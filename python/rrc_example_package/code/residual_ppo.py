"""Define networks and ResidualPPO2."""
from dl.rl import PolicyBase, ValueFunctionBase, Policy, ValueFunction
from dl.rl import VecEpisodeLogger, VecRewardNormWrapper, RolloutDataManager
from dl.rl.util import misc, rl_evaluate, rl_record
from dl.modules import DiagGaussian, ProductDistribution, Normal
from dl import Checkpointer, logger, Algorithm, nest
import torch
import torch.nn as nn
import torch.nn.functional as F
import gin
import os
import time
import numpy as np
from dl.rl.envs import SubprocVecEnv, DummyVecEnv, EpisodeInfo, VecObsNormWrapper
from .training_env import make_training_env


@gin.configurable
def make_pybullet_env(nenv, reward_fn, termination_fn, initializer, action_space,
                      init_joint_conf=False, residual=False, kp_coef=None,
                      kd_coef=None, frameskip=1, seed=0,
                      norm_observations=False, visualization=False,
                      grasp='pinch', monitor=False):

    def _env(rank):
        def _thunk():
            env = make_training_env(reward_fn, termination_fn, initializer,
                                    action_space, init_joint_conf, residual,
                                    kp_coef, kd_coef, frameskip, rank,
                                    visualization, grasp, monitor)
            env = EpisodeInfo(env)
            env.seed(seed + rank)
            return env
        return _thunk

    if nenv > 1:
        env = SubprocVecEnv([_env(i) for i in range(nenv)], context='fork')
    else:
        env = DummyVecEnv([_env(0)])
        env.reward_range = env.envs[0].reward_range

    if norm_observations:
        env = VecObsNormWrapper(env)
    return env


class ScaledNormal(Normal):
    def __init__(self, loc, scale, fac):
        super().__init__(loc, scale)
        self.fac = fac

    def mode(self):
        return self.mean * self.fac

    def sample(self):
        return super().sample() * self.fac

    def rsample(self):
        return super().rsample() * self.fac

    def log_prob(self, ac):
        return super().log_prob(ac / self.fac)

    def to_tensors(self):
        return {'loc': self.mean, 'scale': self.stddev}

    def from_tensors(self, tensors):
        return ScaledNormal(tensors['loc'], tensors['scale'], fac=self.fac)


class PolicyNet(PolicyBase):
    """Policy network."""

    def __init__(self, observation_space, action_space, torque_std=0.05):
        self.torque_std = torque_std
        super().__init__(observation_space, action_space)

    def build(self):
        """Build."""
        self.fc1 = nn.Linear(self.observation_space.shape[0], 256)
        self.fc2 = nn.Linear(256, 256)
        self.dist = DiagGaussian(256, self.action_space.shape[0],
                                 constant_log_std=False)
        for p in self.dist.fc_mean.parameters():
            nn.init.constant_(p, 0.)

    def forward(self, x):
        """Forward."""
        x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        dist = self.dist(x)
        return ScaledNormal(dist.mean, dist.stddev, fac=self.torque_std)


class TorqueAndPositionPolicyNet(PolicyBase):
    """Policy network."""

    def __init__(self, observation_space, action_space, torque_std=0.05,
                 position_std=0.001):
        self.torque_std = torque_std
        self.position_std = position_std
        super().__init__(observation_space, action_space)

    def build(self):
        """Build."""
        self.fc1 = nn.Linear(self.observation_space.shape[0], 256)
        self.fc2 = nn.Linear(256, 256)
        self.dist_torque = DiagGaussian(256,
                                        self.action_space['torque'].shape[0],
                                        constant_log_std=False)
        self.dist_position = DiagGaussian(256,
                                          self.action_space['position'].shape[0],
                                          constant_log_std=False)
        for p in self.dist_torque.fc_mean.parameters():
            nn.init.constant_(p, 0.)
        for p in self.dist_position.fc_mean.parameters():
            nn.init.constant_(p, 0.)

    def forward(self, x):
        """Forward."""
        x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        d_torque = self.dist_torque(x)
        d_torque = ScaledNormal(d_torque.mean, d_torque.stddev,
                                fac=self.torque_std)
        d_position = self.dist_position(x)
        d_position = ScaledNormal(d_position.mean, d_position.stddev,
                                  fac=self.position_std)
        return ProductDistribution({'torque': d_torque,
                                    'position': d_position})


class VFNet(ValueFunctionBase):
    """Value Function."""

    def build(self):
        """Build."""
        self.fc1 = nn.Linear(self.observation_space.shape[0], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.vf = nn.Linear(256, 1)

    def forward(self, x):
        """Forward."""
        x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.vf(x)


@gin.configurable
def policy_fn(env, torque_std=0.05):
    """Create policy."""
    return Policy(PolicyNet(env.observation_space, env.action_space,
                            torque_std=torque_std))


@gin.configurable
def torque_and_position_policy_fn(env, torque_std=0.05, position_std=0.001):
    """Create policy."""
    return Policy(TorqueAndPositionPolicyNet(env.observation_space,
                                             env.action_space,
                                             torque_std=torque_std,
                                             position_std=position_std))


@gin.configurable
def value_fn(env):
    """Create value function network."""
    return ValueFunction(VFNet(env.observation_space, env.action_space))


class ResidualPPOActor(object):
    """Actor."""

    def __init__(self, pi, vf, policy_training_start):
        """Init."""
        self.pi = pi
        self.vf = vf
        self.policy_training_start = policy_training_start
        self.t = 0

    def __call__(self, ob, state_in=None):
        """Produce decision from model."""
        if self.t < self.policy_training_start:
            outs = self.pi(ob, state_in, deterministic=True)
        else:
            outs = self.pi(ob, state_in)

        def _res_norm(ac):
            return ac.abs().sum(dim=1).mean()
        residual_norm = nest.map_structure(_res_norm, outs.action)
        if isinstance(residual_norm, torch.Tensor):
            logger.add_scalar('actor/l1_residual_norm', residual_norm, self.t,
                              time.time())
            self.t += outs.action.shape[0]
        else:
            self.t += nest.flatten(outs.action)[0].shape[0]
            for k, v in residual_norm.items():
                logger.add_scalar(f'actor/{k}_residual_norm', v, self.t,
                                  time.time())
        data = {'action': outs.action,
                'value': self.vf(ob).value,
                'logp': outs.dist.log_prob(outs.action),
                'dist': outs.dist.to_tensors()}
        if outs.state_out:
            data['state'] = outs.state_out
        return data

    def state_dict(self):
        """State dict."""
        return {'t': self.t}

    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.t = state_dict['t']


@gin.configurable(blacklist=['logdir'])
class ResidualPPO2(Algorithm):
    """PPO algorithm with upgrades.

    This version is described in https://arxiv.org/abs/1707.02286 and
    https://github.com/joschu/modular_rl/blob/master/modular_rl/ppo.py
    """

    def __init__(self,
                 logdir,
                 env_fn,
                 policy_fn,
                 value_fn,
                 nenv=1,
                 opt_pi=torch.optim.Adam,
                 opt_vf=torch.optim.Adam,
                 batch_size=32,
                 rollout_length=None,
                 gamma=0.99,
                 lambda_=0.95,
                 ent_coef=0.01,
                 norm_advantages=False,
                 epochs_pi=10,
                 epochs_vf=10,
                 max_grad_norm=None,
                 kl_target=0.01,
                 alpha=1.5,
                 policy_training_start=10000,
                 eval_num_episodes=10,
                 record_num_episodes=0,
                 gpu=True):
        """Init."""
        self.logdir = logdir
        self.ckptr = Checkpointer(os.path.join(logdir, 'ckpts'))
        self.env_fn = env_fn
        self.nenv = nenv
        self.eval_num_episodes = eval_num_episodes
        self.record_num_episodes = record_num_episodes
        self.ent_coef = ent_coef
        self.epochs_pi = epochs_pi
        self.epochs_vf = epochs_vf
        self.max_grad_norm = max_grad_norm
        self.kl_target = kl_target
        self.initial_kl_weight = 0.2
        self.kl_weight = self.initial_kl_weight
        self.alpha = alpha
        self.policy_training_start = policy_training_start
        self.device = torch.device('cuda:0' if gpu and torch.cuda.is_available()
                                   else 'cpu')

        self.env = VecEpisodeLogger(VecRewardNormWrapper(env_fn(nenv=nenv),
                                                         gamma))

        self.pi = policy_fn(self.env).to(self.device)
        self.vf = value_fn(self.env).to(self.device)
        self.opt_pi = opt_pi(self.pi.parameters())
        self.opt_vf = opt_vf(self.vf.parameters())
        self._actor = ResidualPPOActor(self.pi, self.vf, policy_training_start)
        self.data_manager = RolloutDataManager(
            self.env,
            self._actor,
            self.device,
            batch_size=batch_size,
            rollout_length=rollout_length,
            gamma=gamma,
            lambda_=lambda_,
            norm_advantages=norm_advantages)

        self.mse = nn.MSELoss()

        self.t = 0

    def compute_kl(self):
        """Compute KL divergence of new and old policies."""
        kl = 0
        n = 0
        for batch in self.data_manager.sampler():
            outs = self.pi(batch['obs'])
            old_dist = outs.dist.from_tensors(batch['dist'])
            k = old_dist.kl(outs.dist).mean().detach().cpu().numpy()
            s = nest.flatten(batch['action'])[0].shape[0]
            kl = (n / (n + s)) * kl + (s / (n + s)) * k
            n += s
        return kl

    def loss_pi(self, batch):
        """Compute loss."""
        outs = self.pi(batch['obs'])

        # compute policy loss
        logp = outs.dist.log_prob(batch['action'])
        assert logp.shape == batch['logp'].shape
        ratio = torch.exp(logp - batch['logp'])
        assert ratio.shape == batch['atarg'].shape

        old_dist = outs.dist.from_tensors(batch['dist'])
        kl = old_dist.kl(outs.dist)
        kl_pen = (kl - 2 * self.kl_target).clamp(min=0).pow(2)
        losses = {}
        losses['pi'] = -(ratio * batch['atarg']).mean()
        losses['ent'] = -outs.dist.entropy().mean()
        losses['kl'] = kl.mean()
        losses['kl_pen'] = kl_pen.mean()
        losses['total'] = (losses['pi'] + self.ent_coef * losses['ent']
                           + self.kl_weight * losses['kl'] + 1000 * losses['kl_pen'])
        return losses

    def loss_vf(self, batch):
        return self.mse(self.vf(batch['obs']).value, batch['vtarg'])

    def step(self):
        """Compute rollout, loss, and update model."""
        self.pi.train()
        self.t += self.data_manager.rollout()
        losses = {'pi': [], 'vf': [], 'ent': [], 'kl': [], 'total': [],
                  'kl_pen': []}

        #######################
        # Update pi
        #######################

        if self.t >= self.policy_training_start:
            kl_too_big = False
            for _ in range(self.epochs_pi):
                if kl_too_big:
                    break
                for batch in self.data_manager.sampler():
                    self.opt_pi.zero_grad()
                    loss = self.loss_pi(batch)
                    # break if new policy is too different from old policy
                    if loss['kl'] > 4 * self.kl_target:
                        kl_too_big = True
                        break
                    loss['total'].backward()

                    for k, v in loss.items():
                        losses[k].append(v.detach().cpu().numpy())

                    if self.max_grad_norm:
                        norm = nn.utils.clip_grad_norm_(self.pi.parameters(),
                                                        self.max_grad_norm)
                        logger.add_scalar('alg/grad_norm', norm, self.t,
                                          time.time())
                        logger.add_scalar('alg/grad_norm_clipped',
                                          min(norm, self.max_grad_norm),
                                          self.t, time.time())
                    self.opt_pi.step()

        #######################
        # Update value function
        #######################
        for _ in range(self.epochs_vf):
            for batch in self.data_manager.sampler():
                self.opt_vf.zero_grad()
                loss = self.loss_vf(batch)
                losses['vf'].append(loss.detach().cpu().numpy())
                loss.backward()
                if self.max_grad_norm:
                    norm = nn.utils.clip_grad_norm_(self.vf.parameters(),
                                                    self.max_grad_norm)
                    logger.add_scalar('alg/vf_grad_norm', norm, self.t,
                                      time.time())
                    logger.add_scalar('alg/vf_grad_norm_clipped',
                                      min(norm, self.max_grad_norm),
                                      self.t, time.time())
                self.opt_vf.step()

        for k, v in losses.items():
            if len(v) > 0:
                logger.add_scalar(f'loss/{k}', np.mean(v), self.t, time.time())

        # update weight on kl to match kl_target.
        if self.t >= self.policy_training_start:
            kl = self.compute_kl()
            if kl > 10.0 * self.kl_target and self.kl_weight < self.initial_kl_weight:
                self.kl_weight = self.initial_kl_weight
            elif kl > 1.3 * self.kl_target:
                self.kl_weight *= self.alpha
            elif kl < 0.7 * self.kl_target:
                self.kl_weight /= self.alpha

            logger.add_scalar('alg/kl', kl, self.t, time.time())
            logger.add_scalar('alg/kl_weight', self.kl_weight, self.t, time.time())

        data = self.data_manager.storage.get_rollout()
        value_error = data['vpred'].data - data['q_mc'].data
        logger.add_scalar('alg/value_error_mean',
                          value_error.mean().cpu().numpy(), self.t, time.time())
        logger.add_scalar('alg/value_error_std',
                          value_error.std().cpu().numpy(), self.t, time.time())
        return self.t

    def evaluate(self):
        """Evaluate model."""
        self.pi.eval()
        misc.set_env_to_eval_mode(self.env)

        # Eval policy
        os.makedirs(os.path.join(self.logdir, 'eval'), exist_ok=True)
        outfile = os.path.join(self.logdir, 'eval',
                               self.ckptr.format.format(self.t) + '.json')
        stats = rl_evaluate(self.env, self.pi, self.eval_num_episodes,
                            outfile, self.device)
        logger.add_scalar('eval/mean_episode_reward', stats['mean_reward'],
                          self.t, time.time())
        logger.add_scalar('eval/mean_episode_length', stats['mean_length'],
                          self.t, time.time())

        # Record policy
        os.makedirs(os.path.join(self.logdir, 'video'), exist_ok=True)
        outfile = os.path.join(self.logdir, 'video',
                               self.ckptr.format.format(self.t) + '.mp4')
        rl_record(self.env, self.pi, self.record_num_episodes, outfile,
                  self.device)

        self.pi.train()
        misc.set_env_to_train_mode(self.env)

    def save(self):
        """State dict."""
        state_dict = {
            'pi': self.pi.state_dict(),
            'vf': self.vf.state_dict(),
            'opt_pi': self.opt_pi.state_dict(),
            'opt_vf': self.opt_vf.state_dict(),
            'kl_weight': self.kl_weight,
            'env': misc.env_state_dict(self.env),
            '_actor': self._actor.state_dict(),
            't': self.t
        }
        self.ckptr.save(state_dict, self.t)

    def load(self, t=None):
        """Load state dict."""
        state_dict = self.ckptr.load(t)
        if state_dict is None:
            self.t = 0
            return self.t
        self.pi.load_state_dict(state_dict['pi'])
        self.vf.load_state_dict(state_dict['vf'])
        self.opt_pi.load_state_dict(state_dict['opt_pi'])
        self.opt_vf.load_state_dict(state_dict['opt_vf'])
        self.kl_weight = state_dict['kl_weight']
        misc.env_load_state_dict(self.env, state_dict['env'])
        self._actor.load_state_dict(state_dict['_actor'])
        self.t = state_dict['t']
        return self.t

    def close(self):
        """Close environment."""
        try:
            self.env.close()
        except Exception:
            pass


if __name__ == '__main__':

    import unittest
    import shutil
    from dl import train
    from dl.rl.envs import make_env
    from dl.rl.modules import PolicyBase, ValueFunctionBase
    from functools import partial

    class TestPPO2(unittest.TestCase):
        """Test case."""

        def test_feed_forward_ppo2(self):
            """Test feed forward ppo2."""
            def env_fn(nenv):
                return make_env('LunarLanderContinuous-v2', nenv)

            def policy_fn(env):
                return Policy(PolicyNet(env.observation_space,
                                        env.action_space))

            def vf_fn(env):
                return ValueFunction(VFNet(env.observation_space,
                                           env.action_space))

            ppo = partial(ResidualPPO2, env_fn=env_fn, policy_fn=policy_fn,
                          value_fn=vf_fn)
            train('test', ppo, maxt=1000, eval=True, eval_period=1000)
            shutil.rmtree('test')

    unittest.main()
