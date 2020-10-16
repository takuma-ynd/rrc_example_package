#!/usr/bin/env python3
from .fc_force_control import ForceControlPolicy, Viz, grasp_force_control
from rrc_simulation.gym_wrapper.envs import cube_env
from .wholebody_planning import WholeBodyPlanner
from .training_env import make_training_env
from rrc_simulation import TriFingerPlatform
from gym.wrappers import Monitor, TimeLimit
import pybullet as p
import numpy as np
import time
import gym

from .training_env.wrappers import RenderWrapper
from gym.wrappers import Monitor
from .save_video import merge_videos

from .utils import set_seed, action_type_to, repeat
import argparse
import functools
from .const import MU, VIRTUAL_CUBE_HALFWIDTH
from collections import namedtuple

class PlanningAndForceControlPolicy:
    def __init__(self, env, obs, fc_policy, action_repeat=2, align_goal_ori=True,
                 use_rrt=False, use_incremental_rrt=False, constants=None):
        if constants is None:
            constants = self._get_default_constants()
        self.env = env
        self.fc_policy = fc_policy
        self.planner = WholeBodyPlanner(env)
        if align_goal_ori:
            goal_ori = obs['goal_object_orientation']
        else:
            goal_ori = obs['object_orientation']
        path = self.planner.plan(obs, obs['goal_object_position'], goal_ori,
                                 retry_grasp=10, mu=constants.mu,
                                 cube_halfwidth=constants.halfwidth,
                                 use_rrt=use_rrt,
                                 use_incremental_rrt=use_incremental_rrt,
                                 use_ori=align_goal_ori)
        if path.cube is None:
            raise RuntimeError('No feasible path is found.')

        self.path = path
        self.joint_sequence = repeat(path.joint_conf, num_repeat=action_repeat)
        self.cube_sequence = repeat(path.cube, num_repeat=action_repeat)
        self._step = 0
        self._actions_in_progress = False

    @property
    def path_length(self):
        return len(self.path.cube)

    def get_grasp_pose(self):
        return self.path.joint_conf[0]

    def get_cube_tip_pos(self):
        return self.path.cube_tip_pos

    def get_init_cube_pose(self):
        return self.path.cube[0]

    def _get_default_constants(self):
        from .const import MU, VIRTUAL_CUBE_HALFWIDTH
        Constants = namedtuple('Constants', ['mu', 'halfwidth'])
        return Constants(mu=MU, halfwidth=VIRTUAL_CUBE_HALFWIDTH)

    def _initialize_joint_poses(self, obs):
        self.env.platform.simfinger.reset_finger_positions_and_velocities(self.path.joint_conf[0])
        obs['robot_position'] = self.path.joint_conf[0]
        self.env.cube_tip_positions = self.path.cube_tip_pos
        return obs

    def at_end_of_sequence(self, step):
        len_sequence = len(self.cube_sequence) -1
        return (step > len_sequence)

    def get_steps_past_sequence(self, step):
        len_sequence = len(self.cube_sequence) -1
        return min(0, step - len_sequence)

    def get_action(self, obs, step=None):
        # NOTE: It assumes that fingers have already been in the grasp pose specified in path.joint_conf[0] at the beginning
        if not self._actions_in_progress:
            if np.linalg.norm(obs['robot_position'] - self.path.joint_conf[0]).sum() > 0.25:
                print('large initial joint conf error:', np.linalg.norm(obs['robot_position'] - self.path.joint_conf[0]))
        self._actions_in_progress = True

        step = self._step if step is None else step
        step = min(step, len(self.cube_sequence) - 1)
        target_cube_pose = self.cube_sequence[step]
        target_joint_conf = self.joint_sequence[step]

        torque = self.fc_policy(obs, target_cube_pose[:3],
                                p.getQuaternionFromEuler(target_cube_pose[3:]))
        action = {
            'position': np.asarray(target_joint_conf),
            'torque': torque
        }
        if step is None:
            self._step += 1
        return action


def run_episode_cube_pose(env, viz=None):
    obs = env.reset()
    # plan a path
    planner = WholeBodyPlanner(env)
    path = planner.plan(obs, obs['goal_object_position'], obs['goal_object_orientation'], retry_grasp=100, mu=MU)

    if path.cube is None:
        return

    target_sequence = repeat(path.cube, num_repeat=5) + [path.cube[-1]] * 100
    # set joint to the initial configuration and update observation
    env.platform.simfinger.reset_finger_positions_and_velocities(path.joint_conf[0])
    obs['robot_position'] = path.joint_conf[0]
    env.cube_tip_positions = path.cube_tip_pos
    print('path.cube_tip_pos', path.cube_tip_pos)


    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=0,
                                 cameraPitch=-40,
                                 cameraTargetPosition=[0, 0, 0])
    if viz is not None:
        viz.reset(obs)
    obs = grasp(env, obs)
    # tip_pd = TipPD([10, 1], 0.7 * env.cube_tip_positions)
    tip_pd = None
    controller = ForceControlPolicy(env, True, tip_pd, mu=MU, viz=viz)

    # Then move toward the goal positions
    env.unwrapped.action_space = TriFingerPlatform.spaces.robot_torque.gym
    env.unwrapped.action_type = cube_env.ActionType.TORQUE
    done = False
    for cube_pose in target_sequence:
        # transform wrenches to base frame
        cube_pos = cube_pose[:3]
        cube_quat = p.getQuaternionFromEuler(cube_pose[3:])
        torque = controller(obs, cube_pos, cube_quat)
        obs, reward, done, info = env.step(torque)
        viz.update_cube_orientation(obs)
        time.sleep(0.01)
        if done:
            break
    time.sleep(2)
    # env.close()


def run_episode_joint_conf(env, viz=None):
    obs = env.reset()
    # plan a path
    planner = WholeBodyPlanner(env)
    path = planner.plan(obs, obs['goal_object_position'], obs['goal_object_orientation'], retry_grasp=100, mu=MU)

    if path.cube is None:
        return

    target_sequence = repeat(path.joint_conf, num_repeat=1) + [path.joint_conf[-1]] * 100
    # set joint to the initial configuration and update observation
    env.platform.simfinger.reset_finger_positions_and_velocities(path.joint_conf[0])
    obs['robot_position'] = path.joint_conf[0]
    env.cube_tip_positions = path.cube_tip_pos
    print('path.cube_tip_pos', path.cube_tip_pos)


    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=0,
                                    cameraPitch=-40,
                                    cameraTargetPosition=[0, 0, 0])
    if viz is not None:
        viz.reset(obs)
    obs = grasp(env, obs)

    # Then move toward the goal positions
    env.unwrapped.action_space = TriFingerPlatform.spaces.robot_position.gym
    env.unwrapped.action_type = cube_env.ActionType.POSITION
    for joint_conf in target_sequence:
        print(joint_conf)
        obs, reward, done, info = env.step(np.asarray(joint_conf))
        viz.update_cube_orientation(obs)
        time.sleep(0.1)
        if done:
            break


def run_episode_hybrid(env, viz=None):
    obs = env.reset()

    # set up the policy
    Constants = namedtuple('Const', ['mu', 'halfwidth'])
    fc_policy = ForceControlPolicy(env, apply_torques=True, mu=MU, grasp_force=0.05,
                                    viz=viz, use_inv_dynamics=True)
    planning_fc_policy = PlanningAndForceControlPolicy(env, obs, fc_policy, action_repeat=2)

    # set joint to the initial configuration and update observation
    obs = planning_fc_policy._initialize_joint_poses(obs)

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=0,
                                    cameraPitch=-40,
                                    cameraTargetPosition=[0, 0, 0])
    if viz is not None:
        viz.reset(obs)
    obs = grasp_force_control(env, obs, fc_policy, grasp_force=0.8)

    # Then move toward the goal positions
    extra_steps = 100
    from rrc_simulation.gym_wrapper.envs.cube_env import ActionType
    with action_type_to(ActionType.TORQUE_AND_POSITION, env):
        for i in range(planning_fc_policy.path_length + extra_steps):
            action = planning_fc_policy.get_action(obs, step=i)
            obs, reward, done, info = env.step(action)
            viz.update_cube_orientation(obs)
            if done:
                break
            time.sleep(0.01)


def run_episodes(neps, control, seed=0, filename=None):
    reward_fn = 'task1_reward'
    termination_fn = 'pos_and_rot_close_to_goal'
    # termination_fn = 'position_close_to_goal'
    initializer = 'task4_small_rot_init'
    env = make_training_env(reward_fn, termination_fn, initializer,
                            action_space='torque_and_position',
                            init_joint_conf=True,
                            visualization=True,
                            frameskip=3,  # Idk how it affects the performance...
                            grasp='pinch', rank=seed)
    env = env.env  # HACK to remove FLatObservationWrapper

    if filename is not None:
        tmp_dir = '/tmp/video'
        env = Monitor(RenderWrapper(TimeLimit(env, 1000)), tmp_dir,
                    video_callable=lambda episode_id: True, mode='evaluation',
                    force=True)
    env = TimeLimit(env, 10000)
    viz = Viz()
    for _ in range(neps):
        if control == 'hybrid':
            run_episode_hybrid(env, viz)
        elif control in ['cube_pose', 'joint_conf']:
            RuntimeError('control mode: cube_pose, joint_conf are not maintained anymore')
        else:
            ValueError(f'unknown control method {control}')
    env.close()

    if filename is not None:
        merge_videos(filename, tmp_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--num_episodes", type=int, default=1, help="number of episodes")
    parser.add_argument("--record", default=None, help="save file name")
    args = parser.parse_args()
    control = 'hybrid'
    if args.record is not None:
        filename = f"videos/fc_plan_{control}_{args.record}.mp4"
    else:
        filename = None
    set_seed(args.seed)
    run_episodes(args.num_episodes, control, args.seed, filename)
