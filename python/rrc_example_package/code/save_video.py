#!/usr/bin/env python3
from rrc_simulation.code.training_env import make_training_env
from rrc_simulation.code.training_env.wrappers import RenderWrapper
from rrc_simulation.code.utils import set_seed
from gym.wrappers import Monitor
from rrc_simulation.code.const import TMP_VIDEO_DIR

import statistics
import subprocess
import argparse
import os
import pybullet as p
import time
import shutil
from rrc_simulation.gym_wrapper.envs import cube_env

import torch
import numpy as np
import dl
from dl import nest
from dl.rl import set_env_to_eval_mode
import torch
from rrc_simulation.code.residual_ppo import ResidualPPO2
dl.rng.seed(0)


def merge_videos(target_path, src_dir):
    src_videos = sorted([os.path.join(src_dir, v) for v in os.listdir(src_dir) if v.endswith('.mp4')])
    command = ['ffmpeg']
    for src_video in src_videos:
        command += ['-i', src_video]
    command += ['-filter_complex', f'concat=n={len(src_videos)}:v=1[outv]', '-map', '[outv]', target_path]
    subprocess.run(command)
    # remove_temp_dir(src_dir)


def remove_temp_dir(directory):
    assert directory.startswith('/tmp/') or directory.startswith('./local_tmp/'), 'This function can only remove directories under /tmp/'
    shutil.rmtree(directory)


def _init_env_and_policy(difficulty, policy):
    if policy == 'ppo':
        if difficulty == 4:
            expdir = f'../../../models/mpfc_level_{difficulty}'
            is_level_4 = True
        else:
            expdir = f'../../../models/fc_level_{difficulty}'
            is_level_4 = False
        bindings = [
            f'make_pybullet_env.reward_fn="task{difficulty}_competition_reward"',
            'make_pybullet_env.termination_fn="{}"'.format('stay_close_to_goal_level_4' if is_level_4 else 'stay_close_to_goal'),
            f'make_pybullet_env.initializer="task{difficulty}_init"',
            'make_pybullet_env.visualization=True',
            'make_pybullet_env.monitor=True',
        ]
        from rrc_simulation.code.utils import set_seed
        set_seed(0)
        dl.load_config(expdir + '/config.gin', bindings)
        ppo = ResidualPPO2(expdir, nenv=1)
        ppo.load()
        env = ppo.env
        set_env_to_eval_mode(env)
        return env, ppo

    else:
        eval_config = {
            'action_space': 'torque_and_position' if args.policy == 'mpfc' else 'torque',
            'frameskip': 3,
            'residual': True,
            'reward_fn': f'task{difficulty}_competition_reward',
            'termination_fn': 'no_termination',
            'initializer': f'task{difficulty}_init',
            'monitor': True,
            'rank': 0
        }

        from rrc_simulation.code.utils import set_seed
        set_seed(0)
        env = make_training_env(visualization=False, **eval_config)
        return env, None


def main(args):
    env, ppo = _init_env_and_policy(args.difficulty, args.policy)

    acc_rewards = []
    wallclock_times = []
    # aligning_steps = []
    env_steps = []
    avg_rewards = []
    for i in range(args.num_episodes):
        start = time.time()
        is_done = False
        obs = env.reset()
        accumulated_reward = 0
        # aligning_steps.append(env.unwrapped.step_count)

        if args.policy == 'ppo':
            done = False
            accumulated_reward = 0
            while not done:
                obs = torch.from_numpy(obs).float().to(ppo.device)
                with torch.no_grad():
                    action = ppo.pi(obs, deterministic=True).action
                    action = nest.map_structure(lambda x: x.cpu().numpy(), action)
                obs, reward, done, info = env.step(action)
                accumulated_reward += reward
        else:
            done = False
            accumulated_reward = 0
            if env.action_type == cube_env.ActionType.TORQUE_AND_POSITION:
                zero_action = {
                    'torque': (env.action_space['torque'].sample() * 0).astype(np.float64),
                    'position': (env.action_space['position'].sample() * 0).astype(np.float64)
                }
            else:
                zero_action = np.array(env.action_space.sample() * 0).astype(np.float64)
                assert zero_action.dtype == np.float64
            while not done:
                obs, reward, done, info = env.step(zero_action)
                accumulated_reward += reward

        end = time.time()
        print("Accumulated reward: {}".format(accumulated_reward))
        print('Elapsed:', end - start)
        acc_rewards.append(accumulated_reward)
        wallclock_times.append(end - start)

    env.close()

    def _print_stats(name, data):
        print('======================================')
        print(f'Mean   {name}\t{np.mean(data):.2f}')
        print(f'Max    {name}\t{max(data):.2f}')
        print(f'Min    {name}\t{min(data):.2f}')
        print(f'Median {name}\t{statistics.median(data):2f}')

    # print('Total elapsed time\t{:.2f}'.format(sum(wallclock_times)))
    # print('Mean elapsed time\t{:.2f}'.format(sum(wallclock_times) / len(wallclock_times)))
    # _print_stats('acc reward', acc_rewards)
    # _print_stats('aligning steps', aligning_steps)

    if args.time_steps is None:
        args.time_steps = ppo.ckptr.ckpts()[-1]
    video_file = "{}_{}_steps_level_{}.mp4".format(args.filename, args.time_steps, args.difficulty)
    merge_videos(video_file, TMP_VIDEO_DIR)
    print('Video is saved at {}'.format(video_file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", help="experiment_dir")
    parser.add_argument("--time_steps", type=int, default=None, help="time steps")
    parser.add_argument("--difficulty", type=int, default=1, help="difficulty")
    parser.add_argument("--num_episodes", default=10, type=int, help="number of episodes to record a video")
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("--policy", default='ppo', choices=["fc", "mpfc", "ppo"], help="which policy to run")
    parser.add_argument("--filename", help="filename")
    args = parser.parse_args()

    print('For faster recording, run `git apply faster_recording_patch.diff`. This temporary changes episode length and window size.')

    # check if `ffmpeg` is available
    if shutil.which('ffmpeg') is None:
        raise OSError('ffmpeg is not available. To record a video, you need to install ffmpeg.')

    if os.path.isdir(TMP_VIDEO_DIR):
        remove_temp_dir(TMP_VIDEO_DIR)
    os.mkdir(TMP_VIDEO_DIR)

    if args.policy in ['fc', 'mpfc']:
        scripted_video_dir = 'rrc_videos'
        args.time_steps = 0
        if not os.path.isdir(scripted_video_dir):
            os.makedirs(scripted_video_dir)
        args.exp_dir = scripted_video_dir

    set_seed(args.seed)
    # gym.logger.set_level(gym.logger.INFO)
    main(args)
