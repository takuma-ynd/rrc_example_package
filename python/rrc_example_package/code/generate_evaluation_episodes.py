#!/usr/bin/env python3

import os
import argparse
from rrc_simulation.tasks.move_cube import sample_goal, Pose
from .utils import set_seed


def generate_eval_episodes(num_episodes):
    '''genereate code/eval_episodes/level[n] directories that contain evaluation episodes.'''
    for level in range(1, 4 + 1):
        eval_dir = os.environ['RRC_SIM_ROOT'] + '/code/eval_episodes/level{}'.format(level)
        if not os.path.isdir(eval_dir):
            os.makedirs(eval_dir)

            for i in range(num_episodes):
                # sample init_pos, goal and convert it to json
                init = sample_goal(-1)
                goal = sample_goal(level)
                with open(os.path.join(eval_dir, '{:05d}-init.json'.format(i)), 'w') as f:
                    f.write(init.to_json())
                with open(os.path.join(eval_dir, '{:05d}-goal.json'.format(i)), 'w') as f:
                    f.write(goal.to_json())
        else:
            print('directory {} already exists. skipping...'.format(eval_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate episodes for evaluation')
    parser.add_argument('num_episodes', type=int, help='number of episodes')
    args = parser.parse_args()
    set_seed()
    generate_eval_episodes(args.num_episodes)
