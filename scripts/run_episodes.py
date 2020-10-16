#!/usr/bin/env python3

import numpy as np
from rrc_example_package.code.training_env import make_training_env
from rrc_example_package.code.training_env.env import ActionType
import rrc_example_package.move_cube
import rrc_example_package.cube_env

# env = make_training_env(visualization=False, **eval_config)
# env.unwrapped.initializer = initializer

# eval_config = {
#         'action_space': 'torque_and_position',
#         'frameskip': 3,
#         'residual': True,
#         'reward_fn': f'task{difficulty}_competition_reward',
#         'termination_fn': 'no_termination',
#         'initializer': f'task{difficulty}_init',
#         'monitor': False,
#         'rank': 0
# }

goal = move_cube.sample_goal(3)
goal_dict = {
    'position': move_cube.position,
    'orientation': move_cube.orientation
}

env = cube_env.RealRobotCubeEnv(goal_dict, 3)

obs = env.reset()
done = False
accumulated_reward = 0
if env.action_type == ActionType.TORQUE_AND_POSITION:
        zero_action = {
        'torque': (env.action_space['torque'].sample() * 0).astype(np.float64),
        'position': (env.action_space['position'].sample() * 0).astype(np.float64)
        }
        assert zero_action['torque'].dtype == np.float64
        assert zero_action['position'].dtype == np.float64
else:
        zero_action = np.array(env.action_space.sample() * 0).astype(np.float64)
        assert zero_action.dtype == np.float64

while not done:
        obs, reward, done, info = env.step(zero_action)
        accumulated_reward += reward
print("Accumulated reward: {}".format(accumulated_reward))
