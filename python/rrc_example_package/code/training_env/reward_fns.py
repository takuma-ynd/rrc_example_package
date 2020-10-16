"""Place reward functions here.

These will be passed as an arguement to the training env, allowing us to
easily try out new reward functions.
"""


import numpy as np
from rrc_simulation.tasks import move_cube
from scipy.spatial.transform import Rotation


###############################
# Competition Reward Functions
###############################


def _competition_reward(observation, difficulty):
    object_pose = move_cube.Pose(
        observation['object_position'],
        observation['object_orientation']
    )
    goal_pose = move_cube.Pose(
        observation['goal_object_position'],
        observation['goal_object_orientation']
    )
    return -move_cube.evaluate_state(goal_pose, object_pose, difficulty)


def task1_competition_reward(previous_observation, observation, action):
    return _competition_reward(observation, 1)


def task2_competition_reward(previous_observation, observation, action):
    return _competition_reward(observation, 2)


def task3_competition_reward(previous_observation, observation, action):
    return _competition_reward(observation, 3)


def task4_competition_reward(previous_observation, observation, action):
    return _competition_reward(observation, 4)


##############################
# Training Reward functions
##############################


def _shaping_distance_from_tips_to_cube(previous_observation, observation):
    # calculate first reward term
    current_distance_from_block = np.linalg.norm(
        observation["robot_tip_positions"] - observation["object_position"]
    )
    previous_distance_from_block = np.linalg.norm(
        previous_observation["robot_tip_positions"]
        - previous_observation["object_position"]
    )

    return previous_distance_from_block - current_distance_from_block


def _action_reg(previous_observation, action):
    v = previous_observation['robot_velocity']
    t = np.array(action.torque)
    velocity_reg = v.dot(v)
    torque_reg = t.dot(t)
    return 0.1 * velocity_reg + torque_reg


def task1_reward(previous_observation, observation, action):
    shaping = _shaping_distance_from_tips_to_cube(previous_observation,
                                                  observation)
    return _competition_reward(observation, 1) + 500 * shaping


def task1_reward_reg(previous_observation, observation, action):
    shaping = _shaping_distance_from_tips_to_cube(previous_observation,
                                                  observation)
    reg = _action_reg(previous_observation, action)
    r = _competition_reward(observation, 1)
    return r - 0.1 * reg + 500 * shaping


def task2_reward(previous_observation, observation, action):
    shaping = _shaping_distance_from_tips_to_cube(previous_observation,
                                                  observation)
    return _competition_reward(observation, 2) + 500 * shaping


def task2_reward_reg(previous_observation, observation, action):
    shaping = _shaping_distance_from_tips_to_cube(previous_observation,
                                                  observation)
    reg = _action_reg(previous_observation, action)
    r = _competition_reward(observation, 2)
    return r - 0.1 * reg + 500 * shaping


def task3_reward(previous_observation, observation, action):
    shaping = _shaping_distance_from_tips_to_cube(previous_observation,
                                                  observation)
    return _competition_reward(observation, 3) + 500 * shaping


def task3_reward_reg(previous_observation, observation, action):
    shaping = _shaping_distance_from_tips_to_cube(previous_observation,
                                                  observation)
    reg = _action_reg(previous_observation, action)
    r = _competition_reward(observation, 3)
    return r - 0.1 * reg + 500 * shaping


def task4_reward(previous_observation, observation, action):
    shaping = _shaping_distance_from_tips_to_cube(previous_observation,
                                                  observation)
    return _competition_reward(observation, 4) + 500 * shaping


def task4_reward_reg(previous_observation, observation, action):
    shaping = _shaping_distance_from_tips_to_cube(previous_observation,
                                                  observation)
    reg = _action_reg(previous_observation, action)
    r = _competition_reward(observation, 4)
    return r - 0.1 * reg + 500 * shaping


def task1_reward_reg_slippage(previous_observation, observation, action):
    shaping = _tip_slippage(previous_observation, observation, action)
    return task1_reward_reg(previous_observation, observation, action) + 300 * shaping


def task2_reward_reg_slippage(previous_observation, observation, action):
    shaping = _tip_slippage(previous_observation, observation, action)
    return task2_reward_reg(previous_observation, observation, action) + 300 * shaping


def task3_reward_reg_slippage(previous_observation, observation, action):
    shaping = _tip_slippage(previous_observation, observation, action)
    return task3_reward_reg(previous_observation, observation, action) + 300 * shaping


def task4_reward_reg_slippage(previous_observation, observation, action):
    shaping = _tip_slippage(previous_observation, observation, action)
    return task4_reward_reg(previous_observation, observation, action) + 300 * shaping


def _orientation_error(observation):
    goal_rot = Rotation.from_quat(observation['goal_object_orientation'])
    actual_rot = Rotation.from_quat(observation['object_orientation'])
    error_rot = goal_rot.inv() * actual_rot
    return error_rot.magnitude() / np.pi


def match_orientation_reward(previous_observation, observation, action):
    shaping = _shaping_distance_from_tips_to_cube(previous_observation,
                                                  observation)
    return -_orientation_error(observation) + 500 * shaping


def match_orientation_reward_shaped(previous_observation, observation, action):
    shaping = _shaping_distance_from_tips_to_cube(previous_observation,
                                                  observation)
    ori_shaping = (_orientation_error(previous_observation)
                   - _orientation_error(observation))
    return 500 * shaping + 100 * ori_shaping


def _tip_slippage(previous_observation, observation, action):
    obj_rot = Rotation.from_quat(observation['object_orientation'])
    prev_obj_rot = Rotation.from_quat(previous_observation['object_orientation'])
    relative_tip_pos = obj_rot.apply(observation["robot_tip_positions"] - observation["object_position"])
    prev_relative_tip_pos = prev_obj_rot.apply(previous_observation["robot_tip_positions"] - previous_observation["object_position"])
    return - np.linalg.norm(relative_tip_pos - prev_relative_tip_pos)
