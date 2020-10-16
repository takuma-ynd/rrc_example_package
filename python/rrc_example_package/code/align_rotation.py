import os
import random
import itertools
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R
from .const import *

from pybullet_planning import plan_joint_motion

from .utils import apply_transform, VisualCubeOrientation, set_seed

def sort_map_positions_to_fingers(tips, goal_tips):
    inds = np.asarray([ v for v in itertools.permutations([0,1,2])])
    cost = np.asarray([ np.linalg.norm(tips - goal_tips[v, :]) for v in inds])

    inds_sort = inds[np.argsort(cost), :]

    return inds_sort

def pinocchio_solve_ik(solve_ik, obs, target_tip_positions):
    joints = obs['robot_position']
    target_robot_joint_angles = []
    for i in range(3):
        joint_pos = solve_ik(i, target_tip_positions[i], joints)
        try:
            target_robot_joint_angles.append(joint_pos[3*i:3*(i+1)])
        except TypeError:
            return target_robot_joint_angles

    target_joint_angles = np.concatenate(target_robot_joint_angles)
    return target_joint_angles

def project_cube_xy_plane(orientation):
    rot = R.from_quat(orientation)

    #define axis
    axes = np.eye(3)

    #rotate axis by cube orientation
    axes_rotated = rot.apply(axes)

    #calculate the angle between each rotated axis and xy plane
    cos = axes_rotated[:,2] #dot product with z_axis

    #choose the nearest axis to z_axis
    idx = np.argmax(np.abs(cos))
    sign = np.sign(cos[idx])

    #calculate align rotation
    rot_align = vector_align_rotation(axes_rotated[idx], sign * axes[2])

    return (rot_align * rot).as_quat()


def vector_align_rotation(a, b):
    """
    return Rotation that transform vector a to vector b

    input
    a : np.array(3)
    b : np.array(3)

    return
    rot : scipy.spatial.transform.Rotation
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    assert norm_a != 0 and norm_b != 0

    a = a / norm_a
    b = b / norm_b

    cross = np.cross(a, b)
    norm_cross = np.linalg.norm(cross)
    cross = cross / norm_cross
    dot = np.dot(a, b)

    if norm_cross < 1e-8 and dot > 0:
        '''same direction, no rotation a == b'''
        return R.from_quat([0,0,0,1])
    elif norm_cross < 1e-8 and dot < 0:
        '''opposite direction a == -b'''
        c = np.eye(3)[np.argmax(np.linalg.norm(np.eye(3) - a, axis=1))]
        cross = np.cross(a, c)
        norm_cross = np.linalg.norm(cross)
        cross = cross / norm_cross

        return R.from_rotvec(cross * np.pi)

    rot = R.from_rotvec(cross * np.arctan2(norm_cross, dot))

    assert np.linalg.norm(rot.apply(a) - b) < 1e-7
    return rot


def pitch_rotation_times(cube_orientation, goal_orientation):
    rot_cube = R.from_quat(cube_orientation)
    rot_goal = R.from_quat(goal_orientation)

    rot = rot_goal * rot_cube.inv()

    z_axis = np.array([0,0,1])

    z_axis_rotated = rot.apply(z_axis)

    cos_z = z_axis.dot(z_axis_rotated)

    if   cos_z > np.cos( np.pi / 4):
        return 0
    elif cos_z > np.cos( np.pi * 3 / 4):
        return 1
    else:
        return 2

def calc_pitching_cube_tip_position(pitch_axis, pitch_angle):
    x = np.asarray([0.05, 0, 0])
    y = np.asarray([0, 0.05, 0])
    if   pitch_angle > 0 and pitch_axis == 'x':
        cube_tip_positions = np.asarray([x,  y, -x])
    elif pitch_angle > 0 and pitch_axis == 'y':
        cube_tip_positions = np.asarray([y, -x, -y])
    elif pitch_angle < 0 and pitch_axis == 'x':
        cube_tip_positions = np.asarray([x, -y, -x])
    elif pitch_angle < 0 and pitch_axis == 'y':
        cube_tip_positions = np.asarray([y,  x, -y])
    else:
        cube_tip_positions = calc_yawing_cube_tip_position()

    return cube_tip_positions

def calc_yawing_cube_tip_position():
    x = np.asarray([0.05, 0, 0])
    y = np.asarray([0, 0.05, 0])
    cube_tip_positions_list = [np.asarray([x,  y, -x]),
                               np.asarray([y, -x, -y]),
                               np.asarray([x, -y, -x]),
                               np.asarray([y,  x, -y])]

    return cube_tip_positions_list[random.randint(0, 3)]


def align_z(cube_orientation, projected_goal_orientation):
    pitch_times = pitch_rotation_times(cube_orientation, projected_goal_orientation)
    rot = R.from_quat(projected_goal_orientation)
    rot_cube = R.from_quat(cube_orientation)

    if pitch_times == 0:
        return rot.as_quat(), 'z', 0

    if pitch_times == 1:
        axes = ['x', 'x', 'y', 'y']
        angles = [np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2]

        rot_aligns = [R.from_euler(axis, angle) for axis, angle in zip(axes, angles)]
        rot_diff = [ (rot * rot_align * rot_cube.inv()) for rot_align in rot_aligns]
        expmap = [ rot.as_rotvec() for rot in rot_diff]
        norm_expmap_xy = [np.linalg.norm( vec[:2] ) for vec in expmap]

        idx = np.argmin(norm_expmap_xy)

        return (rot * rot_aligns[idx]).as_quat(), axes[idx], -angles[idx]

    if pitch_times == 2:
        axes = ['x' , 'y']

        rot_aligns = [R.from_euler(axis, np.pi) for axis in axes]
        diff_mag = [ (rot * rot_align * rot_cube.inv()).magnitude() for rot_align in rot_aligns]

        idx = np.argmin(diff_mag)

        return (rot * rot_aligns[idx]).as_quat(), axes[idx], np.pi

def pitch_rotation_axis_and_angle(cube_tip_positions):
    x_mean = np.mean(np.abs(cube_tip_positions[:, 0]))
    y_mean = np.mean(np.abs(cube_tip_positions[:, 1]))
    if x_mean > y_mean:
        rotate_axis = "x"
    else:
        rotate_axis = "y"

    if rotate_axis == "x":
        idx = np.argmax(np.abs(cube_tip_positions[:, 1]))
        if cube_tip_positions[idx, 1] > 0:
            rotate_angle = np.pi / 2
        else:
            rotate_angle = -np.pi / 2
    else:
        idx = np.argmax(np.abs(cube_tip_positions[:, 0]))
        if cube_tip_positions[idx, 0] > 0:
            rotate_angle = -np.pi / 2
        else:
            rotate_angle = np.pi / 2

    return rotate_axis, rotate_angle



def cube_rotation_aligned(obs):
    rot = R.from_quat(obs['object_orientation'])
    rot_goal = R.from_quat(obs['goal_object_orientation'])

    angle = (rot.inv() * rot_goal).magnitude()

    return (angle < np.pi/2)

def cube_centered(obs):
    pos_xy = np.asarray([obs['object_position'][0], obs['object_position'][1], 0])

    dist = np.linalg.norm(pos_xy)

    return (dist < 0.07)

def align_rotation(env, obs, cube_manipulator, yaw_planning=True):
    import time
    from .utils import frameskip_to

    with frameskip_to(1, env):
        #p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, 'rotation_{}.mp4'.format(_))
        ik = env.unwrapped.platform.simfinger.pinocchio_utils.inverse_kinematics

        step_start = env.unwrapped.step_count

        #stop when cube is already aligned
        if cube_rotation_aligned(obs):
            print('align_rotation succeeded!')
            return obs

        #centering cube
        if not cube_centered(obs):
            obs = cube_manipulator.move_to_center(obs, force_control=False)
            obs = cube_manipulator.holds_until_cube_stops(obs)


        obs, cube_tip_positions = cube_manipulator.align_pitch(obs)
        obs = cube_manipulator.holds_until_cube_stops(obs)

        if yaw_planning:
            obs = cube_manipulator.align_yaw(obs, planning=yaw_planning)
            obs = cube_manipulator.holds_until_cube_stops(obs)
        else:
            projected_goal_orientation = project_cube_xy_plane(obs['goal_object_orientation'])
            z_aligned_goal_orientation, pitch_axis, pitch_angle = align_z(obs['object_orientation'], projected_goal_orientation)
            obs = cube_manipulator.align_yaw(obs,
                                             planning=yaw_planning,
                                             cube_tip_positions=cube_tip_positions,
                                             pitch_axis=pitch_axis,
                                             pitch_angle=pitch_angle)
            obs = cube_manipulator.holds_until_cube_stops(obs)


        step_end = env.unwrapped.step_count
        print("total steps for aligning: {}".format(step_end - step_start))

    return obs

def run_episode(args):
    from .training_env import make_training_env
    from gym.wrappers import Monitor, TimeLimit
    from .save_video import merge_videos
    from .training_env.wrappers import RenderWrapper
    from .cube_manipulator import CubeManipulator
    import time

    def calc_tip_positions(env, obs):
        from code.grasp_sampling import GraspSampler
        sample_fc_grasp = GraspSampler(env, obs, mu=MU)
        cube_tip_positions, current_tip_positions, joint_conf = sample_fc_grasp(VIRTUAL_CUBE_HALFWIDTH, shrink_region=0.46)
        return cube_tip_positions

    config = {
        'action_space': 'position',
        'frameskip': 3,
        'reward_fn': 'task4_competition_reward',
        'termination_fn': 'position_close_to_goal',
        'initializer': 'task4_init',
        'rank': args.seed
    }
    if args.record is not None:
        config.update({'monitor': True})
    set_seed(args.seed)
    env = make_training_env(visualization=True, **config)
    env = env.env  # HACK to remove FLatObservationWrapper
    # env = TimeLimit(env, 3000)

    print('before start.. frameskip', env.unwrapped.frameskip)
    for i in range(args.num_episodes):
        if args.record is not None:
            env.stats_recorder.save_complete()
            env.stats_recorder.done = True
        obs = env.reset()
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=0,
                                        cameraPitch=-40,
                                        cameraTargetPosition=[0, 0, 0])
        # env.cube_tip_positions = calc_tip_positions(env, obs)
        cube_manipulator = CubeManipulator(env)
        align_rotation(env, obs, cube_manipulator)
        time.sleep(1.0)
    env.close()
    if args.record is not None:
        os.mkdir('rot_videos') if not os.path.isdir('rot_videos') else None
        merge_videos(f"rot_videos/{args.record}.mp4", TMP_VIDEO_DIR)


if __name__ == '__main__':
    from .save_video import remove_temp_dir
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--num_episodes", type=int, default=1, help="number of episodes")
    parser.add_argument("--record", default=None, help="save file name")
    args = parser.parse_args()

    if args.record is not None:
        if os.path.isdir(TMP_VIDEO_DIR):
            remove_temp_dir(TMP_VIDEO_DIR)
    run_episode(args)
