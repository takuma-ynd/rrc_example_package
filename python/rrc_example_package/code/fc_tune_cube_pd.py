#!/usr/bin/env python3
'''
Use custom_utils.sample_cube_surface_points and visualize the sampled points
'''

import random
from .training_env import make_training_env
from .training_env.wrappers import RenderWrapper
from gym.wrappers import Monitor, TimeLimit
import numpy as np
import pybullet as p
import time
import json
from .grasping import CubePD
from .utils import VisualMarkers
from matplotlib import pyplot as plt
from matplotlib import cm
from save_video import merge_videos
random.seed(int(time.time()))
np.random.seed(int(time.time()))


if __name__ == '__main__':
    reward_fn = 'task1_reward'
    termination_fn = 'no_termination'
    initializer = 'task4_init'
    env = make_training_env(reward_fn, termination_fn, initializer,
                            action_space='position',
                            init_joint_conf=False,
                            visualization=False)
    env = env.env  # HACK to remove FLatObservationWrapper

    obs = env.reset()
    vis_markers = VisualMarkers()
    # p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, 'improved_saved.mp4')
    ik = env.unwrapped.platform.simfinger.pinocchio_utils.inverse_kinematics
    pinocchio = env.unwrapped.platform.simfinger.pinocchio_utils
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=0,
                                 cameraPitch=-40,
                                 cameraTargetPosition=[0, 0, 0])

    def get_force_gains(i):
        step_size = 4 / 20
        x = i // 21
        y = i % 21
        return np.array([-1 + step_size*x, -1 + step_size*y])

    def get_torque_gains(i):
        step_size = 3 / 20
        x = i // 21
        y = i % 21
        return np.array([-2 + step_size*x, -2 + step_size*y])

    def error_norm(errors):
        low = np.min(errors)
        high = np.quantile(errors, 0.8)
        errors = (errors - low) / (high - low)
        return np.clip(errors, 0, 1)

    def test_force_gains(gains):
        c = CubePD(force_gains=gains)
        obs = env.reset()
        cube = env.unwrapped.platform.cube
        p.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=0,
                                     cameraPitch=-40,
                                     cameraTargetPosition=[0, 0, 0])
        robot_pos = obs['robot_position']
        obs['object_position'] = np.array([0, 0, 0.0325])
        cube.set_state(obs['object_position'], obs['object_orientation'])
        goal = np.array([0, 0, 0.135])
        goal_ori = np.array([0, 0, 0, 1])

        total_error = 0
        for _ in range(1000):
            force, _ = c(goal, goal_ori, obs['object_position'],
                         obs['object_orientation'])
            p.applyExternalForce(objectUniqueId=cube.block,
                                 linkIndex=-1,
                                 forceObj=force,
                                 posObj=np.zeros(3),
                                 flags=p.LINK_FRAME)
            obs, _, _, _ = env.step(robot_pos)
            total_error += np.linalg.norm(goal - obs['object_position'])
            # time.sleep(0.01)
        return total_error

    def test_torque_gains(gains, force_gains):
        c = CubePD(force_gains=force_gains, torque_gains=gains)
        obs = env.reset()
        cube = env.unwrapped.platform.cube
        p.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=0,
                                     cameraPitch=-40,
                                     cameraTargetPosition=[0, 0, 0])

        # HARD EXAMPLE
        init_pos = np.array([0., 0., 0.0325])
        init_ori = np.array([0.0, 0.0, -1.5634019394850234])
        goal_pos = np.array([-0.04638887, -0.00215872, 0.06769445])
        goal_ori = np.array([-2.2546160857726445, 0.5656166523145225, 1.9566104949225405])
        goal_ori = np.array(p.getQuaternionFromEuler(goal_ori))

        robot_pos = obs['robot_position']
        obs['object_position'] = init_pos
        cube.set_state(init_pos, p.getQuaternionFromEuler(init_ori))

        total_error = 0
        for _ in range(1000):
            force, torque = c(goal_pos, goal_ori, obs['object_position'],
                              obs['object_orientation'])
            p.applyExternalForce(objectUniqueId=cube.block,
                                 linkIndex=-1,
                                 forceObj=force,
                                 posObj=np.zeros(3),
                                 flags=p.LINK_FRAME)

            p.applyExternalTorque(objectUniqueId=cube.block,
                                  linkIndex=-1,
                                  torqueObj=torque,
                                  flags=p.LINK_FRAME)
            obs, _, _, _ = env.step(robot_pos)
            total_error += c._rotation_error(
                                goal_ori, obs['object_orientation']).magnitude()
        return total_error

    def search(test, get_gains, outfile):
        viridis = cm.get_cmap('viridis')
        gains = []
        errors = []
        for i in range(21 * 21):
            print(i)
            # gains.append(sample_gains())
            gains.append(get_gains(i))
            errors.append(test(10 ** gains[-1]))

        env.close()

        gains = np.array(gains)
        errors = np.array(errors)
        colors = viridis(error_norm(errors))
        print(errors)
        plt.scatter(gains[:, 0], gains[:, 1], c=colors, cmap='viridis')
        plt.xlabel("KP")
        plt.ylabel("KD")
        plt.colorbar()
        plt.savefig(f'{outfile}.pdf')

        best = np.argmin(errors)
        print(gains[best])
        print(errors[best])
        with open(f'{outfile}.json', 'w') as f:
            json.dump({'gains': gains[best].tolist(), 'error': errors[best]}, f)

        # plt.show()
        return 10 ** gains[best]

    #######################
    # Tune gains
    #######################

    force_gains = search(test_force_gains, get_force_gains, 'gains')
    torque_gains = search(lambda gains: test_torque_gains(gains, force_gains),
                          get_torque_gains,
                          outfile='torque_gains')
    print(force_gains)
    print(torque_gains)

    def test_pd_controller():
        c = CubePD()
        cube = env.unwrapped.platform.cube
        obs = env.reset()
        p.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=0,
                                     cameraPitch=-40,
                                     cameraTargetPosition=[0, 0, 0])
        robot_pos = obs['robot_position']
        obs['object_position'] = np.array([0, 0, 0.0325])
        goal_pos = obs['goal_object_position']
        goal_ori = obs['goal_object_orientation']

        done = False
        while not done:
            force, torque = c(goal_pos, goal_ori, obs['object_position'],
                              obs['object_orientation'])
            p.applyExternalForce(objectUniqueId=cube.block,
                                 linkIndex=-1,
                                 forceObj=force,
                                 posObj=np.zeros(3),
                                 flags=p.LINK_FRAME)

            p.applyExternalTorque(objectUniqueId=cube.block,
                                  linkIndex=-1,
                                  torqueObj=torque,
                                  flags=p.LINK_FRAME)
            obs, _, done, _ = env.step(robot_pos)
            time.sleep(0.01)

    # tmp_dir = '/tmp/video'
    # env = Monitor(RenderWrapper(TimeLimit(env, 200)), tmp_dir,
    #               video_callable=lambda episode_id: True, mode='evaluation',
    #               force=True)
    # for _ in range(10):
    #     test_pd_controller()
    # video_file = "./force_torque_PD_cube.mp4"
    # env.close()
    # merge_videos(video_file, tmp_dir)
