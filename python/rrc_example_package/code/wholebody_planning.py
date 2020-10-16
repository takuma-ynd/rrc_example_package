#!/usr/bin/env python3
import pybullet as p
import random
from .training_env import make_training_env
from .training_env.initializers import task1_init
import numpy as np
import pybullet as p
import time
from rrc_simulation import visual_objects
from .utils import sample_cube_surface_points, apply_transform, set_seed, VisualCubeOrientation
from pybullet_planning import plan_joint_motion, plan_wholebody_motion
import argparse
from itertools import product
from collections import namedtuple
from rrc_simulation.tasks.move_cube import _CUBE_WIDTH, _ARENA_RADIUS, _min_height, _max_height
from .grasp_sampling import GraspSampler
from .const import COLLISION_TOLERANCE

dummy_links = [-2, -3, -4, -100, -101, -102]  # link indices <=100 denotes circular joints
custom_limits = {
    -2:(-_ARENA_RADIUS, _ARENA_RADIUS),
    -3:(-_ARENA_RADIUS, _ARENA_RADIUS),
    -4:(_min_height, _max_height),
    -100:(-np.pi, np.pi),
    -101:(-np.pi, np.pi),
    -102:(-np.pi, np.pi)
}
    # -102:(0, np.pi)

Path = namedtuple('Path', ['cube', 'joint_conf', 'tip_path', 'cube_tip_pos'])

def get_joint_states(robot_id, link_indices):
    joint_states = [joint_state[0] for joint_state in p.getJointStates(robot_id, link_indices)]
    return np.asarray(joint_states)


def disable_tip_collisions(env):
    disabled_collisions = set()
    for tip_link in env.platform.simfinger.pybullet_tip_link_indices:
        disabled_collisions.add(((env.platform.cube.block, -1), (env.platform.simfinger.finger_id, tip_link)))
    return disabled_collisions


class WholeBodyPlanner:
    def __init__(self, env):
        self.env = env

        # disable collision check for tip
        self._disabled_collisions = disable_tip_collisions(self.env)

    def _get_disabled_colilsions(self):
        disabled_collisions = set()
        for tip_link in self.env.platform.simfinger.pybullet_tip_link_indices:
            disabled_collisions.add(((self.env.platform.cube.block, -1), (self.env.platform.simfinger.finger_id, tip_link)))
        return disabled_collisions

    def _get_tip_path(self, cube_tip_positions, cube_path):
        def get_quat(euler):
            return p.getQuaternionFromEuler(euler)
        return [apply_transform(cube_pose[:3], get_quat(cube_pose[3:]), cube_tip_positions) for cube_pose in cube_path]

    def plan(self, obs, goal_pos=None, goal_quat=None, retry_grasp=10, mu=1.0,
             cube_halfwidth=0.0425, use_rrt=False, use_incremental_rrt=False,
             min_goal_threshold=0.01, max_goal_threshold=0.8, use_ori=False):
        goal_pos = obs['goal_object_position'] if goal_pos is None else goal_pos
        goal_quat = obs['goal_object_orientation'] if goal_quat is None else goal_quat
        resolutions = 0.03 * np.array([0.3, 0.3, 0.3, 1, 1, 1])  # roughly equiv to the lengths of one step.

        goal_ori = p.getEulerFromQuaternion(goal_quat)
        target_pose = np.concatenate([goal_pos, goal_ori])
        sample_fc_grasp = GraspSampler(self.env, obs, mu=mu, slacky_collision=True)
        org_joint_conf = obs['robot_position']
        org_joint_vel = obs['robot_velocity']

        # if self.env.visualization:
        #     vis_cubeori = VisualCubeOrientation(obs['object_position'], obs['object_orientation'])
        #     vis_goalori = VisualCubeOrientation(goal_pos, goal_quat)
        # else:
        #     vis_cubeori = None

        counter = 0
        cube_path = None
        from .utils import keep_state
        while cube_path is None and counter < retry_grasp:
            with keep_state(self.env):
                goal_threshold = ((counter / retry_grasp)
                                * (max_goal_threshold - min_goal_threshold)
                                + min_goal_threshold)
                cube_tip_positions, current_tip_positions, joint_conf = sample_fc_grasp(cube_halfwidth, shrink_region=0.35)
                self.env.platform.simfinger.reset_finger_positions_and_velocities(joint_conf)
                cube_path, joint_conf_path = plan_wholebody_motion(
                    self.env.platform.cube.block,
                    dummy_links,
                    self.env.platform.simfinger.finger_id,
                    self.env.platform.simfinger.pybullet_link_indices,
                    target_pose,
                    current_tip_positions,
                    cube_tip_positions,
                    init_joint_conf=joint_conf,
                    ik=self.env.platform.simfinger.pinocchio_utils.inverse_kinematics,
                    obstacles=[self.env.platform.simfinger.finger_id],
                    disabled_collisions=self._disabled_collisions,
                    custom_limits=custom_limits,
                    resolutions=resolutions,
                    diagnosis=False,
                    max_distance=-COLLISION_TOLERANCE,
                    # vis_fn=vis_cubeori.set_state,
                    iterations=20,
                    use_rrt=use_rrt,
                    use_incremental_rrt=use_incremental_rrt,
                    use_ori=use_ori,
                    goal_threshold=goal_threshold,
                    restarts=1
                )
                counter += 1

                # # reset cube position
                # self.env.platform.cube.block.set_state(obs['object_position'],
                #                                        obs['object_orientation'])
                # reset cube position and velocity
                # set_body_state(self.env.platform.cube.block,
                #                obs['object_position'], obs['object_orientation'],
                #                org_obj_vel)


        # reset joint conf
        self.env.platform.simfinger.reset_finger_positions_and_velocities(org_joint_conf, org_joint_vel)

        one_pose = (len(np.shape(cube_path)) == 1)
        if one_pose: cube_path = [cube_path]
        tip_path = self._get_tip_path(cube_tip_positions, cube_path)
        return Path(cube_path, joint_conf_path, tip_path, cube_tip_positions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--num_episodes", type=int, default=1, help="number of episodes")
    args = parser.parse_args()

    set_seed(args.seed)
    reward_fn = 'task1_reward'
    termination_fn = 'position_close_to_goal'
    initializer = 'task4_small_rot_init'
    env = make_training_env(reward_fn, termination_fn, initializer,
                            action_space='position', visualization=True, rank=args.seed)
    env = env.env  # HACK to remove FlatObservationWrapper
    for i in range(args.num_episodes):
        obs = env.reset()

        goal_pos = obs["goal_object_position"]
        goal_quat = obs["goal_object_orientation"]
        planner = WholeBodyPlanner(env)
        path = planner.plan(obs, goal_pos, goal_quat, use_rrt=True,
                            use_ori=True, min_goal_threshold=0.01,
                            max_goal_threshold=0.3)

        if path.cube is None:
            print('PATH is NOT found...')
            quit()

        vis_cubeori = VisualCubeOrientation(obs['object_position'], obs['object_orientation'])

        # clear some windows in GUI
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        # change camera parameters # You can also rotate the camera by CTRL + drag
        p.resetDebugVisualizerCamera( cameraDistance=0.6, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0,0,0])
        # p.startStateLogging( p.STATE_LOGGING_VIDEO_MP4, f'wholebody_planning_{args.seed}.mp4')
        for cube_pose, joint_conf in zip(path.cube, path.joint_conf):
            point, ori = cube_pose[:3], cube_pose[3:]
            quat = p.getQuaternionFromEuler(ori)
            for i in range(3):
                p.resetBasePositionAndOrientation(env.platform.cube.block, point, quat)
                env.platform.simfinger.reset_finger_positions_and_velocities(joint_conf)
                vis_cubeori.set_state(point, quat)
                time.sleep(0.01)
