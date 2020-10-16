#!/usr/bin/env python3
'''
Use custom_utils.sample_cube_surface_points and visualize the sampled points
'''

import random
from .training_env import make_training_env
import numpy as np
import pybullet as p
import time
from .utils import apply_transform, CylinderMarker, VisualCubeOrientation
from rrc_simulation import TriFingerPlatform
from rrc_simulation.gym_wrapper.envs import cube_env
from .training_env.wrappers import RenderWrapper
from gym.wrappers import Monitor, TimeLimit
import gym
from .grasping import Cube, CoulombFriction, Transform, CubePD, get_rotation_between_vecs, Contact, PDController
from .save_video import merge_videos
from .utils import set_seed, action_type_to
import argparse


def get_contacts(T):
    contacts = [Contact(c, T) for c in p.getContactPoints()]
    # arm body_id = 1, cube_body_id = 4
    contacts = [c for c in contacts if c.bodyA == 1 and c.bodyB == 4]
    return [c for c in contacts if c.linkA % 5 == 0]  # just look at tip contacts


def get_torque_shrink_coef(org_torque, add_torque):
    slack = 0.36 - org_torque
    neg_torque = np.sign(add_torque) == -1
    slack[neg_torque] = 0.36 + org_torque[neg_torque]
    slack[np.isclose(slack, 0.0)] = 0  # avoid numerical instability
    assert np.all(slack >= 0), f'slack: {slack}'
    ratio = slack / np.abs(add_torque)
    return min(1.0, np.min(ratio))


# def get_tip_forces(obs, wrench):
#     cube = Cube(0.0325, CoulombFriction(mu=1.0))
#     T_cube_to_base = Transform(pos=obs['object_position'],
#                                ori=obs['object_orientation'])
#     R_cube_to_base = Transform(pos=np.zeros(3), ori=obs['object_orientation'])
#     T_base_to_cube = T_cube_to_base.inverse()
#     contacts = get_contacts(T_base_to_cube)
#     if len(contacts) == 0:
#         return get_tip_forces_back_up(obs, wrench)
#     forces = cube.solve_for_tip_forces([c.TA for c in contacts], wrench)
#     if forces is None:
#         return get_tip_forces_back_up(obs, wrench)
#
#     # print(forces)
#     tips_cube_frame = T_base_to_cube(obs['robot_tip_positions'])
#     wrench_at_tips = np.zeros((3, 6))
#     for f, c in zip(forces, contacts):
#         ind = (c.linkA - 1) // 5
#         T = Transform(pos=tips_cube_frame[ind] - c.contact_posA,
#                       ori=np.array([0, 0, 0, 1]))
#         # rotate from contact frame to cube_frame
#         f[:3] = c.TA.adjoint().dot(f)[:3]
#         # translate to tip position and rotate to world axes
#         f = R_cube_to_base(T).adjoint().dot(f)
#
#         # add to tip wrenches
#         wrench_at_tips[ind] += f
#     # print(wrench_at_tips)
#     return wrench_at_tips


class Viz(object):
    def __init__(self):
        self.goal_viz = None
        self.cube_viz = None
        self.initialized = False
        self.markers = []

    def reset(self, obs):
        if self.goal_viz:
            del self.goal_viz
            del self.cube_viz

        self.goal_viz = VisualCubeOrientation(obs['goal_object_position'],
                                              obs['goal_object_orientation'])
        self.cube_viz = VisualCubeOrientation(obs['object_position'],
                                              obs['object_orientation'])
        if self.markers:
            for m in self.markers:
                del m
        self.markers = []
        self.initialized = True

    def update_cube_orientation(self, obs):
        self.cube_viz.set_state(obs['object_position'],
                                obs['object_orientation'])

    def update_tip_force_markers(self, obs, tip_wrenches, force):
        cube_force = apply_transform(np.zeros(3), obs['object_orientation'],
                                     force[None])[0]
        self._set_markers([w[:3] for w in tip_wrenches],
                          obs['robot_tip_positions'],
                          cube_force, obs['object_position'])

    def _set_markers(self, forces, tips, cube_force, cube_pos):
        from const import FORCE_COLOR_RED, FORCE_COLOR_GREEN, FORCE_COLOR_BLUE
        R = 0.005
        L = 0.2
        ms = []
        cube_force = cube_force / np.linalg.norm(cube_force)
        force_colors = [FORCE_COLOR_RED, FORCE_COLOR_GREEN, FORCE_COLOR_BLUE]
        q = get_rotation_between_vecs(np.array([0, 0, 1]), cube_force)
        if not self.markers:
            ms.append(CylinderMarker(R, L, cube_pos + 0.5 * L * cube_force,
                                     q, color=(0, 0, 1, 0.5)))
        else:
            self.markers[0].set_state(cube_pos + 0.5 * L * cube_force, q)
            ms.append(self.markers[0])

        for i, (f, t) in enumerate(zip(forces, tips)):
            f = f / np.linalg.norm(f)
            q = get_rotation_between_vecs(np.array([0, 0, 1]), f)
            if not self.markers:
                color = force_colors[i]
                ms.append(CylinderMarker(R, L, t + 0.5 * L * f,
                                         q, color=color))
            else:
                self.markers[i+1].set_state(t + 0.5 * L * f, q)
                ms.append(self.markers[i+1])
        self.markers = ms


class TipPD(object):
    def __init__(self, gains, tips_cube_frame, max_force=1.0):
        self.pd_tip1 = PDController(*gains)
        self.pd_tip2 = PDController(*gains)
        self.pd_tip3 = PDController(*gains)
        self.tips_cube_frame = tips_cube_frame
        self.max_force = max_force

    def _scale(self, x, lim):
        norm = np.linalg.norm(x)
        if norm > lim:
            return x * lim / norm
        return x

    def __call__(self, obs):
        tips = obs['robot_tip_positions']
        goal_tips = apply_transform(obs['object_position'],
                                    obs['object_orientation'],
                                    self.tips_cube_frame)
        f = [self.pd_tip1(goal_tips[0] - tips[0]),
             self.pd_tip2(goal_tips[1] - tips[1]),
             self.pd_tip3(goal_tips[2] - tips[2])]
        return [self._scale(ff, self.max_force) for ff in f]


class ForceControlPolicy(object):
    def __init__(self, env, apply_torques=True, tip_pd=None, mu=1.0,
                 grasp_force=0.0, viz=None, use_inv_dynamics=True):
        self.cube_pd = CubePD()
        self.tip_pd = tip_pd
        self.cube = Cube(0.0325, CoulombFriction(mu=mu))
        self.env = env
        self.apply_torques = apply_torques
        self.viz = viz
        self.use_inv_dynamics = use_inv_dynamics
        self.grasp_force = grasp_force

    def __call__(self, obs, target_pos=None, target_quat=None):
        target_pos = obs['goal_object_position'] if target_pos is None else target_pos
        target_quat = obs['goal_object_orientation'] if target_quat is None else target_quat
        force, torque = self.cube_pd(target_pos,
                                     target_quat,
                                     obs['object_position'],
                                     obs['object_orientation'])
        if not self.apply_torques:
            torque[:] = 0
        wrench_cube_frame = np.concatenate([force, torque])
        tip_wrenches = self.get_tip_forces(obs, wrench_cube_frame)
        if self.tip_pd:
            tip_forces = self.tip_pd(obs)
            for i in range(3):
                tip_wrenches[i][:3] += tip_forces[i]
        if self.viz:
            self.viz.update_tip_force_markers(obs, tip_wrenches, force)

        Js = self._calc_jacobians(obs)
        torque = sum([Js[i].T.dot(tip_wrenches[i]) for i in range(3)])
        if self.use_inv_dynamics:
            stable_torque = self.inverse_dynamics(
                                         obs, self.env.platform.simfinger.finger_id)

            # shrink torque control to be within torque limits
            if np.any(np.abs(stable_torque + torque) > 0.36):
                slack = 0.36 - stable_torque
                neg_torque = np.sign(torque) == -1
                slack[neg_torque] = 0.36 + stable_torque[neg_torque]
                assert np.all(slack > 0)
                ratio = slack / np.abs(torque)
                torque = torque * min(1.0, np.min(ratio))
        else:
            stable_torque = np.zeros_like(torque)
            if np.any(np.abs(torque) > 0.36):
                torque = torque * 0.36 / np.max(np.abs(torque))

        if self.grasp_force != 0:
            grasp_torque = self.get_grasp_torque(obs, f=self.grasp_force, Js=Js)
            coef = get_torque_shrink_coef(torque + stable_torque, grasp_torque)
            grasp_torque = coef * grasp_torque
        else:
            grasp_torque = torque * 0
        # if coef > 0:
        #     print('grasp torque!!', coef)
        return torque + stable_torque + grasp_torque

    def get_tip_forces(self, obs, wrench):
        T_cube_to_base = Transform(pos=obs['object_position'],
                                   ori=obs['object_orientation'])
        tips = obs['robot_tip_positions']
        tips_cube_frame = T_cube_to_base.inverse()(tips)
        contacts = [self.cube.contact_from_tip_position(tip)
                    for tip in tips_cube_frame]
        tip_forces = self.cube.solve_for_tip_forces(contacts, wrench)
        if tip_forces is None:
            tip_forces = np.zeros((3, 6))
        # Rotate forces to world frame axes.
        for i, (f, c) in enumerate(zip(tip_forces, contacts)):
            tip_forces[i][:3] = T_cube_to_base(c).adjoint().dot(f)[:3]
        return tip_forces

    def inverse_dynamics(self, obs, id):
        torque = p.calculateInverseDynamics(id, list(obs['robot_position']),
                                            list(obs['robot_velocity']),
                                            [0. for _ in range(9)])
        return np.array(torque)

    def get_grasp_torque(self, obs, f, Js=None):
        if Js is None:
            Js = self._calc_jacobians(obs)
        tip_to_center = obs['object_position'] - obs['robot_tip_positions']
        inward_force_vec = f * (tip_to_center / np.linalg.norm(obs['robot_tip_positions'] - obs['object_position'], axis=1).T)
        inward_force_vec = np.hstack((inward_force_vec, np.zeros((3,3))))
        torque = sum([Js[i].T.dot(inward_force_vec[i]) for i in range(3)])
        return torque

    def _calc_jacobians(self, obs):
        Js = [self.env.unwrapped.platform.simfinger.pinocchio_utils.compute_jacobian(
                                    i, obs['robot_position']) for i in range(3)]
        return Js


def grasp_tippos_control(env, obs):
    ik = env.unwrapped.platform.simfinger.pinocchio_utils.inverse_kinematics

    grasp_target_cube_positions = env.cube_tip_positions * 0.5
    grasp_target_tip_positions = apply_transform(obs['object_position'],
                                                 obs['object_orientation'],
                                                 grasp_target_cube_positions)
    target_joint_conf = []
    for i in range(3):
        target_joint = ik(i, grasp_target_tip_positions[i], obs['robot_position'])
        target_joint_conf.append(target_joint[3*i:3*(i+1)])
    grasp_action = np.concatenate(target_joint_conf)

    # Grasp the cube first
    env.unwrapped.action_space = gym.spaces.Dict(
                {
                    "torque": TriFingerPlatform.spaces.robot_torque.gym,
                    "position": TriFingerPlatform.spaces.robot_position.gym,
                })
    env.unwrapped.action_type = cube_env.ActionType.TORQUE_AND_POSITION
    for i in range(10):
        obs, reward, done, info = env.step({'torque': np.zeros(9),
                                            'position': grasp_action})
        time.sleep(0.01)
    return obs


def grasp_force_control(env, obs, force_controller, grasp_force, steps=10, sleep=0):
    # Then move toward the goal positions
    from rrc_simulation.gym_wrapper.envs.cube_env import ActionType
    import time
    with action_type_to(ActionType.TORQUE, env):
        for i in range(steps):
            grasp_torq = force_controller.get_grasp_torque(obs, f=grasp_force)
            stable_torq = force_controller.inverse_dynamics(obs, env.platform.simfinger.finger_id)
            coef = get_torque_shrink_coef(stable_torq, grasp_torq)
            torq = stable_torq + coef * grasp_torq
            obs, reward, done, info = env.step(torq)
            if sleep > 0:
                time.sleep(sleep)
    return obs

def run_episodes(neps, seed):
    reward_fn = 'task1_reward'
    termination_fn = 'pos_and_rot_close_to_goal'
    # termination_fn = 'position_close_to_goal'
    initializer = 'task4_init'
    env = make_training_env(reward_fn, termination_fn, initializer,
                            action_space='torque_and_position',
                            init_joint_conf=True,
                            visualization=True,
                            grasp='pinch',
                            rank=seed)
    env = env.env  # HACK to remove FLatObservationWrapper
    # tmp_dir = '/tmp/video'
    # env = Monitor(RenderWrapper(TimeLimit(env, 1000)), tmp_dir,
    #               video_callable=lambda episode_id: True, mode='evaluation',
    #               force=True)
    env = TimeLimit(env, 1000)
    viz = Viz()
    for _ in range(neps):
        obs = env.reset()

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=0,
                                     cameraPitch=-40,
                                     cameraTargetPosition=[0, 0, 0])
        viz.reset(obs)
        # tip_pd = TipPD([10, 1], 0.7 * env.cube_tip_positions)
        tip_pd = None
        controller = ForceControlPolicy(env, True, tip_pd)
        # obs = grasp_force_control(env, obs, controller.get_grasp_torque)
        obs = grasp_tippos_control(env, obs)

        # Then move toward the goal positions
        env.unwrapped.action_space = TriFingerPlatform.spaces.robot_torque.gym
        env.unwrapped.action_type = cube_env.ActionType.TORQUE
        done = False
        while not done:
            # transform wrenches to base frame
            torque = controller(obs)
            obs, reward, done, info = env.step(torque)
            viz.update_cube_orientation(obs)
            time.sleep(0.01)

    env.close()
    # video_file = "./force_control.mp4"
    # merge_videos(video_file, tmp_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--num_episodes", type=int, default=1, help="number of episodes")
    args = parser.parse_args()
    set_seed(args.seed)
    run_episodes(args.num_episodes, args.seed)
