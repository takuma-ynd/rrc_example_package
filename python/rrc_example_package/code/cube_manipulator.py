#!/usr/bin/env python3
from rrc_simulation.gym_wrapper.envs.cube_env import ActionType
from .fc_force_control import ForceControlPolicy, Viz, grasp_force_control
from .grasp_sampling import GraspSampler
from pybullet_planning import plan_joint_motion
from .const import *
from collections import namedtuple
from .utils import action_type_to, apply_transform, repeat, frameskip_to
from scipy.spatial.transform import Rotation as R
import functools
import numpy as np
from .align_rotation import project_cube_xy_plane, pitch_rotation_times, calc_pitching_cube_tip_position, pitch_rotation_axis_and_angle, align_z, pinocchio_solve_ik, sort_map_positions_to_fingers

class CubeManipulator:
    def __init__(self, env, viz=None, visualization=False):
        self.vis_markers = None
        self.viz = viz
        if visualization:
            from .utils import VisualMarkers
            self.vis_markers = VisualMarkers()
        self.visualization = visualization

        # TEMP:
        self.vis_markers = None

        self.env = env
        self.fc_policy = ForceControlPolicy(self.env, apply_torques=True, mu=MU, grasp_force=0.0,
                                            viz=None, use_inv_dynamics=True)  # TEMP (viz)

    def set_viz(self, viz):
        if self.visualization:
            print('viz is updated!')
            self.viz = viz

    def move_to_center(self, obs, orientation=None, force_control=True, skip_planned_motions=False):
        center = (0, 0, 0)
        orientation = obs['object_orientation'] if orientation is None else orientation
        return self.move_to_target(obs, center, orientation, force_control=force_control, avoid_top=True)

    def move_to_target(self, obs, target_pos, target_ori, force_control=True, skip_planned_motions=False, avoid_top=False, flipping=False):
        # move to grasp pose
        print('approach to grasp pose...')
        obs = self.grasp_approach(obs, avoid_top=avoid_top)

        # tighten the grasp
        print('tightening the grasp...')
        obs = self.tighten_grasp(obs)

        if force_control:
            print('moving to the goal...')
            if target_pos[2] < MIN_HEIGHT + 0.02:
                # force control towards the target position
                obs = self._run_reactive_actions(
                    obs,
                    functools.partial(self.fc_policy, target_pos=target_pos, target_quat=target_ori),
                    lambda obs: np.linalg.norm(obs['object_position'] - target_pos) < 0.05
                )
            else:
                # motion planning + force control
                raise RuntimeError('Whooa, the goal is in the sky! WIP.')
        else:
            #rotating not implemented yet
            print('moving to the goal...')
            obs = self.moving_cube(obs, target_pos, num_repeat=40)

        return obs

    def align_rotation(self, obs, yaw_planning=True):
        from .align_rotation import cube_rotation_aligned, cube_centered

        step_start = self.env.unwrapped.step_count

        #centering cube
        if not cube_centered(obs):
            obs = self.move_to_center(obs, force_control=False)
            obs = self.holds_until_everything_stops(obs)

        #stop when cube is already aligned
        if cube_rotation_aligned(obs):
            print('align_rotation succeeded!')
            return obs


        obs, cube_tip_positions = self.align_pitch(obs)
        obs = self.holds_until_everything_stops(obs)

        if yaw_planning:
            obs = self.align_yaw(obs, planning=yaw_planning)
            obs = self.holds_until_everything_stops(obs)
        else:
            projected_goal_orientation = project_cube_xy_plane(obs['goal_object_orientation'])
            z_aligned_goal_orientation, pitch_axis, pitch_angle = align_z(obs['object_orientation'], projected_goal_orientation)
            obs = self.align_yaw(obs,
                                 planning=yaw_planning,
                                 cube_tip_positions=cube_tip_positions,
                                 pitch_axis=pitch_axis,
                                 pitch_angle=pitch_angle)
            obs = self.holds_until_everything_stops(obs)

        step_end = self.env.unwrapped.step_count
        print("total steps for aligning: {}".format(step_end - step_start))
        return obs

    def align_pitch(self, obs):
        projected_goal_orientation = project_cube_xy_plane(obs['goal_object_orientation'])
        z_aligned_goal_orientation, pitch_axis, pitch_angle = align_z(obs['object_orientation'], projected_goal_orientation)

        pitch_times = pitch_rotation_times(obs['object_orientation'], obs['goal_object_orientation'])
        print('pitch_times', pitch_times)
        #tip positions relative to cube position for kicking up cube
        cube_tip_positions = calc_pitching_cube_tip_position(pitch_axis, pitch_angle)

        for i in range(pitch_times):
            print("reaching grasp position...")
            obs, cube_tip_positions, suc = self.reach_tip_positions(obs, cube_tip_positions)
            if not suc:
                break #stop before mess up environment
            print("pitching cube...")
            obs = self.pitching_cube(obs, cube_tip_positions, num_repeat=20, final_pitch=(i == pitch_times - 1))

            rotated_axis = pitch_axis
            rotated_angle = np.sign(pitch_angle) * np.pi /2
            cube_tip_positions = apply_transform(np.array([0,0,0]), R.from_euler(rotated_axis, -rotated_angle).as_quat(), cube_tip_positions)

        return obs, cube_tip_positions

    def align_yaw(self, obs, planning=True, cube_tip_positions=None, pitch_axis=None, pitch_angle=None):

        rotated_axis = None
        step_yaw_angle = np.pi/3
        for i in range(int(np.pi / step_yaw_angle)):
            if np.abs(self.get_yaw_diff(obs)) < step_yaw_angle:
                break #stop condition

            print("reaching grasp position...")
            if planning:
                cube_tip_positions, cube_pose = self.calc_yaw_tip_positions(obs)
                obs = self.grasp_approach(obs, cube_tip_pos=cube_tip_positions, cube_pose=cube_pose, in_rep=3, out_rep=8, margin_coef=1.5, flipping=False)
            else:
                assert(cube_tip_positions is not None and pitch_axis is not None and pitch_angle is not None)
                obs, cube_tip_positions, suc = self.reach_tip_positions(obs, cube_tip_positions)
                if not suc:
                    break #stop before mess up environment

            print("yawing cube...")
            #self._run_planned_actions(obs, path.joint_conf, ActionType.POSITION)
            obs, angle = self.yawing_cube(obs, cube_tip_positions, step_angle=step_yaw_angle,num_repeat=20)
            obs = self.holds_until_everything_stops(obs)

            if planning:
                pass
            else:
                rotated_axis = 'z'
                rotated_angle= angle
                cube_tip_positions = calc_pitching_cube_tip_position(pitch_axis, pitch_angle)
                cube_tip_positions = R.from_euler(rotated_axis, -rotated_angle).apply(cube_tip_positions)
                cube_tip_positions = R.from_euler(pitch_axis, -pitch_angle).apply(cube_tip_positions)

        return obs

    def calc_yaw_tip_positions(self, obs):
        from .wholebody_planning import WholeBodyPlanner

        projected_goal_orientation = project_cube_xy_plane(obs['goal_object_orientation'])
        rot_diff = (R.from_quat(obs['object_orientation']) * R.from_quat(projected_goal_orientation).inv())
        angle = -rot_diff.as_euler('ZXY')[0]
        angle_clip = np.clip(angle, -np.pi/3, np.pi/3)

        goal_quat =  (R.from_euler('Z', angle_clip) * R.from_quat(obs['object_orientation'])).as_quat()

        planner = WholeBodyPlanner(self.env)
        path = planner.plan(obs, goal_pos=obs['object_position'], goal_quat=goal_quat, retry_grasp=1000, use_incremental_rrt=True)

        cube_tip_positions = path.cube_tip_pos
        cube_pose = path.cube[0]

        return cube_tip_positions, cube_pose

    def reach_tip_positions(self, obs, cube_tip_positions):
        from .utils import repeat, ease_out
        act_seq, cube_tip_positions = self.motion_planning(cube_tip_positions, obs)
        suc = False
        if act_seq is None:
            print('Motion Planning failed!!')
            return obs, cube_tip_positions, suc
        else:
            suc = True
        act_seq = ease_out(act_seq, in_rep=3, out_rep=6)
        obs = self._run_planned_actions(obs, act_seq, ActionType.POSITION)
        return obs, cube_tip_positions, suc

    def grasp_approach(self, obs, avoid_top=False, in_rep=3, out_rep=8, **kwargs):
        from .utils import repeat, ease_out
        act_seq = self.get_grasp_approach_actions(obs, avoid_top=avoid_top, **kwargs)
        act_seq = ease_out(act_seq, in_rep=in_rep, out_rep=out_rep)
        obs = self._run_planned_actions(obs, act_seq, ActionType.POSITION, frameskip=1)
        return obs

    def pitching_cube(self, obs, cube_tip_positions, num_repeat=30, final_pitch=False):
        act_seq = self.get_actions_pitching_cube(cube_tip_positions, obs, final_pitch)
        act_seq = repeat(act_seq, num_repeat)
        obs = self._run_planned_actions(obs, act_seq, ActionType.POSITION)
        return obs

    def yawing_cube(self, obs, cube_tip_positions, step_angle=np.pi/3, num_repeat=30):

        act_seq, angle = self.get_actions_yawing_cube(cube_tip_positions, obs, step_angle=step_angle)
        act_seq = repeat(act_seq, num_repeat=num_repeat)
        obs = self._run_planned_actions(obs, act_seq, ActionType.POSITION)
        return obs, angle

    def motion_planning(self, cube_tip_positions, obs):
        grasp_target_cube_positions = cube_tip_positions * 1.5
        grasp_target_tip_positions = apply_transform(obs['object_position'], obs['object_orientation'], grasp_target_cube_positions)

        if self.vis_markers is not None:
            self.vis_markers.remove()
            self.vis_markers.add(grasp_target_tip_positions, color=TRANSLU_CYAN)

        init_tip_pos = self.env.platform.forward_kinematics(INIT_JOINT_CONF)
        inds = sort_map_positions_to_fingers(init_tip_pos, grasp_target_tip_positions)

        planned_motion = None

        for ind in inds:
            done = False

            grasp_target_tip_positions_ = grasp_target_tip_positions[ind, :]
            cube_tip_positions_ = cube_tip_positions[ind, :]

            target_joint_angles = pinocchio_solve_ik(
                self.env.platform.simfinger.pinocchio_utils.inverse_kinematics,
                obs,
                grasp_target_tip_positions_)

            if len(target_joint_angles) != 9:
                continue

            cube_id = self.env.platform.cube.block

            for i in range(30):
                planned_motion = plan_joint_motion(**self._get_grasp_conf(target_joint_angles, flipping=True))
                #reset the robot position that pybullet-planning changed
                self.env.platform.simfinger.reset_finger_positions_and_velocities(obs['robot_position'], obs['robot_velocity'])
                if planned_motion is not None:
                    done = True
                    break

            if done:
                break

        return planned_motion, cube_tip_positions_

    def get_grasp_approach_actions(self, obs, cube_tip_pos=None, cube_pose=None,
                                   margin_coef=1.3, n_trials=100, avoid_top=False,
                                   flipping=False, **kwargs):
        '''return grasp action sequence'''

        from .wholebody_planning import disable_tip_collisions
        import pybullet as p
        import time
        if (cube_tip_pos is None) != (cube_pose is None):
            raise ValueError('You must set both cube_tip_pos and cube_pose to be non-None or both to be None.')

        org_joint_conf = obs['robot_position']
        org_joint_vel = obs['robot_velocity']
        init_joint_conf = None
        # disabled_collisions = disable_tip_collisions(self.env)


        if cube_tip_pos is not None:
            from .grasping import Transform
            from .utils import IKUtils
            m_cube_tip_pos = cube_tip_pos * margin_coef  # add some margin
            cube_pos = cube_pose[:3]
            cube_quat = p.getQuaternionFromEuler(cube_pose[3:])
            m_tip_pos = Transform(cube_pos, cube_quat)(m_cube_tip_pos)

            tip_pos = Transform(cube_pos, cube_quat)(cube_tip_pos)
            if self.vis_markers is not None:
                self.vis_markers.remove()
                # self.vis_markers.add(tip_pos, color=TRANSLU_BLUE)
                self.vis_markers.add(m_tip_pos, color=TRANSLU_CYAN)

            ik_utils = IKUtils(self.env)
            jconfs = ik_utils.sample_no_collision_ik(m_tip_pos, sort_tips=False)
            self.env.platform.simfinger.reset_finger_positions_and_velocities(org_joint_conf, org_joint_vel)
            if len(jconfs) == 0:
                raise ValueError('The given tip position is not feasible')

            init_joint_conf = jconfs[0]

        for i in range(n_trials):
            if init_joint_conf is None:
                sample_fc_grasp = GraspSampler(self.env, obs, mu=MU)
                print('init_joint_conf is None --> sampling grasp points ad hoc...')
                _, _, joint_conf = sample_fc_grasp(cube_halfwidth=0.06,
                                                   shrink_region=0.2)
            else:
                joint_conf = init_joint_conf
            action_seq = plan_joint_motion(**self._get_grasp_conf(joint_conf, flipping=flipping))

            if action_seq is not None:
                break
            else:
                print('grasp approach sequence is None. Retrying...')
        if action_seq is None:
            self.env.platform.simfinger.reset_finger_positions_and_velocities(org_joint_conf, org_joint_vel)
            raise ValueError('grasp_approach failed. No path to the grasp is found.')

        # show action sequence
        # print('VISUALIZING ACTION SEQUENCE...')
        # for act in action_seq:
        #     self.env.platform.simfinger.reset_finger_positions_and_velocities(act)
        #     time.sleep(0.14)

        self.env.platform.simfinger.reset_finger_positions_and_velocities(org_joint_conf, org_joint_vel)
        return action_seq

    def _get_grasp_conf(self, joint_conf, flipping):
        MaxDist = namedtuple('MaxDist', ['dist', 'body_link_pairs'])
        cube_id = self.env.platform.cube.block
        workspace_id = 0
        if flipping:
            config = {
                'body': self.env.platform.simfinger.finger_id,
                'joints': self.env.platform.simfinger.pybullet_link_indices,
                'end_conf': joint_conf,
                'self_collisions':True,
                'iterations': 2000,
                'obstacles': [cube_id, 0],
                'max_distance': 0.01,
                'diagnosis': False,
                'smooth': 1200
            }
        else:
            config = {
                'body': self.env.platform.simfinger.finger_id,
                'joints': self.env.platform.simfinger.pybullet_link_indices,
                'end_conf': joint_conf,
                'obstacles': [cube_id, 0],
                'self_collisions': True,
                'ignore_collision_steps': 6,
                'iterations': 2000,
                'max_distance': -COLLISION_TOLERANCE,
                'max_dist_on': [
                    MaxDist(dist=-1e-03, body_link_pairs=self._create_body_fingerlink_pairs(workspace_id)),
                    MaxDist(dist=0.017, body_link_pairs=self._create_body_fingerlink_pairs(cube_id))
                ],
                'diagnosis': False,
                'smooth':1200
            }
        return config

    def get_actions_moving_cube(self, obs, target_pos):
        tip_positions_list = []

        cube_tip_positions = apply_transform([0,0,0], R.from_quat(obs['object_orientation']).inv().as_quat(), obs['robot_tip_positions'] - obs['object_position'])

        #grasp
        grasp_target_cube_positions = cube_tip_positions * 0.7
        grasp_target_tip_positions = apply_transform(obs['object_position'], obs['object_orientation'], grasp_target_cube_positions)
        tip_positions_list.append(grasp_target_tip_positions)

        position = target_pos - obs['object_position']
        position[2] = 0
        dist = np.linalg.norm(position)

        if dist > 0.1:
            #waypoint
            position = obs['object_position']
            position[0] = (position[0] + target_pos[0]) /2
            position[1] = (position[1] + target_pos[1]) /2
            target_tip_positions = apply_transform(position,
                                                obs['object_orientation'],
                                                grasp_target_cube_positions)
            tip_positions_list.append(target_tip_positions)

        #move
        position = obs['object_position']
        position[0] = target_pos[0]
        position[1] = target_pos[1]
        target_tip_positions = apply_transform(position,
                                            obs['object_orientation'],
                                            grasp_target_cube_positions)
        tip_positions_list.append(target_tip_positions)

        #release
        grasp_target_cube_positions = cube_tip_positions * 2.0
        grasp_target_tip_positions = apply_transform(position+np.array([0, 0, 0.02]), obs['object_orientation'], grasp_target_cube_positions)
        tip_positions_list.append(grasp_target_tip_positions)

        return self.tip_positions_to_actions(tip_positions_list, obs)

    def get_actions_pitching_cube(self, cube_tip_positions, obs, final_pitch=False):
        '''return list of actions that achieve grasp -> liftup -> pitch -> place -> release'''
        '''assume cube_tip_positions are on the centers of three faces'''

        from .action_sequences import ScriptedActions
        action_sequence = ScriptedActions(cube_tip_positions)

        action_sequence.add_grasp(obs, coef=0.6)
        action_sequence.add_liftup(obs, coef=0.6)
        rotate_axis, rotate_angle = pitch_rotation_axis_and_angle(cube_tip_positions)
        action_sequence.add_pitch_rotation(obs, rotate_axis, rotate_angle, coef=0.6, vis_markers=self.vis_markers)
        action_sequence.add_place_cube(obs, coef=0.6)

        if final_pitch:
            # Apply yaw rotation after placing the cube
            # HACK
            import copy
            dummy_obs = copy.deepcopy(obs)
            dummy_obs['object_orientation'] = action_sequence.orientation

            # check if we should run yaw rotation
            center_to_tip = action_sequence.get_tip_sequence()[-1] - dummy_obs['object_position']
            top_idx = np.argmax(center_to_tip[:, 2])
            indices = set(range(3))
            indices.remove(top_idx)
            radians = [np.arctan2(center_to_tip[idx, 1], center_to_tip[idx, 0]) for idx in indices]
            # print('degrees', [rad * 180 / np.pi for rad in radians])

            apply_yaw_rotation = True
            for rad in radians:
                if min(abs(rad % (np.pi / 2)), np.pi / 2 - abs(rad % (np.pi / 2))) > np.pi / 6:
                    apply_yaw_rotation = False

            if apply_yaw_rotation:
                action_sequence.add_yaw_rotation(dummy_obs, vis_markers=self.vis_markers)

        action_sequence.add_release(obs)
        return self.tip_positions_to_actions(action_sequence.get_tip_sequence(), obs)

    def get_actions_yawing_cube(self, cube_tip_positions, obs, step_angle=np.pi/3):
        '''return list of actions that achieve grasp -> yaw -> release'''

        from .action_sequences import ScriptedActions
        action_sequence = ScriptedActions(cube_tip_positions)

        action_sequence.add_grasp(obs)
        angle_clipped = action_sequence.add_yaw_rotation(obs, step_angle=step_angle, vis_markers=self.vis_markers)
        action_sequence.add_release(obs)
        return self.tip_positions_to_actions(action_sequence.get_tip_sequence(), obs), angle_clipped

    def get_yaw_diff(self, obs):
        projected_goal_orientation = project_cube_xy_plane(obs['goal_object_orientation'])
        rot_diff = (R.from_quat(obs['object_orientation']) * R.from_quat(projected_goal_orientation).inv())
        yaw_diff = -rot_diff.as_euler('ZXY')[0]

        return yaw_diff

    def _create_body_fingerlink_pairs(self, body_id):
        body_link_pairs = set()
        for link_id in self.env.platform.simfinger.pybullet_link_indices:
            body_link_pairs.add(((body_id, -1), (self.env.platform.simfinger.finger_id, link_id)))
        return body_link_pairs

    def tighten_grasp(self, obs):
        obs = grasp_force_control(self.env, obs, self.fc_policy, grasp_force=0.8,
                                  steps=6, sleep=0)
        return obs

    def release(self, obs):
        '''loose grasp of the cube and drop it'''
        with action_type_to(ActionType.TORQUE, self.env):
            obs = grasp_force_control(self.env, obs, self.fc_policy, grasp_force=-0.2,
                                      steps=4, sleep=0)

        #obs = self._move_joints_toward_home(obs, steps=50)
        return obs

    def moving_cube(self, obs, target_pos, num_repeat=3):
        from .utils import repeat
        act_seq = self.get_actions_moving_cube(obs, target_pos)
        act_seq = repeat(act_seq, num_repeat)
        obs = self._run_planned_actions(obs, act_seq, ActionType.POSITION)

        return obs

    def holds_until_everything_stops(self, obs, pos_tolerance=5e-4,
                                     ori_tolerance=5e-3, robot_vel_tolerance=1.5, max_steps=30,
                                     frameskip=1):
        from scipy.spatial.transform import Rotation
        def is_moving(prev_obs, curr_obs):
            if prev_obs is None:
                return True
            prev_pos = prev_obs['object_position']
            curr_pos = curr_obs['object_position']
            prev_rot = Rotation.from_quat(prev_obs['object_orientation'])
            curr_rot = Rotation.from_quat(curr_obs['object_orientation'])

            ori_error = (curr_rot.inv() * prev_rot).magnitude()
            pos_error = np.linalg.norm(prev_pos - curr_pos)
            robot_vel = np.linalg.norm(curr_obs['robot_velocity'])
            # print('is_moving (pos):\t{:.6f}'.format(pos_error))
            # print('is_moving (ori):\t{:.6f}'.format(ori_error))
            # print('is_moving (robot vel):\t{:.6f}'.format(robot_vel))
            return pos_error > pos_tolerance or ori_error > ori_tolerance \
                or robot_vel > robot_vel_tolerance

        step = 0
        prev_obs = None
        org_robot_pos = obs['robot_position']
        with frameskip_to(frameskip, self.env):
            with action_type_to(ActionType.POSITION, self.env):
                while is_moving(prev_obs, obs) and step < max_steps:
                    prev_obs = obs
                    obs, _, _, _ = self.env.step(org_robot_pos)
                    step += 1
                    self._maybe_update_markers(obs)
                    self._maybe_wait()
        return obs

    def _move_joints_toward_home(self, obs, steps=10):
        '''move joints toward the very initial configuration'''
        sleep = 0.02 if self.env.visualization else 0
        seq = [INIT_JOINT_CONF] * steps
        obs = self._run_planned_actions(obs, seq, ActionType.POSITION)
        return obs

    def _run_planned_actions(self, obs, action_seq, action_type, teleport=False, frameskip=2):
        # run action sequence according to action_type
        assert action_type == ActionType.POSITION
        if teleport:
            # FIXME: this doesn't work somehow
            # Force Controller fails after this reset...
            self.env.platform.simfinger.reset_finger_positions_and_velocities(action_seq[-1])
            obs['robot_position'] = action_seq[-1]
            obs['robot_tip_positions'] = self.env.platform.simfinger.pinocchio_utils.forward_kinematics(action_seq[-1])
            return obs

        # run the action sequence
        step_start = self.env.unwrapped.step_count
        with frameskip_to(frameskip, self.env):
            with action_type_to(action_type, self.env):
                for action in action_seq:
                    action = np.asarray(action)
                    obs, reward, done, info = self.env.step(action)
                    self._maybe_wait()
        step_end = self.env.unwrapped.step_count
        print("step: {}".format(step_end - step_start))
        return obs

    def _run_planned_actions2(self, obs, action_seq, action_type, frameskip=2):
        assert action_type == ActionType.POSITION
        all_action = {
            'torque': (self.env.action_space['torque'].sample() * 0).astype(np.float64),
            'position': (INIT_JOINT_CONF).astype(np.float64)
        }
        with frameskip_to(frameskip, self.env):
            for action in action_seq:
                all_action['position'] = np.asarray(action)
                obs, reward, done, info = self.env.step(all_action)
        return obs

    def _run_reactive_actions(self, obs, policy, success_cond, max_steps=300, frameskip=2):
        steps = 0
        with frameskip_to(frameskip, self.env):
            with action_type_to(ActionType.TORQUE, self.env):
                while not success_cond(obs) and steps < max_steps:
                    torque = policy(obs)
                    obs, reward, done, info = self.env.step(torque)
                    steps += 1
                    self._maybe_wait()
        return obs

    def tip_positions_to_actions(self, tip_positions_list, obs):
        ik = self.env.unwrapped.platform.simfinger.pinocchio_utils.inverse_kinematics

        actions = []
        for tip_positions in tip_positions_list:
            target_joint_conf = []
            for i in range(3):
                target_joint = ik(i, tip_positions[i], obs['robot_position'])
                try:
                    target_joint_conf.append(target_joint[3*i:3*(i+1)])
                except TypeError:
                    return actions
            action = np.concatenate(target_joint_conf)
            actions.append(action)

        return actions

    def _maybe_wait(self):
        import time
        if self.env.visualization:
            time.sleep(0.01)

    def _maybe_update_markers(self, obs):
        if self.viz is not None:
            if not self.viz.initialized:
                self.viz.reset(obs)
            else:
                self.viz.update_cube_orientation(obs)



def run_episode(args):
    from .training_env import make_training_env
    import pybullet as p
    reward_fn = 'task1_reward'
    termination_fn = 'pos_and_rot_close_to_goal'
    # initializer = 'task4_small_rot_init'
    initializer = 'task1_init'
    env = make_training_env(reward_fn, termination_fn, initializer,
                            action_space='torque',
                            init_joint_conf=False,
                            visualization=True,
                            frameskip=3,  # Idk how it affects the performance...
                            grasp='pinch', rank=args.seed)
    env = env.env  # HACK to remove FLatObservationWrapper
    cube_manipulator = CubeManipulator(env)

    obs = env.reset()
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(cameraDistance=0.6, cameraYaw=0,
                                 cameraPitch=-40,
                                 cameraTargetPosition=[0, 0, 0])

    goal_pos = obs['goal_object_position']
    goal_pos[2] += 0.01
    goal_ori = obs['object_orientation']  # NOTE: I'm using object ori, but not goal ori
    cube_manipulator.move_to_target(obs, goal_pos, goal_ori)  #, skip_planned_motions=True)
    # cube_manipulator.move_to_center(obs)  #, skip_planned_motions=True)


def run_align_episode(args):
    from .training_env import make_training_env
    import time
    import pybullet as p
    reward_fn = 'task4_reward'
    termination_fn = 'pos_and_rot_close_to_goal'
    initializer = 'task4_init'
    env = make_training_env(reward_fn, termination_fn, initializer,
                            action_space='torque_and_position',
                            visualization=args.visualize,
                            frameskip=3,
                            monitor=args.record,
                            grasp='pinch', rank=args.seed)
    env = env.env  # HACK to remove FLatObservationWrapper
    cube_manipulator = CubeManipulator(env)
    obs = env.reset()

    cube_manipulator.align_rotation(obs)

    # get rotation difference
    rot = R.from_quat(obs['object_orientation'])
    rot_goal = R.from_quat(obs['goal_object_orientation'])
    print('rot diff', (rot.inv() * rot_goal).magnitude() / np.pi)
    """
    while p.isConnected():
        env.platform.simfinger.reset_finger_positions_and_velocities(obs['robot_position'])
        p.stepSimulation()
        time.sleep(0.01)
    """

if __name__ == '__main__':
    import argparse
    from .utils import set_seed
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--num_episodes", type=int, default=1, help="number of episodes")
    parser.add_argument("--visualize", action='store_true')
    parser.add_argument("--record", action='store_true', help="save file name")
    args = parser.parse_args()
    set_seed(args.seed)
    # run_episode(args)
    run_align_episode(args)
