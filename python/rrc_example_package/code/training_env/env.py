"""Gym environment for training in simulation."""
import numpy as np
import gym

import rrc_simulation
from rrc_simulation.gym_wrapper.envs import cube_env
from rrc_simulation.tasks import move_cube
from rrc_simulation import visual_objects
from rrc_simulation import TriFingerPlatform
from rrc_simulation.action import Action
from rrc_simulation.code.utils import VisualCubeOrientation


class TrainingEnv(gym.Env):
    """Gym environment for moving cubes with simulated TriFingerPro."""

    def __init__(
        self,
        reward_fn,
        termination_fn,
        initializer=None,
        action_type=cube_env.ActionType.POSITION,
        kp_coef=None,
        kd_coef=None,
        frameskip=1,
        visualization=False,
        is_level_4=False
    ):
        """Initialize.

        Args:
            reward_fn: The reward fn to use.
            initializer: Initializer class for providing initial cube pose and
                goal pose. If no initializer is provided, we will initialize in a way
                which is be helpful for learning.
            action_type (ActionType): Specify which type of actions to use.
                See :class:`ActionType` for details.
            kp_coef float:  Coefficient for P-gain for position controller.  Set to
                NaN to use default gain for the corresponding joint.
            kd_coef float:  Coefficient for D-gain for position controller.  Set to
                NaN to use default gain for the corresponding joint.
            frameskip (int):  Number of actual control steps to be performed in
                one call of step().
            visualization (bool): If true, the pyBullet GUI is run for
                visualization.
        """
        # Basic initialization
        # ====================

        self._compute_reward = reward_fn
        self._termination_fn = termination_fn
        self.initializer = initializer
        self.action_type = action_type
        self.visualization = visualization
        self.is_level_4 = is_level_4
        self.ori_marker = None
        self.ori_goal_marker = None

        self.kp_coef = 1.0 if kp_coef is None else kp_coef
        self.kd_coef = 1.0 if kd_coef is None else kd_coef

        if frameskip < 1:
            raise ValueError("frameskip cannot be less than 1.")
        self.frameskip = frameskip

        # will be initialized in reset()
        self.platform = None

        # Create the action and observation spaces
        # ========================================

        spaces = TriFingerPlatform.spaces

        if self.action_type == cube_env.ActionType.TORQUE:
            self.action_space = spaces.robot_torque.gym
        elif self.action_type == cube_env.ActionType.POSITION:
            self.action_space = spaces.robot_position.gym
        elif self.action_type == cube_env.ActionType.TORQUE_AND_POSITION:
            self.action_space = gym.spaces.Dict(
                {
                    "torque": spaces.robot_torque.gym,
                    "position": spaces.robot_position.gym,
                }
            )
        else:
            raise ValueError("Invalid action_type")

        self.observation_names = [
            "robot_position",
            "robot_velocity",
            "robot_tip_positions",
            "object_position",
            "object_orientation",
            "goal_object_position",
            "goal_object_orientation",
            "tip_force",
        ]

        self.observation_space = gym.spaces.Dict(
            {
                "robot_position": spaces.robot_position.gym,
                "robot_velocity": spaces.robot_velocity.gym,
                "robot_tip_positions": gym.spaces.Box(
                    low=np.array([spaces.object_position.low] * 3),
                    high=np.array([spaces.object_position.high] * 3),
                ),
                "object_position": spaces.object_position.gym,
                "object_orientation": spaces.object_orientation.gym,
                "goal_object_position": spaces.object_position.gym,
                "goal_object_orientation": spaces.object_orientation.gym,
                "tip_force": gym.spaces.Box(
                    low=np.zeros(3),
                    high=np.ones(3),
                )
            }
        )

    def step(self, action):
        if self.platform is None:
            raise RuntimeError("Call `reset()` before starting to step.")

        if not self.action_space.contains(action):
            raise ValueError(
                "Given action is not contained in the action space."
            )

        num_steps = self.frameskip

        # ensure episode length is not exceeded due to frameskip
        step_count_after = self.step_count + num_steps
        if step_count_after > move_cube.episode_length:
            excess = step_count_after - move_cube.episode_length
            num_steps = max(1, num_steps - excess)

        reward = 0.0
        for _ in range(num_steps):
            self.step_count += 1
            if self.step_count > move_cube.episode_length:
                raise RuntimeError("Exceeded number of steps for one episode.")

            # send action to robot
            robot_action = self._gym_action_to_robot_action(action)
            t = self.platform.append_desired_action(robot_action)

            previous_observation = self._create_observation(t)
            observation = self._create_observation(t + 1)
            self._maybe_update_visualization(observation)
            applied_action = self.platform.get_applied_action(t)

            reward += self._compute_reward(
                previous_observation=previous_observation,
                observation=observation,
                action=applied_action,
            )

        is_done = self.step_count == move_cube.episode_length
        is_done = is_done or self._termination_fn(observation)

        return observation, reward, is_done, {}

    def reset(self):
        # reset simulation
        del self.platform
        if hasattr(self, 'goal_marker'):
            del self.goal_marker

        # initialize simulation
        if self.initializer is None:
            # if no initializer is given (which will be the case during training),
            # we can initialize in any way desired. here, we initialize the cube always
            # in the center of the arena, instead of randomly, as this appears to help
            # training
            initial_robot_position = TriFingerPlatform.spaces.robot_position.default
            default_object_position = (
                TriFingerPlatform.spaces.object_position.default
            )
            default_object_orientation = (
                TriFingerPlatform.spaces.object_orientation.default
            )
            initial_object_pose = move_cube.Pose(
                position=default_object_position,
                orientation=default_object_orientation,
            )
            goal_object_pose = move_cube.sample_goal(difficulty=1)
        else:
            # if an initializer is given, i.e. during evaluation, we need to initialize
            # according to it, to make sure we remain coherent with the standard CubeEnv.
            # otherwise the trajectories produced during evaluation will be invalid.
            initial_robot_position = TriFingerPlatform.spaces.robot_position.default
            initial_object_pose=self.initializer.get_initial_state()
            goal_object_pose = self.initializer.get_goal()

        self.platform = TriFingerPlatform(
            visualization=self.visualization,
            initial_robot_position=initial_robot_position,
            initial_object_pose=initial_object_pose,
        )

        self.goal = {
            "position": goal_object_pose.position,
            "orientation": goal_object_pose.orientation,
        }
        # visualize the goal
        if self.visualization:
            if self.is_level_4:
                self.goal_marker = visual_objects.CubeMarker(
                    width=0.065,
                    position=goal_object_pose.position,
                    orientation=goal_object_pose.orientation,
                )
                self.ori_goal_marker = VisualCubeOrientation(
                    goal_object_pose.position,
                    goal_object_pose.orientation
                )

            else:
                self.goal_marker = visual_objects.SphereMaker(
                    radius=0.065 / 2,
                    position=goal_object_pose.position,
                )

        self.info = dict()

        self.step_count = 0
        init_obs = self._create_observation(0)

        if self.visualization:
            self.ori_marker = VisualCubeOrientation(
                init_obs['object_position'],
                init_obs['object_orientation']
            )

        return init_obs

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.action_space.seed(seed)
        move_cube.random = self.np_random
        return [seed]

    def _create_observation(self, t):
        robot_observation = self.platform.get_robot_observation(t)
        object_observation = self.platform.get_object_pose(t)
        robot_tip_positions = self.platform.forward_kinematics(
            robot_observation.position
        )
        robot_tip_positions = np.array(robot_tip_positions)

        observation = {
            "robot_position": robot_observation.position,
            "robot_velocity": robot_observation.velocity,
            "robot_tip_positions": robot_tip_positions,
            "object_position": object_observation.position,
            "object_orientation": object_observation.orientation,
            "goal_object_position": self.goal["position"],
            "goal_object_orientation": self.goal["orientation"],
            "tip_force": robot_observation.tip_force
        }
        return observation

    def _gym_action_to_robot_action(self, gym_action):
        # construct robot action depending on action type

        if self.action_type == cube_env.ActionType.TORQUE:
            robot_action = Action(
                torque=gym_action,
                position=np.array([np.nan] * 3 * self.platform.simfinger.number_of_fingers),
                kp=self.kp_coef * self.platform.simfinger.position_gains,
                kd=self.kd_coef * self.platform.simfinger.velocity_gains
            )
        elif self.action_type == cube_env.ActionType.POSITION:
            robot_action = Action(
                torque=np.array([0.0] * 3 * self.platform.simfinger.number_of_fingers),
                position=gym_action,
                kp=self.kp_coef * self.platform.simfinger.position_gains,
                kd=self.kd_coef * self.platform.simfinger.velocity_gains
            )
        elif self.action_type == cube_env.ActionType.TORQUE_AND_POSITION:
            robot_action = Action(
                torque=gym_action["torque"],
                position=gym_action["position"],
                kp=self.kp_coef * self.platform.simfinger.position_gains,
                kd=self.kd_coef * self.platform.simfinger.velocity_gains
            )
        else:
            raise ValueError("Invalid action_type")

        return robot_action

    def _maybe_update_visualization(self, obs):
        if self.visualization:
            self.ori_marker.set_state(
                obs['object_position'],
                obs['object_orientation']
            )
