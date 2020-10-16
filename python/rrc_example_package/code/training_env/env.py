"""Gym environment for training in simulation."""
import numpy as np
import gym

import rrc_simulation
from rrc_simulation.tasks import move_cube
from rrc_simulation import visual_objects
from rrc_simulation import TriFingerPlatform
from rrc_simulation.action import Action
from rrc_simulation.code.utils import VisualCubeOrientation
import robot_fingers


class ActionType(enum.Enum):
    """Different action types that can be used to control the robot."""

    #: Use pure torque commands.  The action is a list of torques (one per
    #: joint) in this case.
    TORQUE = enum.auto()
    #: Use joint position commands.  The action is a list of angular joint
    #: positions (one per joint) in this case.  Internally a PD controller is
    #: executed for each action to determine the torques that are applied to
    #: the robot.
    POSITION = enum.auto()
    #: Use both torque and position commands.  In this case the action is a
    #: dictionary with keys "torque" and "position" which contain the
    #: corresponding lists of values (see above).  The torques resulting from
    #: the position controller are added to the torques in the action before
    #: applying them to the robot.
    TORQUE_AND_POSITION = enum.auto()


class TrainingEnv(gym.Env):
    """Gym environment for moving cubes with simulated TriFingerPro."""

    def __init__(
        self,
        reward_fn,
        termination_fn,
        initializer=None,
        action_type=ActionType.POSITION,
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

        # self._compute_reward = reward_fn
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

        robot_torque_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_torque.low,
            high=trifingerpro_limits.robot_torque.high,
        )
        robot_position_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_position.low,
            high=trifingerpro_limits.robot_position.high,
        )
        robot_velocity_space = gym.spaces.Box(
            low=trifingerpro_limits.robot_velocity.low,
            high=trifingerpro_limits.robot_velocity.high,
        )

        object_state_space = gym.spaces.Dict(
            {
                "position": gym.spaces.Box(
                    low=trifingerpro_limits.object_position.low,
                    high=trifingerpro_limits.object_position.high,
                ),
                "orientation": gym.spaces.Box(
                    low=trifingerpro_limits.object_orientation.low,
                    high=trifingerpro_limits.object_orientation.high,
                ),
            }
        )
        # spaces = TriFingerPlatform.spaces

        if self.action_type == ActionType.TORQUE:
            self.action_space = robot_torque_space
            self._initial_action = trifingerpro_limits.robot_torque.default
        elif self.action_type == ActionType.POSITION:
            self.action_space = robot_position_space
            self._initial_action = trifingerpro_limits.robot_position.default
        elif self.action_type == ActionType.TORQUE_AND_POSITION:
            self.action_space = gym.spaces.Dict(
                {
                    "torque": robot_torque_space,
                    "position": robot_position_space
                }
            )
            self._initial_action = {
                "torque": trifingerpro_limits.robot_torque.default,
                "position": trifingerpro_limits.robot_position.default,
            }
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
                "robot_position": robot_position_space,
                "robot_velocity": robot_velocity_space,
                "robot_tip_positions": gym.spaces.Box(
                    low=np.array([trifingerpro_limits.object_position.low] * 3),
                    high=np.array([trifingerpro_limits.object_position.high] * 3),
                ),
                "object_position": object_state_space['position'],
                "object_orientation": object_state_space['orientation'],
                "goal_object_position": object_state_space['position'],
                "goal_object_orientation": object_state_space['orientation'],
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
            # send action to robot
            robot_action = self._gym_action_to_robot_action(action)
            t = self.platform.append_desired_action(robot_action)

            # previous_observation = self._create_observation(t)
            # observation = self._create_observation(t + 1)
            observation = self._create_observation(t, action)
            # observation = self._create_observation(t + 1)
            self._maybe_update_visualization(observation)
            # applied_action = self.platform.get_applied_action(t)

            # reward += self._compute_reward(
            #     previous_observation=previous_observation,
            #     observation=observation,
            #     action=applied_action,
            # )

            reward += self.compute_reward(
                observation["achieved_goal"],
                observation["desired_goal"],
                self.info,
            )

            self.step_count = t
            # make sure to not exceed the episode length
            if self.step_count >= move_cube.episode_length - 1:
                break

        is_done = self.step_count == move_cube.episode_length
        is_done = is_done or self._termination_fn(observation)

        return observation, reward, is_done, self.info

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
            # default_object_position = (
            #     TriFingerPlatform.spaces.object_position.default
            # )
            # default_object_orientation = (
            #     TriFingerPlatform.spaces.object_orientation.default
            # )
            # initial_object_pose = move_cube.Pose(
            #     position=default_object_position,
            #     orientation=default_object_orientation,
            # )
            goal_object_pose = move_cube.sample_goal(difficulty=1)
        else:
            # if an initializer is given, i.e. during evaluation, we need to initialize
            # according to it, to make sure we remain coherent with the standard CubeEnv.
            # otherwise the trajectories produced during evaluation will be invalid.
            initial_robot_position = TriFingerPlatform.spaces.robot_position.default
            # initial_object_pose=self.initializer.get_initial_state()
            goal_object_pose = self.initializer.get_goal()

        # self.platform = TriFingerPlatform(
        #     visualization=self.visualization,
        #     initial_robot_position=initial_robot_position,
        #     initial_object_pose=initial_object_pose,
        # )
        self._reset_platform_frontend()

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
        observation, _, _, _ = self.step(self._initial_action)
        # init_obs = self._create_observation(0)

        if self.visualization:
            self.ori_marker = VisualCubeOrientation(
                init_obs['object_position'],
                init_obs['object_orientation']
            )

        return init_obs


    # Copied from cube_env.py
    def compute_reward(self, achieved_goal, desired_goal, info):
        """Compute the reward for the given achieved and desired goal.

        Args:
            achieved_goal (dict): Current pose of the object.
            desired_goal (dict): Goal pose of the object.
            info (dict): An info dictionary containing a field "difficulty"
                which specifies the difficulty level.

        Returns:
            float: The reward that corresponds to the provided achieved goal
            w.r.t. to the desired goal. Note that the following should always
            hold true::

                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(
                    ob['achieved_goal'],
                    ob['desired_goal'],
                    info,
                )
        """
        return -move_cube.evaluate_state(
            move_cube.Pose.from_dict(desired_goal),
            move_cube.Pose.from_dict(achieved_goal),
            info["difficulty"],
        )
    def _reset_platform_frontend(self):
        """Reset the platform frontend."""
        # reset is not really possible
        if self.platform is not None:
            raise RuntimeError(
                "Once started, this environment cannot be reset."
            )

        self.platform = robot_fingers.TriFingerPlatformFrontend()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.action_space.seed(seed)
        move_cube.random = self.np_random
        return [seed]

    def _create_observation(self, t, action):
        robot_observation = self.platform.get_robot_observation(t)
        # object_observation = self.platform.get_object_pose(t)
        robot_tip_positions = self.platform.forward_kinematics(
            robot_observation.position
        )
        robot_tip_positions = np.array(robot_tip_positions)

        observation = {
            "robot_position": robot_observation.position,
            "robot_velocity": robot_observation.velocity,
            "robot_tip_positions": robot_tip_positions,
            # "object_position": object_observation.position,
            # "object_orientation": object_observation.orientation,
            "goal_object_position": self.goal["position"],
            "goal_object_orientation": self.goal["orientation"],
            "tip_force": robot_observation.tip_force
        }
        observation.update({
            "object_position": camera_observation.object_pose.position,  # achieved_goal.position
            "object_orientation": camera_observation.object_pose.orientation,  # achieved_goal.orientation
            "action": action,
        })
        return observation

    def _gym_action_to_robot_action(self, gym_action):
        # construct robot action depending on action type

        if self.action_type == ActionType.TORQUE:
            robot_action = Action(
                torque=gym_action,
                position=np.array([np.nan] * 3 * self.platform.simfinger.number_of_fingers),
                kp=self.kp_coef * self.platform.simfinger.position_gains,
                kd=self.kd_coef * self.platform.simfinger.velocity_gains
            )
        elif self.action_type == ActionType.POSITION:
            robot_action = Action(
                torque=np.array([0.0] * 3 * self.platform.simfinger.number_of_fingers),
                position=gym_action,
                kp=self.kp_coef * self.platform.simfinger.position_gains,
                kd=self.kd_coef * self.platform.simfinger.velocity_gains
            )
        elif self.action_type == ActionType.TORQUE_AND_POSITION:
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
