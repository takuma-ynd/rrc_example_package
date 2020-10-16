from .env import TrainingEnv
from .wrappers import IKActionWrapper, FlatObservationWrapper, JointConfInitializationWrapper, ResidualLearningWrapper, ResidualLearningFCWrapper, CubeRotationAlignWrapper, ResidualLearningMotionPlanningFCWrapper, PyBulletClearGUIWrapper


def get_initializer(name):
    from . import initializers
    if hasattr(initializers, name):
        return getattr(initializers, name)
    else:
        raise ValueError(f"Can't find initializer: {name}")


def get_reward_fn(name):
    from . import reward_fns
    if hasattr(reward_fns, name):
        return getattr(reward_fns, name)
    else:
        raise ValueError(f"Can't find reward function: {name}")


def get_termination_fn(name):
    from . import termination_fns
    if hasattr(termination_fns, name):
        return getattr(termination_fns, name)
    elif hasattr(termination_fns, "generate_" + name):
        return getattr(termination_fns, "generate_" + name)()
    else:
        raise ValueError(f"Can't find termination function: {name}")


def make_training_env(reward_fn, termination_fn, initializer, action_space,
                      init_joint_conf=False, residual=False, kp_coef=None,
                      kd_coef=None, frameskip=1, rank=0, visualization=False,
                      grasp='pinch', monitor=False):
    from rrc_simulation.gym_wrapper.envs import cube_env
    is_level_4 = 'task4' in reward_fn
    reward_fn = get_reward_fn(reward_fn)
    initializer = get_initializer(initializer)
    termination_fn = get_termination_fn(termination_fn)
    if action_space not in ['torque', 'position', 'ik', 'torque_and_position', 'position_and_torque']:
        raise ValueError("Action Space must be one of: 'torque', 'position', 'ik'.")
    if action_space == 'torque':
        action_type = cube_env.ActionType.TORQUE
    elif action_space in ['torque_and_position', 'position_and_torque']:
        action_type = cube_env.ActionType.TORQUE_AND_POSITION
    else:
        action_type = cube_env.ActionType.POSITION
    env = TrainingEnv(reward_fn=reward_fn,
                      termination_fn=termination_fn,
                      initializer=initializer,
                      kp_coef=kp_coef,
                      kd_coef=kd_coef,
                      frameskip=frameskip,
                      action_type=action_type,
                      visualization=visualization,
                      is_level_4=is_level_4)
    env.seed(seed=rank)
    env.action_space.seed(seed=rank)
    if visualization:
        env = PyBulletClearGUIWrapper(env)
    if monitor:
        from gym.wrappers import Monitor
        from rrc_simulation.code.training_env.wrappers import RenderWrapper
        from rrc_simulation.code.const import TMP_VIDEO_DIR
        env = Monitor(
            RenderWrapper(env),
            TMP_VIDEO_DIR,
            video_callable=lambda episode_id: True,
            mode='evaluation'
        )

    if action_space == 'ik':
        env = IKActionWrapper(env)
    if residual:
        if action_space == 'torque':
            # env = JointConfInitializationWrapper(env, heuristic=grasp)
            env = ResidualLearningFCWrapper(env, apply_torques=is_level_4,
                                            is_level_4=is_level_4)
        elif action_space == 'torque_and_position':
            env = ResidualLearningMotionPlanningFCWrapper(env, apply_torques=is_level_4,
                                                          action_repeat=2, align_goal_ori=is_level_4,
                                                          use_rrt=is_level_4,
                                                          init_cube_manip='flip_and_grasp' if is_level_4 else 'grasp',
                                                          evaluation=False)
        else:
            raise ValueError(f"Can't do residual learning with {action_space}")
    else:
        if init_joint_conf:
            env = JointConfInitializationWrapper(env, heuristic=grasp)
            if is_level_4:
                env = CubeRotationAlignWrapper(env)
    env = FlatObservationWrapper(env)
    return env


def make_vec_training_env(nenvs, reward_fn, termination_fn, initializer,
                          action_space, init_joint_conf=False, residual=False, kp_coef=None, kd_coef=None,
                          frameskip=1):
    from stable_baselines.common.vec_env import SubprocVecEnv

    def _make_env(rank):
        def _init():
            return make_training_env(reward_fn, termination_fn, initializer,
                                     action_space, init_joint_conf, residual, kp_coef, kd_coef, frameskip,
                                     rank)
        return _init

    return SubprocVecEnv([_make_env(rank=i) for i in range(nenvs)])
