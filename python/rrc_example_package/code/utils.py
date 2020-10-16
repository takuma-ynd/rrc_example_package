import random
import numpy as np
import pybullet as p
import itertools
from rrc_simulation import visual_objects
from scipy.spatial.transform import Rotation as R


def set_seed(seed=0):
    import random
    import numpy as np
    import tensorflow as tf
    import torch
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_random_seed(seed)
    torch.manual_seed(0)


def apply_rotation_z(org_pos, theta):
    '''
    Apply 3 x 3 rotation matrix for rotation on xy-plane
    '''
    x_, y_, z_ = org_pos
    x = x_ * np.cos(theta) - y_ * np.sin(theta)
    y = x_ * np.sin(theta) + y_ * np.cos(theta)
    z = z_
    return x, y, z


def apply_transform(pos, ori, points):
    T = np.eye(4)
    T[:3, :3] = np.array(p.getMatrixFromQuaternion(ori)).reshape((3, 3))
    T[:3, -1] = pos
    if len(points.shape) == 1:
        points = points[None]
    homogeneous = points.shape[-1] == 4
    if not homogeneous:
        points_homo = np.ones((points.shape[0], 4))
        points_homo[:, :3] = points
        points = points_homo

    points = T.dot(points.T).T
    if not homogeneous:
        points = points[:, :3]
    return points


def sample_from_normal_cube(cube_halfwidth, face=None, shrink_region=1.0, avoid_top=False,
                            sample_from_all_faces=False):
    '''
    sample from hypothetical cube that has no rotation and is located at (0, 0, 0)
    NOTE: It does NOT sample point from the bottom face

    It samples points with the following procedure:
    1. choose one of the 5 faces (except the bottom)
    2a. if the top face is chosen, just sample from there
    2b. if a side face is chosen:
        1. sample points from the front face
        2. rotate the sampled points properly according to the selected face
    '''
    # 1. choose one of the faces:
    if avoid_top:
        faces = [0, 1, 2, 3]
    elif sample_from_all_faces:
        faces = [-2, -1, 0, 1, 2, 3]
    else:
        faces = [-1, 0, 1, 2, 3]

    if face is None:
        face = random.choice(faces)

    if face not in faces:
        raise KeyError(f'face {face} is not in the list of allowed faces: {faces}')

    if face == -1:
        # top
        x, y = np.random.uniform(low=-cube_halfwidth * shrink_region,
                                 high=cube_halfwidth * shrink_region, size=2)
        z = cube_halfwidth
    elif face == -2:
        # bottom (only allowed when sample_from_all_faces is enabled)
        x, y = np.random.uniform(low=-cube_halfwidth * shrink_region,
                                 high=cube_halfwidth * shrink_region, size=2)
        z = -cube_halfwidth
    else:
        # one of the side faces
        # sample on the front xz-face
        x_, z_ = np.random.uniform(low=-cube_halfwidth * shrink_region,
                                   high=cube_halfwidth * shrink_region, size=2)
        y_ = -cube_halfwidth

        # apply rotation to the points according to its face direction
        rot_theta = face * np.pi / 2
        x, y, z = apply_rotation_z((x_, y_, z_), rot_theta)

    return x, y, z


def sample_heuristic_points(cube_halfwidth=0.0325, shrink_region=1.0):
    '''
    Sample three points on the normal cube heurisitcally.

    One point is sampled on a side face, and the other two points are sampled
    from the face stading on the other side.

    The two points are sampled in a way that they are point symmetric w.r.t.
    the center of the face.
    '''

    min_dist = cube_halfwidth * 0.1

    # center of the front face
    x_, z_ = 0, 0
    y_ = -cube_halfwidth
    center_point = (x_, y_, z_)

    # two points that are point symmetric w.r.t. the center of the face
    x_, z_ = 0, 0
    while np.sqrt(x_ ** 2 + z_ ** 2) < min_dist:  # rejection sampling
        x_, z_ = np.random.uniform(low=-cube_halfwidth * shrink_region,
                                   high=cube_halfwidth * shrink_region, size=2)
    y_ = -cube_halfwidth

    x__, z__ = -x_, -z_  # point symetric w.r.t. the center point
    y__ = y_

    support_point1 = (x_, y_, z_)
    support_point2 = (x__, y__, z__)

    # sample two faces that are in parallel
    faces = [0, 1, 2, 3]
    face = random.choice(faces)
    parallel_face = face + 2 % 4

    # apply rotation to the points according to its face direction
    sample_points = []
    rot_theta = face * np.pi / 2
    sample_points.append(np.asarray(apply_rotation_z(center_point, rot_theta),
                                    dtype=np.float))

    for point in [support_point1, support_point2]:
        rot_theta = parallel_face * np.pi / 2
        sample_points.append(np.asarray(apply_rotation_z(point, rot_theta),
                             dtype=np.float))

    return sample_points

def sample_center_of_three(cube_halfwidth=0.0325, shrink_region=1.0):

    # center of the front face
    x_, z_ = 0, 0
    y_ = -cube_halfwidth
    center_point = (x_, y_, z_)

    faces = [0, 1, 2, 3]

    sample_points = []
    start = random.choice(faces)
    for i in range(3):
        rot_theta = ((i + start) % 4 )* np.pi / 2
        sample_points.append(np.asarray(apply_rotation_z(center_point, rot_theta),
                             dtype=np.float))

    return sample_points

def sample_center_of_two(cube_halfwidth=0.0325, shrink_region=1.0):
    # center of the front face
    x_, z_ = 0, 0
    y_ = -cube_halfwidth
    center_point = (x_, y_, z_)

    faces = [0, 1, 2, 3]

    sample_points = []
    start = random.choice(faces)
    for i in range(2):
        rot_theta = ((2 * i + start) % 4 )* np.pi / 2
        sample_points.append(np.asarray(R.from_euler('z', rot_theta).apply(center_point),
                             dtype=np.float))

    #hacky position definition
    sample_points.append(np.asarray([np.inf, np.inf, np.inf]))

    return np.asarray(sample_points)

def sample_cube_surface_points(cube_halfwidth=0.0325,
                               shrink_region=0.8,
                               num_samples=3,
                               heuristic='pinch'):
    '''
    sample points on the surfaces of the cube except the one at the bottom.
    NOTE: This function only works when the bottom face is fully touching on
          the table.

    Args:
        cube_pos: Position (x, y, z)
        cube_orientation: Orientation as quaternion (x, y, z, w)
        cube_halfwidth: halfwidth of the cube (float)
        shrink_region: shrink the sample region on each plane by the specified
                       coefficient (float)
        num_samples: number of points to sample (int)
    Returns:
        List of sampled positions
    '''
    # Backward compatibility
    if heuristic == 'pinch':
        assert num_samples == 3, 'heuristic sampling only supports 3 samples'
        norm_cube_samples = sample_heuristic_points(cube_halfwidth=cube_halfwidth,
                                                    shrink_region=shrink_region)
    elif heuristic == 'center_of_three':
        assert num_samples == 3
        norm_cube_samples = sample_center_of_three(cube_halfwidth=cube_halfwidth)
    elif heuristic == 'center_of_two':
        assert num_samples == 3 #don't use this flag
        norm_cube_samples = sample_center_of_two(cube_halfwidth=cube_halfwidth)
    elif heuristic is None:
        norm_cube_samples = [sample_from_normal_cube(cube_halfwidth,
                                                     shrink_region=shrink_region)
                             for _ in range(num_samples)]
    else:
        raise KeyError('Unrecognized heuristic value: {}. Use one of ["pinch", "center_of_three", None]'.format(heuristic))

    # apply transformation
    return np.array(norm_cube_samples)
    # sample_points = apply_transform(cube_pos, cube_orientation,
    #                                 np.array(norm_cube_samples))
    #
    # return sample_points


class VisualMarkers:
    '''Visualize spheres on the specified points'''
    def __init__(self):
        self.markers = []

    def add(self, points, radius=0.015, color=None):
        if isinstance(points[0], (int, float)):
            points = [points]
        if color is None:
            color = (0, 1, 1, 0.5)
        for point in points:
            self.markers.append(
                        visual_objects.SphereMaker(radius, point, color=color))

    def remove(self):
        self.markers = []


class VisualCubeOrientation:
    '''visualize cube orientation by three cylinder'''
    def __init__(self, cube_position, cube_orientation, cube_halfwidth=0.0325):
        self.markers = []
        self.cube_halfwidth = cube_halfwidth

        color_cycle = [[1, 0, 0, 0.6], [0, 1, 0, 0.6], [0, 0, 1, 0.6]]

        self.z_axis = np.asarray([0,0,1])

        const = 1 / np.sqrt(2)
        x_rot = R.from_quat([const, 0, const, 0])
        y_rot = R.from_quat([0, const, const, 0])
        z_rot = R.from_quat([0,0,0,1])

        assert( np.linalg.norm( x_rot.apply(self.z_axis) - np.asarray([1., 0., 0.]) ) < 0.00000001)
        assert( np.linalg.norm( y_rot.apply(self.z_axis) - np.asarray([0., 1., 0.]) ) < 0.00000001)
        assert( np.linalg.norm( z_rot.apply(self.z_axis) - np.asarray([0., 0., 1.]) ) < 0.00000001)

        self.rotations = [x_rot, y_rot, z_rot]
        cube_rot = R.from_quat(cube_orientation)

        #x: red , y: green, z: blue
        for rot, color in zip(self.rotations, color_cycle):
            rotation = cube_rot * rot
            orientation = rotation.as_quat()
            bias = rotation.apply(self.z_axis) * cube_halfwidth
            self.markers.append(
                CylinderMarker(radius=cube_halfwidth/20,
                               length=cube_halfwidth*2,
                               position=cube_position + bias,
                               orientation=orientation,
                               color=color)
            )

    def set_state(self, position, orientation):
        cube_rot = R.from_quat(orientation)
        for rot, marker in zip(self.rotations, self.markers):
            rotation = cube_rot * rot
            orientation = rotation.as_quat()
            bias = rotation.apply(self.z_axis) * self.cube_halfwidth
            marker.set_state(position=position + bias,
                             orientation=orientation)


class CylinderMarker:
    """Visualize a cylinder."""

    def __init__(
        self, radius, length, position, orientation, color=(0, 1, 0, 0.5)):
        """
        Create a cylinder marker for visualization

        Args:
            radius (float): radius of cylinder.
            length (float): length of cylinder.
            position: Position (x, y, z)
            orientation: Orientation as quaternion (x, y, z, w)
            color: Color of the cube as a tuple (r, b, g, q)
        """

        self.shape_id = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=radius,
            length=length,
            rgbaColor=color
        )
        self.body_id = p.createMultiBody(
            baseVisualShapeIndex=self.shape_id,
            basePosition=position,
            baseOrientation=orientation
        )

    def set_state(self, position, orientation):
        """Set pose of the marker.

        Args:
            position: Position (x, y, z)
            orientation: Orientation as quaternion (x, y, z, w)
        """
        p.resetBasePositionAndOrientation(
            self.body_id,
            position,
            orientation
        )



def is_valid_action(action, action_type='position'):
    from rrc_simulation.trifinger_platform import TriFingerPlatform
    spaces = TriFingerPlatform.spaces

    if action_type == 'position':
        action_space = spaces.robot_position
    elif action_type == 'torque':
        action_space = spaces.robot_position

    return (action_space.low <= action).all() and (action <= action_space.high).all()


import copy
from rrc_simulation.gym_wrapper.envs.cube_env import ActionType
class action_type_to:
    '''
    A Context Manager that sets action type and action space temporally
    This applies to all wrappers and the origianl environment recursively ;)
    '''
    def __init__(self, action_type, env):
        self.action_type = action_type
        self.action_space = self._get_action_space(action_type)
        self.org_action_type = env.action_type
        self.org_action_space = env.action_space
        self.env = env

    def __enter__(self):
        current_env = self.env
        self.set_action_type_and_space(current_env)
        while hasattr(current_env, 'env'):
            current_env = current_env.env
            self.set_action_type_and_space(current_env)

    def __exit__(self, type, value, traceback):
        current_env = self.env
        self.revert_action_type_and_space(current_env)
        while hasattr(current_env, 'env'):
            current_env = current_env.env
            self.revert_action_type_and_space(current_env)

    def set_action_type_and_space(self, env):
        env.action_space = self.action_space
        env.action_type = self.action_type

    def revert_action_type_and_space(self, env):
        env.action_space = self.org_action_space
        env.action_type = self.org_action_type

    def _get_action_space(self, action_type):
        import gym
        from rrc_simulation import TriFingerPlatform
        spaces = TriFingerPlatform.spaces
        if action_type == ActionType.TORQUE:
            action_space = spaces.robot_torque.gym
        elif action_type == ActionType.POSITION:
            action_space = spaces.robot_position.gym
        elif action_type == ActionType.TORQUE_AND_POSITION:
            action_space = gym.spaces.Dict(
                {
                    "torque": spaces.robot_torque.gym,
                    "position": spaces.robot_position.gym,
                }
            )
        else:
            ValueError('unknown action type')
        return action_space


def repeat(sequence, num_repeat=3):
    '''
    [1,2,3] with num_repeat = 3  --> [1,1,1,2,2,2,3,3,3]
    '''
    return list(e for e in sequence for _ in range(num_repeat))


def ease_out(sequence, in_rep=1, out_rep=5):
    '''
    create "ease out" motion where an action is repeated for *out_rep* times at the end.
    '''
    in_seq_length = len(sequence[:-len(sequence) // 3])
    out_seq_length = len(sequence[-len(sequence) // 3:])
    x = [0, out_seq_length - 1]
    rep = [in_rep, out_rep]
    out_repeats = np.interp(list(range(out_seq_length)), x, rep).astype(int).tolist()

    #in_repeats = np.ones(in_seq_length).astype(int).tolist()
    in_repeats = np.ones(in_seq_length) * in_rep
    in_repeats = in_repeats.astype(int).tolist()
    repeats = in_repeats + out_repeats
    assert len(repeats) == len(sequence)

    seq = [repeat([e], n_rep) for e, n_rep in zip(sequence, repeats)]
    seq = [y for x in seq for y in x]  # flatten it

    return seq

class frameskip_to:
    '''
    A Context Manager that sets action type and action space temporally
    This applies to all wrappers and the origianl environment recursively ;)
    '''
    def __init__(self, frameskip, env):
        self.frameskip = frameskip
        self.env = env
        self.org_frameskip = env.unwrapped.frameskip

    def __enter__(self):
        self.env.unwrapped.frameskip = self.frameskip

    def __exit__(self, type, value, traceback):
        self.env.unwrapped.frameskip = self.org_frameskip


class keep_state:
    '''
    A Context Manager that preserves the state of the simulator
    '''
    def __init__(self, env):
        self.finger_id = env.platform.simfinger.finger_id
        self.joints = env.platform.simfinger.pybullet_link_indices
        self.cube_id = env.platform.cube.block

    def __enter__(self):
        self.state_id = p.saveState()

    def __exit__(self, type, value, traceback):
        p.restoreState(stateId=self.state_id)


class IKUtils:
    def __init__(self, env):
        from .const import INIT_JOINT_CONF
        self.fk = env.platform.simfinger.pinocchio_utils.forward_kinematics
        self.ik = env.platform.simfinger.pinocchio_utils.inverse_kinematics
        self.finger_id = env.platform.simfinger.finger_id
        self.tip_ids = env.platform.simfinger.pybullet_tip_link_indices
        self.link_ids = env.platform.simfinger.pybullet_link_indices
        self.cube_id = env.platform.cube.block
        self.env = env
        self.tips_init = self.fk(INIT_JOINT_CONF)

    def sample_no_collision_ik(self, target_tip_positions, sort_tips=False, slacky_collision=False):
        from pybullet_planning.interfaces.kinematics.ik_utils import sample_multiple_ik_with_collision

        with keep_state(self.env):
            if sort_tips:
                target_tip_positions, _ = self._assign_positions_to_fingers(target_tip_positions)
            collision_fn = self._get_collision_fn(slacky_collision)
            sample_fn = self._get_sample_fn()
            solutions = sample_multiple_ik_with_collision(self.ik, collision_fn, sample_fn,
                                                          target_tip_positions, num_samples=3)
            return solutions

    def _get_collision_fn(self, slacky_collision):
        from pybullet_planning.interfaces.robots.collision import get_collision_fn
        return get_collision_fn(**self._get_collision_conf(slacky_collision))

    def _get_collision_conf(self, slacky_collision):
        from .const import COLLISION_TOLERANCE
        if slacky_collision:
            disabled_collisions = [((self.finger_id, tip_id), (self.cube_id, -1))
                                   for tip_id in self.tip_ids]
            config = {
                'body': self.finger_id,
                'joints': self.link_ids,
                'obstacles': [self.cube_id],
                'self_collisions': True,
                'extra_disabled_collisions': disabled_collisions,
                'max_distance': -COLLISION_TOLERANCE
            }
        else:
            config = {
                'body': self.finger_id,
                'joints': self.link_ids,
                'obstacles': [self.cube_id],
                'self_collisions': False
            }

        return config

    def _get_sample_fn(self):
        space = self.env.platform.spaces.robot_position.gym
        def _sample_fn():
            s = np.random.rand(space.shape[0])
            return s * (space.high - space.low) + space.low
        return _sample_fn

    def _assign_positions_to_fingers(self, tips):
        min_cost = 1000000
        opt_tips = []
        opt_inds = [0, 1, 2]
        for v in itertools.permutations([0, 1, 2]):
            sorted_tips = tips[v, :]
            cost = np.linalg.norm(sorted_tips - self.tips_init)
            if min_cost > cost:
                min_cost = cost
                opt_tips = sorted_tips
                opt_inds = v

        return opt_tips, opt_inds

    def get_joint_conf(self):
        obs = self.env.platform.simfinger._get_latest_observation()
        return obs.position, obs.velocity


def get_body_state(body_id):
    position, orientation = p.getBasePositionAndOrientation(
        body_id
    )
    velocity = p.getBaseVelocity(body_id)
    return list(position), list(orientation), list(velocity)


def set_body_state(body_id, position, orientation, velocity):
    p.resetBasePositionAndOrientation(
        body_id,
        position,
        orientation,
    )
    linear_vel, angular_vel = velocity
    p.resetBaseVelocity(body_id, linear_vel, angular_vel)


class AssertNoStateChanges:
    def __init__(self, env):
        self.cube_id = env.platform.cube.block
        self.finger_id = env.platform.simfinger.finger_id
        self.finger_links = env.platform.simfinger.pybullet_link_indices

    def __enter__(self):
        from .utils import get_body_state, set_body_state
        from pybullet_planning.interfaces.robots.joint import get_joint_velocities, get_joint_positions
        org_obj_pos, org_obj_ori, org_obj_vel = get_body_state(self.cube_id)
        self.org_obj_pos = org_obj_pos
        self.org_obj_ori = org_obj_ori
        self.org_obj_vel = org_obj_vel

        self.org_joint_pos = get_joint_positions(self.finger_id, self.finger_links)
        self.org_joint_vel = get_joint_velocities(self.finger_id, self.finger_links)

    def __exit__(self, type, value, traceback):
        from pybullet_planning.interfaces.robots.joint import get_joint_velocities, get_joint_positions
        obj_pos, obj_ori, obj_vel = get_body_state(self.cube_id)
        np.testing.assert_array_almost_equal(self.org_obj_pos, obj_pos)
        np.testing.assert_array_almost_equal(self.org_obj_ori, obj_ori)
        np.testing.assert_array_almost_equal(self.org_obj_vel[0], obj_vel[0])
        np.testing.assert_array_almost_equal(self.org_obj_vel[1], obj_vel[1])

        joint_pos = get_joint_positions(self.finger_id, self.finger_links)
        joint_vel = get_joint_velocities(self.finger_id, self.finger_links)
        np.testing.assert_array_almost_equal(self.org_joint_pos, joint_pos)
        np.testing.assert_array_almost_equal(self.org_joint_vel, joint_vel)
