import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import Delaunay
from scipy.spatial.qhull import QhullError
from cvxopt import matrix
from cvxopt import solvers
import cvxpy as cp
import pybullet as p


def get_rotation_between_vecs(v1, v2):
    """Rotation from v1 to v2."""
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    axis = np.cross(v1, v2)
    if np.linalg.norm(axis) == 0:
        if np.dot(v1, v2) > 0:
            # zero rotation
            return np.array([0, 0, 0, 1])
        else:
            # 180 degree rotation

            # get perp vec
            axis = np.random.rand(3)
            axis /= np.linalg.norm(axis)
            axis -= np.dot(axis, v1)
            axis /= np.linalg.norm(axis)
            assert np.dot(axis, v1) == 0
            assert np.dot(axis, v2) == 0
            return np.array([axis[0], axis[1], axis[2], 0])
    axis /= np.linalg.norm(axis)
    angle = np.arccos(v1.dot(v2))
    quat = np.zeros(4)
    quat[:3] = axis * np.sin(angle / 2)
    quat[-1] = np.cos(angle / 2)
    return quat


def solve_qp(P, q, G, h, A, b, **kwargs):
    P, q, G, h, A, b = [matrix(x) for x in [P, q, G, h, A, b]]
    solvers.options['show_progress'] = False
    try:
        solution = solvers.qp(P, q, G, h, A, b, **kwargs)
        if solution['status'] != 'optimal':
            raise ValueError("No feasible solution.")
        return np.array(solution['x'])[:, 0]
    except ValueError as e:
        # print(f"Unable to solve for forces: {repr(e)}")
        return None


def solve_conic_program(G, wrench, mu):
    # the optimization variables are the forces on each tip
    f = cp.Variable(9)
    # dummy = cp.Variable(1)

    # the optimization objective is constant since it's actually a constraint satisfication problem
    objective = cp.Minimize(cp.sum_squares(f))

    # two constraints:
    # 1. forces meet the external force
    constraint1 = G @ f == wrench
    # 2. forces inside the friction cone
    constraint21 = cp.SOC(mu * f[2], f[0:2])
    constraint22 = cp.SOC(mu * f[5], f[3:5])
    constraint23 = cp.SOC(mu * f[8], f[6:8])
    constraints = [constraint1, constraint21, constraint22, constraint23]
    # constraints = [cp.norm(dummy) <= -1]

    # the optimization problem
    prob = cp.Problem(objective, constraints)

    # solve the problem
    prob.solve()
    print("The optimal value is", prob.value)
    print("A solution f is", f.value)
    if f.value is None:
        return None
    return np.array(f.value).reshape((3, 3))


class Transform(object):
    def __init__(self, pos=None, ori=None, T=None):
        if pos is not None and ori is not None:
            self.R = np.array(p.getMatrixFromQuaternion(ori)).reshape((3, 3))
            self.pos = pos
            self.T = np.eye(4)
            self.T[:3, :3] = self.R
            self.T[:3, -1] = self.pos
        elif T is not None:
            self.T = T
            self.R = T[:3, :3]
            self.pos = T[:3, -1]
        else:
            raise ValueError("You must specify T or both pos and ori.")

    def adjoint(self):
        def _skew(p):
            return np.array([
                [0, -p[2], p[1]],
                [p[2], 0, -p[0]],
                [-p[1], p[0], 0],
            ])

        adj = np.zeros((6, 6))
        adj[:3, :3] = self.R
        adj[3:, 3:] = self.R
        adj[3:, :3] = _skew(self.pos).dot(self.R)
        return adj

    def inverse(self):
        T = np.eye(4)
        T[:3, :3] = self.R.T
        T[:3, -1] = -self.R.T.dot(self.pos)
        return Transform(T=T)

    def __call__(self, x):
        if isinstance(x, Transform):
            return Transform(T=self.T.dot(x.T))
        else:
            # check for different input forms
            one_dim = len(x.shape) == 1
            homogeneous = x.shape[-1] == 4
            if one_dim:
                x = x[None]
            if not homogeneous:
                x_homo = np.ones((x.shape[0], 4))
                x_homo[:, :3] = x
                x = x_homo

            # transform points
            x = self.T.dot(x.T).T

            # create output to match input form
            if not homogeneous:
                x = x[:, :3]
            if one_dim:
                x = x[0]
            return x


class FrictionModel:
    def wrench_basis(self):
        pass

    def is_valid(self, wrench):
        pass

    def approximate_cone(self, contacts):
        pass

    def get_forces_from_approx(self, forces):
        pass


class NoFriction(FrictionModel):
    def __init__(self):
        self.basis = np.array([0, 0, 1, 0, 0, 0])[:, None]

    def wrench_basis(self):
        return self.basis

    def is_valid(self, wrench):
        return wrench >= 0

    def approximate_cone(self, contacts):
        return contacts, self

    def get_forces_from_approx(self, forces):
        return forces


class CoulombFriction(FrictionModel):
    def __init__(self, mu):
        self.mu = mu
        self.basis = np.eye(6)[:, :3]
        self.cone_corners, self.corner_transforms = self._get_cone_corners()

    def wrench_basis(self):
        return self.basis

    def is_valid(self, wrench):
        return np.linalg.norm(wrench[:2]) <= wrench[2]

    def _get_cone_corners(self):
        # approximate cone with an inscribed square
        contact_normal = np.array([0, 0, 1])
        fac = self.mu * np.sqrt(2) / 2
        corners = []
        transforms = []
        for i in [-1, +1]:
            for j in [-1, +1]:
                corner = np.array([i * fac, j * fac, 1])
                corner /= np.linalg.norm(corner)
                q = get_rotation_between_vecs(contact_normal, corner)
                corners.append(corner)
                transforms.append(Transform(pos=np.zeros(3), ori=q))
        return corners, transforms

    def approximate_cone(self, contacts):
        """
        Returns a set of contacts under a simpler friction model
        which approximate the friction cone of contacts.
        """
        new_contacts = []
        for c in contacts:
            new_contacts.extend([c] + [c(T) for T in self.corner_transforms])
        return new_contacts, NoFriction()

    def get_forces_from_approx(self, forces):
        n = len(forces) // 5
        contact_normal = np.array([0, 0, 1])[None]
        contact_forces = []
        assert np.all(forces >= 0)
        for i in range(n):
            force = contact_normal * forces[5 * i]
            for j, c in enumerate(self.corner_transforms):
                f = c(contact_normal * forces[5 * i + j + 1])
                force += f
            contact_forces.append(force[0])
        return contact_forces


class Contact(object):
    def __init__(self, c, T=None):
        """
        Creates a contact point in the reference frame defined by T
        (world frame if T is None).

        c is an element in the output of p.getContactPoints()
        """
        self.bodyA, self.bodyB, self.linkA, self.linkB = c[1:5]
        self.contact_posA = T(np.array(c[5]))
        self.contact_posB = T(np.array(c[6]))
        self.contact_normal = T(np.array(c[7]))
        q = get_rotation_between_vecs(np.array([0, 0, 1]), self.contact_normal)
        self.TA = Transform(pos=self.contact_posA, ori=q)
        self.TB = Transform(pos=self.contact_posB, ori=q)


class Cube(object):
    def __init__(self, halfwidth, friction_model):
        self.w = halfwidth
        self.friction = friction_model

    def _compute_grasp_matrix(self, contacts, friction=None):
        if friction is None:
            friction = self.friction
        return np.concatenate([c.adjoint().dot(friction.wrench_basis())
                               for c in contacts], axis=1)

    def contact_from_tip_position(self, pos):
        """
        Compute contact frame from tip positions in the cube
        center of mass frame.
        """
        axis = np.argmax(np.abs(pos))
        sign = np.sign(pos[axis])
        z_contact = np.zeros(3)
        z_contact[axis] = -sign
        q = get_rotation_between_vecs(np.array([0, 0, 1]), z_contact)
        T_contact_to_cube = Transform(pos=pos, ori=q)
        return T_contact_to_cube

    def solve_for_tip_forces(self, contacts, wrench, force_min=0.01):
        new_contacts, friction = self.friction.approximate_cone(contacts)
        G = self._compute_grasp_matrix(new_contacts, friction)
        n = len(new_contacts)
        f = solve_qp(np.eye(n), np.zeros(n), -np.eye(n),
                     -force_min * np.ones(n), G, wrench)
        if f is None:
            # G_old = self._compute_grasp_matrix(contacts)
            # f = solve_conic_program(G_old, wrench, self.friction.mu)
            return None
        else:
            f = self.friction.get_forces_from_approx(f)

        # transform wrenches to cube frame from contact frames
        # f = [c.adjoint().dot(self.friction.wrench_basis()).dot(f)
        #      for c, f in zip(contacts, f)]
        f = [self.friction.wrench_basis().dot(f) for f in f]
        # print(sum(f))
        return np.array(f)

    def force_closure_test(self, contacts):
        new_contacts, friction = self.friction.approximate_cone(contacts)
        G = self._compute_grasp_matrix(new_contacts, friction)
        try:
            hull = Delaunay(G.T)
        except QhullError:
            return False
        return hull.find_simplex(np.zeros((6))) >= 0


class PDController(object):
    def __init__(self, kp, kd):
        self.kp = kp
        self.kd = kd
        self.prev_error = None

    def __call__(self, error, velocity_error=None):
        if self.prev_error is None:
            self.prev_error = error

        control = self.kp * error + self.kd * (error - self.prev_error)
        self.prev_error = error
        return control


class TorquePDController(PDController):
    def __call__(self, error, velocity_error=None):
        if self.prev_error is None:
            self.prev_error = error

        rot = error.inv().as_rotvec()
        drot = (error.inv() * self.prev_error).as_rotvec()
        control = self.kp * rot + self.kd * drot
        self.prev_error = error
        return control


class CubePD(object):
    def __init__(self,
                 mass=0.02,
                 max_force=1.0,
                 max_torque=0.05,
                 force_gains=[100, 631],
                 torque_gains=[0.31623, 0.89125]):

        self.max_force = max_force
        self.max_torque = max_torque
        self.force_pd = PDController(*force_gains)
        self.torque_pd = TorquePDController(*torque_gains)
        self.mass = mass
        self.g = np.array([0, 0, -9.81])

    def _rotation_error(self, goal_quat, quat):
        goal_rot = Rotation.from_quat(goal_quat)
        actual_rot = Rotation.from_quat(quat)
        return goal_rot.inv() * actual_rot

    def _scale(self, x, lim):
        norm = np.linalg.norm(x)
        if norm > lim:
            return x * lim / norm
        return x

    def __call__(self, goal_pos, goal_quat, pos, quat):
        force = self.force_pd(goal_pos - pos)
        force = self._scale(force, self.max_force)
        force -= self.mass * self.g
        torque = self.torque_pd(self._rotation_error(goal_quat, quat))
        torque = self._scale(torque, self.max_torque)

        T = Transform(pos=pos, ori=quat)
        return T.R.T.dot(force), T.R.dot(torque)
