from dataclasses import dataclass
import numpy as np


@dataclass
class BaseState:
    position: np.ndarray
    orientation: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray

    def __post_init__(self):
        # Eagerly compute and cache derived quantities (accessed multiple times per step)
        w, x, y, z = self.orientation

        # Euler angles
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        self._roll = float(np.arctan2(t0, t1))

        t2 = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
        self._pitch = float(np.arcsin(t2))

        self._yaw = float(np.arctan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        ))

        # Rotation matrix
        R = np.empty((3, 3))
        R[0, 0] = 1 - 2*(y*y + z*z)
        R[0, 1] = 2*(x*y - z*w)
        R[0, 2] = 2*(x*z + y*w)
        R[1, 0] = 2*(x*y + z*w)
        R[1, 1] = 1 - 2*(x*x + z*z)
        R[1, 2] = 2*(y*z - x*w)
        R[2, 0] = 2*(x*z - y*w)
        R[2, 1] = 2*(y*z + x*w)
        R[2, 2] = 1 - 2*(x*x + y*y)
        self._rotation_matrix = R

    @property
    def rotation_matrix(self):
        return self._rotation_matrix

    @property
    def roll(self):
        return self._roll

    @property
    def pitch(self):
        return self._pitch

    @property
    def yaw(self):
        return self._yaw

    def to_mpc_vector(self):
        """Convert to 12-dim MPC state: [pos, rpy, lin_vel, ang_vel]."""
        x = np.empty(12)
        x[0:3] = self.position
        x[3] = self._roll
        x[4] = self._pitch
        x[5] = self._yaw
        x[6:9] = self.linear_velocity
        x[9:12] = self.angular_velocity
        return x


@dataclass
class JointState:
    positions: np.ndarray
    velocities: np.ndarray


@dataclass(frozen=True)
class State:
    base: BaseState
    joints: JointState

    @classmethod
    def from_vector(cls, vec: np.ndarray, num_joints: int):
        """
        Construct State from flat vector.
        Assumes layout:
        [pos(3), quat(4), lin_vel(3), ang_vel(3),
         joint_pos(N), joint_vel(N)]
        """
        expected = 3 + 4 + 3 + 3 + 2 * num_joints
        assert vec.shape[0] == expected, \
            f"Expected state dim {expected}, got {vec.shape[0]}"

        idx = 0

        position = vec[idx:idx+3]
        idx += 3
        orientation = vec[idx:idx+4]
        norm = np.linalg.norm(orientation)
        if norm > 0:
            orientation /= norm
        idx += 4
        linear_velocity = vec[idx:idx+3]
        idx += 3
        angular_velocity = vec[idx:idx+3]
        idx += 3

        joint_positions = vec[idx:idx+num_joints]
        idx += num_joints
        joint_velocities = vec[idx:idx+num_joints]

        base = BaseState(
            position=position,
            orientation=orientation,
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
        )

        joints = JointState(
            positions=joint_positions,
            velocities=joint_velocities,
        )

        return cls(base=base, joints=joints)

    def to_vector(self):
        return np.concatenate([
            self.base.position,
            self.base.orientation,
            self.base.linear_velocity,
            self.base.angular_velocity,
            self.joints.positions,
            self.joints.velocities,
        ])
