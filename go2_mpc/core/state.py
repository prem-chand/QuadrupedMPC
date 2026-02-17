from dataclasses import dataclass
import numpy as np


# def _quat_to_euler(quat):
#     """Converts [w,x,y,z] to [roll, pitch, yaw] (reuses internal buffer)"""
#     w, x, y, z = quat
#     # Standard conversion
#     t0 = +2.0 * (w * x + y * z)
#     t1 = +1.0 - 2.0 * (x * x + y * y)
#     rpy = np.zeros(3)
#     rpy[0] = np.arctan2(t0, t1)

#     t2 = +2.0 * (w * y - z * x)
#     t2 = np.clip(t2, -1.0, 1.0)
#     rpy[1] = np.arcsin(t2)

#     t3 = +2.0 * (w * z + x * y)
#     t4 = +1.0 - 2.0 * (y * y + z * z)
#     rpy[2] = np.arctan2(t3, t4)
#     return rpy


@dataclass
class BaseState:
    position: np.ndarray
    orientation: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray

    # @property
    # def euler(self):
    #     return _quat_to_euler(self.orientation)

    @property
    def rotation_matrix(self):
        w, x, y, z = self.orientation
        R = np.zeros((3, 3))

        R[0, 0] = 1 - 2*(y*y + z*z)
        R[0, 1] = 2*(x*y - z*w)
        R[0, 2] = 2*(x*z + y*w)

        R[1, 0] = 2*(x*y + z*w)
        R[1, 1] = 1 - 2*(x*x + z*z)
        R[1, 2] = 2*(y*z - x*w)

        R[2, 0] = 2*(x*z - y*w)
        R[2, 1] = 2*(y*z + x*w)
        R[2, 2] = 1 - 2*(x*x + y*y)
        return R

    @property
    def roll(self):
        w, x, y, z = self.orientation
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        return np.arctan2(t0, t1)

    @property
    def pitch(self):
        w, x, y, z = self.orientation
        t2 = +2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        return np.arcsin(t2)

    @property
    def yaw(self):
        w, x, y, z = self.orientation
        return np.arctan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z)
        )


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
