"""
State representations for the quadruped robot.

Coordinate frame conventions
----------------------------
- **World frame (W)**: Fixed inertial frame, z-up. Base position and
  linear velocity are expressed in this frame.
- **Body frame (B)**: Attached to the robot's trunk CoM, rotates with
  the robot. Angular velocity is expressed in this frame (MuJoCo
  convention: ``qvel[3:6]`` is body-frame angular velocity).
- **Yaw frame (Y)**: World frame rotated about z by the robot's yaw
  angle only (no roll/pitch). The MPC uses this frame for foot
  positions so that body tilt does not contaminate horizontal
  foot placement.  R_YW = R_z(-yaw).

Quaternion convention: scalar-first ``(w, x, y, z)``, consistent with
MuJoCo.

Unit conventions: SI throughout — metres, radians, seconds, kilograms.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class BaseState:
    """Floating-base state of the robot trunk.

    Attributes
    ----------
    position : np.ndarray, shape (3,)
        CoM position ``[x, y, z]`` in **world frame** (m).
    orientation : np.ndarray, shape (4,)
        Unit quaternion ``[w, x, y, z]`` representing R_WB
        (rotation from body → world).
    linear_velocity : np.ndarray, shape (3,)
        CoM translational velocity ``[vx, vy, vz]`` in **world frame**
        (m/s).
    angular_velocity : np.ndarray, shape (3,)
        Body angular velocity ``[wx, wy, wz]`` in **body frame**
        (rad/s). This follows the MuJoCo convention where ``qvel[3:6]``
        lives in the body's local frame.

    Cached derived quantities (computed once in ``__post_init__``):
        ``roll``, ``pitch``, ``yaw`` — intrinsic ZYX Euler angles (rad).
        ``rotation_matrix`` — R_WB, shape (3, 3).
    """

    position: np.ndarray         # (3,)  world frame, m
    orientation: np.ndarray      # (4,)  quaternion [w, x, y, z]
    linear_velocity: np.ndarray  # (3,)  world frame, m/s
    angular_velocity: np.ndarray # (3,)  body frame, rad/s

    def __post_init__(self):
        # --- Quaternion → intrinsic ZYX Euler angles ---
        # Convention: q = w + xi + yj + zk  (scalar-first)
        w, x, y, z = self.orientation

        # Roll (rotation about body x-axis)
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        self._roll = float(np.arctan2(t0, t1))

        # Pitch (rotation about body y-axis), clamped to avoid
        # numerical issues at gimbal lock (±pi/2)
        t2 = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
        self._pitch = float(np.arcsin(t2))

        # Yaw (rotation about world z-axis)
        self._yaw = float(np.arctan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        ))

        # --- Quaternion → rotation matrix R_WB ---
        # Maps body-frame vectors into the world frame: v_W = R_WB @ v_B
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
    def rotation_matrix(self) -> np.ndarray:
        """R_WB: (3, 3) rotation matrix, world ← body."""
        return self._rotation_matrix

    @property
    def roll(self) -> float:
        """Roll angle about the x-axis (rad), range [-pi, pi]."""
        return self._roll

    @property
    def pitch(self) -> float:
        """Pitch angle about the y-axis (rad), range [-pi/2, pi/2]."""
        return self._pitch

    @property
    def yaw(self) -> float:
        """Yaw angle about the z-axis (rad), range [-pi, pi]."""
        return self._yaw

    def to_mpc_vector(self) -> np.ndarray:
        """Pack into the 12-dim centroidal MPC state vector.

        Layout::

            [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
             ──────── ──────────────── ──────────── ────────────
             pos (W)  Euler angles     lin vel (W)  ang vel (W)
                      (rad)            (m/s)        (rad/s)

        Angular velocity is rotated from body frame (MuJoCo convention)
        to **world frame** using R_WB, matching the Di Carlo et al.
        centroidal MPC formulation where ``θ̇ ≈ ω_world`` for small
        roll/pitch.

        Returns
        -------
        np.ndarray, shape (12,)
        """
        x = np.empty(12)
        x[0:3] = self.position
        x[3] = self._roll
        x[4] = self._pitch
        x[5] = self._yaw
        x[6:9] = self.linear_velocity
        # Rotate ω from body frame → world frame: ω_W = R_WB @ ω_B
        x[9:12] = self._rotation_matrix @ self.angular_velocity
        return x


@dataclass
class JointState:
    """Joint-space state for all actuated degrees of freedom.

    Leg ordering: ``[FL, FR, RL, RR]``, each with 3 DOFs
    ``[hip_abduction, hip_flexion, knee_flexion]``, giving 12 total.

    Attributes
    ----------
    positions : np.ndarray, shape (n_joints,)
        Joint angles (rad). Zero = nominal standing configuration
        as defined by the URDF/XML.
    velocities : np.ndarray, shape (n_joints,)
        Joint angular velocities (rad/s).
    """

    positions: np.ndarray   # (n_joints,) rad
    velocities: np.ndarray  # (n_joints,) rad/s


@dataclass(frozen=True)
class State:
    """Complete robot state: floating base + joints.

    Frozen dataclass — immutable after creation to prevent accidental
    mutation during a control step.

    Attributes
    ----------
    base : BaseState
        Floating-base state (position/lin_vel in world frame,
        angular velocity in body frame).
    joints : JointState
        Joint-space state (joint angles and velocities).
    """

    base: BaseState
    joints: JointState

    @classmethod
    def from_vector(cls, vec: np.ndarray, num_joints: int) -> "State":
        """Construct State from a flat vector.

        Expected layout::

            [pos_W(3), quat_wxyz(4), lin_vel_W(3), ang_vel_B(3),
             joint_pos(N), joint_vel(N)]

        Total length: ``13 + 2 * num_joints``.

        Parameters
        ----------
        vec : np.ndarray, shape (13 + 2*num_joints,)
            Flat state vector.
        num_joints : int
            Number of actuated joints (typically 12 for a quadruped).

        Returns
        -------
        State
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

    def to_vector(self) -> np.ndarray:
        """Flatten to a single array (inverse of ``from_vector``).

        Returns
        -------
        np.ndarray, shape (13 + 2*n_joints,)
            Layout matches ``from_vector``.
        """
        return np.concatenate([
            self.base.position,
            self.base.orientation,
            self.base.linear_velocity,
            self.base.angular_velocity,
            self.joints.positions,
            self.joints.velocities,
        ])
