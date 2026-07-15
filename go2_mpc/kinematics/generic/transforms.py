"""
Homogeneous-transform and rotation utilities shared by the generic
URDF-driven kinematics engine.

URDF convention: <origin rpy="r p y"/> is a fixed-axis (extrinsic)
roll-pitch-yaw rotation, composed as R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
"""

import numpy as np


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """Convert URDF fixed-axis roll-pitch-yaw to a (3, 3) rotation matrix."""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)

    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return Rz @ Ry @ Rx


def make_transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    """Build a (4, 4) homogeneous transform from a translation + RPY rotation."""
    T = np.eye(4)
    T[:3, :3] = rpy_to_matrix(rpy)
    T[:3, 3] = xyz
    return T


def axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' rotation formula. `axis` must be a unit vector."""
    K = skew(axis)
    c, s = np.cos(angle), np.sin(angle)
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix such that skew(a) @ b == cross(a, b)."""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])
