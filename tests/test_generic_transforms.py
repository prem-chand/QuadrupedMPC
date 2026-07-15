import numpy as np
from go2_mpc.kinematics.generic.transforms import (
    rpy_to_matrix, make_transform, axis_angle_to_matrix, skew,
)


def test_rpy_to_matrix_zero_is_identity():
    R = rpy_to_matrix(np.zeros(3))
    np.testing.assert_allclose(R, np.eye(3), atol=1e-12)


def test_rpy_to_matrix_roll_maps_y_to_z():
    # Roll +90deg about X: Y axis -> Z axis
    R = rpy_to_matrix(np.array([np.pi / 2, 0.0, 0.0]))
    np.testing.assert_allclose(R @ np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), atol=1e-9)


def test_rpy_to_matrix_yaw_maps_x_to_y():
    # Yaw +90deg about Z: X axis -> Y axis
    R = rpy_to_matrix(np.array([0.0, 0.0, np.pi / 2]))
    np.testing.assert_allclose(R @ np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), atol=1e-9)


def test_make_transform_translation_only():
    T = make_transform(np.array([1.0, 2.0, 3.0]), np.zeros(3))
    np.testing.assert_allclose(T[:3, 3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-12)


def test_axis_angle_to_matrix_quarter_turn_about_z():
    R = axis_angle_to_matrix(np.array([0.0, 0.0, 1.0]), np.pi / 2)
    np.testing.assert_allclose(R @ np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), atol=1e-9)


def test_axis_angle_to_matrix_zero_angle_is_identity():
    R = axis_angle_to_matrix(np.array([0.0, 1.0, 0.0]), 0.0)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-12)


def test_skew_matches_cross_product():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, -1.0, 2.0])
    np.testing.assert_allclose(skew(a) @ b, np.cross(a, b), atol=1e-12)
