import numpy as np
from go2_mpc.kinematics.generic.generic_kinematics import GenericKinematics

# Planar 2-joint arm in the XY plane, both joints rotate about world Z:
# base -(0.5, 0, 0)-> j1 (revolute, axis Z) -(0.3, 0, 0)-> foot
_TWO_JOINT_URDF = """<?xml version="1.0"?>
<robot name="r">
  <link name="base"/>
  <link name="mid"/>
  <link name="foot"/>
  <joint name="j1" type="revolute">
    <origin xyz="0.5 0 0" rpy="0 0 0"/>
    <parent link="base"/><child link="mid"/>
    <axis xyz="0 0 1"/>
  </joint>
  <joint name="j2" type="revolute">
    <origin xyz="0.3 0 0" rpy="0 0 0"/>
    <parent link="mid"/><child link="foot"/>
    <axis xyz="0 0 1"/>
  </joint>
</robot>
"""


def _write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content)
    return p


def _make_kin(tmp_path):
    path = _write(tmp_path, "arm.urdf", _TWO_JOINT_URDF)
    return GenericKinematics(urdf_path=path, base_link="base", foot_links=["foot"])


def test_forward_kinematics_zero_config(tmp_path):
    kin = _make_kin(tmp_path)
    p_foot = kin.foot_position_world(0, p_base=np.zeros(3), R_base=np.eye(3), q_leg=np.zeros(3)[:2])
    np.testing.assert_allclose(p_foot, [0.8, 0.0, 0.0], atol=1e-9)


def test_forward_kinematics_quarter_turn_first_joint(tmp_path):
    kin = _make_kin(tmp_path)
    # j1 rotates 90deg: the (0.3, 0, 0) second link now points along +Y
    # from j1's position (0.5, 0, 0).
    q_leg = np.array([np.pi / 2, 0.0])
    p_foot = kin.foot_position_world(0, p_base=np.zeros(3), R_base=np.eye(3), q_leg=q_leg)
    np.testing.assert_allclose(p_foot, [0.5, 0.3, 0.0], atol=1e-9)


def test_leg_jacobian_matches_finite_difference(tmp_path):
    kin = _make_kin(tmp_path)
    q0 = np.array([0.3, -0.4])
    p_base, R_base = np.zeros(3), np.eye(3)

    J = kin.leg_jacobian(0, p_base, R_base, q0)

    eps = 1e-6
    J_fd = np.zeros((3, 2))
    for k in range(2):
        dq = np.zeros(2)
        dq[k] = eps
        p_plus = kin.foot_position_world(0, p_base, R_base, q0 + dq)
        p_minus = kin.foot_position_world(0, p_base, R_base, q0 - dq)
        J_fd[:, k] = (p_plus - p_minus) / (2 * eps)

    np.testing.assert_allclose(J[:, :2], J_fd, atol=1e-6)


def test_foot_position_translates_with_base(tmp_path):
    kin = _make_kin(tmp_path)
    p_base = np.array([1.0, 2.0, 3.0])
    p_foot = kin.foot_position_world(0, p_base=p_base, R_base=np.eye(3), q_leg=np.zeros(2))
    np.testing.assert_allclose(p_foot, p_base + [0.8, 0.0, 0.0], atol=1e-9)


def test_foot_velocity_matches_finite_difference_time_derivative(tmp_path):
    kin = _make_kin(tmp_path)
    q0 = np.array([0.2, 0.5])
    qd = np.array([1.0, -0.5])
    p_base, R_base = np.zeros(3), np.eye(3)
    v_base, omega_body = np.zeros(3), np.zeros(3)

    v_analytic = kin.foot_velocity(0, p_base, R_base, q0, v_base, omega_body, qd)

    dt = 1e-6
    p_plus = kin.foot_position_world(0, p_base, R_base, q0 + qd * dt)
    p_minus = kin.foot_position_world(0, p_base, R_base, q0 - qd * dt)
    v_fd = (p_plus - p_minus) / (2 * dt)

    np.testing.assert_allclose(v_analytic, v_fd, atol=1e-4)
