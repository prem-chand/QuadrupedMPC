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


# 3-joint leg with hip(axis X)/thigh(axis Y)/calf(axis Y) joints plus a
# trailing fixed foot joint -- mirrors real quadruped leg topology (Go2)
# closely enough to exercise full_jacobian's fixed 3-DOF-per-leg column
# layout and forward_kinematics' foot_offset_transform path, neither of
# which the 2-joint planar fixture above can reach.
_THREE_JOINT_WITH_FOOT_OFFSET_URDF = """<?xml version="1.0"?>
<robot name="leg">
  <link name="base"/>
  <link name="hip"/>
  <link name="thigh"/>
  <link name="calf"/>
  <link name="foot"/>
  <joint name="hip_joint" type="revolute">
    <origin xyz="0.2 0.1 0" rpy="0 0 0"/>
    <parent link="base"/><child link="hip"/>
    <axis xyz="1 0 0"/>
  </joint>
  <joint name="thigh_joint" type="revolute">
    <origin xyz="0 0.05 0" rpy="0 0 0"/>
    <parent link="hip"/><child link="thigh"/>
    <axis xyz="0 1 0"/>
  </joint>
  <joint name="calf_joint" type="revolute">
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <parent link="thigh"/><child link="calf"/>
    <axis xyz="0 1 0"/>
  </joint>
  <joint name="foot_joint" type="fixed">
    <origin xyz="0 0 -0.3" rpy="0 0 0"/>
    <parent link="calf"/><child link="foot"/>
  </joint>
</robot>
"""


def _make_leg_kin(tmp_path):
    path = _write(tmp_path, "leg.urdf", _THREE_JOINT_WITH_FOOT_OFFSET_URDF)
    return GenericKinematics(urdf_path=path, base_link="base", foot_links=["foot"])


def test_foot_position_includes_trailing_fixed_offset(tmp_path):
    kin = _make_leg_kin(tmp_path)
    p_foot = kin.foot_position_world(0, p_base=np.zeros(3), R_base=np.eye(3), q_leg=np.zeros(3))
    # base -> hip (0.2,0.1,0) -> thigh (0,0.05,0) -> calf (0,0,-0.3, via
    # calf_joint's own origin) -> foot (0,0,-0.3, via the TRAILING fixed
    # foot_joint -- must not be dropped).
    np.testing.assert_allclose(p_foot, [0.2, 0.15, -0.6], atol=1e-9)


def test_foot_position_rotates_with_nonidentity_base_orientation(tmp_path):
    kin = _make_leg_kin(tmp_path)
    # 90deg yaw base orientation: Rz(90) @ [0.2, 0.15, -0.6]
    R_base = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    p_foot = kin.foot_position_world(0, p_base=np.zeros(3), R_base=R_base, q_leg=np.zeros(3))
    np.testing.assert_allclose(p_foot, [-0.15, 0.2, -0.6], atol=1e-9)


def test_full_jacobian_base_blocks_and_leg_columns(tmp_path):
    kin = _make_leg_kin(tmp_path)
    p_base, R_base, q_leg = np.zeros(3), np.eye(3), np.zeros(3)

    J_full = kin.full_jacobian(0, p_base, R_base, q_leg)
    assert J_full.shape == (3, 18)

    # Base linear block: identity.
    np.testing.assert_allclose(J_full[:, 0:3], np.eye(3), atol=1e-12)

    # Base angular block: -skew(r) @ R_base, r = p_foot - p_base.
    p_foot = kin.foot_position_world(0, p_base, R_base, q_leg)
    r = p_foot - p_base
    skew_r = np.array([
        [0.0, -r[2], r[1]],
        [r[2], 0.0, -r[0]],
        [-r[1], r[0], 0.0],
    ])
    np.testing.assert_allclose(J_full[:, 3:6], -skew_r @ R_base, atol=1e-9)

    # Leg columns (leg 0 -> qvel cols 6:9) must match leg_jacobian exactly.
    J_leg = kin.leg_jacobian(0, p_base, R_base, q_leg)
    np.testing.assert_allclose(J_full[:, 6:9], J_leg, atol=1e-12)

    # All other legs' columns (9:18) are zero for a single-leg query.
    np.testing.assert_allclose(J_full[:, 9:18], np.zeros((3, 9)), atol=1e-12)


def test_leg_jacobian_matches_finite_difference_with_rotated_base(tmp_path):
    kin = _make_leg_kin(tmp_path)
    q0 = np.array([0.2, -0.3, 0.4])
    p_base = np.zeros(3)
    R_base = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # 90deg yaw

    J = kin.leg_jacobian(0, p_base, R_base, q0)

    eps = 1e-6
    J_fd = np.zeros((3, 3))
    for k in range(3):
        dq = np.zeros(3)
        dq[k] = eps
        p_plus = kin.foot_position_world(0, p_base, R_base, q0 + dq)
        p_minus = kin.foot_position_world(0, p_base, R_base, q0 - dq)
        J_fd[:, k] = (p_plus - p_minus) / (2 * eps)

    np.testing.assert_allclose(J, J_fd, atol=1e-6)


def test_full_jacobian_base_angular_block_with_rotated_base(tmp_path):
    kin = _make_leg_kin(tmp_path)
    q_leg = np.array([0.1, 0.2, -0.15])
    p_base = np.zeros(3)
    R_base = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # 90deg yaw

    J_full = kin.full_jacobian(0, p_base, R_base, q_leg)
    p_foot = kin.foot_position_world(0, p_base, R_base, q_leg)
    r = p_foot - p_base
    skew_r = np.array([
        [0.0, -r[2], r[1]],
        [r[2], 0.0, -r[0]],
        [-r[1], r[0], 0.0],
    ])
    np.testing.assert_allclose(J_full[:, 3:6], -skew_r @ R_base, atol=1e-9)
