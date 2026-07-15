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


_ONE_JOINT_WITH_MASS_URDF = """<?xml version="1.0"?>
<robot name="r">
  <link name="base"/>
  <link name="foot">
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0.1 0 0" rpy="0 0 0"/>
    </inertial>
  </link>
  <joint name="j1" type="revolute">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="base"/><child link="foot"/>
    <axis xyz="0 1 0"/>
  </joint>
</robot>
"""


def test_gravity_compensation_single_link_horizontal_arm(tmp_path):
    path = _write(tmp_path, "arm.urdf", _ONE_JOINT_WITH_MASS_URDF)
    kin = GenericKinematics(urdf_path=path, base_link="base", foot_links=["foot"])

    # Joint axis is world Y, link CoM sits 0.1m along local X at q=0, so the
    # arm points along world X: a static torque balancing gravity about the
    # Y axis is expected. tau = cross(z=[0,1,0], r=[0.1,0,0]) . (m*mg) with
    # mg = [0,0,+9.81] (matches Go2Kinematics' sign convention):
    # cross([0,1,0],[0.1,0,0]) = [0*0-0*0, 0*0.1-0*0, 0*0-1*0.1] = [0,0,-0.1]
    # tau = [0,0,-0.1] . [0,0,2.0*9.81] = -0.1 * 19.62 = -1.962
    tau = kin.gravity_compensation(0, p_base=np.zeros(3), R_base=np.eye(3), q_leg=np.array([0.0]))
    np.testing.assert_allclose(tau, [-1.962], atol=1e-6)


def test_gravity_compensation_zero_when_arm_hangs_straight_down(tmp_path):
    path = _write(tmp_path, "arm.urdf", _ONE_JOINT_WITH_MASS_URDF)
    kin = GenericKinematics(urdf_path=path, base_link="base", foot_links=["foot"])

    # Rotate -90deg about Y so the link's local +X (where the CoM sits)
    # points along world -Z (hanging straight down) -- lever arm about the
    # Y axis has zero moment, so gravity torque should vanish.
    tau = kin.gravity_compensation(0, p_base=np.zeros(3), R_base=np.eye(3), q_leg=np.array([-np.pi / 2]))
    np.testing.assert_allclose(tau, [0.0], atol=1e-9)


# base -(j1, revolute, axis Y)-> mid (NO <inertial> at all) -(fixed_offset,
# fixed, translation only)-> real_mass_link (the real 2.0kg mass). Mirrors
# ANYmal's actual LF_HAA -> LF_HIP (0.001kg placeholder) -> LF_hip_fixed
# (0.74kg real mass) pattern: the mass sits one `fixed`-joint hop past the
# joint's immediate child, not on the immediate child itself.
_INTERMEDIATE_FIXED_MASS_URDF = """<?xml version="1.0"?>
<robot name="r">
  <link name="base"/>
  <link name="mid"/>
  <link name="real_mass_link">
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0.0 0 0" rpy="0 0 0"/>
    </inertial>
  </link>
  <joint name="j1" type="revolute">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="base"/><child link="mid"/>
    <axis xyz="0 1 0"/>
  </joint>
  <joint name="fixed_offset" type="fixed">
    <origin xyz="0.1 0 0" rpy="0 0 0"/>
    <parent link="mid"/><child link="real_mass_link"/>
  </joint>
</robot>
"""


def test_gravity_compensation_attributes_mass_on_intermediate_fixed_link(tmp_path):
    path = _write(tmp_path, "arm.urdf", _INTERMEDIATE_FIXED_MASS_URDF)
    kin = GenericKinematics(urdf_path=path, base_link="base", foot_links=["real_mass_link"])

    # Same geometry as test_gravity_compensation_single_link_horizontal_arm
    # (0.1m lever arm along X, 2.0kg, axis Y) so the expected torque is
    # identical: -1.962 Nm. A design keyed only on j1's immediate child
    # ("mid", which has no <inertial>) would wrongly compute zero here.
    tau = kin.gravity_compensation(0, p_base=np.zeros(3), R_base=np.eye(3), q_leg=np.array([0.0]))
    np.testing.assert_allclose(tau, [-1.962], atol=1e-6)


# 2-joint leg where joint0 has TWO rigidly-attached links carrying mass:
# its immediate child "mid" AND "mid2", reached via a fixed-joint hop that
# sits SERIALLY on the path to the foot (mid -> [fixed] -> mid2 -> [j2] ->
# foot), unlike a dead-end side branch. joint1's link "foot" is distal to
# joint0. This is the coverage gap the brief's own gravity tests don't
# reach: they use exactly one massed link per test and only ever one
# actuated joint. A design that dropped all but the first entry of
# rigid_links, or that failed to accumulate a distal joint's mass into a
# proximal joint's torque, would still pass those tests but fail here.
_TWO_JOINT_MULTI_LINK_URDF = """<?xml version="1.0"?>
<robot name="r">
  <link name="base"/>
  <link name="mid">
    <inertial>
      <mass value="1.0"/>
      <origin xyz="0.05 0 0" rpy="0 0 0"/>
    </inertial>
  </link>
  <link name="mid2">
    <inertial>
      <mass value="0.5"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
    </inertial>
  </link>
  <link name="foot">
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0.05 0 0" rpy="0 0 0"/>
    </inertial>
  </link>
  <joint name="j1" type="revolute">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="base"/><child link="mid"/>
    <axis xyz="0 1 0"/>
  </joint>
  <joint name="mid_fixed_hop" type="fixed">
    <origin xyz="0.02 0 0" rpy="0 0 0"/>
    <parent link="mid"/><child link="mid2"/>
  </joint>
  <joint name="j2" type="revolute">
    <origin xyz="0.08 0 0" rpy="0 0 0"/>
    <parent link="mid2"/><child link="foot"/>
    <axis xyz="0 1 0"/>
  </joint>
</robot>
"""


def test_gravity_compensation_sums_multiple_rigid_links_and_distal_joint(tmp_path):
    path = _write(tmp_path, "arm.urdf", _TWO_JOINT_MULTI_LINK_URDF)
    kin = GenericKinematics(urdf_path=path, base_link="base", foot_links=["foot"])

    # At q=[0,0], with axis Y on both joints:
    #   p_com_mid  = (0.05, 0, 0), m=1.0, owned by joint0
    #   p_com_mid2 = (0.02, 0, 0), m=0.5, owned by joint0 (fixed hop, serial)
    #   p_com_foot = (0.15, 0, 0), m=2.0, owned by joint1
    # tau[1] sums only foot (owning_k=1 >= 1):
    #   cross([0,1,0],[0.05,0,0]).(2.0*[0,0,9.81]) = -0.05*19.62 = -0.981
    # tau[0] sums mid + mid2 + foot (all owning_k >= 0):
    #   mid:  -0.05*9.81  = -0.4905
    #   mid2: -0.02*4.905 = -0.0981
    #   foot: -0.15*19.62 = -2.943
    #   total = -3.5316
    tau = kin.gravity_compensation(0, p_base=np.zeros(3), R_base=np.eye(3), q_leg=np.array([0.0, 0.0]))
    np.testing.assert_allclose(tau, [-3.5316, -0.981], atol=1e-6)
