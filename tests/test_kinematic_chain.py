import numpy as np
import pytest
from go2_mpc.kinematics.generic.urdf_parser import parse_urdf
from go2_mpc.kinematics.generic.kinematic_chain import (
    build_leg_chains, URDFTopologyError, UnsupportedJointTypeError,
)

# Straight-line chain, no fixed joints at all: base -> j1 (revolute) -> foot.
_NO_FIXED_URDF = """<?xml version="1.0"?>
<robot name="r">
  <link name="base"/>
  <link name="foot"/>
  <joint name="j1" type="revolute">
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <parent link="base"/><child link="foot"/>
    <axis xyz="0 0 1"/>
  </joint>
</robot>
"""

# base -> j1 (revolute) -> mid -> fixed_rot (fixed, 90deg yaw) -> pre_foot
#      -> j2 (revolute) -> post -> fixed_trans (fixed, pure translation) -> foot
# Exercises: a rotating fixed joint BETWEEN two actuated joints, and a
# trailing fixed joint AFTER the last actuated joint.
_MIXED_URDF = """<?xml version="1.0"?>
<robot name="r">
  <link name="base"/>
  <link name="mid"/>
  <link name="pre_foot"/>
  <link name="post"/>
  <link name="foot"/>
  <joint name="j1" type="revolute">
    <origin xyz="0.5 0 0" rpy="0 0 0"/>
    <parent link="base"/><child link="mid"/>
    <axis xyz="0 0 1"/>
  </joint>
  <joint name="fixed_rot" type="fixed">
    <origin xyz="0 0 0" rpy="0 0 1.5707963267948966"/>
    <parent link="mid"/><child link="pre_foot"/>
  </joint>
  <joint name="j2" type="revolute">
    <origin xyz="0.3 0 0" rpy="0 0 0"/>
    <parent link="pre_foot"/><child link="post"/>
    <axis xyz="1 0 0"/>
  </joint>
  <joint name="fixed_trans" type="fixed">
    <origin xyz="0.2 0 0" rpy="0 0 0"/>
    <parent link="post"/><child link="foot"/>
  </joint>
</robot>
"""

_DISCONNECTED_URDF = """<?xml version="1.0"?>
<robot name="r">
  <link name="base"/>
  <link name="leg1"/>
  <link name="leg1_foot"/>
  <link name="leg2"/>
  <link name="leg2_foot"/>
  <joint name="j1" type="fixed">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="base"/><child link="leg1"/>
  </joint>
  <joint name="j2" type="fixed">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="leg1"/><child link="leg1_foot"/>
  </joint>
  <joint name="j3" type="fixed">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="base"/><child link="leg2"/>
  </joint>
  <joint name="j4" type="fixed">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="leg2"/><child link="leg2_foot"/>
  </joint>
</robot>
"""

_PRISMATIC_URDF = """<?xml version="1.0"?>
<robot name="r">
  <link name="base"/>
  <link name="foot"/>
  <joint name="j1" type="prismatic">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="base"/><child link="foot"/>
    <axis xyz="0 0 1"/>
  </joint>
</robot>
"""


def _write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content)
    return p


def test_no_fixed_joints_single_chain(tmp_path):
    tree = parse_urdf(_write(tmp_path, "r.urdf", _NO_FIXED_URDF))
    chains = build_leg_chains(tree, base_link="base", foot_links=["foot"])

    assert len(chains) == 1
    chain = chains[0]
    assert chain.foot_link == "foot"
    assert len(chain.joints) == 1
    assert chain.joints[0].joint_name == "j1"
    np.testing.assert_allclose(chain.joints[0].offset_transform[:3, 3], [1, 0, 0])
    np.testing.assert_allclose(chain.foot_offset_transform, np.eye(4), atol=1e-12)


def test_fixed_joint_rotation_folds_into_next_actuated_joint(tmp_path):
    tree = parse_urdf(_write(tmp_path, "r.urdf", _MIXED_URDF))
    chain = build_leg_chains(tree, base_link="base", foot_links=["foot"])[0]

    assert [cj.joint_name for cj in chain.joints] == ["j1", "j2"]

    # j2's axis is [1,0,0] in its OWN local frame, but a 90deg yaw fixed
    # joint precedes it, so in the frame chain up to j2 the effective axis
    # (offset_transform's rotation applied to [1,0,0]) must be [0,1,0].
    j2 = chain.joints[1]
    effective_axis = j2.offset_transform[:3, :3] @ j2.axis
    np.testing.assert_allclose(effective_axis, [0.0, 1.0, 0.0], atol=1e-9)

    # j2's translation offset composes j1's revolute origin's rotation (identity
    # here) with the fixed joint's rotation+translation, then j2's own origin.
    # fixed_rot contributes zero translation, so offset should just be j2's
    # own origin (0.3, 0, 0) expressed after the folded rotation.
    np.testing.assert_allclose(j2.offset_transform[:3, 3], [0.0, 0.3, 0.0], atol=1e-9)


def test_trailing_fixed_joint_becomes_foot_offset(tmp_path):
    tree = parse_urdf(_write(tmp_path, "r.urdf", _MIXED_URDF))
    chain = build_leg_chains(tree, base_link="base", foot_links=["foot"])[0]

    # fixed_trans: pure +0.2 translation along X in `post`'s frame, with no
    # rotation, applied after j2. Must show up as foot_offset_transform, NOT
    # be silently dropped.
    np.testing.assert_allclose(chain.foot_offset_transform[:3, 3], [0.2, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(chain.foot_offset_transform[:3, :3], np.eye(3), atol=1e-9)


def test_rigid_links_captures_intermediate_and_trailing_links(tmp_path):
    tree = parse_urdf(_write(tmp_path, "r.urdf", _MIXED_URDF))
    chain = build_leg_chains(tree, base_link="base", foot_links=["foot"])[0]

    j1, j2 = chain.joints

    # j1's own immediate child is "mid" -- recorded at transform=identity
    # (relative to j1's own post-rotation frame).
    assert [name for name, _ in j1.rigid_links] == ["mid", "pre_foot"]
    np.testing.assert_allclose(j1.rigid_links[0][1], np.eye(4), atol=1e-12)
    # "pre_foot" is reached via fixed_rot (90deg yaw, zero translation) from
    # "mid" -- rotation-only offset relative to j1's own frame.
    np.testing.assert_allclose(j1.rigid_links[1][1][:3, :3],
                                [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-9)

    # j2's own immediate child is "post", plus the trailing fixed_trans link
    # "foot" -- both must be present, not just the immediate child.
    assert [name for name, _ in j2.rigid_links] == ["post", "foot"]
    np.testing.assert_allclose(j2.rigid_links[0][1], np.eye(4), atol=1e-12)
    np.testing.assert_allclose(j2.rigid_links[1][1][:3, 3], [0.2, 0.0, 0.0], atol=1e-9)


def test_missing_foot_link_raises(tmp_path):
    tree = parse_urdf(_write(tmp_path, "r.urdf", _NO_FIXED_URDF))
    with pytest.raises(URDFTopologyError):
        build_leg_chains(tree, base_link="base", foot_links=["nonexistent"])


def test_disconnected_foot_raises(tmp_path):
    tree = parse_urdf(_write(tmp_path, "r.urdf", _DISCONNECTED_URDF))
    # leg2_foot is not reachable from leg1 (siblings cannot reach each other)
    with pytest.raises(URDFTopologyError):
        build_leg_chains(tree, base_link="leg1", foot_links=["leg2_foot"])


def test_unsupported_joint_type_raises(tmp_path):
    tree = parse_urdf(_write(tmp_path, "r.urdf", _PRISMATIC_URDF))
    with pytest.raises(UnsupportedJointTypeError):
        build_leg_chains(tree, base_link="base", foot_links=["foot"])
