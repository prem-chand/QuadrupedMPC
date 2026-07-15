import numpy as np
import pytest
from go2_mpc.kinematics.generic.urdf_parser import parse_urdf, URDFParseError

_SIMPLE_URDF = """<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base">
    <inertial>
      <mass value="5.0"/>
      <origin xyz="0.01 0.02 0.03" rpy="0 0 0"/>
    </inertial>
  </link>
  <link name="link1">
    <inertial>
      <mass value="1.5"/>
      <origin xyz="0 0 -0.1" rpy="0 0 0"/>
    </inertial>
  </link>
  <link name="foot">
  </link>
  <joint name="joint1" type="revolute">
    <origin xyz="0.2 0.1 0" rpy="0 0 0"/>
    <parent link="base"/>
    <child link="link1"/>
    <axis xyz="1 0 0"/>
  </joint>
  <joint name="foot_joint" type="fixed">
    <origin xyz="0 0 -0.2" rpy="0 0 0"/>
    <parent link="link1"/>
    <child link="foot"/>
  </joint>
</robot>
"""

_MALFORMED_URDF = "<robot><link name=\"base\"></robot>"  # unclosed <link>

_NO_ROOT_URDF = """<?xml version="1.0"?>
<robot name="cyclic">
  <link name="a"/>
  <link name="b"/>
  <joint name="j1" type="fixed">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="a"/>
    <child link="b"/>
  </joint>
  <joint name="j2" type="fixed">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="b"/>
    <child link="a"/>
  </joint>
</robot>
"""

_ZERO_AXIS_URDF = """<?xml version="1.0"?>
<robot name="bad_axis">
  <link name="base"/>
  <link name="link1"/>
  <joint name="joint1" type="revolute">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="base"/>
    <child link="link1"/>
    <axis xyz="0 0 0"/>
  </joint>
</robot>
"""


def _write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content)
    return p


def test_parse_basic_tree_structure(tmp_path):
    path = _write(tmp_path, "simple.urdf", _SIMPLE_URDF)
    tree = parse_urdf(path)

    assert tree.root_link == "base"
    assert set(tree.links.keys()) == {"base", "link1", "foot"}
    assert len(tree.joints) == 2


def test_parse_inertial_values(tmp_path):
    path = _write(tmp_path, "simple.urdf", _SIMPLE_URDF)
    tree = parse_urdf(path)

    base_inertial = tree.links["base"].inertial
    assert base_inertial.mass == pytest.approx(5.0)
    np.testing.assert_allclose(base_inertial.com, [0.01, 0.02, 0.03])

    assert tree.links["foot"].inertial is None


def test_parse_joint_origin_and_axis(tmp_path):
    path = _write(tmp_path, "simple.urdf", _SIMPLE_URDF)
    tree = parse_urdf(path)

    joint1 = next(j for j in tree.joints if j.name == "joint1")
    assert joint1.joint_type == "revolute"
    assert joint1.parent == "base"
    assert joint1.child == "link1"
    np.testing.assert_allclose(joint1.origin_xyz, [0.2, 0.1, 0.0])
    np.testing.assert_allclose(joint1.axis, [1.0, 0.0, 0.0])

    foot_joint = next(j for j in tree.joints if j.name == "foot_joint")
    assert foot_joint.joint_type == "fixed"
    assert foot_joint.axis is None


def test_malformed_xml_raises_urdf_parse_error(tmp_path):
    path = _write(tmp_path, "bad.urdf", _MALFORMED_URDF)
    with pytest.raises(URDFParseError):
        parse_urdf(path)


def test_ambiguous_root_raises_urdf_parse_error(tmp_path):
    path = _write(tmp_path, "cyclic.urdf", _NO_ROOT_URDF)
    with pytest.raises(URDFParseError):
        parse_urdf(path)


def test_zero_length_axis_raises_urdf_parse_error(tmp_path):
    path = _write(tmp_path, "bad_axis.urdf", _ZERO_AXIS_URDF)
    with pytest.raises(URDFParseError):
        parse_urdf(path)
