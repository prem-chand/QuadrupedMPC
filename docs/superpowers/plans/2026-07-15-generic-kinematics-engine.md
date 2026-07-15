# Generic URDF-Driven Kinematics Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a URDF-driven kinematics engine (FK, per-leg Jacobians, gravity compensation) that works for any 3-DOF-per-leg × 4-leg quadruped, additive alongside the existing hand-derived `Go2Kinematics`, validated first on Go2 (two independent ground truths) then on ANYmal (a genuinely new robot).

**Architecture:** A new `go2_mpc/kinematics/generic/` package: a stdlib-only URDF parser builds a plain kinematic tree; a chain-builder walks each foot link back to the base link, folding `fixed` joints into composed static offsets; `GenericKinematics` computes FK via a chained 4×4-homogeneous-transform walk and Jacobians via the classical per-joint-column formula `J[:,k] = z_k × (p_foot − p_k)`, mirroring `go2_kinematics.py`'s math but driven by the parsed tree instead of hand-copied constants. `MujocoRobot` gets one new optional constructor parameter so it can use either kinematics backend.

**Tech Stack:** Python, NumPy, `xml.etree.ElementTree` (stdlib), MuJoCo (validation only), pytest.

## Global Constraints

- No new dependencies: `xml.etree.ElementTree` (stdlib) + NumPy only in the core `generic/` package. `mujoco` is used only in the validation scripts (Tasks 7–8), not in the core library.
- Confirmed choice: custom minimal parser. No Pinocchio, urdfpy, or yourdfpy.
- Kinematics only — no mass matrix `M(q)`, no Coriolis terms. Only FK, per-leg Jacobians, and static gravity compensation (same scope as `Go2Kinematics`).
- Fixed topology only: 3 revolute joints per leg × 4 legs = 12 actuated joints, floating 6-DOF base (18 total `nv`, matching MuJoCo's `qvel` layout: `[vx,vy,vz,wx,wy,wz | leg0(3) | leg1(3) | leg2(3) | leg3(3)]`).
- `go2_mpc/kinematics/go2_kinematics.py` is **not modified**. The generic engine is additive.
- `GenericKinematics` must expose the same public method surface as `Go2Kinematics`: `forward_kinematics`, `foot_position_world`, `leg_jacobian`, `full_jacobian`, `foot_velocity`, `gravity_compensation` — same parameter order and shapes (`leg_idx`, `p_base (3,)`, `R_base (3,3)`, `q_leg (3,)`, ...).
- `MujocoRobot(model, data)` — the exact call `main.py` uses today — must keep working unchanged after Task 6.
- Leg/foot ordering convention throughout: index 0=FL, 1=FR, 2=RL, 3=RR (matches `HIP_ORIGINS`, `foot_stance_offsets` elsewhere in the codebase).

---

### Task 1: Homogeneous transform utilities

**Files:**
- Create: `go2_mpc/kinematics/generic/__init__.py` (empty)
- Create: `go2_mpc/kinematics/generic/transforms.py`
- Test: `tests/test_generic_transforms.py`

**Interfaces:**
- Produces: `rpy_to_matrix(rpy: np.ndarray) -> np.ndarray` (3,3); `make_transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray` (4,4); `axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray` (3,3); `skew(v: np.ndarray) -> np.ndarray` (3,3). All consumed by Tasks 3–5.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_generic_transforms.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_generic_transforms.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'go2_mpc.kinematics.generic'`

- [ ] **Step 3: Write the implementation**

```python
# go2_mpc/kinematics/generic/transforms.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_generic_transforms.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add go2_mpc/kinematics/generic/__init__.py go2_mpc/kinematics/generic/transforms.py tests/test_generic_transforms.py
git commit -m "feat: add homogeneous-transform utilities for generic kinematics engine"
```

---

### Task 2: URDF parser

**Files:**
- Create: `go2_mpc/kinematics/generic/urdf_parser.py`
- Test: `tests/test_urdf_parser.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `URDFParseError` (Exception); `Inertial(mass: float, com: np.ndarray)`, `Link(name: str, inertial: Inertial | None)`, `Joint(name, joint_type, parent, child, origin_xyz, origin_rpy, axis)`, `KinematicTree(links: dict[str, Link], joints: list[Joint], root_link: str)` (all dataclasses); `parse_urdf(path) -> KinematicTree`. Consumed by Task 3.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_urdf_parser.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_urdf_parser.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'go2_mpc.kinematics.generic.urdf_parser'`

- [ ] **Step 3: Write the implementation**

```python
# go2_mpc/kinematics/generic/urdf_parser.py
"""
Minimal URDF parser producing a plain kinematic tree.

Only <link> (name, <inertial> mass/CoM) and <joint> (name, type, parent,
child, <origin>, <axis>) are read. <visual>/<collision>/mesh geometry is
ignored entirely -- irrelevant to kinematics.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET

import numpy as np


class URDFParseError(Exception):
    """Raised when a URDF file is malformed or its topology is invalid."""


@dataclass
class Inertial:
    mass: float
    com: np.ndarray  # (3,) CoM position in the link's local frame


@dataclass
class Link:
    name: str
    inertial: Optional[Inertial]


@dataclass
class Joint:
    name: str
    joint_type: str  # 'revolute', 'continuous', 'fixed', 'prismatic', 'planar', 'floating'
    parent: str
    child: str
    origin_xyz: np.ndarray  # (3,)
    origin_rpy: np.ndarray  # (3,)
    axis: Optional[np.ndarray]  # (3,) unit vector; None for joints with no <axis>


@dataclass
class KinematicTree:
    links: dict
    joints: list
    root_link: str


def _parse_vec3(s) -> np.ndarray:
    if s is None:
        return np.zeros(3)
    return np.array([float(x) for x in s.split()], dtype=np.float64)


def parse_urdf(path) -> KinematicTree:
    path = Path(path)
    try:
        xml_tree = ET.parse(path)
    except ET.ParseError as e:
        raise URDFParseError(f"Failed to parse URDF at {path}: {e}") from e

    root = xml_tree.getroot()
    if root.tag != "robot":
        raise URDFParseError(f"{path}: root element is <{root.tag}>, expected <robot>")

    links = {}
    for link_el in root.findall("link"):
        name = link_el.get("name")
        if name is None:
            raise URDFParseError(f"{path}: <link> missing 'name' attribute")

        inertial = None
        inertial_el = link_el.find("inertial")
        if inertial_el is not None:
            mass_el = inertial_el.find("mass")
            mass = float(mass_el.get("value")) if mass_el is not None else 0.0
            origin_el = inertial_el.find("origin")
            com = _parse_vec3(origin_el.get("xyz") if origin_el is not None else None)
            inertial = Inertial(mass=mass, com=com)

        links[name] = Link(name=name, inertial=inertial)

    if not links:
        raise URDFParseError(f"{path}: no <link> elements found")

    joints = []
    child_links = set()
    for joint_el in root.findall("joint"):
        name = joint_el.get("name")
        joint_type = joint_el.get("type")
        if name is None or joint_type is None:
            raise URDFParseError(f"{path}: <joint> missing 'name' or 'type' attribute")

        parent_el = joint_el.find("parent")
        child_el = joint_el.find("child")
        if parent_el is None or child_el is None:
            raise URDFParseError(f"{path}: joint '{name}' missing <parent> or <child>")
        parent = parent_el.get("link")
        child = child_el.get("link")

        origin_el = joint_el.find("origin")
        origin_xyz = _parse_vec3(origin_el.get("xyz") if origin_el is not None else None)
        origin_rpy = _parse_vec3(origin_el.get("rpy") if origin_el is not None else None)

        axis = None
        if joint_type in ("revolute", "continuous", "prismatic"):
            axis_el = joint_el.find("axis")
            axis_xyz = _parse_vec3(axis_el.get("xyz") if axis_el is not None else "1 0 0")
            norm = np.linalg.norm(axis_xyz)
            if norm < 1e-9:
                raise URDFParseError(f"{path}: joint '{name}' has a zero-length <axis>")
            axis = axis_xyz / norm

        joints.append(Joint(
            name=name, joint_type=joint_type, parent=parent, child=child,
            origin_xyz=origin_xyz, origin_rpy=origin_rpy, axis=axis,
        ))
        child_links.add(child)

    root_candidates = [name for name in links if name not in child_links]
    if len(root_candidates) != 1:
        raise URDFParseError(
            f"{path}: expected exactly one root link (a link that is never "
            f"a <child>), found {len(root_candidates)}: {root_candidates}"
        )

    return KinematicTree(links=links, joints=joints, root_link=root_candidates[0])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_urdf_parser.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add go2_mpc/kinematics/generic/urdf_parser.py tests/test_urdf_parser.py
git commit -m "feat: add minimal URDF parser for generic kinematics engine"
```

---

### Task 3: Leg chain builder

This is the trickiest correctness point in the whole engine: a URDF frequently has (a) `fixed` joints *between* two actuated joints whose `rpy` is **not** zero (this reorients the next actuated joint's effective world-frame axis — ANYmal's URDF does this for every leg joint), and (b) a trailing `fixed` joint *after* the last actuated joint down to the actual foot/toe frame (Go2's URDF does exactly this: `FL_foot_joint` is `fixed`, child of `FL_calf`, translating `(0,0,-0.213)` to reach `FL_foot`). Both must be composed via full 4×4 transforms, not naive angle addition, and the trailing case needs its own field since there's no "next actuated joint" to attach it to.

**Files:**
- Create: `go2_mpc/kinematics/generic/kinematic_chain.py`
- Test: `tests/test_kinematic_chain.py`

**Interfaces:**
- Consumes: `KinematicTree`, `Joint` from Task 2 (`go2_mpc.kinematics.generic.urdf_parser`); `make_transform` from Task 1 (`go2_mpc.kinematics.generic.transforms`).
- Produces: `URDFTopologyError`, `UnsupportedJointTypeError` (Exceptions); `ChainJoint(joint_name: str, child_link: str, axis: np.ndarray, offset_transform: np.ndarray)`, `KinematicChain(foot_link: str, joints: list[ChainJoint], foot_offset_transform: np.ndarray)` (dataclasses); `build_leg_chains(tree: KinematicTree, base_link: str, foot_links: list[str]) -> list[KinematicChain]`. Consumed by Task 4.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_kinematic_chain.py
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
  <link name="island"/>
  <link name="foot"/>
  <joint name="j1" type="fixed">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="island"/><child link="foot"/>
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


def test_missing_foot_link_raises(tmp_path):
    tree = parse_urdf(_write(tmp_path, "r.urdf", _NO_FIXED_URDF))
    with pytest.raises(URDFTopologyError):
        build_leg_chains(tree, base_link="base", foot_links=["nonexistent"])


def test_disconnected_foot_raises(tmp_path):
    tree = parse_urdf(_write(tmp_path, "r.urdf", _DISCONNECTED_URDF))
    with pytest.raises(URDFTopologyError):
        build_leg_chains(tree, base_link="base", foot_links=["foot"])


def test_unsupported_joint_type_raises(tmp_path):
    tree = parse_urdf(_write(tmp_path, "r.urdf", _PRISMATIC_URDF))
    with pytest.raises(UnsupportedJointTypeError):
        build_leg_chains(tree, base_link="base", foot_links=["foot"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_kinematic_chain.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'go2_mpc.kinematics.generic.kinematic_chain'`

- [ ] **Step 3: Write the implementation**

```python
# go2_mpc/kinematics/generic/kinematic_chain.py
"""
Builds base->foot kinematic chains from a parsed URDF tree.

Fixed joints are folded into a composed static 4x4 transform: those
between two actuated joints get attached to the *next* actuated joint's
offset; any fixed joints remaining after the *last* actuated joint (e.g.
a foot/toe frame rigidly bolted to the last leg link) become the chain's
`foot_offset_transform`, applied once at the very end of FK.
"""

from dataclasses import dataclass, field
import numpy as np

from .urdf_parser import KinematicTree
from .transforms import make_transform


class URDFTopologyError(Exception):
    """Raised when a foot link cannot be traced back to the base link."""


class UnsupportedJointTypeError(Exception):
    """Raised when a leg chain contains a joint type this engine can't handle."""


_ACTUATED_TYPES = ("revolute", "continuous")
_FIXED_TYPES = ("fixed",)


@dataclass
class ChainJoint:
    joint_name: str
    child_link: str          # link this joint moves -- needed to look up mass/CoM later
    axis: np.ndarray          # (3,) unit vector, in this joint's own local frame
    offset_transform: np.ndarray  # (4,4) static transform from the previous actuated
                                   # joint's frame (or base) up to this joint's frame,
                                   # BEFORE this joint's own DOF rotation is applied


@dataclass
class KinematicChain:
    foot_link: str
    joints: list  # list[ChainJoint], base -> foot order
    foot_offset_transform: np.ndarray = field(default_factory=lambda: np.eye(4))


def build_leg_chains(tree: KinematicTree, base_link: str, foot_links: list) -> list:
    child_to_joint = {j.child: j for j in tree.joints}

    chains = []
    for foot_link in foot_links:
        if foot_link not in tree.links:
            raise URDFTopologyError(f"foot link '{foot_link}' not found in URDF")

        # Walk from the foot up to the base, collecting joints foot->base.
        path_joints = []
        current = foot_link
        visited = {current}
        while current != base_link:
            joint = child_to_joint.get(current)
            if joint is None:
                raise URDFTopologyError(
                    f"foot link '{foot_link}' does not connect to base link "
                    f"'{base_link}' (stopped at '{current}')"
                )
            path_joints.append(joint)
            current = joint.parent
            if current in visited:
                raise URDFTopologyError(f"cycle detected while tracing '{foot_link}' to base")
            visited.add(current)
        path_joints.reverse()  # now base -> foot order

        chain_joints = []
        pending_T = np.eye(4)
        for joint in path_joints:
            joint_T = make_transform(joint.origin_xyz, joint.origin_rpy)

            if joint.joint_type in _FIXED_TYPES:
                pending_T = pending_T @ joint_T
                continue

            if joint.joint_type not in _ACTUATED_TYPES:
                raise UnsupportedJointTypeError(
                    f"joint '{joint.name}' has type '{joint.joint_type}'; "
                    f"only {_ACTUATED_TYPES + _FIXED_TYPES} are supported"
                )

            chain_joints.append(ChainJoint(
                joint_name=joint.name,
                child_link=joint.child,
                axis=joint.axis,
                offset_transform=pending_T @ joint_T,
            ))
            pending_T = np.eye(4)

        # Anything left in pending_T is fixed joints AFTER the last actuated
        # joint -- e.g. a foot/toe frame. Do not drop this silently.
        chains.append(KinematicChain(
            foot_link=foot_link,
            joints=chain_joints,
            foot_offset_transform=pending_T,
        ))

    return chains
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_kinematic_chain.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add go2_mpc/kinematics/generic/kinematic_chain.py tests/test_kinematic_chain.py
git commit -m "feat: add leg chain builder with fixed-joint folding for generic kinematics"
```

---

### Task 4: `GenericKinematics` — forward kinematics and Jacobians

**Files:**
- Create: `go2_mpc/kinematics/generic/generic_kinematics.py`
- Test: `tests/test_generic_kinematics.py`

**Interfaces:**
- Consumes: `parse_urdf`, `KinematicTree` from Task 2; `build_leg_chains`, `KinematicChain` from Task 3; `axis_angle_to_matrix`, `skew` from Task 1.
- Produces: `GenericKinematics(urdf_path, base_link, foot_links, leg_order=None)` with methods `forward_kinematics(leg_idx, p_base, R_base, q_leg) -> dict`, `foot_position_world(leg_idx, p_base, R_base, q_leg) -> np.ndarray (3,)`, `leg_jacobian(leg_idx, p_base, R_base, q_leg) -> np.ndarray (3,3)`, `full_jacobian(leg_idx, p_base, R_base, q_leg) -> np.ndarray (3,18)`, `foot_velocity(leg_idx, p_base, R_base, q_leg, v_base, omega_body, qd_leg) -> np.ndarray (3,)`. Consumed by Task 5 (same class, gravity_compensation) and Task 6 (`MujocoRobot`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_generic_kinematics.py
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
```

Note: `q_leg`/`qd_leg` are 2-dim for this synthetic 2-joint fixture (not the usual 3), which is fine — `GenericKinematics` must work for whatever number of actuated joints a chain actually has; the "3-DOF-per-leg" assumption is a property of the *real* robot configs (Task 7/8), not a hardcoded array size inside `GenericKinematics` itself. `leg_jacobian`'s returned matrix is always `(3, len(chain.joints))`; the test above slices `J[:, :2]` since the real Go2/ANYmal chains will produce `(3, 3)`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_generic_kinematics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'go2_mpc.kinematics.generic.generic_kinematics'`

- [ ] **Step 3: Write the implementation**

```python
# go2_mpc/kinematics/generic/generic_kinematics.py
"""
URDF-driven analytical kinematics for any 3-DOF-per-leg x 4-leg quadruped.

Public method surface mirrors go2_mpc.kinematics.go2_kinematics.Go2Kinematics
so the two are interchangeable behind the Robot interface.
"""

import numpy as np

from .urdf_parser import parse_urdf
from .kinematic_chain import build_leg_chains
from .transforms import axis_angle_to_matrix, skew

# Column indices into the (3, 18) full Jacobian for each leg's joint DOFs.
# qvel layout: [vx,vy,vz | wx_B,wy_B,wz_B | leg0(3) | leg1(3) | leg2(3) | leg3(3)]
_LEG_DOF_COLS = [
    [6, 7, 8],
    [9, 10, 11],
    [12, 13, 14],
    [15, 16, 17],
]


class GenericKinematics:
    """
    Analytical FK/Jacobians/gravity-comp for a robot described by a URDF.

    All methods are stateless -- each call takes the full leg state
    (p_base, R_base, q_leg) and returns the requested quantity, exactly
    like Go2Kinematics.
    """

    def __init__(self, urdf_path, base_link, foot_links, leg_order=None):
        self._tree = parse_urdf(urdf_path)
        ordered_foot_links = (
            list(foot_links) if leg_order is None
            else [foot_links[i] for i in leg_order]
        )
        self._chains = build_leg_chains(self._tree, base_link, ordered_foot_links)
        self._gravity = np.array([0.0, 0.0, -9.81])

    # ------------------------------------------------------------------
    # Forward kinematics
    # ------------------------------------------------------------------

    def forward_kinematics(self, leg_idx, p_base, R_base, q_leg):
        """
        Compute the full FK frame chain for one leg.

        Returns a dict with:
          joint_frames -- list of {"p": (3,), "R": (3,3), "z": (3,)} per
                           actuated joint, base->foot order. "p"/"R" are the
                           joint's position/cumulative rotation BEFORE its
                           own DOF rotation is applied; "z" is its rotation
                           axis expressed in the world frame.
          p_foot -- (3,) foot position in world frame.
        """
        chain = self._chains[leg_idx]

        T_current = np.eye(4)
        T_current[:3, :3] = R_base
        T_current[:3, 3] = p_base

        joint_frames = []
        for cj, qi in zip(chain.joints, q_leg):
            T_current = T_current @ cj.offset_transform
            p_i = T_current[:3, 3].copy()
            R_i = T_current[:3, :3].copy()
            z_i = R_i @ cj.axis

            R_qi = np.eye(4)
            R_qi[:3, :3] = axis_angle_to_matrix(cj.axis, qi)
            T_current = T_current @ R_qi

            joint_frames.append({"p": p_i, "R": R_i, "z": z_i})

        T_current = T_current @ chain.foot_offset_transform

        return {"joint_frames": joint_frames, "p_foot": T_current[:3, 3].copy()}

    def foot_position_world(self, leg_idx, p_base, R_base, q_leg):
        """Foot position in world frame. Shape: (3,)."""
        return self.forward_kinematics(leg_idx, p_base, R_base, q_leg)["p_foot"].copy()

    # ------------------------------------------------------------------
    # Jacobians
    # ------------------------------------------------------------------

    def leg_jacobian(self, leg_idx, p_base, R_base, q_leg):
        """
        Geometric Jacobian mapping leg joint velocities -> foot Cartesian
        velocity in world frame. Shape: (3, n_joints_in_leg).

        J[:, k] = z_k x (p_foot - p_joint_k)
        """
        fk = self.forward_kinematics(leg_idx, p_base, R_base, q_leg)
        p_foot = fk["p_foot"]
        joint_frames = fk["joint_frames"]

        J = np.empty((3, len(joint_frames)), dtype=np.float64)
        for k, jf in enumerate(joint_frames):
            J[:, k] = np.cross(jf["z"], p_foot - jf["p"])
        return J

    def full_jacobian(self, leg_idx, p_base, R_base, q_leg):
        """
        Full positional Jacobian mapping all 18 generalised velocities
        (6 floating-base DOFs + 12 leg joint DOFs) -> foot Cartesian
        velocity in world frame. Shape: (3, 18).
        """
        fk = self.forward_kinematics(leg_idx, p_base, R_base, q_leg)
        p_foot = fk["p_foot"]

        J_full = np.zeros((3, 18), dtype=np.float64)
        J_full[:, 0:3] = np.eye(3)

        r = p_foot - p_base
        J_full[:, 3:6] = -skew(r) @ R_base

        J_leg = self.leg_jacobian(leg_idx, p_base, R_base, q_leg)
        for k, col in enumerate(_LEG_DOF_COLS[leg_idx]):
            J_full[:, col] = J_leg[:, k]

        return J_full

    # ------------------------------------------------------------------
    # Foot velocity
    # ------------------------------------------------------------------

    def foot_velocity(self, leg_idx, p_base, R_base, q_leg, v_base, omega_body, qd_leg):
        """
        Foot Cartesian velocity in world frame. Shape: (3,).

        v_base     : (3,) base linear velocity, world frame.
        omega_body : (3,) base angular velocity, body frame.
        qd_leg     : joint velocities for this leg.
        """
        fk = self.forward_kinematics(leg_idx, p_base, R_base, q_leg)
        p_foot = fk["p_foot"]

        omega_world = R_base @ omega_body
        J_leg = self.leg_jacobian(leg_idx, p_base, R_base, q_leg)

        return v_base + np.cross(omega_world, p_foot - p_base) + J_leg @ qd_leg
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_generic_kinematics.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add go2_mpc/kinematics/generic/generic_kinematics.py tests/test_generic_kinematics.py
git commit -m "feat: add GenericKinematics forward kinematics and Jacobians"
```

---

### Task 5: `GenericKinematics.gravity_compensation`

Mirrors `Go2Kinematics.gravity_compensation`'s formula exactly: `tau[j] = sum over links distal to joint j of cross(z_j, p_link_com_world - p_joint_j) @ (m_link * mg_world)`, where `mg_world = -gravity` (matches the existing code's sign convention so it lines up with MuJoCo's `qfrc_bias`). The one thing this version doesn't have to do that `go2_kinematics.py` does: manually transcribe `HIP_COM`/`THIGH_COM`/`CALF_COM` arrays — mass and CoM come straight from each link's URDF `<inertial>` tag, parsed in Task 2.

**Files:**
- Modify: `go2_mpc/kinematics/generic/generic_kinematics.py` (add method to the same class)
- Modify: `tests/test_generic_kinematics.py` (add tests)

**Interfaces:**
- Consumes: `self._tree.links[name].inertial` (mass, com) from Task 2; `self._chains[leg_idx].joints[k].child_link` from Task 3.
- Produces: `GenericKinematics.gravity_compensation(leg_idx, p_base, R_base, q_leg) -> np.ndarray (3,)`. Consumed by Task 6 (`MujocoRobot`).

- [ ] **Step 1: Write the failing tests**

```python
# Append to tests/test_generic_kinematics.py

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_generic_kinematics.py -k gravity -v`
Expected: FAIL with `AttributeError: 'GenericKinematics' object has no attribute 'gravity_compensation'`

- [ ] **Step 3: Write the implementation**

Add to `go2_mpc/kinematics/generic/generic_kinematics.py`, inside the `GenericKinematics` class:

```python
    # ------------------------------------------------------------------
    # Gravity compensation
    # ------------------------------------------------------------------

    def gravity_compensation(self, leg_idx, p_base, R_base, q_leg):
        """
        Static gravity compensation torques for one leg. Shape: (n_joints,).

        tau[j] = sum over links distal to joint j of:
                   cross(z_j, p_link_com_world - p_joint_j) . (m_link * mg_world)

        mg_world = -self._gravity, matching Go2Kinematics' MuJoCo qfrc_bias
        sign convention. Coriolis/centrifugal terms are NOT included (same
        static-only approximation Go2Kinematics makes).
        """
        chain = self._chains[leg_idx]
        fk = self.forward_kinematics(leg_idx, p_base, R_base, q_leg)
        joint_frames = fk["joint_frames"]
        mg_world = -self._gravity

        # World-frame CoM position of the link each joint moves.
        link_mass_com = []
        for k, cj in enumerate(chain.joints):
            jf = joint_frames[k]
            link = self._tree.links[cj.child_link]
            if link.inertial is None:
                link_mass_com.append((0.0, jf["p"]))
                continue
            R_after_own_rotation = jf["R"] @ axis_angle_to_matrix(cj.axis, q_leg[k])
            p_com_world = jf["p"] + R_after_own_rotation @ link.inertial.com
            link_mass_com.append((link.inertial.mass, p_com_world))

        n = len(chain.joints)
        tau = np.zeros(n, dtype=np.float64)
        for j in range(n):
            z_j, p_j = joint_frames[j]["z"], joint_frames[j]["p"]
            for k in range(j, n):
                m_k, p_com_k = link_mass_com[k]
                if m_k == 0.0:
                    continue
                tau[j] += np.cross(z_j, p_com_k - p_j) @ (m_k * mg_world)
        return tau
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_generic_kinematics.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

```bash
git add go2_mpc/kinematics/generic/generic_kinematics.py tests/test_generic_kinematics.py
git commit -m "feat: add gravity compensation to GenericKinematics"
```

---

### Task 6: Wire `MujocoRobot` to accept a pluggable kinematics backend

**Files:**
- Create: `go2_mpc/kinematics/generic/robot_config.py`
- Modify: `go2_mpc/core/mujoco_robot.py` (constructor only)
- Test: `tests/test_mujoco_robot_kinematics_injection.py`

**Interfaces:**
- Consumes: `GenericKinematics` from Task 4/5; `Go2Kinematics` (existing, unmodified) from `go2_mpc.kinematics.go2_kinematics`.
- Produces: `GenericRobotConfig(urdf_path: Path, base_link: str, foot_links: list[str])` dataclass; `MujocoRobot.__init__(self, model, data, kinematics=None)` — `self._kin = kinematics or Go2Kinematics()`. Consumed by Tasks 7–8.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mujoco_robot_kinematics_injection.py
"""
Verifies MujocoRobot's kinematics backend is now injectable without
changing its default (backward-compatible) behavior.
"""
import mujoco
import numpy as np
from unittest.mock import MagicMock

from go2_mpc.core.mujoco_robot import MujocoRobot
from go2_mpc.kinematics.go2_kinematics import Go2Kinematics


def _make_model_data():
    model = mujoco.MjModel.from_xml_path("go2_mpc/robot/scene.xml")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_default_constructor_still_uses_go2_kinematics():
    model, data = _make_model_data()
    robot = MujocoRobot(model, data)
    assert isinstance(robot._kin, Go2Kinematics)


def test_kinematics_backend_is_injectable():
    model, data = _make_model_data()
    fake_kin = MagicMock()
    robot = MujocoRobot(model, data, kinematics=fake_kin)
    assert robot._kin is fake_kin
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_mujoco_robot_kinematics_injection.py -v`
Expected: FAIL with `TypeError: MujocoRobot.__init__() got an unexpected keyword argument 'kinematics'`

- [ ] **Step 3: Write the implementation**

First, `go2_mpc/kinematics/generic/robot_config.py`:

```python
"""
Per-robot configuration for GenericKinematics: which URDF to parse and
which links are the base and the four feet.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class GenericRobotConfig:
    urdf_path: Path
    base_link: str
    foot_links: List[str]  # order: [FL, FR, RL, RR]
```

Then, in `go2_mpc/core/mujoco_robot.py`, find the `__init__` (currently something like):

```python
    def __init__(self, model, data):
        ...
        self._kin = Go2Kinematics()
```

Change it to:

```python
    def __init__(self, model, data, kinematics=None):
        ...
        self._kin = kinematics if kinematics is not None else Go2Kinematics()
```

(Only the signature and the `self._kin` assignment line change — every other line of `__init__` stays exactly as-is.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_mujoco_robot_kinematics_injection.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the full existing test suite to confirm no regression**

Run: `python -m pytest tests/ -v`
Expected: All existing tests still PASS (the `MujocoRobot(model, data)` call in `main.py` and `tests/test_controllers.py` is unaffected since `kinematics` defaults to `None`)

- [ ] **Step 6: Commit**

```bash
git add go2_mpc/kinematics/generic/robot_config.py go2_mpc/core/mujoco_robot.py tests/test_mujoco_robot_kinematics_injection.py
git commit -m "feat: make MujocoRobot's kinematics backend injectable"
```

---

### Task 7: Vendor Go2's official URDF and validate GenericKinematics against it (Stage A)

Sourced and verified during planning: [`unitreerobotics/unitree_ros`](https://github.com/unitreerobotics/unitree_ros), path `robots/go2_description/urdf/go2_description.urdf`, **BSD 3-Clause License** (confirmed via the repo's `LICENSE` file — permissive, safe to vendor). Cross-checked its `FL_hip_joint`/`FL_thigh_joint`/`FL_calf_joint` origins and axes against `go2_kinematics.py`'s `HIP_ORIGINS`/`THIGH_OFFSETS`/`L_THIGH`/`L_CALF` constants — they match exactly (e.g. `FL_hip_joint origin="0.1934 0.0465 0"` = `HIP_ORIGINS[0] = [0.1934, 0.0465, 0.0]`), which is strong evidence this is the right URDF for the robot this codebase already models. Confirmed topology: base link is `"base"`; each leg's foot link (`FL_foot`, `FR_foot`, `RL_foot`, `RR_foot`) is a **`fixed` child of the calf link** (`FL_foot_joint`, translation `(0,0,-0.213)`) — i.e. exactly the "trailing fixed joint after the last actuated joint" case Task 3 was built to handle correctly. There are also dead-end `fixed`-jointed sibling links (`*_hip_rotor`, `*_calflower`, etc., for rotor inertia / collision geometry) that are not on the foot's ancestry path and are correctly ignored by `build_leg_chains`'s foot-to-base walk.

**Files:**
- Create: `go2_mpc/robot/urdf/go2/go2_description.urdf` (vendored)
- Create: `go2_mpc/robot/urdf/go2/LICENSE` (vendored, BSD 3-Clause from `unitreerobotics/unitree_ros`)
- Create: `go2_mpc/robot/urdf/go2/README.md` (source + license attribution)
- Modify: `go2_mpc/kinematics/generic/robot_config.py` (add `go2_config()`)
- Create: `go2_mpc/kinematics/validate_generic_kinematics.py`
- Test: `tests/test_validate_generic_kinematics_go2.py`

**Interfaces:**
- Consumes: `GenericKinematics` (Task 4/5), `Go2Kinematics` (existing), `GenericRobotConfig` (Task 6).
- Produces: `go2_config() -> GenericRobotConfig`; `strip_visual_and_collision(urdf_path) -> Path` (temp file, used so MuJoCo doesn't need to resolve mesh files it doesn't need for kinematics); consumed by Task 8 too.

- [ ] **Step 1: Vendor the files**

```bash
mkdir -p go2_mpc/robot/urdf/go2
curl -sL https://raw.githubusercontent.com/unitreerobotics/unitree_ros/master/robots/go2_description/urdf/go2_description.urdf \
  -o go2_mpc/robot/urdf/go2/go2_description.urdf
curl -sL https://raw.githubusercontent.com/unitreerobotics/unitree_ros/master/LICENSE \
  -o go2_mpc/robot/urdf/go2/LICENSE
```

Verify the download succeeded and looks like a URDF before proceeding:

Run: `grep -c '<joint name=' go2_mpc/robot/urdf/go2/go2_description.urdf`
Expected: a number >= 12 (at minimum the 12 leg joints)

Create `go2_mpc/robot/urdf/go2/README.md`:

```markdown
# Go2 URDF (vendored)

Source: https://github.com/unitreerobotics/unitree_ros/blob/master/robots/go2_description/urdf/go2_description.urdf
License: BSD 3-Clause (see LICENSE in this directory), Copyright Unitree Robotics.

Used only for kinematics validation (`go2_mpc/kinematics/validate_generic_kinematics.py`)
-- not for simulation. The simulated Go2 model remains `go2_mpc/robot/go2.xml` (MJCF).
```

- [ ] **Step 2: Add the Go2 robot config**

Append to `go2_mpc/kinematics/generic/robot_config.py`:

```python
def go2_config() -> "GenericRobotConfig":
    return GenericRobotConfig(
        urdf_path=Path("go2_mpc/robot/urdf/go2/go2_description.urdf"),
        base_link="base",
        foot_links=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
    )
```

- [ ] **Step 3: Write the validation script**

```python
# go2_mpc/kinematics/validate_generic_kinematics.py
"""
Validates GenericKinematics against trusted ground truth.

Stage A (Go2): vs the existing hand-derived Go2Kinematics, AND vs MuJoCo
   loading the vendored Go2 URDF directly -- Go2 has two independent
   oracles, making it the cheapest place to catch a parser/Jacobian bug.
Stage B (ANYmal, added in a later task): vs MuJoCo loading the vendored
   ANYmal URDF directly (no independent hand-derived ANYmal ground truth
   exists -- that's the actual point of this engine).

Run: python go2_mpc/kinematics/validate_generic_kinematics.py
"""

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import mujoco

from go2_mpc.kinematics.generic.generic_kinematics import GenericKinematics
from go2_mpc.kinematics.generic.robot_config import go2_config
from go2_mpc.kinematics.go2_kinematics import Go2Kinematics


def strip_visual_and_collision(urdf_path) -> Path:
    """
    Return a temp copy of urdf_path with <visual>/<collision> removed.
    Mesh/visual geometry never affects kinematics or <inertial> values, so
    this lets MuJoCo compile a kinematics oracle without needing to resolve
    the URDF's (often ROS package://) mesh file paths.
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    for link in root.findall("link"):
        for tag in ("visual", "collision"):
            for el in list(link.findall(tag)):
                link.remove(el)
    fd, tmp_path = tempfile.mkstemp(suffix=".urdf")
    tree.write(tmp_path)
    return Path(tmp_path)


def _random_leg_angles(n_samples, n_joints, seed=0):
    rng = np.random.default_rng(seed)
    # +/- 0.5 rad sweep around zero -- safely within typical revolute limits
    # for hip/thigh/calf joints on these robots.
    return rng.uniform(-0.5, 0.5, size=(n_samples, n_joints))


def validate_go2_against_hand_derived(n_samples=200, tol=1e-9):
    """Stage A part 1: GenericKinematics vs Go2Kinematics, same random states."""
    cfg = go2_config()
    generic = GenericKinematics(cfg.urdf_path, cfg.base_link, cfg.foot_links)
    hand_derived = Go2Kinematics()

    p_base = np.array([0.1, -0.2, 0.32])
    R_base = np.eye(3)  # zero base tilt is enough to validate the leg chain math
    max_err = 0.0

    for q_leg in _random_leg_angles(n_samples, 3, seed=1):
        for leg_idx in range(4):
            p_generic = generic.foot_position_world(leg_idx, p_base, R_base, q_leg)
            p_hand = hand_derived.foot_position_world(leg_idx, p_base, R_base, q_leg)
            max_err = max(max_err, np.linalg.norm(p_generic - p_hand))

    print(f"[Stage A.1] GenericKinematics vs Go2Kinematics: max foot-position error = {max_err:.3e} m")
    assert max_err < tol, f"max error {max_err} exceeds {tol}"
    return max_err


def validate_against_mujoco_urdf(urdf_path, base_link, foot_links, leg_joint_names,
                                  n_samples=50, tol=1e-6, label=""):
    """
    Stage A part 2 / Stage B: GenericKinematics vs MuJoCo loading the same
    URDF directly. `leg_joint_names` is a list of 4 lists of 3 joint names
    (base->foot order per leg), used to look up each joint's qpos address
    by NAME rather than assuming a fixed numeric layout.
    """
    stripped = strip_visual_and_collision(urdf_path)
    model = mujoco.MjModel.from_xml_path(str(stripped))
    data = mujoco.MjData(model)

    generic = GenericKinematics(urdf_path, base_link, foot_links)

    qpos_addrs = [
        [model.joint(name).qposadr[0] for name in leg_names]
        for leg_names in leg_joint_names
    ]
    base_body_id = model.body(base_link).id
    foot_body_ids = [model.body(name).id for name in foot_links]

    max_err = 0.0
    for q_leg_sample in _random_leg_angles(n_samples, 3, seed=2):
        for leg_idx in range(4):
            data.qpos[:] = 0.0
            for addr, qi in zip(qpos_addrs[leg_idx], q_leg_sample):
                data.qpos[addr] = qi
            mujoco.mj_forward(model, data)

            p_base_mj = data.xpos[base_body_id].copy()
            R_base_mj = data.xmat[base_body_id].reshape(3, 3).copy()
            p_foot_mj = data.xpos[foot_body_ids[leg_idx]].copy()

            p_foot_generic = generic.foot_position_world(leg_idx, p_base_mj, R_base_mj, q_leg_sample)
            max_err = max(max_err, np.linalg.norm(p_foot_mj - p_foot_generic))

    print(f"[{label}] GenericKinematics vs MuJoCo-from-URDF: max foot-position error = {max_err:.3e} m")
    assert max_err < tol, f"max error {max_err} exceeds {tol}"
    return max_err


if __name__ == "__main__":
    validate_go2_against_hand_derived()

    go2_leg_joint_names = [
        ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"],
        ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"],
        ["RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"],
        ["RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"],
    ]
    cfg = go2_config()
    validate_against_mujoco_urdf(
        cfg.urdf_path, cfg.base_link, cfg.foot_links, go2_leg_joint_names,
        label="Stage A.2 (Go2)",
    )
```

**Verification note for whoever runs this**: `mujoco.MjModel.from_xml_path` compiling a raw (stripped) URDF directly is a real MuJoCo feature, but exact behavior (e.g. whether it auto-adds a floating `freejoint` for the root body, and the resulting `qpos`/`nq` layout) should be confirmed empirically the first time this runs — that's exactly why `qpos_addrs` are looked up by joint *name* via `model.joint(name).qposadr`, not by a hardcoded numeric offset. If `MjModel.from_xml_path` errors on the stripped URDF, print the error; likely causes are leftover `<transmission>`/`<gazebo>` tags MuJoCo's URDF compiler doesn't understand, which can be stripped the same way `<visual>`/`<collision>` are.

- [ ] **Step 4: Run the validation script**

Run: `python go2_mpc/kinematics/validate_generic_kinematics.py`
Expected: both `[Stage A.1]` and `[Stage A.2 (Go2)]` print a max error near machine precision (Stage A.1, since both are analytical) or below `1e-6` m (Stage A.2, since MuJoCo's own kinematics is an independent numerical path), with no assertion failure. If it fails, do not proceed to Task 8 until Stage A passes — it is the highest-confidence check available and any failure here indicates a bug in Tasks 1-5, not in how a new robot is configured.

- [ ] **Step 5: Wrap Stage A in a pytest test for CI**

```python
# tests/test_validate_generic_kinematics_go2.py
from go2_mpc.kinematics.validate_generic_kinematics import (
    validate_go2_against_hand_derived,
    validate_against_mujoco_urdf,
)
from go2_mpc.kinematics.generic.robot_config import go2_config


def test_generic_kinematics_matches_go2_kinematics():
    validate_go2_against_hand_derived(n_samples=50, tol=1e-9)


def test_generic_kinematics_matches_mujoco_from_go2_urdf():
    cfg = go2_config()
    leg_joint_names = [
        ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"],
        ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"],
        ["RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"],
        ["RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"],
    ]
    validate_against_mujoco_urdf(
        cfg.urdf_path, cfg.base_link, cfg.foot_links, leg_joint_names,
        n_samples=20, tol=1e-6, label="test",
    )
```

Run: `python -m pytest tests/test_validate_generic_kinematics_go2.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add go2_mpc/robot/urdf/go2/ go2_mpc/kinematics/generic/robot_config.py \
        go2_mpc/kinematics/validate_generic_kinematics.py \
        tests/test_validate_generic_kinematics_go2.py
git commit -m "feat: vendor Go2 URDF and validate GenericKinematics against it (Stage A)"
```

---

### Task 8: Vendor ANYmal's official URDF and validate GenericKinematics against it (Stage B)

Sourced and verified during planning: [`ANYbotics/anymal_c_simple_description`](https://github.com/ANYbotics/anymal_c_simple_description), path `urdf/anymal.urdf`, **BSD-style license** (confirmed via the repo's `LICENSE` file, Copyright ANYbotics AG — permissive, safe to vendor). Confirmed topology: base link `"base"`; foot links `LF_FOOT`, `RF_FOOT`, `LH_FOOT`, `RH_FOOT` (ANYmal's own left/right-front/hind naming, mapped here to this codebase's `[FL, FR, RL, RR]` convention); leg joints named `{LEG}_HAA` (hip abduction, revolute), `{LEG}_HFE` (hip flexion, revolute), `{LEG}_KFE` (knee flexion, revolute). This URDF is a substantially better stress test of Task 3's fixed-joint folding than Go2's: **every** actuated joint (`LF_HFE`, `LF_KFE`, etc.) is preceded by an intermediate `fixed` joint with a **non-zero rotation** (e.g. `LF_hip_fixed_LF_HFE`) that reorients the next joint's effective world-frame axis — unlike Go2, where the leg's fixed joints only ever appear as dead-end sibling branches (rotor/collision links) or as a single trailing translation-only offset to the foot. If Stage A passed but this fails, look first at whether fixed-joint *rotations* (not just translations) are being composed correctly in `kinematic_chain.py`.

**Files:**
- Create: `go2_mpc/robot/urdf/anymal/anymal.urdf` (vendored)
- Create: `go2_mpc/robot/urdf/anymal/LICENSE` (vendored)
- Create: `go2_mpc/robot/urdf/anymal/README.md` (source + license attribution)
- Modify: `go2_mpc/kinematics/generic/robot_config.py` (add `anymal_config()`)
- Modify: `go2_mpc/kinematics/validate_generic_kinematics.py` (add Stage B call)
- Test: `tests/test_validate_generic_kinematics_anymal.py`

**Interfaces:**
- Consumes: `GenericKinematics` (Task 4/5); `validate_against_mujoco_urdf`, `strip_visual_and_collision` (Task 7).
- Produces: `anymal_config() -> GenericRobotConfig`.

- [ ] **Step 1: Vendor the files**

```bash
mkdir -p go2_mpc/robot/urdf/anymal
curl -sL https://raw.githubusercontent.com/ANYbotics/anymal_c_simple_description/master/urdf/anymal.urdf \
  -o go2_mpc/robot/urdf/anymal/anymal.urdf
curl -sL https://raw.githubusercontent.com/ANYbotics/anymal_c_simple_description/master/LICENSE \
  -o go2_mpc/robot/urdf/anymal/LICENSE
```

Run: `grep -c 'type="revolute"' go2_mpc/robot/urdf/anymal/anymal.urdf`
Expected: `12` (3 revolute joints x 4 legs)

Create `go2_mpc/robot/urdf/anymal/README.md`:

```markdown
# ANYmal C URDF (vendored)

Source: https://github.com/ANYbotics/anymal_c_simple_description/blob/master/urdf/anymal.urdf
License: see LICENSE in this directory, Copyright ANYbotics AG.

Used only for kinematics validation (`go2_mpc/kinematics/validate_generic_kinematics.py`).
This is the first robot GenericKinematics supports with no independent
hand-derived kinematics implementation to cross-check against -- validated
against MuJoCo loading this same URDF directly instead.
```

- [ ] **Step 2: Add the ANYmal robot config**

Append to `go2_mpc/kinematics/generic/robot_config.py`:

```python
def anymal_config() -> "GenericRobotConfig":
    return GenericRobotConfig(
        urdf_path=Path("go2_mpc/robot/urdf/anymal/anymal.urdf"),
        base_link="base",
        foot_links=["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"],  # [FL, FR, RL, RR]
    )
```

- [ ] **Step 3: Add the Stage B call to the validation script**

Add this import alongside the other module-level imports at the top of `go2_mpc/kinematics/validate_generic_kinematics.py` (next to the existing `from go2_mpc.kinematics.generic.robot_config import go2_config`):

```python
from go2_mpc.kinematics.generic.robot_config import anymal_config
```

Then append to the `if __name__ == "__main__":` block:

```python
    anymal_leg_joint_names = [
        ["LF_HAA", "LF_HFE", "LF_KFE"],
        ["RF_HAA", "RF_HFE", "RF_KFE"],
        ["LH_HAA", "LH_HFE", "LH_KFE"],
        ["RH_HAA", "RH_HFE", "RH_KFE"],
    ]
    anymal_cfg = anymal_config()
    validate_against_mujoco_urdf(
        anymal_cfg.urdf_path, anymal_cfg.base_link, anymal_cfg.foot_links,
        anymal_leg_joint_names, label="Stage B (ANYmal)",
    )
```

- [ ] **Step 4: Run the full validation script**

Run: `python go2_mpc/kinematics/validate_generic_kinematics.py`
Expected: `[Stage A.1]`, `[Stage A.2 (Go2)]`, and `[Stage B (ANYmal)]` all print and pass. If Stage B fails while Stage A passes, per the note above, check `kinematic_chain.py`'s handling of *rotating* fixed joints first (add a `print` of `j2.offset_transform` in a REPL against ANYmal's `LF` chain and compare against manual composition of `LF_hip_fixed_LF_HFE`'s `rpy`).

- [ ] **Step 5: Wrap Stage B in a pytest test for CI**

```python
# tests/test_validate_generic_kinematics_anymal.py
from go2_mpc.kinematics.validate_generic_kinematics import (
    validate_against_mujoco_urdf,
)
from go2_mpc.kinematics.generic.robot_config import anymal_config


def test_generic_kinematics_matches_mujoco_from_anymal_urdf():
    cfg = anymal_config()
    leg_joint_names = [
        ["LF_HAA", "LF_HFE", "LF_KFE"],
        ["RF_HAA", "RF_HFE", "RF_KFE"],
        ["LH_HAA", "LH_HFE", "LH_KFE"],
        ["RH_HAA", "RH_HFE", "RH_KFE"],
    ]
    validate_against_mujoco_urdf(
        cfg.urdf_path, cfg.base_link, cfg.foot_links, leg_joint_names,
        n_samples=20, tol=1e-6, label="test",
    )
```

Run: `python -m pytest tests/test_validate_generic_kinematics_anymal.py -v`
Expected: PASS (1 test)

- [ ] **Step 6: Run the entire test suite one final time**

Run: `python -m pytest tests/ -v`
Expected: All tests pass, including the pre-existing `tests/test_controllers.py` (unmodified) and every new test file from Tasks 1-8.

- [ ] **Step 7: Commit**

```bash
git add go2_mpc/robot/urdf/anymal/ go2_mpc/kinematics/generic/robot_config.py \
        go2_mpc/kinematics/validate_generic_kinematics.py \
        tests/test_validate_generic_kinematics_anymal.py
git commit -m "feat: vendor ANYmal URDF and validate GenericKinematics against it (Stage B)"
```

---

## Deliberately Not Done Here (per spec's non-goals)

- No ANYmal MJCF scene, no tuned actuator/contact params, no standing/walking ANYmal in `main.py` — that's a future "wire up ANYmal end-to-end" sub-project.
- No mass matrix / Coriolis terms.
- No support for robots with a different leg-DOF count or leg count than 3x4.
- `go2_kinematics.py` is untouched.
