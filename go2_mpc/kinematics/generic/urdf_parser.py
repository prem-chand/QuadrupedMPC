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
