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
    axis: np.ndarray          # (3,) unit vector, in this joint's own local frame
    offset_transform: np.ndarray  # (4,4) static transform from the previous actuated
                                   # joint's frame (or base) up to this joint's frame,
                                   # BEFORE this joint's own DOF rotation is applied
    rigid_links: list = field(default_factory=list)
    # list[(link_name: str, transform: np.ndarray (4,4))] -- EVERY link
    # rigidly attached to this joint: its own immediate child link (at
    # transform=I, relative to this joint's own POST-rotation frame), plus
    # any further links reached by a chain of `fixed` joints before the
    # next actuated joint (or the foot). Real URDFs commonly put the joint's
    # actual mass on such a fixed-jointed descendant, not on the immediate
    # child -- e.g. ANYmal's LF_HAA (revolute) moves link "LF_HIP" (a
    # near-massless 0.001kg placeholder), while the real 0.74kg mass sits on
    # "LF_hip_fixed", one `fixed` joint further out. A design that only
    # tracked the immediate child link would silently drop that mass from
    # gravity compensation. `rigid_links` fixes this by capturing every
    # link rigidly attached to a given DOF, however many fixed-joint hops
    # away, each with its own transform back to that joint's frame.


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
            pending_T = pending_T @ joint_T

            if joint.joint_type in _FIXED_TYPES:
                # This fixed joint's child link is rigidly attached to the
                # most recently emitted actuated joint (if any -- fixed
                # joints between base and the FIRST actuated joint carry no
                # ChainJoint to attach to, and their mass is correctly the
                # base's concern, not this leg's).
                if chain_joints:
                    chain_joints[-1].rigid_links.append((joint.child, pending_T.copy()))
                continue

            if joint.joint_type not in _ACTUATED_TYPES:
                raise UnsupportedJointTypeError(
                    f"joint '{joint.name}' has type '{joint.joint_type}'; "
                    f"only {_ACTUATED_TYPES + _FIXED_TYPES} are supported"
                )

            chain_joints.append(ChainJoint(
                joint_name=joint.name,
                axis=joint.axis,
                offset_transform=pending_T.copy(),
            ))
            # Reset to accumulate the NEW joint's own rigid attachments,
            # starting with its own immediate child link at transform=I
            # (relative to this joint's own post-rotation frame).
            pending_T = np.eye(4)
            chain_joints[-1].rigid_links.append((joint.child, np.eye(4)))

        # Anything left in pending_T is fixed joints AFTER the last actuated
        # joint -- e.g. a foot/toe frame. Do not drop this silently.
        chains.append(KinematicChain(
            foot_link=foot_link,
            joints=chain_joints,
            foot_offset_transform=pending_T,
        ))

    return chains
