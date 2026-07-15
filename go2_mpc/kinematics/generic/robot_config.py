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
