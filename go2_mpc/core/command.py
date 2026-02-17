from dataclasses import dataclass
import numpy as np


@dataclass
class Command:
    v_cmd_global: np.ndarray
    yaw_rate: float
    default_height: float
