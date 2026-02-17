from dataclasses import dataclass
import numpy as np
from pathlib import Path


# ==========================================================
# Simulation
# ==========================================================

@dataclass
class SimulationConfig:
    sim_dt: float
    model_path: Path


# ==========================================================
# MPC
# ==========================================================

@dataclass
class MPCConfig:
    mass: float
    inertia: np.ndarray
    horizon: int
    dt: float
    Q: np.ndarray
    R: np.ndarray
    mu: float
    f_max: float


# ==========================================================
# Gait
# ==========================================================

@dataclass
class GaitConfig:
    gait_period: float
    stance_ratio: float


# ==========================================================
# Controller
# ==========================================================

@dataclass
class ControllerConfig:
    mpc_decimation: int
    force_smooth_alpha: float
    default_height: float
    torque_limit: float
    swing_kp: float
    swing_kd: float


# ==========================================================
# Full System Config
# ==========================================================

@dataclass
class SystemConfig:
    simulation: SimulationConfig
    mpc: MPCConfig
    gait: GaitConfig
    controller: ControllerConfig


# ==========================================================
# Default Factory
# ==========================================================

def default_config() -> SystemConfig:
    return SystemConfig(
        simulation=SimulationConfig(
            sim_dt=0.001,
            model_path=Path("go2_mpc/robot/scene.xml"),
        ),
        mpc=MPCConfig(
            mass=15.2,
            inertia=np.diag([0.1, 0.1, 0.02]),
            horizon=10,
            dt=0.03,
            Q=np.diag([1, 5, 100, 3, 10, 0.1, 5, 5, 12, 2, 3, 2]),
            R=np.diag([1e-6] * 12),
            mu=0.6,
            f_max=180.0,
        ),
        gait=GaitConfig(
            gait_period=0.45,
            stance_ratio=0.65,
        ),
        controller=ControllerConfig(
            mpc_decimation=3,
            force_smooth_alpha=0.9,
            default_height=0.32,
            torque_limit=35.0,
            swing_kp=400.0,
            swing_kd=10.0,
        ),
    )
