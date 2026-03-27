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
    force_control_decimation: int  # New: force control loop frequency
    motion_control_decimation: int  # New: motion control loop frequency
    force_smooth_alpha: float
    default_height: float
    torque_limit: float
    swing_kp: float
    swing_kd: float
    foot_stance_offsets: np.ndarray


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
            inertia=np.diag([0.18, 0.35, 0.3]),
            horizon=10,
            dt=0.01,  # 100 Hz MPC (was 0.03 = 33 Hz)
            Q=np.diag([5, 5, 50, 20, 20, 10, 8, 8, 15, 15, 5, 3]),  # Tuned for Go2
            R=np.diag([1e-3] * 12),
            mu=0.6,
            f_max=180.0,
        ),
        gait=GaitConfig(
            gait_period=0.45,
            stance_ratio=0.65,
        ),
        controller=ControllerConfig(
            mpc_decimation=10,      # 100 Hz MPC (sim_dt * 10 = 0.01s)
            force_control_decimation=2,  # 500 Hz force control
            motion_control_decimation=2,  # 500 Hz motion control
            force_smooth_alpha=0.1,
            default_height=0.32,
            torque_limit=35.0,
            swing_kp=500.0,  # Tuned: higher for more responsive swing
            swing_kd=15.0,   # Tuned: higher damping
            foot_stance_offsets=np.array([
                [ 0.1934,  0.142, 0.0],  # FL
                [ 0.1934, -0.142, 0.0],  # FR
                [-0.1934,  0.142, 0.0],  # RL
                [-0.1934, -0.142, 0.0],  # RR
            ]),
        ),
    )
