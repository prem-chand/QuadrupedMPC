"""
Pytest configuration and shared fixtures for Go2 MPC tests.
"""

import sys
from pathlib import Path

import pytest
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def mujoco_model():
    """Load MuJoCo model."""
    import mujoco
    import os

    model_path = os.path.join(
        os.path.dirname(__file__), '..', 'go2_mpc', 'robot', 'scene.xml'
    )
    return mujoco.MjModel.from_xml_path(model_path)


@pytest.fixture
def mujoco_data(mujoco_model):
    """Create MuJoCo data and initialize pose."""
    import mujoco

    data = mujoco.MjData(mujoco_model)

    # Initialize standing pose
    data.qpos[2] = 0.28  # Height
    data.qpos[3] = 1.0   # Quaternion w
    data.qpos[7:] = np.array([0.0, 0.9, -1.8] * 4)  # Joint angles

    mujoco.mj_forward(mujoco_model, data)

    return data


@pytest.fixture
def gait_generator():
    """Create default GaitGenerator."""
    from go2_mpc.controller.gait_generator import GaitGenerator
    return GaitGenerator(gait_period=0.5, stance_phase_ratio=0.5,
                        prediction_horizon=10, dt=0.02)


@pytest.fixture
def srb_dynamics():
    """Create default SRBDynamics."""
    from go2_mpc.controller.srb_dynamics import SRBDynamics
    return SRBDynamics(mass=15.2, inertia=np.diag([0.07, 0.26, 0.25]))


@pytest.fixture
def convex_mpc(srb_dynamics):
    """Create default ConvexMPC."""
    from go2_mpc.controller.convex_mpc import ConvexMPC
    Q = np.diag([10, 10, 100, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    R = np.diag([1e-3] * 12)
    return ConvexMPC(srb_dynamics, prediction_horizon=10, dt=0.02,
                    Q=Q, R=R, mu=0.6, f_max=150.0)
