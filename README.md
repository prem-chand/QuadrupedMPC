# QuadrupedMPC - Model Predictive Control for Quadruped Robots

This repository contains a Model Predictive Control (MPC) and Whole Body Control (WBC) implementation for quadruped robots (Unitree Go1/Go2) using MuJoCo.

## Structure

- `main.py`: Main entry point for the simulation
- `go2_mpc/controller/`: Core controller implementation
    - `convex_mpc.py`: Convex MPC using CVXPY
    - `wbc.py`: Whole Body Controller
    - `state_estimator.py`, `gait_scheduler.py`, etc.: Helper modules
- `go2_mpc/robot/`: Robot description files (MJCF)
- `archive/`: Legacy code (ignored by git)

## Installation

1. Create a Python virtual environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

```bash
# On macOS, MuJoCo passive viewer requires running with mjpython
mjpython main.py

# On Linux
python main.py
```

### Controls
- **W/S**: Move Forward/Backward
- **A/D**: Strafe Left/Right
- **Q/E**: Turn Left/Right
- **SPACE**: Stop
- **X**: Exit

## Key Components

- **Convex MPC**: Solves for ground reaction forces to track a reference trajectory.
- **Whole Body Control**: Maps forces to joint torques while respecting constraints.
- **Swing Leg Control**: Computes foot trajectories using Bezier curves and IK.
# QuadrupedMPC
