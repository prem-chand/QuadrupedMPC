# GEMINI.md - QuadrupedMPC Project Context

## Project Overview
QuadrupedMPC is a modular Model Predictive Control (MPC) framework for quadruped locomotion, specifically targeting the Unitree Go2 robot. It implements centroidal dynamics MPC with Whole-Body Control (WBC), separating physics simulation, state estimation, control logic, and numerical optimization.

### Key Technologies
- **Language**: Python 3.9+
- **Math/Linear Algebra**: NumPy
- **Physics Engine**: MuJoCo (default), IsaacLab (supported)
- **Optimization (QP)**: CVXPY, CLARABEL, PyTorch (for GPU batched MPC)
- **Robot Platform**: Unitree Go2

### Architecture
The project follows a modular, simulator-agnostic architecture:
- `go2_mpc/core/`: Robot abstraction (`Robot` ABC) and state definitions.
- `go2_mpc/controller/`: Control algorithms (MPC, WBC, Gait Scheduler, Trajectory Generator).
- `go2_mpc/kinematics/`: Analytical kinematics (FK, Jacobians, gravity compensation) without simulator dependencies.
- `go2_mpc/config/`: Centralized configuration for all parameters.
- `go2_mpc/utils/`: Utilities like data logging and teleop.

## Building and Running

### Installation
```bash
pip install numpy mujoco cvxpy clarabel
# Optional for GPU Batched MPC:
pip install torch
```

### Running Simulation
```bash
python main.py
```
**Controls**:
- `W/A/S/D`: Forward/turn commands
- `Space`: Stop
- `Q/E`: Increase/decrease speed
- `R/F`: Increase/decrease yaw rate

### Testing
The project includes an interactive test suite for evaluating walking quality:
```bash
python tests/test_suite.py
```
You can also run specific categories or tests via command line arguments (see `--help`).

## Development Conventions

### 1. Simulator-Agnostic Design
All controller code MUST depend only on the `Robot` ABC defined in `go2_mpc/core/robot.py`. This ensures the controller can be used with different backends (MuJoCo, IsaacLab, or hardware) without modification.

### 2. Analytical Kinematics
Avoid using simulator-specific kinematics built-ins. Use the analytical implementations in `go2_mpc/kinematics/go2_kinematics.py` to maintain portability.

### 3. Centralized Configuration
Tunable parameters should be added to `go2_mpc/config/config.py` rather than being hardcoded in controller files.

### 4. Modular Solvers
The framework supports multiple QP solvers. Implement the `QPSolver` interface in `go2_mpc/controller/solver.py` when adding a new optimization backend.

### 5. Type Hinting
Use Python type hints consistently across the codebase to ensure clarity and support for static analysis.

### 6. Testing First
When adding new features or fixing bugs in the controller, verify them using the `tests/test_suite.py` scenarios to ensure no regressions in walking stability or performance.
