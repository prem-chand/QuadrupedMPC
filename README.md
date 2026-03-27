# QuadrupedMPC

A modular Model Predictive Control (MPC) framework for quadruped locomotion, implementing MIT Cheetah-style convex MPC for the Unitree Go2 robot.

## Overview

QuadrupedMPC implements centroidal dynamics MPC with whole-body control (WBC) for the Unitree Go2 quadruped robot. The architecture separates physics simulation, state estimation, control logic, and numerical optimization into distinct layers—enabling backend swaps (MuJoCo, IsaacLab), solver experimentation, and future integration with learning-based systems.

### Key Features

- **Centroidal Dynamics MPC**: Convex QP formulation following Di Carlo et al., IROS 2018
- **Whole-Body Control**: Jacobian-transpose WBC for stance/swing leg control
- **Raibert Foot Placement**: Heuristic swing foot targeting
- **Cubic Bezier Swing Trajectories**: Smooth foot motion planning
- **Phase-Based Gait Scheduling**: Support for trot, bound, and pace gaits
- **Simulator-Agnostic Controller**: Zero simulator dependencies in control code
- **GPU Batched MPC**: Optional PyTorch-based batched solver for RL training

---

## Architecture

### Control Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Simulation Backend                          │
│                    (MuJoCo / IsaacLab / PyBullet)                  │
└─────────────────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Robot Interface                             │
│            (MujocoRobot / IsaacRobot — implements Robot ABC)        │
└─────────────────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       State Estimator                               │
│              (Base pose, velocity, foot positions)                │
└─────────────────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     ControllerCore (1 kHz)                          │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────────┐  │
│  │GaitScheduler │→ │TrajectoryGen │→ │     ConvexMPC (33 Hz)   │  │
│  │  (100 Hz)    │  │  (33 Hz)     │  │    (QP Build + Solve)   │  │
│  └──────────────┘  └──────────────┘  └───────────┬─────────────┘  │
│                                                     │               │
│                                                     ▼               │
│                              ┌──────────────────────────────────┐   │
│                              │   WholeBodyController (100 Hz)   │   │
│                              │   (Stance: J^T·F + Grav Comp)    │   │
│                              │   (Swing: Cartesian PD)          │   │
│                              └─────────────────┬──────────────────┘   │
│                                                    │                │
│                                                    ▼                │
│                                         ┌─────────────────────┐      │
│                                         │   Torque Merging    │      │
│                                         │   (clip, combine)   │      │
│                                         └──────────┬──────────┘      │
└─────────────────────────────────────────────────────┼────────────────┘
                                                      │
                                                      ▼
                                        ┌─────────────────────────┐
                                        │   Joint Torques (12,)   │
                                        └─────────────────────────┘
```

### Directory Structure

```
go2_mpc/
├── core/                          # Robot abstraction & state
│   ├── robot.py                   # Abstract Robot interface (ABC)
│   ├── mujoco_robot.py            # MuJoCo implementation
│   ├── isaac_robot.py            # IsaacLab implementation
│   ├── state.py                   # BaseState, JointState, State
│   ├── command.py                 # Command dataclass
│   └── ...
│
├── controller/                    # Control algorithms (simulator-agnostic)
│   ├── controller_manager.py     # ControllerCore orchestration
│   ├── convex_mpc.py             # Centroidal dynamics MPC
│   ├── cvxpy_solver.py          # CLARABEL/CVXPY QP backend
│   ├── gpu_qp_solver.py         # GPU batched QP solver (PyTorch)
│   ├── batched_mpc.py           # Batched MPC for parallel envs
│   ├── batched_wbc.py           # Batched WBC for parallel envs
│   ├── wbc.py                   # Whole-body controller
│   ├── gait_scheduler.py        # Phase-based gait (trot/bound/pace)
│   ├── trajectory_generator.py  # Reference trajectory generation
│   ├── foot_swing_trajectory.py # Cubic Bezier swing trajectories
│   ├── state_estimator.py       # State extraction from Robot interface
│   └── solver.py                # Abstract QPSolver interface
│
├── kinematics/                   # Analytical kinematics (no simulator deps)
│   ├── go2_kinematics.py        # FK, Jacobians, gravity compensation
│   ├── contact_force_estimator.py # GRF estimation from residuals
│   ├── frame_assertions.py      # Debug validation
│   └── validate_kinematics.py   # Verification vs MuJoCo
│
├── config/
│   └── config.py                 # All tunable parameters
│
└── main.py                       # Simulation loop
```

---

## Installation

### Prerequisites

- Python 3.9+
- NumPy
- MuJoCo
- CVXPY
- CLARABEL

```bash
pip install numpy mujoco cvxpy clarabel
```

### Optional (for GPU Batched MPC)

```bash
pip install torch
```

### Running

```bash
python main.py
```

Controls:
- `W/A/S/D`: Forward/turn commands
- `Space`: Stop
- `Q/E`: Increase/decrease speed
- `R/F`: Increase/decrease yaw rate

---

## Core Concepts

### 1. Robot Abstraction (Interface Segregation)

The `Robot` ABC (`go2_mpc/core/robot.py`) defines a simulator-agnostic interface:

```python
class Robot(ABC):
    def step(self): ...
    def get_time(self) -> float: ...
    def set_torques(self, tau: np.ndarray): ...
    def get_base_pose(self) -> tuple[np.ndarray, np.ndarray]: ...
    def get_base_velocity(self) -> tuple[np.ndarray, np.ndarray]: ...
    def get_joint_state(self) -> tuple[np.ndarray, np.ndarray]: ...
    def get_foot_positions_world(self) -> list[np.ndarray]: ...
    def get_foot_jacobian(self, foot_index: int) -> np.ndarray: ...
    def get_foot_velocity(self, foot_index: int) -> np.ndarray: ...
    def get_gravity_compensation(self, leg_index: int) -> np.ndarray: ...
    def get_leg_jacobian(self, foot_index: int) -> np.ndarray: ...
```

All controller code operates on this interface—simulation backends are interchangeable.

### 2. Analytical Kinematics

All forward kinematics, Jacobians, and gravity compensation are computed analytically (`go2_mpc/kinematics/go2_kinematics.py`) with zero simulator dependencies:

- **FK**: Foot positions via frame chain (base → hip → thigh → calf → foot)
- **Jacobians**: Geometric Jacobian via cross products
- **Gravity Compensation**: Link CoM positions × masses × gravity

Validation confirms < 1e-15 error vs MuJoCo built-ins.

### 3. Centroidal Dynamics MPC

The convex MPC (`go2_mpc/controller/convex_mpc.py`) formulates the control problem as:

**State** (12-dim): `[p, θ, v, ω] ∈ ℝ¹²`

**Decision Variables**: Ground reaction forces for 4 legs × 3 axes

**Dynamics** (linearized):
```
ṗ = v
θ̇ ≈ ω
v̇ = (1/m) ΣF + g
ω̇ = I⁻¹ Σ(r × F)
```

**Constraints**:
- Normal force: `0 ≤ Fz ≤ f_max` per leg
- Friction cone: `|Fx|, |Fy| ≤ μ · Fz`

### 4. Gait Scheduling

Phase-based gait patterns (`go2_mpc/controller/gait_scheduler.py`):

| Gait   | Phase Offsets (FL, FR, RL, RR) | Contact Pattern |
|--------|--------------------------------|-----------------|
| Trot   | [0, 0.5, 0.5, 0]              | Diagonal pairs  |
| Bound  | [0, 0, 0.5, 0.5]              | Front/rear pairs|
| Pace   | [0, 0.5, 0, 0.5]              | Lateral pairs   |

### 5. Whole-Body Control

Stance legs use Jacobian-transpose control:
```
τ_stance = J_leg^T · F_GRF + τ_gravity
```

Swing legs use Cartesian PD:
```
τ_swing = J_leg^T · (Kp · (p_target - p_foot) - Kd · v_foot)
```

---

## Configuration

All parameters are centralized in `go2_mpc/config/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `mpc.mass` | 15.2 kg | Robot mass |
| `mpc.inertia` | diag(0.1, 0.1, 0.02) | Body inertia |
| `mpc.horizon` | 10 steps | MPC prediction horizon |
| `mpc.dt` | 0.03s | MPC timestep (33 Hz) |
| `gait.gait_period` | 0.45s | Gait period |
| `gait.stance_ratio` | 0.65 | Stance phase fraction |
| `mpc.f_max` | 180 N | Max force per leg |
| `mpc.mu` | 0.6 | Friction coefficient |
| `controller.torque_limit` | 35 Nm | Joint torque limit |
| `controller.swing_kp` | 400 | Swing position gain |
| `controller.swing_kd` | 10 | Swing velocity gain |

---

## Extension Guide

### Adding a New Simulator Backend

1. Implement the `Robot` ABC in a new file:

```python
# go2_mpc/core/pybullet_robot.py
from .robot import Robot

class PybulletRobot(Robot):
    def __init__(self, robot_id):
        self._robot = robot_id
        self._kin = Go2Kinematics()
    
    def get_base_pose(self) -> tuple[np.ndarray, np.ndarray]:
        # Extract from PyBullet state
        ...
```

2. Swap in `main.py`:

```python
# Before
robot = MujocoRobot(model, data)

# After
robot = PybulletRobot(robot_id)
```

No changes to controller code required.

### Adding a New QP Solver

1. Implement the solver interface:

```python
# go2_mpc/controller/osqp_solver.py
from .solver import QPSolver

class OSQPSolver(QPSolver):
    def solve(self, H, f, A_eq, b_eq, A_ineq, b_ineq):
        # Solve using OSQP
        ...
```

2. Swap in `main.py`:

```python
solver = OSQPSolver()
```

### Using IsaacLab Backend

```python
from go2_mpc.core.isaac_robot import IsaacRobot
import isaaclab.sim as sim

# Create IsaacLab simulation
sim_cfg = sim.SimulationCfg(dt=0.001)
sim = sim.create_simulation_cfg(sim_cfg)

# Load robot (requires IsaacLab asset)
robot_cfg = ...  # Unitree Go2 USD config
robot = isaaclab.utils.WCfgOf(robot_cfg).to(sim)

# Wrap with IsaacRobot
go2 = IsaacRobot(robot, sim)
```

---

## API Reference

### Core Classes

| Class | File | Description |
|-------|------|-------------|
| `Robot` | `core/robot.py` | Abstract interface for robot access |
| `MujocoRobot` | `core/mujoco_robot.py` | MuJoCo implementation |
| `IsaacRobot` | `core/isaac_robot.py` | IsaacLab implementation |
| `Go2Kinematics` | `kinematics/go2_kinematics.py` | Analytical FK/Jacobians |

### Controller Classes

| Class | File | Description |
|-------|------|-------------|
| `ControllerCore` | `controller/controller_manager.py` | Main orchestration |
| `ConvexMPC` | `controller/convex_mpc.py` | Centroidal dynamics MPC |
| `WholeBodyController` | `controller/wbc.py` | Jacobian-transpose WBC |
| `GaitScheduler` | `controller/gait_scheduler.py` | Phase-based gait timing |
| `TrajectoryGenerator` | `controller/trajectory_generator.py` | Reference trajectory |
| `FootSwingTrajectory` | `controller/foot_swing_trajectory.py` | Bezier swing planning |
| `StateEstimator` | `controller/state_estimator.py` | State extraction |

### Solver Classes

| Class | File | Description |
|-------|------|-------------|
| `ClarabelSolver` | `controller/cvxpy_solver.py` | CLARABEL QP backend |
| `CVXPYSolver` | `controller/cvxpy_solver.py` | CVXPY wrapper |
| `BatchedQPSolver` | `controller/gpu_qp_solver.py` | GPU batched ADMM |

---

## Performance

| Component | Frequency | Notes |
|-----------|-----------|-------|
| Simulation step | 1 kHz | MuJoCo timestep |
| Gait scheduling | 100 Hz | Contact schedule update |
| WBC | 100 Hz | Torque computation |
| MPC solve | 33 Hz | QP build + solve |

Typical solve time: ~5-10ms per MPC iteration (CPU).

---

## References

- Di Carlo et al., "Dynamic Locomotion in the MIT Cheetah 3 Through Convex Model-Predictive Control," IROS 2018
- Unitree Go2 Technical Specifications
- IsaacLab Documentation

---

## License

MIT License
