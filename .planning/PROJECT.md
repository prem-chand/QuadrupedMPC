# Project: QuadrupedMPC

MIT Cheetah-style convex MPC controller for a Unitree Go2 quadruped robot.

## What This Is

A modular, simulator-agnostic controller stack for quadrupedal locomotion. The system implements centroidal dynamics MPC with whole-body control (Jacobian transpose), Raibert heuristic foot placement, cubic Bezier swing trajectories, and phase-based gait scheduling. Currently tested with MuJoCo backend on flat ground with trot gait.

## Why It Matters

Model Predictive Control (MPC) for legged robots requires real-time optimization at control frequencies. This project demonstrates a working implementation of MIT Cheetah-style convex MPC that can run at 33 Hz on a Unitree Go2, with clean separation between the control algorithms and the simulation backend.

## Core Value

A working MPC-WBC controller stack for quadruped robots that can be extended to different simulators (MuJoCo, IsaacLab, PyBullet) and used as a foundation for MPC-augmented RL research.

## Context

- **Robot**: Unitree Go2 (15.2 kg)
- **Backend**: MuJoCo (current), designed for simulator-agnostic operation
- **Stack**: Python, NumPy, CVXPY, CLARABEL solver
- **User**: Robotics Engineer / RL Researcher

## Architecture

```
main.py                          # Simulation loop & component wiring
go2_mpc/
├── core/
│   ├── state.py                 # BaseState (cached RPY/rotmat), JointState, State
│   ├── command.py               # Command dataclass (v_cmd, yaw_rate, height)
│   ├── robot.py                 # Abstract Robot interface (ABC, full API)
│   └── mujoco_robot.py          # MuJoCo-specific Robot implementation
├── config/
│   └── config.py                # All tunable parameters (dataclass-based)
├── controller/                  # NO mujoco imports - fully simulator-agnostic
│   ├── controller_manager.py    # ControllerCore: orchestrates gait→MPC→WBC→swing
│   ├── convex_mpc.py            # Centroidal dynamics QP formulation
│   ├── solver.py                # Abstract QPSolver + MPCSolver interfaces
│   ├── cvxpy_solver.py          # CVXPY/CLARABEL QP backend (parametric, warm-started)
│   ├── wbc.py                   # Jacobian-transpose WBC (takes Robot interface)
│   ├── gait_scheduler.py        # Phase-based gait (trot/bound/pace)
│   ├── trajectory_generator.py  # Reference trajectory for MPC horizon
│   ├── foot_swing_trajectory.py # Cubic Bezier swing foot trajectory
│   └── state_estimator.py       # State extraction from Robot interface
└── utils/
    └── teleop.py                # Keyboard teleoperation
```

## Key Design Principles

- Controller code has ZERO simulator dependencies — all `controller/` files use only the `Robot` ABC from `core/robot.py`
- Robot interface defines: `get_base_pose()`, `get_base_velocity()`, `get_joint_state()`, `get_foot_positions_world()`, `get_leg_jacobian()`, `get_foot_velocity()`, `get_gravity_compensation()`, `set_torques()`
- Solver hierarchy: `QPSolver` (low-level H/f/A/b) and `MPCSolver` (high-level state/ref/contacts)

## Current Status

- **Working**: Trot walking tested in MuJoCo simulation on flat ground
- **Current task**: Replace MuJoCo dependency for kinematics/dynamics computation with manual implementations
- **Open questions**: Can this framework be elevated for truly batched MPC for MPC-augmented RL? What are its limitations?

## Requirements

### Validated

- ✓ Centroidal dynamics MPC — convex QP formulation with friction cone constraints
- ✓ Whole-body control via Jacobian transpose
- ✓ Raibert heuristic foot placement
- ✓ Cubic Bezier swing trajectories
- ✓ Phase-based gait scheduler (trot/bound/pace)
- ✓ Simulator-agnostic controller stack via Robot ABC
- ✓ MuJoCo backend integration

### Active

- [ ] Replace MuJoCo kinematics/dynamics with manual computation
- [ ] Extend to IsaacLab backend
- [ ] Implement batched GPU solver for parallel environments

### Out of Scope

- [Nonlinear MPC] — linear MPC sufficient for current speed requirements
- [Force sensor contact estimation] — using gait schedule instead
- [Terrain adaptation] — flat ground only for now

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| CVXPY/CLARABEL solver | Warm-started parametric QP, sufficient for 33 Hz | Working |
| Jacobian transpose WBC | Simpler than full operational space control | Working |
| Phase-based gait scheduler | Predictable contact patterns | Working |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-03-27 after initialization (brownfield project)*
