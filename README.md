# QuadrupedMPC

A modular Model Predictive Control (MPC) framework for quadruped locomotion built on MuJoCo.

The project separates physics, estimation, control logic, optimization, and numerical solving into clearly defined layers. The result is a structured and extensible control stack that supports solver swapping, controller experimentation, and future integration with learning-based systems.

---

## Architecture Overview

The control pipeline is layered:

```
MuJoCo Simulator
      ↓
Robot Interface (MujocoRobot)
      ↓
StateEstimator
      ↓
ControllerCore
      ↓
ConvexMPC (QP Builder)
      ↓
QPSolver Backend
      ↓
WholeBodyController
      ↓
Joint Torques
```

Each layer has a single responsibility.

---

## Project Structure

```
go2_mpc/
│
├── core/
│   ├── mujoco_robot.py      # MuJoCo plant abstraction
│   ├── state.py             # Structured robot state
│   └── command.py           # Command definition
│
├── controller/
│   ├── controller_manager.py
│   ├── state_estimator.py
│   ├── convex_mpc.py
│   ├── cvxpy_solver.py
│   ├── wbc.py
│   ├── gait_scheduler.py
│   ├── trajectory_generator.py
│   └── foot_swing_trajectory.py
│
├── config/
│   └── config.py            # Centralized system configuration
│
├── robot/
│   └── scene.xml            # MuJoCo model
│
└── main.py                  # System orchestration
```

---

## Core Concepts

### 1. Structured State Representation

Robot state is represented using typed dataclasses instead of raw vector slicing.
This makes control logic explicit and readable.

### 2. Robot Abstraction

`MujocoRobot` encapsulates all simulator-specific logic.
Controllers operate only on abstracted plant data.

### 3. Explicit Controller State

Controller memory (e.g., gait phase, MPC counters) is stored in a dedicated `ControllerState` object.
No hidden internal state.

### 4. Solver-Agnostic MPC

`ConvexMPC` builds quadratic program (QP) matrices directly and delegates solving to a generic `QPSolver` interface.

This allows swapping numerical backends without modifying control logic.

### 5. Centralized Configuration

All system parameters are defined in `config/config.py`.
`main.py` contains no hardcoded tuning values.

---

## Execution Flow

`main.py` performs:

1. Load configuration
2. Initialize MuJoCo
3. Construct robot interface
4. Construct estimator and controller stack
5. Run simulation loop
6. Compute torques
7. Apply torques

The control logic itself is not implemented in `main.py`.

---

## Extensibility

The architecture supports:

* Replacing the QP solver backend
* Integrating alternative controllers
* Swapping simulation backends
* Adding hardware interfaces
* Integrating learning-based modules

---

## Current Capabilities

* Centroidal dynamics MPC
* Contact scheduling
* Friction cone constraints
* Whole-body torque computation
* Modular solver backend

---

## Design Philosophy

* Separate physics from control.
* Separate control logic from numerical solvers.
* Keep state explicit.
* Keep configuration centralized.
* Keep orchestration minimal.

The system is structured to evolve without entangling responsibilities.

---