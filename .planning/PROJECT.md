# Project: QuadrupedMPC

MIT Cheetah-style convex MPC controller for a Unitree Go2 quadruped robot.

## What This Is

A modular, simulator-agnostic controller stack for quadrupedal locomotion. The system implements centroidal dynamics MPC with whole-body control (Jacobian transpose), Raibert heuristic foot placement, cubic Bezier swing trajectories, and phase-based gait scheduling.

## Milestone v1.0 (Completed)

Implemented core MPC-WBC stack:
- ✓ Centroidal dynamics MPC with convex QP
- ✓ Whole-body control via Jacobian transpose
- ✓ Raibert foot placement
- ✓ Cubic Bezier swing trajectories
- ✓ Phase-based gait scheduler
- ✓ Simulator-agnostic Robot ABC
- ✓ MuJoCo + IsaacLab backends
- ✓ Analytical kinematics
- ✓ GPU batched MPC (requires PyTorch)

## Milestone v1.1: MIT Cheetah Parity

### Gap Analysis

| Aspect | MIT Cheetah | QuadrupedMPC | Priority |
|--------|-------------|--------------|----------|
| MPC Frequency | 400 Hz | 33 Hz | **HIGH** |
| WBC Frequency | 500 Hz | 100 Hz | HIGH |
| State Estimation | Linear Kalman Filter | Simple integration | HIGH |
| QP Solver | qpOASES | CLARABEL | MEDIUM |
| Balance Controller | Yes | No | HIGH |
| Terrain Adaptation | 30° slope | No | MEDIUM |
| Stair Climbing | 9cm | No | MEDIUM |

### Key References
- Di Carlo et al., IROS 2018 — Original convex MPC paper
- Bledt et al., IROS 2018 — MIT Cheetah 3 design paper
- [A1-QP-MPC-Controller](https://github.com/ShuoYangRobotics/A1-QP-MPC-Controller) — 812 stars, C++ implementation
- [MIT-Cheetah-Note](https://github.com/Technician13/MIT-Cheetah-Note) — Source code analysis

## Core Value

A working MPC-WBC controller stack for quadruped robots that can be extended to different simulators and used as a foundation for MPC-augmented RL research.

## Architecture

```
go2_mpc/
├── core/              # Robot abstraction (Robot ABC)
├── controller/        # MPC, WBC, gait (simulator-agnostic)
├── kinematics/        # Analytical FK/Jacobians
└── config/           # Parameters
```

## Context

- **Robot**: Unitree Go2 (15.2 kg)
- **Backend**: MuJoCo, IsaacLab
- **Stack**: Python, NumPy, PyTorch (optional)

## Requirements

### v1.1 Active (MIT Cheetah Parity)

- [ ] Increase MPC frequency to 100+ Hz
- [ ] Increase WBC frequency to 500 Hz
- [ ] Implement Linear Kalman Filter for state estimation
- [ ] Add Balance Controller for push recovery
- [ ] Tune parameters for Go2

### Out of Scope

- [Nonlinear MPC] — linear sufficient for current scope
- [Hardware deployment] — simulation only

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| CLARABEL solver | Sufficient for 33 Hz | Working |
| Jacobian transpose WBC | Simpler than full OBC | Working |
| Python implementation | Prototyping | Consider C++ for production |

---

*Last updated: 2026-03-27*
