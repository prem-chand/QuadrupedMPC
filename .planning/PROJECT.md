# Project: QuadrupedMPC

MIT Cheetah-style convex MPC controller for a Unitree Go2 quadruped robot.

## Current State (v1.1)

A working MPC-WBC controller stack implementing MIT Cheetah-style convex MPC for quadrupedal locomotion.

### Completed Features
- Centroidal dynamics MPC (100 Hz)
- Whole-body control via Jacobian transpose (500 Hz)
- Analytical kinematics (no simulator deps)
- Raibert foot placement + Bezier swing trajectories
- Phase-based gait scheduler (trot/bound/pace)
- Simulator-agnostic Robot ABC (MuJoCo, IsaacLab)
- Kalman filter state estimation
- Balance controller for push recovery
- GPU batched MPC (requires PyTorch)

### Architecture
```
go2_mpc/
├── core/           # Robot abstraction (Robot ABC)
├── controller/     # MPC, WBC, gait, balance (simulator-agnostic)
├── kinematics/     # Analytical FK/Jacobians
└── config/         # Parameters
```

---

## v1.1 Milestone Summary

**Completed:** 2026-03-27

| Phase | Feature |
|-------|---------|
| 4 | State Estimation (Kalman filter) |
| 5 | Controller Frequency (100/500 Hz) + Tuning |
| 6 | Balance Controller (push recovery) |

---

## Next Goals

1. **Test performance** at increased frequencies (100/500 Hz)
2. **C++ porting** if Python is too slow (swap to qpOASES)
3. **Terrain adaptation** (stretch goal)
4. **Hardware deployment** (stretch goal)

---

## Key References

- Di Carlo et al., IROS 2018 — Convex MPC
- Bledt et al., IROS 2018 — MIT Cheetah 3 design
- [A1-QP-MPC-Controller](https://github.com/ShuoYangRobotics/A1-QP-MPC-Controller) — Reference implementation

---

*Last updated: 2026-03-27*
