# Project: QuadrupedMPC

MIT Cheetah-style convex MPC controller for a Unitree Go2 quadruped robot.

## Current State (v1.2)

A production-quality MPC-WBC controller stack with optimized QP solver.

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
- **Optimized QP solver (quadprog — 20x faster than CLARABEL)**

### Architecture
```
go2_mpc/
├── core/           # Robot abstraction (Robot ABC)
├── controller/     # MPC, WBC, gait, balance (simulator-agnostic)
├── kinematics/     # Analytical FK/Jacobians
└── config/         # Parameters
```

---

## v1.2 Milestone Summary

**Completed:** 2026-03-27

| Phase | Feature |
|-------|---------|
| 7 | Unified QP solver backend |
| 8 | Benchmark suite |
| 9 | Solver selection (quadprog default) |

---

## Next Goals

1. **Terrain adaptation** — Rough terrain + stairs
2. **Nonlinear MPC** — acados for agile motions
3. **Hardware deployment** — C++ port if needed

---

## Key References

- Di Carlo et al., IROS 2018 — Convex MPC
- Bledt et al., IROS 2018 — MIT Cheetah 3 design
- [qpsolvers](https://qpsolvers.github.io/qpsolvers/) — Unified QP solver API

---

*Last updated: 2026-03-27*
