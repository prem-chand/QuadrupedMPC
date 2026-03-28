# Project: QuadrupedMPC

MIT Cheetah-style convex MPC controller for a Unitree Go2 quadruped robot.

## Current State (v1.3)

Production-quality MPC-WBC controller with terrain adaptation.

### Completed Features
- Centroidal dynamics MPC (100 Hz, quadprog solver)
- Whole-body control via Jacobian transpose (500 Hz)
- Analytical kinematics (no simulator deps)
- Phase-based gait scheduler (trot/bound/pace)
- Simulator-agnostic Robot ABC (MuJoCo, IsaacLab)
- Kalman filter state estimation
- Balance controller for push recovery
- GPU batched MPC (requires PyTorch)
- Optimized QP solver (quadprog — 20x faster)
- **Terrain adaptation (rough terrain, stairs, slope estimation)**

### Architecture
```
go2_mpc/
├── core/           # Robot abstraction (Robot ABC)
├── controller/     # MPC, WBC, gait, balance (simulator-agnostic)
├── kinematics/     # Analytical FK/Jacobians
├── robot/          # MuJoCo scenes (flat, rough_terrain, stairs)
└── config/         # Parameters
```

---

## Key References

- Di Carlo et al., IROS 2018 — Convex MPC
- Bledt et al., IROS 2018 — MIT Cheetah 3 design
- [qpsolvers](https://qpsolvers.github.io/qpsolvers/) — Unified QP solver API

---

*Last updated: 2026-03-27*
