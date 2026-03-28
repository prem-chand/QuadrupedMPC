# Project: QuadrupedMPC

MIT Cheetah-style convex MPC controller for a Unitree Go2 quadruped robot.

## Current State (v1.2)

Production-quality MPC-WBC controller with optimized QP solver.

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

### Architecture
```
go2_mpc/
├── core/           # Robot abstraction (Robot ABC)
├── controller/     # MPC, WBC, gait, balance (simulator-agnostic)
├── kinematics/     # Analytical FK/Jacobians
└── config/         # Parameters
```

---

## Milestone v1.3: Terrain Adaptation

**Goal:** Walk on rough terrain and stairs (up to 20° slopes, 5cm stairs).

### Requirements
- TERR-01: Rough terrain MuJoCo scene
- TERR-02: Contact-embedded height estimation
- TERR-03: Slope adaptation for MPC
- TERR-04: Stair climbing capability

---

## Key References

- Di Carlo et al., IROS 2018 — Convex MPC
- Bledt et al., IROS 2018 — MIT Cheetah 3 design
- [qpsolvers](https://qpsolvers.github.io/qpsolvers/) — Unified QP solver API

---

*Last updated: 2026-03-27*
