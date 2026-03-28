# Roadmap: QuadrupedMPC

## Project Overview

MIT Cheetah-style convex MPC controller for Unitree Go2 quadruped robot.

## Milestones

- [v1.0](./milestones/v1.0-ROADMAP.md) — Core MPC-WBC stack
- [v1.1](./milestones/v1.1-ROADMAP.md) — MIT Cheetah parity
- [v1.2](./milestones/v1.2-ROADMAP.md) — Solver ablation (quadprog 20x faster)
- **v1.3** — Terrain adaptation (in progress)

## v1.3: Terrain Adaptation

### Phase 10: Terrain Scene ✓
**Status:** Complete (PR #3)
- rough_terrain.xml, stairs.xml

### Phase 11: Height Estimation ✓
**Status:** Complete (PR #3)
- TerrainEstimator class

### Phase 12: Slope Adaptation
**Status:** Pending
**Requirements:** SLOPE-01, SLOPE-02, SLOPE-03

### Phase 13: Stair Climbing
**Status:** Pending
**Requirements:** STAIR-01, STAIR-02, STAIR-03

---

*Updated: 2026-03-27*
