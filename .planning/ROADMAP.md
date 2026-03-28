# Roadmap: QuadrupedMPC

## Project Overview

MIT Cheetah-style convex MPC controller for Unitree Go2 quadruped robot.

## Milestones

- [v1.0](./milestones/v1.0-ROADMAP.md) — Core MPC-WBC stack
- [v1.1](./milestones/v1.1-ROADMAP.md) — MIT Cheetah parity
- [v1.2](./milestones/v1.2-ROADMAP.md) — Solver ablation (quadprog 20x faster)
- **v1.3** — Terrain adaptation (in progress)

## v1.3: Terrain Adaptation

### Phase 10: Terrain Scene
**Goal:** Create MuJoCo scenes with rough terrain and stairs.
**Requirements:** TERR-01, TERR-02, TERR-03
**Status:** Pending

### Phase 11: Height Estimation
**Goal:** Contact-embedded height and slope estimation.
**Requirements:** HEIGHT-01, HEIGHT-02, HEIGHT-03
**Status:** Pending

### Phase 12: Slope Adaptation
**Goal:** MPC adapts to terrain slope.
**Requirements:** SLOPE-01, SLOPE-02, SLOPE-03
**Status:** Pending

### Phase 13: Stair Climbing
**Goal:** Robot climbs 5cm stairs.
**Requirements:** STAIR-01, STAIR-02, STAIR-03
**Status:** Pending

---

*Updated: 2026-03-27*
