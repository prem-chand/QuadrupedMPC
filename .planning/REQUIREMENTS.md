# Requirements: QuadrupedMPC v1.3

**Defined:** 2026-03-27
**Core Value:** Enable quadruped locomotion on rough terrain and stairs.

## v1.3 Requirements

### Terrain Scene (Phase 10)

- [ ] **TERR-01**: Create MuJoCo scene with rough terrain (random bumps, height variation)
- [ ] **TERR-02**: Create MuJoCo scene with stairs (5cm rise, 25cm depth)
- [ ] **TERR-03**: Terrain provides realistic friction (mu >= 0.6)

### Height Estimation (Phase 11)

- [ ] **HEIGHT-01**: Contact-embedded height estimation per foot
- [ ] **HEIGHT-02**: Terrain slope estimation from stance legs
- [ ] **HEIGHT-03**: Normal vector estimation for each contact surface

### Slope Adaptation (Phase 12)

- [ ] **SLOPE-01**: MPC state includes terrain slope information
- [ ] **SLOPE-02**: Gait adapts to slope (stair gait vs flat gait)
- [ ] **SLOPE-03**: Robot walks on 20° slope without falling

### Stair Climbing (Phase 13)

- [ ] **STAIR-01**: Contact detection for stair edges
- [ ] **STAIR-02**: Foot placement on stair surface
- [ ] **STAIR-03**: Robot climbs 5cm stairs

## Out of Scope

| Feature | Reason |
|---------|--------|
| Vision-based terrain sensing | Contact estimation sufficient for MPC |
| Dynamic stair running | Walk only for safety |
| Steep slopes (> 30°) | Limited by friction |

---

*Requirements defined: 2026-03-27*
