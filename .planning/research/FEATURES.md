# Feature Landscape: Classical MPC Controllers for Quadruped Robots (2023-2025)

**Domain:** Quadrupedal Locomotion — Classical Model Predictive Control
**Researched:** March 2026

---

## Current Project Baseline

The project implements MIT Cheetah-style convex MPC with the following components:

| Component | Status | Implementation |
|-----------|--------|----------------|
| Centroidal Dynamics MPC | Working | Convex QP with CLARABEL |
| Jacobian Transpose WBC | Working | Simple J^T F + gravity compensation |
| Raibert Foot Placement | Working | Heuristic, proven approach |
| Bezier Swing Trajectories | Working | Cubic interpolation |
| Phase-based Gait Scheduler | Working | Trot/bound/pace patterns |
| Friction Cone Constraints | Working | μ = 0.6 |
| MuJoCo Backend | Working | Flat ground locomotion |

---

## Table Stakes Features

Features expected in any production quadruped controller. Missing these = fundamentally incomplete.

### 1. Contact Force Estimation

| Attribute | Details |
|-----------|---------|
| **Why Expected** | Gait schedule assumes contact but doesn't measure it. Force estimation needed for robust locomotion on slippery/uneven surfaces. |
| **Complexity** | Medium |
| **Implementation** | Model-based estimation from joint torques, or hardware force sensors |

### 2. Disturbance Rejection / Push Recovery

| Attribute | Details |
|-----------|---------|
| **Why Expected** | Real-world deployment requires handling external forces |
| **Complexity** | Medium |
| **Implementation** | Reactive gait switching (trot→bound), external force estimation |

### 3. Fall Detection and Recovery

| Attribute | Details |
|-----------|---------|
| **Why Expected** | Robot should detect falls and attempt recovery |
| **Complexity** | Low-Medium |
| **Implementation** | Pitch/roll threshold monitoring → stand-up motion primitive |

### 4. Multi-Gait Support

| Attribute | Details |
|-----------|---------|
| **Why Expected** | Different gaits optimal for different speeds/conditions |
| **Complexity** | Low |
| **Current** | Trot/bound/pace implemented; gallop is research-level |

### 5. Sim-to-Real Transfer Validation

| Attribute | Details |
|-----------|---------|
| **Why Expected** | Controller developed in simulation must work on hardware |
| **Complexity** | Medium (requires hardware) |

---

## Recent Differentiators (2023-2025)

Features that set modern controllers apart from basic MIT Cheetah implementations.

### 1. Cascaded Fidelity MPC (Cafe-MPC)

| Attribute | Details |
|-----------|---------|
| **What** | Multiple MPC layers with different fidelity levels |
| **Complexity** | High |
| **Reference** | Cafe-MPC (arXiv:2403.03995, CMU 2024) |
| **Benefit** | Tuning-free whole-body control; reduces manual parameter tuning |

### 2. Nonlinear MPC for Agile Motions

| Attribute | Details |
|-----------|---------|
| **What** | Uses nonlinear dynamics instead of linearized centroidal model |
| **Complexity** | High |
| **Reference** | ETH Zurich perceptive locomotion; IIT DLSLab Quadruped-PyMPC |
| **When Needed** | Speeds >2 m/s, jumping, dynamic maneuvers |

### 3. Robust / Chance-Constrained MPC

| Attribute | Details |
|-----------|---------|
| **What** | Accounts for model uncertainties in constraints |
| **Complexity** | High |
| **Reference** | Chance-Constrained Convex MPC (arXiv:2411.03481, 2024) |
| **Benefit** | More robust to parameter variations, disturbances |

### 4. Dual-MPC Footstep Planning

| Attribute | Details |
|-----------|---------|
| **What** | Separate MPC for footstep planning + force optimization |
| **Complexity** | High |
| **Reference** | Dual-MPC Footstep Planning (arXiv:2511.07921, 2025) |
| **Benefit** | Reactive obstacle avoidance, improved robustness |

### 5. Adaptive MPC with Stability Guarantees

| Attribute | Details |
|-----------|---------|
| **What** | Adaptive parameters with formal stability proofs |
| **Complexity** | Very High |
| **Reference** | Adaptive Non-linear Centroidal MPC (arXiv:2409.01144, 2024) |
| **Benefit** | Handles changing terrain, wear-and-tear |

---

## Anti-Features

Features to deliberately NOT build unless explicitly required.

| Anti-Feature | Why Avoid | Alternative |
|--------------|-----------|-------------|
| Replace linear with nonlinear MPC by default | 10x compute cost; linear sufficient for <2 m/s | Offer nonlinear as optional mode |
| Force sensors (hardware) | Cost, reliability issues | Contact force estimation |
| Generic "general-purpose" controller | Feature creep | Focus on Unitree Go2 + locomotion |
| Built-in perception stack | Application-specific | Keep controller perception-agnostic |

---

## Feature Dependencies

```
Contact Force Estimation
    ├── Gait schedule (existing)
    └── State estimation (existing)

Disturbance Rejection
    ├── State estimation (existing)
    ├── MPC solver (existing)
    └── Reactive gait switching (new)

Cafe-MPC / Cascaded Fidelity
    ├── Multiple MPC layers (new)
    └── Automatic parameter tuning (new)

Nonlinear MPC (acados)
    ├── CasADi/acados solver (new)
    └── Longer solve time budget (new)

Dual-MPC Footstep Planning
    ├── Perception (new)
    └── Separate footstep planning layer (new)
```

---

## MVP Recommendation

### Prioritize (Next Phases)

1. **Contact force estimation** — Low complexity, high robustness gain
2. **Disturbance rejection** — Medium complexity, essential for deployment
3. **Solver upgrade (OSQP/ProxQP)** — Low complexity, immediate performance gain

### Prioritize (Later Phases)

4. **Robust/chance-constrained MPC** — Higher complexity, better hardware robustness
5. **GPU batched MPC (ReLU-QP)** — For RL training integration
6. **Nonlinear MPC exploration** — For high-speed/agile motions

### Defer

| Feature | Reason |
|--------|--------|
| Nonlinear MPC as default | Linear sufficient for current speeds |
| Terrain adaptation | Out of scope |
| Contact-implicit MPC | Research-level complexity |
| Full whole-body MPC | Centroidal + WBC sufficient |

---

## Sources

### Key Papers (2023-2025)

- Cafe-MPC: Cascaded-Fidelity MPC with Tuning-Free WBC (arXiv:2403.03995, 2024)
- Adaptive Non-linear Centroidal MPC (arXiv:2409.01144, 2024)
- Dual-MPC Footstep Planning (arXiv:2511.07921, 2025)
- Chance-Constrained Convex MPC (arXiv:2411.03481, 2024)
- ReLU-QP: GPU-Accelerated QP Solver (Bishop et al., ICRA 2024)

### Open Source Implementations

- ShuoYangRobotics/A1-QP-MPC-Controller (812 stars) — Production MIT Cheetah clone
- iit-DLSLab/Quadruped-PyMPC (440 stars) — Python with acados/JAX
- Unitree MPC implementations (various GitHub repos)

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Table stakes | HIGH | Standard in production systems |
| Recent differentiators | HIGH | Well-documented in 2023-2025 papers |
| Anti-features | HIGH | Established best practices |
| Implementation complexity | MEDIUM | Some variations between labs |
