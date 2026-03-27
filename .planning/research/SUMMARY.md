# Project Research Summary

**Project:** QuadrupedMPC — Extension for IsaacLab & Batched GPU MPC
**Domain:** Legged Robot Control / Model Predictive Control
**Researched:** 2026-03-27
**Confidence:** HIGH (verified via official docs, GitHub, arXiv)

---

## Executive Summary

This project extends an existing MuJoCo-based MIT Cheetah-style convex MPC controller for Unitree Go2 with an IsaacLab GPU-accelerated simulation backend and batched GPU MPC solver for RL-augmented training. Research confirms IsaacLab + RSL-RL is the industry-standard stack for quadruped locomotion research, offering 10,000+ parallel environments on a single RTX GPU. The existing controller architecture—with its clean separation between MPC, WBC, and gait scheduling—translates well to IsaacLab via a custom Robot ABC that handles batched tensor operations.

The primary differentiator is **batched GPU MPC**: solving thousands of MPC problems in parallel to enable RL training where MPC serves as "teacher" or reward component. This requires replacing CVXPY/CLARABEL (CPU-only) with GPU-accelerated solvers like ReLU-QP or custom Torch-based QP. The main risks are physics mismatch between simulators causing controller instability and QP solver infeasibility during aggressive commands. Mitigation involves parameter abstraction, constraint validation, and fallback solvers.

---

## Key Findings

### Recommended Stack

Research identifies IsaacLab v2.0+ as the GPU simulation backend and RSL-RL v5.0+ for RL training as the de facto standard for quadruped locomotion research. The current CVXPY/CLARABEL solver is unsuitable for batched MPC and must be replaced.

**Core technologies:**
- **IsaacLab v2.0+** — GPU-parallelized physics simulation (10K+ envs per GPU), native Unitree Go2 USD support, deterministic physics for sim-to-real transfer
- **RSL-RL v5.0+** — RL algorithms integrated with IsaacLab, multi-GPU support, proven on Spot/ANYmal/Unitree platforms
- **ReLU-QP** (or custom Torch batched QP) — GPU-accelerated QP solver, solves 1000+ QPs in parallel, 10-100x faster than CPU solvers
- **Custom Robot ABC** — Existing `core/robot.py` pattern extended for batched tensors, abstracts simulator from controller stack

**NOT recommended:**
- CVXPY/CLARABEL (current): CPU-only, single-solve, too slow for batched MPC
- MuJoCo MJX: Intermediate GPU acceleration, but IsaacLab preferred for full RL integration

### Expected Features

The project already implements core MPC-WBC functionality (centroidal dynamics MPC, Jacobian-transpose WBC, Raibert foot placement, cubic Bezier swing trajectories, phase-based gait scheduler). Research identifies what to add:

**Must have (table stakes):**
- **Contact force estimation** — Detect slip, toe-strike, lift-off; enables robust locomotion without hardware force sensors
- **Disturbance rejection / push recovery** — Handle external forces via reactive gait switching (trot→bound)
- **Fall detection + recovery** — Monitor pitch/roll, trigger stand-up motion to prevent hardware damage

**Should have (competitive):**
- **Batched GPU MPC** — Primary differentiator for RL-augmented training; enables parallel MPC solves across 1000+ environments
- **IsaacLab backend** — Natural fit with batched MPC; provides GPU simulation + RL training pipeline
- **RL-Augmented MPC** — RL predicts MPC parameters (gait timing, foot placement weights) while MPC handles real-time control

**Defer (v2+):**
- Nonlinear MPC — Linear MPC sufficient for speeds <2 m/s; 10x compute cost for marginal benefit
- Terrain adaptation — Out of scope per project goals
- Contact-implicit MPC — Research-level complexity; phase-based gait sufficient
- Parameter auto-tuning — Manual tuning acceptable for research platform

### Architecture Approach

The recommended architecture follows a **hierarchical controller with simulator-agnostic boundaries** pattern, extending the existing design:

```
IsaacLab (GPU Batched) → Robot ABC (Batched) → Controller Stack (Batched Tensors)
                                        ↓
                    GaitScheduler → ConvexMPC → WBC → SwingTraj
```

**Major components:**
1. **Simulation Backend (IsaacLab)** — GPU-parallel physics, handles thousands of envs, provides batched state/action tensors
2. **IsaacLab Managers** — Observation, Reward, Termination, Command managers for RL task specification
3. **Controller Stack** — Simulator-agnostic MPC/WBC/gait, now operates on batched tensors `[B, ...]`
4. **Robot Interface (ABC)** — Abstracts simulator, provides `get_base_pose()`, `get_foot_positions()`, `set_torques()` for batched data

### Critical Pitfalls

Research identifies top risks during implementation:

1. **QP Solver Infeasibility** — Contact schedule conflicts, poor reference trajectories cause solver failure → Robot falls
   - *Prevention:* Constraint redundancy checks, fallback to previous solution, regularization (H + 1e-6*I), clamp friction μ=0.5-0.7

2. **Jacobian Computation Errors** — Manual Jacobian doesn't match simulator convention → Incorrect WBC forces, oscillations
   - *Prevention:* Verify against simulator's built-in Jacobian (`mj_jac`, `compute_anc_jacobian_world`), compare analytical vs numerical

3. **Simulator Physics Mismatch (MuJoCo → IsaacLab)** — Friction models, contact stiffness, actuator models differ → Controller fails in IsaacLab
   - *Prevention:* Expose friction/contact parameters as configurable, match mass/inertia via IsaacLab scales, test contact detection separately

4. **GPU Solver Numerical Instability** — Batched solver produces NaN/Inf for some batch elements → Training instability
   - *Prevention:* Per-problem regularization, NaN checks with CPU fallback, mixed-precision careful accumulation

---

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Replace MuJoCo Kinematics with Manual Computation
**Rationale:** Foundation for all subsequent work; must verify Jacobian/FK implementation before simulator migration
**Delivers:** Manual forward kinematics and analytical Jacobian for 4 legs (FL, FR, BL, BR)
**Addresses:** Contact force estimation, disturbance rejection (basic)
**Avoids:** Jacobian computation errors (Pitfall #2), frame transformation confusion (Pitfall #9)

### Phase 2: IsaacLab Backend Integration
**Rationale:** Required for GPU simulation; establishes new simulation contract before batched solver
**Delivers:** IsaacRobot implementation of Robot ABC, single-env controller validation in IsaacLab
**Uses:** IsaacLab v2.0+, existing controller stack
**Implements:** Robot Interface layer with batched tensor API
**Avoids:** Physics mismatch (Pitfall #3), contact timing (Pitfall #6), state estimation latency (Pitfall #5)

### Phase 3: Batched GPU MPC Solver
**Rationale:** Core differentiator; enables parallel RL training at scale
**Delivers:** GPU-accelerated QP solver (ReLU-QP or custom Torch), batched ConvexMPC operating on `[B, 12]` states
**Uses:** ReLU-QP or custom CUDA kernels, PyTorch batched operations
**Implements:** Batched ConvexMPC component
**Avoids:** GPU solver instability (Pitfall #4), memory layout issues (Pitfall #11)

### Phase 4: RL Training Integration
**Rationale:** Complete the training pipeline; enables MPC-augmented RL research
**Delivers:** Direct Workflow IsaacLab environment, reward shaping, integration with RSL-RL
**Uses:** RSL-RL v5.0+, batched MPC from Phase 3
**Implements:** RL Training Pipeline

### Phase 5: Advanced Features (Optional)
- RL-Augmented MPC (RL predicts MPC parameters)
- Differentiable MPC for end-to-end learning
- Terrain adaptation

### Phase Ordering Rationale

- **Phase 1 before 2:** Manual kinematics must work independently of simulator to isolate Jacobian errors
- **Phase 2 before 3:** IsaacLab integration establishes simulation contract; physics mismatch must be resolved before batched solver development
- **Phase 3 before 4:** Batched MPC is prerequisite for parallel RL training at scale
- Grouped phases: 1-2 isolate simulation/backend issues; 3-4 enable training; 5 extends for advanced research

### Research Flags

Phases likely needing deeper research during planning:
- **Phase 3 (GPU Batched MPC):** ReLU-QP API integration, constraint formulation for batched problems — needs API research
- **Phase 4 (RL Integration):** Reward shaping strategy, curriculum learning design — established patterns but project-specific

Phases with standard patterns (skip research-phase):
- **Phase 1 (Kinematics):** Well-documented FK/Jacobian patterns for quadrupeds
- **Phase 2 (IsaacLab):** IsaacLab documentation comprehensive; direct workflow pattern established

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | IsaacLab/RSL-RL are industry standards with strong documentation; ReLU-QP published in IEEE ICRA 2024 |
| Features | HIGH | Table stakes well-established; differentiators based on recent research (2025-2026 arXiv papers) |
| Architecture | HIGH | IsaacLab documentation covers integration patterns; batched MPC patterns from mpc.pytorch/GATO |
| Pitfalls | HIGH | Derived from real-world deployment issues documented in IsaacLab GitHub, arXiv practitioner guides |

**Overall confidence:** HIGH

### Gaps to Address

- **ReLU-QP constraint handling:** Research did not verify ReLU-QP handles friction cone constraints natively — may need custom constraint formulation
- **IsaacLab + custom MPC integration:** Exact API for per-timestep torque application needs verification during Phase 2
- **Sim-to-real transfer:** Hardware validation path not documented; requires external research during Phase 2 or 4

---

## Sources

### Primary (HIGH confidence)
- IsaacLab GitHub (6,752 stars) — Official documentation, manager-based and direct workflow environments
- RSL-RL GitHub (2,400 stars) — Canonical RL library for IsaacLab
- IsaacLab Paper (Mittal et al., arXiv:2511.04831, 2025) — GPU simulation for robot learning
- ReLU-QP Paper (Bishop et al., IEEE ICRA 2024) — GPU-accelerated batched QP for MPC
- NVIDIA Blog: Spot Quadruped with IsaacLab — Industry validation

### Secondary (MEDIUM confidence)
- mpc.pytorch — Differentiable MPC solver patterns
- GATO (Du et al., arXiv:2510.07625, 2025) — GPU batched trajectory optimization
- rl-mpc-locomotion (GitHub) — RL-MPC integration pattern
- Various arXiv papers on RL-augmented MPC (2025-2026)

### Tertiary (LOW confidence)
- Community extensions (IsaacLab-Quadruped-Tasks) — Less maintained, reference only

---

*Research completed: 2026-03-27*
*Ready for roadmap: yes*
