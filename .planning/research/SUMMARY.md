# Research Summary: Classical MPC Controllers for Quadruped Robots (2023-2025)

**Domain:** Robotics Control — Quadrupedal Locomotion (Classical MPC)
**Researched:** March 2026
**Overall confidence:** HIGH

## Executive Summary

The field of classical MPC-based quadruped control has matured significantly beyond the foundational MIT Cheetah work (2018). Research from 2023-2025 reveals three major trends: (1) **GPU-accelerated batched MPC** enabling parallel solves for sim-to-real pipelines, (2) **Nonlinear MPC** becoming computationally tractable with improved solvers (acados, ProxQP), and (3) **Cascaded fidelity approaches** (Cafe-MPC) reducing tuning burden. The dominant paradigm remains centroidal dynamics MPC + whole-body control, but solver choices have diversified from CVXPY/CLARABEL to include OSQP, ProxQP, qpOASES, and GPU-native options like ReLU-QP.

Key lab contributions: ETH Zurich's perceptive NMPC for ANYmal, IIT DLSLab's Python-based Quadruped-PyMPC with acados/JAX, CMU's Cafe-MPC for tuning-free control, and multiple production-grade qpOASES/OSQP implementations for Unitree robots.

## Key Findings

**Stack:** Convex MPC with OSQP/ProxQP remains production standard; acados for nonlinear MPC; ReLU-QP/GATO for GPU batched scenarios

**Architecture:** Two-layer (MPC → WBC) or cascaded multi-fidelity (Cafe-MPC) remain dominant patterns

**Critical pitfall:** Single-solve CPU MPC cannot scale to RL training pipelines; requires GPU-batched solver

## Implications for Roadmap

Based on research, suggested phase structure:

### Phase 1: Baseline Stabilization
Verify current CVXPY/CLARABEL implementation works reliably
- Addresses: Working flat-ground trot
- Avoids: Premature optimization

### Phase 2: Solver Upgrade
Replace CVXPY with OSQP or ProxQP for real-time performance
- Addresses: Single-solve latency (~10ms target)
- Avoids: CVXPY overhead

### Phase 3: Robustness Enhancement
Add chance-constrained or robust MPC variants
- Addresses: Model uncertainties, disturbances
- Avoids: Brittle behavior on real hardware

### Phase 4: GPU Batched MPC
Integrate ReLU-QP or custom Torch batched solver for RL training
- Addresses: Scalability for MPC-augmented RL
- Avoids: CPU bottleneck in parallel training

### Phase 5: Nonlinear MPC Exploration
Evaluate acados for terrain/agile motions
- Addresses: High-speed locomotion, rough terrain
- Avoids: Linearization errors

**Phase ordering rationale:**
- Solver upgrade before GPU batching (dependency: need fast single-solve first)
- Robustness before nonlinear (build on stable baseline)
- RL integration last (requires mature controller)

**Research flags for phases:**
- Phase 3 (Robustness): May need deeper research on uncertainty quantification methods
- Phase 5 (Nonlinear MPC): acados integration complexity; may need feasibility study

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | Multiple verified sources, active OSS repos |
| Features | HIGH | Well-documented in papers and implementations |
| Architecture | HIGH | Consensus on two-layer/cascaded patterns |
| Pitfalls | MEDIUM | Some emerging trends not fully validated |

## Gaps to Address

- Real-world deployment data from labs using nonlinear MPC (limited publications)
- Direct comparisons between OSQP vs ProxQP for quadruped-specific QPs
- Integration complexity of ReLU-QP for production systems

---

## Sources

### Primary (HIGH confidence)
- Cafe-MPC (arXiv:2403.03995, 2024) — Cascaded-fidelity MPC
- ReLU-QP Paper (Bishop et al., IEEE ICRA 2024) — GPU-accelerated batched QP
- iit-DLSLab/Quadruped-PyMPC (GitHub, 440 stars) — Python MPC with acados/JAX
- ShuoYangRobotics/A1-QP-MPC-Controller (GitHub, 812 stars) — Production MIT Cheetah clone
- ETH Zurich perceptive locomotion papers (2022-2024)

### Secondary (MEDIUM confidence)
- GATO (Du et al., arXiv:2510.07625, 2025) — GPU batched trajectory optimization
- Dual-MPC Footstep Planning (arXiv:2511.07921, 2025)
- Adaptive Non-linear Centroidal MPC (arXiv:2409.01144, 2024)
- Various Unitree MPC implementations on GitHub

### Tertiary (LOW confidence)
- Community implementations, less maintained repositories
