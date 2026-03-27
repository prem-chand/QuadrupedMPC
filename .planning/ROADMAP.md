# Roadmap: QuadrupedMPC

## Project Overview

Extension of existing MuJoCo-based MIT Cheetah-style convex MPC controller to support IsaacLab backend and batched GPU MPC for RL-augmented training.

## Phases

- [ ] **Phase 1: Manual Kinematics & Dynamics** - Replace MuJoCo FK/Jacobian with verified analytical implementations
- [ ] **Phase 2: IsaacLab Backend** - Implement IsaacRobot ABC, validate controller behavior in IsaacLab
- [ ] **Phase 3: Batched GPU MPC** - GPU-accelerated QP solver for parallel environments

---

## Phase Details

### Phase 1: Manual Kinematics & Dynamics

**Goal:** Replace MuJoCo-dependent kinematics/dynamics with verified analytical implementations, enabling simulator-agnostic controller operation.

**Depends on:** Nothing (first phase)

**Requirements:** KND-01, KND-02, KND-03, KND-04, KND-05, ROB-01, ROB-03

**Success Criteria** (what must be TRUE):
  1. Forward kinematics computes all 4 foot positions in body frame from joint angles (FL, FR, BL, BR legs)
  2. Analytical Jacobian matrices for all 4 legs match MuJoCo built-in functions with error < 1e-5
  3. Foot velocity computation via Jacobian derivative produces accurate results
  4. Gravity compensation torques computed correctly for all 12 DOF
  5. Contact force estimation from residuals enables robust locomotion without force sensors
  6. Frame transformation assertions validate all coordinate conversions

**Plans:** TBD

---

### Phase 2: IsaacLab Backend

**Goal:** Implement IsaacLab backend via IsaacRobot ABC, enabling GPU-accelerated simulation and RL training pipeline.

**Depends on:** Phase 1

**Requirements:** ISAAC-01, ISAAC-02, ISAAC-03, ISAAC-04, ISAAC-05, ISAAC-06, ISAAC-07, ISAAC-08, ROB-02

**Success Criteria** (what must be TRUE):
  1. IsaacRobot class fully implements Robot ABC interface
  2. get_base_pose() returns position quaternion and RPY correctly
  3. get_base_velocity() returns linear and angular velocity in world frame
  4. get_joint_state() returns joint positions and velocities for all 12 DOF
  5. get_foot_positions_world() returns foot positions in world frame for all 4 legs
  6. get_leg_jacobian() returns analytical Jacobian for each leg
  7. set_torques() applies torques to IsaacLab simulation correctly
  8. Single-step controller validation matches MuJoCo behavior (identical torques within tolerance)
  9. QP solver infeasibility handled gracefully with fallback

**Plans:** TBD

---

### Phase 3: Batched GPU MPC

**Goal:** Implement GPU-accelerated batched MPC solver for parallel environment training, enabling RL-augmented locomotion research.

**Depends on:** Phase 2

**Requirements:** BATCH-01, BATCH-02, BATCH-03, BATCH-04, BATCH-05

**Success Criteria** (what must be TRUE):
  1. Controller accepts batched state input with shape (num_envs, 12) or (num_envs, state_dim)
  2. WBC computes batched torques with shape (num_envs, 12)
  3. GPU-accelerated QP solver solves batched problems (1000+ parallel solves)
  4. Solver handles infeasible batch elements gracefully with fallback to previous solution
  5. NaN/Inf detection triggers CPU fallback for problematic batch elements without crashing

**Plans:** TBD

---

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Manual Kinematics | 0/6 | Not started | - |
| 2. IsaacLab Backend | 0/9 | Not started | - |
| 3. Batched GPU MPC | 0/5 | Not started | - |

---

## Coverage Notes

- All 21 v1 requirements mapped to phases ✓
- ROB-01 (contact force estimation) assigned to Phase 1 (foundation)
- ROB-02 (QP infeasibility) assigned to Phase 2 (after IsaacLab integration)
- ROB-03 (frame assertions) assigned to Phase 1 (kinematics verification)
- Granularity: standard (3 phases for focused work)

---

*Roadmap created: 2026-03-27*
