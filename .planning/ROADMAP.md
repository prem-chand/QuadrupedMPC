# Roadmap: QuadrupedMPC

## Project Overview

Extension of existing MuJoCo-based MIT Cheetah-style convex MPC controller to support IsaacLab backend and batched GPU MPC for RL-augmented training.

## Phases

- [x] **Phase 1: Manual Kinematics & Dynamics** - Replace MuJoCo FK/Jacobian with verified analytical implementations
- [x] **Phase 2: IsaacLab Backend** - Implement IsaacRobot ABC, validate controller behavior in IsaacLab
- [x] **Phase 3: Batched GPU MPC** - GPU-accelerated QP solver for parallel environments

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

**Plans:** Not created yet (Phase 1 must complete first - requires Go2Kinematics verified)

---

### Phase 2: IsaacLab Backend

**Plans:** 2 plans created
- [x] 2-01-PLAN.md — IsaacRobot core implementation (Robot ABC methods)
- [x] 2-02-PLAN.md — Validation + QP infeasibility handling

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

**Plans:** 1 plan created
- [x] 3-01-PLAN.md — Batched MPC implementation (GPU solver + BatchedConvexMPC + BatchedWBC)

---

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Manual Kinematics | ✓ | Complete | ✓ |
| 2. IsaacLab Backend | 2/2 | Complete | ✓ |
| 3. Batched GPU MPC | 1/1 | Complete | ✓ |

---

## Coverage Notes

- All 21 v1 requirements mapped to phases ✓
- ROB-01 (contact force estimation) assigned to Phase 1 (foundation)
- ROB-02 (QP infeasibility) assigned to Phase 2 (after IsaacLab integration)
- ROB-03 (frame assertions) assigned to Phase 1 (kinematics verification)
- Granularity: standard (3 phases for focused work)

---

## Milestone v1.1: MIT Cheetah Parity

### Phase 4: State Estimation (Kalman Filter)

**Goal:** Implement Linear Kalman Filter for robust state estimation matching MIT Cheetah.

**Requirements:** EST-01, EST-02, EST-03

**Success Criteria:**
1. LKF estimates base position/velocity from IMU + encoders
2. Fused orientation from gyro has < 1° drift over 10s
3. Contact-embedded velocity integration reduces position drift

### Phase 5: Controller Frequency & Tuning

**Goal:** Increase control frequencies and tune parameters.

**Requirements:** CTRL-01, CTRL-02, TUNE-01, TUNE-02, TUNE-03

**Success Criteria:**
1. MPC runs at 100 Hz with < 5ms solve time
2. WBC runs at 500 Hz
3. Stable walking on flat ground with tuned parameters

### Phase 6: Balance Controller

**Goal:** Add push recovery and disturbance rejection.

**Requirements:** BAL-01, BAL-02

**Success Criteria:**
1. Robot recovers from 5N push within 0.5s
2. Reactive gait switching triggers on disturbance

---

*Roadmap created: 2026-03-27*
*Updated: 2026-03-27 with MIT Cheetah parity roadmap*
