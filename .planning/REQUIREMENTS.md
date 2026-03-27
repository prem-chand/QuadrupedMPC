# Requirements: QuadrupedMPC

**Defined:** 2026-03-27
**Core Value:** A working MPC-WBC controller stack for quadruped robots that can be extended to different simulators and used as a foundation for MPC-augmented RL research.

## v1 Requirements

Requirements for extending to IsaacLab and batched GPU MPC.

### Kinematics & Dynamics

- [ ] **KND-01**: Forward kinematics computes all 4 foot positions in body frame from joint angles
- [ ] **KND-02**: Leg Jacobian matrices computed for all 4 legs (3x3 or 6x3 depending on formulation)
- [ ] **KND-03**: Foot velocity computed via Jacobian derivative or numerical differentiation
- [ ] **KND-04**: Gravity compensation torques computed correctly for 12-DOF
- [ ] **KND-05**: Kinematics verified against MuJoCo built-in functions (误差 < 1e-5)

### IsaacLab Backend

- [ ] **ISAAC-01**: IsaacRobot class implements Robot ABC interface
- [ ] **ISAAC-02**: get_base_pose() returns position quaternion and RPY
- [ ] **ISAAC-03**: get_base_velocity() returns linear and angular velocity
- [ ] **ISAAC-04**: get_joint_state() returns joint positions and velocities
- [ ] **ISAAC-05**: get_foot_positions_world() returns foot positions in world frame
- [ ] **ISAAC-06**: get_leg_jacobian() returns analytical Jacobian for each leg
- [ ] **ISAAC-07**: set_torques() applies torques to simulation
- [ ] **ISAAC-08**: Single-step controller validation matches MuJoCo behavior

### Batched MPC

- [ ] **BATCH-01**: Controller accepts batched state input (num_envs, state_dim)
- [ ] **BATCH-02**: WBC computes batched torques (num_envs, 12)
- [ ] **BATCH-03**: GPU-accelerated QP solver handles batched problems
- [ ] **BATCH-04**: Solver handles infeasible problems gracefully (fallback)
- [ ] **BATCH-05**: NaN/Inf detection with CPU fallback

### Robustness

- [ ] **ROB-01**: Contact force estimation from residuals
- [ ] **ROB-02**: QP solver infeasibility handling with fallback
- [ ] **ROB-03**: Frame transformation assertions for debugging

## v2 Requirements

### RL Integration

- **RL-01**: RSL-RL integration with custom environment wrapper
- **RL-02**: Reward shaping for MPC-augmented locomotion
- **RL-03**: Policy training with MPC as teacher

### Advanced Features

- **ADV-01**: Nonlinear MPC for high-speed maneuvers
- **ADV-02**: Terrain adaptation with height map
- **ADV-03**: Contact-implicit MPC

## Out of Scope

| Feature | Reason |
|---------|--------|
| Hardware deployment | Simulation-only for now |
| Force sensor contact estimation | Using gait schedule instead |
| Whole-body MPC | Current centroidal+WBC sufficient |
| Footstep planning | Requires perception stack |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| KND-01 | Phase 1 | Pending |
| KND-02 | Phase 1 | Pending |
| KND-03 | Phase 1 | Pending |
| KND-04 | Phase 1 | Pending |
| KND-05 | Phase 1 | Pending |
| ISAAC-01 | Phase 2 | Pending |
| ISAAC-02 | Phase 2 | Pending |
| ISAAC-03 | Phase 2 | Pending |
| ISAAC-04 | Phase 2 | Pending |
| ISAAC-05 | Phase 2 | Pending |
| ISAAC-06 | Phase 2 | Pending |
| ISAAC-07 | Phase 2 | Pending |
| ISAAC-08 | Phase 2 | Pending |
| BATCH-01 | Phase 3 | Pending |
| BATCH-02 | Phase 3 | Pending |
| BATCH-03 | Phase 3 | Pending |
| BATCH-04 | Phase 3 | Pending |
| BATCH-05 | Phase 3 | Pending |
| ROB-01 | Phase 1 | Pending |
| ROB-02 | Phase 2 | Pending |
| ROB-03 | Phase 1 | Pending |

**Coverage:**
- v1 requirements: 21 total
- Mapped to phases: 21
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-27*
*Last updated: 2026-03-27 after research synthesis*
