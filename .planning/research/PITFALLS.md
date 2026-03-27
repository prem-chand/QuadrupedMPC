# Domain Pitfalls: Classical MPC Controllers for Quadruped Robots (2023-2025)

**Domain:** Legged Robot Control — Model Predictive Control
**Researched:** March 2026

---

## Overview

This document catalogs critical, moderate, and minor pitfalls specific to classical quadruped MPC development. These are derived from practitioner experience, published post-mortems, and common implementation issues documented in the literature.

---

## Critical Pitfalls

These mistakes cause system instability, complete controller failure, or require fundamental rewrites.

### 1. QP Solver Infeasibility

**What goes wrong:** QP solver fails to converge or returns infeasible solutions, causing immediate robot fall.

**Why it happens:**
- Contact schedule conflicts (force requested from swing leg)
- Friction cone constraints too restrictive (μ too high)
- Poor reference trajectory (physically impossible desired states)
- Poorly conditioned Hessian matrices (numerical issues)

**Consequences:**
- Robot falls immediately
- Torque spikes → joint damage
- Complete locomotion failure

**Prevention:**
- Regularization: `H + 1e-6 * I` on Hessian diagonal
- Conservative friction coefficient: μ = 0.5–0.7 (not higher)
- Fallback to previous valid solution with EMA smoothing
- Validate contact schedule: at least 2 legs in stance for trot
- Constraint redundancy checks before solver call

**Detection:**
- Monitor solver status flags (infeasible, unbounded, max iterations)
- Log residual norms to detect convergence degradation
- Watch for NaN/Inf in force solutions

**Phase mapping:** Any phase with QP solver; especially critical when changing solvers

---

### 2. Jacobian Computation Errors

**What goes wrong:** Incorrect leg Jacobian matrices cause wrong force distributions and instability.

**Why it happens:**
- Manual Jacobian doesn't match simulator/robot convention
- URDF joint axis misalignment
- Forward kinematics chain errors
- Row ordering mismatch (FL, FR, BL, BR)

**Consequences:**
- WBC produces incorrect torques
- Foot position drift over time
- Oscillations that amplify until fall

**Prevention:**
- Verify against simulator's built-in Jacobian (MuJoCo: `mj_jac`, IsaacLab: `compute_anc_jacobian_world`)
- Test with zero-velocity joints → Jacobian should give zero velocity
- Compare analytical vs numerical Jacobian: `J_num = (f(q+δ) - f(q-δ)) / (2δ)`
- Explicitly document leg ordering and match to force vector

**Detection:**
- Compare computed vs finite-differenced foot velocities
- Monitor systematic foot position drift over gait cycles

---

### 3. Frame Transformation Confusion

**What goes wrong:** Mixing world frame, body frame, and joint frame coordinates causes incorrect force application.

**Why it happens:**
- MPC computes forces in body frame but WBC expects world frame
- Foot positions from robot interface in different frames
- Rotational transformations not applied consistently

**Consequences:**
- Robot drifts sideways unexpectedly
- Asymmetric gaits between left/right legs
- Oscillations at turn initiation

**Prevention:**
- Document frame convention explicitly:
  - World = global inertial frame
  - Body = yaw-rotated (Rᵀ × world)
  - Joint = local leg coordinates
- Add frame assertion checks in debug builds
- Visualize forces in simulator to verify direction

---

### 4. Contact Schedule Timing Mismatch

**What goes wrong:** Gait scheduler predicts contact events at different times than physics simulation.

**Why it happens:**
- Phase computation uses wall-clock time but simulation may step differently
- Contact detection threshold differs between simulators
- Swing leg lands early/late relative to schedule

**Consequences:**
- WBC applies stance logic to swinging legs (pushes foot into ground)
- Robot trips or lurches unexpectedly

**Prevention:**
- Use actual contact state from simulator rather than schedule for WBC switching
- Add grace period around phase transitions (10-20 ms)
- Implement contact force threshold monitoring
- Separate contact detection from contact scheduling

---

## Moderate Pitfalls

These cause degraded performance, require workarounds, or limit capabilities.

### 5. State Estimation Latency

**What goes wrong:** State estimator introduces delay between true robot state and controller input.

**Why it happens:**
- Filtering (EKF, moving average) introduces phase lag
- Callback-based state extraction has asynchronous timing
- Default observation buffers have delays

**Consequences:**
- Degraded tracking at high speeds
- Oscillations appearing as "tuning issues"
- MPC horizon becomes less accurate

**Prevention:**
- Timestamp all state measurements and compensate for delay
- Use higher-rate state estimation (IMU integration at 1 kHz)
- Account for observation delay in MPC prediction: `x_ref(t + dt_delay)`

---

### 6. Reference Trajectory Mismatch

**What goes wrong:** MPC reference trajectory doesn't match achievable robot behavior.

**Why it happens:**
- Assumed velocity/acceleration limits don't match robot limits
- No momentum planning — MPC assumes instant velocity changes
- Body orientation not accounted for in 2D projection

**Consequences:**
- MPC solves for forces robot cannot execute
- Persistent position error during aggressive commands

**Prevention:**
- Clip commanded velocity to achievable range (<1.5 m/s for Go2)
- Include angular velocity in reference trajectory
- Add acceleration constraints to trajectory generator

---

### 7. Solver Performance Regression

**What goes wrong:** Solver becomes slower over time or with different parameters.

**Why it happens:**
- Matrix conditioning changes with robot configuration
- Different contact schedules produce different QP structures
- Memory allocation overhead in Python solvers

**Consequences:**
- Control loop misses real-time deadline
- Reduced effectiveness (lower MPC rate)

**Prevention:**
- Profile solver across diverse configurations
- Use warm-starting from previous solution
- Set maximum iteration limits to bound worst-case time

---

### 8. Force Smoothing Causes Phase Lag

**What goes wrong:** EMA on contact forces introduces lag, causing late touchdown detection.

**Why it happens:**
- `force_smooth = α * force_raw + (1-α) * force_smooth` with α too low
- Default smoothing parameters tuned for one gait don't work for others

**Consequences:**
- Late swing-to-stance transition
- Foot penetrates ground before force detected
- Gait appears "sluggish"

**Prevention:**
- Use higher α for faster response (0.7–0.9)
- Separate smooth force (for WBC) from raw force (for contact detection)
- Tune per gait pattern

---

## Minor Pitfalls

These cause minor issues, require debugging effort, or represent missed opportunities.

### 9. Parameter Tuning Doesn't Transfer

**What goes wrong:** Controller parameters optimized in simulation fail on hardware.

**Why it happens:**
- Simulated friction, delay, and noise don't match reality
- Actuator model differences (ideal torque vs hardware with back-EMF, friction)
- Sensor noise characteristics differ

**Consequences:**
- "Works in simulation but not on robot"
- Overly aggressive gains causing oscillation

**Prevention:**
- Add 20% margin to all gain values
- Test with injected noise to approximate hardware
- Keep hardware-specific parameters in separate config

---

### 10. Gait Transition Instability

**What goes wrong:** Switching between gaits (trot→bound) causes transients.

**Why it happens:**
- Contact schedule changes discontinuously
- Foot positions from previous gait persist
- MPC horizon has different requirements

**Prevention:**
- Smooth contact schedule transition
- Reset swing leg targets on gait change
- Use consistent foot placement logic across gaits

---

### 11. Memory Allocation in Control Loop

**What goes wrong:** Allocating new arrays/matrices each control cycle causes GC overhead.

**Why it happens:**
- Creating numpy arrays in Python loop
- Not pre-allocating solver workspaces

**Consequences:**
- Variable latency (non-deterministic timing)
- CPU spikes causing missed control deadlines

**Prevention:**
- Pre-allocate all matrices and vectors
- Use in-place operations where possible
- Reuse solver objects across timesteps

---

## Phase-Specific Warning Summary

| Phase | Primary Pitfalls | Mitigation Priority |
|-------|------------------|---------------------|
| Solver Upgrade (OSQP/ProxQP) | Infeasibility (#1), Warm-start issues | Test across diverse conditions |
| Kinematics Implementation | Jacobian errors (#2), Frame confusion (#3) | Verify against simulator |
| Multi-Gait Support | Contact timing (#4), Transition instability (#10) | Test gait switching extensively |
| Hardware Deployment | Tuning transfer (#9), Latency (#5) | Add margins, test with noise |

---

## Sources

### Practitioner Guides

- Real-Time QP Solvers: Practical Guide Towards Legged Robots (arXiv:2510.21773, 2025)
- Benchmarking QP Formulations and Solvers for Dynamic Quadrupedal Walking (ICRA 2024)
- Various GitHub issues and discussions from quadruped MPC repositories

### Academic Sources

- MIT Cheetah MPC papers (Di Carlo et al., 2018)
- Cafe-MPC (arXiv:2403.03995, 2024)
- ETH locomotion papers (2022-2024)

### Community Knowledge

- ShuoYangRobotics/A1-QP-MPC-Controller issues and discussions
- Unitree community forums
- ROS/walking_robotics discussions

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| QP solver pitfalls | HIGH | Well-documented in practitioner guides |
| Jacobian/frame issues | HIGH | Common implementation errors |
| Performance issues | MEDIUM | Hardware-dependent |
| Transfer to hardware | MEDIUM | Limited publications |
