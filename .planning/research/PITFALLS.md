# Domain Pitfalls: Quadruped MPC Controllers

**Domain:** Legged robot control / Model Predictive Control
**Researched:** 2026-03-27
**Project Context:** Adding IsaacLab backend + GPU batched MPC solver to existing MuJoCo-based MPC-WBC controller

## Overview

This document catalogs critical, moderate, and minor pitfalls specific to quadruped MPC development, with emphasis on the transition from CPU-based solvers (CVXPY/CLARABEL) to GPU-accelerated batched MPC, and simulator backend migration (MuJoCo → IsaacLab).

---

## Critical Pitfalls

These are mistakes that cause system instability, complete controller failure, or require fundamental architectural rewrites.

### 1. QP Solver Infeasibility and Divergence

**What goes wrong:** The quadratic program solver fails to converge or returns infeasible solutions, causing the robot to fall or exhibit erratic behavior.

**Why it happens:** 
- Contact schedule conflicts (e.g., requesting force from a foot that should be in swing phase)
- Friction cone constraints too restrictive for the commanded velocity
- Poor reference trajectory causing physically impossible desired states
- Numerical conditioning issues with poorly scaled Hessian matrices

**Consequences:** 
- Robot falls immediately
- Torque spikes causing joint damage
- Complete loss of locomotion

**Prevention:**
- Implement constraint redundancy checks before solver call
- Add fallback to previous valid solution with smoothing
- Use regularization (small epsilon on diagonal of Hessian): `H + 1e-6 * I`
- Clamp friction coefficient to conservative values (μ = 0.5–0.7 for indoor surfaces)
- Validate contact schedule consistency: at least 2 legs in stance for stable trot

**Detection:**
- Monitor solver status flags (infeasible, unbounded, max iterations)
- Log residual norms to detect convergence degradation
- Watch for NaN/Inf in force solutions

**Phase mapping:** Phase 2 (IsaacLab Integration) — simulator physics differences can cause previously stable contact schedules to become infeasible

---

### 2. Jacobian Computation Errors

**What goes wrong:** Incorrect leg Jacobian matrices cause wrong force distributions, WBC failures, or instability.

**Why it happens:**
- Manual Jacobian implementation doesn't match simulator's convention
- Rotation convention mismatch (active vs. passive joints)
- Forward kinematics chain errors in URDF/mesh parsing

**Consequences:**
- WBC produces incorrect torques
- Foot position drift over time
- Oscillations that amplify until fall

**Prevention:**
- Verify Jacobian against simulator's built-in computation (use MuJoCo's `mj_jac` or IsaacLab's `compute_anc_jacobian_world`)
- Test with zero-velocity joint positions — Jacobian should produce zero translational velocity
- Implement analytical Jacobian and compare against numerical Jacobian: `J_num = (f(q+δ) - f(q-δ)) / (2δ)`
- Check row ordering matches your force stacking order (FL, FR, BL, BR)

**Detection:**
- Compare computed foot velocities against finite-differenced positions
- Monitor for systematic drift in foot positions over gait cycles

**Phase mapping:** Phase 1 (Replace MuJoCo Kinematics) — manual Jacobian implementation is directly tested here

---

### 3. Simulator Physics Mismatch (MuJoCo → IsaacLab)

**What goes wrong:** Controller works in MuJoCo but fails immediately in IsaacLab due to physics differences.

**Why it happens:**
- Different friction models (MuJoCo uses smooth friction, IsaacLab's PhysX uses different formulation)
- Contact stiffness/damping parameters differ
- Actuator model differences (MuJoCo direct torque vs. IsaacLab's implicit motor models)
- Gravity direction or scale differences

**Consequences:**
- Robot slips in IsaacLab but not MuJoCo
- Contact detection timing mismatch
- Controller parameters (gains, thresholds) need complete retuning

**Prevention:**
- Expose friction coefficient, contact stiffness, and damping as configurable parameters
- Use IsaacLab's `scales` to match Go2 mass/inertia to MuJoCo values
- Test contact detection separately in both simulators before integrating controller
- Implement contact force monitoring to detect simulation-specific behavior

**Detection:**
- Compare base height, joint torques, and contact forces between simulators
- Log foot contact forces and compare timing patterns

**Phase mapping:** Phase 2 (IsaacLab Integration) — primary target of this phase

---

### 4. GPU Solver Numerical Instability

**What goes wrong:** Batched GPU solver produces NaN/Inf or diverges for some batch elements while others solve correctly.

**Why it happens:**
- Heterogeneous problem conditioning across batch elements (some easy, some hard)
- Memory access patterns causing thread divergence
- Floating-point precision issues with large horizon lengths
- Lack of warm-starting in GPU implementation

**Consequences:**
- Some environments fail while others work
- Training instability in RL-MPC setups
- Silent failures that corrupt learning

**Prevention:**
- Implement per-problem regularization tuning
- Add NaN/Inf checks on output and fallback to CPU solver
- Use mixed-precision carefully — accumulate in FP64 for critical computations
- Batch similar-difficulty problems together (sort by velocity magnitude)

**Detection:**
- Monitor solution quality metrics (primal/dual residuals) per batch element
- Log failure counts per batch iteration

**Phase mapping:** Phase 3 (GPU Batched MPC) — direct challenge of this phase

---

## Moderate Pitfalls

These cause degraded performance, require workarounds, or limit controller capabilities.

### 5. State Estimation Latency

**What goes wrong:** State estimator introduces delay between true robot state and controller input, causing predictive mismatch.

**Why it happens:**
- Filtering (EKF, moving average) introduces phase lag
- Callback-based state extraction has asynchronous timing
- IsaacLab's observation buffer has default delays

**Consequences:**
- Degraded tracking at high speeds
- Oscillations that appear as "controller tuning" issues
- MPC horizon becomes less accurate

**Prevention:**
- Timestamp all state measurements and compensate for delay
- Use higher-rate state estimation (IMU integration at 1 kHz)
- Account for observation delay in MPC prediction: `x_ref(t + dt_delay)`

**Phase mapping:** Phase 2 (IsaacLab Integration) — IsaacLab's rigid body physics may have different latency characteristics

---

### 6. Contact Schedule Timing Mismatch

**What goes wrong:** Gait scheduler predicts contact events at different times than simulator detects them.

**Why it happens:**
- Phase computation uses wall-clock time but simulation may step differently
- Contact detection threshold differs between MuJoCo and IsaacLab
- Swing leg lands early/late relative to schedule

**Consequences:**
- WBC applies stance logic to swinging legs (pushes foot into ground)
- Robot trips or lurches unexpectedly

**Prevention:**
- Use simulator's actual contact state rather than schedule for WBC switching
- Add small grace period around phase transitions (10-20 ms)
- Implement contact force threshold monitoring

**Phase mapping:** Phase 2 (IsaacLab Integration) — contact detection differs between simulators

---

### 7. Reference Trajectory Generation Mismatch

**What goes wrong:** MPC reference trajectory doesn't match what the robot can actually achieve, causing persistent tracking errors.

**Why it happens:**
- Assumed velocity/acceleration limits in trajectory don't match robot limits
- No momentum planning — MPC assumes instant velocity changes
- Body orientation not accounted for in 2D projection

**Consequences:**
- MPC solves for forces the robot cannot execute
- Persistent position error during aggressive commands

**Prevention:**
- Clip commanded velocity to achievable range (< 1.5 m/s for Go2)
- Include angular velocity in reference trajectory
- Add acceleration constraints to trajectory generator

**Phase mapping:** Any phase — affects all controller operation

---

### 8. Force Smoothing Causes Phase Lag

**What goes wrong:** Exponential moving average (EMA) on contact forces introduces lag, causing late touchdown detection.

**Why it happens:**
- `force_smooth = alpha * force_raw + (1 - alpha) * force_smooth` with alpha too low
- Default smoothing parameters tuned for one gait don't work for others

**Consequences:**
- Late swing-to-stance transition
- Foot penetrates ground before force is detected
- Gait appears "sluggish"

**Prevention:**
- Use higher alpha for faster response (0.7–0.9) or disable smoothing for contact detection
- Separate smooth force (for WBC) from raw force (for contact detection)
- Tune per gait pattern

**Phase mapping:** Phase 2 — gait pattern changes may require retuning

---

## Minor Pitfalls

These cause minor issues, require debugging effort, or represent missed optimization opportunities.

### 9. Frame Transformation Confusion

**What goes wrong:** Mixing world frame, body frame, and joint frame coordinates causes incorrect force application.

**Why it happens:**
- MPC computes forces in body frame but WBC expects world frame
- Foot positions from robot interface in different frames than expected
- Rotational transformations not applied consistently

**Consequences:**
- Robot drifts sideways unexpectedly
- Asymmetric gaits between left/right legs

**Prevention:**
- Document frame convention explicitly: world = global, body = yaw-rotated, joint = local
- Add frame assertion checks in debug builds
- Visualize forces in RViz/Omniverse to verify direction

**Phase mapping:** Phase 1 — manual kinematics implementation must match frame conventions

---

### 10. Parameter Tuning Doesn't Transfer

**What goes wrong:** Controller parameters (gains, thresholds) optimized in simulation fail on hardware or different simulator.

**Why it happens:**
- Simulated friction, delay, and noise don't match reality
- Actuator model differences (simulation uses ideal torque, hardware has back-EMF, friction)
- Sensor noise characteristics differ

**Consequences:**
- "It works in simulation but not on robot" scenario
- Overly aggressive gains that cause oscillation

**Prevention:**
- Add 20% margin to all gain values
- Test with injected noise to approximate hardware conditions
- Keep hardware-specific parameters in separate config section

**Phase mapping:** Phase 2 — new simulator backend reveals transferability issues

---

### 11. Memory Layout Causes GPU Performance Issues

**What goes wrong:** Batched MPC runs slower than expected on GPU due to memory access patterns.

**Why it happens:**
- Non-contiguous memory for batched problems
- CPU-GPU transfers in inner loop
- Each batch element allocates separate GPU memory

**Consequences:**
- Training too slow to be practical
- Can't achieve desired batch size

**Prevention:**
- Use pinned memory and CUDA streams for async transfers
- Pre-allocate batch buffers and reuse
- Profile with NVIDIA Nsight to identify bottlenecks

**Phase mapping:** Phase 3 (GPU Batched MPC)

---

## Phase-Specific Warning Summary

| Phase | Primary Pitfalls | Mitigation Priority |
|-------|------------------|---------------------|
| Phase 1: Replace MuJoCo Kinematics | Jacobian errors (#2), Frame transformation (#9) | Verify FK/Jacobian against simulator |
| Phase 2: IsaacLab Integration | Physics mismatch (#3), Contact timing (#6), State latency (#5) | Parameter abstraction, contact force monitoring |
| Phase 3: GPU Batched MPC | Solver instability (#4), Memory layout (#11) | Fallback to CPU, batch profiling |

---

## Sources

- IsaacLab GitHub Issues: #2174 (friction/performance), #1898 (contact penetration), #1436 (instability)
- NVIDIA Developer Forum: "Differences between Isaac Sim and MuJoCo Jacobians"
- arXiv:2510.21773 — "Real-Time QP Solvers: Practical Guide Towards Legged Robots"
- arXiv:2502.01329 — "Benchmarking Different QP Formulations and Solvers for Dynamic Quadrupedal Walking"
- MuJoCo to IsaacLab friction model differences documented in IsaacLab discussions
