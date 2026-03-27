# Feature Landscape: Quadruped MPC Controllers

**Domain:** Legged robot locomotion control
**Researched:** 2026-03-27
**Project Context:** MIT Cheetah-style convex MPC for Unitree Go2 (existing implementation has centroidal dynamics MPC, Jacobian transpose WBC, Raibert foot placement, Bezier swing trajectories, phase-based gait scheduler)

---

## Current Baseline

The project already implements the core MPC-WBC stack:

| Component | Status | Notes |
|----------|--------|-------|
| Centroidal dynamics MPC | ✓ Working | Convex QP with CLARABEL solver |
| Jacobian transpose WBC | ✓ Working | Simple but effective |
| Raibert foot placement | ✓ Working | Heuristic, proven approach |
| Bezier swing trajectories | ✓ Working | Cubic interpolation |
| Phase-based gait scheduler | ✓ Working | Trot/bound/pace patterns |
| Friction cone constraints | ✓ Working | μ = 0.6 |
| MuJoCo backend | ✓ Working | Flat ground locomotion |

---

## Table Stakes

Features users expect in any production quadruped controller. Missing these = product feels fundamentally incomplete.

### 1. Contact Force Estimation

| Attribute | Value |
|-----------|-------|
| **Why Expected** | Gait schedule assumes contact but doesn't measure it. Force sensors or estimation needed for robust locomotion on slippery/uneven surfaces. |
| **Complexity** | Medium |
| **Dependencies** | Foot force sensors (optional), state estimation, contact detection logic |

**What it does:** Estimates actual contact forces vs. assumed contact from gait schedule. Enables detecting slip, toe-strike, and lift-off events.

**Current gap:** Project uses fixed gait schedule (assumes contacts), no force feedback.

**Implementation options:**

- **Force sensors:** Direct measurement (high accuracy, requires hardware)
- **Contact estimation:** Learn from residuals, torque sensors, or model-based (No-op, leverages existing model)

**Recommendation:** Start with contact estimation (no hardware changes). Add force sensors only if research requires precise force tracking.

---

### 2. Disturbance Rejection / Push Recovery

| Attribute | Value |
|-----------|-------|
| **Why Expected** | Real-world deployment requires handling external forces (door pushing, walking on moving platforms). |
| **Complexity** | Medium |
| **Dependencies** | State estimation, MPC horizon, reactive gait switching |

**What it does:** Detects external disturbances and adjusts gait/foot placement to maintain balance. Often implements "wander" behavior or紧急 gait changes.

**Current gap:** No explicit disturbance detection or reactive response.

**Implementation options:**

- **Reactive gait switching:** Switch to faster gait (e.g., trot→bound) when disturbance detected
- **MPC horizon extension:** Use longer horizon to plan recovery
- **External force estimation:** Estimate disturbance from momentum change

**Recommendation:** Implement reactive gait switching first (low complexity, high robustness gain).

---

### 3. Fall Detection and Recovery

| Attribute | Value |
|-----------|-------|
| **Why Expected** | When things go wrong, robot should detect fall and attempt recovery instead of lying damaged. |
| **Complexity** | Low-Medium |
| **Dependencies** | State estimation (pitch/roll limits), motion primitives for stand-up |

**What it does:** Monitors orientation, detects when robot is on its side or upside-down, triggers recovery motion (flip or stand-up).

**Current gap:** No fall detection implemented.

**Implementation:** Simple pitch/roll threshold check → trigger pre-recorded stand-up trajectory.

---

### 4. Multi-Gait Support

| Attribute | Value |
|-----------|-------|
| **Why Expected** | Different gaits optimal for different speeds/conditions. |
| **Complexity** | Low |
| **Dependencies** | Gait scheduler (already exists) |

**What it does:** Supports trot (default), bound (fast running), pace (lateral stability), crawl (low speed, high stability), gallop (high speed).

**Current state:** Trot/bound/pace implemented in gait_scheduler.py. Gallop/crawl would extend capabilities.

**Recommendation:** Current gaits sufficient for most use cases. Gallop is research-level (no stable open-loop gallop exists).

---

### 5. Sim-to-Real Transfer Validation

| Attribute | Value |
|-----------|-------|
| **Why Expected** | Controller developed in simulation must work on hardware. |
| **Complexity** | Medium (requires hardware) |
| **Dependencies** | Hardware interface, parameter tuning |

**What it does:** Validation that simulation parameters (mass, inertia, friction) match hardware, and controller handles real-world delays/noise.

**Current gap:** No hardware validation path documented.

**Recommendation:** Document required hardware setup and validation procedure. Critical for any "product" claim.

---

### 6. Parameter Auto-Tuning / Adaptation

| Attribute | Value |
|-----------|-------|
| **Why Expected** | Manual tuning of 20+ MPC/WBC parameters is painful. |
| **Complexity** | High |
| **Dependencies** | System identification, adaptation algorithms |

**What it does:** Automatically tunes MPC weights, WBC gains based on observed performance.

**Current gap:** All parameters hardcoded in config.py.

**Recommendation:** Defer to later phase. Manual tuning acceptable for research platform.

---

## Differentiators

Features that set a controller apart from competitors. Not expected by default, but valued by advanced users.

### 1. Batched GPU MPC for RL Training

| Attribute | Value |
|-----------|-------|
| **Why Expected By** | RL researchers wanting to train policies with MPC in the loop |
| **Complexity** | High |
| **Dependencies** | PyTorch-based MPC, GPU-accelerated simulation (IsaacLab) |

**What it does:** Solves multiple MPC problems in parallel on GPU, enabling batched training where MPC serves as "teacher" or reward component.

**Why differentiate:** Most MPC controllers run at 33 Hz on CPU. Batched GPU MPC runs thousands of instances in parallel for training.

**Implementation approaches:**

| Approach | Speed | Accuracy | Complexity |
|----------|-------|----------|------------|
| PyTorch QP (parametric) | Medium (GPU) | Same as CPU | Medium |
| JAX/cuOpt | Fast (GPU) | Similar | High |
| OSQP (batch) | Fast (CPU) | Similar | Medium |
| Learnable MPC proxy | Fast (GPU) | Approximate | High |

**Relevance to project:** The PROJECT.md explicitly lists "Implement batched GPU solver for parallel environments" as an active requirement.

**Recommendation:** This is the highest-value differentiator for the stated goal of "MPC-augmented RL research."

---

### 2. Nonlinear MPC for Agile Motions

| Attribute | Value |
|-----------|-------|
| **Why Expected By** | Users wanting high-speed running, jumping, dynamic maneuvers |
| **Complexity** | High |
| **Dependencies** | CasADi/ACADOS solver, longer solve time budget |

**What it does:** Uses nonlinear dynamics model instead of linearized centroidal model. Enables capturing leg mass, velocity-dependent effects, better high-speed performance.

**Trade-off:**

| Criterion | Linear MPC (current) | Nonlinear MPC |
|-----------|----------------------|---------------|
| Solve time | ~5ms (33 Hz possible) | ~20-50ms |
| Accuracy at high speed | Degrades | Maintained |
| Implementation complexity | Low | High |
| Parameter sensitivity | High | Lower |

**When nonlinear matters:** Speeds > 2 m/s, jumping, highly dynamic maneuvers.

**Current PROJECT.md stance:** "Nonlinear MPC — linear MPC sufficient for current speed requirements" (out of scope).

**Recommendation:** Keep linear MPC as default. Offer nonlinear as optional high-performance mode if research requires.

---

### 3. RL-Augmented / Learning-Enhanced MPC

| Attribute | Value |
|-----------|-------|
| **Why Expected By** | Researchers combining MPC with reinforcement learning |
| **Complexity** | High |
| **Dependencies** | RL framework, training pipeline, differentiable MPC (optional) |

**What it does:** Two integration patterns:

1. **RL → MPC:** Policy provides reference trajectories or gait parameters, MPC executes
2. **MPC → RL:** MPC provides trajectories/forces as supervision for learning a faster policy
3. **Learnable components:** Neural networks predict MPC parameters (e.g., foot placement, gait timing)

**Recent research (2025-2026):**

- "RL-Augmented MPC for Non-Gaited Locomotion" (arXiv 2603.10878) — RL learns contact schedules, MPC executes
- "Real-Time Gait Adaptation using MPC and RL" (arXiv 2510.20706) — Online gait parameter adaptation
- "Rambo: RL-Augmented Model-Based Whole-Body Control" — Locomanipulation combining learned and model-based

**Relevance to project:** PROJECT.md asks "Can this framework be elevated for truly batched MPC for MPC-augmented RL?"

**Recommendation:** This is the primary differentiator if targeting RL research applications.

---

### 4. Contact-Implicit MPC

| Attribute | Value |
|-----------|-------|
| **Why Expected By** | Advanced users wanting emergent contact patterns |
| **Complexity** | Very High |
| **Dependencies** | Mixed-integer programming or smoothing techniques |

**What it does:** Does not pre-plan contact schedule. Instead, optimizes both forces AND contact timing simultaneously. Enables non-periodic gaits, handling unexpected terrain.

**Current state:** Uses fixed gait schedule (phase-based). Contact-implicit would replace this.

**When it matters:** Unknown terrain, highly dynamic maneuvers where pre-planned gait is suboptimal.

**Implementation complexity:** High. Requires either:

- Mixed-integer quadratic programming (slow)
- Smoothing/relaxation techniques (faster but approximate)

**Recommendation:** Defer. Phase-based gait sufficient for most applications. Contact-implicit is research-level.

---

### 5. Whole-Body MPC (vs. Centroidal Only)

| Attribute | Value |
|-----------|-------|
| **Why Expected By** | Applications requiring precise task-space tracking (manipulation, whole-body coordination) |
| **Complexity** | High |
| **Dependencies** | Full dynamics model, operational space control |

**What it does:** Optimizes over full robot state (joint positions + velocities), not just centroidal dynamics.

**Current state:** Centroidal MPC + WBC separation (current architecture).

**Trade-off:**

| Criterion | Centroidal + WBC | Whole-Body MPC |
|-----------|------------------|----------------|
| Compute cost | Low (separate) | High (coupled) |
| Coordination | WBC handles | MPC handles |
| Flexibility | High (modular) | Lower |

**Recommendation:** Keep current separation. WBC provides sufficient coordination for locomotion.

---

### 6. Terrain Adaptation / Height Map Integration

| Attribute | Value |
|-----------|-------|
| **Why Expected By** | Users deploying on stairs, rough terrain |
| **Complexity** | Medium-High |
| **Dependencies** | Terrain sensing (camera/IMU), terrain mapping, body height planning |

**What it does:** Uses terrain information to:

- Plan foot placement on visible surfaces
- Adjust body height for clearance
- Adapt swing trajectories to terrain

**Current state:** Flat ground only.

**Recent work:**

- "MPC-based Controller with Terrain Insight" (Villarreal et al., ICRA 2020) — Uses terrain geometry
- "Nonlinear MPC for Touch-Down in Complex Terrain" (IEEE 2025)
- "Stable Trajectory Planning Using Terrain Features" (RAL 2026)

**Implementation levels:**

1. **Body height adaptation:** Simple, adjust z Desired based on terrain
2. **Foot placement on terrain:** Medium, requires terrain height at foot target
3. **Full terrain-aware planning:** High, requires mapping + planning

**Recommendation:** Keep flat ground for now. Terrain adaptation is out of scope per PROJECT.md.

---

### 7. Real-Time Footstep Planning

| Attribute | Value |
|-----------|-------|
| **Why Expected By** | Users wanting reactive obstacle avoidance |
| **Complexity** | High |
| **Dependencies** | Perception (camera/LIDAR), planning algorithm |

**What it does:** Online planning of future footstep locations to avoid obstacles while satisfying dynamic constraints.

**Recent work:** "Dual-MPC Footstep Planning" (arXiv 2511.07921) — Uses separate MPC for footstep planning vs. force optimization.

**Recommendation:** Defer. Requires perception stack.

---

## Anti-Features

Features to deliberately NOT build unless explicitly required.

### 1. Force Sensors (Hardware)

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Dedicated foot force sensors | Hardware changes, cost, reliability | Use contact force estimation from model residuals |

**Rationale:** Force sensors add hardware complexity and failure modes. Contact estimation achieves ~80-90% accuracy without hardware.

---

### 2. Full Nonlinear MPC as Default

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Replace linear with nonlinear MPC | 10x compute cost, marginal benefit at low speeds | Offer nonlinear as optional mode |

**Rationale:** Current linear MPC handles speeds up to ~2 m/s adequately. Nonlinear adds complexity without proportional benefit.

---

### 3. Generic "General-Purpose" Controller

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Try to handle any robot, any task | Feature creep, maintenance burden | Focus on Unitree Go2 + locomotion |

**Rationale:** Project scope is quadruped locomotion. Keep modularity but don't over-abstract.

---

### 4. Real-Time Iteration Nonlinear MPC

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Advanced nonlinear MPC variants | Complexity for research platform | Use ACADOS if nonlinear needed later |

**Rationale:** Real-time iteration, multi-stage NMPC are overkill. CasADi with standard SQP sufficient if nonlinear needed.

---

### 5. Built-in Perception Stack

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Integrate camera/LIDAR processing | Scope expansion, sensor-specific code | Keep perception external, accept terrain info via API |

**Rationale:** Perception is application-specific. Keep controller agnostic.

---

## Feature Dependencies

```
Contact Force Estimation
    ├── Gait schedule (existing)
    ├── State estimation (existing)
    └── Optional: Force sensors

Disturbance Rejection
    ├── State estimation (existing)
    ├── MPC solver (existing)
    └── Reactive gait switching (new)

Fall Recovery
    ├── State estimation (existing)
    └── Motion primitives (new)

Batched GPU MPC
    ├── PyTorch-based MPC solver (new)
    ├── GPU-accelerated forward dynamics (new)
    └── IsaacLab integration (new)

RL-Augmented MPC
    ├── Batched MPC (enables training)
    ├── Differentiable components (optional)
    └── Training pipeline (external)

Terrain Adaptation
    ├── Body height planning (new)
    ├── Terrain height API (new)
    └── Foot placement on terrain (new)
```

---

## MVP Recommendation

For a research platform targeting MPC-augmented RL:

### Prioritize (Next Phase)

1. **Contact force estimation** — Low complexity, high robustness
   - Enables slip detection, better contact scheduling
   - No hardware changes required

2. **Disturbance rejection** — Medium complexity, high deployment value
   - Push recovery, walking on moving platforms
   - Reactive gait switching is simple

3. **Fall detection + recovery** — Low complexity, prevents hardware damage
   - Essential for any autonomous deployment

### Prioritize (Later Phase)

4. **Batched GPU MPC** — High complexity, primary differentiator for RL
   - Enables parallel training at scale
   - Requires PyTorch rewrite of MPC core

5. **IsaacLab backend** — Enables GPU simulation + RL training
   - Natural fit with batched MPC goal

### Defer

| Feature | Reason |
|---------|--------|
| Nonlinear MPC | Linear sufficient for current speeds |
| Terrain adaptation | Out of scope per project goals |
| Contact-implicit | Research-level complexity |
| Whole-body MPC | Centroidal + WBC sufficient |
| Footstep planning | Requires perception stack |
| Parameter auto-tuning | Manual tuning acceptable |

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Table stakes features | HIGH | Standard in production quadruped controllers |
| Differentiators | MEDIUM | Research landscape, specific implementations vary |
| RL-MPC integration | HIGH | Active research area, clear patterns emerging |
| Anti-features | HIGH | Well-established what to avoid |

---

## Sources

### Academic Papers

- "Dynamic Locomotion in the MIT Cheetah 3 Through Convex Model-Predictive Control" (Di Carlo et al., 2018) — Foundational convex MPC
- "RL-Augmented MPC for Non-Gaited Legged and Hybrid Locomotion" (Patrizi et al., arXiv 2603.10878, 2026) — RL-MPC integration
- "Real-Time Gait Adaptation for Quadrupeds using MPC and RL" (arXiv 2510.20706, 2025) — Online gait adaptation
- "Dual-MPC Footstep Planning for Robust Quadruped Locomotion" (Ham et al., arXiv 2511.07921, 2025) — Footstep planning
- "MPC-based Controller with Terrain Insight" (Villarreal et al., ICRA 2020) — Terrain adaptation
- "Nonlinear MPC-Based Control Framework for Quadruped Robots: Touch-Down in Complex Terrain" (IEEE 2025)
- "Whole-Body Model-Predictive Control of Legged Robots with MuJoCo" (Zhang et al., arXiv 2503.04613, 2025)
- "Cafe-MPC: Cascaded-Fidelity MPC with Tuning-Free WBC" (arXiv 2403.03995, 2024)
- "Rambo: RL-Augmented Model-Based Whole-Body Control" (Cheng et al., arXiv 2504.06662, 2025)

### Open Source Projects

- iit-DLSLab/Quadruped-PyMPC (GitHub) — Python MPC with acados/JAX options
- dyumanaditya/isaac-quad-loco — IsaacLab + RL + MPC integration
- felipemohr/IsaacLab-Quadruped-Tasks — IsaacLab quadruped tasks

### Industry

- Unitree Go2 documentation
- NVIDIA Isaac Lab documentation (GPU simulation for robot learning)
- Boston Dynamics, ANYbotics, DEEP Robotics quadruped platforms
