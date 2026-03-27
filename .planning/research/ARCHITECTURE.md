# Architecture Patterns: Classical MPC Controllers for Quadruped Robots (2023-2025)

**Domain:** Quadrupedal Locomotion Control — MPC-WBC Architecture
**Researched:** March 2026
**Confidence:** HIGH

---

## Dominant Architecture Patterns

Research from 2023-2025 confirms two primary architectural patterns for classical quadruped MPC:

### Pattern 1: Two-Layer MPC + WBC (Most Common)

This is the MIT Cheetah architecture, still dominant in production systems:

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONTROLLER LOOP (30-100 Hz)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐                                               │
│  │   Gait       │  Phase-based contact scheduling              │
│  │ Scheduler    │  → Binary contact schedule [4 legs × H]      │
│  └──────┬───────┘                                              │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
│  │ Trajectory   │     │   Convex     │     │  Reference   │   │
│  │ Generator    │────▶│     MPC      │────▶│  Forces (12) │   │
│  │ (x_ref)      │     │   (QP Solve) │     │              │   │
│  └──────────────┘     └──────────────┘     └──────┬───────┘   │
│                                                   │             │
└───────────────────────────────────────────────────┼─────────────┘
                                                    │
┌───────────────────────────────────────────────────┼─────────────┐
│                    LOW-LEVEL (1 kHz)              ▼             │
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
│  │     WBC      │     │    Swing     │     │    Torque    │   │
│  │ (J^T × F +   │────▶│   Leg PD     │────▶│   Commands   │   │
│  │  gravity)    │     │  + Bezier    │     │   (12 joints)│   │
│  └──────────────┘     └──────────────┘     └──────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Components:**
- **Gait Scheduler:** Phase-based (trot, bound, pace, crawl)
- **Trajectory Generator:** Reference state for MPC horizon (position, velocity)
- **Convex MPC:** Centroidal dynamics QP with friction cone constraints
- **WBC:** Jacobian-transpose + gravity compensation
- **Swing Controller:** Cartesian PD + cubic Bezier trajectories

**Data Flow:**
1. Gait scheduler outputs contact schedule [4 × H]
2. Trajectory generator creates reference [12 × (H+1)]
3. MPC solves QP → contact forces [12]
4. WBC converts forces → joint torques
5. Swing leg PD adds swing leg torques
6. Combined torques → robot actuators

---

### Pattern 2: Cascaded Fidelity MPC (Cafe-MPC)

Emerging architecture from CMU (2024) that reduces tuning burden:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Cafe-MPC Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Fidelity Level 1: Centroidal (Fast)         │  │
│  │         Low-dimensional QP (~30 variables)                │  │
│  └─────────────────────────┬────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Fidelity Level 2: Full WBC (Accurate)       │  │
│  │         Higher-dimensional QP (~50+ variables)          │  │
│  └─────────────────────────┬────────────────────────────────┘  │
│                            │                                     │
│                            ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         Automatic Weight Tuning (No Manual Tuning)        │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Key Innovation:** Automatic weight balancing between fidelity levels eliminates manual tuning.

**Reference:** Cafe-MPC (arXiv:2403.03995, 2024)

---

### Pattern 3: Nonlinear MPC Architecture

For high-speed or terrain-adaptive locomotion:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Nonlinear MPC Architecture                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐                                               │
│  │   acados /   │  Nonlinear optimal control                   │
│  │   CasADi     │  - Full robot dynamics                        │
│  │   Solver     │  - Contact dynamics                            │
│  └──────┬───────┘  - SQP or Gauss-Newton iterations             │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Solution: Full Trajectory                    │  │
│  │         [joint positions, velocities, torques]            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Characteristics:**
- Full nonlinear dynamics (no linearization)
- Longer solve time (20-50ms vs 5-10ms for linear)
- Better accuracy at high speeds
- Used by ETH Zurich for ANYmal perceptive locomotion

---

## Component Boundaries

### Layer 1: Gait & Reference

| Component | Responsibility | Input | Output |
|-----------|---------------|-------|--------|
| GaitScheduler | Phase-based contact scheduling | time, gait params | contact_schedule [4 × H] |
| TrajectoryGenerator | Reference trajectory | state, command | x_ref [12 × (H+1)] |
| FootPlacement (Raibert) | Heuristic foot targets | velocity, yaw rate | target_positions |

### Layer 2: Optimization

| Component | Responsibility | Input | Output |
|-----------|---------------|-------|--------|
| ConvexMPC | Centroidal dynamics QP | state, x_ref, contacts | forces [12] |
| QPSolver | Generic QP interface | H, f, A, b | solution |
| WBC | Whole-body control | forces, jacobians | torques [12] |

### Layer 3: Execution

| Component | Responsibility | Input | Output |
|-----------|---------------|-------|--------|
| SwingTrajectory | Bezier swing path | swing_targets, phase | joint_targets |
| SwingController | Cartesian PD | foot_target, foot_pos | swing_torques |
| TorqueBlender | Merge stance/swing | tau_stance, tau_swing | tau_final |

---

## Key Architectural Decisions

### QP Formulation Variants

| Variant | Variables | Constraints | Solvers |
|---------|-----------|-------------|---------|
| **Force-based (standard)** | 3 forces × 4 legs = 12 | Friction cones, force limits | OSQP, ProxQP, qpOASES |
| **Force + acceleration** | 12 + 6 acceleration | Plus CoP constraints | More complex |
| **Unified leg model** | Combine stance/swing | Phase-dependent | Research-level |

**Current project:** Force-based with friction cone constraints.

### Solver Integration Patterns

```
Pattern A: Embedded Solver (OSQP, ProxQP)
    MPC Controller → QP matrices → Solver → Forces
    
Pattern B: Code Generation (acados)
    MPC Controller → acados OCP → C code generation → Compiled solver
    
Pattern C: GPU Batched (ReLU-QP)
    Batch states → GPU QP solve → Batch forces (parallel)
```

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Per-Timestep Re-initialization

**Bad:** Create new solver object each control cycle
```python
# BAD: Overhead kills performance
for step in control_loop:
    solver = OSQP()
    solver.setup(Q, A, ...)
    result = solver.solve()
```

**Good:** Warm-start with previous solution
```python
# GOOD: Reuse solver, warm-start
solver = OSQP()
solver.setup(Q, A, ...)
for step in control_loop:
    solver.update(...)
    result = solver.solve(warm_start=prev_solution)
```

### Anti-Pattern 2: Hardcoded Constraint Bounds

**Bad:** Fixed friction coefficient, force limits
```python
# BAD: No adaptability
mu = 0.6  # hardcoded
max_force = 180  # hardcoded
```

**Good:** Configurable with safety margins
```python
# GOOD: Configurable, with margins
mu = config.friction_coefficient  # configurable
max_force = config.max_force * 0.8  # 20% margin
```

### Anti-Pattern 3: Mixing Frames Incorrectly

**Bad:** Forces in body frame, Jacobian in world frame
```python
# BAD: Frame mismatch
forces_body = mpc_output()
torques = J_world.T @ forces_body  # WRONG
```

**Good:** Consistent frame transformations
```python
# GOOD: Explicit frame handling
forces_world = R_body_to_world @ forces_body
torques = J_world.T @ forces_world
```

---

## Scalability Characteristics

| Configuration | MPC Solve Time | WBC Time | Total Latency |
|---------------|----------------|----------|---------------|
| Single QP (CPU) | 5-10ms | <1ms | ~10ms (100 Hz) |
| Single QP (OSQP optimized) | 2-5ms | <1ms | ~5ms (200 Hz) |
| Batched 1000 (GPU) | 2-5ms | <1ms | ~5ms |
| Batched 10000 (GPU) | 10-20ms | 2-5ms | ~25ms |

---

## Sources

### Architecture Papers

- Cafe-MPC: Cascaded-Fidelity MPC (arXiv:2403.03995, 2024)
- MIT Cheetah MPC (Di Carlo et al., 2018) — Foundational
- ETH Perceptive Locomotion (2022-2024)
- Inverse-Dynamics MPC via Nullspace Resolution (Mastalli et al., 2022)

### Implementation References

- ShuoYangRobotics/A1-QP-MPC-Controller — Production architecture
- iit-DLSLab/Quadruped-PyMPC — Python architecture
- Unitree official implementations

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Two-layer architecture | HIGH | Standard across labs |
| Cafe-MPC pattern | HIGH | Documented in 2024 paper |
| Nonlinear MPC | MEDIUM | Varied implementations |
| Scalability data | MEDIUM | Varies by hardware |
