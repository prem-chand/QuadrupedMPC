# Architecture Patterns: Batched GPU MPC with IsaacLab

**Domain:** Quadruped Robot Control — MPC-WBC with GPU-Parallel RL Training
**Researched:** 2026-03-27
**Confidence:** MEDIUM-HIGH

## Recommended Architecture

For extending the existing MPC-WBC controller to IsaacLab with batched GPU MPC, the recommended architecture follows a **hierarchical controller with simulator-agnostic boundaries** pattern. This architecture enables parallel training across thousands of GPU environments while maintaining the clean separation between control algorithms and simulation backend already established in the project.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           RL TRAINING PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                   │
│  │   IsaacLab   │────▶│   RL Agent   │────▶│  Env Reset   │                   │
│  │  Simulator   │     │  (PPO/PPGR) │     │   Handler    │                   │
│  │ (GPU Batched)│◀────│             │◀────│              │                   │
│  └──────┬───────┘     └──────────────┘     └──────────────┘                   │
│         │                                                                      │
│         ▼                                                                      │
│  ┌──────────────────────────────────────────────────────────────────┐          │
│  │                    IsaacLab Environment                         │          │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │          │
│  │  │ Scene Mgr   │  │ Obs Manager │  │    Action Manager      │  │          │
│  │  │ (terrain,   │  │ (policy,    │  │    (joint torque,      │  │          │
│  │  │  lights)    │  │  history)   │  │     position target)  │  │          │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────┘  │          │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │          │
│  │  │ Reward Mgr  │  │Term Manager │  │   Command Manager      │  │          │
│  │  │ (tracking,  │  │ (fall, time │  │   (velocity cmd,       │  │          │
│  │  │  stability) │  │  out, etc)  │  │    yaw rate, height)  │  │          │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────┘  │          │
│  └──────────────────────────────────────────────────────────────────┘          │
│                                    │                                            │
│                                    ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────┐          │
│  │              CONTROLLER STACK (Simulator-Agnostic)               │          │
│  │                                                                   │          │
│  │  ┌────────────────────────────────────────────────────────────┐  │          │
│  │  │              ControllerCore (Orchestrator)                 │  │          │
│  │  │  - GaitScheduler → contact_schedule (B×4×H)               │  │          │
│  │  │  - TrajectoryGenerator → x_ref (B×12×(H+1))               │  │          │
│  │  │  - Solver orchestration (CPU/GPU)                          │  │          │
│  │  └────────────────────────────────────────────────────────────┘  │          │
│  │                              │                                    │          │
│  │         ┌────────────────────┼────────────────────┐                │          │
│  │         ▼                    ▼                    ▼                │          │
│  │  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │          │
│  │  │ ConvexMPC   │     │     WBC     │     │   Swing     │       │          │
│  │  │ (QP Solver) │     │ (J^T×F +    │     │  Trajectory │       │          │
│  │  │ Centroidal  │     │  gravity)   │     │   (Bezier)  │       │          │
│  │  │ Dynamics    │     │             │     │             │       │          │
│  │  └─────────────┘     └─────────────┘     └─────────────┘       │          │
│  │                                                                   │          │
│  └──────────────────────────────────────────────────────────────────┘          │
│                                    │                                            │
│                                    ▼                                            │
│  ┌──────────────────────────────────────────────────────────────────┐          │
│  │                   Robot Interface (ABC)                          │          │
│  │  ┌─────────────────────────────────────────────────────────────┐ │          │
│  │  │ get_base_pose() → (pos, quat)        get_joint_state()    │ │          │
│  │  │ get_base_velocity() → (lin, ang)    get_foot_positions()  │ │          │
│  │  │ get_foot_jacobian() → (3, nv)       get_foot_velocity()   │ │          │
│  │  │ get_gravity_compensation() → (3,)   set_torques()         │ │          │
│  │  └─────────────────────────────────────────────────────────────┘ │          │
│  └──────────────────────────────────────────────────────────────────┘          │
│                                    │                                            │
│         ┌──────────────────────────┼──────────────────────────┐                │
│         ▼                          ▼                          ▼                │
│  ┌─────────────┐            ┌─────────────┐            ┌─────────────┐        │
│  │ MujocoRobot │            │ IsaacRobot  │            │ PyBullet    │        │
│  │ (current)   │            │ (to build)  │            │ (optional)  │        │
│  └─────────────┘            └─────────────┘            └─────────────┘        │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

Legend: B = batch size (num_envs), H = MPC horizon steps
```

## Component Boundaries

### Layer 1: Simulation Backend (IsaacLab)

| Component | Responsibility | Communicates With |
|-----------|---------------|-------------------|
| `IsaacEnv` | GPU-parallel physics simulation, handles thousands of envs | Scene Manager, Controller Stack |
| `SceneManager` | Terrain spawning, lighting, robot asset loading | IsaacEnv, Observation Manager |
| `SimWriter` | Batched tensor writes (torques → GPU) | Controller Stack |

**Boundary Contract:** IsaacLab provides batched state tensors (`[B, state_dim]`) and accepts batched action tensors. The controller stack processes these as batched operations.

### Layer 2: IsaacLab Managers (Task Specification)

| Component | Responsibility | Communicates With |
|-----------|---------------|-------------------|
| `ObservationManager` | Constructs policy observations from state | IsaacEnv → RL Agent |
| `RewardManager` | Computes scalar rewards per environment | IsaacEnv, Controller Stack |
| `TerminationManager` | Determines episode end conditions | IsaacEnv |
| `CommandManager` | Generates velocity/yaw commands (goal-conditioned) | IsaacEnv |
| `EventManager` | Handles reset logic, domain randomization | IsaacEnv |

**Key Insight:** IsaacLab uses a **manager-based configuration** pattern where the environment is defined declaratively via dataclasses (`ManagerBasedRLEnvCfg`). This differs from the current project's imperative style but can coexist.

### Layer 3: Controller Stack (Simulator-Agnostic)

| Component | Responsibility | Input | Output |
|-----------|---------------|-------|--------|
| `ControllerCore` | Orchestrates gait→MPC→WBC→swing at control freq | State, Command, contact_schedule | torques [B, 12] |
| `GaitScheduler` | Phase-based contact scheduling | gait_params, time | contact_schedule [B, 4, H] |
| `TrajectoryGenerator` | Reference state for MPC horizon | state, command, contacts | x_ref [B, 12, H+1] |
| `ConvexMPC` | Centroidal dynamics QP | state, x_ref, contacts, feet | forces [B, 12] |
| `WBC` | Jacobian-transpose whole-body control | forces, state, jacobians | stance_torques [B, 12] |
| `FootSwingTraj` | Cubic Bezier swing trajectory | swing_params, phase | swing_targets [B, 4, 3] |

**Critical:** All controller components must operate on batched tensors `[B, ...]` for GPU parallelism.

### Layer 4: Robot Interface (Abstract Boundary)

| Method | Purpose | Tensor Shape |
|--------|---------|--------------|
| `get_base_pose()` | Base position and orientation | [B, 7] (pos + quat) |
| `get_base_velocity()` | Base linear/angular velocity | [B, 6] |
| `get_joint_state()` | Joint positions and velocities | [B, 12] each |
| `get_foot_positions_world()` | Foot positions in world frame | [B, 4, 3] |
| `get_leg_jacobian()` | Leg Jacobian matrices | [B, 4, 3, 3] |
| `set_torques()` | Apply joint torques | [B, 12] |

**This layer already exists** in the project (`core/robot.py`) — it needs a batched tensor variant for IsaacLab.

## Data Flow

### Training Data Flow (Batched)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STEP k: ENVIRONMENT LOOP                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. ISAACLAB SIMULATOR                                                     │
│     state_k [B, state_dim] ──────────────────────────────────────────┐     │
│                                                                        │     │
│  2. OBSERVATION CONSTRUCTION (ObservationManager)                     │     │
│     state_k ──▶ obs_k [B, obs_dim] ──▶ RL Agent                      │     │
│                                                                        │     │
│  3. RL AGENT FORWARD PASS                                               │     │
│     obs_k ──▶ action_k [B, action_dim] ──▶ either:                    │     │
│                      │                                                   │     │
│                      ├── Option A: Direct torque commands [B, 12]      │     │
│                      │        (simpler, used in this project)          │     │
│                      │                                                   │     │
│                      └── Option B: MPC weight parameters [B, mpc_dim]  │     │
│                               (for learned MPC, e.g., rl-mpc-locomotion)│     │
│                                                                        │     │
│  4. CONTROLLER STACK (per environment)                                 │     │
│     action_k + state_k ──▶ ControllerCore ──▶ torques [B, 12]        │     │
│        │                                                                  │     │
│        ├── GaitScheduler: time_k ──▶ contacts [B, 4, H]              │     │
│        ├── TrajectoryGenerator: state + cmd ──▶ x_ref [B, 12, H+1]   │     │
│        ├── ConvexMPC: QP solve ──▶ forces [B, 12]                    │     │
│        ├── WBC: forces + jacobians ──▶ stance_tau [B, 12]            │     │
│        └── Swing: swing_targets + PD ──▶ swing_tau [B, 12]           │     │
│                                                                        │     │
│  5. TORQUE APPLICATION                                                  │     │
│     torques [B, 12] ──▶ IsaacEnv.set_dof_actions() ──▶ physics step  │     │
│                                                                        │     │
│  6. REWARD COMPUTATION (RewardManager)                                  │     │
│     state_k+1 + torques ──▶ rewards [B, 1]                            │     │
│                                                                        │     │
│  7. TERMINATION CHECK (TerminationManager)                             │     │
│     state_k+1 ──▶ terminated [B], truncated [B]                      │     │
│                                                                        │     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Data Transformation Points

| Point | Transformation | Batch-Aware? |
|-------|---------------|--------------|
| State extraction | Simulator state → Robot ABC interface | Yes — `[B, ...]` |
| Gait scheduling | Scalar time → binary contact schedule | Yes — broadcasts to `[B, ...]` |
| MPC solve | Single QP → batched QP (one per env) | **Requires implementation** |
| WBC | Single Jacobian → batched Jacobians `[B, 4, 3, 3]` | Yes |
| Torque application | `[B, 12]` → IsaacLab action buffer | Yes |

## Patterns to Follow

### Pattern 1: Batched QP Solver for GPU MPC

For true GPU parallelism in MPC, the QP solve must operate on batched matrices. Two approaches:

**Approach A: OSQP/CLARABEL with Batched API**
```python
# Each environment solves its own QP, but matrices are stacked
H_batch = torch.stack([H_0, H_1, ..., H_B-1], dim=0)  # [B, 12+H*12, 12+H*12]
f_batch = torch.stack([f_0, f_1, ..., f_B-1], dim=0)  # [B, 12+H*12]

# OSQP has limited batch support; CLARABEL is better
import clarabel
solver = clarabel.DefaultSolver(...)  # May need per-env instantiation
solutions = [solver.solve(H_i, f_i) for i in range(B)]  # Sequential :(
```

**Approach B: Differentiable MPC Layer (Recommended for RL-MPC)**
```python
# mpc.pytorch provides batched iLQR/DDP on GPU
from mpc import mpc

x_lqr, u_lqr, objs = mpc.MPC(
    n_state=12,
    n_ctrl=12,
    T=horizon,
    u_lower=force_bounds,
    u_upper=force_bounds,
    lqr_iter=10,
    batch_size=num_envs,
)(x_init_batch, quad_cost, lin_dynamics)
```

**Recommendation:** For pure MPC-WBC (not learning MPC weights), use **Approach A** with CLARABEL or custom CUDA kernels. For RL-augmented MPC (where RL predicts MPC parameters), use **Approach B** (mpc.pytorch).

### Pattern 2: IsaacLab Environment Integration

IsaacLab supports two workflows:

| Workflow | Use Case | Complexity |
|----------|----------|------------|
| **Manager-Based** | Standard RL tasks, reward shaping | Low — declarative config |
| **Direct Workflow** | Custom controllers, tight simulation coupling | Medium — imperative Python |

**For this project:** Use **Direct Workflow** because:
1. The existing controller stack is mature and simulator-agnostic
2. Manager-based would require reimplementing controller logic in IsaacLab's manager patterns
3. Direct workflow allows direct `step()` calls into the controller stack

```python
# Direct Workflow Pattern (recommended)
class QuadrupedRlEnv:
    def __init__(self, cfg):
        self.robot = IsaacRobot(cfg)  # implements Robot ABC
        self.controller = ControllerCore(cfg.controller)
        
    def step(self, action):
        # action can be: direct torques OR MPC weight predictions
        self.controller.update(self.robot.get_state(), action)
        torques = self.controller.compute_torques()
        self.robot.set_torques(torques)
        self.sim.step()
        
        return obs, reward, terminated, info
```

### Pattern 3: Hybrid RL-MPC Architecture

For advanced use cases (MPC-augmented RL), the pattern from `rl-mpc-locomotion`:

```
┌─────────────────┐     ┌─────────────────┐
│   RL Policy     │────▶│  MPC Weight     │
│  (neural net)   │     │  Predictor      │
└─────────────────┘     └────────┬────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │   Convex MPC Controller │
                    │  (uses predicted params)│
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Whole-Body Control    │
                    └─────────────────────────┘
```

This allows RL to learn what MPC parameters work best for different situations, while MPC handles real-time control.

## Anti-Patterns to Avoid

### Anti-Pattern 1: Per-Environment Python Loops in Controller

**What:** Iterating over batch dimension in Python
```python
# BAD: Sequential loop loses GPU parallelism
for i in range(num_envs):
    mpc_result = solve_mpc(state[i], ...)
    torques[i] = mpc_result.torques
```

**Instead:** Vectorized tensor operations
```python
# GOOD: Full GPU parallelism
H_batch = self._build_QP_matrices(state_batch)  # [B, ...]
forces_batch = self.solver.solve_batch(H_batch)  # [B, 12]
```

### Anti-Pattern 2: Breaking the Robot ABC for IsaacLab

**What:** Adding IsaacLab-specific methods to the controller
```python
# BAD: Controller knows about IsaacLab tensors
class ConvexMPC:
    def solve(self, env):
        # directly accesses env._physics_view
```

**Instead:** Keep the Robot ABC pure, implement batched variant
```python
# GOOD: Batched robot interface
class IsaacRobot(Robot):
    def get_foot_positions_world(self) -> torch.Tensor:  # [B, 4, 3]
        return self._foot_pos_buffer  # pre-allocated GPU tensor
```

### Anti-Pattern 3: Synchronous GPU-CPU Data Transfer Each Control Step

**What:** Moving tensors CPU→GPU→CPU each frame
```python
# BAD: Overhead kills performance
for env in envs:
    state = env.get_state().cpu().numpy()  # CPU transfer
    result = mpc.solve(state)  # CPU solve
    torques = torch.tensor(result.torques).cuda()  # GPU transfer
```

**Instead:** Keep all computation on GPU
```python
# GOOD: Fully GPU-resident
state_batch = env.get_state()  # [B, ...] already on GPU
forces_batch = mpc_batch(state_batch)  # GPU solve
env.set_actions(forces_batch)  # direct GPU write
```

## Scalability Considerations

| Concern | 256 envs | 4096 envs | 16384 envs |
|---------|----------|-----------|------------|
| **Simulation** | IsaacLab native | IsaacLab native | May need GPU memory tuning |
| **QP Solve (CPU)** | ~5ms/solve × 256 | Timeout likely | Not viable |
| **QP Solve (GPU batched)** | ~0.5ms/solve | ~2ms/solve | ~8ms/solve |
| **WBC Compute** | ~0.1ms | ~0.5ms | ~2ms |
| **Total control loop** | <10ms | <15ms | <25ms |

**Key insight:** The bottleneck shifts from simulation (IsaacLab handles efficiently) to the QP solve. GPU-batched solvers are essential for 4096+ environments.

## Build Order Recommendations

Based on dependencies and testability:

### Phase 1: IsaacLab Integration (Foundation)
1. **IsaacRobot implementation** — implements Robot ABC for IsaacLab
   - Batch state extraction from IsaacLab tensors
   - Batch Jacobian computation (analytical URDF or finite-diff)
   - Test: Single-env stepping matches MuJoCo
   
2. **Batched WBC** — extend existing WBC to batched operation
   - Input: `[B, 4, 3, 3]` Jacobians, `[B, 12]` forces
   - Output: `[B, 12]` torques
   - Test: Batched WBC produces same results as single-env

### Phase 2: Batched MPC Solver
3. **GPU-accelerated QP** — replace CVXPY with batched solver
   - Option A: Custom CUDA QP kernel (OSQP-style)
   - Option B: CLARABEL with batched setup
   - Option C: mpc.pytorch iLQR for differentiable use
   
4. **Batched ConvexMPC** — wrap QP solver with MPC formulation
   - Input: `[B, 12]` state, `[B, 12, H+1]` ref trajectory
   - Output: `[B, 12]` optimal forces
   - Test: Batch solve ≈ N single solves

### Phase 3: RL Training Integration
5. **Direct Workflow Environment** — create IsaacLab env that uses controller
   - Integrates with RL libraries (RSL-RL, stable-baselines3)
   - Handles reset, reward computation, termination
   
6. **Reward shaping** — addIsaacLab reward terms
   - Tracking reward, stability penalty, effort penalty
   - Curriculum learning support

### Phase 4: Advanced (Optional)
7. **Differentiable MPC** — integrate mpc.pytorch for RL-MPC
8. **Hybrid RL-MPC** — RL predicts MPC parameters

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| IsaacLab architecture | HIGH | Official documentation and source patterns |
| Batched MPC patterns | MEDIUM | mpc.pytorch well-documented; GATO is new (2025) |
| Scalability estimates | MEDIUM | Based on IsaacLab benchmarks; actual depends on implementation |
| Build order | HIGH | Standard embedded systems dependencies |

## Sources

- Isaac Lab Documentation — Manager-Based and Direct Workflow environments (2026-03)
- NVIDIA Isaac Lab arXiv paper (arXiv:2511.04831, 2025-11)
- mpc.pytorch — Differentiable MPC solver (locuslab.github.io)
- GATO — GPU-Accelerated Batched Trajectory Optimization (arXiv:2510.07625, 2025-10)
- rl-mpc-locomotion — RL-MPC integration pattern (GitHub silvery107, 931 stars)
- IsaacLab-Quadruped-Tasks — Community quadruped extensions (GitHub felipemohr)
