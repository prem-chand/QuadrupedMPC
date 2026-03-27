# Technology Stack

**Project:** QuadrupedMPC - Extension for IsaacLab & Batched GPU MPC
**Researched:** 2026-03-27
**Confidence:** HIGH (verified via official docs, GitHub, arXiv)

## Recommended Stack

### Core Simulation Backend

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **IsaacLab** | v2.0+ | GPU-accelerated simulation | Industry standard for quadruped RL; 6,752 stars; native support for Unitree Go2; deterministic physics; 10,000+ fps on RTX GPU |
| **Isaac Sim** | Latest | Underlying physics engine | NVIDIA Omniverse-based; physically accurate; supports USD workflow |
| **MuJoCo** | v3.0+ | Fallback / dev simulation | Keep for rapid prototyping;MJX provides GPU acceleration |

**Rationale:** IsaacLab is the de facto standard for quadruped locomotion research (Spot, ANYmal, Unitree). It provides:
- GPU-parallelized simulation (10K+ envs on single RTX GPU)
- Native Unitree Go2 robot description (USD format)
- Integrated RL training pipeline with RSL-RL
- Deterministic physics for reproducible sim-to-real transfer

**Confidence:** HIGH — IsaacLab paper published 2025, actively maintained by NVIDIA, used in Spot quadruped training (NVIDIA blog 2024).

---

### RL Training (if MPC-augmented RL is goal)

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **RSL-RL** | v5.0+ | RL algorithms (PPO) | 2.4K stars; integrated with IsaacLab out-of-the-box; multi-GPU support; proven on all major quadrupeds |
| **legged_gym** | Latest | Isaac Gym legacy | Still relevant for Isaac Gym users; IsaacLab is successor |

**Rationale:** RSL-RL is the canonical choice for IsaacLab. It provides clean extension points, multi-GPU training, and is used by NVIDIA for Spot training (see NVIDIA Technical Blog 2024).

**Confidence:** HIGH — RSL-RL v5.0 released March 2026; IsaacLab docs recommend it.

---

### GPU-Accelerated Batched MPC Solvers

| Technology | Purpose | When to Use | Why |
|------------|---------|-------------|-----|
| **ReLU-QP** | GPU QP solver for MPC | Batch solving across 1000+ envs | Uses ADMM with GPU kernels; 10-100x faster than CPU solvers; specifically designed for MPC; arXiv 2023 |
| **OSQP + cu12 backend** | Differentiable QP | Need gradient flow to policy | Native CUDA support; differentiable via osqpth; well-documented |
| **Custom Torch QP** | Simple batched MPC | Full control over solver | Use `torch.linalg.solve` for unconstrained or custom ADMM loop; maximum flexibility |
| **ProxQP** | Efficient QP (CPU) | Prototyping / smaller batches | Fast CPU solver; good for <100 parallel envs; Python-native API |

**NOT RECOMMENDED:**
- **CVXPY/CLARABEL** (current): Not GPU-accelerated; single-solve only; too slow for batched MPC across 1000+ envs
- **qpSWIFT**: CPU-only; no GPU support

**Rationale for ReLU-QP:** 
- Specifically designed for batched MPC in robotics (paper: IEEE ICRA 2024)
- GPU-native via CUDA
- Solves multiple QPs in parallel (one per environment)
- MIT license, actively maintained

**Confidence:** MEDIUM — ReLU-QP has 142 stars, published in IEEE ICRA 2024. GATO (2025) shows newer research direction but less mature.

---

### IsaacLab Robot Interface

| Component | Purpose | Integration |
|-----------|---------|-------------|
| `isaaclab.assets.articulation.Articulation` | Robot state access | Get joint positions, velocities, apply torques |
| `isaaclab.assets.articulation.ArticulationCfg` | Robot configuration | Load USD, configure joints, control modes |
| `isaaclab.controllers.OperationalSpaceController` | WBC example | Reference implementation for whole-body control |
| Custom `Robot` ABC | Your controller interface | Mirror existing `core/robot.py` pattern |

**Integration Pattern:**
```
IsaacLab Articulation → Custom Robot ABC → Your existing controller/
                                              (convex_mpc.py, wbc.py, etc.)
```

**Confidence:** HIGH — IsaacLab documentation covers this pattern; `run_osc.py` tutorial shows exact integration.

---

### Optional: GPU-Accelerated MuJoCo

| Technology | Purpose | When to Use |
|------------|---------|-------------|
| **MuJoCo MJX** | GPU backend for MuJoCo | If staying with MuJoCo but need GPU speedup |
| **MuJoCo Warp** | Differentiable physics | For model-based RL with gradients |

**Note:** If IsaacLab integration is complex, MJX provides intermediate GPU acceleration for MuJoCo without full simulator switch.

---

## Installation

### IsaacLab (recommended path)

```bash
# 1. Install Isaac Sim (NVIDIA Omniverse)
# Download from NVIDIA website (requires RTX GPU for best performance)

# 2. Clone Isaac Lab
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# 3. Create environment and install
./scripts/create_conda_env.sh  # Creates isaaclab conda env
source scripts/isaaclab_shell.sh

# 4. Install dependencies
pip install torch  # Use CUDA-enabled torch for GPU training
pip install rsl-rl-lib  # RL training
```

### ReLU-QP (GPU-accelerated QP solver)

```bash
# Install from source (requires CUDA)
git clone https://github.com/RoboticExplorationLab/ReLUQP-py.git
cd ReLUQP-py
pip install -e .
```

### For custom batched Torch MPC

```bash
# Standard PyTorch with CUDA (likely already installed)
# Use torch.compile for JIT acceleration
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Architecture for Batched MPC + RL

### Option A: IsaacLab + RSL-RL + ReLU-QP (Recommended)

```
┌─────────────────────────────────────────────────────────────┐
│                    IsaacLab Simulation                       │
│  (10,000 parallel environments, GPU-accelerated physics)   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Custom Robot Interface (ABC)                   │
│  get_joint_state() → get_foot_positions() → etc.           │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
     ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
     │ GaitScheduler│  │ Convex MPC   │  │ WBC (J^T F)  │
     │ (phase-based)│  │(ReLU-QP GPU)│  │(Torch batch) │
     └──────────────┘  └──────────────┘  └──────────────┘
              │               │               │
              └───────────────┼───────────────┘
                              ▼
              ┌──────────────────────────────┐
              │    Torque Commands (batched) │
              │   Shape: (num_envs, 12)      │
              └──────────────────────────────┘
                              │
                              ▼
              ┌──────────────────────────────┐
              │     RSL-RL PPO Agent         │
              │  (optional: learn residual)  │
              └──────────────────────────────┘
```

**Why this architecture:**
- IsaacLab handles physics in parallel on GPU
- Your MPC controller processes batched states (num_envs, 12) each step
- ReLU-QP solves QP for each environment in parallel on GPU
- RSL-RL trains on collected experience

### Option B: Keep MuJoCo, Add MJX (Simpler)

If IsaacLab integration is too complex:
- Use MuJoCo MJX for GPU-accelerated physics
- Keep existing controller structure
- Implement batched MPC via custom Torch solver

---

## Summary Recommendations

| Goal | Recommended Stack |
|------|-------------------|
| **IsaacLab backend** | IsaacLab v2.0+ + RSL-RL v5.0+ |
| **GPU batched MPC** | ReLU-QP or custom Torch batched QP |
| **RL training** | RSL-RL (integrates with IsaacLab) |
| **Keep MuJoCo** | Stick with CVXPY/CLARABEL for single-solve; too slow for batching |
| **Differentiable MPC** | OSQP with osqpth or torch custom |

---

## Sources

| Source | Type | Confidence |
|--------|------|------------|
| IsaacLab GitHub (6,752 stars) | Official | HIGH |
| RSL-RL GitHub (2,400 stars) | Official | HIGH |
| IsaacLab Paper (Mittal et al., 2025) | arXiv | HIGH |
| ReLU-QP Paper (Bishop et al., ICRA 2024) | arXiv | HIGH |
| NVIDIA Blog: Spot Quadruped with IsaacLab | Official | HIGH |
| GATO (Du et al., 2025) | arXiv | MEDIUM (new research) |

---

## Open Questions

1. **IsaacLab + custom MPC integration complexity?** — Need to verify exact API for applying torques per timestep
2. **ReLU-QP maturity?** — 142 stars vs. 2,400 for RSL-RL; verify it handles friction cone constraints natively
3. **Single-GPU training vs. multi-GPU?** — For 10K envs, may need multi-GPU; RSL-RL supports this

**Recommendation:** Start with IsaacLab + single-solve MPC (keeping CLARABEL) as baseline; profile before adding ReLU-QP complexity.
