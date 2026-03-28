# QuadrupedMPC — User Guide

## Quick Start

```bash
# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mj2

# Run simulation
cd /Users/premchand/Documents/GitHub/QuadrupedMPC
python main.py
```

---

## What to Expect

### MuJoCo Viewer

When you run `python main.py`, a MuJoCo viewer window opens showing the Unitree Go2 robot.

**Initial state:** Robot stands on flat ground.

**Standing phase (first 2-3 seconds):**
- Robot balances with all four feet on ground
- Base height ~0.32m
- Torques ~10-30 Nm per joint

**Walking phase:**
- Trot gait (diagonal foot pairs)
- Base moves forward at commanded speed
- MPC solves every 10ms (100 Hz)
- WBC computes torques every 2ms (500 Hz)

### Keyboard Controls

| Key | Action |
|-----|--------|
| `W` | Move forward |
| `S` | Move backward |
| `A` | Turn left |
| `D` | Turn right |
| `Space` | Stop |
| `Q` | Increase speed |
| `E` | Decrease speed |
| `R` | Increase yaw rate |
| `F` | Decrease yaw rate |

### Performance Indicators

**Console output (every 100 steps):**
```
step=100  t=0.100s  height=0.321m  mpc=0.078ms
```

- `height`: Base height (should stay ~0.30-0.32m)
- `mpc`: MPC solve time (should be <1ms with quadprog)

---

## Terrain Adaptation

### Running on Rough Terrain

```bash
python main.py --scene rough_terrain
```

### Running on Stairs

```bash
python main.py --scene stairs
```

### Terrain Estimator

The controller estimates terrain height and slope from stance feet:
- Uses EMA smoothing (α=0.1) to prevent jitter
- Computes surface normal from 3+ stance feet
- Filters swing feet (airborne) from height estimation

### Stair Controller

When approaching stairs:
- Detects stair region via x position (configurable `stair_start`)
- Increases swing height (0.15m vs 0.08m normal)
- Slows swing duration (0.3s vs 0.2s normal)
- Adjusts foot target to terrain-relative height

---

## Running Benchmarks

### QP Solver Benchmark

```bash
conda activate mj2
python tests/benchmark_solvers.py
```

**Expected output:**
```
======================================================================
QP Solver Benchmark — MPC Problem
======================================================================
Solver       Mean (ms)    Median (ms)  Success 
----------------------------------------------------------------------
clarabel     1.780        1.625        100     
osqp         2.727        2.662        100     
quadprog     0.088        0.079        100     
scs          2.219        1.999        100     
======================================================================

Fastest solver: quadprog
```

### Kinematics Validation

```bash
conda activate mj2
python go2_mpc/kinematics/validate_kinematics.py
```

**Expected output:**
All tests pass with max error < 1e-5.

---

## Configuration

### Changing Solver

Edit `go2_mpc/config/config.py`:
```python
solver=SolverConfig(
    solver_name='quadprog',  # or 'clarabel', 'osqp', 'scs'
    verbose=False,
)
```

### Changing Gait

Edit `go2_mpc/config/config.py`:
```python
gait=GaitConfig(
    gait_period=0.45,   # seconds per full gait cycle
    stance_ratio=0.65,  # fraction of cycle with foot on ground
)
```

### Changing MPC Frequency

```python
mpc=MPCConfig(
    dt=0.01,  # 0.01s = 100 Hz, 0.03s = 33 Hz
    ...
)
```

### Enabling/Disabling Balance Controller

```python
balance=BalanceConfig(
    enabled=True,  # Set False to disable push recovery
    ...
)
```

---

## Testing

### Run All Tests

```bash
conda activate mj2
python -m pytest tests/ -v
```

### Run Specific Test

```bash
python -m pytest tests/test_controllers.py::test_convex_mpc -v
```

---

## Troubleshooting

### MuJoCo Viewer Doesn't Open

```bash
# Check MuJoCo installation
python -c "import mujoco; print(mujoco.__version__)"

# If error, reinstall
pip install mujoco --upgrade
```

### MPC Solve Time Too High

```bash
# Check solver
python -c "from qpsolvers import available_solvers; print(available_solvers)"

# Install quadprog (fastest)
pip install quadprog
```

### Robot Falls Immediately

- Check Q/R weights in config
- Verify friction coefficient (mu=0.6 for concrete)
- Increase swing_kp (more aggressive swing control)

### Low Base Height

- Increase default_height in config (default: 0.32m)
- Check Q[1,1] weight (height tracking, default: 50)

---

## Architecture Summary

```
main.py
    └── ControllerCore
        ├── GaitScheduler (100 Hz)
        ├── TrajectoryGenerator (33 Hz)
        ├── ConvexMPC (100 Hz, quadprog solver)
        ├── WholeBodyController (500 Hz)
        └── BalanceController (on disturbance)
```

**Timing:**
- Simulation: 1 kHz
- Gait + WBC: 100-500 Hz
- MPC: 100 Hz

---

## Environment

```bash
conda env: mj2
python: 3.9
dependencies: mujoco, numpy, cvxpy, qpsolvers, quadprog, clarabel
```

---

*Last updated: 2026-03-28*
