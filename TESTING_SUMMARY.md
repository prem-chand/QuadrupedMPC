# Go2 MPC Testing Summary

This document summarizes the tests performed during debugging and validation of the MIT Cheetah MPC implementation for the Go2 robot.

## Bugs Found and Fixed

### 1. Wrong Pinocchio Package
- **Issue**: Installed `pinocchio` was a nose testing plugin (v0.4.3), not the robotics library
- **Fix**: Rewrote `state_estimator.py` and `wbc.py` to use MuJoCo's native kinematics
- **Test**: Verified imports work without Pinocchio dependency

### 2. Incorrect Robot Mass
- **Issue**: Code assumed 12 kg, but actual MuJoCo model mass is 15.2 kg
- **Fix**: Updated `mass = 15.2` in main.py
- **Test**: Verified with `sum(model.body_mass)` = 15.206 kg

### 3. WBC Sign Error
- **Issue**: Jacobian transpose method had wrong sign (`tau = J^T @ F` instead of `tau = -J^T @ F`)
- **Fix**: Changed to `leg_torques = -J_leg.T @ desired_forces[i]`
- **Test**: Applied positive calf torque and verified robot lifts (z increases)

### 4. Actuator Order Mismatch
- **Issue**: Joint order in qpos (FL, FR, RL, RR) differs from actuator order (FR, FL, RR, RL)
- **Fix**: Created proper mapping dictionaries in WBC and JointPDController
- **Test**: Verified actuator-to-joint mapping with `model.actuator_trnid`

### 5. Swing Phase Calculation Bug
- **Issue**: Swing phase could exceed [0, 1] range, causing invalid trajectories
- **Fix**: Added `get_swing_phase()` method with proper per-leg calculation
- **Test**: Verified swing phase stays in [0, 1] for all legs at various times

### 6. Simulation Loop Timing Bug
- **Issue**: `mj_step()` was inside control condition, so simulation wouldn't advance
- **Fix**: Moved `mj_step()` outside control condition, added proper timing
- **Test**: Verified simulation time advances correctly

---

## Sanity Tests Performed

### Test 1: Component Import Test
```python
# Verified all modules import correctly
from go2_mpc.controller.gait_generator import GaitGenerator
from go2_mpc.controller.convex_mpc import ConvexMPC
from go2_mpc.controller.srb_dynamics import SRBDynamics
from go2_mpc.controller.state_estimator import StateEstimator
from go2_mpc.controller.wbc import WBC
from go2_mpc.controller.swing_leg_controller import SwingLegController
```
**Result**: All imports successful

### Test 2: Gait Generator Swing Phase
```python
gg = GaitGenerator(0.5, 0.5, 10, 0.02)
# t=0.0: FL=-1.00 (stance), FR=0.00 (swing start)
# t=0.125: FL=-1.00, FR=0.50 (mid-swing)
# t=0.25: FL=0.00 (swing start), FR=-1.00 (stance)
```
**Result**: Swing phases correctly bounded to [0, 1] or -1 for stance

### Test 3: State Estimator
```python
q = np.zeros(19)
q[3] = 1.0  # quaternion w
v = np.zeros(18)
state = state_estimator.estimate(q, v)
# state.shape = (12,) -> [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
```
**Result**: Returns correct 12D state vector

### Test 4: MuJoCo Model Properties
```python
model.opt.timestep = 0.002  # 2ms simulation timestep
model.opt.gravity = [0, 0, -9.81]
sum(model.body_mass) = 15.206 kg
model.nq = 19  # 7 freejoint + 12 joints
model.nv = 18  # 6 freejoint + 12 joints
model.nu = 12  # 12 actuators
```

### Test 5: Joint-Actuator Mapping
```
Actuator 0 (FR_hip) -> Joint FR_hip_joint -> qvel[9]
Actuator 1 (FR_thigh) -> Joint FR_thigh_joint -> qvel[10]
Actuator 2 (FR_calf) -> Joint FR_calf_joint -> qvel[11]
Actuator 3 (FL_hip) -> Joint FL_hip_joint -> qvel[6]
...
```
**Result**: Mapping verified correct

### Test 6: Jacobian Analysis
```python
# FL_foot Jacobian z-row for columns [6,7,8]:
# [0.0955, ~0, -0.167]
# - Hip has small z contribution
# - Thigh has ~0 z contribution at standing pose
# - Calf has -0.167 z contribution
```
**Result**: Jacobian structure matches expected kinematics

### Test 7: Torque Sign Convention
```python
# Apply +10 Nm to all calf joints:
# Initial z=0.30 -> After 200 steps z=0.425 (robot lifts)
# Apply -10 Nm to all calf joints:
# Initial z=0.30 -> After 200 steps z=0.110 (robot falls)
```
**Result**: Positive torque extends leg and lifts robot

### Test 8: Pitch/Roll Convention
```python
# Positive Y rotation (pitch):
# FL foot z: -0.018, RL foot z: 0.059
# -> Front is lower (positive pitch = nose down)

# Positive X rotation (roll):
# FL foot z: 0.049, FR foot z: -0.008
# -> Right side is lower (positive roll = tilted right)
```

### Test 9: Joint PD Controller Standing
```python
# 5 second simulation with joint PD control (kp=60, kd=5)
# t=0.0s: z=0.280, roll=0.00, pitch=0.00
# t=5.0s: z=0.269, roll=0.00, pitch=-0.01
```
**Result**: Robot stands stably with minimal drift

### Test 10: MPC Solver
```python
# MPC with fallback forces when solver fails
problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
# Returns optimal forces or fallback (weight/4 per leg)
```
**Result**: Solver runs, fallback works on infeasibility

---

## Test Commands

### Run Full Simulation
```bash
cd /Users/premchand/Documents/GitHub/Go1MPC
mjpython -m go2_mpc.main
```

### Run Unit Tests
```bash
cd /Users/premchand/Documents/GitHub/Go1MPC
python -m pytest tests/ -v
```

---

## Known Limitations

1. **Force-based WBC is unstable for trotting** - The Jacobian transpose method with the current gains causes oscillations. Joint PD control is used for standing instead.

2. **MPC dynamics are simplified** - The B matrix doesn't include angular dynamics from foot forces (torque contribution).

3. **No foot contact detection** - The swing leg controller doesn't detect actual ground contact.

---

## Files Modified

| File | Changes |
|------|---------|
| `main.py` | Fixed mass, timing, added joint PD controller |
| `state_estimator.py` | Rewrote for MuJoCo, added foot velocity |
| `wbc.py` | Rewrote for MuJoCo, fixed sign, added mapping |
| `gait_generator.py` | Added `get_swing_phase()`, renamed params |
| `convex_mpc.py` | Added solver failure handling |
| `swing_leg_controller.py` | Fixed velocity calculation |
| `joint_pd_controller.py` | New file for stable standing |
| `stand_controller.py` | Fixed orientation signs (unused) |
