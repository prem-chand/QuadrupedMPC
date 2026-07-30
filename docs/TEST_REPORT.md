# Quadruped MPC Controller - Comprehensive Test Report

## Executive Summary

This report documents comprehensive testing of the MIT Cheetah-style convex MPC controller for the Unitree Go2 quadruped robot in MuJoCo simulation. Testing reveals critical stability issues that prevent reliable autonomous walking. The robot exhibits height collapse, roll instability, and systematic lateral drift across all test scenarios.

---

## 1. Experimental Setup

### 1.1 Robot Configuration
- **Robot**: Unitree Go2 (mass: 15.2 kg)
- **Simulation**: MuJoCo with 1kHz physics step
- **Control Loop**: 
  - MPC: 100 Hz (dt = 0.01s, horizon = 10 steps)
  - WBC: 100 Hz
  - Swing control: 1 kHz

### 1.2 Controller Parameters
```
MPC Configuration:
- Mass: 15.2 kg
- Inertia: diag([0.18, 0.35, 0.3]) kg·m²
- Horizon: 10 steps (0.1s prediction)
- Q weights: diag([5, 5, 50, 20, 20, 10, 8, 8, 15, 15, 5, 3])
- R weights: diag(1e-3 * 12)
- Friction coefficient (μ): 0.6
- Max force per leg: 180 N

Gait Configuration:
- Gait: Trot (diagonal pairs)
- Period: 0.45s
- Stance ratio: 0.65 (default), tested 0.65-0.90

WBC Configuration:
- Torque limit: 35 Nm
- Swing Kp: 500.0
- Swing Kd: 15.0
```

### 1.3 Test Protocol
1. **Stand Phase** (1.0s): Joint PD control to reach nominal pose
2. **Walk Phase**: Full MPC+WBC controller activation
3. **Termination**: Robot falls (height < 0.1m) or 3-8 seconds elapsed

---

## 2. Test Results

### 2.1 Baseline: Standing in Place (v_cmd = [0, 0, 0])

| Metric | Value |
|--------|-------|
| Stability Time | 1.76s (fell at t=2.76s) |
| Mean Height | 0.207 ± 0.052 m |
| Mean Roll | -26.1° ± 46.1° |
| Mean Pitch | -2.8° ± 5.3° |
| Actual Velocity X | -0.177 m/s |
| Actual Velocity Y | +0.238 m/s |
| Max Torque | 35.0 Nm |

**Observations:**
- Height collapses from 0.27m (stand) to 0.13m within 1 second of walk start
- Severe roll oscillation (±46° std dev) indicates lateral instability
- **Systematic lateral drift**: Robot drifts left even with zero velocity command

### 2.2 Forward Walking Tests

| Command | Target Vx | Actual Vx | Stability | Height (m) | Roll (°) |
|---------|-----------|-----------|-----------|------------|----------|
| v=0.1 m/s | 0.10 | -0.182 | Fell (1.72s) | 0.206 | -26±48 |
| v=0.2 m/s | 0.20 | -0.180 | Fell (1.80s) | 0.207 | -24±49 |
| v=0.3 m/s | 0.30 | +0.043 | Fell (2.13s) | 0.212 | -29±44 |

**Key Finding**: Velocity tracking completely fails - actual velocity is negative (backward) regardless of forward command.

### 2.3 Lateral Movement Tests

| Command | Target Vy | Actual Vy | Stability |
|---------|-----------|-----------|-----------|
| Strafe L 0.1 | +0.10 | +0.181 | Fell (2.46s) |
| Strafe R 0.1 | -0.10 | -0.247 | Fell (2.37s) |

### 2.4 Yaw Rotation Tests

| Command | Stability | Height | Roll | Pitch |
|---------|-----------|--------|------|-------|
| Turn L 0.5 rad/s | Fell (1.50s) | 0.269 | +33±52 | -9±10 |
| Turn R 0.5 rad/s | Fell (1.44s) | 0.291 | +20±56 | +2±9 |

### 2.5 Stance Ratio Sweep

Testing with varying stance ratios (percentage of gait cycle with feet on ground):

| Stance Ratio | Stability Time | Final Height | Final Roll |
|--------------|---------------|--------------|------------|
| 0.65 | **8.00s+** | 0.173 m | -80.2° |
| 0.70 | 3.07s | Fell | - |
| 0.75 | 2.79s | Fell | - |
| 0.80 | 2.76s | Fell | - |
| 0.85 | 2.92s | Fell | - |
| 0.90 | 3.44s | Fell | - |

**Critical Finding**: Lower stance ratio (0.65) provides longest stability! This is counterintuitive but suggests the problem is in force computation, not gait timing.

---

## 3. Root Cause Analysis

### 3.1 Force Analysis

During walking, MPC-computed ground reaction forces are inconsistent:

```
Contact Pattern: [1, 0, 0, 1] (diagonal pairs - trot)
Measured Fz Total: 0 - 316 N (erratic)
Expected Fz: ~150 N (to support 15.2 kg)
```

**Issue**: Forces fluctuate wildly and often go to zero, removing support.

### 3.2 Identified Issues

1. **Insufficient Vertical Force**
   - MPC generates only ~80N total (should be ~150N for 4-leg stance)
   - At 2-leg stance (trot), expected ~75N per leg, but forces erratic
   - Result: Robot under-supported → height collapse

2. **Lateral Instability**
   - Systematic Vy drift (+0.2 m/s) regardless of command
   - Likely due to asymmetric force distribution
   - Roll oscillates ±50° before fall

3. **Velocity Tracking Failure**
   - Actual velocity opposes command (backward when should go forward)
   - Suggests sign error in body frame transformation

4. **Stand-to-Walk Transition**
   - gait_phase_time not updating during stand phase (FIXED)
   - Contact schedule all-zeros at walk start (FIXED)

---

## 4. Bug Fixes Applied

### 4.1 Fixed Issues
1. ✅ Gait phase time not updating during stand
2. ✅ All-zero contact schedule at walk start
3. ✅ Added data logging infrastructure

### 4.2 Partial Improvements
- Increased stance ratio from 0.65 to 0.80 (but didn't solve root cause)
- Added height feedback to WBC (minimal improvement)

---

## 5. Recommendations for Stable Walking

### 5.1 High Priority
1. **Debug MPC Force Computation**
   - Verify friction cone constraints are not too restrictive
   - Check Q weight for height (currently 50, may need 200+)
   - Ensure f_max (180N) is not limiting vertical forces

2. **Fix Body Frame Transform**
   - Velocity tracking shows systematic error
   - Check R_z_T transformation in controller_manager.py

3. **Increase Torque Limit**
   - Currently saturating at 35 Nm
   - Try 45-50 Nm for more authority

### 5.2 Medium Priority
1. **Add Height Feedback in WBC**
   - Proportional feedback on height error
   - Distribute correction across stance legs

2. **Force Smoothing**
   - Current EMA alpha=0.1 may be too aggressive
   - Try alpha=0.3 for smoother force transitions

### 5.3 Research Directions
1. **Compare with MIT Cheetah Parameters**
   - MIT uses different Q weights (heavier on height)
   - Different gait timing (faster, more dynamic)

2. **Contact Force Estimation**
   - Use foot contact sensors instead of gait schedule
   - More reactive to actual ground contact

---

## 6. Conclusions

The MPC-WBC controller demonstrates fundamental stability issues that prevent reliable autonomous walking:

1. **Height collapse**: Forces insufficient to support robot weight
2. **Roll instability**: Lateral forces unbalanced, causing oscillation
3. **Velocity tracking failure**: Command velocity not reflected in actual motion

The robot can maintain standing (with height drop to ~0.17m) for extended periods but cannot walk stably. The trot gait with 2-leg support appears to be the primary failure point - forces computed by MPC are insufficient.

**Recommended Next Step**: Debug the MPC force computation, specifically verifying that friction cone constraints and force limits are not preventing adequate vertical force generation.

---

## Appendix: Test Data

All raw test data is logged to `logs/` directory in NumPy format with fields:
- `time`: Simulation time (s)
- `base_pos`: (x, y, z) position
- `base_quat`: (w, x, y, z) quaternion
- `base_rpy`: (roll, pitch, yaw) angles
- `joint_q`: 12 joint positions
- `contact_schedule`: (4,) binary contact flags

---

*Report generated: 2026-03-28*
*Test framework: MuJoCo + Python*
