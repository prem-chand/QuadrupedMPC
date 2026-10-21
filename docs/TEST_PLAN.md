# QuadrupedMPC Walking Quality Test System - Plan

## Objective

Build a comprehensive, verifiable test system for evaluating the MPC-WBC controller on the Unitree Go2 robot. The system enables systematic assessment of walking quality across various conditions.

---

## Test Architecture

```
tests/
├── test_suite.py          # Main test runner (interactive menu)
├── test_config.py         # Test parameters & thresholds
├── scenarios/
│   ├── flat_ground.py     # Flat terrain tests
│   ├── slopes.py          # Slope tests
│   ├── stairs.py          # Stair climbing tests
│   ├── perturbations.py   # Push recovery tests
│   └── gait_transitions.py # Gait pattern switching
├── metrics/
│   ├── collector.py       # Data collection utilities
│   └── evaluator.py       # Pass/fail evaluation
└── reports/
    └── test_report.py     # Report generation
```

---

## Test Scenarios

### 1. Flat Ground Tests

| Test | Command | Duration | Expected Behavior |
|------|---------|----------|-------------------|
| **Stand** | Stand in place | 5s | Height 0.30-0.35m, roll/pitch < 5°, no fall |
| **Walk Forward 0.3 m/s** | Walk forward | 5s | Stable forward motion, height maintained |
| **Walk Forward 0.5 m/s** | Walk forward | 5s | Stable forward, slight oscillation OK |
| **Walk Forward 0.8 m/s** | Walk forward | 5s | Stable forward, may have oscillations |
| **Walk Backward** | Walk backward | 3s | Stable backward motion |
| **Strafe Left** | Strafe left | 3s | Stable lateral motion |
| **Strafe Right** | Strafe right | 3s | Stable lateral motion |
| **Turn Left (yaw)** | Rotate left | 360° | Smooth rotation, no fall |
| **Turn Right (yaw)** | Rotate right | 360° | Smooth rotation, no fall |

### 2. Slope Tests

| Test | Angle | Direction | Expected Behavior |
|------|-------|-----------|-------------------|
| **Slope Up 10°** | 10° | Forward | Ascend without slipping |
| **Slope Down 10°** | 10° | Forward | Descend without slipping |
| **Slope Up 15°** | 15° | Forward | Ascend (may be unstable) |
| **Side Slope 10°** | 10° | Lateral | Maintain balance |

### 3. Stair Tests

| Test | Step Height | Steps | Expected Behavior |
|------|-------------|-------|-------------------|
| **Small Stairs** | 5cm | 3 | Climb without falling |
| **Medium Stairs** | 10cm | 3 | Climb (may struggle) |
| **Down Stairs** | 5cm | 3 | Descend |

### 4. Perturbation Tests

| Test | Type | Magnitude | Expected Behavior |
|------|------|-----------|-------------------|
| **Push Forward** | Impulse | 5 N·s | Recover within 0.5s |
| **Push Lateral** | Impulse | 3 N·s | Recover within 0.5s |
| **Push Backward** | Impulse | 5 N·s | Recover within 0.5s |

### 5. Gait Transition Tests

| Test | From | To | Expected Behavior |
|------|------|----|-------------------|
| **Stand → Trot** | Stand | Trot | Smooth transition |
| **Trot → Stand** | Trot | Stand | Smooth stop |
| **Trot → Bound** | Trot | Bound | Gait switch OK |
| **Bound → Trot** | Bound | Trot | Gait switch OK |

---

## Metrics Collected

| Metric | Description | Unit |
|--------|-------------|------|
| `height_mean` | Average body height | m |
| `height_std` | Height standard deviation | m |
| `roll_mean` | Average roll angle | rad |
| `roll_max` | Maximum absolute roll | rad |
| `pitch_mean` | Average pitch angle | rad |
| `pitch_max` | Maximum absolute pitch | rad |
| `yaw_rate_mean` | Average yaw rate | rad/s |
| `velocity_x_mean` | Average forward velocity | m/s |
| `velocity_error` | Tracking error vs cmd | m/s |
| `torque_max` | Maximum torque applied | Nm |
| `fall_count` | Number of falls | count |
| `step_count` | Total steps taken | count |
| `contact_violations` | Impossible contacts | count |

---

## Pass/Fail Criteria

| Metric | Pass Threshold | Fail Condition |
|--------|----------------|-----------------|
| `fall_count` | 0 | Any fall |
| `height_mean` | 0.25 - 0.38 m | Outside range |
| `roll_max` | < 45° (< 0.78 rad) | Exceeds threshold |
| `pitch_max` | < 45° (< 0.78 rad) | Exceeds threshold |
| `velocity_error` | < 0.2 m/s | Large tracking error |
| `contact_violations` | 0 | Any violation |

---

## Implementation Plan

### Phase 1: Core Infrastructure (Priority: HIGH)

1. **test_config.py** - Define test parameters, scenarios, thresholds
2. **metrics/collector.py** - Data collection during simulation
3. **metrics/evaluator.py** - Pass/fail evaluation logic

### Phase 2: Scenario Implementation (Priority: HIGH)

4. **scenarios/flat_ground.py** - Basic walking tests
5. **scenarios/slopes.py** - Slope tests
6. **scenarios/stairs.py** - Stair tests

### Phase 3: Advanced Scenarios (Priority: MEDIUM)

7. **scenarios/perturbations.py** - Push recovery
8. **scenarios/gait_transitions.py** - Gait switching

### Phase 4: Test Runner (Priority: HIGH)

9. **test_suite.py** - Interactive menu system
10. **reports/test_report.py** - Result logging

---

## File Structure

```
QuadrupedMPC/
├── tests/
│   ├── __init__.py
│   ├── test_suite.py           # Main runner
│   ├── test_config.py          # Parameters
│   ├── scenarios/
│   │   ├── __init__.py
│   │   ├── base.py             # Base test class
│   │   ├── flat_ground.py
│   │   ├── slopes.py
│   │   ├── stairs.py
│   │   ├── perturbations.py
│   │   └── gait_transitions.py
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── collector.py
│   │   └── evaluator.py
│   └── reports/
│       ├── __init__.py
│       └── test_report.py
├── main.py                      # Existing (for reference)
└── docs/
    └── TEST_PLAN.md           # This file
```

---

## Usage

```bash
# Run interactive test menu
python tests/test_suite.py

# Run specific test
python tests/test_suite.py --test flat_walk_forward

# Run all flat ground tests
python tests/test_suite.py --category flat

# Run batch with auto-evaluation
python tests/test_suite.py --batch --report
```

---

## Timeline

| Phase | Estimated Time |
|-------|----------------|
| Phase 1: Core | 30 min |
| Phase 2: Scenarios | 45 min |
| Phase 3: Advanced | 30 min |
| Phase 4: Runner | 30 min |
| **Total** | **~2 hours** |

---

## Acceptance Criteria

- [ ] Test system runs without errors
- [ ] Each scenario executes and collects metrics
- [ ] Pass/fail evaluation works correctly
- [ ] Interactive menu allows test selection
- [ ] Results are logged to file
- [ ] Tests are repeatable
