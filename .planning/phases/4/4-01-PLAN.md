---
phase: 4-state-estimation
plan: 01
type: execute
wave: 1
depends_on: []
files_modified: []
autonomous: true
requirements: [EST-01, EST-02, EST-03]

must_haves:
  truths:
    - "LKF estimates base position and velocity from IMU + encoder data"
    - "Fused orientation from gyro has < 1° drift over 10s"
    - "Contact-embedded velocity integration reduces position drift"
    - "StateEstimator returns smoothed state for controller"
  artifacts:
    - path: "go2_mpc/controller/kalman_filter.py"
      provides: "Linear Kalman Filter for state estimation"
      min_lines: 200
      exports: ["class LinearKalmanFilter", "class OrientationFilter"]
    - path: "go2_mpc/controller/state_estimator.py"
      provides: "Updated StateEstimator using LKF"
      min_lines: 100
  key_links:
    - from: "go2_mpc/controller/kalman_filter.py"
      to: "go2_mpc/core/state.py"
      via: "returns BaseState dataclass"
      pattern: "BaseState(position.*orientation.*linear_velocity)"
    - from: "go2_mpc/controller/state_estimator.py"
      to: "go2_mpc/controller/kalman_filter.py"
      via: "composition"
      pattern: "self.lkf = LinearKalmanFilter"
---

<objective>
Implement Linear Kalman Filter (LKF) for robust state estimation matching MIT Cheetah, replacing the current pass-through StateEstimator.

Purpose: Reduce sensor noise and drift for reliable MPC/WBC operation. Match MIT Cheetah's state estimation approach.
Output: kalman_filter.py with LKF + OrientationFilter, updated StateEstimator
</objective>

<context>
@go2_mpc/core/state.py         # BaseState, JointState, State dataclasses
@go2_mpc/core/robot.py         # Robot ABC - provides raw sensor data
@go2_mpc/controller/state_estimator.py  # Current implementation (pass-through)
@go2_mpc/config/config.py      # Config dataclasses for parameters

# MIT Cheetah LKF Reference:
# - State: [pos(3), vel(3), rpy(3)] = 9 states OR extended with gyro bias
# - IMU provides: accelerometer (a_B), gyroscope (ω_B)
# - Prediction: integrate velocity → position, use gyro for orientation
# - Update: fuse encoder-derived velocity (contact-embedded) with IMU
# - Complementary filter: high-pass gyro + low-pass accelerometer for orientation
</context>

<interfaces>
<!-- Key interfaces for implementation -->

From go2_mpc/core/state.py:
```python
@dataclass
class BaseState:
    position: np.ndarray        # (3,) world frame, m
    orientation: np.ndarray    # (4,) quaternion [w, x, y, z]
    linear_velocity: np.ndarray # (3,) world frame, m/s
    angular_velocity: np.ndarray # (3,) body frame, rad/s
    
    # Cached derived
    rotation_matrix: np.ndarray  # (3, 3) R_WB
    roll, pitch, yaw: float     # Euler angles (rad)
    
    def to_mpc_vector(self) -> np.ndarray:  # (12,)
```

From go2_mpc/core/robot.py:
```python
class Robot(ABC):
    def get_base_pose(self) -> tuple[np.ndarray, np.ndarray]:
        # Returns (position (3,), quaternion (4,))
    
    def get_base_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        # Returns (linear_velocity_world (3,), angular_velocity_body (3,))
    
    def get_joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        # Returns (joint_positions (12,), joint_velocities (12,))
    
    def get_foot_positions_world(self) -> list[np.ndarray]:
        # Returns list of 4 foot positions (3,) in world frame
```

From go2_mpc/controller/state_estimator.py (current):
```python
class StateEstimator:
    def __init__(self, robot): ...
    
    def estimate(self) -> tuple[State, np.ndarray]:
        # Current: just passes through robot data
        # Need to wrap with LKF filtering
```
</interfaces>

<tasks>

<task type="auto">
  <name>Task 1: Create LinearKalmanFilter class for base state estimation</name>
  <files>go2_mpc/controller/kalman_filter.py</files>
  <action>
    Create LinearKalmanFilter class implementing:
    
    1. State vector: x = [pos(3), vel(3), rpy(3), gyro_bias(3)] = 12 states
    2. Prediction step:
       - State transition: constant velocity model
       - Process noise: Q (tunable, ~0.01 for position, 0.1 for velocity)
       - Predict state and covariance: x_pred = F @ x, P_pred = F @ P @ F.T + Q
    3. Update step (contact-embedded):
       - When foot in stance: use foot position delta for velocity update
       - Measurement: encoder-derived velocity from foot positions
       - Measurement noise: R (tunable, ~0.1)
       - Kalman gain: K = P @ H.T @ (H @ P @ H.T + R)^-1
       - Update: x = x + K @ (z - H @ x), P = (I - K @ H) @ P
    
    Parameters (class arguments or config):
    - dt: timestep (typically 0.001 for prediction at 1kHz)
    - process_noise: dict with 'pos', 'vel', 'orientation', 'gyro_bias' scales
    - measurement_noise: dict with 'position', 'velocity' scales
    
    Use numpy for all matrix operations. Handle covariance propagation carefully.
  </action>
  <verify>
    <automated>python -c "
import numpy as np
from go2_mpc.controller.kalman_filter import LinearKalmanFilter

lkf = LinearKalmanFilter(dt=0.001)
# Test prediction
x = lkf.predict(dt=0.001)
assert x.shape == (12,), f'Expected state shape (12,), got {x.shape}'
# Test update
z = np.zeros(6)  # [pos(3), vel(3)] measurement
x_updated = lkf.update(z, contact_mask=np.array([1, 1, 1, 1]))
assert x_updated.shape == (12,), f'Updated state shape mismatch'
print('LKF basic test: PASSED')
"</automated>
  </verify>
  <done>LinearKalmanFilter class created with predict/update methods</done>
</task>

<task type="auto">
  <name>Task 2: Create OrientationFilter for IMU gyro fusion</name>
  <files>go2_mpc/controller/kalman_filter.py</files>
  <action>
    Create OrientationFilter class implementing complementary filter:
    
    1. State: quaternion q (4,) representing orientation
    2. Prediction (gyro integration):
       - q_pred = q ⊗ q(ω*dt)  (quaternion multiplication)
       - Propagate covariance for uncertainty
    3. Update (accelerometer correction):
       - When stationary: use accelerometer for gravity vector
       - Compute error between measured gravity direction and expected
       - Apply correction: q = q ⊗ q(error_axis * gain)
       - Gain α ~ 0.98-0.99 (high-pass on gyro, low-pass on accel)
    
    Key methods:
    - predict(gyro: np.ndarray, dt: float) -> quaternion
    - update(accel: np.ndarray) -> quaternion  
    - fused() -> quaternion (after both predict and update)
    
    Implement quaternion integration using first-order Euler:
    q_dot = 0.5 * Ω(ω) @ q  where Ω is the quaternion derivative matrix
    
    Handle quaternion normalization after each step.
  </action>
  <verify>
    <python -c "
import numpy as np
from go2_mpc.controller.kalman_filter import OrientationFilter

of = OrientationFilter()
# Test gyro integration
q = of.predict(np.array([0.0, 0.0, 0.1]), dt=0.001)
assert np.abs(np.linalg.norm(q) - 1.0) < 1e-6, 'Quaternion not normalized'
# Test accelerometer correction (gravity pointing down in body frame)
q = of.update(np.array([0.0, 0.0, 9.81]))  # gravity in z
print('OrientationFilter basic test: PASSED')
"</python>
  </verify>
  <done>OrientationFilter class with gyro prediction and accel correction</done>
</task>

<task type="auto">
  <name>Task 3: Integrate LKF with StateEstimator</name>
  <files>go2_mpc/controller/state_estimator.py</files>
  <action>
    Update StateEstimator to use LKF:
    
    1. In __init__:
       - Create self._lkf = LinearKalmanFilter(dt=sim_dt)
       - Create self._orientation_filter = OrientationFilter()
       - Store last foot positions for velocity computation
       - Initialize filter state from first robot measurement
    
    2. In estimate():
       a. Get raw sensor data from robot:
          - pos_raw, quat_raw = robot.get_base_pose()
          - lin_vel_raw, ang_vel_raw = robot.get_base_velocity()
          - foot_pos_world = robot.get_foot_positions_world()
       
       b. Run orientation filter:
          - orientation_filter.predict(ang_vel_raw, dt)
          - If stationary (all feet in stance): orientation_filter.update(accel)
          - Get fused quaternion
       
       c. Compute contact-embedded velocity:
          - Track foot positions over time
          - When foot in stance: velocity = (pos_current - pos_prev) / dt
          - Build measurement z = [pos_measured(3), vel_embedded(3)]
       
       d. Run LKF update with contact mask:
          - contact_mask = gait_scheduler.get_contact() or compute from forces
          - Pass to lkf.update(z, contact_mask)
       
       e. Return filtered State and foot positions
    
    3. Add reset() method to reinitialize filters
    
    Keep the same interface: estimate() returns (State, foot_pos_rel)
  </action>
  <verify>
    <automated>python -c "
import numpy as np
from unittest.mock import Mock

# Mock robot for testing
mock_robot = Mock()
mock_robot.get_base_pose.return_value = (np.array([0., 0., 0.3]), np.array([1., 0., 0., 0.]))
mock_robot.get_base_velocity.return_value = (np.array([0., 0., 0.]), np.array([0., 0., 0.]))
mock_robot.get_joint_state.return_value = (np.zeros(12), np.zeros(12))
mock_robot.get_foot_positions_world.return_value = [np.array([0.2, 0.15, 0.]) for _ in range(4)]

from go2_mpc.controller.state_estimator import StateEstimator

se = StateEstimator(mock_robot)
state, foot_pos = se.estimate()

assert state.base.position.shape == (3,), 'Position shape mismatch'
assert state.base.orientation.shape == (4,), 'Orientation shape mismatch'
assert foot_pos.shape == (4, 3), 'Foot pos shape mismatch'
print('StateEstimator integration test: PASSED')
"</automated>
  </verify>
  <done>StateEstimator uses LKF for state filtering while maintaining same API</done>
</task>

</tasks>

<verification>
- [ ] LinearKalmanFilter implements 12-state (pos, vel, rpy, gyro_bias) LKF
- [ ] OrientationFilter fuses gyro + accelerometer with complementary filter
- [ ] StateEstimator.estimate() returns filtered State and foot_pos_rel
- [ ] Contact-embedded velocity reduces drift during stance
- [ ] Orientation drift < 1° over 10s when stationary
- [ ] Compatible with existing controller stack (same return types)
</verification>

<success_criteria>
Complete LKF state estimation system. Files: kalman_filter.py (~200 lines), state_estimator.py updated (~100 lines). Filters reduce noise and drift while maintaining controller compatibility.
</success_criteria>

<output>
After completion, create `.planning/phases/4-state-estimation/4-01-SUMMARY.md`
</output>
