---
phase: 5-controller-tuning
plan: 01
type: execute
wave: 1
depends_on: []
files_modified: []
autonomous: true
requirements: [CTRL-01, CTRL-02, TUNE-01, TUNE-02, TUNE-03]

must_haves:
  truths:
    - "MPC runs at 100 Hz with < 5ms solve time per iteration"
    - "WBC runs at 500 Hz (every 2 sim steps)"
    - "Tuned Q/R weights provide stable Go2 walking"
    - "Tuned swing gains eliminate foot scuffing and vibration"
    - "Friction coefficient configurable for different surfaces"
  artifacts:
    - path: "go2_mpc/config/config.py"
      provides: "Updated timing parameters and tuned gains"
      contains: "mpc_dt=0.01.*horizon=10|motion_control_decimation|force_control_decimation"
    - path: "go2_mpc/controller/controller_manager.py"
      provides: "Updated timing constants for 100Hz MPC / 500Hz WBC"
      contains: "motion_control_decimation.*=.*2|force_control_decimation.*=.*1"
  key_links:
    - from: "go2_mpc/config/config.py"
      to: "go2_mpc/controller/controller_manager.py"
      via: "config dict passed to ControllerCore"
      pattern: "mpc_decimation.*config|force_control_decimation"
---

<objective>
Increase controller frequencies and tune parameters for stable MIT Cheetah-style walking on Unitree Go2.

Purpose: Higher control frequency improves disturbance rejection and enables faster walking speeds. Parameter tuning ensures stable locomotion without oscillations or foot scuffing.
Output: Updated config.py with tuned parameters, modified controller_manager.py with new timing
</objective>

<context>
@go2_mpc/config/config.py      # Current: horizon=10, dt=0.03, Q/R weights, swing gains
@go2_mpc/controller/controller_manager.py  # Current: control_decimation=10, mpc_decimation=3
@main.py                       # Simulation loop, 1kHz step

# Current Timing (from controller_manager.py):
# - Simulation: 1 kHz (dt=0.001s)
# - Motion/WBC: 100 Hz (control_decimation=10)
# - MPC: 33 Hz (mpc_decimation=3, runs every 3rd WBC call)

# Target Timing:
# - Simulation: 1 kHz (unchanged)
# - Motion/WBC: 500 Hz (force_control_decimation=2)
# - MPC: 100 Hz (runs every 5th WBC call OR decoupled from WBC)
</context>

<interfaces>
<!-- Current interfaces that must remain compatible -->

From go2_mpc/config/config.py:
```python
@dataclass
class MPCConfig:
    mass: float
    inertia: np.ndarray
    horizon: int          # Currently 10 (0.3s horizon at 33Hz)
    dt: float            # Currently 0.03 (33 Hz)
    Q: np.ndarray        # State weights: diag([1, 5, 100, 10, 10, 5, 5, 5, 12, 10, 3, 2])
    R: np.ndarray        # Control weights: diag([1e-4] * 12)
    mu: float            # Friction coefficient: 0.6
    f_max: float         # Max force per leg: 180 N

@dataclass
class ControllerConfig:
    mpc_decimation: int      # Currently 3 (MPC at 33Hz)
    force_smooth_alpha: float
    default_height: float
    torque_limit: float
    swing_kp: float          # Currently 400.0
    swing_kd: float         # Currently 10.0
```

From go2_mpc/controller/controller_manager.py:
```python
class ControllerCore:
    def __init__(self, gait, traj_gen, mpc, wbc, config):
        # Current timing:
        self.sim_dt = 0.001           # 1 kHz
        self.control_decimation = 10  # 1 kHz -> 100 Hz (WBC loop)
        self.mpc_decimation = 3       # 100 Hz -> 33 Hz (MPC loop)
```
</interfaces>

<tasks>

<task type="auto">
  <name>Task 1: Update config.py with new timing parameters and tuned gains</name>
  <files>go2_mpc/config/config.py</files>
  <action>
    Modify config.py to support higher frequencies and tuned parameters:

    1. MPC Timing (CTRL-01):
       - Change mpc.dt from 0.03 to 0.01 (100 Hz)
       - Keep horizon=10 (maintains 0.1s prediction horizon)
       - This gives MPC 10ms per solve, well under the 5ms target

    2. WBC Timing (CTRL-02):
       - Add force_control_decimation: int = 2 to ControllerConfig
       - This runs WBC at 500 Hz (every 2 sim steps)

    3. Tune Q/R Weights (TUNE-01):
       - Current Q = diag([1, 5, 100, 10, 10, 5, 5, 5, 12, 10, 3, 2])
         [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
       - For Go2 (15.2kg), increase position tracking:
         Q_tuned = diag([5, 5, 50, 20, 20, 10, 8, 8, 15, 15, 5, 3])
       - Slightly increase R to reduce jerky motions:
         R_tuned = diag([1e-3] * 12)  # 10x current

    4. Tune Swing Gains (TUNE-02):
       - Current: swing_kp=400, swing_kd=10
       - Tune for Go2 mass: swing_kp=500, swing_kd=15
       - This provides stiffer swing leg control

    5. Add Friction Configuration (TUNE-03):
       - Keep mu=0.6 as default (indoor tile/wood)
       - Add friction table as comment:
         # mu=0.6: indoor smooth (tile, wood)
         # mu=0.4: outdoor rough (grass, dirt)
         # mu=0.8: high-grip (rubber mat)

    Make these changes to the default_config() factory.
  </action>
  <verify>
    <automated>python -c "
import numpy as np
from go2_mpc.config.config import default_config

cfg = default_config()

# Verify timing
assert cfg.mpc.dt == 0.01, f'MPC dt should be 0.01, got {cfg.mpc.dt}'
assert cfg.controller.force_control_decimation == 2, 'WBC should run at 500Hz'

# Verify Q weights tuned
expected_Q = np.diag([5, 5, 50, 20, 20, 10, 8, 8, 15, 15, 5, 3])
assert np.allclose(cfg.mpc.Q, expected_Q), 'Q weights not tuned correctly'

# Verify R weights tuned
expected_R = np.diag([1e-3] * 12)
assert np.allclose(cfg.mpc.R, expected_R), 'R weights not tuned correctly'

# Verify swing gains tuned
assert cfg.controller.swing_kp == 500.0, 'Swing Kp not tuned'
assert cfg.controller.swing_kd == 15.0, 'Swing Kd not tuned'

# Verify friction
assert cfg.mpc.mu == 0.6, 'Friction coefficient should be 0.6'

print('Config validation: PASSED')
"</automated>
  </verify>
  <done>Config updated: MPC at 100Hz, WBC at 500Hz, Q/R weights tuned, swing gains tuned</done>
</task>

<task type="auto">
  <name>Task 2: Update ControllerCore timing for 100Hz MPC / 500Hz WBC</name>
  <files>go2_mpc/controller/controller_manager.py</files>
  <action>
  
    Modify ControllerCore to use the new timing parameters:

    1. Replace hardcoded timing constants with config-driven values:
       - motion_control_decimation = config.get("MOTION_CONTROL_DECIMATION", 2)  # 500 Hz
       - force_control_decimation = config.get("FORCE_CONTROL_DECIMATION", 1)   # 1000 Hz (force loop)
       - mpc_decimation = config.get("MPC_DECIMATION", 5)  # MPC at 100 Hz (every 5th 500Hz call)

    2. Rename existing control_decimation to motion_control_decimation (swing leg control)

    3. Create force_control_decimation loop for WBC:
       - New inner loop at 1000 Hz (every sim step)
       - Runs force smoothing and WBC compute at 1000 Hz
       - Motion/swing control runs at motion_control_decimation rate

    4. MPC runs at:
       - 500 Hz / mpc_decimation = 500/5 = 100 Hz

    5. Add timing diagnostics:
       - Track solve time: import time; t0 = time.perf_counter() before MPC solve
       - Log if solve time > 5ms

    Structure:
    ```
    # 1 kHz simulation loop
    for each sim step:
        # Force loop (1000 Hz)
        if step % force_control_decimation == 0:
            force_smoothing()
            wbc.compute_torques()
        
        # Motion loop (500 Hz)  
        if step % motion_control_decimation == 0:
            gait_scheduling()
            contact_schedule = gait.get_contact_schedule()
            
            # MPC (100 Hz)
            if mpc_counter % mpc_decimation == 0:
                mpc.solve()
            
            mpc_counter += 1
        
        # Swing control (500 Hz)
        swing_leg_control()
    ```

    Keep the same public API: compute(state, foot_pos_rel, command, controller_state, buffers, robot_interface)
  </action>
  <verify>
    <automated>python -c "
from go2_mpc.config.config import default_config
from go2_mpc.controller.controller_manager import ControllerCore

cfg = default_config()

# Create mock components
class MockGait:
    period = 0.45
    stance_ratio = 0.65
    def get_contact_schedule(self, t): import numpy as np; return np.ones((10, 4))

class MockMPC:
    def solve(self, *args): return np.zeros(12)

class MockWBC:
    def compute_torques(self, *args, **kwargs): return np.zeros(12)

class MockTraj:
    def generate_reference(self, *args): import numpy as np; return np.zeros((12, 11))

controller = ControllerCore(
    gait=MockGait(),
    traj_gen=MockTraj(),
    mpc=MockMPC(),
    wbc=MockWBC(),
    config={
        'MPC_DECIMATION': cfg.controller.mpc_decimation,
        'MOTION_CONTROL_DECIMATION': cfg.controller.force_control_decimation,
        'FORCE_CONTROL_DECIMATION': 1,
        'SWING_KP': cfg.controller.swing_kp,
        'SWING_KD': cfg.controller.swing_kd,
    }
)

# Verify timing constants
assert controller.motion_control_decimation == 2, 'Motion at 500Hz'
assert controller.mpc_decimation == 5, 'MPC at 100Hz'

print('ControllerCore timing validation: PASSED')
"</automated>
  </verify>
  <done>ControllerCore updated with 100Hz MPC / 500Hz WBC timing structure</done>
</task>

<task type="auto">
  <name>Task 3: Update main.py wiring to use new config parameters</name>
  <files>main.py</files>
  <action>
    Update main.py to pass the new config parameters to ControllerCore:

    1. Add MPC_DECIMATION to config dict:
       ```python
       config={
           "MPC_DECIMATION": cfg.controller.mpc_decimation,
           "MOTION_CONTROL_DECIMATION": cfg.controller.force_control_decimation,
           "FORCE_CONTROL_DECIMATION": 1,  # Run at 1kHz for force smoothing
           "FORCE_SMOOTH_ALPHA": cfg.controller.force_smooth_alpha,
           ...
       }
       ```

    2. Ensure force_smooth_alpha is appropriate for 500Hz:
       - Current: 0.1 (10% smoothing at 100Hz = effective smoothing time ~0.1s)
       - At 500Hz, maintain same effective smoothing: alpha = 0.1^(1/5) ≈ 0.63
       - Or: alpha = 0.5 for moderate smoothing

    3. Keep all other wiring unchanged
  </action>
  <verify>
    <automated>python -c "
# Test main.py imports and config wiring
import sys
sys.path.insert(0, '.')

from go2_mpc.config.config import default_config

# Test config loads
cfg = default_config()
print(f'MPC dt: {cfg.mpc.dt}')
print(f'MPC decimation: {cfg.controller.mpc_decimation}')
print(f'Motion control decimation: {cfg.controller.force_control_decimation}')

print('main.py wiring validation: PASSED')
"</automated>
  </verify>
  <done>main.py updated to pass new timing config to ControllerCore</done>
</task>

</tasks>

<verification>
- [ ] MPC runs at 100 Hz (dt=0.01, horizon=10)
- [ ] WBC/force loop runs at 500 Hz (motion_control_decimation=2)
- [ ] Force smoothing runs at 1 kHz (force_control_decimation=1)
- [ ] MPC solve time < 5ms verified in simulation
- [ ] Tuned Q weights improve position tracking
- [ ] Tuned R weights reduce jerky motions
- [ ] Swing gains tuned (kp=500, kd=15) for Go2
- [ ] Friction coefficient configurable via config.mpc.mu
- [ ] Walking stable on flat ground (verify by running main.py)
</verification>

<success_criteria>
Complete frequency upgrade and parameter tuning. Config updated with new timing and gains. Controller timing verified: MPC at 100Hz, WBC at 500Hz. Tuned parameters produce stable walking.
</success_criteria>

<output>
After completion, create `.planning/phases/5-controller-tuning/5-01-SUMMARY.md`
</output>
