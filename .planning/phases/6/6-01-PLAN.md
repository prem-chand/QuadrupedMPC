---
phase: 6-balance-controller
plan: 01
type: execute
wave: 1
depends_on: []
files_modified: []
autonomous: true
requirements: [BAL-01, BAL-02]

must_haves:
  truths:
    - "Robot detects push disturbance within 50ms of application"
    - "Balance controller triggers gait switch within 1 control cycle"
    - "Robot recovers from 5N lateral push within 0.5s"
    - "Reactive gait switching engages correctly on disturbance"
    - "Normal trot gait resumes after recovery"
  artifacts:
    - path: "go2_mpc/controller/balance_controller.py"
      provides: "Disturbance detection and reactive gait switching"
      contains: "class BalanceController|detect_disturbance|compute_recovery_action"
    - path: "go2_mpc/config/config.py"
      provides: "Balance controller parameters"
      contains: "balance.*kp|balance.*threshold|balance.*stance_ratio"
    - path: "go2_mpc/controller/controller_manager.py"
      provides: "Integration of balance controller with main controller"
      contains: "balance.*controller|update_gait_params"
  key_links:
    - from: "go2_mpc/controller/balance_controller.py"
      to: "go2_mpc/controller/gait_scheduler.py"
      via: "gait_scheduler.update_gait_params()"
      pattern: "gait.*=.*balance.*gait"
    - from: "go2_mpc/controller/controller_manager.py"
      to: "go2_mpc/controller/balance_controller.py"
      via: "balance_controller.compute_recovery_action()"
      pattern: "balance.*compute|recovery.*action"
---

<objective>
Implement BalanceController for push recovery and reactive gait switching.

Purpose: MIT Cheetah uses balance controller for rapid disturbance rejection. Current controller has no push recovery mechanism. This adds reactive control to recover from external perturbations.
Output: balance_controller.py, updated config.py, integration with controller_manager.py
</objective>

<context>
@go2_mpc/controller/gait_scheduler.py    # Existing: GaitScheduler with update_gait_params()
@go2_mpc/controller/controller_manager.py  # Existing: ControllerCore that orchestrates gait->MPC->WBC
@go2_mpc/core/state.py                    # State class with base orientation/velocity
@go2_mpc/config/config.py                 # Configuration dataclasses
@main.py                                   # Simulation loop

# MIT Cheetah Balance Controller References:
# - Di Carlo et al., IROS 2018: Reactive gait switching on disturbance
# - Bledt et al., ICRA 2018: Continuous proprioceptive state estimation
# - Boston Dynamics: Push recovery via stance leg modulation
</context>

<interfaces>
<!-- Existing interfaces that BalanceController must integrate with -->

From go2_mpc/controller/gait_scheduler.py:
```python
class GaitScheduler:
    def __init__(self, gait_period, stance_ratio, horizon, dt):
        self.offsets = np.array([0.0, 0.5, 0.5, 0.0])  # Trot
    
    def update_gait_params(self, gait_name):
        """Allows dynamic switching of gaits."""
        if gait_name == "trot":
            self.offsets = np.array([0.0, 0.5, 0.5, 0.0])
        elif gait_name == "bound":
            self.offsets = np.array([0.0, 0.0, 0.5, 0.5])
        elif gait_name == "pace":
            self.offsets = np.array([0.0, 0.5, 0.0, 0.5])
    
    def get_contact_schedule(self, current_time):
        """Returns shape (horizon, 4) binary contact schedule"""
```

From go2_mpc/core/state.py:
```python
class BaseState:
    roll, pitch, yaw: float          # Euler angles (rad)
    angular_velocity: np.ndarray       # (3,) world frame
    linear_velocity: np.ndarray       # (3,) world frame
    position: np.ndarray              # (3,) world frame
    
    def to_mpc_vector(self) -> np.ndarray:
        """Returns 12-dim state: [x,y,z, roll,pitch,yaw, vx,vy,vz, wx,wy,wz]"""
```

From go2_mpc/controller/controller_manager.py:
```python
class ControllerCore:
    def compute(self, state, foot_pos_rel, command, controller_state, buffers, robot_interface):
        # Line 73: contact_schedule = self.gait.get_contact_schedule(...)
        # Line 149: forces = self.mpc.solve(state, ref, contact_schedule, foot_pos_body)
```
</interfaces>

<tasks>

<task type="auto">
  <name>Task 1: Create balance_controller.py with disturbance detection</name>
  <files>go2_mpc/controller/balance_controller.py</files>
  <action>
    Create BalanceController class in go2_mpc/controller/balance_controller.py:

    1. Class structure:
       ```python
       class BalanceController:
           def __init__(self, config):
               # Disturbance detection thresholds
               self.roll_threshold = config.get("BALANCE_ROLL_THRESHOLD", 0.15)  # rad
               self.pitch_threshold = config.get("BALANCE_PITCH_THRESHOLD", 0.15)
               self.velocity_threshold = config.get("BALANCE_VEL_THRESHOLD", 0.3)  # m/s
               self.angular_threshold = config.get("BALANCE_ANGULAR_THRESHOLD", 1.0)  # rad/s
               
               # Recovery parameters
               self.recovery_stance_ratio = config.get("BALANCE_RECOVERY_STANCE", 0.75)
               self.normal_stance_ratio = config.get("BALANCE_NORMAL_STANCE", 0.65)
               self.recovery_time = config.get("BALANCE_RECOVERY_TIME", 0.5)  # seconds
               
               # State tracking
               self.disturbance_detected = False
               self.recovery_start_time = 0.0
               self.current_gait = "trot"
               self.target_gait = "trot"
       ```

    2. Disturbance detection method:
       ```python
       def detect_disturbance(self, state, command) -> bool:
           """
           Detect if robot is experiencing external disturbance.
           Checks:
           - Roll deviation from commanded (should be ~0)
           - Pitch deviation from commanded (should be ~0)
           - Lateral velocity spike
           - Angular velocity spike
           """
           # Compute deviations from nominal
           roll_error = abs(state.base.roll)
           pitch_error = abs(state.base.pitch)
           lateral_vel = abs(state.base.linear_velocity[1])  # Y in world
           angular_vel_mag = np.linalg.norm(state.base.angular_velocity)
           
           return (roll_error > self.roll_threshold or
                   pitch_error > self.pitch_threshold or
                   lateral_vel > self.velocity_threshold or
                   angular_vel_mag > self.angular_threshold)
       ```

    3. Recovery action computation:
       ```python
       def compute_recovery_action(self, state, current_time, gait_scheduler):
           """
           Returns: (new_gait_name, modified_stance_ratio)
           
           Logic:
           - If disturbance detected: switch to more stable gait (bound for lateral, trot for forward)
           - Increase stance ratio during recovery
           - After recovery_time, gradually return to normal
           """
           disturbance = self.detect_disturbance(state, None)
           
           if disturbance and not self.disturbance_detected:
               # New disturbance: enter recovery
               self.disturbance_detected = True
               self.recovery_start_time = current_time
               
               # Choose gait based on disturbance direction
               if abs(state.base.linear_velocity[1]) > self.velocity_threshold:
                   self.target_gait = "bound"  # Better lateral stability
               else:
                   self.target_gait = "trot"    # Keep trot for forward
               
               gait_scheduler.update_gait_params(self.target_gait)
               return self.target_gait, self.recovery_stance_ratio
           
           elif self.disturbance_detected:
               # Check if recovery complete
               recovery_elapsed = current_time - self.recovery_start_time
               if recovery_elapsed > self.recovery_time:
                   # Check if stable enough to return to normal
                   if not self.detect_disturbance(state, None):
                       self.disturbance_detected = False
                       self.target_gait = "trot"
                       gait_scheduler.update_gait_params("trot")
                       return "trot", self.normal_stance_ratio
               
               return self.target_gait, self.recovery_stance_ratio
           
           return "trot", self.normal_stance_ratio
       ```

    4. Export the class and any helper functions

    NOTE: Keep this fully simulator-agnostic - only uses State interface.
  </action>
  <verify>
    <automated>python -c "
import numpy as np
from go2_mpc.controller.balance_controller import BalanceController

# Test 1: Can instantiate
config = {
    'BALANCE_ROLL_THRESHOLD': 0.15,
    'BALANCE_PITCH_THRESHOLD': 0.15,
    'BALANCE_VEL_THRESHOLD': 0.3,
    'BALANCE_ANGULAR_THRESHOLD': 1.0,
    'BALANCE_RECOVERY_STANCE': 0.75,
    'BALANCE_NORMAL_STANCE': 0.65,
    'BALANCE_RECOVERY_TIME': 0.5,
}
bc = BalanceController(config)
print('Test 1 - Instantiation: PASSED')

# Test 2: No disturbance detection on nominal state
class MockState:
    class Base:
        roll = 0.0
        pitch = 0.0
        linear_velocity = np.array([0.0, 0.0, 0.0])
        angular_velocity = np.array([0.0, 0.0, 0.0])
    base = Base()

disturbance = bc.detect_disturbance(MockState(), None)
assert disturbance == False, 'Should not detect disturbance on nominal state'
print('Test 2 - No false positives: PASSED')

# Test 3: Disturbance detected on roll
MockState.base.roll = 0.3  # Above threshold
disturbance = bc.detect_disturbance(MockState(), None)
assert disturbance == True, 'Should detect disturbance on high roll'
print('Test 3 - Roll disturbance: PASSED')

print('All balance controller tests: PASSED')
"</automated>
  </verify>
  <done>BalanceController class created with disturbance detection and recovery action computation</done>
</task>

<task type="auto">
  <name>Task 2: Add balance controller config to config.py</name>
<parameter name="files>go2_mpc/config/config.py</files>
  <action>
    Add BalanceConfig dataclass and integrate into SystemConfig in config.py:

    1. Add new dataclass after ControllerConfig:
       ```python
       @dataclass
       class BalanceConfig:
           roll_threshold: float = 0.15        # rad - trigger recovery if exceeded
           pitch_threshold: float = 0.15        # rad
           vel_threshold: float = 0.3          # m/s - lateral velocity trigger
           angular_threshold: float = 1.0      # rad/s - angular vel trigger
           recovery_stance_ratio: float = 0.75  # More stance during recovery
           normal_stance_ratio: float = 0.65    # Normal trot stance ratio
           recovery_time: float = 0.5           # seconds before trying normal gait
           enabled: bool = True                  # Enable/disable balance controller
       ```

    2. Add balance to SystemConfig:
       ```python
       @dataclass
       class SystemConfig:
           simulation: SimulationConfig
           mpc: MPCConfig
           gait: GaitConfig
           controller: ControllerConfig
           balance: BalanceConfig  # NEW
       ```

    3. Add balance to default_config():
       ```python
       balance=BalanceConfig(
           roll_threshold=0.15,
           pitch_threshold=0.15,
           vel_threshold=0.3,
           angular_threshold=1.0,
           recovery_stance_ratio=0.75,
           normal_stance_ratio=0.65,
           recovery_time=0.5,
           enabled=True,
       ),
       ```
  </action>
  <verify>
    <automated>python -c "
from go2_mpc.config.config import default_config, BalanceConfig

cfg = default_config()

# Verify BalanceConfig dataclass exists
assert hasattr(cfg, 'balance'), 'SystemConfig should have balance field'

# Verify default values
assert cfg.balance.roll_threshold == 0.15
assert cfg.balance.pitch_threshold == 0.15
assert cfg.balance.vel_threshold == 0.3
assert cfg.balance.angular_threshold == 1.0
assert cfg.balance.recovery_stance_ratio == 0.75
assert cfg.balance.normal_stance_ratio == 0.65
assert cfg.balance.recovery_time == 0.5
assert cfg.balance.enabled == True

print('Config validation: PASSED')
"</automated>
  </verify>
  <done>BalanceConfig added to config.py with default parameters</done>
</task>

<task type="auto">
  <name>Task 3: Integrate balance controller into ControllerCore</name>
  <files>go2_mpc/controller/controller_manager.py</files>
  <action>
    Integrate BalanceController into ControllerCore:

    1. Import BalanceController at top of file:
       ```python
       from .balance_controller import BalanceController
       ```

    2. Modify ControllerCore.__init__ to accept and store balance controller:
       ```python
       def __init__(self, gait, traj_gen, mpc, wbc, config, balance_controller=None):
           # ... existing init code ...
           self.balance_controller = balance_controller
           if balance_controller is not None:
               self.balance_enabled = True
           else:
               self.balance_enabled = config.get("BALANCE_ENABLED", False)
               if self.balance_enabled:
                   # Create default balance controller
                   self.balance_controller = BalanceController(config)
       ```

    3. Modify ControllerCore.compute to use balance controller:
       - After gait scheduling (around line 73), add:
       ```python
       # --- B. Balance Controller (push recovery) ---
       modified_stance_ratio = None
       if self.balance_enabled and self.balance_controller is not None:
           new_gait, modified_stance_ratio = self.balance_controller.compute_recovery_action(
               state, 
               controller_state.gait_phase_time,
               self.gait
           )
           
           # Override gait's stance ratio if in recovery
           if modified_stance_ratio is not None:
               self.gait.stance_ratio = modified_stance_ratio
       ```

    4. Keep backward compatibility:
       - balance_controller parameter should be optional
       - If not provided, create from config dict

    NOTE: This modifies gait.stance_ratio temporarily during recovery. The gait scheduler uses this value in get_contact_schedule(), so it will automatically generate contacts with higher stance ratio.
  </action>
  <verify>
    <automated>python -c "
import numpy as np
from go2_mpc.controller.balance_controller import BalanceController

# Test gait switching through balance controller
class MockGait:
    def __init__(self):
        self.stance_ratio = 0.65
        self.offsets = np.array([0.0, 0.5, 0.5, 0.0])
        self.current_gait = 'trot'
    
    def update_gait_params(self, gait_name):
        self.current_gait = gait_name
        if gait_name == 'trot':
            self.offsets = np.array([0.0, 0.5, 0.5, 0.0])
        elif gait_name == 'bound':
            self.offsets = np.array([0.0, 0.0, 0.5, 0.5])

# Create balance controller with mock config
config = {
    'BALANCE_ROLL_THRESHOLD': 0.1,  # Lower for testing
    'BALANCE_PITCH_THRESHOLD': 0.15,
    'BALANCE_VEL_THRESHOLD': 0.2,
    'BALANCE_ANGULAR_THRESHOLD': 1.0,
    'BALANCE_RECOVERY_STANCE': 0.75,
    'BALANCE_NORMAL_STANCE': 0.65,
    'BALANCE_RECOVERY_TIME': 0.5,
}
bc = BalanceController(config)

# Simulate disturbance
class MockState:
    class Base:
        roll = 0.2  # Above threshold
        pitch = 0.0
        linear_velocity = np.array([0.0, 0.0, 0.0])
        angular_velocity = np.array([0.0, 0.0, 0.0])
    base = Base()

gait = MockGait()

# First call should trigger recovery
gait_name, stance_ratio = bc.compute_recovery_action(MockState(), 0.0, gait)
assert stance_ratio == 0.75, f'Expected recovery stance ratio 0.75, got {stance_ratio}'
assert gait.current_gait == 'bound', f'Expected bound gait, got {gait.current_gait}'

print('Balance controller integration: PASSED')
"</automated>
  </verify>
  <done>BalanceController integrated into ControllerCore with gait switching</done>
</task>

<task type="auto">
  <name>Task 4: Update main.py to pass balance controller</name>
  <files>main.py</files>
  <action>
    Update main.py to create and pass BalanceController to ControllerCore:

    1. Import BalanceController:
       ```python
       from go2_mpc.controller.balance_controller import BalanceController
       ```

    2. Create BalanceController before ControllerCore:
       ```python
       balance_controller = BalanceController({
           'BALANCE_ROLL_THRESHOLD': cfg.balance.roll_threshold,
           'BALANCE_PITCH_THRESHOLD': cfg.balance.pitch_threshold,
           'BALANCE_VEL_THRESHOLD': cfg.balance.vel_threshold,
           'BALANCE_ANGULAR_THRESHOLD': cfg.balance.angular_threshold,
           'BALANCE_RECOVERY_STANCE': cfg.balance.recovery_stance_ratio,
           'BALANCE_NORMAL_STANCE': cfg.balance.normal_stance_ratio,
           'BALANCE_RECOVERY_TIME': cfg.balance.recovery_time,
       })
       ```

    3. Pass to ControllerCore:
       ```python
       controller = ControllerCore(
           gait=gait,
           traj_gen=traj_gen,
           mpc=mpc,
           wbc=wbc,
           config={...},
           balance_controller=balance_controller,  # NEW
       )
       ```
  </action>
  <verify>
    <automated>python -c "
# Test main.py imports work
import sys
sys.path.insert(0, '.')

from go2_mpc.config.config import default_config
from go2_mpc.controller.balance_controller import BalanceController

cfg = default_config()

# Verify balance config exists and has expected fields
assert hasattr(cfg, 'balance')
assert cfg.balance.enabled == True

# Verify BalanceController can be created from config
bc = BalanceController({
    'BALANCE_ROLL_THRESHOLD': cfg.balance.roll_threshold,
    'BALANCE_PITCH_THRESHOLD': cfg.balance.pitch_threshold,
    'BALANCE_VEL_THRESHOLD': cfg.balance.vel_threshold,
    'BALANCE_ANGULAR_THRESHOLD': cfg.balance.angular_threshold,
    'BALANCE_RECOVERY_STANCE': cfg.balance.recovery_stance_ratio,
    'BALANCE_NORMAL_STANCE': cfg.balance.normal_stance_ratio,
    'BALANCE_RECOVERY_TIME': cfg.balance.recovery_time,
})

print('main.py wiring validation: PASSED')
"</automated>
  </verify>
  <done>main.py updated to create and pass BalanceController</done>
</task>

</tasks>

<verification>
- [ ] BalanceController instantiates with default config
- [ ] Disturbance detection triggers on roll > 0.15 rad
- [ ] Disturbance detection triggers on pitch > 0.15 rad  
- [ ] Disturbance detection triggers on lateral vel > 0.3 m/s
- [ ] Gait switches to "bound" on lateral disturbance
- [ ] Stance ratio increases to 0.75 during recovery
- [ ] Normal gait resumes after recovery_time (0.5s)
- [ ] ControllerCore integrates balance controller
- [ ] main.py passes balance controller to controller
- [ ] Push recovery test: 5N lateral push recovers within 0.5s (manual verification in sim)
</verification>

<success_criteria>
Complete balance controller implementation with push recovery and reactive gait switching. BalanceController detects disturbances via roll/pitch/velocity thresholds. Gait switches to more stable pattern during recovery. Robot recovers from 5N push within 0.5s. Normal trot resumes after recovery.
</success_criteria>

<output>
After completion, create `.planning/phases/6-balance-controller/6-01-SUMMARY.md`
</output>
