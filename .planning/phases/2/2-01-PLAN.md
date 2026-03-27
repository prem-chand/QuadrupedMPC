---
phase: 2-isaaclab-backend
plan: 01
type: execute
wave: 1
depends_on: []
files_modified: []
autonomous: true
requirements: [ISAAC-01, ISAAC-02, ISAAC-03, ISAAC-04, ISAAC-07]

must_haves:
  truths:
    - "IsaacRobot class fully implements Robot ABC interface"
    - "get_base_pose() returns position (3,) and quaternion (4,) in world frame"
    - "get_base_velocity() returns linear (3,) and angular (3,) velocity"
    - "get_joint_state() returns joint positions (12,) and velocities (12,)"
    - "set_torques() applies torques to IsaacLab simulation via Articulation API"
  artifacts:
    - path: "go2_mpc/core/isaac_robot.py"
      provides: "IsaacLab-backed Robot implementation"
      min_lines: 150
      exports: ["class IsaacRobot"]
  key_links:
    - from: "go2_mpc/core/isaac_robot.py"
      to: "go2_mpc/core/robot.py"
      via: "class IsaacRobot(Robot)"
      pattern: "implements all @abstractmethod"
    - from: "go2_mpc/core/isaac_robot.py"
      to: "isaaclab.assets.Articulation"
      via: "constructor injection"
      pattern: "def __init__.*Articulation"
---

<objective>
Implement IsaacRobot class that wraps IsaacLab Articulation and fully implements the Robot ABC interface.

Purpose: Enable IsaacLab as an alternative simulation backend to MuJoCo, using the same controller stack.
Output: go2_mpc/core/isaac_robot.py with complete Robot ABC implementation
</objective>

<context>
@go2_mpc/core/robot.py        # Robot ABC interface - MUST implement all abstract methods
@go2_mpc/core/mujoco_robot.py # Reference implementation - follow same patterns
@go2_mpc/kinematics/go2_kinematics.py  # Analytical FK/Jacobians (Phase 1 output)

# IsaacLab API patterns (from research):
# - robot.data.root_pose_w        → (N, 7) [pos(3) + quat(4)]
# - robot.data.root_lin_vel_b     → (N, 3) linear velocity BODY frame
# - robot.data.root_ang_vel_b    → (N, 3) angular velocity BODY frame
# - robot.data.joint_pos          → (N, nj) joint positions
# - robot.data.joint_vel           → (N, nj) joint velocities
# - robot.set_joint_effort_target() → apply torques
# - robot.write_data_to_sim()      → write to PhysX
# - robot.update(dt)                → update internal buffers
# - sim.step()                      → advance simulation
</context>

<interfaces>
<!-- Key interfaces from Robot ABC that IsaacRobot must implement -->

From go2_mpc/core/robot.py:
```python
class Robot(ABC):
    @abstractmethod
    def step(self): ...
    
    @abstractmethod
    def get_time(self) -> float: ...
    
    @abstractmethod
    def set_torques(self, tau: np.ndarray): ...
    
    @abstractmethod
    def get_base_pose(self) -> tuple[np.ndarray, np.ndarray]: ...
    # Returns (position (3,), quaternion (4,)) [w,x,y,z]
    
    @abstractmethod
    def get_base_velocity(self) -> tuple[np.ndarray, np.ndarray]: ...
    # Returns (linear_velocity_world (3,), angular_velocity_body (3,))
    
    @abstractmethod
    def get_joint_state(self) -> tuple[np.ndarray, np.ndarray]: ...
    # Returns (joint_positions (N,), joint_velocities (N,))
    
    @abstractmethod
    def get_foot_positions_world(self) -> list[np.ndarray]: ...
    # Returns list of 4 foot positions (3,) in world frame
    
    @abstractmethod
    def get_foot_jacobian(self, foot_index: int) -> np.ndarray: ...
    # Returns (3, 18) full Jacobian
    
    @abstractmethod
    def get_foot_velocity(self, foot_index: int) -> np.ndarray: ...
    # Returns (3,) foot velocity in world frame
    
    @abstractmethod
    def get_gravity_compensation(self, leg_index: int) -> np.ndarray: ...
    # Returns (3,) bias torques for leg
    
    @abstractmethod
    def get_leg_jacobian(self, foot_index: int) -> np.ndarray: ...
    # Returns (3, 3) leg-local Jacobian
```

IsaacLab Articulation conventions (matching MuJoCo):
- root_pose_w[:, 3:7] is quaternion [w, x, y, z] (scalar-first)
- root_lin_vel_b is BODY frame (not world!)
- root_ang_vel_b is BODY frame (not world!)
- joint ordering matches MuJoCo XML: [FL(3), FR(3), RL(3), RR(3)]
</interfaces>

<tasks>

<task type="auto">
  <name>Task 1: Create IsaacRobot class skeleton with simulation methods</name>
  <files>go2_mpc/core/isaac_robot.py</files>
  <action>
    Create IsaacRobot class that:
    1. Takes `articulation: Articulation` and `sim: SimulationContext` in constructor
    2. Stores Go2Kinematics instance (self._kin) for analytical FK
    3. Implements step() → calls sim.step() then robot.update(dt)
    4. Implements get_time() → returns sim.get_physics_dt() accumulated or sim.current_time
    5. Implements set_torques(tau) → calls robot.set_joint_effort_target(tau) then robot.write_data_to_sim()
    
    NOTE: For single-env use (batch dim = 1), squeeze batch dimension from Articulation data.
  </action>
  <verify>
    <automated>MISSING — Create test_isaac_robot.py after implementation</automated>
  </verify>
  <done>IsaacRobot can be instantiated and stepped without errors</done>
</task>

<task type="auto">
  <name>Task 2: Implement base state and joint state getters</name>
  <files>go2_mpc/core/isaac_robot.py</files>
  <action>
    Implement in IsaacRobot:
    1. get_base_pose() → extract position from root_pose_w[0, :3], quaternion from root_pose_w[0, 3:7]
       - Convert to [w,x,y,z] format if IsaacLab uses [x,y,z,w]
    2. get_base_velocity() → extract root_lin_vel_b[0] (BODY frame!) and root_ang_vel_b[0]
       - CRITICAL: IsaacLab stores angular velocity in BODY frame (like MuJoCo), return as-is
       - For linear: MuJoCo uses WORLD frame, but IsaacLab may use BODY - verify and convert if needed
    3. get_joint_state() → extract robot.data.joint_pos[0], robot.data.joint_vel[0]
    
    FollowMujocoRobot._get_base_state() pattern for extracting R_base from quaternion.
  </action>
  <verify>
    <automated>MISSING — Add to test_isaac_robot.py after implementation</automated>
  </verify>
  <done>get_base_pose, get_base_velocity, get_joint_state return correct shapes and data types</done>
</task>

<task type="auto">
  <name>Task 3: Implement foot kinematics via Go2Kinematics</name>
  <files>go2_mpc/core/isaac_robot.py</files>
  <action>
    Implement remaining Robot ABC methods using Go2Kinematics (same pattern as MujocoRobot):
    
    1. get_foot_positions_world() → call self._kin.foot_position_world for each leg
    2. get_leg_jacobian(foot_index) → call self._kin.leg_jacobian
    3. get_foot_jacobian(foot_index) → call self._kin.full_jacobian
    4. get_foot_velocity(foot_index) → call self._kin.foot_velocity
    5. get_gravity_compensation(leg_index) → call self._kin.gravity_compensation
    
    All methods need p_base, R_base from get_base_pose() and q_joints from get_joint_state().
    Use _get_base_state() helper pattern from MujocoRobot to avoid redundant computation.
  </action>
  <verify>
    <automated>MISSING — Add to test_isaac_robot.py after implementation</automated>
  </verify>
  <done>All 5 foot kinematics methods implemented, use analytical Go2Kinematics</done>
</task>

</tasks>

<verification>
- [ ] IsaacRobot imports Robot ABC and implements all 10 abstract methods
- [ ] Constructor accepts Articulation and SimulationContext
- [ ] set_torques applies torques via IsaacLab API
- [ ] get_base_pose returns (3,) position + (4,) quaternion
- [ ] get_base_velocity returns linear (world) + angular (body)
- [ ] get_joint_state returns (12,) pos + (12,) vel
- [ ] All foot kinematics delegate to Go2Kinematics (no IsaacLab FK used)
</verification>

<success_criteria>
IsaacRobot class complete. File: go2_mpc/core/isaac_robot.py (~150 lines). All Robot ABC methods implemented. Uses analytical Go2Kinematics for feet (same as MujocoRobot).
</success_criteria>

<output>
After completion, create `.planning/phases/2-isaaclab-backend/2-01-SUMMARY.md`
</output>
