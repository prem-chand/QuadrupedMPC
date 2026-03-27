---
phase: 2-isaaclab-backend
plan: 02
type: execute
wave: 1
depends_on: []
files_modified: []
autonomous: true
requirements: [ISAAC-08, ROB-02]

must_haves:
  truths:
    - "Single-step controller produces identical torques from IsaacRobot and MujocoRobot within tolerance"
    - "QP solver handles infeasible problems gracefully without crashing"
    - "Fallback solution returned when QP is infeasible"
  artifacts:
    - path: "tests/test_isaac_robot.py"
      provides: "Validation test comparing IsaacLab to MuJoCo"
      min_lines: 50
    - path: "go2_mpc/controller/cvxpy_solver.py"
      provides: "QP infeasibility handling"
      modifies: "solve() method to return fallback"
  key_links:
    - from: "tests/test_isaac_robot.py"
      to: "go2_mpc/core/isaac_robot.py"
      via: "import IsaacRobot"
      pattern: "test.*MujocoRobot.*IsaacRobot"
    - from: "go2_mpc/controller/cvxpy_solver.py"
      to: "go2_mpc/controller/convex_mpc.py"
      via: "solver.solve() call"
      pattern: "forces = solver.solve.*if forces is None"
---

<objective>
Validate IsaacLab backend matches MuJoCo behavior and add QP infeasibility handling.

Purpose: Verify numerical correctness of IsaacRobot implementation; ensure controller robustness when QP solver fails.
Output: Validation test + robustified QP solver
</objective>

<execution_context>
@$HOME/.config/opencode/get-shit-done/workflows/execute-plan.md
@$HOME/.config/opencode/get-shit-done/templates/summary.md
</execution_context>

<context>
@go2_mpc/core/isaac_robot.py     # From Plan 2-01 - IsaacRobot implementation
@go2_mpc/core/mujoco_robot.py   # Reference for validation comparison
@go2_mpc/controller/cvxpy_solver.py  # Current solver - returns None on infeasibility
@go2_mpc/controller/convex_mpc.py    # Uses solver.solve() - needs fallback handling
@go2_mpc/config/config.py            # Torque limits, tolerances

# Validation approach:
# 1. Create test with identical initial state in both simulators
# 2. Run single controller step (same input to ControllerCore)
# 3. Compare output torques - must match within tolerance (e.g., 1e-5)
# 4. Use same Go2Kinematics for both (should be exact match)
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create validation test comparing IsaacRobot to MujocoRobot</name>
  <files>tests/test_isaac_robot.py</files>
  <action>
    Create test_isaac_robot.py with:
    
    1. Fixtures for both MujocoRobot and IsaacRobot with identical initial state
    2. Single controller step test:
       - Extract state from both robots (pose, velocity, joints)
       - Run ControllerCore.compute_torques() with same inputs
       - Compare output torques: assert_allclose(tau_mujoco, tau_isaac, rtol=1e-4)
    
    3. Test all Robot ABC methods return correct shapes:
       - get_base_pose() → (3,) + (4,)
       - get_base_velocity() → (3,) + (3,)
       - get_joint_state() → (12,) + (12,)
       - get_foot_positions_world() → list of 4×(3,)
       - get_leg_jacobian(i) → (3, 3) for each leg
    
    NOTE: This test requires IsaacLab to be installed. Skip if not available (pytest.importorskip).
  </action>
  <verify>
    <automated>pytest tests/test_isaac_robot.py -v --tb=short</automated>
  </verify>
  <done>Test runs and validates IsaacRobot matches MujocoRobot output</done>
</task>

<task type="auto">
  <name>Task 2: Add QP infeasibility handling with fallback</name>
  <files>go2_mpc/controller/cvxpy_solver.py</files>
  <action>
    Modify ClarabelSolver.solve() to handle infeasibility gracefully:
    
    1. Current behavior: returns None on infeasible/unsolved
    2. New behavior: return fallback solution
    
    Fallback strategy options:
    - Option A (recommended): Return zero forces (12,) - safe, robot falls but doesn't explode
    - Option B: Return previous solution (requires storing state)
    - Option C: Relax constraints and re-solve
    
    Implement Option A for simplicity:
    ```python
    if sol.status not in (clarabel.SolverStatus.Solved, clarabel.SolverStatus.AlmostSolved):
        # Infeasible or unsolved - return zero forces as fallback
        return np.zeros(n, dtype=np.float64)
    ```
    
    Also update CVXPYSolver similarly for consistency.
    
    Document in docstring that solver now returns fallback on infeasibility.
  </action>
  <verify>
    <automated>pytest tests/test_controllers.py::TestQPSolver::test_infeasible_returns_none -v</automated>
  </verify>
  <done>QP solver returns fallback (zeros) instead of None on infeasibility</done>
</task>

<task type="auto">
  <name>Task 3: Update ConvexMPC to handle None from solver (if still needed)</name>
<files>go2_mpc/controller/convex_mpc.py</files>
<action>
    Check if convex_mpc.py handles None from solver.solve():
    
    1. Find where solver.solve() is called
    2. If currently checks for None and raises error:
       - Remove error-raising (now solver returns fallback)
       - Or keep None-check but return zero forces as fallback
    
    The solver now returns zeros on infeasibility, so downstream code should work without changes.
    Verify and update if needed.
  </action>
  <verify>
    <automated>pytest tests/test_controllers.py -v -k mpc</automated>
  </verify>
  <done>ConvexMPC handles solver fallback gracefully</done>
</task>

</tasks>

<verification>
- [ ] test_isaac_robot.py created with shape validation tests
- [ ] Single-step controller test compares IsaacRobot to MujocoRobot
- [ ] QP solver returns zeros instead of None on infeasibility
- [ ] ConvexMPC works with solver fallback (no crashes)
</verification>

<success_criteria>
IsaacLab backend validated against MuJoCo (within tolerance). QP solver handles infeasibility without crashing. Controller remains stable with fallback forces.
</success_criteria>

<output>
After completion, create `.planning/phases/2-isaaclab-backend/2-02-SUMMARY.md`
</output>
