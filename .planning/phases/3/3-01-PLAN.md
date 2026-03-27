---
phase: 3-batched-gpu-mpc
plan: 01
type: execute
wave: 1
depends_on: []
files_modified: []
autonomous: true
requirements: [BATCH-01, BATCH-02, BATCH-03, BATCH-04, BATCH-05]

must_haves:
  truths:
    - "Controller accepts batched state input with shape (num_envs, state_dim)"
    - "WBC computes batched torques with shape (num_envs, 12)"
    - "GPU-accelerated QP solver handles batched problems efficiently"
    - "Solver handles infeasible batch elements gracefully"
    - "NaN/Inf detection triggers CPU fallback without crashing"
  artifacts:
    - path: "go2_mpc/controller/batched_mpc.py"
      provides: "Batched convex MPC with Torch-based GPU solver"
      min_lines: 250
      exports: ["class BatchedConvexMPC"]
    - path: "go2_mpc/controller/batched_wbc.py"
      provides: "Batched whole-body controller"
      min_lines: 80
      exports: ["class BatchedWBC"]
    - path: "go2_mpc/controller/gpu_qp_solver.py"
      provides: "GPU-accelerated batched QP solver"
      min_lines: 150
      exports: ["class BatchedQPSolver"]
  key_links:
    - from: "go2_mpc/controller/batched_mpc.py"
      to: "go2_mpc/controller/gpu_qp_solver.py"
      via: "solve() method call"
      pattern: "self.solver.solve_batch(H, f, A_eq, ...)"
    - from: "go2_mpc/controller/batched_wbc.py"
      to: "go2_mpc/controller/gpu_qp_solver.py"
      via: "Jacobian transpose on GPU"
      pattern: "J_leg.T @ forces"
---

<objective>
Implement GPU-accelerated batched MPC and WBC for parallel environment simulation.

Purpose: Enable batched MPC solving across 1000+ parallel environments for RL training augmentation.
Output: Batched MPC solver, batched WBC, and batched controller interface.
</objective>

<context>
@go2_mpc/controller/convex_mpc.py    # Single-env MPC formulation (reference)
@go2_mpc/controller/wbc.py          # Single-env WBC (reference)
@go2_mpc/controller/cvxpy_solver.py # Current CPU solver (for fallback)
@go2_mpc/config/config.py            # MPC parameters

# Research context (from research/STACK.md):
# - ReLU-QP: GPU batched solver, 142 stars, ICRA 2024
# - Custom Torch: Use torch.linalg for unconstrained or custom ADMM
# - OSQP: Differentiable but CPU-only

# Recommended approach: Custom Torch batched QP
# Rationale:
#   1. No external dependencies (ReLU-QP requires CUDA compilation)
#   2. Full control over solver structure (MPC has specific QP form)
#   3. PyTorch already available in stack
#   4. Can implement custom infeasibility handling per batch element
#   5. Flexibility for NaN/Inf detection and graceful fallback
</context>

<interfaces>
<!-- Key interfaces from existing code that batched versions must match -->

From convex_mpc.py (single-env):
```python
class ConvexMPC:
    def solve(self, state, x_ref, contact_schedule, foot_positions_body):
        # Returns: (12,) optimal forces
```

From wbc.py (single-env):
```python
class WholeBodyController:
    def compute_torques(self, foot_forces, gravity_comp=True):
        # foot_forces: List of 4 arrays (3,) 
        # Returns: (12,) joint torques
```

Expected batched interface:
```python
class BatchedConvexMPC:
    def solve_batch(self, states, x_refs, contact_schedules, foot_positions_body):
        """Solve MPC for batch of environments.
        
        Args:
            states: (num_envs, 12) batched state vectors
            x_refs: (num_envs, 12, N+1) reference trajectories
            contact_schedules: (num_envs, N, 4) contact flags
            foot_positions_body: (num_envs, 4, 3) foot positions
            
        Returns:
            forces: (num_envs, 12) optimal forces per env
        """

class BatchedWBC:
    def compute_torques_batch(self, foot_forces, gravity_comp=True):
        """Compute torques for batch.
        
        Args:
            foot_forces: (num_envs, 4, 3) GRFs
            Returns: (num_envs, 12) torques
        """
```
</interfaces>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Implement BatchedQPSolver with GPU acceleration</name>
  <files>go2_mpc/controller/gpu_qp_solver.py</files>
  <behavior>
    - Test 1: solve_batch accepts (num_envs, n, n) H, (num_envs, n) f, returns (num_envs, n) solution
    - Test 2: Handles varying contact schedules per env (some stance, some swing)
    - Test 3: Detects NaN/Inf in solution and marks those indices
    - Test 4: Falls back to previous solution for infeasible/Nan batch elements
    - Test 5: Returns zero forces for swing-only legs
  </behavior>
  <action>
    Implement BatchedQPSolver class:
    
    1. __init__(self, num_envs, n_vars, device='cuda'):
       - Store batch size, problem dimensions
       - Initialize solution buffer on GPU
       - Initialize previous_solution buffer for fallback
    
    2. solve_batch(H, f, A_eq=None, b_eq=None, A_ineq=None, b_ineq=None):
       - H: (num_envs, n, n) or (n, n) broadcast
       - f: (num_envs, n)
       - Uses ADMM algorithm for batched QP:
         * Warm-start from previous solution
         * Iteratively refine until convergence or max_iter
       - Returns: (num_envs, n) solution, (num_envs,) status flags
    
    3. NaN/Inf detection:
       - After solve, check torch.isnan(solution) | torch.isinf(solution)
       - Mark problematic batch indices
       - Replace with previous solution buffer values
    
    4. Infeasibility handling:
       - Track convergence per batch element
       - If element not converged after max_iter, use previous solution
       - Return status flags: 0=solved, 1=infeasible, 2=nan
    
    NOTE: For the convex MPC QP structure (positive definite H, equality + box constraints),
    use a simplified ADMM that exploits the problem structure. The friction cone
    constraints can be handled via projection.
  </action>
  <verify>
    <automated>pytest tests/test_gpu_qp_solver.py -x</automated>
  </verify>
  <done>BatchedQPSolver solves 1000+ QPs in parallel on GPU, handles NaN/Inf gracefully</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Implement BatchedConvexMPC</name>
  <files>go2_mpc/controller/batched_mpc.py</files>
  <behavior>
    - Test 1: solve_batch with (num_envs=256, 12) states returns (256, 12) forces
    - Test 2: Different contact schedules per env produce different forces
    - Test 3: Foot positions correctly transformed to body frame per env
    - Test 4: Returns zero forces for swing legs (contact=0)
    - Test 5: Output matches single-env ConvexMPC when batch size=1
  </behavior>
  <action>
    Implement BatchedConvexMPC class:
    
    1. __init__(self, mass, inertia, horizon, dt, Q, R, mu, f_max, solver, num_envs, device='cuda'):
       - Store all MPC parameters
       - Convert numpy arrays to torch tensors on GPU
       - Store solver instance
    
    2. build_qp_batch(states, x_refs, contact_schedules, foot_positions_body):
       - Vectorized QP matrix construction for all environments
       - Uses torch operations instead of numpy
       - Returns batched H, f, A_eq, b_eq, A_ineq, b_ineq
    
    3. solve_batch(states, x_refs, contact_schedules, foot_positions_body):
       - Build batched QP matrices
       - Call solver.solve_batch()
       - Extract first control action from each solution
       - Undo force scaling: forces = u0 / force_scale
       - Returns: (num_envs, 12) forces
    
    Key differences from single-env:
    - All loops vectorized with torch operations
    - Dynamics matrices A, B computed per-env (yaw varies)
    - Use torch.scatter or masking for contact-dependent constraints
  </action>
  <verify>
    <automated>pytest tests/test_batched_mpc.py -x</automated>
  </verify>
  <done>BatchedConvexMPC solves MPC for 256+ environments in parallel</done>
</task>

<task type="auto" tdd="true">
  <name>Task 3: Implement BatchedWBC</name>
  <files>go2_mpc/controller/batched_wbc.py</files>
  <behavior>
    - Test 1: compute_torques_batch with (num_envs, 4, 3) forces returns (num_envs, 12) torques
    - Test 2: Swing legs (zero force) produce zero torque contribution
    - Test 3: Gravity compensation correctly added per env
    - Test 4: Torque clipping applied per env
    - Test 5: Output matches single-env WBC when batch size=1
  </behavior>
  <action>
    Implement BatchedWBC class:
    
    1. __init__(self, robot_interface, torque_limit=35.0, num_envs=1, device='cuda'):
       - Store robot interface (for Jacobian access)
       - Store parameters
    
    2. compute_torques_batch(foot_forces, gravity_comp=True):
       - foot_forces: (num_envs, 4, 3) GRFs in world frame
       - Returns: (num_envs, 12) joint torques
       
       Implementation:
       1. Get batched Jacobians: (num_envs, 4, 3, 3) from robot
       2. Compute J^T @ F for each leg: tau_leg = J_leg.T @ F_leg
          - J_leg: (num_envs, 4, 3, 3) → transpose to (num_envs, 4, 3, 3)
          - F_leg: (num_envs, 4, 3) 
          - Result: (num_envs, 4, 3)
       3. Add gravity compensation: (num_envs, 4, 3) + (num_envs, 4, 3)
       4. Reshape to (num_envs, 12)
       5. Clip to torque_limit
       
       Note: Robot interface must provide batched Jacobians.
       If not available, implement analytical batched Jacobian in Go2Kinematics.
  </action>
  <verify>
    <automated>pytest tests/test_batched_wbc.py -x</automated>
  </verify>
  <done>BatchedWBC computes torques for 256+ environments in parallel</done>
</task>

</tasks>

<verification>
- [ ] BatchedQPSolver solves 1000+ QPs in parallel on GPU (< 10ms per batch)
- [ ] NaN/Inf detection works, fallback returns previous solution
- [ ] BatchedConvexMPC matches single-env output for batch_size=1
- [ ] BatchedWBC matches single-env output for batch_size=1
- [ ] All components handle device transfer (CPU fallback for problematic batch elements)
</verification>

<success_criteria>
Complete batched MPC stack: BatchedQPSolver + BatchedConvexMPC + BatchedWBC.
All files in go2_mpc/controller/. Tests verify correctness vs single-env baseline.
</success_criteria>

<output>
After completion, create `.planning/phases/3-batched-gpu-mpc/3-01-SUMMARY.md`
</output>
