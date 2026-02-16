import cvxpy as cp
import numpy as np
from typing import Tuple, Optional

class ConvexMPC:
    def __init__(self, mass, inertia, prediction_horizon, dt, 
                 Q, R, mu, f_max):
        """
        Args:
            mass: Robot mass (kg)
            inertia: Body inertia matrix (3x3)
            prediction_horizon: Steps (N)
            dt: Time step (s)
            Q: State cost weights (12,) or (12,12)
            R: Control cost weights (12,) or (12,12)
            mu: Friction coefficient
            f_max: Maximum force per leg
        """
        self.mass = mass
        self.inertia = inertia
        self.horizon = prediction_horizon
        self.dt = dt
        self.mu = mu
        self.f_max = f_max
        self.g = np.array([0, 0, -9.81])

        # Dimensions
        self.n_state = 12
        self.n_ctrl = 12 

        # --- CVXPY Problem Setup (Compiled ONCE) ---
        self._setup_cvxpy_problem(Q, R)

    def _setup_cvxpy_problem(self, Q, R):
        """
        Sets up the Convex Optimization problem using CVXPY with Parameters for DPP compliance.
        """
        # 1. Parameters (Data we update every loop)
        # Dynamics matrices are now parameters!
        self.param_A = cp.Parameter((12, 12), name="A")
        self.param_B = cp.Parameter((12, 12), name="B")
        self.param_g = cp.Parameter((12,), name="g") # Gravity vector
        
        # References
        self.param_x_init = cp.Parameter((12,), name="x_init")
        self.param_x_ref = cp.Parameter((12, self.horizon + 1), name="x_ref")
        self.param_contact = cp.Parameter((self.horizon, 4), name="contact")

        # 2. Variables (What we solve for)
        self.var_x = cp.Variable((12, self.horizon + 1), name="x")
        self.var_f = cp.Variable((12, self.horizon), name="f")

        # 3. Formulate Constraints & Cost
        cost = 0
        constraints = [self.var_x[:, 0] == self.param_x_init]

        for k in range(self.horizon):
            # --- Cost ---
            # State tracking error
            state_error = self.var_x[:, k+1] - self.param_x_ref[:, k+1]
            cost += cp.quad_form(state_error, Q)
            # Force minimization (Energy/Smoothness)
            cost += cp.quad_form(self.var_f[:, k], R)

            # --- Dynamics Constraint ---
            # x_{k+1} = A * x_k + B * f_k + g
            # Note: We use the *current* A and B for the whole horizon (LTI approx)
            constraints.append(
                self.var_x[:, k+1] == self.param_A @ self.var_x[:, k] + 
                                      self.param_B @ self.var_f[:, k] + 
                                      self.param_g
            )

            # --- Friction & Contact Constraints ---
            for i in range(4): # For each leg
                f_blk = self.var_f[3*i : 3*i+3, k] # [fx, fy, fz]
                contact = self.param_contact[k, i]

                # Normal Force limits
                # If contact=0, max_force=0 -> Force is zeroed out
                constraints.append(f_blk[2] >= 0)
                constraints.append(f_blk[2] <= self.f_max * contact)

                # Friction Pyramid (Linearized Cone) - using linear constraints
                # |fx| <= mu * fz  =>  fx <= mu*fz AND -fx <= mu*fz
                constraints.append(f_blk[0] <= self.mu * f_blk[2])
                constraints.append(f_blk[0] >= -self.mu * f_blk[2])
                constraints.append(f_blk[1] <= self.mu * f_blk[2])
                constraints.append(f_blk[1] >= -self.mu * f_blk[2])

        # 4. Compile
        self.problem = cp.Problem(cp.Minimize(cost), constraints)
        
        # Check if DPP (Disciplined Parametrized Programming) compliant
        # This ensures repeated solves are fast
        assert self.problem.is_dpp()

    def update_dynamics_matrices(self, psi, foot_positions_rel):
        """
        Recomputes A and B based on CURRENT state.
        Args:
            psi: Current Yaw angle (radians)
            foot_positions_rel: (4, 3) Array of foot pos relative to CoM
        """
        # 1. A Matrix (State Transition)
        # Rotation approx for small angles
        cos_yaw = np.cos(psi)
        sin_yaw = np.sin(psi)
        R_z_t = np.array([
            [cos_yaw, sin_yaw, 0],
            [-sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ]) # Transpose of R_z

        A = np.eye(12)
        # Position += Velocity * dt
        A[0:3, 6:9] = np.eye(3) * self.dt 
        # Euler += R_z.T * Omega * dt
        A[3:6, 9:12] = R_z_t * self.dt

        # 2. B Matrix (Control Input)
        B = np.zeros((12, 12))
        I_inv = np.linalg.inv(self.inertia) # Assuming inertia is already World Frame or Body approx
        
        for i in range(4):
            r = foot_positions_rel[i]
            r_skew = np.array([
                [0, -r[2], r[1]],
                [r[2], 0, -r[0]],
                [-r[1], r[0], 0]
            ])
            
            # Linear Accel: F/m * dt
            # Map forces (cols 3i:3i+3) to Linear Vel (rows 6:9)
            B[6:9, 3*i:3*i+3] = (np.eye(3) / self.mass) * self.dt
            
            # Angular Accel: I_inv * (r x F) * dt
            # Map forces to Angular Vel (rows 9:12)
            B[9:12, 3*i:3*i+3] = (I_inv @ r_skew) * self.dt
            
        # 3. Gravity Vector (affine term)
        g_vec = np.zeros(12)
        g_vec[8] = -9.81 * self.dt # Add gravity to Z-velocity
        
        return A, B, g_vec

    def solve(self, current_state: np.ndarray, desired_traj: np.ndarray, 
              contact_schedule: np.ndarray, foot_positions_rel: np.ndarray) -> np.ndarray:
        """
        Solves the MPC problem for the next control step.

        Args:
            current_state: (12,) [x,y,z, r,p,y, vx,vy,vz, wx,wy,wz] - Current robot state
            desired_traj: (12, N+1) - Desired future states
            contact_schedule: (N, 4) - Contact schedule (1=stance, 0=swing)
            foot_positions_rel: (4, 3) - Feet positions relative to CoM

        Returns:
            (12,) Array of ground reaction forces [fx, fy, fz] * 4
        """
        # 1. Update Physics Model
        psi = current_state[5]
        A, B, g = self.update_dynamics_matrices(psi, foot_positions_rel)

        # 2. Update CVXPY Parameters (DPP approach - fast!)
        self.param_A.value = A
        self.param_B.value = B
        self.param_g.value = g
        self.param_x_init.value = current_state
        self.param_x_ref.value = desired_traj
        self.param_contact.value = contact_schedule  # Assume already float

        # 3. Solve (uses pre-compiled problem)
        # Use CLARABEL for better convergence
        self.problem.solve(solver=cp.CLARABEL, verbose=False)

        if self.problem.status not in ["optimal", "optimal_inaccurate"]:
            print(f"[Values] MPC Solve Failed: {self.problem.status}. Returning zero forces.")
            return np.zeros(12)

        return self.var_f[:, 0].value