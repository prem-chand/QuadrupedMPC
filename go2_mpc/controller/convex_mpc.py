import cvxpy as cp
import numpy as np

class ConvexMPC:
    def __init__(self, mass, inertia, prediction_horizon, dt, 
                 Q, R, mu, f_max):
        self.mass = mass
        self.inertia = inertia
        self.horizon = prediction_horizon
        self.dt = dt
        self.mu = mu
        self.f_max = f_max
        self.g = np.array([0, 0, -9.81])

        # --- SCALING (Critical for Solver Stability) ---
        # Scales forces so the solver sees values close to 1.0
        self.force_scale = 1.0 / f_max 
        self.I_inv = np.linalg.inv(self.inertia)

        self._setup_cvxpy_problem(Q, R)

    def _setup_cvxpy_problem(self, Q, R):
        self.param_A = cp.Parameter((12, 12), name="A")
        self.param_B = cp.Parameter((12, 12), name="B") 
        self.param_g = cp.Parameter((12,), name="g")
        self.param_x_init = cp.Parameter((12,), name="x_init")
        self.param_x_ref = cp.Parameter((12, self.horizon + 1), name="x_ref")
        self.param_contact = cp.Parameter((self.horizon, 4), name="contact")

        self.var_x = cp.Variable((12, self.horizon + 1), name="x")
        self.var_u = cp.Variable((12, self.horizon), name="u_scaled")

        cost = 0
        constraints = [self.var_x[:, 0] == self.param_x_init]
        
        # Scale R to match the scaled variable u
        R_scaled = np.array(R) / (self.force_scale**2)

        for k in range(self.horizon):
            # Cost
            error = self.var_x[:, k+1] - self.param_x_ref[:, k+1]
            cost += cp.quad_form(error, Q)
            cost += cp.quad_form(self.var_u[:, k], R_scaled)

            # Dynamics
            constraints.append(
                self.var_x[:, k+1] == self.param_A @ self.var_x[:, k] + 
                                      self.param_B @ self.var_u[:, k] + 
                                      self.param_g
            )

            # Constraints (Applied to scaled forces)
            for i in range(4):
                u_leg = self.var_u[3*i : 3*i+3, k] 
                contact = self.param_contact[k, i]

                # Fz Limits: 0 <= u_z <= 1.0 * contact
                constraints.append(u_leg[2] >= 0)
                constraints.append(u_leg[2] <= 1.0 * contact)

                # Friction Pyramid
                constraints.append(u_leg[0] <= self.mu * u_leg[2])
                constraints.append(u_leg[0] >= -self.mu * u_leg[2])
                constraints.append(u_leg[1] <= self.mu * u_leg[2])
                constraints.append(u_leg[1] >= -self.mu * u_leg[2])

        self.problem = cp.Problem(cp.Minimize(cost), constraints)
        assert self.problem.is_dpp()

    def update_dynamics_matrices(self, psi, foot_positions_body_frame):
        """
        Args:
            foot_positions_body_frame: MUST be in Body Frame!
        """
        cos_yaw = np.cos(psi)
        sin_yaw = np.sin(psi)
        
        # Rotation Matrix (Body -> World)
        R_z = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])

        A = np.eye(12)
        A[0:3, 6:9] = np.eye(3) * self.dt 
        # FIX 1: Use R_z (not Transpose) for angular velocity mapping
        A[3:6, 9:12] = R_z * self.dt 

        B = np.zeros((12, 12))
        
        for i in range(4):
            # FIX 2: Rotate foot from Body to World for Torque Calculation
            # Torque = r_world x F_world. 
            # We have r_body, so we must rotate it.
            r_body = foot_positions_body_frame[i]
            r_world = R_z @ r_body 
            
            r_skew = np.array([
                [0, -r_world[2], r_world[1]],
                [r_world[2], 0, -r_world[0]],
                [-r_world[1], r_world[0], 0]
            ])
            
            # Physics B matrix
            B_f_lin = (np.eye(3) / self.mass) * self.dt
            B_f_ang = (self.I_inv @ r_skew) * self.dt
            
            # Scale B for the solver (so solver sees u ~ 1.0)
            B[6:9, 3*i:3*i+3] = B_f_lin / self.force_scale
            B[9:12, 3*i:3*i+3] = B_f_ang / self.force_scale
            
        g_vec = np.zeros(12)
        g_vec[8] = -9.81 * self.dt
        
        return A, B, g_vec

    def solve(self, current_state, desired_traj, contact_schedule, foot_positions_body):
        psi = current_state[5]
        A, B, g = self.update_dynamics_matrices(psi, foot_positions_body)

        self.param_A.value = A
        self.param_B.value = B
        self.param_g.value = g
        self.param_x_init.value = current_state
        self.param_x_ref.value = desired_traj
        self.param_contact.value = contact_schedule

        # CLARABEL with warm_start=False (Robust)
        try:
            self.problem.solve(solver=cp.CLARABEL, warm_start=False, verbose=False)
        except cp.SolverError:
            return np.zeros(12)

        if self.problem.status not in ["optimal", "optimal_inaccurate"]:
            return np.zeros(12)

        # Un-scale forces back to Newtons
        return self.var_u[:, 0].value / self.force_scale