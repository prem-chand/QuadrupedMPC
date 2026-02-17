import numpy as np


class ConvexMPC:
    def __init__(
        self,
        mass,
        inertia,
        prediction_horizon,
        dt,
        Q,
        R,
        mu,
        f_max,
        solver,  # generic QPSolver
    ):
        self.mass = mass
        self.inertia = inertia
        self.horizon = prediction_horizon
        self.dt = dt
        self.Q = Q
        self.R = R
        self.mu = mu
        self.f_max = f_max
        self.solver = solver

        self.force_scale = 1.0 / f_max
        self.I_inv = np.linalg.inv(self.inertia)

    # ==========================================================
    # Structured State → Vector
    # ==========================================================

    def _structured_to_vector(self, state):
        x = np.zeros(12)
        x[0:3] = state.base.position
        x[3] = state.base.roll
        x[4] = state.base.pitch
        x[5] = state.base.yaw
        x[6:9] = state.base.linear_velocity
        x[9:12] = state.base.angular_velocity
        return x

    # ==========================================================
    # Dynamics Linearization
    # ==========================================================

    def update_dynamics_matrices(self, yaw, foot_positions_body):

        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        R_z = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw,  cos_yaw, 0],
            [0, 0, 1]
        ])

        A = np.eye(12)
        A[0:3, 6:9] = np.eye(3) * self.dt
        A[3:6, 9:12] = R_z * self.dt

        B = np.zeros((12, 12))

        for i in range(4):
            r_body = foot_positions_body[i]
            r_world = R_z @ r_body

            r_skew = np.array([
                [0, -r_world[2], r_world[1]],
                [r_world[2], 0, -r_world[0]],
                [-r_world[1], r_world[0], 0]
            ])

            B_f_lin = (np.eye(3) / self.mass) * self.dt
            B_f_ang = (self.I_inv @ r_skew) * self.dt

            B[6:9, 3*i:3*i+3] = B_f_lin / self.force_scale
            B[9:12, 3*i:3*i+3] = B_f_ang / self.force_scale

        g = np.zeros(12)
        g[8] = -9.81 * self.dt

        return A, B, g

    # ==========================================================
    # Build QP Matrices (Uncondensed Form)
    # ==========================================================

    def build_qp(self, x0, x_ref, contact_schedule, A, B, g):

        nx = 12
        nu = 12
        N = self.horizon

        # Decision variable z = [x0,x1,...,xN,u0,...,uN-1]
        n_vars = nx*(N+1) + nu*N

        H = np.zeros((n_vars, n_vars))
        f = np.zeros(n_vars)

        # Cost terms
        for k in range(N):
            idx_x = nx*(k+1)
            H[idx_x:idx_x+nx, idx_x:idx_x+nx] += self.Q

            idx_u = nx*(N+1) + nu*k
            H[idx_u:idx_u+nu, idx_u:idx_u+nu] += self.R / (self.force_scale**2)

        # Equality constraints
        A_eq = []
        b_eq = []

        # Initial state
        row = np.zeros((nx, n_vars))
        row[:, 0:nx] = np.eye(nx)
        A_eq.append(row)
        b_eq.append(x0)

        # Dynamics constraints
        for k in range(N):
            row = np.zeros((nx, n_vars))

            idx_xk = nx*k
            idx_xkp1 = nx*(k+1)
            idx_uk = nx*(N+1) + nu*k

            row[:, idx_xkp1:idx_xkp1+nx] = np.eye(nx)
            row[:, idx_xk:idx_xk+nx] = -A
            row[:, idx_uk:idx_uk+nu] = -B

            A_eq.append(row)
            b_eq.append(g)

        A_eq = np.vstack(A_eq)
        b_eq = np.concatenate(b_eq)

        # Inequality constraints (friction + contact)
        A_ineq = []
        b_ineq = []

        for k in range(N):
            for leg in range(4):
                idx_u = nx*(N+1) + nu*k + 3*leg

                # Fz >= 0  →  -Fz <= 0
                row = np.zeros(n_vars)
                row[idx_u+2] = -1
                A_ineq.append(row)
                b_ineq.append(0)

                # Fz <= contact
                row = np.zeros(n_vars)
                row[idx_u+2] = 1
                A_ineq.append(row)
                b_ineq.append(contact_schedule[k, leg])

        A_ineq = np.vstack(A_ineq)
        b_ineq = np.array(b_ineq)

        return H, f, A_eq, b_eq, A_ineq, b_ineq

    # ==========================================================
    # Solve
    # ==========================================================

    def solve(self, state, x_ref, contact_schedule, foot_positions_body):

        x0 = self._structured_to_vector(state)
        yaw = state.base.yaw

        A, B, g = self.update_dynamics_matrices(
            yaw,
            foot_positions_body,
        )

        H, f, A_eq, b_eq, A_ineq, b_ineq = self.build_qp(
            x0, x_ref, contact_schedule, A, B, g
        )

        z = self.solver.solve(H, f, A_eq, b_eq, A_ineq, b_ineq)

        if z is None:
            return np.zeros(12)

        nx = 12
        N = self.horizon
        nu = 12

        idx_u0 = nx*(N+1)
        u0 = z[idx_u0:idx_u0+nu]

        return u0 / self.force_scale
