"""
Convex Model-Predictive Controller for centroidal dynamics.

Formulation follows Di Carlo et al., "Dynamic Locomotion in the MIT
Cheetah 3 Through Convex Model-Predictive Control" (IROS 2018).

Coordinate frames
-----------------
- **World (W)**: Fixed inertial frame, z-up.  All MPC state quantities
  (position, velocity, angular velocity) and ground reaction forces
  live in this frame.
- **Body (B)**: Attached to the robot trunk.  The body-frame inertia
  tensor ``I_body`` is constant; the world-frame inertia is obtained
  via ``I_W = R_z · I_body · R_z^T`` under the small roll/pitch
  approximation ``R_WB ≈ R_z(yaw)``.
- **Yaw (Y)**: World frame rotated by yaw only.  Foot positions are
  passed in as *yaw-frame* ("body") coordinates and rotated to world
  frame internally.

MPC state vector (12-dim)
-------------------------
::

    x = [p_x, p_y, p_z,  φ, θ, ψ,  v_x, v_y, v_z,  ω_x, ω_y, ω_z]
         ──────────────── ────────── ──────────────── ────────────────
         position (W)     Euler ZYX  linear vel (W)   angular vel (W)
         (m)              (rad)      (m/s)            (rad/s)

Input vector (12-dim)
---------------------
::

    u = [F0_x, F0_y, F0_z, F1_x, ..., F3_z]
         ────────────────────────────────────
         GRFs per leg [FL, FR, RL, RR] in world frame (N)

Forces are internally scaled by ``1/f_max`` for numerical conditioning;
the solver operates on ``ũ = u · force_scale``.

QP formulation (uncondensed / sparse)
-------------------------------------
Decision variable::

    z = [x_0, x_1, …, x_N, ũ_0, …, ũ_{N-1}]
         ─────────────────  ──────────────────
         (N+1)·12 states    N·12 scaled forces

    dim(z) = 12·(N+1) + 12·N = 12·(2N+1)

Cost::

    min  Σ_{k=1}^{N} (x_k − x_ref_k)^T Q (x_k − x_ref_k)
       + Σ_{k=0}^{N-1} ũ_k^T R̃ ũ_k

    where R̃ = R / force_scale²  (to account for scaled forces).

Equality constraints (dynamics + initial state)::

    x_0 = x_init                       (12 rows)
    x_{k+1} = A x_k + B ũ_k + g       (N·12 rows)

    Total equality rows: 12·(N+1)

Inequality constraints (per leg, per timestep)::

    0 ≤ F_z ≤ contact_k · f_max        (normal force bounds)
    |F_x| ≤ μ · F_z                    (friction pyramid, x)
    |F_y| ≤ μ · F_z                    (friction pyramid, y)

    → 6 rows per leg per step → 6·4·N total inequality rows.
"""

import numpy as np


class ConvexMPC:
    """Centroidal-dynamics convex MPC (QP-based).

    Parameters
    ----------
    mass : float
        Total robot mass (kg).
    inertia : np.ndarray, shape (3, 3)
        Body-frame rotational inertia about CoM (kg·m²).
    prediction_horizon : int
        Number of MPC look-ahead steps ``N``.
    dt : float
        MPC discretisation timestep (s).  Typically 0.03 s (33 Hz).
    Q : np.ndarray, shape (12, 12)
        Diagonal state-tracking weight matrix.
    R : np.ndarray, shape (12, 12)
        Diagonal force-regularisation weight matrix.
    mu : float
        Coulomb friction coefficient.
    f_max : float
        Maximum normal force per leg (N).  Also used as the force
        scaling factor for QP conditioning.
    solver : QPSolver
        Backend that implements ``solve(H, f, A_eq, b_eq, A_ineq, b_ineq)``.
    """

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
        solver,
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

        # Force scaling: ũ = u / f_max  →  improves QP conditioning
        self.force_scale = 1.0 / f_max
        # Body-frame inertia inverse (constant)
        self.I_inv = np.linalg.inv(self.inertia)

    # ==========================================================
    # Dynamics Linearization
    # ==========================================================

    def update_dynamics_matrices(self, yaw, foot_positions_body):
        """Build discrete-time LTI matrices for the centroidal dynamics.

        Linearises the continuous dynamics about the current yaw angle
        using the small roll/pitch approximation ``R_WB ≈ R_z(yaw)``,
        then applies forward-Euler discretisation with timestep ``dt``.

        Continuous dynamics::

            ṗ_W  = v_W
            θ̇    ≈ ω_W                                 (small angle)
            v̇_W  = (1/m) Σ_i F_i_W  +  g_W
            ω̇_W  = I_W⁻¹ Σ_i (r_i_W × F_i_W)

        where ``I_W = R_z · I_body · R_z^T``.

        Discrete (forward Euler): ``x_{k+1} = A x_k + B ũ_k + g``

        Parameters
        ----------
        yaw : float
            Current yaw angle (rad).
        foot_positions_body : list of np.ndarray
            Four (3,) foot position vectors in yaw-aligned body frame (m).

        Returns
        -------
        A : np.ndarray, shape (12, 12)
            State transition matrix.
        B : np.ndarray, shape (12, 12)
            Input matrix (maps *scaled* forces ũ to state).
        g : np.ndarray, shape (12,)
            Gravity affine term.
        """
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        R_z = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw,  cos_yaw, 0],
            [0, 0, 1]
        ])

        # --- A matrix (state transition) ---
        A = np.eye(12)
        # ṗ_W = v_W  →  p_{k+1} = p_k + v_k · dt
        A[0:3, 6:9] = np.eye(3) * self.dt
        # θ̇ ≈ ω_W  →  θ_{k+1} = θ_k + ω_k · dt  (small roll/pitch)
        A[3:6, 9:12] = np.eye(3) * self.dt

        # --- B matrix (input mapping) ---
        B = np.zeros((12, 12))

        # World-frame inertia inverse: I_W⁻¹ = R_z · I_body⁻¹ · R_z^T
        I_world_inv = R_z @ self.I_inv @ R_z.T

        for i in range(4):
            # Foot position in world frame (yaw rotation of body-frame coords)
            r_world = R_z @ foot_positions_body[i]

            # Skew-symmetric matrix [r_world]×  such that [r]× · F = r × F
            rx, ry, rz = r_world
            r_skew = np.array([
                [0,   -rz,  ry],
                [rz,   0,  -rx],
                [-ry,  rx,   0]
            ])

            # Linear: v̇_W = F_W / m
            B_f_lin = (np.eye(3) / self.mass) * self.dt
            # Angular: ω̇_W = I_W⁻¹ · [r_W]× · F_W
            B_f_ang = (I_world_inv @ r_skew) * self.dt

            # Store with force scaling: B maps ũ (not raw F)
            B[6:9, 3*i:3*i+3] = B_f_lin / self.force_scale
            B[9:12, 3*i:3*i+3] = B_f_ang / self.force_scale

        # Gravity affine term: only affects v_z  (world z-down = -9.81)
        g = np.zeros(12)
        g[8] = -9.81 * self.dt

        return A, B, g

    # ==========================================================
    # Build QP Matrices (Uncondensed Form)
    # ==========================================================

    def build_qp(self, x0, x_ref, contact_schedule, A, B, g):
        """Assemble the full QP from dynamics and constraints.

        Builds the uncondensed (sparse-friendly) QP:

        .. math::

            \\min_z \\; \\tfrac{1}{2} z^T H z + f^T z
            \\quad \\text{s.t.} \\quad A_{eq} z = b_{eq}, \\;
            A_{ineq} z \\le b_{ineq}

        Parameters
        ----------
        x0 : np.ndarray, shape (12,)
            Current MPC state (world frame).
        x_ref : np.ndarray, shape (12, N+1)
            Reference trajectory over the horizon.
        contact_schedule : np.ndarray, shape (N, 4)
            Binary contact flags per leg per timestep.  ``1`` = stance
            (leg may push), ``0`` = swing (force = 0).
        A, B, g : np.ndarray
            Discrete dynamics matrices from ``update_dynamics_matrices``.

        Returns
        -------
        H : np.ndarray, shape (n_vars, n_vars)
            Positive semi-definite Hessian.
        f : np.ndarray, shape (n_vars,)
            Linear cost vector.
        A_eq, b_eq : np.ndarray
            Equality constraint matrix and RHS (dynamics + init).
        A_ineq, b_ineq : np.ndarray
            Inequality constraint matrix and RHS (friction + contact).
        """
        nx = 12   # state dimension
        nu = 12   # input dimension (3 force components × 4 legs)
        N = self.horizon

        # Decision variable z = [x_0, x_1, …, x_N, ũ_0, …, ũ_{N-1}]
        n_vars = nx * (N + 1) + nu * N

        # ----------------------------------------------------------
        # Cost: Σ (x_k - x_ref_k)^T Q (x_k - x_ref_k) + ũ_k^T R̃ ũ_k
        #
        # Expanding the quadratic and dropping constants:
        #   H[x_k block] += Q
        #   f[x_k block] += -Q @ x_ref_k
        #   H[ũ_k block] += R / force_scale²
        # ----------------------------------------------------------
        H = np.zeros((n_vars, n_vars))
        f = np.zeros(n_vars)

        for k in range(N):
            # State cost at timestep k+1 (we don't penalise x_0)
            idx_x = nx * (k + 1)
            H[idx_x:idx_x+nx, idx_x:idx_x+nx] += self.Q
            f[idx_x:idx_x+nx] = -self.Q @ x_ref[:, k + 1]

            # Input cost at timestep k (scaled forces)
            idx_u = nx * (N + 1) + nu * k
            H[idx_u:idx_u+nu, idx_u:idx_u+nu] += self.R / (self.force_scale**2)

        # ----------------------------------------------------------
        # Equality constraints
        # ----------------------------------------------------------
        A_eq = []
        b_eq = []

        # (a) Initial state: x_0 = x_init
        row = np.zeros((nx, n_vars))
        row[:, 0:nx] = np.eye(nx)
        A_eq.append(row)
        b_eq.append(x0)

        # (b) Dynamics: x_{k+1} - A x_k - B ũ_k = g   for k = 0…N-1
        for k in range(N):
            row = np.zeros((nx, n_vars))

            idx_xk = nx * k
            idx_xkp1 = nx * (k + 1)
            idx_uk = nx * (N + 1) + nu * k

            row[:, idx_xkp1:idx_xkp1+nx] = np.eye(nx)   #  x_{k+1}
            row[:, idx_xk:idx_xk+nx] = -A                # -A x_k
            row[:, idx_uk:idx_uk+nu] = -B                 # -B ũ_k

            A_eq.append(row)
            b_eq.append(g)                                 # = g

        A_eq = np.vstack(A_eq)
        b_eq = np.concatenate(b_eq)

        # ----------------------------------------------------------
        # Inequality constraints (per leg, per timestep)
        #
        # For each leg i at each step k, the *scaled* force
        # ũ = [F̃x, F̃y, F̃z] satisfies:
        #
        #   (1)  -F̃z ≤ 0                  (Fz ≥ 0, can only push)
        #   (2)   F̃z ≤ contact_k_i        (Fz ≤ f_max if stance, 0 if swing)
        #   (3)   F̃x - μ F̃z ≤ 0          (friction pyramid +x)
        #   (4)  -F̃x - μ F̃z ≤ 0          (friction pyramid -x)
        #   (5)   F̃y - μ F̃z ≤ 0          (friction pyramid +y)
        #   (6)  -F̃y - μ F̃z ≤ 0          (friction pyramid -y)
        #
        # Note: constraint (2) RHS is contact_schedule[k, leg] which
        # is 1.0 for stance (ũz_max = 1 = f_max · force_scale) and
        # 0.0 for swing (forces zeroed out).
        # ----------------------------------------------------------
        A_ineq = []
        b_ineq = []

        mu = self.mu
        for k in range(N):
            for leg in range(4):
                idx_u = nx * (N + 1) + nu * k + 3 * leg

                # (1) -F̃z ≤ 0
                row = np.zeros(n_vars)
                row[idx_u + 2] = -1
                A_ineq.append(row)
                b_ineq.append(0)

                # (2) F̃z ≤ contact
                row = np.zeros(n_vars)
                row[idx_u + 2] = 1
                A_ineq.append(row)
                b_ineq.append(contact_schedule[k, leg])

                # (3) F̃x - μ F̃z ≤ 0
                row = np.zeros(n_vars)
                row[idx_u + 0] = 1
                row[idx_u + 2] = -mu
                A_ineq.append(row)
                b_ineq.append(0)

                # (4) -F̃x - μ F̃z ≤ 0
                row = np.zeros(n_vars)
                row[idx_u + 0] = -1
                row[idx_u + 2] = -mu
                A_ineq.append(row)
                b_ineq.append(0)

                # (5) F̃y - μ F̃z ≤ 0
                row = np.zeros(n_vars)
                row[idx_u + 1] = 1
                row[idx_u + 2] = -mu
                A_ineq.append(row)
                b_ineq.append(0)

                # (6) -F̃y - μ F̃z ≤ 0
                row = np.zeros(n_vars)
                row[idx_u + 1] = -1
                row[idx_u + 2] = -mu
                A_ineq.append(row)
                b_ineq.append(0)

        A_ineq = np.vstack(A_ineq)
        b_ineq = np.array(b_ineq)

        return H, f, A_eq, b_eq, A_ineq, b_ineq

    # ==========================================================
    # Solve
    # ==========================================================

    def solve(self, state, x_ref, contact_schedule, foot_positions_body):
        """Run one MPC iteration: linearise, build QP, solve, extract u_0.

        Parameters
        ----------
        state : State
            Current robot state (``state.base.to_mpc_vector()`` provides
            the 12-dim world-frame MPC state).
        x_ref : np.ndarray, shape (12, N+1)
            Reference trajectory from the trajectory generator.
        contact_schedule : np.ndarray, shape (N, 4)
            Binary stance/swing schedule from the gait scheduler.
        foot_positions_body : list of np.ndarray
            Four (3,) foot positions in yaw-aligned body frame (m).

        Returns
        -------
        np.ndarray, shape (12,)
            Optimal ground reaction forces ``[F_FL, F_FR, F_RL, F_RR]``
            in **world frame** (N).  Returns zeros if the solver fails.
        """
        x0 = state.base.to_mpc_vector()
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

        # Extract first control action and undo force scaling
        idx_u0 = nx * (N + 1)
        u0 = z[idx_u0:idx_u0 + nu]

        return u0 / self.force_scale
