"""
Batched Convex MPC using GPU-accelerated solver.

Vectorized implementation for parallel solving across multiple environments.
Matches the interface and formulation of single-env ConvexMPC.
"""

import torch
import numpy as np


class BatchedConvexMPC:
    """
    Batched centroidal-dynamics convex MPC using GPU acceleration.

    Parameters
    ----------
    mass : float
        Total robot mass (kg).
    inertia : np.ndarray or torch.Tensor, shape (3, 3)
        Body-frame rotational inertia.
    prediction_horizon : int
        Number of MPC look-ahead steps N.
    dt : float
        MPC discretization timestep (s).
    Q : np.ndarray or torch.Tensor, shape (12, 12)
        Diagonal state-tracking weight matrix.
    R : np.ndarray or torch.Tensor, shape (12, 12)
        Diagonal force-regularisation weight matrix.
    mu : float
        Coulomb friction coefficient.
    f_max : float
        Maximum normal force per leg (N).
    solver : BatchedQPSolver
        GPU batched QP solver.
    num_envs : int
        Number of parallel environments.
    device : str
        'cuda' or 'cpu'.
    """

    def __init__(
        self,
        mass: float,
        inertia,
        prediction_horizon: int,
        dt: float,
        Q,
        R: float,
        mu: float,
        f_max: float,
        solver,
        num_envs: int,
        device: str = "cuda",
    ):
        self.mass = mass
        self.horizon = prediction_horizon
        self.dt = dt
        self.mu = mu
        self.f_max = f_max
        self.force_scale = 1.0 / f_max
        self.num_envs = num_envs
        self.device = device

        if isinstance(inertia, np.ndarray):
            inertia = torch.from_numpy(inertia).float()
        if isinstance(Q, np.ndarray):
            Q = torch.from_numpy(Q).float()
        if isinstance(R, np.ndarray):
            R = torch.from_numpy(R).float()

        self.inertia = inertia.to(device)
        self.Q = Q.to(device)
        self.R = R.to(device)
        self.I_inv = torch.linalg.inv(self.inertia)

        self.solver = solver

    def solve_batch(
        self,
        states: torch.Tensor,
        x_refs: torch.Tensor,
        contact_schedules: torch.Tensor,
        foot_positions_body: torch.Tensor,
    ) -> torch.Tensor:
        """
        Solve MPC for batch of environments.

        Parameters
        ----------
        states : torch.Tensor, shape (num_envs, 12)
            Current MPC states (world frame).
        x_refs : torch.Tensor, shape (num_envs, 12, N+1)
            Reference trajectories over horizon.
        contact_schedules : torch.Tensor, shape (num_envs, N, 4)
            Binary contact flags per leg per timestep.
        foot_positions_body : torch.Tensor, shape (num_envs, 4, 3)
            Foot positions in yaw-aligned body frame.

        Returns
        -------
        forces : torch.Tensor, shape (num_envs, 12)
            Optimal ground reaction forces in world frame (N).
        """
        num_envs = states.shape[0]
        nx = 12
        nu = 12
        N = self.horizon

        x0 = states

        yaws = torch.atan2(
            2.0 * (states[:, 5] * states[:, 4] + states[:, 2] * states[:, 3]),
            1.0 - 2.0 * (states[:, 3]**2 + states[:, 4]**2)
        )
        yaws = torch.zeros(num_envs, device=self.device) if yaws.dim() == 0 else yaws

        A, B, g = self._build_dynamics_batch(yaws, foot_positions_body)

        H, f = self._build_cost_batch(x_refs)

        A_eq, b_eq = self._build_eq_constraints_batch(x0, A, B, g)

        A_ineq, b_ineq = self._build_ineq_constraints_batch(contact_schedules)

        solution, status = self.solver.solve_batch(
            H, f, A_eq, b_eq, A_ineq, b_ineq
        )

        idx_u0 = nx * (N + 1)
        u0 = solution[:, idx_u0:idx_u0 + nu]

        forces = u0 / self.force_scale

        return forces

    def _build_dynamics_batch(
        self,
        yaws: torch.Tensor,
        foot_positions_body: torch.Tensor,
    ):
        """Build discrete-time LTI dynamics matrices for batch."""
        N = self.horizon
        nx = 12
        nu = 12
        num_envs = self.num_envs

        A = torch.eye(nx, device=self.device).unsqueeze(0).expand(num_envs, -1, -1)
        A[:, 0:3, 6:9] = torch.eye(3, device=self.device).unsqueeze(0) * self.dt
        A[:, 3:6, 9:12] = torch.eye(3, device=self.device).unsqueeze(0) * self.dt

        B = torch.zeros(num_envs, nx, nu, device=self.device)

        cos_yaw = torch.cos(yaws)
        sin_yaw = torch.sin(yaws)

        R_z = torch.zeros(num_envs, 3, 3, device=self.device)
        R_z[:, 0, 0] = cos_yaw
        R_z[:, 0, 1] = -sin_yaw
        R_z[:, 1, 0] = sin_yaw
        R_z[:, 1, 1] = cos_yaw
        R_z[:, 2, 2] = 1.0

        I_world_inv = torch.matmul(
            torch.matmul(R_z, self.I_inv.unsqueeze(0)),
            R_z.transpose(-2, -1)
        )

        for i in range(4):
            r_body = foot_positions_body[:, i, :]
            r_world = torch.matmul(R_z, r_body.unsqueeze(-1)).squeeze(-1)

            r_skew = torch.zeros(num_envs, 3, 3, device=self.device)
            r_skew[:, 0, 1] = -r_world[:, 2]
            r_skew[:, 0, 2] = r_world[:, 1]
            r_skew[:, 1, 0] = r_world[:, 2]
            r_skew[:, 1, 2] = -r_world[:, 0]
            r_skew[:, 2, 0] = -r_world[:, 1]
            r_skew[:, 2, 1] = r_world[:, 0]

            B_f_lin = (torch.eye(3, device=self.device).unsqueeze(0) / self.mass) * self.dt
            B_f_ang = torch.matmul(I_world_inv, r_skew) * self.dt

            B[:, 7:10, 3*i:3*i+3] = B_f_lin / self.force_scale
            B[:, 9:12, 3*i:3*i+3] = B_f_ang / self.force_scale

        g = torch.zeros(num_envs, nx, device=self.device)
        g[:, 8] = -9.81 * self.dt

        return A, B, g

    def _build_cost_batch(self, x_refs: torch.Tensor):
        """Build QP cost matrices for batch."""
        nx = 12
        nu = 12
        N = self.horizon
        num_envs = self.num_envs

        n_vars = nx * (N + 1) + nu * N

        H = torch.zeros(num_envs, n_vars, n_vars, device=self.device)
        f = torch.zeros(num_envs, n_vars, device=self.device)

        Q = self.Q.unsqueeze(0).unsqueeze(0)
        R = (self.R / (self.force_scale**2)).unsqueeze(0).unsqueeze(0)

        for k in range(N):
            idx_x = nx * (k + 1)
            H[:, idx_x:idx_x+nx, idx_x:idx_x+nx] = Q

            x_ref_k = x_refs[:, :, k + 1]
            f[:, idx_x:idx_x+nx] = -self.Q @ x_ref_k.transpose(-2, -1)

            idx_u = nx * (N + 1) + nu * k
            H[:, idx_u:idx_u+nu, idx_u:idx_u+nu] = R

        return H, f

    def _build_eq_constraints_batch(self, x0, A, B, g):
        """Build equality constraints for batch."""
        nx = 12
        nu = 12
        N = self.horizon
        num_envs = self.num_envs

        n_vars = nx * (N + 1) + nu * N

        A_eq = torch.zeros(num_envs, nx * (N + 1), n_vars, device=self.device)
        b_eq = torch.zeros(num_envs, nx * (N + 1), device=self.device)

        A_eq[:, 0:nx, 0:nx] = torch.eye(nx, device=self.device).unsqueeze(0)
        b_eq[:, 0:nx] = x0

        for k in range(N):
            idx_xk = nx * k
            idx_xkp1 = nx * (k + 1)
            idx_uk = nx * (N + 1) + nu * k

            A_eq[:, idx_xkp1:idx_xkp1+nx, idx_xkp1:idx_xkp1+nx] = torch.eye(nx, device=self.device).unsqueeze(0)
            A_eq[:, idx_xkp1:idx_xkp1+nx, idx_xk:idx_xk+nx] = -A
            A_eq[:, idx_xkp1:idx_xkp1+nx, idx_uk:idx_uk+nu] = -B
            b_eq[:, idx_xkp1:idx_xkp1+nx] = g

        return A_eq, b_eq

    def _build_ineq_constraints_batch(self, contact_schedules: torch.Tensor):
        """Build inequality constraints (friction cone + contact) for batch."""
        nx = 12
        nu = 12
        N = self.horizon
        num_envs = self.num_envs
        mu = self.mu

        n_vars = nx * (N + 1) + nu * N
        n_ineq = 6 * 4 * N

        A_ineq = torch.zeros(num_envs, n_ineq, n_vars, device=self.device)
        b_ineq = torch.zeros(num_envs, n_ineq, device=self.device)

        row = 0
        for k in range(N):
            for leg in range(4):
                idx_u = nx * (N + 1) + nu * k + 3 * leg

                A_ineq[:, row, idx_u + 2] = -1
                b_ineq[:, row] = 0
                row += 1

                A_ineq[:, row, idx_u + 2] = 1
                b_ineq[:, row] = contact_schedules[:, k, leg]
                row += 1

                A_ineq[:, row, idx_u + 0] = 1
                A_ineq[:, row, idx_u + 2] = -mu
                b_ineq[:, row] = 0
                row += 1

                A_ineq[:, row, idx_u + 0] = -1
                A_ineq[:, row, idx_u + 2] = -mu
                b_ineq[:, row] = 0
                row += 1

                A_ineq[:, row, idx_u + 1] = 1
                A_ineq[:, row, idx_u + 2] = -mu
                b_ineq[:, row] = 0
                row += 1

                A_ineq[:, row, idx_u + 1] = -1
                A_ineq[:, row, idx_u + 2] = -mu
                b_ineq[:, row] = 0
                row += 1

        return A_ineq, b_ineq


def batched_mpc_from_config(config, solver, num_envs: int, device: str = "cuda"):
    """
    Create BatchedConvexMPC from config.

    Parameters
    ----------
    config : object
        Configuration object with MPC parameters.
    solver : BatchedQPSolver
        GPU batched QP solver.
    num_envs : int
        Number of parallel environments.
    device : str
        'cuda' or 'cpu'.

    Returns
    -------
    BatchedConvexMPC
    """
    return BatchedConvexMPC(
        mass=config.mpc.mass,
        inertia=config.mpc.inertia,
        prediction_horizon=config.mpc.horizon,
        dt=config.mpc.dt,
        Q=config.mpc.Q,
        R=config.mpc.R,
        mu=config.mpc.mu,
        f_max=config.mpc.f_max,
        solver=solver,
        num_envs=num_envs,
        device=device,
    )
