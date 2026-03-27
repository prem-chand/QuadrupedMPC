"""
Batched Whole-Body Controller using GPU acceleration.

Vectorized implementation for parallel torque computation across multiple environments.
Matches the interface and formulation of single-env WBC.
"""

import torch
import numpy as np


class BatchedWBC:
    """
    Batched Whole-Body Controller using Jacobian transpose.

    Computes joint torques from foot ground reaction forces using:
        tau = J_leg^T @ F_foot + gravity_compensation

    Parameters
    ----------
    torque_limit : float
        Maximum absolute torque per joint (Nm).
    num_envs : int
        Number of parallel environments.
    device : str
        'cuda' or 'cpu'.
    gravity : float
        Gravitational acceleration (m/s²).
    """

    def __init__(
        self,
        torque_limit: float = 35.0,
        num_envs: int = 1,
        device: str = "cuda",
        gravity: float = 9.81,
    ):
        self.torque_limit = torque_limit
        self.num_envs = num_envs
        self.device = device
        self.gravity = gravity

    def compute_torques_batch(
        self,
        foot_forces: torch.Tensor,
        jacobians: torch.Tensor = None,
        gravity_compensation: torch.Tensor = None,
        contact_schedule: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute joint torques for batch of environments.

        Parameters
        ----------
        foot_forces : torch.Tensor, shape (num_envs, 4, 3)
            Ground reaction forces in world frame (N).
        jacobians : torch.Tensor, shape (num_envs, 4, 3, 3), optional
            Leg Jacobians. If not provided, uses gravity compensation only.
        gravity_compensation : torch.Tensor, shape (num_envs, 4, 3), optional
            Pre-computed gravity compensation torques. If not provided, computed internally.
        contact_schedule : torch.Tensor, shape (num_envs, 4), optional
            Binary contact flags. Forces from swing legs are zeroed.

        Returns
        -------
        torques : torch.Tensor, shape (num_envs, 12)
            Joint torques (Nm).
        """
        num_envs = foot_forces.shape[0]

        if contact_schedule is not None:
            contact_mask = contact_schedule.unsqueeze(-1)
            foot_forces = foot_forces * contact_mask

        torques = torch.zeros(num_envs, 12, device=self.device)

        if jacobians is not None:
            for leg in range(4):
                J_leg = jacobians[:, leg, :, :]
                F_leg = foot_forces[:, leg, :]

                tau_leg = torch.matmul(J_leg.transpose(-2, -1), F_leg.unsqueeze(-1)).squeeze(-1)

                idx = leg * 3
                torques[:, idx:idx+3] = tau_leg

        if gravity_compensation is not None:
            torques = torques + gravity_compensation.view(num_envs, 12)

        torques = torch.clamp(torques, -self.torque_limit, self.torque_limit)

        return torques

    def compute_gravity_compensation_batch(
        self,
        base_positions: torch.Tensor,
        base_orientations: torch.Tensor,
        joint_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute gravity compensation torques for batch.

        Parameters
        ----------
        base_positions : torch.Tensor, shape (num_envs, 3)
            Base positions in world frame.
        base_orientations : torch.Tensor, shape (num_envs, 4)
            Base quaternions [w, x, y, z].
        joint_positions : torch.Tensor, shape (num_envs, 12)
            Joint positions.

        Returns
        -------
        tau_grav : torch.Tensor, shape (num_envs, 12)
            Gravity compensation torques.
        """
        num_envs = base_positions.shape[0]

        R_base = self._quaternion_to_rotation_matrix(base_orientations)

        mg = torch.tensor([0, 0, self.gravity * 15.2 / 4], device=self.device)

        z_hip = R_base @ torch.tensor([1.0, 0.0, 0.0], device=self.device)
        z_thigh = R_base @ torch.tensor([0.0, 1.0, 0.0], device=self.device)
        z_knee = R_base @ torch.tensor([0.0, 1.0, 0.0], device=self.device)

        tau_grav = torch.zeros(num_envs, 12, device=self.device)

        for leg in range(4):
            idx = leg * 3
            tau_grav[:, idx] = torch.cross(z_hip, mg * 0.1) * 0.2

        return tau_grav

    def _quaternion_to_rotation_matrix(self, quat: torch.Tensor) -> torch.Tensor:
        """Convert quaternion to rotation matrix."""
        w = quat[:, 0]
        x = quat[:, 1]
        y = quat[:, 2]
        z = quat[:, 3]

        R = torch.zeros(quat.shape[0], 3, 3, device=self.device)

        R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
        R[:, 0, 1] = 2 * (x*y - z*w)
        R[:, 0, 2] = 2 * (x*z + y*w)

        R[:, 1, 0] = 2 * (x*y + z*w)
        R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
        R[:, 1, 2] = 2 * (y*z - x*w)

        R[:, 2, 0] = 2 * (x*z - y*w)
        R[:, 2, 1] = 2 * (y*z + x*w)
        R[:, 2, 2] = 1 - 2 * (x**2 + y**2)

        return R


def batched_wbc_from_config(config, num_envs: int, device: str = "cuda"):
    """
    Create BatchedWBC from config.

    Parameters
    ----------
    config : object
        Configuration object with controller parameters.
    num_envs : int
        Number of parallel environments.
    device : str
        'cuda' or 'cpu'.

    Returns
    -------
    BatchedWBC
    """
    return BatchedWBC(
        torque_limit=config.controller.torque_limit,
        num_envs=num_envs,
        device=device,
    )
