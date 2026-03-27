"""
Contact force estimation from MPC residuals.

Estimates actual ground reaction forces from:
1. Desired forces from MPC
2. Measured base acceleration vs predicted
3. Contact schedule (stance/swing)

This enables slip detection and robust locomotion without force sensors.
"""

import numpy as np


class ContactForceEstimator:
    """
    Estimates contact forces from dynamics residuals.
    
    Uses the difference between predicted and measured base acceleration
    to estimate the actual GRF, accounting for model errors and disturbances.
    
    Parameters
    ----------
    mass : float
        Robot mass (kg).
    alpha : float
        EMA smoothing factor for force estimation (0-1).
        Higher = more responsive, lower = smoother.
    """

    def __init__(self, mass: float = 15.2, alpha: float = 0.3):
        self.mass = mass
        self.alpha = alpha
        self._prev_forces = np.zeros(12)

    def estimate(
        self,
        desired_forces: np.ndarray,
        measured_accel: np.ndarray,
        predicted_accel: np.ndarray,
        contact_schedule: np.ndarray,
        foot_positions_world: list[np.ndarray],
    ) -> np.ndarray:
        """
        Estimate actual contact forces from residuals.
        
        Parameters
        ----------
        desired_forces : np.ndarray, shape (12,)
            MPC desired forces [F_FL, F_FR, F_RL, F_RR] in world frame (N).
        measured_accel : np.ndarray, shape (3,)
            Measured base linear acceleration in world frame (m/s²).
        predicted_accel : np.ndarray, shape (3,)
            Predicted base acceleration from desired forces (m/s²).
        contact_schedule : np.ndarray, shape (4,)
            Binary contact flags [FL, FR, RL, RR]. 1 = stance, 0 = swing.
        foot_positions_world : list of np.ndarray
            Four foot positions in world frame.
            
        Returns
        -------
        estimated_forces : np.ndarray, shape (12,)
            Estimated actual GRF in world frame (N).
        """
        residual_accel = measured_accel - predicted_accel
        
        residual_force = self.mass * residual_accel
        
        total_residual_z = np.abs(residual_force[2])
        
        stance_mask = contact_schedule > 0
        num_stance = np.sum(stance_mask)
        
        if num_stance > 0:
            residual_per_leg_z = total_residual_z / num_stance
            
            correction = np.zeros(12)
            for i in range(4):
                if stance_mask[i]:
                    idx = 3 * i
                    scale = min(1.0, 0.5 * residual_per_leg_z / max(desired_forces[idx + 2], 1.0))
                    correction[idx:idx + 3] = desired_forces[idx:idx + 3] * scale
        else:
            correction = np.zeros(12)
        
        raw_estimate = desired_forces + correction
        
        for i in range(4):
            idx = 3 * i
            if contact_schedule[i] < 0.5:
                raw_estimate[idx:idx + 3] = 0.0
            else:
                raw_estimate[idx + 2] = max(0.0, raw_estimate[idx + 2])
        
        estimated_forces = (
            self.alpha * raw_estimate
            + (1 - self.alpha) * self._prev_forces
        )
        
        self._prev_forces = estimated_forces.copy()
        
        return estimated_forces

    def compute_predicted_accel(
        self,
        forces: np.ndarray,
        gravity: float = 9.81,
    ) -> np.ndarray:
        """
        Compute predicted base acceleration from forces.
        
        Parameters
        ----------
        forces : np.ndarray, shape (12,)
            Contact forces in world frame (N).
        gravity : float
            Gravitational acceleration (m/s²).
            
        Returns
        -------
        accel : np.ndarray, shape (3,)
            Predicted base linear acceleration in world frame (m/s²).
        """
        total_force_xy = np.sum(forces[0::3]), np.sum(forces[1::3])
        total_force_z = np.sum(forces[2::3])
        
        accel = np.array([
            total_force_xy[0] / self.mass,
            total_force_xy[1] / self.mass,
            (total_force_z / self.mass) - gravity,
        ])
        
        return accel

    def detect_slip(
        self,
        estimated_forces: np.ndarray,
        foot_velocities_world: list[np.ndarray],
        friction_coef: float = 0.6,
    ) -> np.ndarray:
        """
        Detect slip for each stance leg.
        
        Parameters
        ----------
        estimated_forces : np.ndarray, shape (12,)
            Estimated contact forces in world frame (N).
        foot_velocities_world : list of np.ndarray
            Foot velocities in world frame for each leg.
        friction_coef : float
            Friction coefficient threshold.
            
        Returns
        -------
        slip : np.ndarray, shape (4,)
            Boolean slip flags for each leg [FL, FR, RL, RR].
        """
        slip = np.zeros(4, dtype=bool)
        
        for i in range(4):
            idx = 3 * i
            
            f_normal = estimated_forces[idx + 2]
            f_x = estimated_forces[idx]
            f_y = estimated_forces[idx + 1]
            
            f_tangent = np.sqrt(f_x**2 + f_y**2)
            
            if f_normal > 5.0:
                ratio = f_tangent / f_normal
                if ratio > friction_coef:
                    slip[i] = True
                    
            vel = foot_velocities_world[i]
            vel_magnitude = np.linalg.norm(vel)
            if vel_magnitude > 0.5:
                slip[i] = True
        
        return slip

    def reset(self):
        """Reset internal state."""
        self._prev_forces = np.zeros(12)
