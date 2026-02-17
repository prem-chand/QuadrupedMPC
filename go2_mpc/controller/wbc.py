import numpy as np


class WholeBodyController:
    def __init__(self, robot, torque_limit=35.0):
        """
        Args:
            robot: Instance implementing the Robot interface.
            torque_limit: Maximum torque per joint (Nm).
        """
        self.robot = robot
        self.torque_limit = torque_limit
        self.tau_ff = np.zeros(12)  # 4 legs x 3 DOF

    def compute_torques(self, foot_forces, gravity_comp=True):
        """
        Computes joint torques from desired ground reaction forces.

        Args:
            foot_forces: List of 4 arrays (3,), desired GRFs [FL, FR, RL, RR] in World Frame
            gravity_comp: If True, add gravity compensation torques

        Returns:
            (12,) array of joint torques for all actuators
        """
        self.tau_ff.fill(0)

        for i in range(4):
            idx = slice(3*i, 3*i+3)

            # Gravity compensation
            if gravity_comp:
                self.tau_ff[idx] = self.robot.get_gravity_compensation(i)

            # Skip Jacobian if force is negligible (swing phase)
            if np.linalg.norm(foot_forces[i]) < 1e-3:
                continue

            # J^T mapping: tau = -J_leg^T @ F
            J_leg = self.robot.get_leg_jacobian(i)
            self.tau_ff[idx] += -J_leg.T @ foot_forces[i]

        np.clip(self.tau_ff, -self.torque_limit, self.torque_limit, out=self.tau_ff)
        return self.tau_ff
