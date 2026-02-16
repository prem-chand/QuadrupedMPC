import numpy as np
import mujoco

class WholeBodyController:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.nv = model.nv
        self.nu = model.nu

        # Foot site configuration
        self.foot_names = ["FL_toe", "FR_toe", "RL_toe", "RR_toe"]
        self.foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n) for n in self.foot_names]

        # Joint & Actuator Mapping
        # Leg DOF indices in qvel (after 6 floating base DOFs)
        # We need: self.leg_dofs = [FL, FR, RL, RR]
        self.leg_dofs = [
            [6, 7, 8],     # FL
            [9, 10, 11],   # FR
            [12, 13, 14],  # RL
            [15, 16, 17]   # RR
        ]

        # Actuator indices corresponding to leg joints
        self.leg_actuators = [
            [0, 1, 2],    # FL
            [3, 4, 5],    # FR
            [6, 7, 8],    # RL
            [9, 10, 11]   # RR
        ]
        
        # Preallocate memory buffers
        self.J_full = np.zeros((3, self.nv))  # Full Jacobian buffer
        self.J_rot = np.zeros((3, self.nv))   # Rotation Jacobian
        self.tau_ff = np.zeros(self.nu)       # Feed-forward torque

    def compute_torques(self, foot_forces, gravity_comp=True):
        """
        Computes joint torques from desired ground reaction forces.

        Args:
            foot_forces: List of 4x3 arrays, desired GRFs [FL, FR, RL, RR] in World Frame
            gravity_comp: If True, add gravity compensation torques

        Returns:
            Array of joint torques for all actuators
        """
        self.tau_ff.fill(0)
        
        # 1. Gravity Compensation (qfrc_bias)
        # qfrc_bias contains Coriolis, Centrifugal, and Gravity forces.
        if gravity_comp:
            # We map the relevant parts of qfrc_bias (6 onwards) to our actuators
            # This handles the heavy lifting of holding the robot up against gravity
            # and swing leg dynamics.
            
            # Fast vectorized map (if indices are aligned, otherwise loop)
            # For robustness, we loop over legs
            for i in range(4):
                dofs = self.leg_dofs[i]
                acts = self.leg_actuators[i]
                
                # qfrc_bias aligns with qvel/dofs
                self.tau_ff[acts] = self.data.qfrc_bias[dofs]

        # Feed-forward force control (J^T * F)
        for i in range(4):
            # If force is zero (swing phase), skip Jacobian calc to save time
            if np.linalg.norm(foot_forces[i]) < 1e-3:
                continue

            # Compute Jacobian for this foot site
            mujoco.mj_jacSite(self.model, self.data, self.J_full, self.J_rot, self.foot_ids[i])

            # Extract 3x3 Block for this leg
            dof_indices = self.leg_dofs[i]
            J_leg = self.J_full[:, dof_indices]

            # Map GRF to torque: tau = -J^T * F
            tau_leg = -J_leg.T @ foot_forces[i]

            # Add to total torque
            act_indices = self.leg_actuators[i]
            self.tau_ff[act_indices] += tau_leg


        # Clip torques to actuator limits
        np.clip(self.tau_ff, -40.0, 40.0, out=self.tau_ff)
        return self.tau_ff