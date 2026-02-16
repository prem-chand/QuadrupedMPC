import numpy as np


class SwingLegIKController:
    def __init__(self, kp, kd, swing_height, dt):
        self.kp = kp
        self.kd = kd
        self.swing_height = swing_height
        self.dt = dt

        # Geometry (Unitree Go2 / A1 standard)
        self.l_hip = 0.0955
        self.l_thigh = 0.213
        self.l_calf = 0.213

        # Signs for [FL, FR, RL, RR]
        # Hip offset direction (y)
        self.side_sign = np.array([1, -1, 1, -1])

        # Hip Positions (for FK/IK reference)
        self.hip_offset_x = 0.1934
        self.hip_offset_y = 0.0465  # Center to hip-servo axis

        # Pre-allocated buffers for performance
        self._dq_des = np.zeros(3)
        self._q_des = np.zeros(3)

    def get_joint_commands(self, foot_pos_des_rel, vel_des_rel, q_curr, dq_curr, leg_idx):
        """
        Calculates torque commands for a single leg.

        Args:
            foot_pos_des_rel (np.array): Desired foot pos relative to Hip (Body Frame)
            vel_des_rel (np.array): Desired foot velocity relative to Hip
            q_curr (np.array): Current joint angles [hip, thigh, calf]
            dq_curr (np.array): Current joint velocities
            leg_idx (int): 0-3

        Returns:
            tau (np.array): Torques [tau_hip, tau_thigh, tau_calf]
        """
        # 1. Inverse Kinematics -> q_des
        try:
            q_des = self._compute_ik(foot_pos_des_rel, leg_idx)
        except ValueError:
            # If target is unreachable, hold current position to prevent crash
            q_des = q_curr

        # 2. Inverse Jacobian -> dq_des (Optional, but better tracking)
        # For simple swing, we can estimate dq_des via Finite Difference
        # if we don't want to compute analytical J_inv
        # Here we assume dq_des = 0 or pass it in if available.
        # A robust swing controller just uses high Kp and lets P-term handle velocity.

        # 3. PD Control (reuse pre-allocated dq_des buffer, always zero)
        # tau = Kp(q_des - q) + Kd(dq_des - dq)
        tau = self.kp * (q_des - q_curr) + self.kd * (self._dq_des - dq_curr)

        return tau, q_des

    def _compute_ik(self, p, leg_idx):
        """
        Analytic inverse kinematics for 3-DOF leg.

        Args:
            p: Foot position relative to hip joint frame [x, y, z]
            leg_idx: Leg index (0-3)

        Returns:
            Joint angles [q_hip, q_thigh, q_calf]
        """
        x, y, z = p[0], p[1], p[2]

        # --- 1. Hip Joint (Abduction/Adduction) ---
        # The hip moves the thigh-calf plane laterally.
        # Project vector onto YZ plane. Length L = sqrt(y^2 + z^2)
        # l_hip forms a right triangle with the leg plane.

        L_yz = np.sqrt(y**2 + z**2)

        # Check reachability
        if L_yz < self.l_hip:
            L_yz = self.l_hip  # Clamp to avoid sqrt(-1)

        # Hip angle calculation
        side = self.side_sign[leg_idx]
        alpha = np.arccos(self.l_hip / L_yz)
        q_hip = np.arctan2(-z, side*y) - alpha
        if leg_idx % 2 != 0:
            q_hip = -q_hip

        # Knee & Thigh (Planar 2-Link IK)

        r_p = np.sqrt(x**2 + h_proj**2)

        # Check reachability
        max_len = self.l_thigh + self.l_calf
        if r_p > max_len:
            r_p = max_len

        # Knee angle (Law of Cosines)
        cos_knee = (self.l_thigh**2 + self.l_calf**2 - r_p**2) / (2 * self.l_thigh * self.l_calf)
        cos_knee = np.clip(cos_knee, -1.0, 1.0)
        internal_angle = np.arccos(cos_knee)
        q_calf = -(np.pi - internal_angle)

        # Thigh angle

        phi_pos = np.arctan2(x, h_proj)
        cos_beta = (self.l_thigh**2 + r_p**2 - self.l_calf**2) / (2 * self.l_thigh * r_p)
        cos_beta = np.clip(cos_beta, -1.0, 1.0)
        beta = np.arccos(cos_beta)
        q_thigh = phi_pos + beta

        # Fill pre-allocated buffer
        self._q_des[0] = q_hip
        self._q_des[1] = q_thigh
        self._q_des[2] = q_calf
        return self._q_des
