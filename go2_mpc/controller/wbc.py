import numpy as np


class WholeBodyController:
    def __init__(self, robot, torque_limit=35.0, height_kp=50.0, roll_kp=30.0, pitch_kp=20.0, pitch_kd=5.0):
        """
        Args:
            robot: Instance implementing the Robot interface.
            torque_limit: Maximum torque per joint (Nm).
            height_kp: Height feedback gain (default 50)
            roll_kp: Roll correction gain (default 30)
            pitch_kp: Pitch correction gain (default 20)
            pitch_kd: Pitch damping gain (default 5) - NEW
        """
        self.robot = robot
        self.torque_limit = torque_limit
        self.height_kp = height_kp
        self.roll_kp = roll_kp
        self.pitch_kp = pitch_kp
        self.pitch_kd = pitch_kd
        self.prev_pitch_error = 0.0  # For derivative term
        self.tau_ff = np.zeros(12)  # 4 legs x 3 DOF
        self.target_height = 0.27  # Default target height

    def set_target_height(self, height):
        """Set target height for body."""
        self.target_height = height

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

        # Get current base state
        base_pos, base_quat = self.robot.get_base_pose()
        base_height = base_pos[2]
        
        # Compute roll and pitch from quaternion
        w, x, y, z = base_quat
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = np.arcsin(2*(w*y - z*x))
        
        # Get angular velocity for pitch damping
        _, omega = self.robot.get_base_velocity()
        pitch_rate = omega[1] if omega is not None else 0.0  # Y-axis rotation
        
        # Height error
        height_error = self.target_height - base_height
        
        # Roll error (want roll = 0)
        roll_error = 0.0 - roll
        
        # Pitch error (want pitch = 0)
        pitch_error = 0.0 - pitch
        
        # Count stance legs
        num_stance = sum(1 for f in foot_forces if np.linalg.norm(f) > 1e-3)
        if num_stance > 0:
            # Height correction
            z_correction = self.height_kp * height_error / num_stance
            
            # Roll correction (differential force: left legs +, right legs -)
            # FL(0), FR(1), RL(2), RR(3)
            roll_correction = self.roll_kp * roll_error / num_stance
            
            # Pitch correction with damping (differential force: front legs +, rear legs -)
            pitch_correction = (self.pitch_kp * pitch_error + self.pitch_kd * pitch_rate) / num_stance
            
            for i in range(4):
                if np.linalg.norm(foot_forces[i]) > 1e-3:
                    # Height correction on Z
                    foot_forces[i][2] += z_correction
                    
                    # Roll correction: add Y force for left legs, subtract for right
                    if i in [0, 2]:  # Left legs (FL, RL)
                        foot_forces[i][1] += roll_correction
                    else:  # Right legs (FR, RR)
                        foot_forces[i][1] -= roll_correction
                    
                    # Pitch correction: add X force for front legs, subtract for rear
                    if i in [0, 1]:  # Front legs (FL, FR)
                        foot_forces[i][0] += pitch_correction
                    else:  # Rear legs (RL, RR)
                        foot_forces[i][0] -= pitch_correction

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
