"""
Balance Controller for push recovery and reactive gait switching.

Implements MIT Cheetah-style balance control:
- Disturbance detection via roll/pitch/velocity thresholds
- Reactive gait switching to more stable patterns
- Automatic recovery to normal gait
"""

import numpy as np


class BalanceController:
    """
    Balance controller for push recovery.
    
    Detects disturbances and triggers reactive gait switching:
    - Roll/pitch deviation > threshold → recovery mode
    - Lateral velocity spike → switch to bound gait
    - Angular velocity spike → recovery mode
    
    Recovery increases stance ratio and returns to normal after recovery_time.
    """

    def __init__(self, config: dict = None):
        """
        Parameters
        ----------
        config : dict
            Configuration dictionary with keys:
            - BALANCE_ROLL_THRESHOLD: rad, default 0.15
            - BALANCE_PITCH_THRESHOLD: rad, default 0.15
            - BALANCE_VEL_THRESHOLD: m/s, default 0.3
            - BALANCE_ANGULAR_THRESHOLD: rad/s, default 1.0
            - BALANCE_RECOVERY_STANCE: stance ratio during recovery, default 0.75
            - BALANCE_NORMAL_STANCE: normal stance ratio, default 0.65
            - BALANCE_RECOVERY_TIME: seconds, default 0.5
        """
        if config is None:
            config = {}
            
        self.roll_threshold = config.get("BALANCE_ROLL_THRESHOLD", 0.15)
        self.pitch_threshold = config.get("BALANCE_PITCH_THRESHOLD", 0.15)
        self.velocity_threshold = config.get("BALANCE_VEL_THRESHOLD", 0.3)
        self.angular_threshold = config.get("BALANCE_ANGULAR_THRESHOLD", 1.0)
        
        self.recovery_stance_ratio = config.get("BALANCE_RECOVERY_STANCE", 0.75)
        self.normal_stance_ratio = config.get("BALANCE_NORMAL_STANCE", 0.65)
        self.recovery_time = config.get("BALANCE_RECOVERY_TIME", 0.5)
        
        self.disturbance_detected = False
        self.recovery_start_time = 0.0
        self.target_gait = "trot"
        self.current_gait = "trot"

    def detect_disturbance(self, state, command) -> bool:
        """
        Detect if robot is experiencing external disturbance.
        
        Parameters
        ----------
        state : State
            Current robot state.
        command : Command
            Current command (unused, for API compatibility).
            
        Returns
        -------
        bool
            True if disturbance detected.
        """
        roll_error = abs(state.base.roll)
        pitch_error = abs(state.base.pitch)
        lateral_vel = abs(state.base.linear_velocity[1])
        angular_vel_mag = np.linalg.norm(state.base.angular_velocity)
        
        return (roll_error > self.roll_threshold or
                pitch_error > self.pitch_threshold or
                lateral_vel > self.velocity_threshold or
                angular_vel_mag > self.angular_threshold)

    def compute_recovery_action(self, state, current_time, gait_scheduler):
        """
        Compute recovery action based on current state.
        
        Parameters
        ----------
        state : State
            Current robot state.
        current_time : float
            Current simulation time (s).
        gait_scheduler : GaitScheduler
            Gait scheduler to modify.
            
        Returns
        -------
        tuple
            (gait_name, modified_stance_ratio)
        """
        disturbance = self.detect_disturbance(state, None)
        
        if disturbance and not self.disturbance_detected:
            self.disturbance_detected = True
            self.recovery_start_time = current_time
            
            lateral_vel = abs(state.base.linear_velocity[1])
            if lateral_vel > self.velocity_threshold:
                self.target_gait = "bound"
            else:
                self.target_gait = "trot"
            
            gait_scheduler.update_gait_params(self.target_gait)
            gait_scheduler.stance_ratio = self.recovery_stance_ratio
            return self.target_gait, self.recovery_stance_ratio
        
        elif self.disturbance_detected:
            recovery_elapsed = current_time - self.recovery_start_time
            
            if recovery_elapsed > self.recovery_time:
                if not self.detect_disturbance(state, None):
                    self.disturbance_detected = False
                    self.target_gait = "trot"
                    gait_scheduler.update_gait_params("trot")
                    gait_scheduler.stance_ratio = self.normal_stance_ratio
                    return "trot", self.normal_stance_ratio
            
            return self.target_gait, self.recovery_stance_ratio
        
        return "trot", self.normal_stance_ratio

    def reset(self):
        """Reset balance controller state."""
        self.disturbance_detected = False
        self.recovery_start_time = 0.0
        self.target_gait = "trot"
        self.current_gait = "trot"
