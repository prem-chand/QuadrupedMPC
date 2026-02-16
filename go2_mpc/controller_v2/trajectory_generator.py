import numpy as np

class TrajectoryGenerator:
    def __init__(self, prediction_horizon, dt):
        self.horizon = prediction_horizon
        self.dt = dt
        # State: [x,y,z,  r,p,y,  vx,vy,vz,  wx,wy,wz] (12)
        self.state_size = 12

        # Pre-allocated buffers for performance
        self._x_ref = np.zeros((self.state_size, self.horizon + 1))
        self._time_steps = (np.arange(self.horizon) + 1) * self.dt

    def generate_reference(self, current_state, v_cmd_global, yaw_rate_cmd, height_cmd):
        """
        Generates the reference trajectory for the MPC.
        
        Args:
            current_state (np.array): (12,) Current robot state.
            v_cmd_global (np.array): (3,) Desired lin vel [vx, vy, vz] in WORLD frame.
            yaw_rate_cmd (float): Desired yaw rate (rad/s).
            height_cmd (float): Desired standing height (z).
            
        Returns:
            x_ref (12, N+1): The reference trajectory.
        """
        # Reuse pre-allocated buffer
        x_ref = self._x_ref
        time_steps = self._time_steps

        # 1. Initialize with current state
        x_ref[:, 0] = current_state

        # --- POSITION (X, Y) Integration ---
        # p_{k+1} = p_k + v_cmd * dt
        # Broadcast current XY position + displacement
        x_ref[0:2, 1:] = current_state[0:2, None] + v_cmd_global[0:2, None] * time_steps
        
        # --- HEIGHT (Z) ---
        # Usually we want to hold a constant height 'height_cmd'
        # Note: We filter this slightly to avoid jumps if current Z is far off? 
        # For MPC, a hard target is usually fine.
        x_ref[2, 1:] = height_cmd
        
        # --- ORIENTATION (Roll, Pitch, Yaw) ---
        # Roll and Pitch are usually 0 for flat ground walking
        x_ref[3:5, 1:] = 0.0 
        
        # Yaw Integration: yaw_{k+1} = yaw_k + yaw_rate * dt
        # Start from current yaw
        current_yaw = current_state[5]
        x_ref[5, 1:] = current_yaw + yaw_rate_cmd * time_steps
        
        # --- LINEAR VELOCITY ---
        # Target velocity is constant over the horizon
        x_ref[6:9, 1:] = v_cmd_global[:, None]
        # Ensure Z velocity ref is 0 if we want to hold height
        x_ref[8, 1:] = 0.0 
        
        # --- ANGULAR VELOCITY ---
        # Roll/Pitch rates are 0
        x_ref[9:11, 1:] = 0.0
        # Yaw rate is the command
        x_ref[11, 1:] = yaw_rate_cmd
        
        return x_ref