import numpy as np


class GaitScheduler:
    def __init__(self, gait_period, stance_ratio, horizon, dt):
        self.period = gait_period
        self.stance_ratio = stance_ratio
        self.horizon = horizon
        self.dt = dt

        # Phase offsets for each leg [FL, FR, RL, RR]
        # Trot: Diagonals are in phase (0 and 3 match, 1 and 2 match)
        self.offsets = np.array([0.0, 0.5, 0.5, 0.0])

    def update_gait_params(self, gait_name):
        """Allows dynamic switching of gaits."""
        if gait_name == "trot":
            self.offsets = np.array([0.0, 0.5, 0.5, 0.0])
        elif gait_name == "bound":
            self.offsets = np.array([0.0, 0.0, 0.5, 0.5])
        elif gait_name == "pace":
            self.offsets = np.array([0.0, 0.5, 0.0, 0.5])

    def get_contact_schedule(self, current_time):
        """
        Generates the MPC contact table (Horizon x 4).
        1 = Stance, 0 = Swing
        """
        # 1. Generate time vector for the horizon
        # shape: (horizon, 1)
        t_future = current_time + np.arange(self.horizon) * self.dt
        t_future = t_future[:, None]

        # 2. Calculate raw phase for each time step
        # shape: (horizon, 1)
        raw_phase = (t_future / self.period) % 1.0

        # 3. Add offsets for each leg
        # shape: (horizon, 4) due to broadcasting (H,1) + (4,)
        leg_phases = (raw_phase + self.offsets) % 1.0

        # 4. Determine contact
        # If phase < stance_ratio, leg is in Stance (1)
        # Otherwise, leg is in Swing (0)
        contacts = (leg_phases < self.stance_ratio).astype(int)

        return contacts

    def get_swing_state(self, current_time, leg_idx):
        """
        Returns normalized swing progress [0, 1] for trajectory generation.
        Returns 0 if in stance.
        """
        raw_phase = (current_time / self.period) % 1.0
        leg_phase = (raw_phase + self.offsets[leg_idx]) % 1.0

        if leg_phase < self.stance_ratio:
            return 0.0  # Stance

        # Normalize swing phase from [stance_ratio, 1.0] -> [0.0, 1.0]
        swing_progress = (leg_phase - self.stance_ratio) / \
            (1.0 - self.stance_ratio)
        return np.clip(swing_progress, 0.0, 1.0)
