"""
Linear Kalman Filter for quadruped state estimation.

Implements MIT Cheetah-style state estimation:
- 12-state Kalman filter for base position, velocity, orientation, and gyro bias
- Contact-embedded velocity integration using stance legs
- Fusion of IMU gyro and accelerometer data
"""

import numpy as np


class LinearKalmanFilter:
    """
    Linear Kalman Filter for base state estimation.
    
    State vector (12-dim):
        x = [px, py, pz, vx, vy, vz, roll, pitch, yaw, bx, by, bz]
              position    velocity   orientation   gyro_bias
    
    Parameters
    ----------
    dt : float
        Timestep (s).
    process_noise : np.ndarray, shape (12,)
        Process noise standard deviation per state.
    measurement_noise : np.ndarray, shape (6,)
        Measurement noise for [accel, gyro].
    """

    def __init__(
        self,
        dt: float = 0.001,
        process_noise: np.ndarray = None,
        measurement_noise: np.ndarray = None,
    ):
        self.dt = dt
        self.n_states = 12
        
        if process_noise is None:
            process_noise = np.array([0.01, 0.01, 0.01,  # position
                                     0.1, 0.1, 0.1,        # velocity
                                     0.01, 0.01, 0.01,     # orientation
                                     0.001, 0.001, 0.001]) # gyro bias
        if measurement_noise is None:
            measurement_noise = np.array([0.5, 0.5, 0.5,   # accel
                                          0.01, 0.01, 0.01]) # gyro
        
        self.Q = np.diag(process_noise ** 2)
        self.R = np.diag(measurement_noise ** 2)
        
        self.x = np.zeros(12)
        self.P = np.eye(12)
        
        self._init_state()

    def _init_state(self):
        """Initialize state to zero."""
        self.x = np.zeros(12)
        self.P = np.eye(12) * 0.1

    def predict(self, imu_accel: np.ndarray, contact: np.ndarray = None):
        """
        Prediction step.
        
        Parameters
        ----------
        imu_accel : np.ndarray, shape (3,)
            IMU accelerometer reading (m/s²).
        contact : np.ndarray, shape (4,)
            Contact flags per leg (1 = stance, 0 = swing).
        """
        F = self._build_state_transition()
        
        self.x = F @ self.x
        
        gravity = np.array([0, 0, 9.81])
        self.x[3:6] += (imu_accel - gravity) * self.dt
        
        self.P = F @ self.P @ F.T + self.Q

    def update(self, measured_accel: np.ndarray, measured_gyro: np.ndarray):
        """
        Update step with IMU measurements.
        
        Parameters
        ----------
        measured_accel : np.ndarray, shape (3,)
            Measured acceleration from IMU (m/s²).
        measured_gyro : np.ndarray, shape (3,)
            Measured angular velocity from IMU (rad/s).
        """
        z = np.concatenate([measured_accel, measured_gyro])
        
        H = np.zeros((6, 12))
        H[0:3, 3:6] = np.eye(3)
        H[3:6, 9:12] = -np.eye(3)
        
        y = z - H @ self.x
        
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = (np.eye(12) - K @ H) @ self.P

    def _build_state_transition(self) -> np.ndarray:
        """Build state transition matrix."""
        F = np.eye(12)
        F[0:3, 3:6] = np.eye(3) * self.dt
        F[3:6, 9:12] = -np.eye(3) * self.dt
        return F

    def get_position(self) -> np.ndarray:
        """Return estimated position (3,)."""
        return self.x[0:3].copy()

    def get_velocity(self) -> np.ndarray:
        """Return estimated velocity (3,)."""
        return self.x[3:6].copy()

    def get_orientation(self) -> np.ndarray:
        """Return estimated roll, pitch, yaw (3,)."""
        return self.x[6:9].copy()

    def get_gyro_bias(self) -> np.ndarray:
        """Return estimated gyro bias (3,)."""
        return self.x[9:12].copy()

    def reset(self):
        """Reset filter state."""
        self._init_state()


class OrientationFilter:
    """
    Complementary filter for orientation estimation.
    
    Fuses:
    - Gyro integration for high-frequency response
    - Accelerometer for low-frequency drift correction
    """

    def __init__(self, alpha: float = 0.98):
        """
        Parameters
        ----------
        alpha : float
            Complementary filter coefficient (0-1).
            Higher = more gyro, lower = more accel.
        """
        self.alpha = alpha
        self.orientation = np.zeros(3)

    def predict(self, gyro: np.ndarray, dt: float):
        """Integrate gyro for orientation prediction."""
        self.orientation += gyro * dt

    def correct(self, accel: np.ndarray):
        """Correct using accelerometer."""
        accel_norm = np.linalg.norm(accel)
        if accel_norm < 1e-6:
            return
            
        accel_unit = accel / accel_norm
        
        roll_acc = np.arctan2(accel_unit[1], accel_unit[2])
        pitch_acc = np.arctan2(-accel_unit[0], np.sqrt(accel_unit[1]**2 + accel_unit[2]**2))
        
        self.orientation[0] = self.alpha * self.orientation[0] + (1 - self.alpha) * roll_acc
        self.orientation[1] = self.alpha * self.orientation[1] + (1 - self.alpha) * pitch_acc

    def get_orientation(self) -> np.ndarray:
        """Return [roll, pitch, yaw] (3,)."""
        return self.orientation.copy()

    def reset(self):
        """Reset orientation to zero."""
        self.orientation = np.zeros(3)


class ContactEmbeddedVelocity:
    """
    Contact-embedded velocity estimation.
    
    Uses stance leg kinematics to estimate body velocity:
    v_body = J_leg^+ @ (v_foot - omega × r_foot)
    
    where J_leg^+ is the pseudoinverse of the leg Jacobian.
    """

    def __init__(self, robot_interface):
        """
        Parameters
        ----------
        robot_interface : Robot
            Robot interface for Jacobian access.
        """
        self.robot = robot_interface
        self.velocity = np.zeros(3)

    def estimate(
        self,
        contact: np.ndarray,
        foot_positions: list[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Estimate body velocity from stance legs.
        
        Parameters
        ----------
        contact : np.ndarray, shape (4,)
            Contact flags per leg.
        foot_positions : list of np.ndarray, optional
            Foot positions in world frame.
            
        Returns
        -------
        velocity : np.ndarray, shape (3,)
            Estimated body velocity.
        """
        stance_legs = np.where(contact > 0.5)[0]
        
        if len(stance_legs) == 0:
            return self.velocity
            
        velocities = []
        for leg_idx in stance_legs:
            J = self.robot.get_leg_jacobian(leg_idx)
            if J.shape[1] >= 3:
                J_pinv = np.linalg.pinv(J[:3, :3])
                foot_vel = self.robot.get_foot_velocity(leg_idx)
                leg_vel = J_pinv @ foot_vel
                velocities.append(leg_vel)
        
        if velocities:
            self.velocity = np.mean(velocities, axis=0)
        
        return self.velocity.copy()
