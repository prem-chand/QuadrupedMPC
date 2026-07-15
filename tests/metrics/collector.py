"""Metrics collector for walking quality tests"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
import time


@dataclass
class TestMetrics:
    """Container for collected test metrics"""
    # Time series data
    timestamps: List[float] = field(default_factory=list)
    
    # Position
    positions: List[np.ndarray] = field(default_factory=list)  # (3,) world pos
    heights: List[float] = field(default_factory=list)  # z position
    
    # Orientation (rpy)
    rolls: List[float] = field(default_factory=list)
    pitches: List[float] = field(default_factory=list)
    yaws: List[float] = field(default_factory=list)
    
    # Velocity
    linear_velocities: List[np.ndarray] = field(default_factory=list)  # (3,)
    angular_velocities: List[np.ndarray] = field(default_factory=list)  # (3,)
    
    # Commands
    commands: List[tuple] = field(default_factory=list)  # (vx, vy, yaw_rate)
    
    # Forces
    mpc_forces: List[np.ndarray] = field(default_factory=list)  # (12,)
    contact_schedule: List[np.ndarray] = field(default_factory=list)  # (4,)
    
    # Torques
    torques: List[np.ndarray] = field(default_factory=list)  # (12,)
    
    # Events
    fall_detected: bool = False
    fall_time: Optional[float] = None
    
    # Test info
    test_name: str = ""
    start_time: float = 0.0
    
    def add_sample(self, t: float, pos: np.ndarray, rpy: np.ndarray, 
                   lin_vel: np.ndarray, ang_vel: np.ndarray,
                   command: tuple, mpc_forces: np.ndarray,
                   contact: np.ndarray, torques: np.ndarray):
        """Add a sample to the metrics"""
        self.timestamps.append(t)
        self.positions.append(pos.copy())
        self.heights.append(pos[2])
        self.rolls.append(rpy[0])
        self.pitches.append(rpy[1])
        self.yaws.append(rpy[2])
        self.linear_velocities.append(lin_vel.copy())
        self.angular_velocities.append(ang_vel.copy())
        self.commands.append(command)
        self.mpc_forces.append(mpc_forces.copy())
        self.contact_schedule.append(contact.copy())
        self.torques.append(torques.copy())
    
    def detect_fall(self, height_threshold: float = 0.15) -> bool:
        """Detect if robot fell during test"""
        if len(self.heights) == 0:
            return False
        
        # Check if height dropped below threshold
        if any(h < height_threshold for h in self.heights):
            self.fall_detected = True
            # Find when fall occurred
            for i, h in enumerate(self.heights):
                if h < height_threshold:
                    self.fall_time = self.timestamps[i]
                    break
            return True
        return False
    
    def get_summary(self) -> dict:
        """Get summary statistics"""
        if len(self.timestamps) == 0:
            return {}
        
        heights = np.array(self.heights)
        rolls = np.array(self.rolls)
        pitches = np.array(self.pitches)
        yaws = np.array(self.yaws)
        torques = np.array(self.torques)
        lin_vels = np.array(self.linear_velocities)
        
        # Compute velocity error if commands were given
        velocity_errors = []
        for cmd, vel in zip(self.commands, lin_vels):
            if cmd is not None and any(cmd):
                vx_cmd, vy_cmd, _ = cmd
                vx_actual = vel[0]
                vy_actual = vel[1]
                velocity_errors.append(np.sqrt((vx_cmd - vx_actual)**2 + (vy_cmd - vy_actual)**2))
        
        return {
            "test_name": self.test_name,
            "duration": self.timestamps[-1] - self.timestamps[0] if len(self.timestamps) > 1 else 0,
            "samples": len(self.timestamps),
            
            # Height
            "height_mean": np.mean(heights),
            "height_std": np.std(heights),
            "height_min": np.min(heights),
            "height_max": np.max(heights),
            
            # Orientation
            "roll_mean": np.mean(rolls),
            "roll_std": np.std(rolls),
            "roll_max": np.max(np.abs(rolls)),
            "pitch_mean": np.mean(pitches),
            "pitch_std": np.std(pitches),
            "pitch_max": np.max(np.abs(pitches)),
            "yaw_total": yaws[-1] - yaws[0] if len(yaws) > 1 else 0,
            
            # Velocity
            "vx_mean": np.mean(lin_vels[:, 0]),
            "vy_mean": np.mean(lin_vels[:, 1]),
            "vz_mean": np.mean(lin_vels[:, 2]),
            "velocity_error_mean": np.mean(velocity_errors) if velocity_errors else 0,
            "velocity_error_max": np.max(velocity_errors) if velocity_errors else 0,
            
            # Angular velocity
            "wx_mean": np.mean([v[0] for v in self.angular_velocities]),
            "wy_mean": np.mean([v[1] for v in self.angular_velocities]),
            "wz_mean": np.mean([v[2] for v in self.angular_velocities]),
            
            # Torques
            "torque_max": np.max(np.abs(torques)),
            "torque_mean": np.mean(np.abs(torques)),
            
            # Events
            "fall_detected": self.fall_detected,
            "fall_time": self.fall_time,
        }


class MetricsCollector:
    """Collects metrics during simulation runs"""
    
    def __init__(self, test_name: str = "test"):
        self.test_name = test_name
        self.metrics = TestMetrics(test_name=test_name)
        self.collecting = False
        self.start_walltime = 0.0
        self.start_simulation_time = 0.0
    
    def start(self, sim_time: float = 0.0):
        """Start collecting metrics"""
        self.metrics = TestMetrics(test_name=self.test_name)
        self.collecting = True
        self.start_walltime = time.time()
        self.start_simulation_time = sim_time
    
    def stop(self):
        """Stop collecting metrics"""
        self.collecting = False
    
    def sample(self, sim_time: float, robot, state, command, 
               mpc_forces: np.ndarray, contact_schedule: np.ndarray,
               torques: np.ndarray):
        """Take a sample of current metrics"""
        if not self.collecting:
            return
        
        # Get robot state
        pos, _ = robot.get_base_pose()
        lin_vel, ang_vel = robot.get_base_velocity()
        
        # Get RPY from state
        rpy = np.array([state.base.roll, state.base.pitch, state.base.yaw])
        
        # Extract command values
        cmd = (command.v_cmd_global[0], command.v_cmd_global[1], command.yaw_rate)
        
        self.metrics.add_sample(
            t=sim_time - self.start_simulation_time,
            pos=pos,
            rpy=rpy,
            lin_vel=lin_vel,
            ang_vel=ang_vel,
            command=cmd,
            mpc_forces=mpc_forces,
            contact=contact_schedule,
            torques=torques,
        )
    
    def get_metrics(self) -> TestMetrics:
        """Get collected metrics"""
        return self.metrics
    
    def get_summary(self) -> dict:
        """Get metrics summary"""
        return self.metrics.get_summary()
