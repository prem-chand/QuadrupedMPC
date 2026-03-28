"""
Comprehensive data logging for quadruped controller debugging.

Logs:
- Timestep, base state (pos, vel, rpy)
- Joint state (q, qd)
- Foot positions (world & body frame)
- Contact schedule & forces
- MPC input/output
- Torques applied
- Errors/warnings
"""

import os
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class LogEntry:
    """Single timestep log entry."""
    t: float
    step: int
    
    base_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    base_quat: np.ndarray = field(default_factory=lambda: np.zeros(4))
    base_rpy: np.ndarray = field(default_factory=lambda: np.zeros(3))
    base_lin_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    base_ang_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    joint_q: np.ndarray = field(default_factory=lambda: np.zeros(12))
    joint_qd: np.ndarray = field(default_factory=lambda: np.zeros(12))
    
    foot_pos_world: np.ndarray = field(default_factory=lambda: np.zeros((4, 3)))
    foot_pos_body: np.ndarray = field(default_factory=lambda: np.zeros((4, 3)))
    foot_vel_world: np.ndarray = field(default_factory=lambda: np.zeros((4, 3)))
    
    contact_schedule: np.ndarray = field(default_factory=lambda: np.zeros(4))
    contact_forces: np.ndarray = field(default_factory=lambda: np.zeros((4, 3)))
    
    gait_phase: float = 0.0
    
    v_cmd: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw_rate_cmd: float = 0.0
    
    mpc_forces: np.ndarray = field(default_factory=lambda: np.zeros((4, 3)))
    mpc_solve_time: float = 0.0
    
    tau: np.ndarray = field(default_factory=lambda: np.zeros(12))
    
    has_nan: bool = False
    error_msg: str = ""


class DataLogger:
    """Comprehensive data logger for controller debugging."""
    
    def __init__(self, log_dir: str = "logs", max_steps: int = 100000):
        self.max_steps = max_steps
        self.entries: list[LogEntry] = []
        self.enabled = True
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(log_dir, timestamp)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self._stats = {
            'total_steps': 0,
            'nan_count': 0,
            'mpc_failures': 0,
            'wbc_failures': 0,
        }
    
    def log(self, entry: LogEntry):
        """Log a single timestep."""
        if not self.enabled:
            return
        
        self.entries.append(entry)
        self._stats['total_steps'] += 1
        
        if entry.has_nan:
            self._stats['nan_count'] += 1
        
        if len(self.entries) >= self.max_steps:
            self.flush()
    
    def check_nan(self, *arrays) -> tuple[bool, str]:
        """Check for NaN/Inf in arrays."""
        for arr in arrays:
            if arr is None:
                continue
            arr = np.asarray(arr)
            if np.any(np.isnan(arr)):
                return True, f"NaN in {type(arr).__name__}"
            if np.any(np.isinf(arr)):
                return True, f"Inf in {type(arr).__name__}"
        return False, ""
    
    def flush(self):
        """Write logs to disk."""
        if not self.entries:
            return
        
        print(f"[DataLogger] Flushing {len(self.entries)} entries to {self.log_dir}")
        
        # Save as numpy for efficiency
        n = len(self.entries)
        
        data = {
            't': np.array([e.t for e in self.entries]),
            'step': np.array([e.step for e in self.entries]),
            
            'base_pos': np.array([e.base_pos for e in self.entries]),
            'base_quat': np.array([e.base_quat for e in self.entries]),
            'base_rpy': np.array([e.base_rpy for e in self.entries]),
            'base_lin_vel': np.array([e.base_lin_vel for e in self.entries]),
            'base_ang_vel': np.array([e.base_ang_vel for e in self.entries]),
            
            'joint_q': np.array([e.joint_q for e in self.entries]),
            'joint_qd': np.array([e.joint_qd for e in self.entries]),
            
            'foot_pos_world': np.array([e.foot_pos_world for e in self.entries]),
            'foot_pos_body': np.array([e.foot_pos_body for e in self.entries]),
            'foot_vel_world': np.array([e.foot_vel_world for e in self.entries]),
            
            'contact_schedule': np.array([e.contact_schedule for e in self.entries]),
            'contact_forces': np.array([e.contact_forces for e in self.entries]),
            
            'gait_phase': np.array([e.gait_phase for e in self.entries]),
            
            'v_cmd': np.array([e.v_cmd for e in self.entries]),
            'yaw_rate_cmd': np.array([e.yaw_rate_cmd for e in self.entries]),
            
            'mpc_forces': np.array([e.mpc_forces for e in self.entries]),
            'mpc_solve_time': np.array([e.mpc_solve_time for e in self.entries]),
            
            'tau': np.array([e.tau for e in self.entries]),
        }
        
        np.savez_compressed(
            os.path.join(self.log_dir, 'trajectory.npz'),
            **data
        )
        
        # Save stats
        with open(os.path.join(self.log_dir, 'stats.json'), 'w') as f:
            json.dump(self._stats, f, indent=2)
        
        # Clear entries
        self.entries.clear()
        print(f"[DataLogger] Saved. Stats: {self._stats}")
    
    def close(self):
        """Final flush and close."""
        self.flush()
        print(f"[DataLogger] Closed. Log dir: {self.log_dir}")


def create_log_entry(
    t: float,
    step: int,
    robot,
    state,
    command,
    contact_schedule,
    gait_phase,
    mpc_forces,
    mpc_solve_time,
    tau,
) -> LogEntry:
    """Create a LogEntry from current state."""
    
    base_pos, base_quat = robot.get_base_pose()
    base_rpy = np.array([state.base.roll, state.base.pitch, state.base.yaw])
    base_lin_vel, base_ang_vel = robot.get_base_velocity()
    
    joint_q, joint_qd = robot.get_joint_state()
    
    foot_pos_world = np.array(robot.get_foot_positions_world())
    foot_pos_body = np.array(robot.get_foot_positions_body())
    foot_vel_world = np.array(robot.get_foot_velocity())
    
    contact_forces = np.array(robot.get_foot_forces())
    
    has_nan, error_msg = check_state(
        base_pos, base_quat, base_lin_vel, base_ang_vel,
        joint_q, joint_qd, foot_pos_world, tau
    )
    
    return LogEntry(
        t=t,
        step=step,
        base_pos=base_pos,
        base_quat=base_quat,
        base_rpy=base_rpy,
        base_lin_vel=base_lin_vel,
        base_ang_vel=base_ang_vel,
        joint_q=joint_q,
        joint_qd=joint_qd,
        foot_pos_world=foot_pos_world,
        foot_pos_body=foot_pos_body,
        foot_vel_world=foot_vel_world,
        contact_schedule=contact_schedule,
        contact_forces=contact_forces,
        gait_phase=gait_phase,
        v_cmd=command.v_cmd_global,
        yaw_rate_cmd=command.yaw_rate,
        mpc_forces=mpc_forces,
        mpc_solve_time=mpc_solve_time,
        tau=tau,
        has_nan=has_nan,
        error_msg=error_msg,
    )


def check_state(*arrays) -> tuple[bool, str]:
    """Check for NaN/Inf in state arrays."""
    for arr in arrays:
        if arr is None:
            continue
        arr = np.asarray(arr)
        if np.any(np.isnan(arr)):
            where = np.argwhere(np.isnan(arr))
            return True, f"NaN at indices {where[:3].tolist()}"
        if np.any(np.isinf(arr)):
            return True, "Inf detected"
    return False, ""
