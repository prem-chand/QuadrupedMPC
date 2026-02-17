from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Robot(ABC):
    """
    Simulator-agnostic robot interface.

    All controller code should depend only on this interface,
    enabling backend swaps (MuJoCo, PyBullet, IsaacLab, hardware).
    """

    # ==========================
    # Simulation
    # ==========================

    @abstractmethod
    def step(self):
        """Advance simulation by one timestep."""
        ...

    @abstractmethod
    def get_time(self) -> float:
        """Return current simulation time in seconds."""
        ...

    @abstractmethod
    def set_torques(self, tau: np.ndarray):
        """Apply joint torques (12,) to actuators."""
        ...

    # ==========================
    # Base State
    # ==========================

    @abstractmethod
    def get_base_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (position (3,), quaternion (4,) [w,x,y,z])."""
        ...

    @abstractmethod
    def get_base_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (linear_velocity_world (3,), angular_velocity_body (3,))."""
        ...

    # ==========================
    # Joints
    # ==========================

    @abstractmethod
    def get_joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (joint_positions (N,), joint_velocities (N,))."""
        ...

    # ==========================
    # Feet
    # ==========================

    @abstractmethod
    def get_foot_positions_world(self) -> list[np.ndarray]:
        """Return list of 4 foot positions (3,) in world frame."""
        ...

    @abstractmethod
    def get_foot_jacobian(self, foot_index: int) -> np.ndarray:
        """
        Return the positional Jacobian for the given foot.
        Shape: (3, nv) where nv is the number of velocity DOFs.
        """
        ...

    @abstractmethod
    def get_foot_velocity(self, foot_index: int) -> np.ndarray:
        """Return Cartesian velocity (3,) of the given foot in world frame."""
        ...

    @abstractmethod
    def get_gravity_compensation(self, leg_index: int) -> np.ndarray:
        """
        Return gravity/Coriolis compensation torques (3,) for the given leg.
        These are the joint-space bias forces for the 3 DOFs of the leg.
        """
        ...

    @abstractmethod
    def get_leg_jacobian(self, foot_index: int) -> np.ndarray:
        """
        Return the (3, 3) Jacobian block mapping leg joint velocities
        to foot Cartesian velocity for the given leg.
        """
        ...
