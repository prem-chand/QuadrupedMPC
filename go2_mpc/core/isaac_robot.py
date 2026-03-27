"""
IsaacLab backend implementation of the Robot interface.

This module wraps an IsaacLab Articulation and exposes all quantities
through the simulator-agnostic :class:`Robot` ABC.

IsaacLab is used for:
  - GPU-accelerated parallel simulation
  - PhysX physics backend
  - Integration with RL training pipelines (RSL-RL)

All kinematics (foot positions, Jacobians, foot velocities) are
computed analytically by :class:`~go2_mpc.kinematics.Go2Kinematics`,
matching the approach used in :class:`MujocoRobot`.

IsaacLab frame conventions (matching MuJoCo):
  - root_pose_w[:, :3] — base position in world frame (m)
  - root_pose_w[:, 3:7] — base orientation as quaternion [w, x, y, z]
  - root_lin_vel_b — base linear velocity in BODY frame (m/s)
  - root_ang_vel_b — base angular velocity in BODY frame (rad/s)
  - joint_pos — joint positions (rad), ordered by USD/Joints definition
  - joint_vel — joint velocities (rad/s)

Leg ordering: [FL, FR, RL, RR], each with 3 DOFs
[hip_abduction, hip_flexion, knee_flexion].
"""

from .robot import Robot
import numpy as np
from go2_mpc.kinematics import Go2Kinematics


class IsaacRobot(Robot):
    """Concrete :class:`Robot` backed by IsaacLab Articulation.

    Parameters
    ----------
    articulation : isaaclab.assets.Articulation
        IsaacLab articulation asset (e.g., Unitree Go2 USD).
    sim : isaaclab.sim.SimulationContext
        IsaacLab simulation context for stepping.
    """

    def __init__(self, articulation, sim):
        self._articulation = articulation
        self._sim = sim
        self._dt = sim.get_physics_dt()

        self._kin = Go2Kinematics()

    def step(self):
        """Advance the simulation by one timestep."""
        self._sim.step()
        self._articulation.update(self._dt)

    def get_time(self) -> float:
        """Current simulation time (s)."""
        return self._sim.current_time

    def set_torques(self, tau: np.ndarray):
        """Apply joint torques to all 12 actuators.

        Parameters
        ----------
        tau : np.ndarray, shape (12,)
            Desired torques (N·m), ordered [FL(3), FR(3), RL(3), RR(3)].
        """
        self._articulation.set_joint_effort_target(tau)
        self._articulation.write_data_to_sim()

    def get_base_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return base pose from Articulation data.

        Returns
        -------
        position : np.ndarray, shape (3,)
            CoM position in world frame (m).
        quaternion : np.ndarray, shape (4,)
            Orientation [w, x, y, z] (scalar-first).
        """
        root_pose = self._articulation.data.root_pose_w[0]
        position = root_pose[:3].copy()
        quaternion = root_pose[3:7].copy()
        return position, quaternion

    def get_base_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        """Return base velocities from Articulation data.

        Returns
        -------
        linear_velocity : np.ndarray, shape (3,)
            Translational velocity in world frame (m/s).
        angular_velocity : np.ndarray, shape (3,)
            Rotational velocity in body frame (rad/s).
        """
        lin_vel = self._articulation.data.root_lin_vel_b[0].copy()
        ang_vel = self._articulation.data.root_ang_vel_b[0].copy()
        return lin_vel, ang_vel

    def get_joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Return joint positions and velocities for all 12 actuated DOFs.

        Returns
        -------
        positions : np.ndarray, shape (12,)
            Joint angles (rad).
        velocities : np.ndarray, shape (12,)
            Joint velocities (rad/s).
        """
        joint_pos = self._articulation.data.joint_pos[0].copy()
        joint_vel = self._articulation.data.joint_vel[0].copy()
        return joint_pos, joint_vel

    def _get_base_state(self):
        """Extract (p_base, R_base, v_base, omega_body) from articulation data.

        Computes the rotation matrix from quaternion.
        """
        p_base, quat = self.get_base_pose()
        v_base, omega_body = self.get_base_velocity()

        w, x, y, z = quat
        n = np.sqrt(w*w + x*x + y*y + z*z)
        w, x, y, z = w/n, x/n, y/n, z/n
        R_base = np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
            [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
            [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
        ], dtype=np.float64)

        return p_base, R_base, v_base, omega_body

    def get_foot_positions_world(self) -> list[np.ndarray]:
        """Foot positions via analytical forward kinematics.

        Returns
        -------
        list of np.ndarray, each shape (3,)
            [FL, FR, RL, RR] foot positions in world frame (m).
        """
        p_base, R_base, _, _ = self._get_base_state()
        q_joints, _ = self.get_joint_state()
        return [
            self._kin.foot_position_world(i, p_base, R_base, q_joints[3*i:3*i+3])
            for i in range(4)
        ]

    def get_foot_jacobian(self, foot_index: int) -> np.ndarray:
        """Full positional Jacobian for a foot.

        Maps generalised velocities (18-dim) to foot Cartesian velocity
        in world frame: v_foot = J @ qvel.

        Parameters
        ----------
        foot_index : int
            Leg index in [0=FL, 1=FR, 2=RL, 3=RR].

        Returns
        -------
        np.ndarray, shape (3, 18)
            Full positional Jacobian in world frame.
        """
        p_base, R_base, _, _ = self._get_base_state()
        q_joints, _ = self.get_joint_state()
        return self._kin.full_jacobian(
            foot_index, p_base, R_base, q_joints[3*foot_index:3*foot_index+3]
        )

    def get_leg_jacobian(self, foot_index: int) -> np.ndarray:
        """Leg-local Jacobian: 3 joint DOFs → foot Cartesian velocity.

        Used by WBC and swing control for the torque mapping
        tau_leg = J_leg^T @ F_foot.

        Parameters
        ----------
        foot_index : int
            Leg index in [0=FL, 1=FR, 2=RL, 3=RR].

        Returns
        -------
        np.ndarray, shape (3, 3)
            Jacobian in world frame.
        """
        p_base, R_base, _, _ = self._get_base_state()
        q_joints, _ = self.get_joint_state()
        return self._kin.leg_jacobian(
            foot_index, p_base, R_base, q_joints[3*foot_index:3*foot_index+3]
        )

    def get_foot_velocity(self, foot_index: int) -> np.ndarray:
        """Cartesian velocity of a foot in world frame.

        Includes base translational, base rotational, and joint velocity
        contributions.

        Parameters
        ----------
        foot_index : int
            Leg index in [0=FL, 1=FR, 2=RL, 3=RR].

        Returns
        -------
        np.ndarray, shape (3,)
            Translational velocity in world frame (m/s).
        """
        p_base, R_base, v_base, omega_body = self._get_base_state()
        q_joints, qd_joints = self.get_joint_state()
        return self._kin.foot_velocity(
            foot_index,
            p_base, R_base,
            q_joints[3*foot_index:3*foot_index+3],
            v_base,
            omega_body,
            qd_joints[3*foot_index:3*foot_index+3],
        )

    def get_gravity_compensation(self, leg_index: int) -> np.ndarray:
        """Static gravity compensation torques for one leg.

        Computed analytically from the link CoM positions and masses.

        Parameters
        ----------
        leg_index : int
            Leg index in [0=FL, 1=FR, 2=RL, 3=RR].

        Returns
        -------
        np.ndarray, shape (3,)
            Bias torques (N·m) for [hip_abd, hip_flex, knee].
        """
        p_base, R_base, _, _ = self._get_base_state()
        q_joints, _ = self.get_joint_state()
        return self._kin.gravity_compensation(
            leg_index, p_base, R_base, q_joints[3*leg_index:3*leg_index+3]
        )
