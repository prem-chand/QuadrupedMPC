from .robot import Robot
import numpy as np
import mujoco


class MujocoRobot(Robot):
    def __init__(self, model, data):
        self.model = model
        self.data = data

        # Discover foot sites internally
        self.foot_names = ["FL_toe", "FR_toe", "RL_toe", "RR_toe"]
        self.foot_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in self.foot_names
        ]

        # Leg DOF indices in qvel (after 6 floating base DOFs)
        self.leg_dof_indices = [
            [6, 7, 8],      # FL
            [9, 10, 11],     # FR
            [12, 13, 14],    # RL
            [15, 16, 17],    # RR
        ]

        # Preallocate jacobian buffers
        self._J = np.zeros((3, model.nv))
        self._Jr = np.zeros((3, model.nv))

    # ==========================
    # Simulation
    # ==========================

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def get_time(self):
        return self.data.time

    def set_torques(self, tau):
        self.data.ctrl[:] = tau

    # ==========================
    # Base State
    # ==========================

    def get_base_pose(self):
        qpos = self.data.qpos
        return qpos[:3], qpos[3:7]

    def get_base_velocity(self):
        qvel = self.data.qvel
        return qvel[0:3], qvel[3:6]

    # ==========================
    # Joints
    # ==========================

    def get_joint_state(self):
        return self.data.qpos[7:], self.data.qvel[6:]

    # ==========================
    # Feet
    # ==========================

    def get_foot_positions_world(self):
        return [
            self.data.site_xpos[self.foot_ids[i]].copy()
            for i in range(4)
        ]

    def get_foot_jacobian(self, foot_index):
        """Full (3, nv) positional Jacobian for the given foot."""
        mujoco.mj_jacSite(
            self.model,
            self.data,
            self._J,
            self._Jr,
            self.foot_ids[foot_index],
        )
        return self._J.copy()

    def get_leg_jacobian(self, foot_index):
        """(3, 3) Jacobian block for leg joints only."""
        mujoco.mj_jacSite(
            self.model,
            self.data,
            self._J,
            self._Jr,
            self.foot_ids[foot_index],
        )
        dofs = self.leg_dof_indices[foot_index]
        return self._J[:, dofs].copy()

    def get_foot_velocity(self, foot_index):
        """Cartesian velocity (3,) of the foot in world frame."""
        mujoco.mj_jacSite(
            self.model,
            self.data,
            self._J,
            self._Jr,
            self.foot_ids[foot_index],
        )
        return self._J @ self.data.qvel

    def get_gravity_compensation(self, leg_index):
        """Gravity/Coriolis torques (3,) for the 3 DOFs of the given leg."""
        dofs = self.leg_dof_indices[leg_index]
        return self.data.qfrc_bias[dofs].copy()

    # ==========================
    # Legacy (kept for backward compat during transition)
    # ==========================

    def get_foot_jacobian_full(self, foot_index):
        """Deprecated: use get_foot_jacobian() or get_leg_jacobian() instead."""
        return self.get_foot_jacobian(foot_index)
