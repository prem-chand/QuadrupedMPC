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

        # Preallocate jacobian buffers
        self._J = np.zeros((3, model.nv))
        self._Jr = np.zeros((3, model.nv))

    # ==========================
    # Simulation
    # ==========================

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def forward_kinematics(self):
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

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

    def get_joint_state(self):
        qpos = self.data.qpos
        qvel = self.data.qvel
        return qpos[7:], qvel[6:]

    # ==========================
    # Feet
    # ==========================

    def get_foot_positions_world(self):
        return [
            self.data.site_xpos[self.foot_ids[i]].copy()
            for i in range(4)
        ]

    def get_foot_jacobian_full(self, foot_index):
        mujoco.mj_jacSite(
            self.model,
            self.data,
            self._J,
            self._Jr,
            self.foot_ids[foot_index],
        )
        return self._J
