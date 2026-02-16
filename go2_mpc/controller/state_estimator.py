import numpy as np
import mujoco

class StateEstimator:
    def __init__(self, mj_model, mj_data):
        self.model = mj_model
        self.data = mj_data
        
        # Hardcoded to ensure order matches MPC (FL, FR, RL, RR)
        # Use SITE objects (toe sites) for accurate foot positions
        self.foot_names = ["FL_toe", "FR_toe", "RL_toe", "RR_toe"]
        self.foot_ids = [
            mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in self.foot_names
        ]
        self.use_sites = True  # Flag to use site_xpos instead of xpos

        # Pre-allocated buffers for performance
        self._rot_mat = np.zeros((3, 3))
        self._state = np.zeros(12)
        self._foot_pos_rel = np.zeros((4, 3))
        self._rpy = np.zeros(3)

    def get_robot_state(self):
        # 1. Update Kinematics
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        
        # 2. Get Raw Data
        qpos = self.data.qpos
        qvel = self.data.qvel
        
        # --- POSITION ---
        pos = qpos[:3].copy()
        quat = qpos[3:7]
        rpy = self._quat_to_euler(quat)
        
        # --- ROTATION MATRIX ---
        R = self._quat_to_rot_mat(quat)

        # --- VELOCITY ---
        # Case A Confirmed: qvel[0:3] is Linear Velocity in Body Frame
        v_local = qvel[0:3]
        w_local = qvel[3:6]
        
        # Rotate to World Frame
        # Since Body Frame is aligned with Visuals (Red=Forward),
        # simple rotation R @ v should give correct World Vector.
        v_world = R @ v_local
        
        # --- STATE VECTOR ---
        # [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
        # We usually keep angular velocity in Body Frame for MPC (standard practice)
        # But some implementations use World. Let's stick to World for linear, Body for angular.
        # Fill pre-allocated buffer in-place (avoids allocation)
        self._state[0:3] = pos
        self._state[3:6] = rpy
        self._state[6:9] = v_world
        self._state[9:12] = w_local
        
        # --- FOOT POSITIONS ---
        foot_pos_rel = self._get_foot_pos_rel(pos)

        return self._state, foot_pos_rel

    def _get_foot_pos_rel(self, com_pos_world):
        """Calculates foot positions relative to CoM in World Frame."""
        for i in range(4):
            # Use site_xpos for site objects
            f_pos_world = self.data.site_xpos[self.foot_ids[i]]
            # MPC needs vector r = p_foot - p_com (in-place update)
            self._foot_pos_rel[i, :] = f_pos_world - com_pos_world

        return self._foot_pos_rel

    def _quat_to_euler(self, q):
        """Converts [w,x,y,z] to [roll, pitch, yaw] (reuses internal buffer)"""
        w, x, y, z = q
        # Standard conversion
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        self._rpy[0] = np.arctan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = np.clip(t2, -1.0, 1.0)
        self._rpy[1] = np.arcsin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        self._rpy[2] = np.arctan2(t3, t4)
        return self._rpy
        
    def _quat_to_rot_mat(self, q):
        """Converts quaternion to 3x3 Rotation Matrix (reuses internal buffer)"""
        w, x, y, z = q
        self._rot_mat[0, 0] = 1 - 2*(y**2 + z**2)
        self._rot_mat[0, 1] = 2*(x*y - z*w)
        self._rot_mat[0, 2] = 2*(x*z + y*w)
        self._rot_mat[1, 0] = 2*(x*y + z*w)
        self._rot_mat[1, 1] = 1 - 2*(x**2 + z**2)
        self._rot_mat[1, 2] = 2*(y*z - x*w)
        self._rot_mat[2, 0] = 2*(x*z - y*w)
        self._rot_mat[2, 1] = 2*(y*z + x*w)
        self._rot_mat[2, 2] = 1 - 2*(x**2 + y**2)
        return self._rot_mat