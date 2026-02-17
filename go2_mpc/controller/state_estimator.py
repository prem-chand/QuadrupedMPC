import numpy as np
from dataclasses import dataclass
from go2_mpc.core.state import BaseState, JointState, State


class StateEstimator:
    def __init__(self, robot):
        self.robot = robot

        self._rot_mat = np.zeros((3, 3))
        self._foot_pos_rel = np.zeros((4, 3))

    def estimate(self):
        """
        Returns:
            structured State
            foot_pos_rel (4,3) in world frame
        """

        self.robot.forward_kinematics()

        pos, quat = self.robot.get_base_pose()
        v_world, w_body = self.robot.get_base_velocity()

        # Normalize quaternion (safety)
        norm = np.linalg.norm(quat)
        if norm > 0:
            quat = quat / norm

        # Build structured State
        base = BaseState(
            position=pos,
            orientation=quat,
            linear_velocity=v_world,
            angular_velocity=w_body,
        )

        # Rotation matrix
        R = base.rotation_matrix

        # Convert linear velocity to body frame
        v_body = R.T @ v_world

        # Joints come directly from robot if needed
        joints = JointState(
            positions=self.robot.data.qpos[7:19],
            velocities=self.robot.data.qvel[6:18],
        )

        state = State(base=base, joints=joints)

        # Foot positions relative to COM
        foot_world = self.robot.get_foot_positions_world()

        for i in range(4):
            self._foot_pos_rel[i] = foot_world[i] - pos

        return state, self._foot_pos_rel
