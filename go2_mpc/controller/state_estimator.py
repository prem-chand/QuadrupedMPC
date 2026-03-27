import numpy as np
from go2_mpc.core.state import BaseState, JointState, State


class StateEstimator:
    def __init__(self, robot):
        self.robot = robot

        self._foot_pos_rel = np.zeros((4, 3))

    def estimate(self):
        """
        Returns:
            structured State
            foot_pos_rel (4,3) in world frame relative to base
        """
        # mj_step already computes full forward kinematics, no need to call again

        pos, quat = self.robot.get_base_pose()
        v_world, w_body = self.robot.get_base_velocity()

        # Normalize quaternion (safety)
        norm = np.linalg.norm(quat)
        if norm > 0:
            quat = quat / norm

        base = BaseState(
            position=pos,
            orientation=quat,
            linear_velocity=v_world,
            angular_velocity=w_body,
        )

        joints = JointState(
            positions=self.robot.get_joint_state()[0],
            velocities=self.robot.get_joint_state()[1],
        )

        state = State(base=base, joints=joints)

        # Foot positions relative to base
        foot_world = self.robot.get_foot_positions_world()
        for i in range(4):
            self._foot_pos_rel[i] = foot_world[i] - pos

        return state, self._foot_pos_rel
