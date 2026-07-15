from dataclasses import dataclass
import numpy as np
from .foot_swing_trajectory import FootSwingTrajectory

@dataclass
class ControllerState:
    step_counter: int          # Counts 1kHz steps
    mpc_counter: int           # Counts MPC solves (at 100Hz rate)
    gait_phase_time: float     # Tracks continuous gait time
    swing_active: np.ndarray   # [4,] bool mask
    swing_start_pos: list      # List of 4 np.arrays (World Frame)
    swing_target_pos: list     # List of 4 np.arrays — frozen landing targets (World Frame)

class ControllerBuffers:
    def __init__(self):
        self.current_forces = np.zeros(12)
        self.smoothed_forces = np.zeros(12)
        self.tau_stance = np.zeros(12)
        self.tau_swing = np.zeros(12)
        self.tau_final = np.zeros(12)
        self.contact_schedule = np.zeros((10, 4))

class ControllerCore:
    def __init__(self, gait, traj_gen, mpc, wbc, config):
        self.gait = gait
        self.traj_gen = traj_gen
        self.mpc = mpc
        self.wbc = wbc
        
        # Timing Constants
        self.sim_dt = config.get("SIM_DT", 0.001)
        self.control_decimation = config.get("CONTROL_DECIMATION", 10)
        self.mpc_decimation = config.get("MPC_DECIMATION", 3)
        self.force_alpha = config.get("FORCE_SMOOTH_ALPHA", 0.1)
        self.torque_limit = config.get("TORQUE_LIMIT", 35.0)

        # Swing Configuration
        self.swing_trajs = [FootSwingTrajectory() for _ in range(4)]
        self.kp_swing = config.get("SWING_KP", 400.0)
        self.kd_swing = config.get("SWING_KD", 10.0)

        # Nominal foot stance positions (body frame) for Raibert foot placement
        self.foot_stance_offsets = config.get("FOOT_STANCE_OFFSETS", np.array([
            [ 0.1934,  0.142, 0.0],
            [ 0.1934, -0.142, 0.0],
            [-0.1934,  0.142, 0.0],
            [-0.1934, -0.142, 0.0],
        ]))

    def compute(self, state, foot_pos_rel, command, controller_state, buffers, robot_interface):
        """Main control loop orchestration."""
        controller_state.step_counter += 1
        controller_state.gait_phase_time += self.sim_dt

        # 1. Low Frequency Loop (100 Hz) - WBC & MPC
        if controller_state.step_counter % self.control_decimation == 0 or controller_state.mpc_counter == 0:
            self._run_low_freq_loop(state, foot_pos_rel, command, controller_state, buffers, robot_interface)

        # 2. High Frequency Loop (1 kHz) - Swing Control
        self._run_high_freq_loop(controller_state, buffers, robot_interface)

        # 3. Merge & Clip
        return self._merge_and_clip_torques(controller_state, buffers)

    def _run_low_freq_loop(self, state, foot_pos_rel, command, controller_state, buffers, robot_interface):
        """Handles gait scheduling, swing/stance transitions, and MPC solves."""
        contact_schedule = self.gait.get_contact_schedule(controller_state.gait_phase_time)
        np.copyto(buffers.contact_schedule, contact_schedule)
        current_contact = contact_schedule[0, :]
        
        # Yaw-aligned rotation matrices
        cos_y, sin_y = np.cos(state.base.yaw), np.sin(state.base.yaw)
        R_z_T = np.array([[cos_y, sin_y, 0], [-sin_y, cos_y, 0], [0, 0, 1]])
        R_z = R_z_T.T

        self._update_swing_states(state, command, controller_state, robot_interface, current_contact, R_z, R_z_T)

        if controller_state.mpc_counter % self.mpc_decimation == 0:
            self._solve_mpc(state, foot_pos_rel, command, buffers, R_z_T)

        controller_state.mpc_counter += 1
        
        # WBC with EMA force smoothing
        buffers.smoothed_forces = self.force_alpha * buffers.smoothed_forces + (1 - self.force_alpha) * buffers.current_forces
        forces_list = [buffers.smoothed_forces[3*i : 3*i+3].copy() for i in range(4)]
        buffers.tau_stance[:] = self.wbc.compute_torques(forces_list, gravity_comp=True)

    def _update_swing_states(self, state, command, controller_state, robot_interface, current_contact, R_z, R_z_T):
        """Updates swing/stance state and computes Raibert landing targets."""
        foot_pos_world = robot_interface.get_foot_positions_world()
        v_body = R_z_T @ state.base.linear_velocity
        v_cmd_body = R_z_T @ command.v_cmd_global

        for i in range(4):
            if current_contact[i] == 1:  # Stance
                controller_state.swing_active[i] = 0
                controller_state.swing_start_pos[i][:] = foot_pos_world[i]
            elif not controller_state.swing_active[i]:  # Swing onset
                raibert_offset = v_body[0:2] * self.gait.period * 0.5 + 0.1 * (v_cmd_body[0:2] - v_body[0:2])
                off_world = R_z @ np.array([raibert_offset[0], raibert_offset[1], 0.0])
                p_stance_world = state.base.position + R_z @ self.foot_stance_offsets[i]
                
                pf = p_stance_world.copy()
                pf[0:2] += off_world[0:2]
                pf[2] = controller_state.swing_start_pos[i][2]
                controller_state.swing_target_pos[i][:] = pf
                controller_state.swing_active[i] = 1

    def _solve_mpc(self, state, foot_pos_rel, command, buffers, R_z_T):
        """Formulates and solves the convex MPC problem."""
        ref = self.traj_gen.generate_reference(
            state.base.to_mpc_vector(), command.v_cmd_global, command.yaw_rate, command.default_height
        )
        foot_pos_body = [R_z_T @ foot_pos_rel[i] for i in range(4)]
        forces = self.mpc.solve(state, ref, buffers.contact_schedule, foot_pos_body)
        np.copyto(buffers.current_forces, forces)

    def _run_high_freq_loop(self, controller_state, buffers, robot_interface):
        """Computes Cartesian PD swing torques for legs in swing phase."""
        buffers.tau_swing.fill(0.0)
        swing_duration = self.gait.period * (1.0 - self.gait.stance_ratio)
        foot_pos_world = robot_interface.get_foot_positions_world()

        for i in range(4):
            if controller_state.swing_active[i]:
                t_swing = self.gait.get_swing_state(controller_state.gait_phase_time, i)
                self.swing_trajs[i].set_initial_position(controller_state.swing_start_pos[i])
                self.swing_trajs[i].set_final_position(controller_state.swing_target_pos[i])
                self.swing_trajs[i].set_height(0.10)
                self.swing_trajs[i].compute_swing_trajectory_bezier(t_swing, swing_duration)

                F_swing = self.kp_swing * (self.swing_trajs[i].get_position() - foot_pos_world[i]) + \
                          self.kd_swing * (self.swing_trajs[i].get_velocity() - robot_interface.get_foot_velocity(i))
                
                buffers.tau_swing[3*i : 3*i+3] = robot_interface.get_leg_jacobian(i).T @ F_swing + \
                                                 robot_interface.get_gravity_compensation(i)

    def _merge_and_clip_torques(self, controller_state, buffers):
        """Merges stance and swing torques and applies safety limits."""
        for i in range(4):
            idx = slice(3*i, 3*i+3)
            buffers.tau_final[idx] = buffers.tau_swing[idx] if controller_state.swing_active[i] else buffers.tau_stance[idx]

        np.clip(buffers.tau_final, -self.torque_limit, self.torque_limit, out=buffers.tau_final)
        return buffers.tau_final, {
            'contact_schedule': buffers.contact_schedule.copy(),
            'swing_active': controller_state.swing_active.copy(),
            'current_forces': buffers.current_forces.copy(),
            'smoothed_forces': buffers.smoothed_forces.copy(),
        }