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
        self.sim_dt = 0.001           # 1 kHz
        self.control_decimation = 10  # 1 kHz -> 100 Hz (WBC loop)
        self.mpc_decimation = 3       # 100 Hz -> 33 Hz (MPC loop)
        
        self.force_alpha = config.get("FORCE_SMOOTH_ALPHA", 0.9)
        self.torque_limit = config.get("TORQUE_LIMIT", 35.0)

        # Swing Configuration
        self.swing_trajs = [FootSwingTrajectory() for _ in range(4)]
        self.kp_swing = config.get("SWING_KP", 400.0)
        self.kd_swing = config.get("SWING_KD", 10.0)

    def compute(self, state, foot_pos_rel, command, controller_state, buffers, robot_interface):
        """
        Args:
            robot_interface: Instance implementing the Robot interface.
        """
        
        # 1. Update Global Time
        controller_state.step_counter += 1
        controller_state.gait_phase_time += self.sim_dt

        # ======================================================
        # LOW FREQUENCY LOOP (100 Hz) - WBC & MPC
        # ======================================================
        # Only run this block every 10 simulation steps
        if controller_state.step_counter % self.control_decimation == 0:
            
            # --- A. Gait Scheduling ---
            contact_schedule = self.gait.get_contact_schedule(controller_state.gait_phase_time)
            np.copyto(buffers.contact_schedule, contact_schedule)
            current_contact = contact_schedule[0, :]
            
            # --- B. Update Swing/Stance State ---
            # We need absolute world positions to know where to start swinging from
            foot_pos_world = robot_interface.get_foot_positions_world()
            
            for i in range(4):
                if current_contact[i] == 1: # Stance
                    controller_state.swing_active[i] = 0
                    # Continuously update start pos so it's fresh upon lift-off
                    controller_state.swing_start_pos[i][:] = foot_pos_world[i]
                else:
                    controller_state.swing_active[i] = 1

            # --- C. MPC (33 Hz) ---
            # Runs every 3rd execution of the 100Hz loop
            if controller_state.mpc_counter % self.mpc_decimation == 0:
                state_vec = state.base.to_mpc_vector()
                
                ref = self.traj_gen.generate_reference(
                    state_vec,
                    command.v_cmd_global,
                    command.yaw_rate,
                    command.default_height,
                )

                # [FIX 2] Coordinate Frame Rotation
                # MPC requires feet in BODY frame, but foot_pos_rel is WORLD frame.
                cos_y = np.cos(state.base.yaw)
                sin_y = np.sin(state.base.yaw)
                R_z_T = np.array([[cos_y, sin_y, 0], [-sin_y, cos_y, 0], [0, 0, 1]])
                
                foot_pos_body = []
                for i in range(4):
                    p_b = R_z_T @ foot_pos_rel[i]
                    foot_pos_body.append(p_b)

                forces = self.mpc.solve(state, ref, buffers.contact_schedule, foot_pos_body)
                np.copyto(buffers.current_forces, forces)
                
            controller_state.mpc_counter += 1

            # --- D. Force Smoothing & WBC ---
            # EMA: alpha close to 1 = heavy smoothing (keep old value)
            buffers.smoothed_forces *= self.force_alpha
            buffers.smoothed_forces += (1 - self.force_alpha) * buffers.current_forces
            
            forces_list = [buffers.smoothed_forces[3*i : 3*i+3] for i in range(4)]
            buffers.tau_stance[:] = self.wbc.compute_torques(forces_list, gravity_comp=True)

        # ======================================================
        # HIGH FREQUENCY LOOP (1 kHz) - Swing Control
        # ======================================================
        buffers.tau_swing.fill(0.0)
        
        # [FIX 3] Re-implement Cartesian PD + Jacobian Transpose (Ghost Legs Fix)
        
        # Calculate Body Velocity for Raibert Heuristic
        cos_y = np.cos(state.base.yaw)
        sin_y = np.sin(state.base.yaw)
        R_z_T = np.array([[cos_y, sin_y, 0], [-sin_y, cos_y, 0], [0, 0, 1]])
        v_body = R_z_T @ state.base.linear_velocity

        for i in range(4):
            if controller_state.swing_active[i]:
                # 1. Trajectory Generation
                t_swing = self.gait.get_swing_state(controller_state.gait_phase_time, i)
                swing_duration = self.gait.period * (1.0 - self.gait.stance_ratio)
                
                # Raibert Heuristic
                raibert_offset = v_body[0:2] * self.gait.period * 0.5
                raibert_offset += 0.5 * (command.v_cmd_global[0:2] - v_body[0:2])
                
                # Rotate offset back to world
                R_z = R_z_T.T
                off_world = R_z @ np.array([raibert_offset[0], raibert_offset[1], 0.0])
                
                p0 = controller_state.swing_start_pos[i]
                pf = p0.copy()
                pf[0] += off_world[0]
                pf[1] += off_world[1]
                pf[2] = 0.02 # Z-target

                self.swing_trajs[i].set_initial_position(p0)
                self.swing_trajs[i].set_final_position(pf)
                self.swing_trajs[i].set_height(0.10)
                self.swing_trajs[i].compute_swing_trajectory_bezier(t_swing, swing_duration)
                
                p_des = self.swing_trajs[i].get_position()
                v_des = self.swing_trajs[i].get_velocity()
                
                # 2. Cartesian PD (simulator-agnostic)
                p_curr = robot_interface.get_foot_positions_world()[i]
                v_curr = robot_interface.get_foot_velocity(i)

                F_swing = self.kp_swing * (p_des - p_curr) + self.kd_swing * (v_des - v_curr)

                # 3. Torque Mapping: tau = J_leg^T @ F + gravity comp
                J_leg = robot_interface.get_leg_jacobian(i)
                grav_comp = robot_interface.get_gravity_compensation(i)

                tau_leg = J_leg.T @ F_swing + grav_comp
                buffers.tau_swing[3*i : 3*i+3] = tau_leg

        # ======================================================
        # Merge & Clip
        # ======================================================
        for i in range(4):
            idx = slice(3*i, 3*i+3)
            if controller_state.swing_active[i]:
                buffers.tau_final[idx] = buffers.tau_swing[idx]
            else:
                buffers.tau_final[idx] = buffers.tau_stance[idx]

        np.clip(buffers.tau_final, -self.torque_limit, self.torque_limit, out=buffers.tau_final)

        return buffers.tau_final