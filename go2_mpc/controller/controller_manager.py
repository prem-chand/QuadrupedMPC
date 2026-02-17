from dataclasses import dataclass
import numpy as np
from .state_estimator import StateEstimator


@dataclass
class ControllerState:
    mpc_counter: int
    gait_phase_time: float
    swing_active: np.ndarray
    swing_start_pos: list


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

        self.mpc_decimation = config["MPC_DECIMATION"]
        self.force_alpha = config["FORCE_SMOOTH_ALPHA"]

    def _structured_to_vector(self, state):
        x = np.zeros(12)

        x[0:3] = state.base.position
        x[3] = state.base.roll
        x[4] = state.base.pitch
        x[5] = state.base.yaw
        x[6:9] = state.base.linear_velocity
        x[9:12] = state.base.angular_velocity

        return x

    def compute(
        self,
        state,
        foot_pos_rel,
        command,
        controller_state: ControllerState,
        buffers: ControllerBuffers,
    ):

        # ======================================================
        # Build centroidal state vector for MPC
        # ======================================================
        state_vec = self._structured_to_vector(state)

        # ======================================================
        # Gait
        # ======================================================
        contact_schedule = self.gait.get_contact_schedule(
            controller_state.gait_phase_time
        )
        np.copyto(buffers.contact_schedule, contact_schedule)

        # ======================================================
        # MPC (Decimated)
        # ======================================================
        if controller_state.mpc_counter % self.mpc_decimation == 0:

            ref = self.traj_gen.generate_reference(
                state_vec,
                command.v_cmd_global,
                command.yaw_rate,
                command.default_height,
            )

            forces = self.mpc.solve(
                state,
                ref,
                buffers.contact_schedule,
                foot_pos_rel,
            )

            np.copyto(buffers.current_forces, forces)

        controller_state.mpc_counter += 1

        # ======================================================
        # Force Smoothing
        # ======================================================
        buffers.smoothed_forces *= (1 - self.force_alpha)
        buffers.smoothed_forces += self.force_alpha * buffers.current_forces

        # ======================================================
        # WBC
        # ======================================================
        forces_list = [
            buffers.smoothed_forces[0:3],
            buffers.smoothed_forces[3:6],
            buffers.smoothed_forces[6:9],
            buffers.smoothed_forces[9:12],
        ]

        buffers.tau_stance[:] = self.wbc.compute_torques(
            forces_list,
            gravity_comp=True,
        )

        # ======================================================
        # Swing (placeholder)
        # ======================================================
        buffers.tau_swing.fill(0.0)

        # ======================================================
        # Merge
        # ======================================================
        buffers.tau_final[:] = buffers.tau_stance
        buffers.tau_final += buffers.tau_swing

        return buffers.tau_final
