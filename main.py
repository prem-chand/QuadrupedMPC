import time
import numpy as np
import mujoco
import mujoco.viewer

from go2_mpc.config.config import default_config
from go2_mpc.utils.data_logger import DataLogger, create_log_entry, LogEntry
from go2_mpc.utils.simulation_utils import setup_simulation, setup_controller
from go2_mpc.core.command import Command


class KeyboardTeleop:
    """GLFW-based keyboard teleop for MuJoCo viewer."""
    
    def __init__(self, vel_speed=0.3, yaw_speed=0.5):
        self.vel_speed = vel_speed
        self.yaw_speed = yaw_speed
        self.v_cmd_global = np.zeros(3)
        self.yaw_rate = 0.0
    
    def key_callback(self, key):
        """GLFW key callback (signature: key only)."""
        if key == 87:  # W
            self.v_cmd_global[0] += self.vel_speed
        elif key == 83:  # S
            self.v_cmd_global[0] -= self.vel_speed
        elif key == 65:  # A
            self.v_cmd_global[1] += self.vel_speed
        elif key == 68:  # D
            self.v_cmd_global[1] -= self.vel_speed
        elif key == 81:  # Q
            self.yaw_rate += self.yaw_speed
        elif key == 69:  # E
            self.yaw_rate -= self.yaw_speed
        elif key == 32:  # Space
            self.v_cmd_global[:] = 0.0
            self.yaw_rate = 0.0
    
    def get_command(self):
        """Get current command."""
        return Command(
            v_cmd_global=self.v_cmd_global.copy(),
            yaw_rate=self.yaw_rate,
            default_height=0.32,
        )


def compute_standing_torque(robot, q_stand, stand_kp, stand_kd, torque_limit):
    """Compute PD + gravity compensation torques for standing."""
    q, qd = robot.get_joint_state()
    tau = np.zeros(12)
    for i in range(4):
        idx = slice(3 * i, 3 * i + 3)
        grav = robot.get_gravity_compensation(i)
        tau[idx] = stand_kp * (q_stand[3*i:3*i+3] - q[3*i:3*i+3]) \
            - stand_kd * qd[3*i:3*i+3] + grav
    return np.clip(tau, -torque_limit, torque_limit)


def main():
    cfg = default_config()
    model, data, robot, estimator = setup_simulation(cfg)
    controller, controller_state, buffers = setup_controller(cfg, robot)
    
    logger = DataLogger(log_dir="logs", max_steps=50000)
    teleop = KeyboardTeleop(vel_speed=0.3, yaw_speed=0.5)
    
    # Stand phase config
    stand_duration = 1.0
    q_stand = np.array([0.0, 0.9, -1.8] * 4)
    
    print("=" * 50)
    print("Controls: W/S=forward/back | A/D=left/right | Q/E=yaw | Space=stop")
    print("=" * 50)

    transition_logged = False

    with mujoco.viewer.launch_passive(model, data, key_callback=teleop.key_callback) as viewer:
        while viewer.is_running():
            step_start = time.time()
            robot.step()
            sim_time = robot.get_time()

            if sim_time < stand_duration:
                tau = compute_standing_torque(robot, q_stand, 300.0, 10.0, cfg.controller.torque_limit)
                controller_state.step_counter += 1
                controller_state.gait_phase_time += cfg.simulation.sim_dt
            else:
                if not transition_logged:
                    print(f"\n=== WALKING MODE START (t={sim_time:.3f}s) ===")
                    transition_logged = True
                
                state, foot_pos_rel = estimator.estimate()
                tau, diag = controller.compute(
                    state=state, foot_pos_rel=foot_pos_rel, command=teleop.get_command(),
                    controller_state=controller_state, buffers=buffers, robot_interface=robot,
                )
                
                if controller_state.step_counter % 10 == 0:
                    logger.log(create_log_entry(
                        t=sim_time, step=controller_state.step_counter, robot=robot, state=state,
                        command=teleop.get_command(), contact_schedule=diag['contact_schedule'],
                        gait_phase=controller_state.gait_phase_time % controller.gait.period,
                        mpc_forces=diag['current_forces'].reshape(4, 3), mpc_solve_time=0.0, tau=tau,
                    ))

            robot.set_torques(tau)
            viewer.sync()
            
            # Sync to real-time
            elapsed = time.time() - step_start
            sleep_time = cfg.simulation.sim_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        logger.close()



if __name__ == "__main__":
    main()
