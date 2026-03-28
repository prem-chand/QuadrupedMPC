import time
import numpy as np
import mujoco
import mujoco.viewer

from go2_mpc.config.config import default_config
from go2_mpc.utils.data_logger import DataLogger, create_log_entry, LogEntry
from go2_mpc.core.mujoco_robot import MujocoRobot
from go2_mpc.controller.state_estimator import StateEstimator
from go2_mpc.controller.controller_manager import (
    ControllerCore,
    ControllerState,
    ControllerBuffers,
)
from go2_mpc.controller.convex_mpc import ConvexMPC
from go2_mpc.controller.cvxpy_solver import ClarabelSolver
from go2_mpc.controller.wbc import WholeBodyController
from go2_mpc.controller.gait_scheduler import GaitScheduler
from go2_mpc.controller.trajectory_generator import TrajectoryGenerator
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


def main():
    global _transition_logged
    _transition_logged = False

    cfg = default_config()

    # ==========================================================
    # Simulation Setup
    # ==========================================================

    model = mujoco.MjModel.from_xml_path(str(cfg.simulation.model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    robot = MujocoRobot(model, data)
    estimator = StateEstimator(robot)

    # ==========================================================
    # Controller Stack
    # ==========================================================

    solver = ClarabelSolver()

    mpc = ConvexMPC(
        mass=cfg.mpc.mass,
        inertia=cfg.mpc.inertia,
        prediction_horizon=cfg.mpc.horizon,
        dt=cfg.mpc.dt,
        Q=cfg.mpc.Q,
        R=cfg.mpc.R,
        mu=cfg.mpc.mu,
        f_max=cfg.mpc.f_max,
        solver=solver,
    )

    gait = GaitScheduler(
        gait_period=cfg.gait.gait_period,
        stance_ratio=cfg.gait.stance_ratio,
        horizon=cfg.mpc.horizon,
        dt=cfg.mpc.dt,
    )

    traj_gen = TrajectoryGenerator(
        prediction_horizon=cfg.mpc.horizon,
        dt=cfg.mpc.dt,
    )

    wbc = WholeBodyController(robot, torque_limit=cfg.controller.torque_limit)

    controller = ControllerCore(
        gait=gait,
        traj_gen=traj_gen,
        mpc=mpc,
        wbc=wbc,
        config={
            "MPC_DECIMATION": cfg.controller.mpc_decimation,
            "FORCE_SMOOTH_ALPHA": cfg.controller.force_smooth_alpha,
            "TORQUE_LIMIT": cfg.controller.torque_limit,
            "SWING_KP": cfg.controller.swing_kp,
            "SWING_KD": cfg.controller.swing_kd,
            "FOOT_STANCE_OFFSETS": cfg.controller.foot_stance_offsets,
        },
    )

    controller_state = ControllerState(
        step_counter=0,
        mpc_counter=0,
        gait_phase_time=0.0,
        swing_active=np.zeros(4),
        swing_start_pos=[
            p.copy() for p in robot.get_foot_positions_world()
        ],
        swing_target_pos=[
            p.copy() for p in robot.get_foot_positions_world()
        ],
    )

    buffers = ControllerBuffers()

    # ==========================================================
    # Data Logger
    # ==========================================================
    
    logger = DataLogger(log_dir="logs", max_steps=50000)
    mpc_forces = np.zeros((4, 3))
    mpc_solve_time = 0.0
    
    # Transition debug flag
    _transition_logged = False

    # ==========================================================
    # Standing Phase Config
    # ==========================================================

    stand_duration = 1.0  # seconds
    # Joint angles from go2.xml default pose: [hip, thigh, calf] x 4 legs
    q_stand = np.array([0.0, 0.9, -1.8] * 4)
    stand_kp = 300.0
    stand_kd = 10.0

    # ==========================================================
    # Main Loop
    # ==========================================================

    teleop = KeyboardTeleop(vel_speed=0.3, yaw_speed=0.5)

    print("=" * 50)
    print("Controls: W/S=forward/back | A/D=left/right | Q/E=yaw | Space=stop")
    print("=" * 50)

    with mujoco.viewer.launch_passive(model, data, key_callback=teleop.key_callback) as viewer:

        while viewer.is_running():

            step_start = time.time()

            robot.step()

            sim_time = robot.get_time()

            if sim_time < stand_duration:
                # --- Standing phase: joint PD + gravity compensation ---
                q, qd = robot.get_joint_state()
                tau = np.zeros(12)
                for i in range(4):
                    idx = slice(3 * i, 3 * i + 3)
                    grav = robot.get_gravity_compensation(i)
                    tau[idx] = stand_kp * (q_stand[3*i:3*i+3] - q[3*i:3*i+3]) \
                        - stand_kd * qd[3*i:3*i+3] + grav
                np.clip(tau, -cfg.controller.torque_limit,
                        cfg.controller.torque_limit, out=tau)
                
                # Update controller state counters even during stand
                controller_state.step_counter += 1
                controller_state.gait_phase_time += cfg.simulation.sim_dt
                
            else:
                # --- Trot phase: full MPC controller ---
                
                # Log transition moment
                if not _transition_logged:
                    print(f"\n=== WALKING MODE START (t={sim_time:.3f}s) ===")
                    _transition_logged = True
                
                state, foot_pos_rel = estimator.estimate()

                command = teleop.get_command()

                tau, diag = controller.compute(
                    state=state,
                    foot_pos_rel=foot_pos_rel,
                    command=command,
                    controller_state=controller_state,
                    buffers=buffers,
                    robot_interface=robot,
                )
                
                # Get gait info for logging
                contact_schedule = diag['contact_schedule']
                gait_phase = controller_state.gait_phase_time % gait.period
                
                # Debug: First 20 steps after transition
                if _transition_logged and controller_state.step_counter < 20:
                    base_pos, _ = robot.get_base_pose()
                    # Compute RPY from quaternion (state already has it computed)
                    print(f"  step={controller_state.step_counter} "
                          f"pos=({base_pos[0]:.3f},{base_pos[1]:.3f},{base_pos[2]:.3f}) "
                          f"rpy=({np.degrees(state.base.roll):.1f},{np.degrees(state.base.pitch):.1f},{np.degrees(state.base.yaw):.1f})deg "
                          f"tau_max={np.max(np.abs(tau)):.1f}")
                
                # Log data every 10 steps
                if controller_state.step_counter % 10 == 0:
                    entry = create_log_entry(
                        t=sim_time,
                        step=controller_state.step_counter,
                        robot=robot,
                        state=state,
                        command=command,
                        contact_schedule=contact_schedule,
                        gait_phase=gait_phase,
                        mpc_forces=diag['current_forces'].reshape(4, 3),
                        mpc_solve_time=0.0,
                        tau=tau,
                    )
                    logger.log(entry)
                    
                    # Print status every 500 steps
                    if controller_state.step_counter % 500 == 0:
                        base_pos, _ = robot.get_base_pose()
                        base_rpy = robot.get_base_rpy()
                        swing_active = diag['swing_active']
                        print(f"step={controller_state.step_counter} t={sim_time:.2f}s "
                              f"pos=({base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f}) "
                              f"rpy=({base_rpy[0]:.3f}, {base_rpy[1]:.3f}, {base_rpy[2]:.3f}) "
                              f"cmd=({command.v_cmd_global[0]:.2f}, {command.v_cmd_global[1]:.2f}) "
                              f"swing={swing_active}")

            robot.set_torques(tau)

            viewer.sync()

            elapsed = time.time() - step_start
            sleep_time = cfg.simulation.sim_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # Cleanup
        logger.close()


if __name__ == "__main__":
    main()
