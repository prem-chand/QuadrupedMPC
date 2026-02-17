import time
import numpy as np
import mujoco
import mujoco.viewer

from go2_mpc.config.config import default_config
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


def main():

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
    )

    buffers = ControllerBuffers()

    # ==========================================================
    # Main Loop
    # ==========================================================

    with mujoco.viewer.launch_passive(model, data) as viewer:

        while viewer.is_running():

            step_start = time.time()

            robot.step()

            state, foot_pos_rel = estimator.estimate()

            command = Command(
                v_cmd_global=np.zeros(3),
                yaw_rate=0.0,
                default_height=cfg.controller.default_height,
            )

            tau = controller.compute(
                state=state,
                foot_pos_rel=foot_pos_rel,
                command=command,
                controller_state=controller_state,
                buffers=buffers,
                robot_interface=robot,
            )

            robot.set_torques(tau)

            viewer.sync()

            elapsed = time.time() - step_start
            sleep_time = cfg.simulation.sim_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
