import time
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path

# --- Your Modules ---
from go2_mpc.core.mujoco_robot import MujocoRobot
from go2_mpc.controller.state_estimator import StateEstimator
from go2_mpc.core.state import State
from go2_mpc.controller.controller_manager import ControllerCore, ControllerState, ControllerBuffers
from go2_mpc.controller.convex_mpc import ConvexMPC
from go2_mpc.controller.cvxpy_solver import CVXPYSolver
from go2_mpc.controller.wbc import WholeBodyController
from go2_mpc.controller.gait_scheduler import GaitScheduler
from go2_mpc.controller.foot_swing_trajectory import FootSwingTrajectory
from go2_mpc.controller.trajectory_generator import TrajectoryGenerator
from go2_mpc.core.command import Command


def main():

    # ==========================================================
    # 1. SIMULATION SETUP
    # ==========================================================

    SIM_DT = 0.001
    MODEL_PATH = Path("go2_mpc/robot/scene.xml")

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)

    # ==========================================================
    # 2. ROBOT INTERFACE
    # ==========================================================

    robot = MujocoRobot(model, data)

    # ==========================================================
    # 3. ESTIMATOR
    # ==========================================================

    estimator = StateEstimator(robot)

    # ==========================================================
    # 4. CONTROLLER STACK
    # ==========================================================

    MASS = 15.2
    INERTIA = np.diag([0.1, 0.1, 0.02])
    HORIZON = 10
    MPC_DT = 0.03

    Q = np.diag([1, 5, 100, 3, 10, 0.1, 5, 5, 12, 2, 3, 2])
    R = np.diag([1e-6] * 12)

    solver = CVXPYSolver(solver_name="CLARABEL")

    mpc = ConvexMPC(
        mass=MASS,
        inertia=INERTIA,
        prediction_horizon=HORIZON,
        dt=MPC_DT,
        Q=Q,
        R=R,
        mu=0.6,
        f_max=180.0,
        solver=solver,
    )

    gait = GaitScheduler(
        gait_period=0.45,
        stance_ratio=0.65,
        horizon=HORIZON,
        dt=MPC_DT,
    )

    traj_gen = TrajectoryGenerator(
        prediction_horizon=HORIZON,
        dt=MPC_DT,
    )

    wbc = WholeBodyController(model, data)

    swing_trajs = [FootSwingTrajectory() for _ in range(4)]

    controller = ControllerCore(
        gait=gait,
        traj_gen=traj_gen,
        mpc=mpc,
        wbc=wbc,
        # swing_trajs=swing_trajs,
        config={
            "MPC_DECIMATION": 3,
            "FORCE_SMOOTH_ALPHA": 0.9,
        },
    )

    controller_state = ControllerState(
        step_counter=0,
        mpc_counter=0,
        gait_phase_time=0.0,
        swing_active=np.zeros(4),
        swing_start_pos=[p.copy() for p in robot.get_foot_positions_world()], # <--- Valid Init
    )

    buffers = ControllerBuffers()

    # ==========================================================
    # 5. MAIN LOOP
    # ==========================================================

    with mujoco.viewer.launch_passive(model, data) as viewer:

        while viewer.is_running():

            step_start = time.time()

            # ---- Step Physics ----
            robot.step()

            # ---- Estimate State ----
            state, foot_pos_rel = estimator.estimate()

            # ---- Build Command (example: zero velocity) ----
            v_cmd_global = np.zeros(3)
            yaw_rate_cmd = 0.0
            default_height = 0.32
            
            command = Command(
                v_cmd_global=v_cmd_global,
                yaw_rate=yaw_rate_cmd,
                default_height=default_height,
            )

            # ---- Compute Control ----
            tau = controller.compute(
                state=state,
                foot_pos_rel=foot_pos_rel,
                command=command,
                controller_state=controller_state,
                buffers=buffers,
                robot_interface=robot  # <--- Pass existing instance
            )

            robot.set_torques(tau)
            robot.step()

            # ---- Sync Viewer ----
            viewer.sync()

            # ---- Real-time sync ----
            elapsed = time.time() - step_start
            sleep_time = SIM_DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
