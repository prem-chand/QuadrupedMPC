import numpy as np
import mujoco
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


def setup_simulation(cfg):
    """Initialize MuJoCo model, data, and robot interface."""
    model = mujoco.MjModel.from_xml_path(str(cfg.simulation.model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    robot = MujocoRobot(model, data)
    estimator = StateEstimator(robot)
    return model, data, robot, estimator


def setup_controller(cfg, robot):
    """Initialize the full MPC-WBC controller stack."""
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

    wbc = WholeBodyController(robot, torque_limit=cfg.controller.torque_limit, height_kp=80.0)
    wbc.set_target_height(cfg.controller.default_height)

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

    state = ControllerState(
        step_counter=0,
        mpc_counter=0,
        gait_phase_time=0.0,
        swing_active=np.zeros(4),
        swing_start_pos=[p.copy() for p in robot.get_foot_positions_world()],
        swing_target_pos=[p.copy() for p in robot.get_foot_positions_world()],
    )

    return controller, state, ControllerBuffers()
