from go2_mpc.controller.joint_pd_controller import JointPDController
from go2_mpc.controller.wbc import WholeBodyController
from go2_mpc.controller.foot_swing_trajectory import FootSwingTrajectory
from go2_mpc.controller.convex_mpc import ConvexMPC
from go2_mpc.controller.trajectory_generator import TrajectoryGenerator
from go2_mpc.controller.gait_scheduler import GaitScheduler
from go2_mpc.controller.state_estimator import StateEstimator
import sys
import time
import threading
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer

import logging
import json


def setup_debug_logger(filename="monolithic_debug.log"):
    logger = logging.getLogger("MPC_DEBUG")
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(filename, mode="w")
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False

    return logger


# Helper class to manage keyboard input


class KeyboardController:
    def __init__(self, vel_speed=0.1, yaw_speed=0.1):
        self.vel_speed = vel_speed
        self.yaw_speed = yaw_speed
        self.v_cmd_body = np.zeros(3)
        self.yaw_rate_cmd = 0.0
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        """Background thread for keyboard input (terminal-based)."""
        import select
        import tty
        import termios

        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setcbreak(sys.stdin.fileno())
            while self.running:
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    key = sys.stdin.read(1).lower()
                    if key == 'w':
                        self.v_cmd_body[0] += self.vel_speed
                    elif key == 's':
                        self.v_cmd_body[0] -= self.vel_speed
                    elif key == 'a':
                        self.v_cmd_body[1] += self.vel_speed
                    elif key == 'd':
                        self.v_cmd_body[1] -= self.vel_speed
                    elif key == 'q':
                        self.yaw_rate_cmd += self.yaw_speed
                    elif key == 'e':
                        self.yaw_rate_cmd -= self.yaw_speed
                    elif key == ' ':
                        self.v_cmd_body[:] = 0.0
                        self.yaw_rate_cmd = 0.0
                    elif key == 'x':
                        self.running = False
        except Exception as e:
            print(f"Keyboard thread error: {e}")
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


# Import controller modules


def main():
    """
    Main simulation loop for the Unitree Go2 robot using MPC and WBC.
    """
    debug_logger = setup_debug_logger()
    DEBUG_EVERY = 50   # log every N control steps

    def debug_log(tag, step, **kwargs):
        if step % DEBUG_EVERY != 0:
            return

        payload = {
            "step": step,
            "tag": tag,
        }

        for k, v in kwargs.items():
            if isinstance(v, np.ndarray):
                payload[k] = v.tolist()
            else:
                payload[k] = v

        debug_logger.info(json.dumps(payload))

    # ============================
    # 1. CONFIGURATION
    # ============================
    SIM_DT = 0.001           # 1 kHz Simulation
    DECIMATION = 10          # Control runs every 10 steps (100 Hz)
    MPC_DECIMATION = 3       # MPC runs every 3 control steps (33 Hz)

    # Robot Physics
    MASS = 15.2  # Unitree Go2 approx
    # INERTIA = np.diag([0.25, 0.45, 0.45])
    # Slightly higher inertia for stability in MPC
    INERTIA = np.diag([0.1, 0.1, 0.02])

    # Initial Pose (Standing)
    START_Q = np.array([0.0, 0.9, -1.8] * 4)  # [Hip, Thigh, Calf] * 4
    DEFAULT_HEIGHT = 0.32

    # Hip link lateral offset (thigh hangs 0.0955m outward from hip joint)
    # Without this, Raibert targets directly under hip body = too far inward
    HIP_LINK_LATERAL = 0.0955
    HIP_SIDE_SIGNS = np.array([1.0, -1.0, 1.0, -1.0])  # [FL, FR, RL, RR]

    # MPC Weights
    # [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
    # Q = np.diag([100, 100, 500, 200, 200, 50, 5, 5, 30, 0.5, 0.5, 0.3])
    Q = np.diag([1, 5, 100, 3, 10, 0.1, 5, 5, 12, 2, 3, 2])
    R = np.diag([1e-6] * 12)  # Slightly higher to smooth forces

    # Load Model
    model_path = Path(__file__).parent / 'go2_mpc' / 'robot' / 'scene.xml'
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    hip_body_ids = [mujoco.mj_name2id(model,
                                      mujoco.mjtObj.mjOBJ_BODY,
                                      name) for name in ['FL_hip', 'FR_hip', 'RL_hip', 'RR_hip']]
    for i in range(4):
        print(i, model.body(hip_body_ids[i]).name)

    # print("hip body ids: ", hip_body_ids)
    # hip_pos_world = [data.xpos[hip_body_ids[i]] for i in range(4)]
    # print("hip pos world: ", hip_pos_world)
    # exit(1)

    # ============================
    # 2. CONTROLLER INITIALIZATION
    # ============================
    print("Initializing Controllers...")

    # A. State Estimator
    estimator = StateEstimator(model, data)

    # B. Gait Scheduler
    gait = GaitScheduler(
        gait_period=0.45, stance_ratio=0.65, horizon=10, dt=0.03)

    # C. Trajectory Generator
    traj_gen = TrajectoryGenerator(prediction_horizon=10, dt=0.03)

    # D. Convex MPC
    mpc = ConvexMPC(MASS, INERTIA, prediction_horizon=10, dt=0.03,
                    Q=Q, R=R, mu=0.6, f_max=180.0)

    # E. Swing Trajectory + Cartesian PD (MIT Cheetah convention)
    SWING_HEIGHT = 0.10
    KP_SWING_CART = 400.0   # Cartesian stiffness [N/m]
    KD_SWING_CART = 10.0    # Cartesian damping   [Ns/m]
    foot_swing_trajectories = [FootSwingTrajectory() for _ in range(4)]
    for _ft in foot_swing_trajectories:
        _ft.set_height(SWING_HEIGHT)

    # F. Whole Body Controller
    wbc = WholeBodyController(model, data)

    # G. Safety PD (Low level fallback)
    # safety_pd = JointPDController(model, kp=2, kd=0.1)

    # ============================
    # 3. SIMULATION SETUP
    # ============================
    # Initialize Robot Pose
    # data.qpos[0:2] = [2,2]
    data.qpos[2] = DEFAULT_HEIGHT  # Lift base
    # data.qpos[3:7] = [0.0, 0.0, 0.0, 1.0]
    data.qpos[7: 7+12] = START_Q

    mujoco.mj_forward(model, data)  # Settle physics

    # Input Command - Thread-safe keyboard controlled
    keyboard = KeyboardController()
    keyboard.start()

    # Standing phase duration before trot
    STAND_DURATION = 1.0  # Stand for 2 seconds before starting trot

    # Loop State
    step_count = 0
    mpc_counter = 0

    # Buffers (pre-allocated to avoid GC pressure)
    current_forces = np.zeros(12)
    smoothed_forces = np.zeros(12)
    FORCE_SMOOTH_ALPHA = 0.99  # Higher = more responsive to new forces

    # Smoothed velocity command (reduces sensitivity to sudden inputs)
    smoothed_v_cmd = np.zeros(3)
    smoothed_yaw_cmd = 0.0
    CMD_SMOOTH_ALPHA = 0.05

    # Pre-allocated buffers for hot loop (avoid allocations at 100Hz)
    R_z = np.zeros((3, 3))
    v_cmd_global = np.zeros(3)
    forces_list = [np.zeros(3) for _ in range(4)]
    contact_schedule_float = np.zeros((10, 4))  # Pre-allocated for MPC
    stance_mask_actuator = np.zeros(12)  # Pre-allocated stance mask
    tau_cmd = np.zeros(12)  # Pre-allocated swing torque buffer
    p_hip = np.zeros((4, 3))  # Pre-allocated hip position
    foot_pos_body = np.zeros((4, 3))  # Pre-allocated body-frame foot positions

    # Swing Cartesian PD buffers (Jacobian-based, MIT convention)
    nv = model.nv
    J_swing = np.zeros((3, nv))   # Position Jacobian buffer
    Jr_swing = np.zeros((3, nv))  # Rotation Jacobian (required by API, unused)
    v_foot_world = np.zeros(3)    # Foot velocity in world frame
    F_swing = np.zeros(3)         # Cartesian PD force
    swing_time = gait.period * (1.0 - gait.stance_ratio)  # Swing duration [s]

    # Initialize Swing Start Position (Fix 1: Teleporting Foot)
    # We must capture where the feet ARE right now to avoid snapping
    mujoco.mj_kinematics(model, data)
    # Use site_xpos since foot_ids are site indices
    swing_start_pos = [data.site_xpos[estimator.foot_ids[i]].copy()
                       for i in range(4)]

    swing_end_pos = [np.zeros(3) for _ in range(4)]  # Target (Raibert)
    swing_active = np.zeros(4)  # 1 if leg is in swing, 0 if stance

    # Actuator Mapping (Must match WBC/XML)
    # Logical [FL, FR, RL, RR] -> Actuator Indices
    actuator_map = [
        [0, 1, 2],    # FL
        [3, 4, 5],    # FR
        [6, 7, 8],    # RL
        [9, 10, 11]   # RR
    ]

    # Initialize tau_stance for 1kHz loop (updated at 100Hz)
    tau_stance = np.zeros(12)

    # Safety state (updated at 100Hz, checked at 1kHz)
    is_fallen = False

    # Safety Ramp (Fix 3: Safety)
    ramp_up = 0.0

    # Pre-allocated torque merge buffer (avoid allocations at 1kHz)
    tau_final = np.zeros(12)

    # Viewer throttle (~60fps)
    last_viewer_sync = 0.0
    VIEWER_DT = 1.0 / 60.0

    print("Starting Simulation...")
    print(f"Standing for {STAND_DURATION}s before starting trot gait...")
    print("\n=== KEYBOARD CONTROLS (press in terminal) ===")
    print("  W/S: Forward/Backward")
    print("  A/D: Strafe Left/Right")
    print("  Q/E: Turn Left/Right")
    print("  SPACE: Stop all motion")
    print("  X: Exit")
    print("=============================================\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and keyboard.running:
            step_start = time.time()

            # ---------------------------------------------
            # 1000 Hz: Physics Integration
            # ---------------------------------------------
            mujoco.mj_step(model, data)

            # ---------------------------------------------
            # 100 Hz: High-Level Control Loop
            # ---------------------------------------------
            if step_count % DECIMATION == 0:

                # --- 1. ESTIMATION ---
                # Get state [Pos, Eu, Vel, Omega] and Feet Rel Positions
                state, foot_pos_rel = estimator.get_robot_state()
                current_time = data.time
                debug_log(
                    "state_estimate",
                    step_count,
                    state=state.copy(),
                    foot_pos_rel=foot_pos_rel.copy(),
                )

                # --- 2. USER INPUT PROCESSING ---
                # Smooth velocity commands for stability (in-place update)
                smoothed_v_cmd *= (1 - CMD_SMOOTH_ALPHA)
                smoothed_v_cmd += CMD_SMOOTH_ALPHA * keyboard.v_cmd_body
                smoothed_yaw_cmd = CMD_SMOOTH_ALPHA * keyboard.yaw_rate_cmd + \
                    (1 - CMD_SMOOTH_ALPHA) * smoothed_yaw_cmd

                # Rotate body command to world frame (reuse R_z buffer)
                yaw = state[5]
                c, s = np.cos(yaw), np.sin(yaw)
                R_z[0, 0] = c
                R_z[0, 1] = -s
                R_z[0, 2] = 0
                R_z[1, 0] = s
                R_z[1, 1] = c
                R_z[1, 2] = 0
                R_z[2, 0] = 0
                R_z[2, 1] = 0
                R_z[2, 2] = 1
                np.dot(R_z, smoothed_v_cmd, out=v_cmd_global)

                # --- 3. GAIT & SCHEDULE ---
                # Standing phase: all legs in stance for first STAND_DURATION seconds
                if current_time < STAND_DURATION:
                    # All stance (reuse buffer)
                    contact_schedule_float.fill(1.0)
                else:
                    contact_schedule = gait.get_contact_schedule(
                        current_time - STAND_DURATION)
                    # Copy to float buffer
                    np.copyto(contact_schedule_float, contact_schedule)
                current_contact = contact_schedule_float[0, :]
                debug_log(
                    "contact_schedule",
                    step_count,
                    contact=current_contact.copy(),
                )

                # --- 4. MPC (33 Hz) ---
                if mpc_counter % MPC_DECIMATION == 0:
                    # Generate Reference Trajectory (Pro Version: Integrates Yaw)
                    ref_traj = traj_gen.generate_reference(
                        state, v_cmd_global, smoothed_yaw_cmd, DEFAULT_HEIGHT
                    )
                    for i in range(4):
                        # Transpose dot: foot_pos_body[i] = R_z.T @ foot_pos_rel[i]
                        foot_pos_body[i] = R_z.T @ foot_pos_rel[i]

                    # Solve MPC (Pro Version: Updates A/B Matrices internally)
                    # Returns (12,) forces - use pre-converted float buffer
                    mpc_f = mpc.solve(
                        state, ref_traj, contact_schedule_float, foot_pos_body)
                    debug_log(
                        "mpc_input",
                        step_count,
                        state=state.copy(),
                        ref=ref_traj.copy(),
                        contact=contact_schedule_float.copy(),
                        foot_body=foot_pos_body.copy(),
                    )

                    debug_log(
                        "mpc_output",
                        step_count,
                        forces=current_forces.copy(),
                    )

                    # Store for WBC (in-place copy to avoid allocation)
                    np.copyto(current_forces, mpc_f)

                mpc_counter += 1

                # Smooth forces to reduce jerking at gait transitions
                smoothed_forces = FORCE_SMOOTH_ALPHA * current_forces + \
                    (1 - FORCE_SMOOTH_ALPHA) * smoothed_forces

                for i in range(4):
                    if current_contact[i] == 1:
                        # === STANCE PHASE ===
                        swing_active[i] = 0
                        # Continuously capture foot position for next swing liftoff
                        swing_start_pos[i][:] = data.site_xpos[estimator.foot_ids[i]]
                    else:
                        # === SWING PHASE ===
                        swing_active[i] = 1

                        # --- Raibert Heuristic (body frame) ---
                        T_stance = gait.period * gait.stance_ratio
                        v_current_world = state[6:9]
                        v_current_body = R_z.T @ v_current_world
                        v_cmd_body = smoothed_v_cmd

                        k_raibert = 0.5
                        vel_error_x = v_cmd_body[0] - v_current_body[0]
                        vel_error_y = v_cmd_body[1] - v_current_body[1]

                        p_offset_body = np.array([
                            v_current_body[0] * T_stance /
                            2 + k_raibert * vel_error_x,
                            v_current_body[1] * T_stance /
                            2 + k_raibert * vel_error_y
                            + HIP_SIDE_SIGNS[i] * HIP_LINK_LATERAL,
                            0.0
                        ])
                        MAX_STEP = 0.25
                        np.clip(p_offset_body[:2], -MAX_STEP,
                                MAX_STEP, out=p_offset_body[:2])

                        # Rotate body offset to world frame
                        p_offset_world = R_z @ p_offset_body

                        # Swing target = hip projection + velocity-based offset
                        p_hip[i] = data.xpos[hip_body_ids[i]]
                        swing_end_pos[i][0] = p_hip[i][0] + p_offset_world[0]
                        swing_end_pos[i][1] = p_hip[i][1] + p_offset_world[1]
                        swing_end_pos[i][2] = 0.02

                        # Set world-frame waypoints on Bezier trajectory
                        foot_swing_trajectories[i].set_initial_position(
                            swing_start_pos[i])
                        foot_swing_trajectories[i].set_final_position(
                            swing_end_pos[i])
                # --- 6. WHOLE BODY CONTROL (Stance) ---
                # Reuse pre-allocated forces_list buffer
                for i in range(4):
                    forces_list[i][:] = smoothed_forces[3*i: 3*i+3]

                # WBC computes tau_stance in the correct ACTUATOR ORDER internally
                # gravity_comp=True adds qfrc_bias so that J^T @ F produces the
                # intended GRF rather than losing torque to leg link gravity
                tau_stance = wbc.compute_torques(
                    forces_list, gravity_comp=True)
                debug_log(
                    "wbc_output",
                    step_count,
                    tau_stance=tau_stance.copy(),
                )


                # Build stance mask (reuse pre-allocated buffer)
                stance_mask_actuator.fill(0)
                for i in range(4):
                    is_stance = current_contact[i]
                    idx_list = actuator_map[i]
                    stance_mask_actuator[idx_list] = is_stance

                # Safety check (update flag at 100Hz)
                # roll/pitch > 45 deg
                is_fallen = np.max(np.abs(state[3:5])) > 0.8

                # Debug Print
                if step_count % 500 == 0:
                    total_fz = np.sum(current_forces[2::3])
                    loop_elapsed = time.time() - step_start
                    print(
                        f"t={current_time:.2f}s | Z: {state[2]:.3f} | Fz: {total_fz:.1f}N | loop: {loop_elapsed*1000:.1f}ms", flush=True)

            # ---------------------------------------------
            # 1000 Hz: Swing Leg Cartesian PD (MIT Cheetah convention)
            # World-frame Bezier trajectory -> Cartesian PD -> J^T mapping
            # ---------------------------------------------
            tau_cmd.fill(0)

            for i in range(4):
                if swing_active[i]:
                    # A. Compute swing phase
                    t_sw = gait.get_swing_state(data.time - STAND_DURATION, i)

                    # B. Bezier trajectory in WORLD frame
                    foot_swing_trajectories[i].compute_swing_trajectory_bezier(
                        t_sw, swing_time)
                    pDesWorld = foot_swing_trajectories[i].get_position()
                    vDesWorld = foot_swing_trajectories[i].get_velocity()

                    # C. Actual foot state in WORLD frame
                    pActWorld = data.site_xpos[estimator.foot_ids[i]]

                    # Foot velocity via full Jacobian: v = J @ qvel
                    mujoco.mj_jacSite(model, data,
                                      J_swing, Jr_swing, estimator.foot_ids[i])
                    np.dot(J_swing, data.qvel, out=v_foot_world)

                    # D. Cartesian PD force (world frame)
                    #    F = Kp * (pDes - pAct) + Kd * (vDes - vAct)
                    F_swing[0] = KP_SWING_CART * (pDesWorld[0] - pActWorld[0]) + \
                        KD_SWING_CART * (vDesWorld[0] - v_foot_world[0])
                    F_swing[1] = KP_SWING_CART * (pDesWorld[1] - pActWorld[1]) + \
                        KD_SWING_CART * (vDesWorld[1] - v_foot_world[1])
                    F_swing[2] = KP_SWING_CART * (pDesWorld[2] - pActWorld[2]) + \
                        KD_SWING_CART * (vDesWorld[2] - v_foot_world[2])

                    # E. Torque = J_leg^T @ F + qfrc_bias (gravity feedforward)
                    dof_indices = wbc.leg_dofs[i]
                    act_indices = actuator_map[i]
                    tau_cmd[act_indices[0]] = (J_swing[0, dof_indices[0]] * F_swing[0] +
                                               J_swing[1, dof_indices[0]] * F_swing[1] +
                                               J_swing[2, dof_indices[0]] * F_swing[2] +
                                               data.qfrc_bias[dof_indices[0]])
                    tau_cmd[act_indices[1]] = (J_swing[0, dof_indices[1]] * F_swing[0] +
                                               J_swing[1, dof_indices[1]] * F_swing[1] +
                                               J_swing[2, dof_indices[1]] * F_swing[2] +
                                               data.qfrc_bias[dof_indices[1]])
                    tau_cmd[act_indices[2]] = (J_swing[0, dof_indices[2]] * F_swing[0] +
                                               J_swing[1, dof_indices[2]] * F_swing[1] +
                                               J_swing[2, dof_indices[2]] * F_swing[2] +
                                               data.qfrc_bias[dof_indices[2]])

            # --- MERGE & SAFETY (in-place to avoid allocations) ---
            np.multiply(tau_stance, stance_mask_actuator, out=tau_final)
            # tau_final += tau_cmd * (1 - stance_mask)
            for _i in range(12):
                tau_final[_i] += tau_cmd[_i] * (1.0 - stance_mask_actuator[_i])

            # Fallback Safety (use flag updated at 100Hz)
            if is_fallen:
                np.multiply(data.qvel[6:18], -5.0, out=tau_final)

            # Safety Ramp
            ramp_up = min(1.0, ramp_up + 0.0005)

            np.clip(tau_final, -35.0, 35.0, out=tau_final)
            debug_log(
                "tau_final",
                step_count,
                tau=tau_final.copy(),
            )

            data.ctrl[:] = tau_final

            # ---------------------------------------------
            # Visualization
            # ---------------------------------------------
            step_count += 1
            viewer.sync()

            # Sync to real time
            time_until_next_step = SIM_DT - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    # Cleanup
    keyboard.running = False
    print("Simulation ended.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        if "mjpython" in str(e):
            print("\n" + "="*60)
            print("ERROR: MuJoCo passive viewer on macOS requires 'mjpython'.")
            print("Please run this script using: mjpython main.py")
            print("="*60 + "\n")
        else:
            raise e
