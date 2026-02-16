import sys
import time
import threading
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer

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
                        self.v_cmd_body[0] = self.vel_speed
                    elif key == 's':
                        self.v_cmd_body[0] = -self.vel_speed
                    elif key == 'a':
                        self.v_cmd_body[1] = self.vel_speed
                    elif key == 'd':
                        self.v_cmd_body[1] = -self.vel_speed
                    elif key == 'q':
                        self.yaw_rate_cmd = self.yaw_speed
                    elif key == 'e':
                        self.yaw_rate_cmd = -self.yaw_speed
                    elif key == ' ':
                        self.v_cmd_body[:] = 0.0
                        self.yaw_rate_cmd = 0.0
                    elif key == 'x':
                        self.running = False
        except Exception as e:
            print(f"Keyboard thread error: {e}")
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from go2_mpc.controller_v2.state_estimator import StateEstimator
from go2_mpc.controller_v2.gait_scheduler import GaitScheduler
from go2_mpc.controller_v2.trajectory_generator import TrajectoryGenerator
from go2_mpc.controller_v2.convex_mpc import ConvexMPC
from go2_mpc.controller_v2.swing_leg_ik_controller import SwingLegIKController
from go2_mpc.controller_v2.wbc import WholeBodyController
from go2_mpc.controller_v2.joint_pd_controller import JointPDController

def main():
    """
    Main simulation loop for the Unitree Go2 robot using MPC and WBC.
    """
    # ============================
    # 1. CONFIGURATION
    # ============================
    SIM_DT = 0.001           # 1 kHz Simulation
    DECIMATION = 10          # Control runs every 10 steps (100 Hz)
    MPC_DECIMATION = 3       # MPC runs every 3 control steps (33 Hz)
    
    # Robot Physics
    MASS = 15.2 # Unitree Go2 approx
    INERTIA = np.diag([0.25, 0.45, 0.45])
    
    # Initial Pose (Standing)
    START_Q = np.array([0.0, 0.9, -1.8] * 4) # [Hip, Thigh, Calf] * 4
    DEFAULT_HEIGHT = 0.32
    
    # MPC Weights
    # [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
    Q = np.diag([100, 100, 500, 200, 200, 50, 5, 5, 30, 0.5, 0.5, 0.3])
    R = np.diag([1e-5] * 12)  # Slightly higher to smooth forces
    
    # Load Model
    model_path = Path(__file__).parent.parent / 'robot' / 'scene.xml'
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # ============================
    # 2. CONTROLLER INITIALIZATION
    # ============================
    print("Initializing Controllers...")
    
    # A. State Estimator
    estimator = StateEstimator(model, data)
    
    # B. Gait Scheduler
    gait = GaitScheduler(gait_period=0.45, stance_ratio=0.65, horizon=10, dt=0.03)
    
    # C. Trajectory Generator
    traj_gen = TrajectoryGenerator(prediction_horizon=10, dt=0.03)
    
    # D. Convex MPC
    mpc = ConvexMPC(MASS, INERTIA, prediction_horizon=10, dt=0.03, 
                    Q=Q, R=R, mu=0.6, f_max=180.0)
    
    # E. Swing Controller
    swing_ctrl = SwingLegIKController(kp=3, kd=0.5, swing_height=0.06, dt=0.01)
    
    # F. Whole Body Controller
    wbc = WholeBodyController(model, data)
    
    # G. Safety PD (Low level fallback)
    # safety_pd = JointPDController(model, kp=2, kd=0.1) 

    # ============================
    # 3. SIMULATION SETUP
    # ============================
    # Initialize Robot Pose
    data.qpos[2] = DEFAULT_HEIGHT # Lift base
    data.qpos[7 : 7+12] = START_Q
    
    mujoco.mj_forward(model, data) # Settle physics
    
    # Input Command - Thread-safe keyboard controlled
    keyboard = KeyboardController()
    keyboard.start()

    # Standing phase duration before trot
    STAND_DURATION = 2.0  # Stand for 2 seconds before starting trot

    # Loop State
    step_count = 0
    mpc_counter = 0

    # Buffers (pre-allocated to avoid GC pressure)
    current_forces = np.zeros(12)
    smoothed_forces = np.zeros(12)
    FORCE_SMOOTH_ALPHA = 0.7  # Higher = more responsive to new forces

    # Smoothed velocity command (reduces sensitivity to sudden inputs)
    smoothed_v_cmd = np.zeros(3)
    smoothed_yaw_cmd = 0.0
    CMD_SMOOTH_ALPHA = 0.05

    # Pre-allocated buffers for hot loop (avoid allocations at 100Hz)
    R_z = np.zeros((3, 3))
    v_cmd_global = np.zeros(3)
    forces_list = [np.zeros(3) for _ in range(4)]
    q_leg_buf = np.zeros(3)
    dq_leg_buf = np.zeros(3)
    contact_schedule_float = np.zeros((10, 4))  # Pre-allocated for MPC
    stance_mask_actuator = np.zeros(12)  # Pre-allocated stance mask
    tau_cmd = np.zeros(12)  # Pre-allocated swing torque buffer
    p_hip = np.zeros(3)  # Pre-allocated hip position
    p_hip_projected = np.zeros(3)  # Pre-allocated projected hip
    p_target = np.zeros(3)  # Pre-allocated target position
    p_des_world = np.zeros(3)  # Pre-allocated desired world position
    vel_zero = np.zeros(3)  # Constant zero velocity for IK calls

    # Initialize Swing Start Position (Fix 1: Teleporting Foot)
    # We must capture where the feet ARE right now to avoid snapping
    mujoco.mj_kinematics(model, data)
    # Use site_xpos since foot_ids are site indices
    swing_start_pos = [data.site_xpos[estimator.foot_ids[i]].copy() for i in range(4)]

    swing_end_pos = [np.zeros(3) for _ in range(4)] # Target (Raibert)

    # Swing leg joint targets (updated at 100Hz, PD applied at 1kHz)
    swing_q_des = np.zeros((4, 3))  # Desired joint angles per leg
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
                
                # --- 2. USER INPUT PROCESSING ---
                # --- 2. USER INPUT PROCESSING ---
                # Smooth velocity commands for stability (in-place update)
                smoothed_v_cmd *= (1 - CMD_SMOOTH_ALPHA)
                smoothed_v_cmd += CMD_SMOOTH_ALPHA * keyboard.v_cmd_body
                smoothed_yaw_cmd = CMD_SMOOTH_ALPHA * keyboard.yaw_rate_cmd + (1 - CMD_SMOOTH_ALPHA) * smoothed_yaw_cmd

                # Rotate body command to world frame (reuse R_z buffer)
                yaw = state[5]
                c, s = np.cos(yaw), np.sin(yaw)
                R_z[0, 0] = c; R_z[0, 1] = -s; R_z[0, 2] = 0
                R_z[1, 0] = s; R_z[1, 1] = c;  R_z[1, 2] = 0
                R_z[2, 0] = 0; R_z[2, 1] = 0;  R_z[2, 2] = 1
                np.dot(R_z, smoothed_v_cmd, out=v_cmd_global)
                
                # --- 3. GAIT & SCHEDULE ---
                # Standing phase: all legs in stance for first STAND_DURATION seconds
                if current_time < STAND_DURATION:
                    contact_schedule_float.fill(1.0)  # All stance (reuse buffer)
                else:
                    contact_schedule = gait.get_contact_schedule(current_time - STAND_DURATION)
                    np.copyto(contact_schedule_float, contact_schedule)  # Copy to float buffer
                current_contact = contact_schedule_float[0, :]
                
                # --- 4. MPC (33 Hz) ---
                if mpc_counter % MPC_DECIMATION == 0:
                    # Generate Reference Trajectory (Pro Version: Integrates Yaw)
                    ref_traj = traj_gen.generate_reference(
                        state, v_cmd_global, smoothed_yaw_cmd, DEFAULT_HEIGHT
                    )
                    
                    # Solve MPC (Pro Version: Updates A/B Matrices internally)
                    # Returns (12,) forces - use pre-converted float buffer
                    mpc_f = mpc.solve(state, ref_traj, contact_schedule_float, foot_pos_rel)
                    
                    # Store for WBC
                    current_forces = mpc_f

                mpc_counter += 1

                # Smooth forces to reduce jerking at gait transitions
                smoothed_forces = FORCE_SMOOTH_ALPHA * current_forces + (1 - FORCE_SMOOTH_ALPHA) * smoothed_forces
                
                # --- 5. SWING LEG LOGIC (compute IK targets at 100Hz) ---
                # Get Joint States for IK/PD
                # Note: data.qpos[7:19] contains JOINT angles.
                # If XML defines Joints in a specific order, we must match it.
                # Usually XML Joint order matches Actuator order, but NOT always.
                # User's findings focused on Actuators. Let's assume standard qpos for now 
                # or we might need to map qpos too.
                # For safety, let's assume qpos indices 7-18 correspond to the same physical
                # order as the actuators if the XML is structured by leg blocks.
                # IF XML is: <body name="FR">...<joint>...<geom>...</body>
                # Then qpos will be FR first too.
                # Let's use the layout: qpos[7:10] -> Actuators[0:3] (FR)
                # This suggests we need to read qpos using the SAME map logic if joints are ordered by leg.
                
                # To be robust, let's assume qpos follows the "Actuator" order (FR, FL, RR, RL)
                # Logical Leg 0 (FL) is at Actuator/Joint index 3,4,5
                
                for i in range(4):
                    # Logical Leg Index i: 0=FL, 1=FR, 2=RL, 3=RR
                    idx_list = actuator_map[i]

                    if current_contact[i] == 1:
                        # === STANCE PHASE ===
                        swing_active[i] = 0
                        # Update Start Pos for next swing (in-place copy to avoid allocation)
                        swing_start_pos[i][:] = data.site_xpos[estimator.foot_ids[i]]
                    else:
                        # === SWING PHASE ===
                        swing_active[i] = 1
                        
                        # 1. Calculate Target (Raibert Heuristic)
                        # We only update the GOAL in this low-frequency loop
                        swing_t = gait.get_swing_state(current_time - STAND_DURATION, i)
                        
                        T_stance = gait.period * gait.stance_ratio
                        T_swing = gait.period * (1 - gait.stance_ratio)
                        v_body = state[6:9]

                        p_hip[:] = data.qpos[0:3]
                        # Projected Hip at landing
                        time_to_land = T_swing * (1 - swing_t)
                        p_hip_projected[:] = p_hip + v_body * time_to_land

                        k_raibert = 0.03
                        foot_offset_x = k_raibert * (v_body[0] - v_cmd_global[0])
                        foot_offset_y = k_raibert * (v_body[1] - v_cmd_global[1])
                        
                        # Store in swing_end_pos
                        p0 = swing_start_pos[i]
                        swing_end_pos[i][:] = p_hip_projected
                        swing_end_pos[i][0] += v_cmd_global[0] * T_stance / 2 + foot_offset_x
                        swing_end_pos[i][1] += v_cmd_global[1] * T_stance / 2 + foot_offset_y
                        swing_end_pos[i][2] = p0[2] # Maintain z-height preference (ground level)

                # --- 6. WHOLE BODY CONTROL (Stance) ---
                # Reuse pre-allocated forces_list buffer
                for i in range(4):
                    forces_list[i][:] = smoothed_forces[3*i : 3*i+3]

                # WBC computes tau_stance in the correct ACTUATOR ORDER internally
                tau_stance = wbc.compute_torques(forces_list, gravity_comp=False)

                # Build stance mask (reuse pre-allocated buffer)
                stance_mask_actuator.fill(0)
                for i in range(4):
                    is_stance = current_contact[i]
                    idx_list = actuator_map[i]
                    stance_mask_actuator[idx_list] = is_stance

                # Safety check (update flag at 100Hz)
                is_fallen = np.max(np.abs(state[3:5])) > 0.8  # roll/pitch > 45 deg

                # Debug Print
                if step_count % 200 == 0:
                    total_fz = smoothed_forces[2] + smoothed_forces[5] + smoothed_forces[8] + smoothed_forces[11]
                    # print(f"Z: {state[2]:.3f} | Fz: {total_fz:.1f}N | cmd: [{keyboard.v_cmd_body[0]:.1f}, {keyboard.v_cmd_body[1]:.1f}] yaw: {keyboard.yaw_rate_cmd:.1f}")

            # ---------------------------------------------
            # 1000 Hz: Swing Leg PD Control (runs every step for stability)
            # ---------------------------------------------
            # ---------------------------------------------
            # 1000 Hz: Swing Leg PD Control (Tactics)
            # ---------------------------------------------
            tau_cmd.fill(0)
            
            # Pre-calc Body Frame for IK (High Frequency)
            p_body_curr = data.qpos[0:3]
            # Fast Quat to Rot (avoid full estimator overhead)
            # q_rot = [w, x, y, z] -> data.qpos[3:7]
            q0, q1, q2, q3 = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
            # First row of rot mat
            r00 = 1 - 2*(q2**2 + q3**2)
            r01 = 2*(q1*q2 - q0*q3)
            r02 = 2*(q1*q3 + q0*q2)
            # Second row
            r10 = 2*(q1*q2 + q0*q3)
            r11 = 1 - 2*(q1**2 + q3**2)
            r12 = 2*(q2*q3 - q0*q1)
            # Third row
            r20 = 2*(q1*q3 - q0*q2)
            r21 = 2*(q2*q3 + q0*q1)
            r22 = 1 - 2*(q1**2 + q2**2)
            
            # Manual Transpose multiply roughly equal to R.T @ vec
            
            for i in range(4):
                if swing_active[i]:
                    idx_list = actuator_map[i]
                    
                    # A. RE-CALCULATE PHASE (Smooth)
                    t_sw = gait.get_swing_state(data.time - STAND_DURATION, i)
                    
                    # B. BEZIER INTERPOLATION
                    p0 = swing_start_pos[i]
                    pf = swing_end_pos[i]
                    
                    # Linear XY, Sine Z
                    # (In-line calc for speed)
                    des_x = p0[0] + (pf[0] - p0[0]) * t_sw
                    des_y = p0[1] + (pf[1] - p0[1]) * t_sw
                    des_z = p0[2] + swing_ctrl.swing_height * np.sin(t_sw * np.pi)
                    
                    # C. TRANSFORM TO HIP FRAME
                    # Rel World
                    rx = des_x - p_body_curr[0]
                    ry = des_y - p_body_curr[1]
                    rz = des_z - p_body_curr[2]
                    
                    # Rel Body (R.T @ rel_world)
                    bx = r00*rx + r10*ry + r20*rz
                    by = r01*rx + r11*ry + r21*rz
                    bz = r02*rx + r12*ry + r22*rz
                    
                    # Rel Hip
                    sign_x = 1 if i in [0, 1] else -1
                    sign_y = 1 if i in [0, 2] else -1
                    hx = bx - (sign_x * swing_ctrl.hip_offset_x)
                    hy = by - (sign_y * swing_ctrl.hip_offset_y)
                    hz = bz 

                    # D. INVERSE KINEMATICS (1kHz)
                    # We pass array to IK
                    target_hip = np.array([hx, hy, hz])
                    
                    try:
                         # Compute q_des immediately
                         q_des_now = swing_ctrl._compute_ik(target_hip, i)
                    except ValueError:
                         # Hold last known good if fail
                         q_des_now = np.array([data.qpos[7+ix] for ix in idx_list])

                    # E. PD CONTROL
                    # Read current joint state
                    for j, ix in enumerate(idx_list):
                        q_curr = data.qpos[7 + ix]
                        dq_curr = data.qvel[6 + ix]
                        
                        # PD Law
                        tau_swing = swing_ctrl.kp * (q_des_now[j] - q_curr) - swing_ctrl.kd * dq_curr
                        tau_cmd[ix] = tau_swing

            # --- MERGE & SAFETY ---
            tau_final = tau_stance * stance_mask_actuator + tau_cmd * (1 - stance_mask_actuator)

            # Fallback Safety (use flag updated at 100Hz)
            if is_fallen:
                tau_final = -5.0 * data.qvel[6:18]

            # Safety Ramp
            ramp_up = min(1.0, ramp_up + 0.0005)  # Slower ramp at 1kHz

            tau_final = np.clip(tau_final, -35.0, 35.0)
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
            print("Please run this script using: mjpython go2_mpc/controller_v2/main.py")
            print("="*60 + "\n")
        else:
            raise e