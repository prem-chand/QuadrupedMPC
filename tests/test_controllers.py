"""
Unit tests for Go2 MPC controllers.

Run with: python -m pytest tests/ -v
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestGaitGenerator:
    """Tests for GaitGenerator class."""

    def test_init(self):
        """Test GaitGenerator initialization."""
        from go2_mpc.controller.gait_generator import GaitGenerator

        gg = GaitGenerator(gait_period=0.5, stance_phase_ratio=0.5,
                          prediction_horizon=10, dt=0.02)

        assert gg.gait_period == 0.5
        assert gg.stance_phase_ratio == 0.5
        assert gg.swing_duration == 0.25  # 0.5 * (1 - 0.5)
        assert gg.num_legs == 4

    def test_swing_phase_bounds(self):
        """Test that swing phase stays in [0, 1] range."""
        from go2_mpc.controller.gait_generator import GaitGenerator

        gg = GaitGenerator(0.5, 0.5, 10, 0.02)

        # Test at multiple time points
        for t in np.linspace(0, 2.0, 100):
            for leg_idx in range(4):
                phase = gg.get_swing_phase(t, leg_idx)
                assert phase == -1.0 or (0.0 <= phase <= 1.0), \
                    f"Swing phase {phase} out of bounds at t={t}, leg={leg_idx}"

    def test_swing_phase_leg_pairs(self):
        """Test that diagonal leg pairs swing together."""
        from go2_mpc.controller.gait_generator import GaitGenerator

        gg = GaitGenerator(0.5, 0.5, 10, 0.02)

        # At t=0: FR and RL should be swinging (phase >= 0)
        #         FL and RR should be in stance (phase == -1)
        assert gg.get_swing_phase(0.0, 0) == -1.0  # FL in stance
        assert gg.get_swing_phase(0.0, 1) >= 0.0   # FR swinging
        assert gg.get_swing_phase(0.0, 2) >= 0.0   # RL swinging
        assert gg.get_swing_phase(0.0, 3) == -1.0  # RR in stance

    def test_contact_schedule_shape(self):
        """Test contact schedule has correct shape."""
        from go2_mpc.controller.gait_generator import GaitGenerator

        gg = GaitGenerator(0.5, 0.5, 10, 0.02)
        schedule = gg.trotting_gait(0.0)

        assert schedule.shape == (10, 4)  # (horizon, num_legs)
        assert schedule.dtype == int

    def test_contact_schedule_values(self):
        """Test contact schedule has only 0s and 1s."""
        from go2_mpc.controller.gait_generator import GaitGenerator

        gg = GaitGenerator(0.5, 0.5, 10, 0.02)
        schedule = gg.trotting_gait(0.0)

        assert np.all((schedule == 0) | (schedule == 1))


class TestSRBDynamics:
    """Tests for SRBDynamics class."""

    def test_init(self):
        """Test SRBDynamics initialization."""
        from go2_mpc.controller.srb_dynamics import SRBDynamics

        mass = 15.0
        inertia = np.diag([0.1, 0.2, 0.1])
        srb = SRBDynamics(mass, inertia)

        assert srb.mass == mass
        np.testing.assert_array_equal(srb.inertia, inertia)
        np.testing.assert_array_equal(srb.g, [0, 0, -9.81])

    def test_continuous_dynamics_gravity(self):
        """Test that gravity is applied correctly."""
        from go2_mpc.controller.srb_dynamics import SRBDynamics

        srb = SRBDynamics(10.0, np.eye(3))

        # State at rest
        state = np.zeros(12)
        forces = [np.zeros(3)] * 4
        foot_pos = [np.zeros(3)] * 4

        state_dot = srb.continuous_dynamics(state, forces, foot_pos)

        # With no forces, acceleration should be gravity
        np.testing.assert_array_almost_equal(state_dot[6:9], [0, 0, -9.81])


class TestConvexMPC:
    """Tests for ConvexMPC class."""

    def test_init(self):
        """Test ConvexMPC initialization."""
        from go2_mpc.controller.srb_dynamics import SRBDynamics
        from go2_mpc.controller.convex_mpc import ConvexMPC

        srb = SRBDynamics(15.0, np.diag([0.1, 0.2, 0.1]))
        Q = np.eye(12)
        R = np.eye(12) * 0.001

        mpc = ConvexMPC(srb, prediction_horizon=10, dt=0.02,
                       Q=Q, R=R, mu=0.6, f_max=150.0)

        assert mpc.prediction_horizon == 10
        assert mpc.dt == 0.02
        assert mpc.mu == 0.6
        assert mpc.f_max == 150.0

    def test_fallback_forces(self):
        """Test fallback forces are computed correctly."""
        from go2_mpc.controller.srb_dynamics import SRBDynamics
        from go2_mpc.controller.convex_mpc import ConvexMPC

        mass = 15.0
        srb = SRBDynamics(mass, np.diag([0.1, 0.2, 0.1]))
        mpc = ConvexMPC(srb, 10, 0.02, np.eye(12), np.eye(12)*0.001, 0.6, 150)

        # Test without contact schedule (distributes evenly)
        fallback = mpc._get_fallback_forces(None)
        assert fallback.shape == (12, 10)

        expected_fz = mass * 9.81 / 4.0
        for k in range(10):
            for i in range(4):
                assert fallback[3*i + 2, k] == pytest.approx(expected_fz, rel=0.01)

        # Test with contact schedule (only stance legs get force)
        contact_schedule = np.zeros((10, 4), dtype=int)
        contact_schedule[:, 0] = 1  # FL in stance
        contact_schedule[:, 3] = 1  # RR in stance

        fallback_trot = mpc._get_fallback_forces(contact_schedule)
        expected_fz_trot = mass * 9.81 / 2.0  # Split between 2 legs

        for k in range(10):
            assert fallback_trot[2, k] == pytest.approx(expected_fz_trot, rel=0.01)  # FL
            assert fallback_trot[5, k] == 0  # FR (swing)
            assert fallback_trot[8, k] == 0  # RL (swing)
            assert fallback_trot[11, k] == pytest.approx(expected_fz_trot, rel=0.01)  # RR


class TestSwingLegController:
    """Tests for SwingLegController class."""

    def test_init(self):
        """Test SwingLegController initialization."""
        from go2_mpc.controller.swing_leg_controller import SwingLegController

        slc = SwingLegController(kp=100.0, kd=10.0, swing_height=0.08)

        assert slc.kp == 100.0
        assert slc.kd == 10.0
        assert slc.swing_height == 0.08

    def test_trajectory_endpoints(self):
        """Test swing trajectory hits start and end points."""
        from go2_mpc.controller.swing_leg_controller import SwingLegController

        slc = SwingLegController(100.0, 10.0, 0.08)
        start = np.array([0.0, 0.0, 0.0])
        end = np.array([0.1, 0.0, 0.0])

        # At phase 0, should be at start
        pos, vel = slc.generate_swing_trajectory(start, end, 0.0, 0.25)
        np.testing.assert_array_almost_equal(pos[:2], start[:2])

        # At phase 1, should be at end
        pos, vel = slc.generate_swing_trajectory(start, end, 1.0, 0.25)
        np.testing.assert_array_almost_equal(pos[:2], end[:2])

    def test_trajectory_height(self):
        """Test swing trajectory reaches max height at mid-swing."""
        from go2_mpc.controller.swing_leg_controller import SwingLegController

        swing_height = 0.08
        slc = SwingLegController(100.0, 10.0, swing_height)
        start = np.array([0.0, 0.0, 0.0])
        end = np.array([0.1, 0.0, 0.0])

        # At phase 0.5, z should be at max height
        pos, vel = slc.generate_swing_trajectory(start, end, 0.5, 0.25)
        assert pos[2] == pytest.approx(swing_height, rel=0.01)

    def test_phase_clamping(self):
        """Test that phase is clamped to [0, 1]."""
        from go2_mpc.controller.swing_leg_controller import SwingLegController

        slc = SwingLegController(100.0, 10.0, 0.08)
        start = np.array([0.0, 0.0, 0.0])
        end = np.array([0.1, 0.0, 0.0])

        # Phase > 1 should be clamped
        pos1, _ = slc.generate_swing_trajectory(start, end, 1.5, 0.25)
        pos2, _ = slc.generate_swing_trajectory(start, end, 1.0, 0.25)
        np.testing.assert_array_almost_equal(pos1, pos2)

        # Phase < 0 should be clamped
        pos1, _ = slc.generate_swing_trajectory(start, end, -0.5, 0.25)
        pos2, _ = slc.generate_swing_trajectory(start, end, 0.0, 0.25)
        np.testing.assert_array_almost_equal(pos1, pos2)


class TestJointPDController:
    """Tests for JointPDController class."""

    def test_init(self):
        """Test JointPDController initialization."""
        from archive.debug.joint_pd_controller import JointPDController

        pd = JointPDController(kp=60.0, kd=5.0)

        assert pd.kp == 60.0
        assert pd.kd == 5.0
        assert len(pd.standing_pose) == 12

    def test_zero_error_zero_torque(self):
        """Test that zero position error gives zero torque (ignoring velocity)."""
        from archive.debug.joint_pd_controller import JointPDController

        pd = JointPDController(kp=60.0, kd=0.0)  # No damping

        # Create q with standing pose
        q = np.zeros(19)
        q[3] = 1.0  # quaternion w
        q[7:] = pd.standing_pose

        v = np.zeros(18)

        tau = pd.compute_torques(q, v)
        np.testing.assert_array_almost_equal(tau, np.zeros(12))

    def test_actuator_mapping(self):
        """Test joint to actuator mapping is correct."""
        from archive.debug.joint_pd_controller import JointPDController

        pd = JointPDController()

        # Check all joints map to valid actuator indices
        for joint_idx, act_idx in pd.joint_to_actuator.items():
            assert 0 <= joint_idx < 12
            assert 0 <= act_idx < 12

        # Check bijection (each actuator appears once)
        actuators = list(pd.joint_to_actuator.values())
        assert len(actuators) == len(set(actuators))


class TestTrajectoryGenerator:
    """Tests for TrajectoryGenerator class."""

    def test_init(self):
        """Test TrajectoryGenerator initialization."""
        from go2_mpc.controller.trajectory_generator import TrajectoryGenerator

        tg = TrajectoryGenerator(prediction_horizon=10, dt=0.02)

        assert tg.prediction_horizon == 10
        assert tg.dt == 0.02
        assert tg.state_size == 12

    def test_reference_trajectory_shape(self):
        """Test reference trajectory has correct shape."""
        from go2_mpc.controller.trajectory_generator import TrajectoryGenerator

        tg = TrajectoryGenerator(10, 0.02)
        current_state = np.zeros(12)
        velocity = np.array([0.5, 0, 0])
        height = 0.28

        x_ref = tg.generate_reference_trajectory(current_state, velocity, height)

        assert x_ref.shape == (12, 11)  # (state_size, horizon + 1)

    def test_reference_trajectory_initial_state(self):
        """Test reference trajectory starts at current state."""
        from go2_mpc.controller.trajectory_generator import TrajectoryGenerator

        tg = TrajectoryGenerator(10, 0.02)
        current_state = np.array([1, 2, 0.3, 0, 0, 0, 0.5, 0, 0, 0, 0, 0])
        velocity = np.array([0.5, 0, 0])
        height = 0.28

        x_ref = tg.generate_reference_trajectory(current_state, velocity, height)

        np.testing.assert_array_equal(x_ref[:, 0], current_state)


class TestMuJoCoIntegration:
    """Tests that require MuJoCo model."""

    @pytest.fixture
    def mujoco_setup(self):
        """Load MuJoCo model and create data."""
        import mujoco
        import os

        model_path = os.path.join(
            os.path.dirname(__file__), '..', 'go2_mpc', 'robot', 'scene.xml'
        )
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)

        # Initialize pose
        data.qpos[2] = 0.28
        data.qpos[3] = 1.0
        data.qpos[7:] = np.array([0.0, 0.9, -1.8] * 4)
        mujoco.mj_forward(model, data)

        return model, data

    def test_model_properties(self, mujoco_setup):
        """Test MuJoCo model has expected properties."""
        model, data = mujoco_setup

        assert model.nq == 19  # 7 freejoint + 12 joints
        assert model.nv == 18  # 6 freejoint DOFs + 12 joint DOFs
        assert model.nu == 12  # 12 actuators

    def test_robot_mass(self, mujoco_setup):
        """Test robot mass is approximately correct."""
        model, data = mujoco_setup

        total_mass = sum(model.body_mass)
        assert total_mass == pytest.approx(15.2, rel=0.05)

    def test_state_estimator(self, mujoco_setup):
        """Test StateEstimator with MuJoCo."""
        from go2_mpc.controller.state_estimator import StateEstimator

        model, data = mujoco_setup
        se = StateEstimator(model, data)

        state = se.estimate(data.qpos, data.qvel)

        assert state.shape == (12,)
        assert state[2] == pytest.approx(0.28, rel=0.01)  # Height

    def test_state_estimator_foot_positions(self, mujoco_setup):
        """Test foot position computation."""
        from go2_mpc.controller.state_estimator import StateEstimator

        model, data = mujoco_setup
        se = StateEstimator(model, data)

        foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        positions = se.compute_foot_positions(foot_names)

        assert len(positions) == 4
        for pos in positions:
            assert pos.shape == (3,)
            # Feet should be near ground
            assert pos[2] < 0.1

    def test_wbc_torque_shape(self, mujoco_setup):
        """Test WBC returns correct torque shape."""
        from go2_mpc.controller.wbc import WBC

        model, data = mujoco_setup
        wbc = WBC(model, data)

        forces = [np.array([0, 0, 30]) for _ in range(4)]
        foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

        tau = wbc.compute_torques(forces, foot_names)

        assert tau.shape == (12,)

    def test_joint_pd_standing(self, mujoco_setup):
        """Test that joint PD controller maintains standing."""
        import mujoco
        from archive.debug.joint_pd_controller import JointPDController

        model, data = mujoco_setup
        pd = JointPDController(kp=60.0, kd=5.0)

        initial_z = data.qpos[2]

        # Run for 500 steps (1 second)
        for _ in range(500):
            tau = pd.compute_torques(data.qpos, data.qvel)
            data.ctrl[:] = tau
            mujoco.mj_step(model, data)

        final_z = data.qpos[2]

        # Height should stay approximately the same
        assert abs(final_z - initial_z) < 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
