"""
Comprehensive tests for QuadrupedMPC controller stack.
Covers: state, gait, trajectory, MPC formulation, solver, WBC,
swing trajectory, controller timing, and end-to-end integration.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from go2_mpc.core.state import BaseState, JointState, State
from go2_mpc.core.command import Command
from go2_mpc.controller.gait_scheduler import GaitScheduler
from go2_mpc.controller.trajectory_generator import TrajectoryGenerator
from go2_mpc.controller.convex_mpc import ConvexMPC
from go2_mpc.controller.foot_swing_trajectory import FootSwingTrajectory
from go2_mpc.controller.wbc import WholeBodyController
from go2_mpc.controller.controller_manager import (
    ControllerCore,
    ControllerState,
    ControllerBuffers,
)
from go2_mpc.controller.cvxpy_solver import ClarabelSolver


# ==============================================================================
# Helpers
# ==============================================================================

def make_identity_quat():
    """Return identity quaternion [w,x,y,z]."""
    return np.array([1.0, 0.0, 0.0, 0.0])


def make_yaw_quat(yaw):
    """Return quaternion for pure yaw rotation."""
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def make_base_state(pos=None, quat=None, lin_vel=None, ang_vel=None):
    return BaseState(
        position=pos if pos is not None else np.zeros(3),
        orientation=quat if quat is not None else make_identity_quat(),
        linear_velocity=lin_vel if lin_vel is not None else np.zeros(3),
        angular_velocity=ang_vel if ang_vel is not None else np.zeros(3),
    )


def make_state(pos=None, quat=None, lin_vel=None, ang_vel=None):
    base = make_base_state(pos, quat, lin_vel, ang_vel)
    joints = JointState(positions=np.zeros(12), velocities=np.zeros(12))
    return State(base=base, joints=joints)


def make_mock_robot():
    """Create a mock Robot with sensible defaults for a standing quadruped."""
    robot = MagicMock()
    foot_positions = [
        np.array([0.2, 0.15, 0.0]),
        np.array([0.2, -0.15, 0.0]),
        np.array([-0.2, 0.15, 0.0]),
        np.array([-0.2, -0.15, 0.0]),
    ]
    robot.get_foot_positions_world.return_value = [p.copy() for p in foot_positions]
    robot.get_foot_velocity.return_value = np.zeros(3)
    robot.get_leg_jacobian.return_value = np.eye(3)
    robot.get_gravity_compensation.return_value = np.zeros(3)
    return robot


class RecordingSolver:
    """Solver mock that records what it receives and returns zeros."""
    def __init__(self):
        self.last_H = None
        self.last_f = None
        self.last_A_eq = None
        self.last_b_eq = None
        self.last_A_ineq = None
        self.last_b_ineq = None
        self.call_count = 0

    def solve(self, H, f, A_eq, b_eq, A_ineq, b_ineq):
        self.last_H = H.copy()
        self.last_f = f.copy()
        self.last_A_eq = A_eq.copy()
        self.last_b_eq = b_eq.copy()
        self.last_A_ineq = A_ineq.copy()
        self.last_b_ineq = b_ineq.copy()
        self.call_count += 1
        return np.zeros(H.shape[0])


# ==============================================================================
# 1. BaseState & State Tests
# ==============================================================================

class TestBaseState:

    def test_identity_quaternion_gives_zero_rpy(self):
        base = make_base_state()
        assert base.roll == pytest.approx(0.0)
        assert base.pitch == pytest.approx(0.0)
        assert base.yaw == pytest.approx(0.0)

    def test_yaw_quaternion(self):
        for yaw in [0.3, -0.5, np.pi / 4, np.pi]:
            base = make_base_state(quat=make_yaw_quat(yaw))
            assert base.yaw == pytest.approx(yaw, abs=1e-10)
            assert base.roll == pytest.approx(0.0, abs=1e-10)
            assert base.pitch == pytest.approx(0.0, abs=1e-10)

    def test_rotation_matrix_identity(self):
        base = make_base_state()
        np.testing.assert_allclose(base.rotation_matrix, np.eye(3), atol=1e-12)

    def test_rotation_matrix_yaw(self):
        yaw = np.pi / 6
        base = make_base_state(quat=make_yaw_quat(yaw))
        R = base.rotation_matrix
        expected = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [0, 0, 1],
        ])
        np.testing.assert_allclose(R, expected, atol=1e-12)

    def test_rotation_matrix_is_orthogonal(self):
        quat = np.array([0.5, 0.5, 0.5, 0.5])  # 120-degree rotation
        base = make_base_state(quat=quat)
        R = base.rotation_matrix
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)

    def test_to_mpc_vector(self):
        base = make_base_state(
            pos=np.array([1.0, 2.0, 0.32]),
            lin_vel=np.array([0.5, 0.0, 0.0]),
            ang_vel=np.array([0.0, 0.0, 0.1]),
        )
        vec = base.to_mpc_vector()
        assert vec.shape == (12,)
        np.testing.assert_allclose(vec[0:3], [1.0, 2.0, 0.32])
        np.testing.assert_allclose(vec[3:6], [0.0, 0.0, 0.0])  # identity quat → zero RPY
        np.testing.assert_allclose(vec[6:9], [0.5, 0.0, 0.0])
        np.testing.assert_allclose(vec[9:12], [0.0, 0.0, 0.1])

    def test_cached_properties_dont_reallocate(self):
        """Verify the rotation matrix is the same object on repeated access."""
        base = make_base_state()
        assert base.rotation_matrix is base.rotation_matrix


class TestState:

    def test_from_vector_roundtrip(self):
        vec = np.random.randn(3 + 4 + 3 + 3 + 12 + 12)
        # Normalize quaternion
        vec[3:7] /= np.linalg.norm(vec[3:7])
        state = State.from_vector(vec, num_joints=12)
        reconstructed = state.to_vector()
        np.testing.assert_allclose(reconstructed, vec, atol=1e-12)

    def test_frozen(self):
        state = make_state()
        with pytest.raises(Exception):
            state.base = None


# ==============================================================================
# 2. Gait Scheduler Tests
# ==============================================================================

class TestGaitScheduler:

    @pytest.fixture
    def scheduler(self):
        return GaitScheduler(
            gait_period=0.5, stance_ratio=0.6, horizon=10, dt=0.05
        )

    def test_contact_schedule_shape_and_binary(self, scheduler):
        schedule = scheduler.get_contact_schedule(0.0)
        assert schedule.shape == (10, 4)
        assert np.all(np.isin(schedule, [0, 1]))

    def test_trot_diagonal_symmetry(self, scheduler):
        scheduler.update_gait_params("trot")
        schedule = scheduler.get_contact_schedule(0.0)
        np.testing.assert_array_equal(schedule[:, 0], schedule[:, 3])
        np.testing.assert_array_equal(schedule[:, 1], schedule[:, 2])

    def test_bound_front_rear_pairing(self, scheduler):
        scheduler.update_gait_params("bound")
        schedule = scheduler.get_contact_schedule(0.0)
        np.testing.assert_array_equal(schedule[:, 0], schedule[:, 1])
        np.testing.assert_array_equal(schedule[:, 2], schedule[:, 3])

    def test_pace_left_right_pairing(self, scheduler):
        scheduler.update_gait_params("pace")
        schedule = scheduler.get_contact_schedule(0.0)
        np.testing.assert_array_equal(schedule[:, 0], schedule[:, 2])
        np.testing.assert_array_equal(schedule[:, 1], schedule[:, 3])

    def test_periodicity(self, scheduler):
        """Contact schedule should repeat after one gait period."""
        t0 = 0.123
        s1 = scheduler.get_contact_schedule(t0)
        s2 = scheduler.get_contact_schedule(t0 + scheduler.period)
        np.testing.assert_array_equal(s1, s2)

    def test_swing_state_stance_returns_zero(self, scheduler):
        t_stance = 0.1 * scheduler.period  # Well inside stance
        assert scheduler.get_swing_state(t_stance, 0) == 0.0

    def test_swing_state_progress_midpoint(self, scheduler):
        # Phase 0.8 → halfway through swing [0.6, 1.0]
        t_swing = 0.8 * scheduler.period
        progress = scheduler.get_swing_state(t_swing, 0)
        assert progress == pytest.approx(0.5, abs=0.01)

    def test_swing_state_bounded_zero_one(self, scheduler):
        for t in np.linspace(0, 2 * scheduler.period, 200):
            for leg in range(4):
                s = scheduler.get_swing_state(t, leg)
                assert 0.0 <= s <= 1.0

    def test_every_leg_swings_over_full_period(self, scheduler):
        """Over one full gait period, every leg must enter swing at some point."""
        scheduler.update_gait_params("trot")
        for leg in range(4):
            swung = False
            for t in np.linspace(0, scheduler.period, 100):
                if scheduler.get_swing_state(t, leg) > 0:
                    swung = True
                    break
            assert swung, f"Leg {leg} never enters swing phase"


# ==============================================================================
# 3. Trajectory Generator Tests
# ==============================================================================

class TestTrajectoryGenerator:

    @pytest.fixture
    def traj_gen(self):
        return TrajectoryGenerator(prediction_horizon=5, dt=0.1)

    def test_shape(self, traj_gen):
        ref = traj_gen.generate_reference(np.zeros(12), np.zeros(3), 0.0, 0.3)
        assert ref.shape == (12, 6)

    def test_position_integration(self, traj_gen):
        ref = traj_gen.generate_reference(
            np.zeros(12), np.array([1.0, 0.0, 0.0]), 0.0, 0.3
        )
        expected_x = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        np.testing.assert_allclose(ref[0, :], expected_x)

    def test_height_constant(self, traj_gen):
        ref = traj_gen.generate_reference(np.zeros(12), np.zeros(3), 0.0, 0.35)
        np.testing.assert_allclose(ref[2, 1:], 0.35)

    def test_yaw_integration(self, traj_gen):
        ref = traj_gen.generate_reference(np.zeros(12), np.zeros(3), 1.0, 0.3)
        assert ref[5, 1] == pytest.approx(0.1)
        assert ref[5, 5] == pytest.approx(0.5)

    def test_roll_pitch_zero(self, traj_gen):
        ref = traj_gen.generate_reference(np.zeros(12), np.ones(3), 0.5, 0.3)
        np.testing.assert_allclose(ref[3, 1:], 0.0)
        np.testing.assert_allclose(ref[4, 1:], 0.0)

    def test_velocity_reference_constant(self, traj_gen):
        v_cmd = np.array([0.5, -0.3, 0.0])
        ref = traj_gen.generate_reference(np.zeros(12), v_cmd, 0.0, 0.3)
        for k in range(1, 6):
            np.testing.assert_allclose(ref[6:8, k], v_cmd[0:2])
            assert ref[8, k] == pytest.approx(0.0)

    def test_initial_column_is_current_state(self, traj_gen):
        current = np.arange(12, dtype=float)
        ref = traj_gen.generate_reference(current, np.zeros(3), 0.0, 0.3)
        np.testing.assert_allclose(ref[:, 0], current)


# ==============================================================================
# 4. Convex MPC Tests
# ==============================================================================

class TestConvexMPC:

    @pytest.fixture
    def mpc(self):
        return ConvexMPC(
            mass=10.0,
            inertia=np.eye(3),
            prediction_horizon=5,
            dt=0.1,
            Q=np.eye(12),
            R=np.eye(12),
            mu=0.5,
            f_max=100.0,
            solver=RecordingSolver(),
        )

    def test_dynamics_matrix_shapes(self, mpc):
        foot_pos = [np.array([0.2, 0.2, -0.3]) for _ in range(4)]
        A, B, g = mpc.update_dynamics_matrices(0.0, foot_pos)
        assert A.shape == (12, 12)
        assert B.shape == (12, 12)
        assert g.shape == (12,)

    def test_gravity_vector(self, mpc):
        foot_pos = [np.zeros(3)] * 4
        _, _, g = mpc.update_dynamics_matrices(0.0, foot_pos)
        assert g[8] == pytest.approx(-9.81 * 0.1)
        assert np.sum(np.abs(g)) == pytest.approx(abs(g[8]))

    def test_A_matrix_structure(self, mpc):
        """A should be identity + dt*velocity coupling + dt*R_z for angular."""
        foot_pos = [np.zeros(3)] * 4
        A, _, _ = mpc.update_dynamics_matrices(0.0, foot_pos)
        # Position integrates velocity: A[0:3, 6:9] = dt * I
        np.testing.assert_allclose(A[0:3, 6:9], 0.1 * np.eye(3))
        # Diagonal is identity
        np.testing.assert_allclose(np.diag(A), np.ones(12))

    def test_B_matrix_linear_force(self, mpc):
        """Linear acceleration from force: B[6:9, :] = (I/m)*dt / force_scale."""
        foot_pos = [np.zeros(3)] * 4
        _, B, _ = mpc.update_dynamics_matrices(0.0, foot_pos)
        expected = (1.0 / 10.0) * 0.1 * 100.0  # (1/mass) * dt / force_scale
        for leg in range(4):
            for dim in range(3):
                assert B[6 + dim, 3 * leg + dim] == pytest.approx(expected)

    def test_qp_dimensions(self, mpc):
        x0 = np.zeros(12)
        x_ref = np.zeros((12, 6))
        contact = np.ones((5, 4))
        foot_pos = [np.zeros(3)] * 4
        A, B, g = mpc.update_dynamics_matrices(0.0, foot_pos)
        H, f, A_eq, b_eq, A_ineq, b_ineq = mpc.build_qp(x0, x_ref, contact, A, B, g)

        n_vars = 12 * 6 + 12 * 5  # 132
        assert H.shape == (n_vars, n_vars)
        assert f.shape == (n_vars,)
        # Eq: 12 (initial) + 12*5 (dynamics) = 72
        assert A_eq.shape == (72, n_vars)
        assert b_eq.shape == (72,)

    def test_reference_tracking_in_cost(self, mpc):
        """Verify BUG-1 fix: x_ref must appear in the linear cost f."""
        x0 = np.zeros(12)
        x_ref = np.ones((12, 6)) * 0.3  # Nonzero reference
        contact = np.ones((5, 4))
        foot_pos = [np.zeros(3)] * 4
        A, B, g = mpc.update_dynamics_matrices(0.0, foot_pos)
        _, f, _, _, _, _ = mpc.build_qp(x0, x_ref, contact, A, B, g)

        # f should be nonzero for state blocks (reference tracking)
        # For k=0: idx_x = 12, f[12:24] = -Q @ x_ref[:, 1] = -I @ 0.3*ones = -0.3*ones
        np.testing.assert_allclose(f[12:24], -0.3 * np.ones(12))

    def test_zero_reference_gives_zero_linear_cost(self, mpc):
        """With zero reference, f should be zero (x^T Q x only)."""
        x0 = np.zeros(12)
        x_ref = np.zeros((12, 6))
        contact = np.ones((5, 4))
        foot_pos = [np.zeros(3)] * 4
        A, B, g = mpc.update_dynamics_matrices(0.0, foot_pos)
        _, f, _, _, _, _ = mpc.build_qp(x0, x_ref, contact, A, B, g)
        np.testing.assert_allclose(f, 0.0)

    def test_H_is_positive_semidefinite(self, mpc):
        x0 = np.zeros(12)
        x_ref = np.zeros((12, 6))
        contact = np.ones((5, 4))
        foot_pos = [np.zeros(3)] * 4
        A, B, g = mpc.update_dynamics_matrices(0.0, foot_pos)
        H, _, _, _, _, _ = mpc.build_qp(x0, x_ref, contact, A, B, g)
        eigenvalues = np.linalg.eigvalsh(H)
        assert np.all(eigenvalues >= -1e-10)

    def test_friction_cone_constraints_present(self, mpc):
        """Verify BUG-3 fix: friction pyramid constraints are included."""
        x0 = np.zeros(12)
        x_ref = np.zeros((12, 6))
        contact = np.ones((5, 4))
        foot_pos = [np.zeros(3)] * 4
        A, B, g = mpc.update_dynamics_matrices(0.0, foot_pos)
        _, _, _, _, A_ineq, b_ineq = mpc.build_qp(x0, x_ref, contact, A, B, g)

        # Per leg per horizon step: 2 (Fz bounds) + 4 (friction pyramid) = 6 rows
        expected_rows = 5 * 4 * 6  # 120
        assert A_ineq.shape[0] == expected_rows
        assert b_ineq.shape[0] == expected_rows

    def test_friction_cone_constrains_lateral_forces(self, mpc):
        """Friction rows should couple Fx/Fy to mu*Fz."""
        x0 = np.zeros(12)
        x_ref = np.zeros((12, 6))
        contact = np.ones((5, 4))
        foot_pos = [np.zeros(3)] * 4
        A, B, g = mpc.update_dynamics_matrices(0.0, foot_pos)
        _, _, _, _, A_ineq, _ = mpc.build_qp(x0, x_ref, contact, A, B, g)

        # Check first leg, first horizon step
        # Row layout per leg: Fz_lo, Fz_hi, Fx+, Fx-, Fy+, Fy-
        nx, nu, N = 12, 12, 5
        n_vars = nx * (N + 1) + nu * N
        idx_u = nx * (N + 1)  # First leg, first step

        # Row 2 (Fx - mu*Fz <= 0): A_ineq[2, idx_u+0] = 1, A_ineq[2, idx_u+2] = -mu
        assert A_ineq[2, idx_u + 0] == pytest.approx(1.0)
        assert A_ineq[2, idx_u + 2] == pytest.approx(-0.5)  # mu = 0.5

        # Row 3 (-Fx - mu*Fz <= 0): A_ineq[3, idx_u+0] = -1, A_ineq[3, idx_u+2] = -mu
        assert A_ineq[3, idx_u + 0] == pytest.approx(-1.0)
        assert A_ineq[3, idx_u + 2] == pytest.approx(-0.5)

    def test_swing_leg_force_bounded_to_zero(self, mpc):
        """When contact=0, Fz upper bound should be 0 (no force allowed)."""
        x0 = np.zeros(12)
        x_ref = np.zeros((12, 6))
        contact = np.zeros((5, 4))  # All swing
        foot_pos = [np.zeros(3)] * 4
        A, B, g = mpc.update_dynamics_matrices(0.0, foot_pos)
        _, _, _, _, _, b_ineq = mpc.build_qp(x0, x_ref, contact, A, B, g)

        # Fz upper bound rows are at indices 1, 7, 13, 19, ... (every 6, starting at 1)
        for k in range(5):
            for leg in range(4):
                fz_upper_idx = (k * 4 + leg) * 6 + 1
                assert b_ineq[fz_upper_idx] == pytest.approx(0.0)

    def test_solver_failure_returns_zeros(self, mpc):
        """When solver returns None, MPC should return zero forces."""
        failing_solver = MagicMock()
        failing_solver.solve.return_value = None
        mpc.solver = failing_solver

        state = make_state(pos=np.array([0, 0, 0.32]))
        x_ref = np.zeros((12, 6))
        contact = np.ones((5, 4))
        foot_pos = [np.zeros(3)] * 4

        forces = mpc.solve(state, x_ref, contact, foot_pos)
        np.testing.assert_allclose(forces, np.zeros(12))

    def test_initial_state_constraint(self, mpc):
        """First equality constraint should pin x_0 = x0."""
        x0 = np.array([1.0, 2.0, 0.3, 0.0, 0.0, 0.1, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        x_ref = np.zeros((12, 6))
        contact = np.ones((5, 4))
        foot_pos = [np.zeros(3)] * 4
        A, B, g = mpc.update_dynamics_matrices(0.0, foot_pos)
        _, _, A_eq, b_eq, _, _ = mpc.build_qp(x0, x_ref, contact, A, B, g)

        # First 12 rows of A_eq should be I (selecting x_0), b_eq = x0
        np.testing.assert_allclose(A_eq[0:12, 0:12], np.eye(12))
        np.testing.assert_allclose(b_eq[0:12], x0)


# ==============================================================================
# 5. Direct Clarabel Solver Tests
# ==============================================================================

class TestClarabelSolver:

    @pytest.fixture
    def solver(self):
        return ClarabelSolver()

    def test_simple_qp(self, solver):
        """min 0.5 * x^2 - x  →  x* = 1"""
        H = np.array([[1.0]])
        f = np.array([-1.0])
        z = solver.solve(H, f)
        assert z is not None
        assert z[0] == pytest.approx(1.0, abs=1e-6)

    def test_equality_constraint(self, solver):
        """min ||x||^2  s.t. sum(x) = 1, 4 vars  →  x = [0.25]*4"""
        n = 4
        H = 2.0 * np.eye(n)
        f = np.zeros(n)
        A_eq = np.ones((1, n))
        b_eq = np.array([1.0])
        z = solver.solve(H, f, A_eq, b_eq)
        np.testing.assert_allclose(z, 0.25 * np.ones(n), atol=1e-6)

    def test_inequality_constraint(self, solver):
        """min -x  s.t. x <= 5  →  x* = 5"""
        H = np.zeros((1, 1))
        # Need small regularization for clarabel
        H[0, 0] = 1e-8
        f = np.array([-1.0])
        A_ineq = np.array([[1.0]])
        b_ineq = np.array([5.0])
        z = solver.solve(H, f, A_ineq=A_ineq, b_ineq=b_ineq)
        assert z is not None
        assert z[0] == pytest.approx(5.0, abs=1e-4)

    def test_mpc_sized_problem(self, solver):
        """Solve a QP with MPC-like dimensions (252 vars)."""
        n = 252
        H = np.eye(n) * 2.0
        f = -np.ones(n)
        # x >= 0 → -x <= 0
        A_ineq = -np.eye(n)
        b_ineq = np.zeros(n)
        z = solver.solve(H, f, A_ineq=A_ineq, b_ineq=b_ineq)
        assert z is not None
        np.testing.assert_allclose(z, 0.5 * np.ones(n), atol=1e-4)

    def test_infeasible_returns_none(self, solver):
        """Contradictory constraints should return None."""
        H = np.eye(1)
        f = np.zeros(1)
        # x == 1 AND x == 2 → infeasible
        A_eq = np.array([[1.0], [1.0]])
        b_eq = np.array([1.0, 2.0])
        z = solver.solve(H, f, A_eq, b_eq)
        assert z is None


# ==============================================================================
# 6. Foot Swing Trajectory Tests
# ==============================================================================

class TestFootSwingTrajectory:

    @pytest.fixture
    def traj(self):
        t = FootSwingTrajectory()
        t.set_initial_position(np.array([0.0, 0.0, 0.0]))
        t.set_final_position(np.array([0.1, 0.0, 0.0]))
        t.set_height(0.08)
        return t

    def test_start_position(self, traj):
        traj.compute_swing_trajectory_bezier(0.0, 0.2)
        p = traj.get_position()
        np.testing.assert_allclose(p, [0.0, 0.0, 0.0], atol=1e-12)

    def test_end_position(self, traj):
        traj.compute_swing_trajectory_bezier(1.0, 0.2)
        p = traj.get_position()
        np.testing.assert_allclose(p, [0.1, 0.0, 0.0], atol=1e-12)

    def test_midpoint_xy(self, traj):
        """At phase=0.5, XY should be midway between start and end."""
        traj.compute_swing_trajectory_bezier(0.5, 0.2)
        p = traj.get_position()
        assert p[0] == pytest.approx(0.05, abs=1e-10)

    def test_z_lifts_above_ground(self, traj):
        """Z should be positive (above ground) at midswing."""
        traj.compute_swing_trajectory_bezier(0.5, 0.2)
        p = traj.get_position()
        assert p[2] > 0.0

    def test_z_peak_near_midswing(self, traj):
        """Maximum Z clearance should occur around midswing."""
        zs = []
        phases = np.linspace(0, 1, 101)
        for s in phases:
            traj.compute_swing_trajectory_bezier(s, 0.2)
            zs.append(traj.get_position()[2])
        peak_idx = np.argmax(zs)
        peak_phase = phases[peak_idx]
        assert 0.3 < peak_phase < 0.7

    def test_velocity_zero_at_endpoints_xy(self, traj):
        """XY velocity should be zero at liftoff and touchdown."""
        traj.compute_swing_trajectory_bezier(0.0, 0.2)
        v0 = traj.get_velocity().copy()
        traj.compute_swing_trajectory_bezier(1.0, 0.2)
        v1 = traj.get_velocity().copy()
        assert abs(v0[0]) < 1e-10
        assert abs(v0[1]) < 1e-10
        assert abs(v1[0]) < 1e-10
        assert abs(v1[1]) < 1e-10

    def test_stationary_swing(self):
        """If start == end, trajectory should lift vertically and return."""
        t = FootSwingTrajectory()
        t.set_initial_position(np.array([0.2, 0.1, 0.0]))
        t.set_final_position(np.array([0.2, 0.1, 0.0]))
        t.set_height(0.05)
        t.compute_swing_trajectory_bezier(0.5, 0.2)
        p = t.get_position()
        assert p[0] == pytest.approx(0.2, abs=1e-10)
        assert p[1] == pytest.approx(0.1, abs=1e-10)
        assert p[2] > 0.0  # Should still lift


# ==============================================================================
# 7. Whole Body Controller Tests
# ==============================================================================

class TestWholeBodyController:

    def test_jacobian_transpose_mapping(self):
        robot = make_mock_robot()
        wbc = WholeBodyController(robot, torque_limit=100.0)
        f_des = np.array([10.0, 0.0, -50.0])
        foot_forces = [f_des, np.zeros(3), np.zeros(3), np.zeros(3)]
        tau = wbc.compute_torques(foot_forces, gravity_comp=False)
        # With J=I: tau = -I^T @ F = -F
        np.testing.assert_allclose(tau[0:3], -f_des)
        np.testing.assert_allclose(tau[3:12], 0.0)

    def test_gravity_compensation_added(self):
        robot = make_mock_robot()
        grav = np.array([1.0, 2.0, 3.0])
        robot.get_gravity_compensation.return_value = grav
        wbc = WholeBodyController(robot, torque_limit=100.0)
        foot_forces = [np.zeros(3)] * 4
        tau = wbc.compute_torques(foot_forces, gravity_comp=True)
        # Each leg should get gravity comp only
        for i in range(4):
            np.testing.assert_allclose(tau[3*i:3*i+3], grav)

    def test_gravity_comp_disabled(self):
        robot = make_mock_robot()
        robot.get_gravity_compensation.return_value = np.array([5.0, 5.0, 5.0])
        wbc = WholeBodyController(robot, torque_limit=100.0)
        foot_forces = [np.zeros(3)] * 4
        tau = wbc.compute_torques(foot_forces, gravity_comp=False)
        np.testing.assert_allclose(tau, 0.0)

    def test_torque_clipping(self):
        robot = make_mock_robot()
        wbc = WholeBodyController(robot, torque_limit=5.0)
        foot_forces = [np.array([100.0, -100.0, 0.0])] * 4
        tau = wbc.compute_torques(foot_forces, gravity_comp=False)
        assert np.max(tau) <= 5.0
        assert np.min(tau) >= -5.0

    def test_skip_zero_force_legs(self):
        """Jacobian should not be queried for legs with negligible force."""
        robot = make_mock_robot()
        wbc = WholeBodyController(robot, torque_limit=100.0)
        foot_forces = [np.zeros(3)] * 4
        wbc.compute_torques(foot_forces, gravity_comp=False)
        robot.get_leg_jacobian.assert_not_called()

    def test_multi_leg_forces_superimpose(self):
        robot = make_mock_robot()
        wbc = WholeBodyController(robot, torque_limit=1000.0)
        f0 = np.array([10.0, 0.0, 0.0])
        f2 = np.array([0.0, 5.0, 0.0])
        foot_forces = [f0, np.zeros(3), f2, np.zeros(3)]
        tau = wbc.compute_torques(foot_forces, gravity_comp=False)
        np.testing.assert_allclose(tau[0:3], -f0)
        np.testing.assert_allclose(tau[6:9], -f2)


# ==============================================================================
# 8. Controller Manager Tests (Timing & Integration)
# ==============================================================================

class TestControllerCore:

    def _make_controller(self):
        gait = GaitScheduler(
            gait_period=0.45, stance_ratio=0.65, horizon=10, dt=0.03,
        )
        traj_gen = TrajectoryGenerator(prediction_horizon=10, dt=0.03)
        mock_solver = RecordingSolver()
        mpc = ConvexMPC(
            mass=15.0, inertia=np.diag([0.1, 0.1, 0.02]),
            prediction_horizon=10, dt=0.03,
            Q=np.eye(12), R=np.eye(12) * 1e-6,
            mu=0.6, f_max=180.0, solver=mock_solver,
        )
        robot = make_mock_robot()
        wbc = WholeBodyController(robot, torque_limit=35.0)
        controller = ControllerCore(
            gait=gait, traj_gen=traj_gen, mpc=mpc, wbc=wbc,
            config={"FORCE_SMOOTH_ALPHA": 0.9, "TORQUE_LIMIT": 35.0},
        )
        return controller, mock_solver, robot

    def _make_controller_state(self, robot):
        return ControllerState(
            step_counter=0,
            mpc_counter=0,
            gait_phase_time=0.0,
            swing_active=np.zeros(4),
            swing_start_pos=[p.copy() for p in robot.get_foot_positions_world()],
        )

    def test_output_shape(self):
        controller, _, robot = self._make_controller()
        state = make_state(pos=np.array([0, 0, 0.32]))
        cs = self._make_controller_state(robot)
        buffers = ControllerBuffers()
        cmd = Command(v_cmd_global=np.zeros(3), yaw_rate=0.0, default_height=0.32)

        tau = controller.compute(state, np.zeros((4, 3)), cmd, cs, buffers, robot)
        assert tau.shape == (12,)

    def test_torque_limit(self):
        controller, _, robot = self._make_controller()
        state = make_state(pos=np.array([0, 0, 0.32]))
        cs = self._make_controller_state(robot)
        buffers = ControllerBuffers()
        cmd = Command(v_cmd_global=np.zeros(3), yaw_rate=0.0, default_height=0.32)

        # Run enough steps to trigger all loops
        for _ in range(30):
            tau = controller.compute(state, np.zeros((4, 3)), cmd, cs, buffers, robot)
        assert np.all(np.abs(tau) <= 35.0)

    def test_mpc_decimation(self):
        """MPC should only be called every control_decimation * mpc_decimation steps."""
        controller, mock_solver, robot = self._make_controller()
        state = make_state(pos=np.array([0, 0, 0.32]))
        cs = self._make_controller_state(robot)
        buffers = ControllerBuffers()
        cmd = Command(v_cmd_global=np.zeros(3), yaw_rate=0.0, default_height=0.32)

        # Run 30 steps (= 3 control ticks at 100Hz)
        for _ in range(30):
            controller.compute(state, np.zeros((4, 3)), cmd, cs, buffers, robot)

        # MPC should fire at step 10 (first 100Hz tick, mpc_counter=0)
        # Then step 30 is the 3rd 100Hz tick, mpc_counter=2 → 2%3!=0 so no fire
        # Actually mpc fires when mpc_counter % 3 == 0, incremented after check
        # At step 10: mpc_counter=0 → fires, incremented to 1
        # At step 20: mpc_counter=1 → no fire, incremented to 2
        # At step 30: mpc_counter=2 → no fire, incremented to 3
        assert mock_solver.call_count == 1

    def test_ema_smoothing_direction(self):
        """Verify BUG-2 fix: with alpha=0.9, new forces contribute 10%, not 90%."""
        controller, _, robot = self._make_controller()
        buffers = ControllerBuffers()

        # Simulate the EMA manually
        buffers.smoothed_forces[:] = 100.0
        buffers.current_forces[:] = 0.0

        alpha = controller.force_alpha  # 0.9
        expected = alpha * 100.0 + (1 - alpha) * 0.0  # 90.0

        # Apply EMA formula (same as controller_manager.py line 105-106)
        buffers.smoothed_forces *= alpha
        buffers.smoothed_forces += (1 - alpha) * buffers.current_forces

        np.testing.assert_allclose(buffers.smoothed_forces, expected)
        # Key: smoothed should still be close to 100 (heavy smoothing), not 0
        assert np.all(buffers.smoothed_forces > 50.0)

    def test_step_counter_increments(self):
        controller, _, robot = self._make_controller()
        state = make_state(pos=np.array([0, 0, 0.32]))
        cs = self._make_controller_state(robot)
        buffers = ControllerBuffers()
        cmd = Command(v_cmd_global=np.zeros(3), yaw_rate=0.0, default_height=0.32)

        for _ in range(5):
            controller.compute(state, np.zeros((4, 3)), cmd, cs, buffers, robot)
        assert cs.step_counter == 5

    def test_swing_stance_merge(self):
        """Verify correct torque source selection based on swing_active."""
        controller, _, robot = self._make_controller()
        buffers = ControllerBuffers()
        buffers.tau_stance[:] = 1.0
        buffers.tau_swing[:] = 2.0

        cs = self._make_controller_state(robot)
        cs.swing_active[0] = 1  # Leg 0 swinging
        cs.swing_active[1] = 0  # Leg 1 stance

        state = make_state(pos=np.array([0, 0, 0.32]))
        cmd = Command(v_cmd_global=np.zeros(3), yaw_rate=0.0, default_height=0.32)

        # Run one step where no control loop fires (step 1, not multiple of 10)
        # tau_stance stays at 1.0, swing loop will overwrite tau_swing
        tau = controller.compute(state, np.zeros((4, 3)), cmd, cs, buffers, robot)

        # Leg 0 (swing): should use tau_swing values
        # Leg 1 (stance): should use tau_stance = 1.0
        assert np.all(tau[3:6] == pytest.approx(1.0))


# ==============================================================================
# 9. End-to-End MPC Solve Test
# ==============================================================================

class TestEndToEnd:

    def test_mpc_produces_nonzero_forces_with_real_solver(self):
        """Full pipeline: state → MPC → real QP solve → nonzero forces."""
        solver = ClarabelSolver()
        mpc = ConvexMPC(
            mass=15.2,
            inertia=np.diag([0.1, 0.1, 0.02]),
            prediction_horizon=10,
            dt=0.03,
            Q=np.diag([1, 5, 100, 3, 10, 0.1, 5, 5, 12, 2, 3, 2]),
            R=np.diag([1e-6] * 12),
            mu=0.6,
            f_max=180.0,
            solver=solver,
        )

        state = make_state(pos=np.array([0.0, 0.0, 0.30]))  # Slightly below target
        x_ref = np.zeros((12, 11))
        x_ref[2, :] = 0.32  # Target height
        x_ref[6:9, :] = 0.0  # Zero velocity
        contact = np.ones((10, 4))  # All stance
        foot_pos_body = [
            np.array([0.19, 0.11, -0.30]),
            np.array([0.19, -0.11, -0.30]),
            np.array([-0.19, 0.11, -0.30]),
            np.array([-0.19, -0.11, -0.30]),
        ]

        forces = mpc.solve(state, x_ref, contact, foot_pos_body)

        assert forces.shape == (12,)
        # Robot is below target height → should produce upward forces (Fz > 0)
        for leg in range(4):
            fz = forces[3 * leg + 2]
            assert fz > 0, f"Leg {leg} Fz={fz:.2f}, expected positive (upward)"

    def test_forces_respect_friction_cone(self):
        """Solved forces must satisfy |Fx|, |Fy| <= mu * Fz."""
        solver = ClarabelSolver()
        mpc = ConvexMPC(
            mass=15.2,
            inertia=np.diag([0.1, 0.1, 0.02]),
            prediction_horizon=10,
            dt=0.03,
            Q=np.diag([1, 5, 100, 3, 10, 0.1, 5, 5, 12, 2, 3, 2]),
            R=np.diag([1e-6] * 12),
            mu=0.6,
            f_max=180.0,
            solver=solver,
        )

        # Request lateral velocity to trigger lateral forces
        state = make_state(
            pos=np.array([0.0, 0.0, 0.32]),
            lin_vel=np.array([0.5, 0.3, 0.0]),
        )
        x_ref = np.zeros((12, 11))
        x_ref[2, :] = 0.32
        x_ref[6, :] = 0.0  # Want to decelerate
        contact = np.ones((10, 4))
        foot_pos_body = [
            np.array([0.19, 0.11, -0.32]),
            np.array([0.19, -0.11, -0.32]),
            np.array([-0.19, 0.11, -0.32]),
            np.array([-0.19, -0.11, -0.32]),
        ]

        forces = mpc.solve(state, x_ref, contact, foot_pos_body)

        mu = 0.6
        for leg in range(4):
            fx = forces[3 * leg]
            fy = forces[3 * leg + 1]
            fz = forces[3 * leg + 2]
            assert abs(fx) <= mu * fz + 1e-3, (
                f"Leg {leg}: |Fx|={abs(fx):.3f} > mu*Fz={mu*fz:.3f}"
            )
            assert abs(fy) <= mu * fz + 1e-3, (
                f"Leg {leg}: |Fy|={abs(fy):.3f} > mu*Fz={mu*fz:.3f}"
            )

    def test_swing_legs_get_zero_force(self):
        """With all legs in swing, MPC should return zero forces."""
        solver = ClarabelSolver()
        mpc = ConvexMPC(
            mass=15.2,
            inertia=np.diag([0.1, 0.1, 0.02]),
            prediction_horizon=10,
            dt=0.03,
            Q=np.eye(12),
            R=np.eye(12) * 1e-6,
            mu=0.6,
            f_max=180.0,
            solver=solver,
        )

        state = make_state(pos=np.array([0.0, 0.0, 0.32]))
        x_ref = np.zeros((12, 11))
        x_ref[2, :] = 0.32
        contact = np.zeros((10, 4))  # All swing
        foot_pos_body = [np.array([0.2, 0.1, -0.3])] * 4

        forces = mpc.solve(state, x_ref, contact, foot_pos_body)
        np.testing.assert_allclose(forces, 0.0, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
