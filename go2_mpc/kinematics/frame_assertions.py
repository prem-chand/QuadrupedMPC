"""
Debug assertions for validating coordinate frame transformations.

These assertions catch errors early in development when integrating
manual kinematics/dynamics computations.
"""

import numpy as np


class FrameAssertions:
    """
    Debug assertions for coordinate frame validation.
    
    Use during development to verify frame transformations are correct.
    Disable in production for performance.
    """
    
    def __init__(self, enabled: bool = True, tolerance: float = 1e-5):
        """
        Parameters
        ----------
        enabled : bool
            If False, all assertions become no-ops.
        tolerance : float
            Numerical tolerance for comparisons.
        """
        self.enabled = enabled
        self.tolerance = tolerance
    
    def rotation_matrix_orthogonal(self, R: np.ndarray, name: str = "R"):
        """Assert R is a valid rotation matrix: R^T @ R = I, det(R) = +1."""
        if not self.enabled:
            return
            
        assert R.shape == (3, 3), f"{name} must be 3x3, got {R.shape}"
        
        identity = R.T @ R
        det = np.linalg.det(R)
        
        if not np.allclose(identity, np.eye(3), atol=self.tolerance):
            raise AssertionError(
                f"{name} is not orthogonal: R^T @ R =\n{identity}\n"
                f"Expected:\n{np.eye(3)}"
            )
        
        if not np.isclose(det, 1.0, atol=self.tolerance):
            raise AssertionError(
                f"{name} has invalid determinant: det(R) = {det}, expected +1"
            )
    
    def rotation_matrix_valid(self, R: np.ndarray, name: str = "R"):
        """Assert R is a valid SO(3) rotation matrix."""
        if not self.enabled:
            return
            
        self.rotation_matrix_orthogonal(R, name)
        
        if not np.allclose(R, R, atol=self.tolerance):
            raise AssertionError(f"{name} contains NaN or Inf")
    
    def quaternion_valid(self, q: np.ndarray, name: str = "q"):
        """Assert quaternion is valid: ||q|| = 1, no NaN/Inf."""
        if not self.enabled:
            return
            
        assert q.shape == (4,), f"{name} must be 4D, got {q.shape}"
        
        norm = np.linalg.norm(q)
        if not np.isclose(norm, 1.0, atol=self.tolerance):
            raise AssertionError(
                f"{name} is not normalized: ||q|| = {norm}, expected 1.0"
            )
        
        if np.any(np.isnan(q)) or np.any(np.isinf(q)):
            raise AssertionError(f"{name} contains NaN or Inf: {q}")
    
    def vector_finite(self, v: np.ndarray, name: str = "v"):
        """Assert vector has no NaN or Inf values."""
        if not self.enabled:
            return
            
        if not np.all(np.isfinite(v)):
            raise AssertionError(f"{name} contains NaN or Inf: {v}")
    
    def vector_shape(self, v: np.ndarray, expected_shape: tuple, name: str = "v"):
        """Assert vector has expected shape."""
        if not self.enabled:
            return
            
        if v.shape != expected_shape:
            raise AssertionError(
                f"{name} has wrong shape: {v.shape}, expected {expected_shape}"
            )
    
    def forces_valid(
        self,
        forces: np.ndarray,
        contact_schedule: np.ndarray,
        f_max: float,
        mu: float,
        name: str = "forces",
    ):
        """
        Assert forces satisfy friction cone and contact constraints.
        
        For stance legs: 0 <= Fz <= f_max, |Fx|, |Fy| <= mu * Fz
        For swing legs: forces should be zero
        """
        if not self.enabled:
            return
            
        assert forces.shape == (12,), f"{name} must be 12D, got {forces.shape}"
        assert contact_schedule.shape == (4,), "contact_schedule must be 4D"
        
        for i in range(4):
            idx = 3 * i
            fx, fy, fz = forces[idx], forces[idx + 1], forces[idx + 2]
            
            if contact_schedule[i] < 0.5:
                if not np.allclose(forces[idx:idx + 3], 0.0, atol=1.0):
                    raise AssertionError(
                        f"{name}[{i}] should be zero for swing leg, got "
                        f"[{fx}, {fy}, {fz}]"
                    )
            else:
                if fz < -self.tolerance:
                    raise AssertionError(
                        f"{name}[{i}] has negative normal force: {fz}"
                    )
                if fz > f_max:
                    raise AssertionError(
                        f"{name}[{i}] exceeds f_max: {fz} > {f_max}"
                    )
                
                friction_limit = mu * fz
                if np.abs(fx) > friction_limit + self.tolerance:
                    raise AssertionError(
                        f"{name}[{i}] exceeds friction in x: "
                        f"{np.abs(fx)} > {friction_limit}"
                    )
                if np.abs(fy) > friction_limit + self.tolerance:
                    raise AssertionError(
                        f"{name}[{i}] exceeds friction in y: "
                        f"{np.abs(fy)} > {friction_limit}"
                    )
    
    def jacobian_shape(self, J: np.ndarray, expected_shape: tuple, name: str = "J"):
        """Assert Jacobian has expected shape."""
        if not self.enabled:
            return
            
        if J.shape != expected_shape:
            raise AssertionError(
                f"{name} has wrong shape: {J.shape}, expected {expected_shape}"
            )
    
    def jacobian_rank(self, J: np.ndarray, min_rank: int = 3, name: str = "J"):
        """Assert Jacobian has sufficient rank (not singular)."""
        if not self.enabled:
            return
            
        rank = np.linalg.matrix_rank(J)
        if rank < min_rank:
            raise AssertionError(
                f"{name} has rank {rank}, expected at least {min_rank}. "
                f"Jacobian may be singular (near kinematic singularity)."
            )
    
    def foot_height_valid(
        self,
        foot_positions_world: list[np.ndarray],
        min_height: float = -0.5,
        max_height: float = 1.5,
        name: str = "foot",
    ):
        """Assert foot positions are in physically reasonable range."""
        if not self.enabled:
            return
            
        for i, p in enumerate(foot_positions_world):
            self.vector_shape(p, (3,), f"{name}_{i}")
            self.vector_finite(p, f"{name}_{i}")
            
            if p[2] < min_height or p[2] > max_height:
                raise AssertionError(
                    f"{name}[{i}] has invalid height: {p[2]} "
                    f"(expected {min_height} <= z <= {max_height})"
                )
    
    def joint_limits(
        self,
        q_leg: np.ndarray,
        leg_name: str,
        q_min: np.ndarray = None,
        q_max: np.ndarray = None,
    ):
        """
        Assert joint angles are within physical limits.
        
        Default limits for Go2:
        - Hip abduction: [-0.5, 0.5] rad
        - Hip flexion: [-1.0, 1.0] rad  
        - Knee flexion: [-2.6, -0.1] rad
        """
        if not self.enabled:
            return
            
        if q_min is None:
            q_min = np.array([-0.5, -1.0, -2.6])
        if q_max is None:
            q_max = np.array([0.5, 1.0, -0.1])
            
        for j in range(3):
            if q_leg[j] < q_min[j] - self.tolerance:
                raise AssertionError(
                    f"{leg_name} joint {j} below min: {q_leg[j]} < {q_min[j]}"
                )
            if q_leg[j] > q_max[j] + self.tolerance:
                raise AssertionError(
                    f"{leg_name} joint {j} above max: {q_leg[j]} > {q_max[j]}"
                )


_default_assertions = FrameAssertions()


def assert_rotation_valid(R: np.ndarray, name: str = "R"):
    """Convenience function using default assertions."""
    _default_assertions.rotation_matrix_valid(R, name)


def assert_quaternion_valid(q: np.ndarray, name: str = "q"):
    """Convenience function using default assertions."""
    _default_assertions.quaternion_valid(q, name)


def assert_forces_valid(
    forces: np.ndarray,
    contact_schedule: np.ndarray,
    f_max: float,
    mu: float,
    name: str = "forces",
):
    """Convenience function using default assertions."""
    _default_assertions.forces_valid(forces, contact_schedule, f_max, mu, name)
