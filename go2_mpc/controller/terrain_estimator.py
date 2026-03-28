"""
Contact-embedded terrain height and slope estimation.

Estimates terrain properties from foot contacts:
- Height per foot
- Surface slope
- Normal vector

Used for MPC state augmentation on rough terrain.
"""

import numpy as np


class TerrainEstimator:
    """
    Estimate terrain properties from foot contacts.
    
    Uses stance foot positions to estimate:
    - Local terrain height at each foot
    - Average terrain slope (roll, pitch)
    - Surface normal vector
    """

    def __init__(self, alpha: float = 0.1):
        """Initialize with EMA smoothing factor."""
        self._alpha = alpha
        self._prev_slope_roll = 0.0
        self._prev_slope_pitch = 0.0

    def estimate(
        self,
        foot_positions: list[np.ndarray],
        contact_schedule: np.ndarray,
        base_height: float = 0.32,
    ) -> dict:
        """
        Estimate terrain from foot contacts.
        
        Parameters
        ----------
        foot_positions : list of np.ndarray
            Four foot positions in world frame.
        contact_schedule : np.ndarray, shape (4,)
            Binary contact flags.
        base_height : float
            Nominal base height.
            
        Returns
        -------
        dict with keys:
            - 'height_per_foot': np.ndarray (4,) foot heights
            - 'slope_roll': float terrain roll angle (rad)
            - 'slope_pitch': float terrain pitch angle (rad)
            - 'normal': np.ndarray (3,) surface normal vector
            - 'avg_height': float average terrain height
        """
        stance_feet = []
        for i in range(4):
            if contact_schedule[i] > 0.5:
                stance_feet.append(foot_positions[i])
        
        if len(stance_feet) < 2:
            return self._default_estimate()
        
        heights = np.array([f[2] for f in stance_feet])
        avg_height = np.mean(heights)
        
        if len(stance_feet) >= 3:
            slope_roll, slope_pitch = self._estimate_slope(stance_feet)
            normal = self._estimate_normal(stance_feet)
            
            # Apply EMA smoothing
            slope_roll = (1 - self._alpha) * self._prev_slope_roll + self._alpha * slope_roll
            slope_pitch = (1 - self._alpha) * self._prev_slope_pitch + self._alpha * slope_pitch
            self._prev_slope_roll = slope_roll
            self._prev_slope_pitch = slope_pitch
        else:
            slope_roll = self._prev_slope_roll
            slope_pitch = self._prev_slope_pitch
            normal = np.array([0, 0, 1])
        
        # Height per foot: only stance feet have valid terrain height
        height_per_foot = np.full(4, np.nan)
        for i in range(4):
            if contact_schedule[i] > 0.5:
                height_per_foot[i] = foot_positions[i][2]
        
        return {
            'height_per_foot': height_per_foot,
            'slope_roll': slope_roll,
            'slope_pitch': slope_pitch,
            'normal': normal,
            'avg_height': avg_height,
        }

    def _estimate_slope(self, feet: list[np.ndarray]) -> tuple[float, float]:
        """Estimate terrain slope from 3+ stance feet."""
        if len(feet) < 3:
            return 0.0, 0.0
        
        pts = np.array(feet[:3])
        
        p1, p2, p3 = pts[0], pts[1], pts[2]
        
        v1 = p2 - p1
        v2 = p3 - p1
        
        normal = np.cross(v1, v2)
        
        if normal[2] < 0:
            normal = -normal
        
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            return 0.0, 0.0
        normal = normal / norm
        
        slope_roll = np.arctan2(normal[1], normal[2])
        slope_pitch = np.arctan2(-normal[0], normal[2])
        
        return slope_roll, slope_pitch

    def _estimate_normal(self, feet: list[np.ndarray]) -> np.ndarray:
        """Estimate surface normal from stance feet."""
        if len(feet) < 3:
            return np.array([0, 0, 1])
        
        pts = np.array(feet[:3])
        
        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[0]
        
        normal = np.cross(v1, v2)
        
        # Ensure upward-facing normal
        if normal[2] < 0:
            normal = -normal
        
        norm = np.linalg.norm(normal)
        
        if norm < 1e-6:
            return np.array([0, 0, 1])
        
        return normal / norm

    def _default_estimate(self) -> dict:
        """Return default estimate when not enough stance feet."""
        return {
            'height_per_foot': np.zeros(4),
            'slope_roll': 0.0,
            'slope_pitch': 0.0,
            'normal': np.array([0, 0, 1]),
            'avg_height': 0.0,
        }

    def get_slope_state(self, estimate: dict) -> np.ndarray:
        """
        Convert terrain estimate to MPC state augmentation.
        
        Returns
        -------
        slope_state : np.ndarray, shape (4,)
            [slope_roll, slope_pitch, avg_height, 0] for MPC state
        """
        return np.array([
            estimate['slope_roll'],
            estimate['slope_pitch'],
            estimate['avg_height'],
            0.0,
        ])
