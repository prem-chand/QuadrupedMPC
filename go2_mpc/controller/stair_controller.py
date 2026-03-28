"""
Stair climbing controller modifications.

Adapts gait and foot placement for stair climbing.
"""

import numpy as np


class StairController:
    """
    Controller for stair climbing.
    
    Adapts:
    - Foot swing height (higher for stairs)
    - Swing duration (slower for stairs)
    - Foot placement (adjust for step depth)
    """

    def __init__(
        self,
        step_height: float = 0.05,  # 5cm stairs
        step_depth: float = 0.25,     # 25cm depth
        swing_height: float = 0.15,    # Higher swing for stairs
        swing_duration: float = 0.3,  # Slower swing
    ):
        self.step_height = step_height
        self.step_depth = step_depth
        self.swing_height = swing_height
        self.swing_duration = swing_duration
        
        self._in_stair_region = False

    def detect_stairs(self, foot_positions: list[np.ndarray], base_x: float) -> bool:
        """Detect if robot is approaching stairs."""
        stair_start = 2.0  # x position where stairs start in scene
        
        if base_x > stair_start - 0.5 and base_x < stair_start + 3.0:
            return True
        return False

    def get_swing_height(self, in_stairs: bool) -> float:
        """Get swing height based on terrain."""
        if in_stairs:
            return self.swing_height
        return 0.08  # Normal swing height

    def get_swing_duration(self, in_stairs: bool) -> float:
        """Get swing duration based on terrain."""
        if in_stairs:
            return self.swing_duration
        return 0.2  # Normal swing duration

    def compute_stair_target(
        self,
        foot_idx: int,
        current_pos: np.ndarray,
        target_base: np.ndarray,
        in_stairs: bool,
    ) -> np.ndarray:
        """
        Compute foot target adjusted for stairs.
        
        Parameters
        ----------
        foot_idx : int
            Foot index (0=FL, 1=FR, 2=RL, 3=RR)
        current_pos : np.ndarray
            Current foot position
        target_base : np.ndarray
            Target base position
        in_stairs : bool
            Whether robot is on stairs
            
        Returns
        -------
        target : np.ndarray
            Adjusted foot target position
        """
        if not in_stairs:
            return current_pos  # Use normal foot placement
        
        # Adjust for stairs
        target = current_pos.copy()
        
        # Lift foot higher for stair
        target[2] = self.step_height + 0.05
        
        return target
