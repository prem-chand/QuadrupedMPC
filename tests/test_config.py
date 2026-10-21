"""Test configuration and thresholds for walking quality tests"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable
import numpy as np


class TestCategory(Enum):
    FLAT_GROUND = "flat"
    SLOPES = "slopes"
    STAIRS = "stairs"
    PERTURBATIONS = "perturbations"
    GAIT_TRANSITIONS = "gait"
    ALL = "all"


@dataclass
class MetricThresholds:
    """Pass/fail thresholds for test metrics"""
    # Height
    height_min: float = 0.25  # m
    height_max: float = 0.38  # m
    height_std_max: float = 0.05  # m
    
    # Orientation
    roll_max: float = 0.78  # rad (~45 deg)
    pitch_max: float = 0.78  # rad (~45 deg)
    yaw_rate_max: float = 2.0  # rad/s
    
    # Velocity
    velocity_error_max: float = 0.2  # m/s
    
    # Torque
    torque_max: float = 45.0  # Nm
    
    # Safety
    fall_height_threshold: float = 0.15  # m - below this = fallen
    contact_violation_max: int = 0
    
    # Recovery
    recovery_time_max: float = 1.0  # seconds
    
    # Oscillation
    oscillation_threshold: float = 0.3  # rad - for pitch/roll oscillation detection


@dataclass
class TestScenario:
    """Definition of a single test scenario"""
    name: str
    category: TestCategory
    description: str
    duration: float  # seconds
    command_generator: Callable  # Function that returns Command
    initial_state: str = "stand"  # "stand", "trot", "bound"
    setup_fn: Callable = None  # Optional setup (e.g., create slope)
    teardown_fn: Callable = None  # Optional cleanup
    expected_outcome: str = ""  # Description of expected behavior
    difficulty: int = 1  # 1-5 scale


class TestConfig:
    """Global test configuration"""
    
    def __init__(self):
        self.thresholds = MetricThresholds()
        self.scenarios = self._build_scenario_library()
        self.default_duration = 5.0  # seconds
        self.warmup_time = 1.0  # seconds before metrics collection
        self.data_collection_rate = 100  # Hz
        
    def get_scenarios(self, category: TestCategory = None) -> list:
        """Get scenarios by category"""
        if category is None or category == TestCategory.ALL:
            return list(self.scenarios.values())
        return [s for s in self.scenarios.values() if s.category == category]
    
    def get_scenario(self, name: str) -> TestScenario:
        """Get scenario by name"""
        return self.scenarios.get(name)
    
    def _build_scenario_library(self) -> dict:
        """Build the library of test scenarios"""
        
        scenarios = {}
        
        # ===== FLAT GROUND TESTS =====
        flat = TestCategory.FLAT_GROUND
        
        scenarios["stand"] = TestScenario(
            name="stand",
            category=flat,
            description="Stand in place",
            duration=5.0,
            command_generator=lambda: None,  # No command = stand
            expected_outcome="Maintain height 0.30-0.35m, roll/pitch < 5deg",
            difficulty=1,
        )
        
        scenarios["walk_forward_0.3"] = TestScenario(
            name="walk_forward_0.3",
            category=flat,
            description="Walk forward at 0.3 m/s",
            duration=5.0,
            command_generator=lambda: self._cmd(vx=0.3),
            expected_outcome="Stable forward motion, height maintained",
            difficulty=1,
        )
        
        scenarios["walk_forward_0.5"] = TestScenario(
            name="walk_forward_0.5",
            category=flat,
            description="Walk forward at 0.5 m/s",
            duration=5.0,
            command_generator=lambda: self._cmd(vx=0.5),
            expected_outcome="Stable forward, slight oscillation OK",
            difficulty=2,
        )
        
        scenarios["walk_forward_0.8"] = TestScenario(
            name="walk_forward_0.8",
            category=flat,
            description="Walk forward at 0.8 m/s",
            duration=5.0,
            command_generator=lambda: self._cmd(vx=0.8),
            expected_outcome="Stable forward, may have oscillations",
            difficulty=3,
        )
        
        scenarios["walk_backward"] = TestScenario(
            name="walk_backward",
            category=flat,
            description="Walk backward",
            duration=3.0,
            command_generator=lambda: self._cmd(vx=-0.3),
            expected_outcome="Stable backward motion",
            difficulty=2,
        )
        
        scenarios["strafe_left"] = TestScenario(
            name="strafe_left",
            category=flat,
            description="Strafe left",
            duration=3.0,
            command_generator=lambda: self._cmd(vy=0.3),
            expected_outcome="Stable lateral motion",
            difficulty=2,
        )
        
        scenarios["strafe_right"] = TestScenario(
            name="strafe_right",
            category=flat,
            description="Strafe right",
            duration=3.0,
            command_generator=lambda: self._cmd(vy=-0.3),
            expected_outcome="Stable lateral motion",
            difficulty=2,
        )
        
        scenarios["turn_left_360"] = TestScenario(
            name="turn_left_360",
            category=flat,
            description="Turn left 360 degrees",
            duration=4.0,
            command_generator=lambda: self._cmd(yaw_rate=1.0),
            expected_outcome="Smooth rotation, no fall",
            difficulty=2,
        )
        
        scenarios["turn_right_360"] = TestScenario(
            name="turn_right_360",
            category=flat,
            description="Turn right 360 degrees",
            duration=4.0,
            command_generator=lambda: self._cmd(yaw_rate=-1.0),
            expected_outcome="Smooth rotation, no fall",
            difficulty=2,
        )
        
        # ===== SLOPE TESTS =====
        slopes = TestCategory.SLOPES
        
        scenarios["slope_up_10"] = TestScenario(
            name="slope_up_10",
            category=slopes,
            description="Walk up 10 degree slope",
            duration=5.0,
            command_generator=lambda: self._cmd(vx=0.3),
            setup_fn=self._setup_slope_10,
            expected_outcome="Ascend without slipping",
            difficulty=3,
        )
        
        scenarios["slope_down_10"] = TestScenario(
            name="slope_down_10",
            category=slopes,
            description="Walk down 10 degree slope",
            duration=5.0,
            command_generator=lambda: self._cmd(vx=-0.3),
            setup_fn=self._setup_slope_10,
            expected_outcome="Descend without slipping",
            difficulty=3,
        )
        
        scenarios["slope_side_10"] = TestScenario(
            name="slope_side_10",
            category=slopes,
            description="Walk on 10 degree side slope",
            duration=5.0,
            command_generator=lambda: self._cmd(vx=0.3),
            setup_fn=self._setup_side_slope_10,
            expected_outcome="Maintain balance",
            difficulty=4,
        )
        
        # ===== STAIR TESTS =====
        stairs = TestCategory.STAIRS
        
        scenarios["stairs_up_small"] = TestScenario(
            name="stairs_up_small",
            category=stairs,
            description="Climb small stairs (5cm)",
            duration=8.0,
            command_generator=lambda: self._cmd(vx=0.2),
            setup_fn=self._setup_stairs_small,
            expected_outcome="Climb without falling",
            difficulty=3,
        )
        
        scenarios["stairs_down_small"] = TestScenario(
            name="stairs_down_small",
            category=stairs,
            description="Descend small stairs (5cm)",
            duration=8.0,
            command_generator=lambda: self._cmd(vx=-0.2),
            setup_fn=self._setup_stairs_small,
            expected_outcome="Descend without falling",
            difficulty=3,
        )
        
        # ===== PERTURBATION TESTS =====
        perturbations = TestCategory.PERTURBATIONS
        
        scenarios["push_forward"] = TestScenario(
            name="push_forward",
            category=perturbations,
            description="Forward push recovery",
            duration=3.0,
            command_generator=lambda: self._cmd(vx=0.0),
            setup_fn=lambda: ("perturbation", "forward", 5.0),
            expected_outcome="Recover within 0.5s",
            difficulty=2,
        )
        
        scenarios["push_lateral"] = TestScenario(
            name="push_lateral",
            category=perturbations,
            description="Lateral push recovery",
            duration=3.0,
            command_generator=lambda: self._cmd(vx=0.0),
            setup_fn=lambda: ("perturbation", "lateral", 3.0),
            expected_outcome="Recover within 0.5s",
            difficulty=2,
        )
        
        scenarios["push_backward"] = TestScenario(
            name="push_backward",
            category=perturbations,
            description="Backward push recovery",
            duration=3.0,
            command_generator=lambda: self._cmd(vx=0.0),
            setup_fn=lambda: ("perturbation", "backward", 5.0),
            expected_outcome="Recover within 0.5s",
            difficulty=2,
        )
        
        # ===== GAIT TRANSITION TESTS =====
        gait = TestCategory.GAIT_TRANSITIONS
        
        scenarios["stand_to_trot"] = TestScenario(
            name="stand_to_trot",
            category=gait,
            description="Transition from stand to trot",
            duration=3.0,
            command_generator=lambda: self._cmd(vx=0.3),
            initial_state="stand",
            expected_outcome="Smooth transition",
            difficulty=2,
        )
        
        scenarios["trot_to_stand"] = TestScenario(
            name="trot_to_stand",
            category=gait,
            description="Transition from trot to stand",
            duration=3.0,
            command_generator=lambda: self._cmd(vx=0.0),
            initial_state="trot",
            expected_outcome="Smooth stop",
            difficulty=2,
        )
        
        scenarios["trot_to_bound"] = TestScenario(
            name="trot_to_bound",
            category=gait,
            description="Transition from trot to bound",
            duration=3.0,
            command_generator=lambda: self._cmd(vx=0.4),
            initial_state="trot",
            expected_outcome="Gait switch OK",
            difficulty=3,
        )
        
        return scenarios
    
    @staticmethod
    def _cmd(vx=0.0, vy=0.0, yaw_rate=0.0):
        """Create a command tuple"""
        return (vx, vy, yaw_rate)
    
    @staticmethod
    def _setup_slope_10():
        """Setup 10 degree slope"""
        return ("terrain", "slope", 10.0)
    
    @staticmethod
    def _setup_side_slope_10():
        """Setup 10 degree side slope"""
        return ("terrain", "side_slope", 10.0)
    
    @staticmethod
    def _setup_stairs_small():
        """Setup small stairs"""
        return ("terrain", "stairs", 0.05)  # 5cm steps


# Global config instance
config = TestConfig()
