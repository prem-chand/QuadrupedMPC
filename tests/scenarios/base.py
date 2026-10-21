"""Base scenario class for walking tests"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import mujoco

from ..test_config import TestConfig, TestScenario
from ..metrics.collector import MetricsCollector
from ..metrics.evaluator import MetricsEvaluator, TestResult, TestStatus


class BaseScenario(ABC):
    """Base class for test scenarios"""
    
    def __init__(self, scenario: TestScenario, config: TestConfig):
        self.scenario = scenario
        self.config = config
        self.collector = MetricsCollector(scenario.name)
        self.evaluator = MetricsEvaluator(config.thresholds)
        
    @abstractmethod
    def setup(self, model: mujoco.MjModel, data: mujoco.MjData) -> bool:
        """Setup the scenario (terrain, initial state, etc.)"""
        pass
    
    @abstractmethod
    def get_command(self, t: float) -> Tuple[float, float, float]:
        """Get command (vx, vy, yaw_rate) at time t"""
        pass
    
    @abstractmethod
    def teardown(self, model: mujoco.MjModel, data: mujoco.MjData):
        """Cleanup after test"""
        pass
    
    def should_start(self, t: float) -> bool:
        """When to start collecting metrics"""
        return t >= self.config.warmup_time
    
    def should_stop(self, t: float) -> bool:
        """When to stop the test"""
        return t >= self.scenario.duration


class FlatGroundScenario(BaseScenario):
    """Flat ground walking scenario"""
    
    def __init__(self, scenario: TestScenario, config: TestConfig):
        super().__init__(scenario, config)
        self.start_time = 0.0
        
    def setup(self, model: mujoco.MjModel, data: mujoco.MjData) -> bool:
        """Setup flat ground - no modification needed"""
        self.start_time = 0.0
        return True
    
    def get_command(self, t: float) -> Tuple[float, float, float]:
        """Get command based on scenario"""
        # Get command from scenario definition
        cmd_gen = self.scenario.command_generator
        if cmd_gen is None:
            return (0.0, 0.0, 0.0)
        
        cmd = cmd_gen()
        if cmd is None:
            return (0.0, 0.0, 0.0)
        
        return cmd
    
    def teardown(self, model: mujoco.MjModel, data: mujoco.MjData):
        """Cleanup - reset to flat ground"""
        pass


def run_scenario(
    scenario: TestScenario,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot,
    estimator,
    controller,
    controller_state,
    buffers,
    config: TestConfig,
    viewer=None,
    verbose: bool = True,
) -> TestResult:
    """
    Run a single test scenario.
    
    Returns TestResult with pass/fail status and metrics.
    """
    
    # Create scenario instance
    scenario_obj = FlatGroundScenario(scenario, config)
    
    # Setup
    if verbose:
        print(f"\n{'='*50}")
        print(f"Running: {scenario.name}")
        print(f"Description: {scenario.description}")
        print(f"Duration: {scenario.duration}s")
        print(f"Expected: {scenario.expected_outcome}")
        print(f"{'='*50}")
    
    setup_success = scenario_obj.setup(model, data)
    if not setup_success:
        return TestResult(
            test_name=scenario.name,
            status=TestStatus.ERROR,
            duration=0,
            metrics={},
            error_message="Setup failed",
        )
    
    # Initialize collector
    collector = MetricsCollector(scenario.name)
    collector.start(robot.get_time())
    
    # Reset controller state
    controller_state.step_counter = 0
    controller_state.mpc_counter = 0
    controller_state.gait_phase_time = 0.0
    
    running = True
    fall_detected = False
    
    try:
        while running:
            # Step simulation
            robot.step()
            sim_time = robot.get_time()
            
            # Get command from scenario
            cmd = scenario_obj.get_command(sim_time)
            
            # Run controller
            if cmd is not None:
                vx, vy, yaw_rate = cmd
                from go2_mpc.core.command import Command
                command = Command(
                    v_cmd_global=np.array([vx, vy, 0.0]),
                    yaw_rate=yaw_rate,
                    default_height=0.32,
                )
                
                # Compute control
                state, foot_pos_rel = estimator.estimate()
                tau, diag = controller.compute(
                    state=state,
                    foot_pos_rel=foot_pos_rel,
                    command=command,
                    controller_state=controller_state,
                    buffers=buffers,
                    robot_interface=robot,
                )
                
                # Collect metrics during test phase
                if collector.collecting and scenario_obj.should_start(sim_time):
                    collector.sample(
                        sim_time=sim_time,
                        robot=robot,
                        state=state,
                        command=command,
                        mpc_forces=diag['current_forces'],
                        contact_schedule=diag['contact_schedule'][0],
                        torques=tau,
                    )
                
                # Apply torques
                robot.set_torques(tau)
            else:
                # Stand mode - no command
                tau = np.zeros(12)
                robot.set_torques(tau)
            
            # Check for fall
            pos, _ = robot.get_base_pose()
            if pos[2] < config.thresholds.fall_height_threshold:
                fall_detected = True
                if verbose:
                    print(f"\n[FALL DETECTED] at t={sim_time:.3f}s")
                running = False
            
            # Check stop condition
            if scenario_obj.should_stop(sim_time):
                running = False
            
            # Sync viewer
            if viewer is not None:
                viewer.sync()
                
    except Exception as e:
        return TestResult(
            test_name=scenario.name,
            status=TestStatus.ERROR,
            duration=robot.get_time(),
            metrics={},
            error_message=str(e),
        )
    
    finally:
        # Teardown
        scenario_obj.teardown(model, data)
        collector.stop()
    
    # Get metrics and evaluate
    metrics = collector.get_summary()
    metrics["fall_detected"] = fall_detected
    
    # Evaluate
    evaluator = MetricsEvaluator(config.thresholds)
    result = evaluator.evaluate(metrics, scenario)
    
    if verbose:
        print(f"\n{result.get_summary()}")
    
    return result
