"""Metrics evaluator for pass/fail determination"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

from ..test_config import MetricThresholds, TestScenario


class TestStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"


@dataclass
class MetricResult:
    """Result of evaluating a single metric"""
    name: str
    value: float
    threshold: float
    passed: bool
    severity: str = "info"  # "info", "warning", "error"


@dataclass
class TestResult:
    """Complete test result"""
    test_name: str
    status: TestStatus
    duration: float
    metrics: dict
    
    # Individual metric results
    metric_results: List[MetricResult] = None
    
    # Summary
    passed_count: int = 0
    failed_count: int = 0
    
    # Error info
    error_message: str = ""
    
    def __post_init__(self):
        if self.metric_results is None:
            self.metric_results = []
    
    def add_metric(self, name: str, value: float, threshold: float, 
                   passed: bool, severity: str = "info"):
        """Add a metric result"""
        self.metric_results.append(MetricResult(
            name=name, value=value, threshold=threshold, 
            passed=passed, severity=severity
        ))
        if passed:
            self.passed_count += 1
        else:
            self.failed_count += 1
    
    def is_passed(self) -> bool:
        """Check if test passed"""
        return self.status == TestStatus.PASS
    
    def get_summary(self) -> str:
        """Get human-readable summary"""
        lines = [
            f"Test: {self.test_name}",
            f"Status: {self.status.value}",
            f"Duration: {self.duration:.2f}s",
            f"Passed: {self.passed_count}, Failed: {self.failed_count}",
        ]
        
        if self.error_message:
            lines.append(f"Error: {self.error_message}")
        
        # Add key metrics
        if "height_mean" in self.metrics:
            lines.append(f"Height: {self.metrics['height_mean']:.3f}m (std: {self.metrics.get('height_std', 0):.3f})")
        if "roll_max" in self.metrics:
            lines.append(f"Roll: {np.degrees(self.metrics['roll_max']):.1f}° max")
        if "pitch_max" in self.metrics:
            lines.append(f"Pitch: {np.degrees(self.metrics['pitch_max']):.1f}° max")
        
        return "\n".join(lines)


class MetricsEvaluator:
    """Evaluates test metrics against thresholds"""
    
    def __init__(self, thresholds: MetricThresholds = None):
        self.thresholds = thresholds or MetricThresholds()
    
    def evaluate(self, metrics: dict, scenario: TestScenario) -> TestResult:
        """Evaluate metrics against thresholds"""
        
        result = TestResult(
            test_name=scenario.name,
            status=TestStatus.PASS,
            duration=metrics.get("duration", 0),
            metrics=metrics,
        )
        
        # Check for errors first
        if not metrics or metrics.get("samples", 0) == 0:
            result.status = TestStatus.ERROR
            result.error_message = "No metrics collected"
            return result
        
        # Check for fall
        if metrics.get("fall_detected", False):
            result.status = TestStatus.FAIL
            result.add_metric("fall", 1, 0, False, "error")
            result.error_message = f"Robot fell at t={metrics.get('fall_time', 0):.2f}s"
            return result
        
        # ===== HEIGHT CHECKS =====
        height_mean = metrics.get("height_mean", 0)
        height_std = metrics.get("height_std", 0)
        
        # Height must be within range
        height_passed = (self.thresholds.height_min <= height_mean <= self.thresholds.height_max)
        result.add_metric(
            "height_mean", height_mean,
            f"{self.thresholds.height_min}-{self.thresholds.height_max}",
            height_passed, "error" if not height_passed else "info"
        )
        
        # Height std check
        height_std_passed = height_std <= self.thresholds.height_std_max
        result.add_metric(
            "height_std", height_std, self.thresholds.height_std_max,
            height_std_passed, "warning" if not height_std_passed else "info"
        )
        
        # ===== ORIENTATION CHECKS =====
        roll_max = metrics.get("roll_max", 0)
        pitch_max = metrics.get("pitch_max", 0)
        
        roll_passed = roll_max <= self.thresholds.roll_max
        result.add_metric(
            "roll_max", np.degrees(roll_max), np.degrees(self.thresholds.roll_max),
            roll_passed, "error" if not roll_passed else "info"
        )
        
        pitch_passed = pitch_max <= self.thresholds.pitch_max
        result.add_metric(
            "pitch_max", np.degrees(pitch_max), np.degrees(self.thresholds.pitch_max),
            pitch_passed, "error" if not pitch_passed else "info"
        )
        
        # ===== VELOCITY CHECKS =====
        velocity_error = metrics.get("velocity_error_mean", 0)
        velocity_passed = velocity_error <= self.thresholds.velocity_error_max
        result.add_metric(
            "velocity_error", velocity_error, self.thresholds.velocity_error_max,
            velocity_passed, "warning" if not velocity_passed else "info"
        )
        
        # ===== TORQUE CHECKS =====
        torque_max = metrics.get("torque_max", 0)
        torque_passed = torque_max <= self.thresholds.torque_max
        result.add_metric(
            "torque_max", torque_max, self.thresholds.torque_max,
            torque_passed, "warning" if not torque_passed else "info"
        )
        
        # ===== DETERMINE OVERALL STATUS =====
        # Count errors (not warnings)
        errors = [m for m in result.metric_results if m.severity == "error" and not m.passed]
        
        if errors:
            result.status = TestStatus.FAIL
            result.error_message = f"Failed: {', '.join([m.name for m in errors])}"
        
        return result
    
    def evaluate_batch(self, results: List[TestResult]) -> dict:
        """Evaluate a batch of test results"""
        
        total = len(results)
        passed = sum(1 for r in results if r.status == TestStatus.PASS)
        failed = sum(1 for r in results if r.status == TestStatus.FAIL)
        errors = sum(1 for r in results if r.status == TestStatus.ERROR)
        
        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "pass_rate": passed / total if total > 0 else 0,
            "results": results,
        }
