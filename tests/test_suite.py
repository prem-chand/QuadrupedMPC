#!/usr/bin/env python3
"""
QuadrupedMPC Walking Quality Test Suite

Interactive test runner for evaluating MPC-WBC controller performance.
"""

import sys
import os
import argparse
from typing import List, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mujoco
import mujoco.viewer

# Import MPC components
from go2_mpc.config.config import default_config
from go2_mpc.utils.simulation_utils import setup_simulation, setup_controller
from go2_mpc.core.command import Command

from tests.test_config import TestConfig, TestCategory
from tests.metrics.evaluator import TestResult, TestStatus
from tests.scenarios.base import run_scenario


class TestSuite:
    """Interactive test suite runner"""
    
    def __init__(self):
        self.config = TestConfig()
        self.results: List[TestResult] = []
        self.cfg = default_config()
        self.model, self.data, self.robot, self.estimator = setup_simulation(self.cfg)
        self.controller, self.controller_state, self.buffers = setup_controller(self.cfg, self.robot)
    
    def run_test(self, scenario_name: str, verbose: bool = True) -> TestResult:
        """Run a single test"""
        
        scenario = self.config.get_scenario(scenario_name)
        if scenario is None:
            print(f"Error: Unknown scenario '{scenario_name}'")
            return None
        
        # Reset simulation and robot
        mujoco.mj_resetData(self.model, self.data)
        self.model, self.data, self.robot, self.estimator = setup_simulation(self.cfg)
        
        # Reset controller state
        self.controller, self.controller_state, self.buffers = setup_controller(self.cfg, self.robot)
        
        # Run scenario
        result = run_scenario(
            scenario=scenario,
            model=self.model,
            data=self.data,
            robot=self.robot,
            estimator=self.estimator,
            controller=self.controller,
            controller_state=self.controller_state,
            buffers=self.buffers,
            config=self.config,
            viewer=None,
            verbose=verbose,
        )
        
        self.results.append(result)
        return result
    
    def run_category(self, category: TestCategory, verbose: bool = True) -> List[TestResult]:
        """Run all tests in a category"""
        
        scenarios = self.config.get_scenarios(category)
        results = []
        
        print(f"\n{'='*60}")
        print(f"Running category: {category.value}")
        print(f"Tests: {len(scenarios)}")
        print(f"{'='*60}")
        
        for scenario in scenarios:
            result = self.run_test(scenario.name, verbose=verbose)
            results.append(result)
            
            # Pause between tests
            if verbose:
                input("\nPress Enter to continue to next test...")
        
        return results
    
    def run_all(self, verbose: bool = True) -> List[TestResult]:
        """Run all tests"""
        return self.run_category(TestCategory.ALL, verbose=verbose)
    
    def print_summary(self):
        """Print summary of all results"""
        
        print(f"\n{'='*60}")
        print("TEST SUITE SUMMARY")
        print(f"{'='*60}")
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r.status == TestStatus.PASS)
        failed = sum(1 for r in self.results if r.status == TestStatus.FAIL)
        errors = sum(1 for r in self.results if r.status == TestStatus.ERROR)
        
        print(f"Total:  {total}")
        print(f"Passed: {passed} ({100*passed/total:.1f}%)" if total > 0 else "Passed: 0")
        print(f"Failed: {failed}")
        print(f"Errors: {errors}")
        
        print(f"\n{'='*60}")
        print("Individual Results:")
        print(f"{'='*60}")
        
        for result in self.results:
            status_symbol = {
                TestStatus.PASS: "✓",
                TestStatus.FAIL: "✗",
                TestStatus.ERROR: "!",
                TestStatus.SKIPPED: "-",
            }.get(result.status, "?")
            
            print(f"  {status_symbol} {result.test_name}: {result.status.value}")
            if result.error_message:
                print(f"      Error: {result.error_message}")


def print_menu():
    """Print main menu"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║         QuadrupedMPC Walking Quality Test Suite               ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. Run Stand Test                                           ║
║  2. Run Flat Ground Tests                                    ║
║  3. Run Slope Tests                                          ║
║  4. Run Stair Tests                                          ║
║  5. Run Perturbation Tests                                   ║
║  6. Run Gait Transition Tests                                 ║
║  7. Run All Tests                                            ║
║                                                              ║
║  8. List Available Tests                                     ║
║  9. Run Specific Test                                        ║
║                                                              ║
║  V. View Last Results Summary                                ║
║  Q. Quit                                                     ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


def list_tests(suite: TestSuite):
    """List all available tests"""
    print("\nAvailable Tests:")
    print("-" * 50)
    
    scenarios = suite.config.get_scenarios()
    for i, scenario in enumerate(scenarios, 1):
        print(f"  {i:2d}. {scenario.name:30s} [{scenario.category.value}] ({scenario.difficulty}*)")
        print(f"       {scenario.description}")


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description="QuadrupedMPC Test Suite")
    parser.add_argument("--test", type=str, help="Run specific test")
    parser.add_argument("--category", type=str, choices=["flat", "slopes", "stairs", "perturbations", "gait", "all"],
                       help="Run tests in category")
    parser.add_argument("--batch", action="store_true", help="Run without prompts")
    args = parser.parse_args()
    
    # Create test suite
    print("Initializing test suite...")
    suite = TestSuite()
    print("Done.\n")
    
    # Handle command line arguments
    if args.test:
        result = suite.run_test(args.test, verbose=not args.batch)
        suite.print_summary()
        return
    
    if args.category:
        category_map = {
            "flat": TestCategory.FLAT_GROUND,
            "slopes": TestCategory.SLOPES,
            "stairs": TestCategory.STAIRS,
            "perturbations": TestCategory.PERTURBATIONS,
            "gait": TestCategory.GAIT_TRANSITIONS,
            "all": TestCategory.ALL,
        }
        results = suite.run_category(category_map[args.category], verbose=not args.batch)
        suite.print_summary()
        return
    
    # Interactive mode
    while True:
        print_menu()
        choice = input("Select option: ").strip().upper()
        
        if choice == "Q":
            print("Goodbye!")
            break
        
        elif choice == "1":
            suite.run_test("stand", verbose=True)
            
        elif choice == "2":
            suite.run_category(TestCategory.FLAT_GROUND, verbose=True)
            
        elif choice == "3":
            suite.run_category(TestCategory.SLOPES, verbose=True)
            
        elif choice == "4":
            suite.run_category(TestCategory.STAIRS, verbose=True)
            
        elif choice == "5":
            suite.run_category(TestCategory.PERTURBATIONS, verbose=True)
            
        elif choice == "6":
            suite.run_category(TestCategory.GAIT_TRANSITIONS, verbose=True)
            
        elif choice == "7":
            suite.run_all(verbose=True)
            
        elif choice == "8":
            list_tests(suite)
            
        elif choice == "9":
            name = input("Enter test name: ").strip()
            suite.run_test(name, verbose=True)
            
        elif choice == "V":
            suite.print_summary()
            
        else:
            print("Invalid option. Please try again.")


if __name__ == "__main__":
    main()
