"""
QP Solver Benchmark Script.

Runs systematic benchmarking of QP solvers on the actual MPC problem
and generates a results table.
"""

import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

from qpsolvers import solve_qp
from go2_mpc.config.config import default_config
from go2_mpc.controller.unified_solver import MultiSolverComparison


def generate_mpc_problem():
    """Generate MPC-sized QP problem for benchmarking."""
    cfg = default_config()
    n = 12 * (cfg.mpc.horizon + 1) + 12 * cfg.mpc.horizon  # Full MPC variable count
    
    H = np.eye(n) * cfg.mpc.R[0, 0]
    f = np.random.randn(n) * 0.1
    
    A_eq = np.random.randn(12, n) * 0.5
    b_eq = np.random.randn(12) * 0.1
    
    A_ineq = np.random.randn(24, n) * 0.3
    b_ineq = np.ones(24) * cfg.mpc.f_max
    
    return H, f, A_eq, b_eq, A_ineq, b_ineq


def run_benchmark(num_trials: int = 100):
    """Run benchmark across all available solvers."""
    print("\n" + "="*70)
    print("QP Solver Benchmark — MPC Problem")
    print("="*70)
    
    H, f, A_eq, b_eq, A_ineq, b_ineq = generate_mpc_problem()
    
    comparison = MultiSolverComparison(
        solver_names=['clarabel', 'osqp', 'quadprog', 'scs'],
        num_trials=num_trials
    )
    
    results = comparison.run_comparison(H, f, A_eq, b_eq, A_ineq, b_ineq)
    comparison.print_results()
    
    fastest = comparison.get_fastest()
    print(f"\nFastest solver: {fastest}")
    
    return results


def main():
    """Run benchmarks."""
    print("Running QP solver benchmarks...\n")
    
    results = run_benchmark(num_trials=100)
    
    print("\nBenchmark complete. Use results to select solver in config.py")


if __name__ == '__main__':
    main()
