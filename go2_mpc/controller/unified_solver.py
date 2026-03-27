"""
Unified QP solver backend using qpsolvers library.

Wraps multiple QP solvers with a unified API for easy comparison:
- CLARABEL (current, warm-started, CVXPY)
- OSQP (fast, widely used in robotics)
- qpOASES (real-time, embedded)
- ProxQP (fast CPU solver)
- quadprog (simple baseline)

All solvers are tested via benchmarks to find the fastest for our MPC problem.
"""

import numpy as np
from qpsolvers import solve_qp
from typing import Optional


class UnifiedSolver:
    """
    Unified QP solver backend for multiple solvers.
    
    Supports:
    - clarabel (current CVXPY backend)
    - osqp (fast, warm-started)
    - qpoases (real-time)
    - proxqp (fast CPU)
    - quadprog (simple baseline)
    
    Parameters
    ----------
    solver_name : str
        Name of solver to use ('clarabel', 'osqp', 'qpoases', 'proxqp', 'quadprog')
    verbose : bool
        Enable verbose output
    """
    
    SOLVERS = ['clarabel', 'osqp', 'qpoases', 'proxqp', 'quadprog']
    
    def __init__(self, solver_name: str = 'clarabel', verbose: bool = False):
        if solver_name not in self.SOLVERS:
            raise ValueError(f"Unknown solver: {solver_name}. Choose from {self.SOLVERS}")
        
        self.solver_name = solver_name
        self.verbose = verbose
        
        self._solver_kwargs = self._get_solver_defaults(solver_name)

    def _get_solver_defaults(self, solver_name: str) -> dict:
        """Get default solver-specific parameters."""
        defaults = {
            'clarabel': {'warm_start': True},
            'osqp': {'warm_start': True, 'eps_abs': 1e-6, 'eps_rel': 1e-6},
            'qpoases': {'nWSR': 100},
            'proxqp': {'warm_start': True, 'eps_abs': 1e-6},
            'quadprog': {},
        }
        return defaults.get(solver_name, {})

    def solve(
        self,
        H: np.ndarray,
        f: np.ndarray,
        A_eq: Optional[np.ndarray] = None,
        b_eq: Optional[np.ndarray] = None,
        A_ineq: Optional[np.ndarray] = None,
        b_ineq: Optional[np.ndarray] = None,
        lb: Optional[np.ndarray] = None,
        ub: Optional[np.ndarray] = None,
        initvals: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        Solve QP: min 0.5 x^T H x + f^T x
        
        Subject to:
            A_eq x = b_eq
            A_ineq x <= b_ineq
            lb <= x <= ub
        
        Parameters
        ----------
        H : np.ndarray, shape (n, n)
            Symmetric cost matrix
        f : np.ndarray, shape (n,)
            Linear cost vector
        A_eq : np.ndarray, shape (m_eq, n)
            Equality constraint matrix
        b_eq : np.ndarray, shape (m_eq,)
            Equality constraint vector
        A_ineq : np.ndarray, shape (m_ineq, n)
            Inequality constraint matrix
        b_ineq : np.ndarray, shape (m_ineq,)
            Inequality constraint vector
        lb : np.ndarray, shape (n,)
            Lower bounds
        ub : np.ndarray, shape (n,)
            Upper bounds
        initvals : np.ndarray, shape (n,)
            Initial values for warm start
            
        Returns
        -------
        solution : np.ndarray, shape (n,)
            Optimal solution, or zeros on failure
        """
        try:
            kwargs = {**self._solver_kwargs}
            if initvals is not None:
                kwargs['initvals'] = initvals
            
            x = solve_qp(
                H, f, 
                G=A_ineq, h=b_ineq,
                A=A_eq, b=b_eq,
                lb=lb, ub=ub,
                solver=self.solver_name,
                verbose=self.verbose,
                **kwargs
            )
            
            if x is None:
                return np.zeros(H.shape[0], dtype=np.float64)
            
            return np.array(x, dtype=np.float64)
            
        except Exception as e:
            print(f"Warning: {self.solver_name} failed: {e}")
            return np.zeros(H.shape[0], dtype=np.float64)

    def __repr__(self):
        return f"UnifiedSolver({self.solver_name})"


class MultiSolverComparison:
    """
    Compare multiple QP solvers on the same problem.
    
    Parameters
    ----------
    solver_names : list of str
        Solvers to compare
    num_trials : int
        Number of trials per solver
    """
    
    def __init__(self, solver_names: list[str] = None, num_trials: int = 100):
        self.solver_names = solver_names or ['clarabel', 'osqp', 'qpoases', 'proxqp', 'quadprog']
        self.num_trials = num_trials
        self.results = {}

    def run_comparison(self, H, f, A_eq=None, b_eq=None, A_ineq=None, b_ineq=None) -> dict:
        """
        Run comparison across all solvers.
        
        Parameters
        ----------
        H : np.ndarray, shape (n, n)
            Symmetric cost matrix
        f : np.ndarray, shape (n,)
            Linear cost vector
        A_eq : np.ndarray, shape (m_eq, n)
            Equality constraint matrix
        b_eq : np.ndarray, shape (m_eq,)
            Equality constraint vector
        A_ineq : np.ndarray, shape (m_ineq, n)
            Inequality constraint matrix
        b_ineq : np.ndarray, shape (m_ineq,)
            Inequality constraint vector
            
        Returns
        -------
        results : dict
            Dictionary of solver results with timing and solution
        """
        import time
        
        for solver_name in self.solver_names:
            try:
                solver = UnifiedSolver(solver_name)
                
                times = []
                solutions = []
                
                for _ in range(self.num_trials):
                    t0 = time.perf_counter()
                    x = solver.solve(H, f, A_eq, b_eq, A_ineq, b_ineq)
                    elapsed = (time.perf_counter() - t0) * 1000
                    
                    times.append(elapsed)
                    solutions.append(x)
                
                times_arr = np.array(times)
                
                self.results[solver_name] = {
                    'mean_ms': np.mean(times_arr),
                    'std_ms': np.std(times_arr),
                    'median_ms': np.median(times_arr),
                    'p95_ms': np.percentile(times_arr, 95),
                    'p99_ms': np.percentile(times_arr, 99),
                    'max_ms': np.max(times_arr),
                    'solution': solutions[-1],
                    'success': not np.all(solutions[-1] == 0),
                }
                
            except Exception as e:
                self.results[solver_name] = {'error': str(e), 'success': False}
        
        return self.results

    def print_results(self):
        """Print comparison results in a formatted table."""
        print("\n" + "="*70)
        print("QP Solver Benchmark Results")
        print("="*70)
        print(f"{'Solver':<12} {'Mean':<10} {'Median':<10} {'P95':<10} {'P99':<10} {'Status':<8}")
        print("-"*70)
        
        for solver_name, result in self.results.items():
            if 'error' in result:
                print(f"{solver_name:<12} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10} {'FAIL':<8}")
                continue
            
            mean = f"{result['mean_ms']:.3f}ms"
            median = f"{result['median_ms']:.3f}ms"
            p95 = f"{result['p95_ms']:.3f}ms"
            p99 = f"{result['p99_ms']:.3f}ms"
            status = 'OK' if result['success'] else 'FAIL'
            
            print(f"{solver_name:<12} {mean:<10} {median:<10} {p95:<10} {p99:<10} {status:<8}")
        
        print("="*70)

    def get_fastest(self) -> str:
        """Return the fastest solver name."""
        fastest = None
        fastest_time = float('inf')
        
        for solver_name, result in self.results.items():
            if 'error' in result or not result.get('success', False):
                continue
            if result['mean_ms'] < fastest_time:
                fastest_time = result['mean_ms']
                fastest = solver_name
        
        return fastest
