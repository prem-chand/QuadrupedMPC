import cvxpy as cp
import numpy as np


class CVXPYSolver:
    def __init__(self, solver_name="CLARABEL"):
        self.solver_name = solver_name

    def solve(self, H, f, A_eq=None, b_eq=None, A_ineq=None, b_ineq=None):
        """
        Solves:
            min 0.5 zᵀ H z + fᵀ z
            s.t. A_eq z = b_eq
                 A_ineq z <= b_ineq

        Returns:
            z (np.ndarray) or None if failure
        """

        n = H.shape[0]

        z = cp.Variable(n)

        # Cost
        cost = 0.5 * cp.quad_form(z, H) + f.T @ z
        constraints = []

        # Equality constraints
        if A_eq is not None and b_eq is not None:
            constraints.append(A_eq @ z == b_eq)

        # Inequality constraints
        if A_ineq is not None and b_ineq is not None:
            constraints.append(A_ineq @ z <= b_ineq)

        problem = cp.Problem(cp.Minimize(cost), constraints)

        try:
            problem.solve(
                solver=getattr(cp, self.solver_name),
                warm_start=False,
                verbose=False,
            )
        except cp.SolverError:
            return None

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            return None

        return z.value
