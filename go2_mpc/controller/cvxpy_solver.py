import numpy as np
import scipy.sparse as sp
import clarabel


class ClarabelSolver:
    """
    Direct Clarabel QP solver — no CVXPY overhead.

    Solves:  min 0.5 z^T P z + q^T z
             s.t. A_eq z = b_eq
                  A_ineq z <= b_ineq

    Clarabel's native form:
        min 0.5 x'Px + q'x
        s.t. Ax + s = b,  s in K
    where K = ZeroCone (equalities) x NonnegativeCone (inequalities).
    """

    def __init__(self):
        self._settings = clarabel.DefaultSettings()
        self._settings.verbose = False

    def solve(self, H, f, A_eq=None, b_eq=None, A_ineq=None, b_ineq=None):
        n = H.shape[0]

        # Cost: upper-triangular of H as sparse CSC
        P = sp.csc_matrix(sp.triu(H))
        q = f.astype(np.float64)

        # Build combined constraint matrix and RHS
        A_blocks = []
        b_blocks = []
        cones = []

        if A_eq is not None and b_eq is not None:
            m_eq = A_eq.shape[0]
            A_blocks.append(sp.csc_matrix(A_eq))
            b_blocks.append(b_eq.astype(np.float64))
            cones.append(clarabel.ZeroConeT(m_eq))

        if A_ineq is not None and b_ineq is not None:
            m_ineq = A_ineq.shape[0]
            A_blocks.append(sp.csc_matrix(A_ineq))
            b_blocks.append(b_ineq.astype(np.float64))
            cones.append(clarabel.NonnegativeConeT(m_ineq))

        if len(A_blocks) == 0:
            # Unconstrained — shouldn't happen in MPC, but handle gracefully
            A_combined = sp.csc_matrix((0, n))
            b_combined = np.array([], dtype=np.float64)
        else:
            A_combined = sp.vstack(A_blocks, format="csc")
            b_combined = np.concatenate(b_blocks)

        solver = clarabel.DefaultSolver(P, q, A_combined, b_combined, cones, self._settings)
        sol = solver.solve()

        if sol.status not in (clarabel.SolverStatus.Solved, clarabel.SolverStatus.AlmostSolved):
            return None

        return np.array(sol.x)


class CVXPYSolver:
    """
    CVXPY-based QP solver (fallback if direct Clarabel has issues).
    Rebuilds the problem each call — slower but more robust.
    """

    def __init__(self, solver_name="CLARABEL"):
        import cvxpy as cp
        self._cp = cp
        self.solver_name = solver_name

    def solve(self, H, f, A_eq=None, b_eq=None, A_ineq=None, b_ineq=None):
        cp = self._cp
        n = H.shape[0]
        z = cp.Variable(n)

        cost = 0.5 * cp.quad_form(z, H) + f.T @ z
        constraints = []

        if A_eq is not None and b_eq is not None:
            constraints.append(A_eq @ z == b_eq)
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
