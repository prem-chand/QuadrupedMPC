from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class QPSolver(ABC):
    """
    Low-level QP solver interface.

    Solves: min 0.5 z^T H z + f^T z
            s.t. A_eq z = b_eq
                 A_ineq z <= b_ineq

    Used by ConvexMPC to solve the QP it constructs from MPC dynamics.
    Implementations: CVXPYSolver, or direct bindings to OSQP, clarabel, etc.
    """

    @abstractmethod
    def solve(self, H, f, A_eq, b_eq, A_ineq, b_ineq) -> np.ndarray | None:
        ...


class MPCSolver(ABC):
    """
    High-level MPC solver interface.

    Takes the full MPC problem description (state, reference, contacts, dynamics)
    and returns optimal forces. This allows solvers like CasADi or batched Torch
    to formulate and solve the problem themselves, bypassing QP matrix construction.

    ConvexMPC implements this interface internally (wrapping a QPSolver).
    A CasADi or Torch solver can implement this directly.
    """

    @abstractmethod
    def solve(
        self,
        state,
        x_ref: np.ndarray,
        contact_schedule: np.ndarray,
        foot_positions_body: list[np.ndarray],
    ) -> np.ndarray:
        """
        Args:
            state: Structured State object with base and joint state.
            x_ref: (12, N+1) reference trajectory.
            contact_schedule: (N, 4) binary contact matrix.
            foot_positions_body: List of 4 foot positions (3,) in body frame.

        Returns:
            (12,) optimal ground reaction forces for the current timestep.
        """
        ...
