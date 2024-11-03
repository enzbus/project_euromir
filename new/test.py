# Copyright 2024 Enzo Busseti
#
# This file is part of Project Euromir.
#
# Project Euromir is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# Project Euromir is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# Project Euromir. If not, see <https://www.gnu.org/licenses/>.
"""Unit tests of the solver class."""

import time
import warnings
from unittest import TestCase, main

import cvxpy as cp
import numpy as np
import scipy as sp

from solver import Solver

class TestSolverClass(TestCase):
    """Unit tests of the solver class."""

    @staticmethod
    def make_program_from_matrix(matrix, seed=0):
        """Make simple LP program."""
        m,n = matrix.shape
        np.random.seed(seed)
        z = np.random.randn(m)
        y = np.maximum(z, 0.)
        s = y - z
        x = np.random.randn(n)
        b = matrix @ x + s
        c = -matrix.T @ y
        return b, c

    def check_solution_valid(self, matrix, b, c, x, y):
        """Check a LP solution is valid."""
        self.assertGreater(np.min(y), -1e-6)
        s = b - matrix @ x
        self.assertGreater(np.min(s), -1e-6)
        self.assertTrue(np.isclose(c.T @ x + b.T @ y, 0., atol=1e-6, rtol=1e-6))
        self.assertTrue(np.allclose(c, - matrix.T @ y, atol=1e-6, rtol=1e-6))


    @staticmethod
    def solve_program_cvxpy(A, b, c):
        """Solve simple LP with CVXPY."""
        m, n = A.shape
        x = cp.Variable(n)
        constr = [b - A @ x >= 0]
        cp.Problem(cp.Minimize(x.T @ c), constr).solve()
        return x.value, constr[0].dual_value

    def test_m_less_n_full_rank_(self):
        """m<n, matrix full rank."""
        np.random.seed(0)
        print('\nm<n, matrix full rank\n')
        matrix = np.random.randn(2, 5)
        b, c = self.make_program_from_matrix(matrix)
        x, y = self.solve_program_cvxpy(matrix, b, c)
        self.check_solution_valid(matrix, b, c, x, y)
        print('real solution x')
        print(x)
        print('real solution y')
        print(y)
        solver = Solver(matrix, b, c, 0, len(b))
        self.check_solution_valid(matrix, b, c, solver.x, solver.y)



    def test_m_equal_n_full_rank_(self):
        """m=n, matrix full rank."""
        print('\nm=n, matrix full rank\n')
        np.random.seed(0)
        matrix = np.random.randn(3, 3)
        b, c = self.make_program_from_matrix(matrix)
        x, y = self.solve_program_cvxpy(matrix, b, c)
        self.check_solution_valid(matrix, b, c, x, y)
        print('real solution x')
        print(x)
        print('real solution y')
        print(y)
        solver = Solver(matrix, b, c, 0, len(b))
        self.check_solution_valid(matrix, b, c, solver.x, solver.y)

    def test_m_greater_n_full_rank_(self):
        """m>n, matrix full rank."""
        np.random.seed(0)
        print('\nm>n, matrix full rank\n')
        matrix = np.random.randn(5, 2)
        b, c = self.make_program_from_matrix(matrix)
        x, y = self.solve_program_cvxpy(matrix, b, c)
        self.check_solution_valid(matrix, b, c, x, y)
        print('real solution x')
        print(x)
        print('real solution y')
        print(y)
        solver = Solver(matrix, b, c, 0, len(b))
        self.check_solution_valid(matrix, b, c, solver.x, solver.y)

    def test_m_less_n_rank_deficient(self):
        """m<n, matrix rank deficient."""
        print('\nm<n, matrix rank deficient\n')
        np.random.seed(0)
        matrix = np.random.randn(2, 5)
        matrix = np.concatenate([matrix, [matrix.sum(0)]], axis=0)
        matrix = matrix[[0,2,1]]
        b, c = self.make_program_from_matrix(matrix)
        x, y = self.solve_program_cvxpy(matrix, b, c)
        self.check_solution_valid(matrix, b, c, x, y)
        print('real solution x')
        print(x)
        print('real solution y')
        print(y)
        solver = Solver(matrix, b, c, 0, len(b))
        self.check_solution_valid(matrix, b, c, solver.x, solver.y)

    def test_m_equal_n_rank_deficient(self):
        """m=n, matrix rank deficient."""
        print('\nm=n, matrix rank deficient\n')
        np.random.seed(0)
        matrix = np.random.randn(2, 3)
        matrix = np.concatenate([matrix, [matrix.sum(0)]], axis=0)
        matrix = matrix[[0,2,1]]
        b, c = self.make_program_from_matrix(matrix)
        x, y = self.solve_program_cvxpy(matrix, b, c)
        self.check_solution_valid(matrix, b, c, x, y)
        print('real solution x')
        print(x)
        print('real solution y')
        print(y)
        solver = Solver(matrix, b, c, 0, len(b))
        self.check_solution_valid(matrix, b, c, solver.x, solver.y)

    def test_m_greater_n_rank_deficient(self):
        """m>n, matrix rank deficient."""
        print('\nm>n, matrix rank deficient\n')
        np.random.seed(0)
        matrix = np.random.randn(5, 2)
        matrix = np.concatenate([matrix.T, [matrix.sum(1)]], axis=0).T
        # matrix = matrix[[0,2,1]]
        b, c = self.make_program_from_matrix(matrix)
        x, y = self.solve_program_cvxpy(matrix, b, c)
        self.check_solution_valid(matrix, b, c, x, y)
        print('real solution x')
        print(x)
        print('real solution y')
        print(y)
        solver = Solver(matrix, b, c, 0, len(b))
        self.check_solution_valid(matrix, b, c, solver.x, solver.y)

    # def test(self):
    #     matrix = np.random.randn(2,5)
    #     breakpoint()
    #     b, c = self.make_program_from_matrix(matrix)
    #     x, y = self.solve_program_cvxpy(matrix, b, c)
    #     solver = Solver(matrix, b, c, 0, len(b))


if __name__ == '__main__': # pragma: no cover
    main()