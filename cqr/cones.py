# Copyright 2025 Enzo Busseti
#
# This file is part of CQR, the Conic QR Solver.
#
# CQR is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# CQR is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# CQR. If not, see <https://www.gnu.org/licenses/>.
"""Projections on cones."""

import numpy as np

def project_nonsymm_soc(x, a):
    """Project on the non-symmetric second-order cone.

    This is defined as {(t, y) in R x R^n | t >= ||x * a||_2}, with
    scaling vector a in R^n_++ (all strictly positive elements), and the
    multiplication is elementwise. The standard second-order cone has scaling
    equal to all ones.

    This cone is not self dual: its dual has the (elementwise) inverse scaling vector.
    Projection is not as efficient as projection on the standard SOC, it
    requires an iterative procedure, but can be warm-started.
    """

    t = x[0]
    y = x[1:]

    # case 1: vector is in cone
    if t >= np.linalg.norm(y * a):
        print('1')
        return np.copy(x)

    # case 2: negative of vector in dual cone
    if -t >= np.linalg.norm(y/a):
        print('2')
        return np.zeros_like(x)

    # case 3: projection is on non-zero surface of (primal) cone
    #
    # Derivation:
    #
    # Let result be (s, z).
    #
    # Following equations come from zero-gradient of Lagrangian in s and z
    # s = t / (1 + 2 * mu)
    # z = y / (1 - 2 * mu * a**2)
    # mu is the Lagrangia multiplier.
    #
    # We find it by Newton search looking for the solution of this
    # s**2 = sum((z*a)**2)
    # knowing that mu is in the open interval (-0.5, 0).

    print('3')

    s = lambda mu: t / (1 + 2 * mu)
    z = lambda mu: y / (1 - 2 * mu * a**2)
    loss = lambda mu: s(mu)**2 - np.sum((z(mu)*a)**2)

    # placeholder
    import scipy as sp
    result = sp.optimize.root_scalar(loss, x0=-.25)
    print(result)
    mu = result.root

    return np.concatenate([[s(mu)], z(mu)])


if __name__ == "__main__":

    np.random.seed(0)
    N = 100
    NTRIES = 100
    for _ in range(100):
        x = np.random.randn(N)
        # chosen so that 3 clauses are more or less equally likely
        a = np.random.uniform(0, 1e-1 if _ % 2 == 0 else 1000., N-1)
        pi = project_nonsymm_soc(x, a)

        # check pi in cone
        assert pi[0] - np.linalg.norm(pi[1:] * a) >= -1e6

        # check pi - x in dual cone
        diff = pi - x
        assert diff[0] - np.linalg.norm(diff[1:] / a) >= -1e6

        # check pi orthogonal to pi - x
        assert np.dot(pi, diff) < 1e6
