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
"""Solver main function, will be unpacked and call all the rest."""

import time

import numpy as np
import scipy as sp

from project_euromir import equilibrate
from project_euromir.lbfgs import minimize_lbfgs
from project_euromir.refinement import refine

DEBUG = False
if DEBUG:
    import matplotlib.pyplot as plt

USE_MY_LBFGS = False
ACTIVE_SET = False # this doesn't work yet, not sure if worth trying to fix it
IMPLICIT_FORMULATION = True # this does help!!! some minor issues on hessian

QR_PRESOLVE = False

if ACTIVE_SET:
    assert USE_MY_LBFGS

PGTOL = 0. # I tried this as a stopping condition for lbfgs, but it can break
# (meaning that active set is still not robust and when switching to lsqr
# it breaks); you can try e.g. 1e-12

def solve(matrix, b, c, zero, nonneg, lbfgs_memory=10):
    "Main function."

    # some shape checking
    n = len(c)
    m = len(b)
    assert matrix.shape == (m, n)
    assert zero + nonneg == m

    if QR_PRESOLVE:
        q, r = np.linalg.qr(np.vstack([matrix.todense(), c.reshape((1, n))]))
        matrix_transf = q[:-1]
        c_transf = q[-1].A1
        sigma = np.linalg.norm(b)
        b_transf = b/sigma

    else:
        # equilibration
        d, e, sigma, rho, matrix_transf, b_transf, c_transf = \
        equilibrate.hsde_ruiz_equilibration(
                matrix, b, c, dimensions={
                    'zero': zero, 'nonneg': nonneg, 'second_order': ()},
                max_iters=25)

    # temporary, build sparse Q
    Q = sp.sparse.bmat([
        [None, matrix_transf.T, c_transf.reshape(n, 1)],
        [-matrix_transf, None, b_transf.reshape(m, 1)],
        [-c_transf.reshape(1, n), -b_transf.reshape(1, m), None],
        ]).tocsc()

    # breakpoint()

    # temporary, [Q, -I]
    QI = sp.sparse.hstack([Q, -sp.sparse.eye(n+m+1, format='csc')])

    # temporary, remove v in zero cone
    _as = np.concatenate(
        [np.ones(n+m+1, dtype=bool),
        np.zeros(n + zero, dtype=bool),
        np.ones(m+1 - zero, dtype=bool)])

    # so we define the matrix of the LBFGS loop
    system_matrix = QI[:, _as]

    # pre-allocate vars used below
    residual = np.empty(system_matrix.shape[0], dtype=float)
    error = np.empty(system_matrix.shape[1]-n-zero, dtype=float)
    gradient = np.empty(system_matrix.shape[1], dtype=float)

    # temporary, just define loss-gradient function for LPs

    if ACTIVE_SET:
        raise Exception('Need to remove this option.')
        # test using active set instead and internal projection
        # when extending to other cones we'll have to figure out
        def loss_gradient(variable):
            variable[n+zero:] = np.maximum(variable[n+zero:], 0)
            residual[:] = system_matrix @ variable
            # error[:] = np.minimum(variable[n+zero:], 0)
            loss = np.linalg.norm(residual)**2 #+ np.linalg.norm(error)**2
            gradient[:] = 2 * (system_matrix.T @ residual)
            active_set = np.ones_like(variable, dtype=bool)
            active_set[n+zero:] = (variable[n+zero:] > 0) | (gradient[n+zero:] < 0)
            # gradient[n+zero:] += 2 * error
            return loss, gradient, active_set

    else:
        if IMPLICIT_FORMULATION:

            # pre-allocate vars used below
            u_prealloc_err = np.empty(nonneg+1, dtype=float)
            v_prealloc_err = np.empty(m+n+1, dtype=float)
            # error = np.empty(system_matrix.shape[1]-n-zero, dtype=float)
            newgradient = np.empty(m+n+1, dtype=float)

            # variable is only u
            def loss_gradient(u):
                u_prealloc_err = np.minimum(u[n+zero:], 0.)
                v_prealloc_err[:] = Q @ u
                v_prealloc_err[n+zero:] = np.minimum(
                    v_prealloc_err[n+zero:], 0.)
                # resv1 = np.minimum(Q[n+zero:] @ u, 0.)
                # resv2 = Q[:n+zero] @ u
                loss = np.linalg.norm(u_prealloc_err)**2
                loss += np.linalg.norm(v_prealloc_err)**2
                # loss += np.linalg.norm(resv2)**2

                newgradient[:] = 2 * Q.T @ v_prealloc_err
                newgradient[n+zero:] += 2 * u_prealloc_err

                # grad = np.zeros_like(u)
                # grad[n+zero:] += 2 * resu
                # grad += 2 * Q[n+zero:].T @ resv1
                # grad += 2 * Q[:n+zero].T @ resv2

                return loss, newgradient

            def hessian(u): # TODO: this is not correct yet, need to check_grad it (it's close)

                v = Q @ u
                mask_u_cone = np.zeros_like(u, dtype=float)
                mask_v_cone = np.ones_like(u, dtype=float)
                mask_u_cone[n+zero:] = u[n+zero:] < 0.
                mask_v_cone[n+zero:] = v[n+zero:] < 0.

                def _matvec(myvar):

                    tmp = Q @ myvar
                    tmp *= mask_v_cone
                    tmp = Q.T @ tmp
                    tmp += mask_u_cone * myvar
                    return 2 * tmp

                    result = np.zeros_like(u)

                    #resu
                    result[n+zero:][u[n+zero:] < 0] += myvar[n+zero:][u[n+zero:] < 0]

                    #resv1
                    resv1_nonproj = Q[n+zero:] @ u
                    tmp = Q[n+zero:] @ myvar
                    tmp[resv1_nonproj > 0] = 0.
                    result += Q[n+zero:].T @ tmp

                    #resv2
                    result += Q[:n+zero].T @ (Q[:n+zero] @ myvar)

                    return 2 * result
                return sp.sparse.linalg.LinearOperator(
                    shape=(len(u), len(u)),
                    matvec=_matvec
                )

        else:
            def loss_gradient(variable):
                residual[:] = system_matrix @ variable
                error[:] = np.minimum(variable[n+zero:], 0)
                loss = np.linalg.norm(residual)**2 + np.linalg.norm(error)**2
                gradient[:] = 2 * (system_matrix.T @ residual)
                gradient[n+zero:] += 2 * error
                return loss, gradient

            def hessian(variable):
                def _matvec(myvar):
                    result = system_matrix.T @ (system_matrix @ myvar)
                    result[n+zero:][variable[n+zero:] < 0] += myvar[n+zero:][variable[n+zero:] < 0]
                    return 2 * result
                return sp.sparse.linalg.LinearOperator(
                    shape=(len(variable), len(variable)),
                    matvec=_matvec
                )

    if IMPLICIT_FORMULATION:
        x_0 = np.zeros(n+m+1)
        x_0[-1] = 1.
    else:
        # initialize with all zeros and 1 only on the HSDE feasible flag
        x_0 = np.zeros(system_matrix.shape[1])
        x_0[n+m] = 1.

    # debug mode, plot history of losses
    if DEBUG:
        residual_sqnorms = []
        violation_sqnorms = []
        def _callback(variable):
            assert not IMPLICIT_FORMULATION
            residual[:] = system_matrix @ variable
            error[:] = np.minimum(variable[n+zero:], 0)
            residual_sqnorms.append(np.linalg.norm(residual)**2)
            violation_sqnorms.append(np.linalg.norm(error)**2)

    # call LBFGS
    start = time.time()
    if not USE_MY_LBFGS:
        lbfgs_result = sp.optimize.fmin_l_bfgs_b(
            loss_gradient,
            x0=x_0,
            m=lbfgs_memory,
            maxfun=1e10,
            factr=0.,
            pgtol=PGTOL,
            callback=_callback if DEBUG else None,
            maxiter=1e10)
        # print LBFGS stats
        stats = lbfgs_result[2]
        stats.pop('grad')
        print('LBFGS stats', stats)
        result_variable = lbfgs_result[0]
    else:
        result_variable = minimize_lbfgs(
            loss_and_gradient_function=loss_gradient,
            initial_point=x_0,
            callback=_callback if DEBUG else None,
            memory=lbfgs_memory,
            max_iters=int(1e10),
            # c_1=1e-3, c_2=.9,
            # ls_backtrack=.5,
            # ls_forward=1.1,
            pgtol=PGTOL,
            # hessian_approximator=hessian,
            use_active_set=ACTIVE_SET,
            max_ls=100)
    print('LBFGS took', time.time() - start)

    # debug mode, plot history of losses
    if DEBUG:
        plt.plot(residual_sqnorms, label='residual square norms')
        plt.plot(violation_sqnorms, label='violation square norms')
        plt.yscale('log')
        plt.legend()
        plt.show()

    # extract result
    if IMPLICIT_FORMULATION:
        u = result_variable
        assert u[-1] > 0, 'Certificates not yet implemented'
        # u /= u[-1]
        v = Q @ u
    else:
        u = result_variable[:n+m+1]
        v = np.zeros(n+m+1)
        v[n+zero:] = result_variable[n+m+1:]

    u, v = refine(
        z=u-v, matrix=matrix_transf, b=b_transf, c=c_transf, zero=zero,
        nonneg=nonneg)

    # u, v = refine(
    #     z=u-v, matrix=matrix_transf, b=b_transf, c=c_transf, zero=zero,
    #     nonneg=nonneg)

    # Transform back into problem format
    u1, u2, u3 = u[:n], u[n:n+m], u[-1]
    v2, v3 = v[n:n+m], v[-1]

    if v3 > u3:
        raise NotImplementedError('Certificates not yet implemented.')

    # Apply HSDE scaling
    x = u1 / u3
    y = u2 / u3
    s = v2 / u3

    if QR_PRESOLVE:
        x_orig = np.linalg.solve(r, x) * sigma
        y_orig = y
        s_orig = s * sigma

    else:
        # invert Ruiz scaling, copied from other repo
        x_orig =  e * x / sigma
        y_orig = d * y / rho
        s_orig = (s/d) / sigma

    return x_orig, y_orig, s_orig
