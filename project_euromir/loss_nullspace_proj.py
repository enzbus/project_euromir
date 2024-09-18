"""Loss and related functions for nullspace projection model."""

import numpy as np
import scipy as sp


def _densify_also_nonsquare(linear_operator):
    result = np.empty(linear_operator.shape)
    for j in range(linear_operator.shape[1]):
        e_j = np.zeros(linear_operator.shape[1])
        e_j[j] = 1.
        result[:, j] = linear_operator.matvec(e_j)
    return result


class NullSpaceModel:

    def __init__(
        self, m, n, zero, nonneg, matrix_transfqr, b, c, nullspace_projector):
        self.m = m
        self.n = n
        self.zero = zero
        self.nonneg = nonneg
        self.matrix = matrix_transfqr
        self.b = b
        self.c = c
        self.nullspace_projector = nullspace_projector
        self.b_proj = self.b @ self.nullspace_projector

        # since matrix is from QR
        self.y0 = -self.c @ self.matrix.T

    def loss(self, variable):
        x = variable[:self.n]
        y_null = variable[self.n:]
        y = self.y0 + self.nullspace_projector @ y_null
        s = -self.matrix @ x + self.b
        y_loss = np.linalg.norm(np.minimum(y[self.zero:], 0.))**2
        s_loss_zero = np.linalg.norm(s[:self.zero])**2
        s_loss_nonneg = np.linalg.norm(np.minimum(s[self.zero:], 0.))**2
        gap_loss = (self.c.T @ x + self.b.T @ y)**2
        return (y_loss + s_loss_zero + s_loss_nonneg + gap_loss
            ) / 2.

    def gradient(self, variable):
        x = variable[:self.n]
        y_null = variable[self.n:]
        y = self.y0 + self.nullspace_projector @ y_null
        s = -self.matrix @ x + self.b
        y_error = np.minimum(y[self.zero:self.zero+self.nonneg], 0.)
        s_error = np.copy(s)
        s_error[self.zero:self.zero+self.nonneg] = np.minimum(
            s[self.zero:self.zero+self.nonneg], 0.)
        gap = self.c.T @ x + self.b.T @ y

        gradient = np.empty_like(variable)
        gradient[:self.n] = -self.matrix.T @ s_error
        gradient[self.n:] = self.nullspace_projector[self.zero:].T @ y_error

        gradient[:self.n] += gap * self.c
        gradient[self.n:] += gap * self.b_proj
        return gradient

    def hessian(self, variable): #, regularizer = 0.):
        x = variable[:self.n]
        y_null = variable[self.n:]
        y = self.y0 + self.nullspace_projector @ y_null
        s = -self.matrix @ x + self.b
        y_error = np.minimum(y[self.zero:self.zero+self.nonneg], 0.)
        s_error = np.copy(s)
        s_error[self.zero:self.zero+self.nonneg] = np.minimum(
            s[self.zero:self.zero+self.nonneg], 0.)
        gap = self.c.T @ x + self.b.T @ y

        # this is DProj
        s_mask = np.ones(self.m, dtype=float)
        s_mask[self.zero:] = s_error[self.zero:] < 0.
        y_mask = np.zeros(self.m, dtype=float)
        y_mask[self.zero:] = y_error < 0.

        def _matvec(dvar):
            result = np.empty_like(dvar)
            dx = dvar[:self.n]
            dy_null = dvar[self.n:]

            # s_error sqnorm
            result[:self.n] = (self.matrix.T @ (s_mask * (self.matrix @ dx)))

            # y_error sqnorm
            result[self.n:] = (self.nullspace_projector.T @ (
                y_mask * (self.nullspace_projector @ dy_null)))

            # gap
            constants = np.concatenate([self.c, self.b_proj])
            result[:] += constants * (constants @ dvar)

            return result # + regularizer * dxdy

        return sp.sparse.linalg.LinearOperator(
            shape=(len(variable), len(variable)),
            matvec=_matvec
        )

    def dense_hessian(self, variable):
        return _densify_also_nonsquare(self.hessian(variable=variable))


if __name__ == '__main__': # pragma: no cover

    from scipy.optimize import check_grad

    # create consts
    np.random.seed(0)
    m = 20
    n = 10
    zero = 5
    nonneg = 15
    matrix = np.random.randn(m, n)
    q, r = np.linalg.qr(matrix, mode='complete')
    matrix_transfqr = q[:, :n]
    nullspace_proj = q[:, n:]

    b = np.random.randn(m)
    c = np.random.randn(n)

    model = NullSpaceModel(
        m=m, n=n, zero=zero, nonneg=nonneg, matrix_transfqr=matrix_transfqr,
        b=b, c=c, nullspace_projector=nullspace_proj)

    print('\nCHECKING GRADIENT')
    for i in range(10):
        print(check_grad(model.loss, model.gradient, np.random.randn(m)))

    print('\nCHECKING HESSIAN')
    for i in range(10):
        print(check_grad(model.gradient, model.dense_hessian,
            np.random.randn(m)))
