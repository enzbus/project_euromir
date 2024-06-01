# BSD 3-Clause License

# Copyright (c) 2024-, Enzo Busseti

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Pure Python implementation of L-BFGS for testing.
 
We only need the multiplication of the gradient at the current point by the
approximate inverse of the second derivative. Line search and choosing next
point are done externally.

References:

- Original paper:
    Updating quasi-Newton matrices with limited storage, Nocedal 1980
    https://doi.org/10.1090/S0025-5718-1980-0572855-7 
    (easy to find non-paywalled)

- Wikipedia page:
    https://en.wikipedia.org/wiki/Limited-memory_BFGS,
    https://web.archive.org/web/20240515120721/https://en.wikipedia.org/wiki/Limited-memory_BFGS

- Blog post:
    https://aria42.com/blog/2014/12/understanding-lbfgs
    https://web.archive.org/web/20231002054213/https://aria42.com/blog/2014/12/understanding-lbfgs
"""

from __future__ import annotations

import numpy as np


def lbfgs_multiply(
    current_gradient: np.array,
    past_steps: np.array,
    past_grad_diffs: np.array,
    base_inverse_diagonal: float | np.array = 1.
):
    r"""Multiply current gradient by the approximate inverse second derivative.

    :param current_gradient:
    :type current_gradient: np.array (1-dimensional)
    :param past_steps: First dimension is L-BFGS memory. Most recent step is
        last row.
    :type past_steps: np.array (2-dimensional)
    :param past_grad_diffs: First dimension is L-BFGS memory. Most recent 
        gradient difference is last row.
    :type past_grad_diffs: np.array (2-dimensional)
    :param base_inverse_diagonal: Diagonal of :math:`H_0`, the base inverse
        Hessian, before the L-BFGS corrections. By default 1, meaning we
        take the identity as base.
    :type base_inverse_diagonal: float or np.array (1-dimensional)
    """

    memory = past_steps.shape[0]
    assert past_grad_diffs.shape[0] == memory

    # compute rhos;
    # this needs to be optimized for numerical stability, normalize by geo mean
    # of l2 norms of s,y pairs
    rhos = np.empty(memory, dtype=float)
    for i in range(memory):
        rhos[i] = 1. / np.dot(past_steps[i], past_grad_diffs[i])

    # using paper notation
    q = current_gradient

    # right part, backward iteration
    alphas = np.empty(memory, dtype=float)
    for i in range(memory-1, -1, -1):
        alphas[i] = rhos[i] * np.dot(past_steps[i], q)
        q -= alphas[i] * past_grad_diffs[i]

    # center part
    r = base_inverse_diagonal * q

    # wikipedia does this correction, not found in original paper
    # gamma_correction = np.dot(
    #     past_steps[-1], past_grad_diffs[-1]) / np.dot(
    #         past_grad_diffs[-1], past_grad_diffs[-1])
    # r = gamma_correction * (base_inverse_diagonal * q)

    # left part, forward iteration
    betas = np.empty(memory, dtype=float)
    for i in range(memory):
        betas[i] = rhos[i] * np.dot(past_grad_diffs[i], r)
        r += (alphas[i] - betas[i]) * past_steps[i]

    return r


def _lbfgs_multiply_dense(
    current_gradient: np.array,
    past_steps: np.array,
    past_grad_diffs: np.array,
    memory: int = 5,
    base_inverse_diagonal: float | np.array = 1.
):
    """Same as above using dense matrix."""

    memory = past_steps.shape[1]
    assert past_grad_diffs.shape[1] == memory

    H = np.diag(base_inverse_diagonal)

    for i in range(memory):
        rho = 1. / np.dot(past_steps[i], past_grad_diffs[i])
        left = np.eye(len(H)) - rho * np.outer(
            past_grad_diffs[i], past_steps[i])
        right = left.T
        H = left @ H @ right + rho * np.outer(past_steps[i], past_steps[i])

    return H @ current_gradient
