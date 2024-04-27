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

import time
from unittest import TestCase, main

import numpy as np
import scipy.sparse as sp

import project_euromir as lib


class TestLinearAlgebra(TestCase):

    # def test_import(self):
    #     """Test that the wheel installed correctly."""
    #     import project_euromir

    def test_csc(self):
        """Test CSC matvec."""

        m = 100
        n = 100
        timer = 0.
        for seed in range(100):
            np.random.seed(seed)
            mat = sp.random(m=m, n=n, dtype=float).tocsc()
            inp = np.random.randn(n)
            out = np.zeros(m, dtype=float)
            mult = np.random.choice([-1., None, 1.])
            if mult is None:
                mult = np.random.randn()
            s = time.time()
            lib.add_csc_matvec(
                n=n, col_pointers=mat.indptr, row_indexes=mat.indices,
                mat_elements=mat.data, input=inp, output=out, mult=mult)
            timer += time.time() - s
            self.assertTrue(np.allclose(out, mult * (mat @ inp)))
        print('timer CSC', timer)

    def test_csr(self):
        """Test CSR matvec."""

        m = 100
        n = 100
        timer = 0.

        for seed in range(100):
            np.random.seed(seed)
            mat = sp.random(m=m, n=n, dtype=float).tocsr()
            inp = np.random.randn(n)
            out = np.zeros(m, dtype=float)
            mult = np.random.choice([-1., None, 1.])
            if mult is None:
                mult = np.random.randn()

            s = time.time()
            lib.add_csr_matvec(
                m=m, row_pointers=mat.indptr, col_indexes=mat.indices,
                mat_elements=mat.data, input=inp, output=out, mult=mult)
            timer += time.time() - s
            self.assertTrue(np.allclose(out, mult * (mat @ inp)))

        print('timer CSR', timer)
