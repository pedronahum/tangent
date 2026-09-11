# Copyright 2018 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Gradients for SciPy special functions and a few linear-algebra ops.

Scientific code leans on `scipy.special` (erf, the gamma family, logistic,
log-sum-exp) and `scipy.linalg`. This module registers adjoints and, where
mechanical, forward-mode tangents for them, so a simulator or statistical model
written against SciPy differentiates like NumPy code.

Loaded optionally by `tangent/__init__.py`; a missing SciPy is silent.
"""

from __future__ import absolute_import

import numpy
import scipy.linalg
import scipy.special as _special

import tangent
from tangent.grads import adjoint
from tangent.tangents import tangent_

#
# scipy.special - error function family
#


@adjoint(_special.erf)
def aerf(y, x):
    d[x] = d[y] * (2.0 / numpy.sqrt(numpy.pi)) * numpy.exp(-(x**2))


@tangent_(_special.erf)
def terf(y, x):
    d[y] = d[x] * (2.0 / numpy.sqrt(numpy.pi)) * numpy.exp(-(x**2))


@adjoint(_special.erfc)
def aerfc(y, x):
    d[x] = -d[y] * (2.0 / numpy.sqrt(numpy.pi)) * numpy.exp(-(x**2))


@tangent_(_special.erfc)
def terfc(y, x):
    d[y] = -d[x] * (2.0 / numpy.sqrt(numpy.pi)) * numpy.exp(-(x**2))


#
# scipy.special - gamma family
#


# d/dx gammaln(x) = digamma(x) = psi(x)
@adjoint(_special.gammaln)
def agammaln(y, x):
    d[x] = d[y] * tangent.scipy_special.psi(x)


@tangent_(_special.gammaln)
def tgammaln(y, x):
    d[y] = d[x] * tangent.scipy_special.psi(x)


# d/dx psi(x) = polygamma(1, x)
@adjoint(_special.psi)
def apsi(y, x):
    d[x] = d[y] * tangent.scipy_polygamma1(x)


@tangent_(_special.psi)
def tpsi(y, x):
    d[y] = d[x] * tangent.scipy_polygamma1(x)


# d/dx gamma(x) = gamma(x) * psi(x) = y * psi(x)
@adjoint(_special.gamma)
def agamma(y, x):
    d[x] = d[y] * y * tangent.scipy_special.psi(x)


@tangent_(_special.gamma)
def tgamma(y, x):
    d[y] = d[x] * y * tangent.scipy_special.psi(x)


#
# scipy.special - logistic family
#


# expit(x) = 1 / (1 + exp(-x));  d/dx = y (1 - y)
@adjoint(_special.expit)
def aexpit(y, x):
    d[x] = d[y] * y * (1.0 - y)


@tangent_(_special.expit)
def texpit(y, x):
    d[y] = d[x] * y * (1.0 - y)


# logit(x) = log(x / (1 - x));  d/dx = 1 / (x (1 - x))
@adjoint(_special.logit)
def alogit(y, x):
    d[x] = d[y] / (x * (1.0 - x))


@tangent_(_special.logit)
def tlogit(y, x):
    d[y] = d[x] / (x * (1.0 - x))


#
# scipy.special - xlogy(x, y) = x * log(y)
#


@adjoint(_special.xlogy)
def axlogy(z, x, y):
    d[x] = tangent.unbroadcast(d[z] * numpy.log(y), x)
    d[y] = tangent.unbroadcast(d[z] * x / y, y)


@tangent_(_special.xlogy)
def txlogy(z, x, y):
    d[z] = numpy.log(y) * d[x] + (x / y) * d[y]


#
# scipy.special.logsumexp (a reduction; adjoint routes through the softmax)
#


@adjoint(_special.logsumexp)
def alogsumexp(y, x, axis=None):
    d[x] = tangent.scipy_logsumexp_grad(d[y], x, y, axis)


#
# scipy.linalg - solve / inv (mirror numpy's linalg adjoints)
#


@adjoint(scipy.linalg.solve)
def asolve(y, a, b):
    # For y = A^-1 b: d[b] = A^-T d[y]; d[A] = -d[b] outer y.
    d[b] = tangent.scipy_linalg_solve(numpy.swapaxes(a, -1, -2), d[y])
    d[a] = tangent.scipy_solve_grad_a(d[b], y)


@adjoint(scipy.linalg.inv)
def ainv(y, x):
    d[x] = -numpy.matmul(numpy.swapaxes(y, -1, -2), numpy.matmul(d[y], numpy.swapaxes(y, -1, -2)))


#
# Runtime helpers referenced by the adjoints above.
#


def _polygamma1(x):
    return _special.polygamma(1, x)


def _linalg_solve(a, b):
    return scipy.linalg.solve(a, b)


def _solve_grad_a(db, y):
    """d[A] for y = solve(A, b): -db outer y (vector) or -db @ y^T (matrix)."""
    db = numpy.asarray(db)
    y = numpy.asarray(y)
    if y.ndim == 1:
        return -numpy.outer(db, y)
    return -numpy.matmul(db, numpy.swapaxes(y, -1, -2))


def _logsumexp_grad(dy, x, y, axis):
    """Adjoint of logsumexp: dy * softmax(x) along `axis`."""
    softmax = numpy.exp(x - numpy.expand_dims(y, axis) if axis is not None else x - y)
    if axis is None:
        return numpy.asarray(dy) * softmax
    return numpy.expand_dims(dy, axis) * softmax


# Expose the runtime helpers and scipy.special on the tangent namespace so the
# generated code (which references `tangent.<name>`) can resolve them.
tangent.scipy_special = _special
tangent.scipy_polygamma1 = _polygamma1
tangent.scipy_logsumexp_grad = _logsumexp_grad
tangent.scipy_linalg_solve = _linalg_solve
tangent.scipy_solve_grad_a = _solve_grad_a
