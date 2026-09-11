# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#      Unless required by applicable law or agreed to in writing, software
#      distributed under the License is distributed on an "AS IS" BASIS,
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#      See the License for the specific language governing permissions and
#      limitations under the License.
"""Automatically test gradients with multiple inputs, modes and motions."""

import os

import numpy as np

import functions
import tfe_utils

# Random test inputs (vectors, matrices, TF tensors) are drawn at collection
# time. Seed them so every run exercises the same inputs and failures
# reproduce; export TANGENT_TEST_SEED to fuzz with different draws, e.g.
#   TANGENT_TEST_SEED=$RANDOM pytest tests/
TEST_SEED = int(os.environ.get('TANGENT_TEST_SEED', '20260908'))
np.random.seed(TEST_SEED)

# The persistent disk cache must not leak between test runs (a disk hit skips
# compilation, so compile-time warnings would not be re-emitted) or pollute
# the developer's real ~/.cache. Tests that exercise the disk cache re-enable
# it explicitly with monkeypatch.
os.environ.setdefault('TANGENT_DISK_CACHE', '0')


# Functions excluded from the parametrized gradient harness. Each entry carries
# a reason. (Previously this was a silent blacklist that hid broken behavior -
# keep it audited.) Every remaining entry is excluded by design, not because a
# mode is broken: the last mode-gap entries (fn_multiple_return,
# active_subscript, init_array_grad_maybe_active, cart2polar) now pass the full
# harness in every mode.
blacklisted = [
    # insert_grad_of / context-manager inlining path is not harness-compatible.
    'inlining_contextmanager',
    # Nested function definitions are rejected with TangentParseError by design
    # (single-exit reverse transform); see tests/test_fence.py.
    'iterpower_with_nested_def',
    # Custom-gradient insertion helper; not a plain differentiable function.
    'insert_grad_of',
    # TensorFlow tracing helpers - require TF and the tracing path; exercised in
    # tests/test_tensorflow.py when TF is installed, not in the core harness.
    '_trace_mul',
    '_nontrace_mul',
]

funcs = [f for f in functions.__dict__.values() if callable(f)]
whitelist = [f for f in funcs if f.__name__ not in blacklisted]
blacklist = [f for f in funcs if f.__name__ in blacklisted]


def pytest_addoption(parser):
    # Only test with one input
    parser.addoption('--short', action='store_true')
    # Only test with all inputs
    parser.addoption('--all', action='store_true')
    # Restrict to certain functions by name
    parser.addoption('--func_filter', action='store')


def pytest_generate_tests(metafunc):
    # Parametrize the functions
    if 'func' in metafunc.fixturenames:
        func_filter = metafunc.config.option.func_filter

        # Test takes args, only pass funcs with same signature
        args = tuple(
            arg
            for arg in metafunc.fixturenames
            if arg not in ('func', 'motion', 'optimized', 'preserve_result')
        )
        if args:
            func_args = []
            for f in whitelist:
                fc = f.__code__
                if fc.co_varnames[: fc.co_argcount] == args:
                    func_args.append(f)
        else:
            func_args = funcs

        if func_filter:
            func_args = [f for f in func_args if func_filter in f.__name__]

        func_names = [f.__name__ for f in func_args]
        metafunc.parametrize('func', func_args, ids=func_names)

    if 'motion' in metafunc.fixturenames:
        metafunc.parametrize('motion', ('split', 'joint'))

    if 'optimized' in metafunc.fixturenames:
        metafunc.parametrize('optimized', (True, False), ids=('optimized', 'unoptimized'))

    if 'preserve_result' in metafunc.fixturenames:
        metafunc.parametrize('preserve_result', (True, False))

    # Parametrize the arguments
    short = metafunc.config.option.short

    bools = [True, False]
    for arg in ['boolean', 'boolean1', 'boolean2']:
        if arg in metafunc.fixturenames:
            metafunc.parametrize(arg, bools)

    scalars = [2.0] if short else [-2.0, -1.5, -1.0, -0.5, -0.1, 0.1, 0.5, 1.0, 1.5, 2.0]
    for arg in 'abc':
        if arg in metafunc.fixturenames:
            metafunc.parametrize(arg, scalars)

    integers = [1] if short else [1, 2, 3]
    if 'n' in metafunc.fixturenames:
        metafunc.parametrize('n', integers)

    vectors = [np.random.randn(i) for i in ((3,) if short else (3, 5, 10))]
    if 'x' in metafunc.fixturenames:
        metafunc.parametrize('x', vectors)

    square_matrices = [np.random.randn(*i) for i in (((3, 3),) if short else ((1, 1), (5, 5)))]
    if 'sqm' in metafunc.fixturenames:
        metafunc.parametrize('sqm', square_matrices)

    tfe_utils.register_parametrizations(metafunc, short)
