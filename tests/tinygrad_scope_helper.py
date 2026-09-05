# Copyright 2026 Tangent contributors
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
"""NumPy-only functions used to test tinygrad method-resolution scoping.

This module deliberately does NOT import tinygrad. It exists so that
``test_tinygrad.py`` can verify that the tinygrad method resolver only claims
method calls for functions defined in namespaces that actually use tinygrad,
leaving ordinary NumPy ``.sum()``/``.mean()`` calls untouched even when
tinygrad is imported elsewhere in the process.
"""
import numpy as np  # noqa: F401  (available to the differentiated functions)


def np_method_sum(x):
    return x.sum()


def np_method_mean(x):
    return x.mean()
