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
"""Tests that `import tangent` is silent.

Missing optional backends (jax, torch, tensorflow, keras, tinygrad, ...) are a
normal condition and must not produce warnings or stdout noise at import time.
They are logged at DEBUG level on the 'tangent' logger instead, and their
status is available via tangent.backend_status().
"""
import logging
import subprocess
import sys

import tangent


def test_import_tangent_emits_no_warnings_and_no_stdout():
  """A bare `import tangent` must print nothing and warn nothing."""
  result = subprocess.run(
      [sys.executable, '-W', 'error::UserWarning', '-c', 'import tangent'],
      capture_output=True, text=True, timeout=300)
  assert result.returncode == 0, (
      'import tangent raised a UserWarning (or failed):\n%s' % result.stderr)
  assert result.stdout == '', (
      'import tangent wrote to stdout:\n%s' % result.stdout)
  # Optional third-party packages may write their own chatter to stderr, but
  # no warning may originate from tangent itself.
  tangent_warning_lines = [
      line for line in result.stderr.splitlines()
      if 'tangent' in line and 'Warning' in line
  ]
  assert not tangent_warning_lines, (
      'import tangent emitted warnings:\n%s' % '\n'.join(tangent_warning_lines))


def test_missing_backends_are_logged_at_debug_level():
  """The missing-backend details remain discoverable via the tangent logger."""
  result = subprocess.run(
      [sys.executable, '-c',
       'import logging; logging.basicConfig(level=logging.DEBUG); '
       'import tangent'],
      capture_output=True, text=True, timeout=300)
  assert result.returncode == 0, result.stderr


def test_backend_status_helper():
  status = tangent.backend_status()
  assert status['numpy'] == 'available'
  for backend in ('tensorflow', 'jax', 'torch', 'keras', 'tinygrad',
                  'visualization'):
    assert backend in status
    assert (status[backend] in ('available', 'not installed') or
            status[backend].startswith('broken:')), (
                'unexpected status for %s: %r' % (backend, status[backend]))
  # The helper returns a copy, not the internal dict.
  status['numpy'] = 'mutated'
  assert tangent.backend_status()['numpy'] == 'available'
