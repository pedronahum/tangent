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
"""Persistent on-disk cache for generated gradient source.

Source-to-source AD always pays a compile bill; the in-memory LRU already
makes repeated `grad()` calls cheap within a process. This cache extends that
across processes: the *generated source* is stored keyed by the same tuple as
the memory cache (function source hash, options) plus the Tangent version,
and on a hit the source is recompiled directly - skipping parsing,
desugaring, differentiation, and optimization entirely.

Only the source is persisted, never pickled objects. At load time the module
is executed against a namespace rebuilt from the function's own globals, and
every global name the compiled entry functions actually reference
(its LOAD_GLOBAL instructions, walked through nested code objects) is verified to
resolve; any mismatch - or any error at all - falls back to a full
recompilation, so the cache can only ever cost a miss, not correctness.

Location: `$TANGENT_CACHE_DIR`, else `$XDG_CACHE_HOME/tangent-ad`, else
`~/.cache/tangent-ad`. Disable entirely with `TANGENT_DISK_CACHE=0`.
"""

from __future__ import absolute_import

import builtins
import dis
import hashlib
import json
import os
import types


def enabled():
    return os.environ.get('TANGENT_DISK_CACHE', '1') != '0'


def cache_dir():
    override = os.environ.get('TANGENT_CACHE_DIR')
    if override:
        return override
    xdg = os.environ.get('XDG_CACHE_HOME')
    base = xdg if xdg else os.path.join(os.path.expanduser('~'), '.cache')
    return os.path.join(base, 'tangent-ad')


_package_fingerprint_memo = None


def _package_fingerprint():
    """A cheap fingerprint of Tangent's own sources.

    Generated source depends on the transformer and the gradient templates,
    so entries must be invalidated when *Tangent* changes, not only when the
    user's function does. Version alone is not enough during development
    (editable installs change without a version bump); the newest mtime
    across the package's modules is, and costs ~60 stats once per process.
    """
    global _package_fingerprint_memo
    if _package_fingerprint_memo is None:
        import tangent

        newest = 0
        pkg_dir = os.path.dirname(tangent.__file__)
        try:
            for root, _, files in os.walk(pkg_dir):
                for name in files:
                    if name.endswith('.py'):
                        newest = max(newest, os.stat(os.path.join(root, name)).st_mtime_ns)
        except OSError:
            pass
        _package_fingerprint_memo = '%s|%d' % (getattr(tangent, '__version__', '?'), newest)
    return _package_fingerprint_memo


def _digest(cache_key):
    payload = repr(cache_key) + '|' + _package_fingerprint()
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _entry_path(cache_key):
    return os.path.join(cache_dir(), _digest(cache_key) + '.json')


def _build_namespace(func):
    """The globals the generated module runs against, rebuilt from `func`."""
    import numpy
    import tangent

    unwrapped = func
    seen = set()
    while hasattr(unwrapped, '__wrapped__') and id(unwrapped) not in seen:
        seen.add(id(unwrapped))
        unwrapped = unwrapped.__wrapped__
    namespace = dict(getattr(unwrapped, '__globals__', {}))
    if getattr(unwrapped, '__closure__', None):
        namespace.update(
            dict(
                zip(
                    unwrapped.__code__.co_freevars,
                    (cell.cell_contents for cell in unwrapped.__closure__),
                )
            )
        )
    namespace.setdefault('tangent', tangent)
    namespace.setdefault('numpy', numpy)
    namespace.setdefault('np', numpy)
    return namespace


def _referenced_globals(code, acc=None):
    """Names a code object (recursively) loads as globals.

    Uses the bytecode's LOAD_GLOBAL/LOAD_NAME instructions rather than
    `co_names`, which also contains attribute names (`tangent.push` would
    otherwise contribute a spurious global `push`).
    """
    if acc is None:
        acc = set()
    for instr in dis.get_instructions(code):
        if instr.opname in ('LOAD_GLOBAL', 'LOAD_NAME'):
            acc.add(instr.argval)
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            _referenced_globals(const, acc)
    return acc


def load(cache_key, func, verbose=0):
    """Return the gradient function compiled from disk, or None on any miss."""
    if not enabled():
        return None
    try:
        with open(_entry_path(cache_key)) as f:
            record = json.load(f)
        source = record['source']
        entries = record['entries']
        motion = record.get('motion', 'joint')
        mode = record.get('mode', 'reverse')

        from tangent import compile as compile_

        namespace = _build_namespace(func)
        module = compile_.compile_file(source, namespace)

        # Every global the entry functions reference must resolve in the
        # rebuilt namespace or in the module itself; otherwise the source was
        # generated against context we cannot reconstruct here.
        module_names = set(vars(module))
        for name in entries:
            fn = getattr(module, name)
            needed = _referenced_globals(fn.__code__)
            missing = needed - module_names - set(namespace) - set(dir(builtins))
            if missing:
                if verbose >= 1:
                    print('[Cache] Disk entry unusable (unresolved: %s)' % sorted(missing))
                return None

        if mode == 'forward' or motion == 'joint':
            df = getattr(module, entries[0])
        else:
            forward = getattr(module, entries[0])
            backward = getattr(module, entries[1])
            import tangent

            def df(*args, **kwargs):
                _stack = tangent.Stack()
                init_grad = kwargs.pop('init_grad', 1.0)
                forward(_stack, *args, **kwargs)
                dx = backward(_stack, init_grad, *args, **kwargs)
                if len(dx) == 1:
                    (dx,) = dx
                return dx

        df.__tangent_source__ = source
        df.__tangent_entries__ = entries
        df.__tangent_motion__ = motion
        return df
    except Exception:
        # A disk-cache problem must never break differentiation.
        return None


def store(cache_key, df, motion='joint', mode='reverse'):
    """Persist a gradient function's generated source. Best-effort."""
    if not enabled():
        return
    source = getattr(df, '__tangent_source__', None)
    entries = getattr(df, '__tangent_entries__', None)
    if not source or not entries:
        return
    try:
        os.makedirs(cache_dir(), exist_ok=True)
        path = _entry_path(cache_key)
        tmp = path + '.tmp.%d' % os.getpid()
        with open(tmp, 'w') as f:
            json.dump(
                {'source': source, 'entries': entries, 'motion': motion, 'mode': mode},
                f,
            )
        os.replace(tmp, path)
    except OSError:
        pass


def clear():
    """Remove all disk-cache entries."""
    try:
        for name in os.listdir(cache_dir()):
            if name.endswith('.json'):
                os.remove(os.path.join(cache_dir(), name))
    except OSError:
        pass
