"""Command-line entry point: `python -m tangent doctor`.

Diagnoses the most common installation problems - above all the collision
with Google's abandoned 2017 `tangent` package, which installs the same
`tangent/` module and silently shadows or breaks this one.
"""

from __future__ import absolute_import

import os
import sys


def _doctor():
    import importlib.metadata as md

    ok = True
    print('tangent doctor')
    print('=' * 60)

    # 1. Which distributions claim the `tangent` module?
    dists = {}
    for name in ('tangent-ad', 'tangent'):
        try:
            dists[name] = md.version(name)
        except md.PackageNotFoundError:
            pass
    if 'tangent' in dists:
        ok = False
        if dists['tangent'].startswith('0.1.'):
            print('✗ PyPI package `tangent` %s is installed.' % dists['tangent'])
            print("  That is Google's unmaintained 2017 release; it provides the")
            print('  same `tangent/` module and the two installations clobber')
            print('  each other. Fix:')
            print('      pip uninstall tangent')
            if 'tangent-ad' not in dists:
                print('      pip install tangent-ad')
        else:
            print('✗ A distribution named `tangent` (%s) is installed -' % dists['tangent'])
            print('  probably a leftover install of this fork from before it')
            print('  was renamed to tangent-ad. Clean up with:')
            print('      pip uninstall tangent')
    if 'tangent-ad' in dists:
        print('✓ tangent-ad %s installed' % dists['tangent-ad'])
    elif 'tangent' not in dists:
        print('~ running from a source checkout (no installed distribution)')

    # 2. Import and version.
    import tangent

    print('✓ import tangent -> version %s (%s)' % (tangent.__version__, tangent.__file__))
    if 'tangent' in dists and dists['tangent'].startswith('0.1.'):
        imported_old = tangent.__version__.startswith('0.1.')
        if imported_old:
            print('✗ The OLD 2017 package is the one being imported!')

    # 3. Backends.
    status = tangent.backend_status()
    for backend, state in sorted(status.items()):
        mark = '✓' if state == 'available' else ('✗' if state.startswith('broken') else ' ')
        print('%s backend %-10s %s' % (mark, backend, state))
        if state.startswith('broken'):
            ok = False

    # 4. Disk cache.
    from tangent import disk_cache

    if disk_cache.enabled():
        path = disk_cache.cache_dir()
        try:
            entries = len([f for f in os.listdir(path) if f.endswith('.json')])
        except OSError:
            entries = 0
        print('✓ disk cache at %s (%d entries)' % (path, entries))
    else:
        print('~ disk cache disabled (TANGENT_DISK_CACHE=0)')

    # 5. Smoke gradient (from a real file: `python -c` functions have no source).
    import tempfile

    src = 'def _doctor_probe(x):\n    return x * x * x\n'
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
        f.write(src)
        path = f.name
    import importlib.util

    spec = importlib.util.spec_from_file_location('_tangent_doctor_probe', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['_tangent_doctor_probe'] = mod
    spec.loader.exec_module(mod)
    result = tangent.grad(mod._doctor_probe)(2.0)
    if result == 12.0:
        print('✓ smoke gradient: d/dx x^3 at 2.0 = %s' % result)
    else:
        ok = False
        print('✗ smoke gradient WRONG: got %s, expected 12.0' % result)

    print('=' * 60)
    print('all good' if ok else 'problems found - see above')
    return 0 if ok else 1


def main(argv):
    if len(argv) >= 1 and argv[0] == 'doctor':
        return _doctor()
    print('usage: python -m tangent doctor')
    return 2


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
