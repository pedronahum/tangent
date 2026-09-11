"""Source capture for environments where inspect.getsource fails.

`@tangent.function` grabs a function's source at definition time, and
`tangent.grad(f, source=...)` / `@tangent.function(source=...)` accept it
explicitly - so Tangent works in the REPL, in `exec`-defined functions, and
after the source file is gone.
"""

import importlib.util
import os
import tempfile

import pytest

import tangent


def test_exec_function_with_explicit_source():
    src = 'def cube(x):\n    return x * x * x\n'
    ns = {}
    exec(src, ns)  # no retrievable source
    g = tangent.grad(ns['cube'], source=src)
    assert g(2.0) == pytest.approx(12.0)


def test_decorator_with_explicit_source():
    ns = {}
    exec('def sq(x):\n    return x * x\n', ns)
    sq = tangent.function(ns['sq'], source='def sq(x):\n    return x * x\n')
    assert tangent.grad(sq)(3.0) == pytest.approx(6.0)


def test_decorator_factory_form():
    ns = {}
    exec('def sq(x):\n    return x * x\n', ns)
    decorate = tangent.function(source='def sq(x):\n    return x * x\n')
    sq = decorate(ns['sq'])
    assert tangent.grad(sq)(4.0) == pytest.approx(8.0)


def test_captured_source_survives_file_deletion():
    # The strongest demonstration: capture at import, delete the file, and the
    # gradient still works while a plain function's does not.
    src = (
        'import tangent\n\n'
        '@tangent.function\n'
        'def captured(x):\n'
        '    return x ** 3\n\n'
        'def plain(x):\n'
        '    return x ** 3\n'
    )
    f = tempfile.NamedTemporaryFile('w', suffix='.py', delete=False)
    f.write(src)
    f.close()
    spec = importlib.util.spec_from_file_location('_capture_tmp', f.name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    os.remove(f.name)  # inspect.getsource can no longer read it

    assert tangent.grad(mod.captured)(2.0) == pytest.approx(12.0)
    with pytest.raises(Exception):
        tangent.grad(mod.plain)(2.0)


def test_captured_source_strips_decorator_line():
    # The stored source must start at `def`, not at the @tangent.function line.
    import tangent.capture as capture

    src = '@tangent.function\n@some_other\ndef f(x):\n    return x\n'
    stripped = capture._strip_decorators(src)
    assert stripped.lstrip().startswith('def f(')


def test_error_message_suggests_capture():
    ns = {}
    exec('def nosrc(x):\n    return x * x\n', ns)
    try:
        tangent.grad(ns['nosrc'])(2.0)
        assert False, 'expected a source error'
    except Exception as e:
        assert '@tangent.function' in str(e)


def test_decorated_function_still_callable():
    @tangent.function
    def f(x):
        return x * 2.0

    assert f(3.0) == 6.0  # decorator does not wrap/alter the function
