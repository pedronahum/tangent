# Changelog

## 0.2.0 (unreleased)

First version of this fork to be numbered ahead of Google's final PyPI release
(0.1.9, December 2017), so that installs from this repository upgrade cleanly
over the abandoned upstream package.

### Highlights since upstream 0.1.9

- **New backends**: JAX, TensorFlow 2.x (eager), PyTorch, Keras 3
  (backend-agnostic `keras.ops`), and tinygrad (method-based `Tensor.*` API,
  including conv2d/max_pool2d/avg_pool2d/layernorm/batchnorm).
- **Higher-order differentiation**: second and third derivatives, Hessian-vector
  products, forward-over-reverse.
- **Language coverage**: classes and inheritance, closures, lambdas,
  comprehensions, `enumerate`/`zip`, f-strings, chained and augmented
  assignment (including subscript targets), early returns, tuple unpacking,
  the `@` operator, pytree (container) arguments — tracked in
  `docs/features/PYTHON_FEATURE_SUPPORT.md` and enforced by
  `tests/test_feature_matrix.py`.
- **Gradient checkpointing** with automatic checkpoint placement.
- **Optimizations**: dead-code elimination, common-subexpression elimination,
  algebraic simplification (SymPy), and a straight-line coarsening prototype.
- **Tooling**: gradient-flow and computation-graph visualization,
  gradient-function caching, clear rejection errors for non-differentiable
  syntax, a finite-difference oracle for backend adjoints.
- **Modernization**: Python 3.9–3.13, gast 0.6/0.7, NumPy 2.x, pyproject-based
  packaging, GitHub Actions CI, 76k+ parameterized tests.

### Fixed in 0.2.0

- `ValueError` for mismatched forward-mode derivative arguments was raised
  incorrectly (a `TypeError` masked the real message).
- The stack variable in the subscript-assignment adjoint template was bound via
  a typo'd keyword (`_stack_`), working only because the default name matched.
- Removed the vendored Python 2 `funcsigs` backport in favor of
  `inspect.signature`.
- Package metadata now identifies this fork (maintainer, repository URLs) and
  exposes `tangent.__version__`.
