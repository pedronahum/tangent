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
- **Gradient checkpointing (limited)**: `grad(f, checkpoint=True)`
  automatically checkpoints `for i in range(n)` loops (constant, zero-based
  `n` ≥ 100) but only stores the loop target selectively — measured overall
  tape-memory reduction is ~3%; also a manual `checkpointed_loop` helper for
  O(√n)-memory forward passes (gradients do not flow through it). Automatic
  checkpointing of arbitrary loops is not implemented
  (`grad_with_checkpointing` raises `NotImplementedError`). See
  `docs/checkpointing_user_guide.md`.
- **Optimizations**: dead-code elimination, common-subexpression elimination,
  algebraic simplification (SymPy), and a straight-line coarsening prototype.
- **Tooling**: gradient-flow and computation-graph visualization,
  gradient-function caching, clear rejection errors for non-differentiable
  syntax, a finite-difference oracle for backend adjoints.
- **Modernization**: Python 3.9–3.13, gast 0.6/0.7, NumPy 2.x, pyproject-based
  packaging, GitHub Actions CI, 75k+ parameterized tests.

### Added in 0.2.0

- **Differentiable list building**: `xs.append(v)` and `v = xs.pop()` are now
  differentiated (previously `.append()` was silently treated as
  non-differentiable, returning zero gradients). They are desugared into
  rebindings through three new primitives (`tangent.list_append` /
  `list_last` / `list_init`) whose adjoints are written in terms of each
  other, so forward mode and second/third derivatives work, including appends
  inside dynamic-length loops. In-place mutations that cannot be expressed as
  a rebinding (`extend`/`insert`/`remove`/`sort`/`reverse`, or append through
  an attribute/subscript) are rejected with a clear error instead of silently
  dropping gradients.
- **Compile-time scalability**: functions a few hundred statements long used
  to crash with `RecursionError` before they could be differentiated (the CFG
  dataflow walk recursed once per statement); the walk is now an iterative
  worklist, and a 400-statement chain compiles fine. The optimizer's
  fixpoints report changes directly instead of serializing the entire AST
  with `gast.dump` twice per iteration, dead-code elimination peels cascading
  dead chains in a single dataflow analysis instead of one analysis per
  layer, and the post-advanced-DCE cleanup round only runs when advanced DCE
  changed something. Compilation of a 150-statement function dropped ~33%,
  and the full test suite runs ~25% faster.
- **Real gradient checkpointing**: `tangent.grad(f, checkpoint=True)` now
  implements segment (√n) recomputation — eligible loops run untaped with a
  snapshot of the loop-carried state every ceil(√n) iterations, and the
  backward pass replays one segment at a time. Peak tape memory drops from
  O(n) to O(√n) with gradients *identical* to the fully-taped path (measured:
  49.2 MB → 1.6 MB, 96.8%, on a 2000-iteration loop with 1000-float state;
  `benchmarks/checkpointing_memory.py`). The previous implementation stored
  only the loop target selectively (~0% measured reduction on state-carrying
  loops) and was restricted to zero-based `range(n)`; any constant-bound
  `range(start, stop, step)` is now eligible. First-order reverse mode only.
- **Mode consistency**: every corpus construct now behaves the same in every
  differentiation mode. The last "first-order-only" gaps are closed - multi-
  output functions (`return 2*a, a`; polar transforms) and subscript-scatter
  loops now pass forward mode and reverse-over-reverse too - and the audited
  harness exclusion list contains only by-design entries. Builtin casts and
  comparisons gained symmetric rules in both modes: `float()` (identity),
  `int()` (zero derivative a.e.), and forward-mode twins for `abs`/`min`/`max`.
  Forward mode also seeds loop-counter tangents, fixing a NameError on any
  `x * i` arithmetic inside loops.
- **Clean errors restored for unregistered ops**: NumPy 2.x wraps most public
  functions in `_ArrayFunctionDispatcher`, which the unimplemented-op scan did
  not recognize - the unimplemented sets were nearly empty, so calls like
  `np.sort(x)` recursed into NumPy's own source and crashed with an opaque
  `AttributeError` instead of raising `ReverseNotImplementedError` /
  `ForwardNotImplementedError`. The scan now recognizes dispatchers, and
  forward mode's no-source fallback raises the standard clean error instead of
  a bare `ValueError`.
- **`break` and `continue`**: both now differentiate exactly (historically
  `break` *miscomputed* gradients via tape replay, then both were rejected). A
  new `loop_exit_desugar` pass lowers them into guard flags before
  differentiation: `continue` becomes a per-iteration skip flag guarding the
  rest of the body; `break` adds a loop-level flag that a `while` folds into
  its condition and a `for` uses to skip remaining iterations. Works with
  data-dependent exits, nested loops, forward mode, both motions, and second
  order. Loops with an `else` clause remain rejected (`while/else` now rejected
  explicitly, like `for/else`).
- **`return` inside loops**: lowered into an assignment plus a returning flag
  plus `break` (which the break lowering then handles), with the exit
  propagated past each enclosing loop and the final return lifted by the
  single-exit transform. Previously rejected.
- **Iteration over computed sequences**: `for v in x * 2.0`, `for v in
  np.flip(x)`, and `for v in [x, x * 2]` (a literal of active values) now
  differentiate - the iterable expression is hoisted into a named intermediate
  and indexed, the same rewrite long applied to named sequences. Previously
  these loops were neither rewritten nor rejected and gradients through the
  loop variable were silently dropped. Tuple-unpacking targets (`for a, b in
  pairs`) work; set literals of active values are rejected (no stable order).
- **List comprehensions over runtime iterables**: `[f(v) for v in xs]` with a
  runtime `xs` is now lowered into an indexed loop built on
  `tangent.list_append` (previously rejected; before that, silently wrong).
  Runtime `if` filters and nested comprehensions are supported; multiple
  generators and tuple targets are still rejected cleanly.
- **Container (pytree) return values in reverse mode**: functions returning
  dicts, lists, or nested containers of arrays can now be differentiated
  (previously a `KeyError` crash). The default seed expands to a matching
  pytree of ones (the gradient of the sum of all leaves), a caller-supplied
  seed of the same structure is used as the cotangent, and second derivatives
  through container *arguments* work. A structurally mismatched seed raises a
  clear `ValueError` instead of being silently replaced.
- **New NumPy derivatives** (reverse and forward mode, validated against a
  finite-difference oracle): the ufunc spellings `np.add`, `np.subtract`,
  `np.divide`, `np.negative`, `np.power`, `np.float_power`; `np.arctan2`,
  `np.hypot`, `np.logaddexp`, `np.arcsinh`, `np.arccosh`, `np.arctanh`,
  `np.exp2`, `np.cbrt`, `np.fmax`, `np.fmin`; the shape ops `np.cumsum`,
  `np.flip`, `np.ravel`, `np.swapaxes`, `np.moveaxis`, `np.tile`,
  `np.repeat`, `np.roll`; and `np.linalg.solve` plus `np.linalg.norm`
  (default 2-norm/Frobenius).
- **Backend parity**: the `@` matmul operator now differentiates for PyTorch
  (and Keras); softmax/log_softmax adjoints for PyTorch, Keras, JAX and
  TensorFlow; `concatenate`/`stack` differentiate across NumPy, JAX,
  TensorFlow, PyTorch and Keras (previously JAX-only); elementwise gaps
  closed across backends (e.g. torch `expm1`/`rsqrt`, jax
  `reciprocal`/`log1p`/`expm1`/`exp2`, keras log/exp family,
  `tf.math.reciprocal` — whose old TF-1-only spelling registered nothing on
  TF 2.x). Unary elementwise rules are now generated from one shared
  backend-neutral table, so every table op has both reverse- and forward-mode
  rules on every backend.
- **Forward mode catches up**: the extended NumPy and TensorFlow modules
  previously registered *zero* forward-mode tangents; they now cover their
  op catalogs (abs, clip, where, min/max/prod, trig, linalg.inv, trace,
  softmax, concat/stack, ...).
- `tangent.backend_status()` reports each optional backend as
  `'available'`, `'not installed'`, or `'broken: <error>'`.
- `tangent.passes`: a documented pass registry for the desugaring pipeline
  with `register_pass(..., before=/after=)` and `unregister_pass` as an
  extension hook for inserting custom frontend passes.
- Property-based tests (hypothesis) drive a corpus sample against the
  finite-difference oracle; CI now also runs the tinygrad-backed tests, ruff
  lint + format checks, and a coverage floor.

### Changed in 0.2.0

- **`import tangent` is silent.** Missing optional backends (jax, torch,
  tensorflow, keras, tinygrad) no longer emit a wall of `UserWarning`s and
  stdout banners; they are logged at DEBUG level on the `'tangent'` logger.
  A `UserWarning` is kept only when a backend package is installed but its
  extensions fail to load (a broken install is actionable). Use
  `tangent.backend_status()` to see what loaded.
- **Gradient-cache keys now include compilation settings.** `grad(f)`
  followed by `grad(f, optimizations={...})` (or a different
  `checkpoint_config`/`grad_config`) previously returned the *same* cached
  function; such calls now compile and cache separately, so previously
  colliding call sites will each trigger their own (correct) compilation.
  Cache hits are also much cheaper: source hashes are memoized instead of
  re-reading the source file on every lookup.
- **Checkpointing consolidated to what demonstrably works**:
  `grad(f, checkpoint=True)` (zero-based constant `range(n)` loops) and the
  manual `checkpointed_loop` forward-pass helper. `grad_with_checkpointing`
  now raises `NotImplementedError` at wrapper-creation time with an
  actionable message. `checkpoint=True` now composes with `optimized=True`
  (optimization is no longer force-disabled).
- List/set/dict comprehensions over dynamic iterables are rejected with a
  clear `TangentParseError` at transform time. The old `.append()`-loop
  fallback crashed in return position and silently produced zero gradients in
  assignment position. Comprehensions over compile-time-constant iterables
  are still unrolled and differentiate correctly.
- `numpy`/`tf` `concatenate`/`stack` over a dynamically built list raise
  `NotImplementedError` instead of silently returning zero gradients.
- The codebase is uniformly formatted with `ruff format` (4-space indent,
  100 columns), enforced in CI; the reformat commit is listed in
  `.git-blame-ignore-revs`.
- Docs pruned: ~15 stale point-in-time progress reports (phase "COMPLETE"
  writeups, outdated test-status snapshots) were deleted or folded into the
  living feature guides; remaining status claims match measured reality.

### Fixed in 0.2.0

- `tangent.autodiff(f, optimizations=...)`, `checkpoint_config=...` and
  `grad_config=...` raised `TypeError`: the caching wrapper re-declared a
  narrower parameter list than the real `autodiff`. The full signature is
  restored.
- Wrong or crashing *higher-order* derivatives with `optimized=True`:
  dead-code elimination trusted per-op-id tape annotations that are not
  unique in differentiated gradient code, so it could delete the wrong tape
  push (second derivatives read stale primal values, or tripped the stack
  op-id assertion). DCE now computes its own balanced push/pop pairing and
  never removes an ambiguous pair; the two long-standing expected failures
  in reverse-over-reverse are eliminated.
- Multi-output functions with `check_dims=False` crashed (or could get a
  wrong default seed): the output arity was inferred by scanning generated
  code for the shape-check assert. It is now recorded as an annotation during
  the reverse-mode transformation.
- `grad(..., checkpoint=True)` could produce wrong gradients for
  `for i in range(start, stop)` loops with `start != 0`: the checkpointed
  adjoint reconstructs the loop target as the zero-based iteration index.
  Such loops now fall back to the standard (non-checkpointed) templates.
- Broadcasting operands of `np.multiply` and `np.maximum` received
  wrongly-shaped gradients (missing unbroadcast reduction).
- `ValueError` for mismatched forward-mode derivative arguments was raised
  incorrectly (a `TypeError` masked the real message).
- The stack variable in the subscript-assignment adjoint template was bound via
  a typo'd keyword (`_stack_`), working only because the default name matched.
- Package metadata now identifies this fork (maintainer, repository URLs) and
  exposes `tangent.__version__`.

### Removed in 0.2.0

- An unreachable, unfinished automatic-checkpointing pipeline
  (`tangent/checkpointing/`, `tangent/analysis/`, `tangent/preprocessing/`),
  the placeholder `checkpointed_backward` (its default path returned the
  incoming gradient unchanged), and `checkpointed_grad` (its "simplified
  backward" returned the single-step gradient at the final state, not the
  loop gradient).
- The vendored Python 2 `funcsigs` backport, in favor of
  `inspect.signature`.
