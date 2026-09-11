# How Tangent Works

Tangent is a **source-to-source** differentiator: it transforms the abstract
syntax tree of your function into the AST of a new function that computes
derivatives, then compiles that AST back to ordinary Python. This page walks
the pipeline.

## The pipeline

```
your function
  │  inspect.getsource + parse
  ▼
desugaring passes          (tangent/passes.py - an ordered, extensible registry)
  │  classes → calls, lambdas → defs, early returns → single exit,
  │  break/continue → guard flags, comprehensions → loops,
  │  xs.append(v) → xs = tangent.list_append(xs, v), ...
  ▼
call resolution + fence    (reject unsupported constructs with clear errors)
  ▼
A-normal form              (every intermediate gets a name)
  ▼
reverse or forward AD      (tangent/reverse_ad.py, tangent/forward_ad.py)
  │  each statement is replaced by its primal + adjoint, driven by a
  │  library of per-operation gradient templates (tangent/grads.py)
  ▼
optimization               (dead-code elimination, constant folding,
  │                         assignment propagation - tape-aware)
  ▼
compile + return           an ordinary Python function
```

## Templates, not rules engines

Every differentiable operation has a small **template** written in Python.
The adjoint of `numpy.tanh`, in its entirety:

```python
@adjoint(numpy.tanh)
def tanh(y, x):
    d[x] = d[y] * (1.0 - y * y)
```

`d[x]` means "the gradient flowing to `x`". During differentiation the
template is spliced into the generated code with the call site's actual
variable names substituted. Registering a new gradient is just writing one of
these — see the [API Reference](api.md).

## The tape

Reverse mode needs values from the forward pass. Where an intermediate is
overwritten (loops, reassignment), the generated primal pushes the old value
onto an explicit stack — visible in the generated source as
`tangent.push(_stack, x, 'op_id')` — and the adjoint pops it back. Push/pop
pairs are balanced per iteration, which is what makes two of Tangent's
distinctive features sound:

- **Tape-aware optimization** — dead-code elimination removes a push only
  together with the pop that consumes it, so optimized code never corrupts
  higher-order derivatives.
- **[Segment checkpointing](checkpointing_user_guide.md)** — eligible loops
  run untaped, snapshotting loop state every √n iterations; the backward pass
  replays one segment at a time at O(√n) peak memory.

## Correctness discipline

Three mechanisms keep gradients trustworthy:

1. **The fence.** Constructs Tangent cannot differentiate are rejected at
   compile time with an actionable message (and usually a suggested
   rewrite) — never compiled into something silently wrong.
2. **Finite-difference oracles.** Every registered gradient is tested against
   numerical differentiation; the parametrized suite runs ~79,000 cases
   across both modes, both motions, and second/third order.
3. **Mode consistency.** A supported construct behaves the same in reverse
   mode, forward mode, and higher order. An op without a registered
   derivative raises `ReverseNotImplementedError` /
   `ForwardNotImplementedError` in *both* modes.

## Two modes, two motions

- **Reverse mode** (`tangent.grad`, `mode='reverse'`) computes
  vector-Jacobian products — efficient for many-inputs → scalar losses.
- **Forward mode** (`mode='forward'`) computes Jacobian-vector products —
  efficient for few inputs, and composes with reverse for Hessian-vector
  products.
- **Joint vs. split motion**: the generated primal and adjoint can live in
  one combined function (default) or as separate forward/backward functions
  sharing an explicit tape (`motion='split'`).
