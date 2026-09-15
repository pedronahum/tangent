# Investigation: a small explicit IR between gast and code generation

*Status: investigation / design proposal (not a commitment). Written 2026-09-15.*

**Question.** Could Tangent introduce a small, explicit intermediate
representation (IR) between gast and code generation, so that every desugar
becomes "lower this Python construct into IR," and reverse-/forward-/higher-order
differentiation become IR-to-IR transformations?

**TL;DR.** The IR effectively *already exists*, but implicitly: by the time a
function reaches the differentiator it is a small, restricted sublanguage of
gast (single-exit, no classes/lambdas/comprehensions/break/continue, calls
resolved, A-normal form), and `tangent/verify.py` already writes down part of
its contract. A *full* replacement of gast with a distinct IR datatype — and
rewriting the ~hundreds of source-level adjoint/tangent templates across six
backends — is a large, high-risk rewrite with a bounded payoff, because the
**product is readable Python source** and **templates-as-Python is a deliberate
feature**. The recommended path is **incremental formalization**: promote the
existing restricted-gast-plus-invariants into a named, always-checked IR
contract, fold the out-of-band annotation channel into typed structure, and
(optionally, later) migrate the AD visitors to IR-to-IR to unlock the one thing
the current design genuinely cannot get cheaply — **in-memory higher-order
differentiation** (no per-order source round-trip). Do not big-bang it.

---

## 1. What the "IR" is today

There is no separate IR type. gast (the `gast` library's Python-3-normalized
AST) is the single carrier from parse to code generation. But the frontend
already lowers surface Python into a **restricted sublanguage**, and that
sublanguage — not raw Python — is what the AD core consumes.

### 1.1 The frontend already lowers Python into simpler gast

`tangent/passes.py` runs a fixed, ordered pipeline (`run_passes`). Every pass in
it is one of three things, and the majority are exactly "lower a Python construct
into simpler gast":

| Pass | Role | Lowers |
|---|---|---|
| `sentinel_rename` | normalize | a user variable named `d` (collides with the `d[x]` operator) |
| `class_desugar` | **lower** | class-method calls → inlined bodies |
| `lambda_desugar` | **lower** | lambdas → inlined at call sites |
| `return_desugar` | **lower** | early/loop returns → single trailing `return` + flags |
| `chained_assign_desugar` | **lower** | `a = b = e` → simple assignments |
| `enumerate_desugar`, `zip_desugar` | **lower** | iterator loops → `range(len(...))` indexed loops |
| `comprehension_desugar` | **lower** | comprehensions → literals or `tangent.list_append` loops |
| `dict_method_desugar` | **lower** | `d.get(...)` → subscript/ternary |
| `concat_desugar` | **lower** | `concat([a,b])` → varargs helper |
| `list_method_desugar` | **lower** | `xs.append/pop` → `tangent.list_*` rebindings |
| `loop_exit_desugar` | **lower** | `break`/`continue` → boolean guard flags |
| `checkpoint_annotation` | analysis+ | `with tangent.checkpoint():` → loop annotation |
| `ifexp_desugar` (forward only) | **lower** | ternaries → `if`-statements |
| `resolve_calls` | analysis | annotate every `Call` with a resolved `func` |
| `explicit_loop_indexes` | **lower** | `for a in x` (active) → `for i in range(len(x))` |
| `fence` | validate | reject anything not lowered, with a clear error |
| `anf` | **lower** | A-normal form (see below) |

So "every desugar is *lower this Python construct into simpler gast*" is
**already the design.** What is missing is only that the target of the lowering
is *untyped* — it is "gast that happens to satisfy some invariants," not a
distinct IR.

### 1.2 A-normal form is the near-IR shape

`tangent/anf.py` normalizes the tree so every assignment RHS is a single
operation over **trivial operands** (a `Name`, a literal, or `None`), with
subexpressions hoisted to named temporaries, tuple-unpacking made explicit, and
`AugAssign` split. Its own docstring enumerates the allowed statement forms
(`y = x`, `y = f(x1..xn)`, `z = x + y`, `y = -x`, `y[i] = x`, `y = x[i]`,
`z = x, y`, …). After ANF the function is a flat, single-exit `FunctionDef` in
single-operation-per-assignment form. **That is the IR** — it just is not named
or typed as one.

### 1.3 `verify.py` is the written-down fragment of the IR contract

`tangent/verify.py` already turns the passes' implicit assumptions into
*checkable invariants* (opt-in via `TANGENT_VERIFY_IR=1` or
`run_passes(..., verify=True)`), raising `IRInvariantError` on violation:

- `lambda_desugar` → no `Lambda` nodes remain
- `class_desugar` → single flat function (no nested defs)
- `resolve_calls` → every `Call` carries a `func` annotation
- `anf` → A-normal form (trivial operands everywhere)

This is precisely "a documented invariant the IR must satisfy after a given
pass." The 0.3.0 CHANGELOG names it as *"the safe, shippable realization of 'one
explicit IR with a documented invariant'; a wholesale single-rewrite core …
remain future work."* This investigation is about that future work.

### 1.4 What is *not* in the tree: the two side-channels

Two kinds of semantics do **not** live in the gast structure and are the real
subject of an "explicit IR":

1. **The annotation channel.** `tangent/annotations.py` stores a `_tangent`
   dict on each node and appends `_tangent` to `node._fields` so it *survives
   the gast↔ast source round-trip*. `FIXED_ANNOTATIONS` carries semantic facts
   between passes: `func` (resolved callee), `adjoint_var`/`temp_adjoint_var`
   (gradient↔primal linkage), `pri`/`adj` (primal↔adjoint back-refs),
   `pri_call`/`adj_call`, `push`/`pop`/`push_func` (tape linkage), `active_in`
   (activity), `output_arity`, `pre_anf` (source for comments), `tangent_keep`
   (DCE protection), `force_checkpoint`. The tree holds the *syntax*; the
   annotation side-table holds most of the *semantics*.

2. **The `d[x]` gradient sentinel.** Adjoint/tangent rules are written as
   ordinary Python functions containing `d[x]` ("gradient of x"), represented as
   a `Subscript` of `Name('d')` and reinterpreted by
   `template.ReplaceGradTransformer` at splice time. It is a source-level
   convention, not a node type — which is *why* `sentinel_rename` has to exist.

An explicit IR is, concretely, the act of moving (1) into typed fields and (2)
into a first-class `GradOf` node.

---

## 2. What an explicit IR would (and would not) change

**Would formalize / improve:**

- **Type safety.** "gast that may or may not be ANF, with annotations that may
  or may not be present" becomes a typed structure whose construction *is* the
  invariant. Pass signatures become honestly `IR -> IR`.
- **Metadata in structure.** `func`, `active`, `adjoint_var`, `op_id`,
  `output_arity`, `comment` become fields, not a `_tangent` dict smuggled
  through `gast_to_ast` via the `_fields` trick.
- **`d[x]` as a real node** (`GradOf(name)`), removing the sentinel collision
  and `sentinel_rename` entirely.
- **In-memory higher-order** (the big one — see §3.3).

**Would *not* change (hard constraints):**

- **The product is readable Python source.** Codegen is `gast → ast.unparse`.
  An IR must still lower to gast/ast for `to_source`. The IR sits *between*
  gast-in and gast-out; it does not remove the source boundary, it moves it to
  the edges.
- **Templates are readable Python.** ~Hundreds of adjoint/tangent rules across
  `grads.py`, `tangents.py`, `elementwise_rules.py`, and the six backend
  extension modules are written as Python functions with `d[x]`, parsed via
  `inspect.getsource` + `template.replace`. That readability is a selling point
  ("rules are just Python"). An IR core must either keep them as source and
  *lower their spliced output into IR at the boundary*, or rewrite all of them
  in an IR-builder DSL (large, and it sacrifices the feature).

---

## 3. What makes a full rewrite expensive

### 3.1 The template layer is the load-bearing wall

Reverse mode (`reverse_ad.ReverseAD`) and forward mode (`forward_ad.ForwardAD`)
do not compute derivatives directly; they **splice source templates**:
structural templates (`dfunction_def`, `for_`/`dfor_`, `while_`/`dwhile_`,
`if_`/`dif_`, plus checkpointed variants) and per-op rules (`grads.adjoints`,
`tangents.tangents`, keyed by operator type or function object). Each rule is a
Python function bound to the call site via `inspect.signature(...).bind` and
`template.replace`. Any IR core has to interoperate with this. The cheapest
interop is an adapter: keep templates as Python source, `template.replace` as
today, then `to_ir(spliced_gast)` at the boundary. A pure-IR rule DSL is the
expensive option.

### 3.2 The AD visitors carry an unusual contract

`ReverseAD.visit` returns a `(primal, adjoint)` **pair** per node and threads
the tape (`push`/`pop` with `op_id`s), then `joint`/`split` motion assembles one
or two functions, with `_fix` (via `fixes.py`) and `store_state` doing CFG-driven
cleanup and split-motion state transfer. Reproducing this as IR-to-IR is
mechanical but touches the most intricate ~1800 lines in the codebase. It must
also reproduce the three CFG analyses (`cfg.Active`/`Defined`/`ReachingDefinitions`)
either over the IR or over a gast view of it.

### 3.3 Higher-order currently round-trips through generated source

This is the sharpest finding. Differentiating a gradient a second time
**re-parses the generated Python**: `compile.compile_file` writes the first
gradient to a real temp `.py` file and imports it (so `inspect`/`pdb` work), and
the second `tangent.grad` calls `quoting.parse_function` →
`inspect.getsource` on that file, running the *entire* pipeline again on the
generated text. `unwrap_function` + `__globals__` merging exist specifically to
recover the namespace across this round-trip.

- An IR-to-IR core *could* keep the transformed IR in memory and differentiate
  it directly, eliminating the per-order parse — a real speed and robustness
  win for `grad(grad(...))`.
- **But** the readable-source product means the IR must still emit source at the
  *end* of each order for `verbose`, `explain`, the disk cache, and
  `__tangent_source__`. So higher-order-in-IR removes the *re-parse*, not the
  *emit*. And the namespace recovery that `unwrap_function` does through the
  function object would have to be carried on the IR instead.

---

## 4. A concrete small IR (if pursued)

The IR is genuinely small — it mirrors the post-ANF sublanguage. ~15 node kinds:

```
Module(functions)
FunctionDef(name, params, body, output_arity, motion)
# statements (single-operation)
Assign(target: Name | Subscript | Attribute, value: Expr)
For(index, iterable, body)          # always range(len(...)) form
While(test, body)
If(test, body, orelse)
Return(values: list[Name])          # single-exit
TapePush(stack, value, op_id) / TapePop(stack, target, op_id)
InsertGradOf(var, body)             # user gradient surgery
# expressions (trivial operands only)
Name(id, *, resolved_grad_of=None)
Const(value)
Prim(op | func, args: list[Name|Const], *, resolved_func)   # BinOp/UnaryOp/Call unified
Subscript(value: Name, index: Name|Const)
Attribute(value: Name, attr)
Tuple(elts: list[Name])
GradOf(var)                         # first-class d[x]
```

Metadata that is annotations today becomes fields: `resolved_func` on `Prim`,
`op_id` on tape nodes, `output_arity`/`motion` on `FunctionDef`,
`active`/`adjoint_var` links where needed, `comment`. Adapters `from_gast` /
`to_gast` bridge to the existing parser and to `ast.unparse` codegen. Templates
stay Python source; their spliced gast is lowered with `from_gast` at the
boundary.

---

## 5. Recommended path (incremental, low-risk first)

Ordered by value-to-risk. Each phase stands alone and keeps the ~80k-case suite
green.

- **Phase 1 — formalize the contract that already exists (cheap, high value).**
  Make `verify.py` always-on in CI and in a debug flag; expand its invariants to
  cover the passes that currently have none (e.g. after `loop_exit_desugar`: no
  `Break`/`Continue`; after `return_desugar`: single trailing return; after
  `explicit_loop_indexes`: every active loop is `range(len(...))`). Introduce a
  thin `IRModule` marker type that *wraps* the gast module plus its invariant
  level, so pass and AD signatures read `IRModule -> IRModule` even while the
  payload is still gast. This buys the documentation, the type-level honesty,
  and pass-drift protection with almost no risk.

- **Phase 2 — fold the side-channels into structure.** Give `d[x]` a first-class
  `GradOf` node (retiring `sentinel_rename`), and move the load-bearing
  annotations (`func`, `op_id`, `output_arity`, `active`) onto typed wrappers of
  the ~15 node kinds, with `from_gast`/`to_gast` adapters. Keep templates as
  Python source, lowered at the splice boundary. This is where a real IF distinct
  IR earns its keep, and it is testable node-kind by node-kind.

- **Phase 3 — IR-to-IR AD and in-memory higher-order (largest, do last).**
  Reimplement `ReverseAD`/`ForwardAD` as IR→IR passes over the typed IR, and
  keep the IR in memory across differentiation orders (differentiate the first
  gradient's IR directly instead of re-parsing its source), emitting source only
  at the end of each order. This unlocks the one capability the current
  architecture cannot get cheaply. It is also the riskiest step and should only
  begin once Phases 1–2 have made the contract and the node set solid.

---

## 6. Recommendation

**Feasible: yes. Advisable as a big-bang parallel-IR rewrite: no.** The codebase
is unusually well-positioned for this — the pass registry, `PassContext`, the
ANF contract, and `verify.py` are already the skeleton of "one explicit IR with a
documented invariant." But two facts bound the payoff and should temper ambition:
the deliverable is *readable source* (so the IR cannot remove the source
boundary, only move it to the edges), and *templates-as-Python* is a feature
worth keeping (so the IR should adapt to source templates, not replace them).

The highest value-to-risk action is **Phase 1**: promote the existing
restricted-gast-plus-invariants to an always-checked, named contract. It delivers
most of the "one IR" benefit — a single documented shape, enforced at every pass
boundary, that stops the "works until two passes disagree" failure mode — at a
fraction of the cost, and it lays the exact groundwork a later Phase 2/3 would
build on. Reserve the full IR-to-IR core (Phase 3) for when in-memory
higher-order differentiation becomes a concrete need, since that is the single
benefit the current source-round-trip design cannot otherwise achieve.
