# Tangent 2 — a reproducible writeup

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pedronahum/tangent/blob/master/paper/Tangent2.ipynb)

*Tangent 2: Multi-Backend Source-to-Source Automatic Differentiation in Modern
Python.*

A short paper on this fork of Google's Tangent — written so the code in it
actually runs. Every result is backed by executable code, and there are two
ways to reproduce all of them.

## Reproduce in the browser (one click)

Open **[`Tangent2.ipynb`](https://colab.research.google.com/github/pedronahum/tangent/blob/master/paper/Tangent2.ipynb)**
in Colab. The first cell installs `tangent-ad`; run the notebook top to bottom.
The notebook *is* the paper — prose plus runnable cells with real outputs.

## Reproduce from a shell (a few seconds)

```bash
pip install "tangent-ad[symbolic]>=0.4.0"     # JAX / PyTorch are optional, auto-detected
python paper/reproduce.py              # runs and asserts every quantitative claim
```

`reproduce.py` prints one line per claim and exits non-zero if any no longer
holds, so it doubles as a regression check on the paper.

## Files

| File | What it is |
|---|---|
| [`Tangent2.ipynb`](Tangent2.ipynb) | The runnable paper (executed, with outputs; Colab-ready) |
| [`PAPER.md`](PAPER.md) | The same paper as static Markdown, for reading on GitHub |
| [`reproduce.py`](reproduce.py) | One script that reproduces and asserts every numeric claim |
| [`requirements.txt`](requirements.txt) | Exact dependencies to reproduce |

The notebook and `PAPER.md` are generated from a single source, so they cannot
drift; `reproduce.py` shares the same code.

## Abstract

Tangent works by **source-to-source transformation**: it reads a Python
function and emits a new Python function that computes its gradient — ordinary
code you can read, debug, and edit, rather than an opaque graph or tape. This
writeup describes *Tangent 2*, a modernized fork of Google's 2017 Tangent: one
differentiation API across six array backends (NumPy, JAX, TensorFlow, PyTorch,
Keras 3, tinygrad), optional lowering of the readable adjoint to a fused backend
kernel, memory-bounded differentiation of long simulations (adjoint ODEs and
√n / online checkpointing), an opt-in tape-liveness pass, symbolic straight-line
coarsening, a statically typed public API, and a debuggability toolkit
(`explain`, `source_map`, `insert_grad_of`). Its niche is making the
derivatives of gnarly, loop- and branch-heavy scientific and quantitative code
legible — not replacing tracing AD for large tensor programs.
