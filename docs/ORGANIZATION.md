# Documentation Organization

How the documentation tree is laid out, and where a new document belongs.

## Layout

```
/tangent
├── README.md                    # Main documentation (features, backends, limitations)
├── CONTRIBUTING.md              # Contribution guidelines
└── docs/
    ├── INDEX.md                 # Full documentation index
    ├── README.md                # Short docs overview
    ├── ORGANIZATION.md          # This file
    ├── checkpointing_user_guide.md
    ├── features/                # Feature reference docs
    │   └── PYTHON_FEATURE_SUPPORT.md   # ← canonical feature matrix
    ├── optimizations/           # Optimization deep dives
    ├── benchmarks/              # Benchmark documentation and results
    ├── bugs/                    # Analyses of past bugs
    ├── development/             # Historical roadmaps and plans
    └── plans/                   # Historical modernization plans
```

## Principles

- **The root `README.md` and `docs/features/PYTHON_FEATURE_SUPPORT.md` are the
  living references.** They describe *current* behavior — what works, what
  doesn't, and known limitations — and are kept accurate as the code changes.
  When any other document disagrees with them, they win.
- **`development/` and `plans/` are historical.** They record intent at the
  time of writing and are not updated to track the code.
- **Point-in-time progress reports don't live here.** "Implementation
  complete" write-ups, test-run snapshots, and launch summaries rot quickly;
  their durable content belongs in the living references, and the rest lives
  in git history. (A number of such files were folded into
  `PYTHON_FEATURE_SUPPORT.md` and deleted — see git history if you need
  them.)
- **Benchmarks are dated measurements.** Documents under `benchmarks/` name
  the environment and date they were measured in.

## Where a new doc goes

| Kind of document | Location |
|---|---|
| What a language feature does / doesn't support | Fold into `features/PYTHON_FEATURE_SUPPORT.md` (add a focused `features/*.md` only for genuinely deep material) |
| Backend gradient coverage | Root `README.md` (Backend Support section) |
| Optimization design and analysis | `optimizations/` |
| Benchmark methodology and results | `benchmarks/` |
| Bug post-mortem worth keeping | `bugs/` |
| Design proposal / roadmap | `development/` or `plans/` |

After adding or removing a document, update `docs/INDEX.md` (and
`docs/README.md` if it is user-facing).
