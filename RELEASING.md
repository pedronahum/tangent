# Releasing tangent-ad

The release pipeline is automated: pushing a `vX.Y.Z` tag builds, verifies,
publishes to PyPI, and creates a GitHub Release
(`.github/workflows/release.yml`). Two one-time setup steps must be done in
web UIs first.

## One-time setup

### 1. PyPI: account + Trusted Publisher (no API tokens needed)

The workflow authenticates via [Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
(OIDC): PyPI trusts release runs coming from this exact repo/workflow, so no
secret is stored anywhere.

1. Create/log into your account at <https://pypi.org> (enable 2FA — required
   for publishing).
2. Since `tangent-ad` does not exist on PyPI yet, register it as a **pending
   publisher** (this claims the name at the same time):
   go to <https://pypi.org/manage/account/publishing/> → "Add a new pending
   publisher" → fill in exactly:

   | Field | Value |
   |---|---|
   | PyPI project name | `tangent-ad` |
   | Owner | `pedronahum` |
   | Repository name | `tangent` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

3. In the GitHub repo: **Settings → Environments → New environment** named
   `pypi` (no secrets needed; optionally add yourself as a required
   reviewer so every release needs a manual approval click).

### 2. GitHub Pages (documentation site)

**Settings → Pages → Build and deployment → Source: "GitHub Actions"**.
The docs workflow (`.github/workflows/docs.yml`) then publishes
<https://pedronahum.github.io/tangent/> on every push to `master` that
touches docs.

## Cutting a release

1. Ensure `CHANGELOG.md` has a section for the version and
   `tangent/__init__.py`'s `__version__` matches (the workflow refuses a
   tag/version mismatch).
2. Change the CHANGELOG heading from "(unreleased)" to the release date.
3. Tag and push:

   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```

4. Watch the `release` workflow: build → twine check → clean-venv wheel
   smoke test → PyPI publish → GitHub Release with artifacts.
5. Verify: `pip install tangent-ad` in a fresh venv, then
   `python -c "import tangent; print(tangent.__version__)"`.

## Versioning

- The single source of truth is `__version__` in `tangent/__init__.py`
  (pyproject reads it dynamically).
- Stay ahead of upstream's final `0.1.9` so environments upgrading from the
  abandoned `tangent` package resolve cleanly after
  `pip uninstall tangent && pip install tangent-ad`.
