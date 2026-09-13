# Developer shortcuts. `make check` mirrors the CI lint job exactly, so run it
# before pushing to catch the ruff gates locally instead of in CI.
#
# ruff is version-sensitive: `ruff format` output can change between releases,
# so CI pins ruff==0.16.6 and so does `make install-dev` / the pre-commit hook.
# Override the interpreter/tools if you are not using a local .venv, e.g.
#   make check RUFF=ruff PYTHON=python

PYTHON ?= python
RUFF ?= ruff
RUFF_VERSION := 0.16.6

.DEFAULT_GOAL := help

.PHONY: help
help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "} {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

.PHONY: install-dev
install-dev:  ## Install the package with dev + test tooling (pinned ruff, pre-commit)
	$(PYTHON) -m pip install -e ".[dev,test]"

.PHONY: hooks
hooks:  ## Install the git pre-commit hooks (runs ruff on staged files)
	pre-commit install

.PHONY: fmt
fmt:  ## Auto-format and auto-fix lint (edits files in place)
	$(RUFF) format .
	$(RUFF) check --fix .

.PHONY: fmt-check
fmt-check:  ## Check formatting only (no edits) - the CI `ruff format --check` gate
	$(RUFF) format --check .

.PHONY: lint
lint:  ## Run the linter (no edits) - the CI `ruff check` gate
	$(RUFF) check .

.PHONY: check
check: lint fmt-check  ## Run both CI gates (ruff check + ruff format --check)

.PHONY: test
test:  ## Run the test suite
	$(PYTHON) -m pytest tests/ -q
