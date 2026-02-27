.PHONY: install
install:
	brew install llvm pkg-config ninja ccache

.PHONY: venv
venv:
	uv sync

.PHONY: precommit
precommit:
	uvx isort xdsl_exo tests
	uvx autoflake --remove-all-unused-imports --recursive --in-place xdsl_exo tests
	uvx black --line-length 5000 xdsl_exo tests
	uvx ruff check --fix xdsl_exo tests

.PHONY: tests
tests:
	uv run pytest tests/
	uv run lit tests/filecheck/
