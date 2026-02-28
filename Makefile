.PHONY: install
install:
	brew install llvm pkg-config ninja ccache

.PHONY: venv
venv:
	uv sync

SRC_DIRS = xdsl_exo tests examples
.PHONY: precommit
precommit:
	uvx isort $(SRC_DIRS)
	uvx autoflake --remove-all-unused-imports --recursive --in-place $(SRC_DIRS)
	uvx black --line-length 5000 $(SRC_DIRS)
	uvx ruff check --fix $(SRC_DIRS)

.PHONY: tests
tests:
	uv run pytest tests/
	uv run lit tests/filecheck/
