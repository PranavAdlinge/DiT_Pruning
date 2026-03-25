# Nexus - Python environment and common tasks
# Usage: make [target]

PYTHON ?= python3
VENV_DIR ?= .venv
VENV_BIN = $(VENV_DIR)/bin

.PHONY: venv install install-dev test lint lint-fix format verify clean

# Create virtual environment
venv:
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "Created $(VENV_DIR). Activate with: source $(VENV_BIN)/activate"

# Install package in editable mode (from active env)
install:
	pip install -e .

# Create venv and install (one-shot setup)
install-dev: venv
	$(VENV_BIN)/pip install -e .
	@echo "Setup complete. Activate with: source $(VENV_BIN)/activate"

# Run tests
test:
	pytest -v

# Lint only (no fix)
lint:
	./lint.sh

# Lint with auto-fix
lint-fix:
	./lint.sh --fix

# Format code
format:
	./lint.sh format

# Verify setup (check_setup.py)
verify:
	$(PYTHON) check_setup.py

# Remove build artifacts
clean:
	rm -rf build/ dist/ *.egg-info .eggs/
