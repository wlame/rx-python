.PHONY: help install dev test lint clean

# Default target
help:
	@echo "RX Trace - Development Commands"
	@echo ""
	@echo "Backend:"
	@echo "  install         Install Python dependencies"
	@echo "  dev             Run development server"
	@echo "  test            Run tests"
	@echo "  lint            Run linter"
	@echo "  clean           Clean build artifacts"
	@echo ""
	@echo "Note: Frontend is now managed separately in the rx-viewer repository."
	@echo "      The backend will automatically download the latest frontend on startup."

# Backend commands
install:
	uv sync

dev:
	uv run rx serve --reload

test:
	uv run pytest -v

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

# Clean build artifacts
clean:
	rm -rf .pytest_cache
	rm -rf htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
