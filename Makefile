.PHONY: help install dev test lint build clean frontend-install frontend-dev frontend-build

# Default target
help:
	@echo "RX Trace - Development Commands"
	@echo ""
	@echo "Backend:"
	@echo "  install         Install Python dependencies"
	@echo "  dev             Run development server"
	@echo "  test            Run tests"
	@echo "  lint            Run linter"
	@echo ""
	@echo "Frontend:"
	@echo "  frontend-install  Install frontend dependencies (requires Docker)"
	@echo "  frontend-dev      Start frontend dev server (requires Docker)"
	@echo "  frontend-build    Build frontend for production (requires Docker)"
	@echo ""
	@echo "All:"
	@echo "  build           Build everything"
	@echo "  clean           Clean build artifacts"

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

# Frontend commands (using Docker with Bun)
FRONTEND_DIR = src/rx/frontend
DOCKER_BUN = docker run --rm -v $(PWD)/$(FRONTEND_DIR):/app -w /app oven/bun:1

frontend-install:
	$(DOCKER_BUN) bun install

frontend-dev:
	docker run --rm -it -v $(PWD)/$(FRONTEND_DIR):/app -w /app -p 5173:5173 oven/bun:1 \
		sh -c "bun install && bun run dev --host 0.0.0.0"

frontend-build:
	$(DOCKER_BUN) sh -c "bun install && bun run build"
	@echo "Frontend built to $(FRONTEND_DIR)/dist"

# Combined commands
build: frontend-build
	@echo "Build complete"

clean:
	rm -rf $(FRONTEND_DIR)/dist
	rm -rf $(FRONTEND_DIR)/node_modules
	rm -rf .pytest_cache
	rm -rf htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
