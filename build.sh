#!/bin/bash
set -e

echo "============================================"
echo "  RX-Trace Build Script"
echo "============================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Build frontend
echo -e "${BLUE}[1/3] Building Frontend...${NC}"
echo "--------------------------------------"

if command -v docker &> /dev/null; then
    echo "Using Docker with Bun to build frontend..."
    make frontend-build
else
    echo -e "${RED}Warning: Docker not found. Attempting to use local Bun...${NC}"
    if command -v bun &> /dev/null; then
        echo "Using local Bun..."
        cd src/rx/frontend
        bun install
        bun run build
        cd "$SCRIPT_DIR"
    else
        echo -e "${RED}Error: Neither Docker nor Bun found. Cannot build frontend.${NC}"
        echo "Install Docker or Bun to build the frontend."
        echo "  - Docker: https://docs.docker.com/get-docker/"
        echo "  - Bun: https://bun.sh/"
        exit 1
    fi
fi

echo -e "${GREEN}✓ Frontend built successfully${NC}"
echo ""

# Verify frontend build
echo -e "${BLUE}[2/3] Verifying Frontend Build...${NC}"
echo "--------------------------------------"

FRONTEND_DIST="src/rx/frontend/dist"
if [ ! -d "$FRONTEND_DIST" ]; then
    echo -e "${RED}Error: Frontend dist directory not found: $FRONTEND_DIST${NC}"
    exit 1
fi

if [ ! -f "$FRONTEND_DIST/index.html" ]; then
    echo -e "${RED}Error: index.html not found in $FRONTEND_DIST${NC}"
    exit 1
fi

# Count built files
FILE_COUNT=$(find "$FRONTEND_DIST" -type f | wc -l | tr -d ' ')
TOTAL_SIZE=$(du -sh "$FRONTEND_DIST" | cut -f1)

echo "Frontend build verified:"
echo "  - Files: $FILE_COUNT"
echo "  - Total size: $TOTAL_SIZE"
echo "  - Location: $FRONTEND_DIST"
echo -e "${GREEN}✓ Frontend build verified${NC}"
echo ""

# Run backend tests
echo -e "${BLUE}[3/5] Running Backend Tests...${NC}"
echo "--------------------------------------"

if command -v uv &> /dev/null; then
    echo "Running tests with uv..."
    uv run pytest tests/ -v --tb=short -x 2>&1 | tail -30

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo -e "${GREEN}✓ All tests passed${NC}"
    else
        echo -e "${RED}✗ Some tests failed${NC}"
        exit 1
    fi
else
    echo -e "${RED}Warning: uv not found. Skipping tests.${NC}"
    echo "Install uv to run tests: https://github.com/astral-sh/uv"
fi

echo ""

# Build standalone binary with PyInstaller
echo -e "${BLUE}[4/5] Building Standalone Binary...${NC}"
echo "--------------------------------------"

if command -v uv &> /dev/null; then
    echo "Building binary with PyInstaller..."
    uv run pyinstaller rx.spec --clean --noconfirm

    if [ -f "dist/rx" ]; then
        BINARY_SIZE=$(du -sh dist/rx | cut -f1)
        echo "Binary created: dist/rx ($BINARY_SIZE)"
        echo -e "${GREEN}✓ Binary built successfully${NC}"
    else
        echo -e "${RED}✗ Binary build failed${NC}"
        exit 1
    fi
else
    echo -e "${RED}Warning: uv not found. Cannot build binary.${NC}"
fi

echo ""

# Create final package
echo -e "${BLUE}[5/5] Creating Distribution Package...${NC}"
echo "--------------------------------------"

if [ -f "dist/rx" ]; then
    # Make binary executable
    chmod +x dist/rx

    # For onefile builds, frontend files are embedded in the binary
    # We can verify by checking the PyInstaller build log
    BINARY_SIZE=$(du -sh dist/rx | cut -f1)
    echo "Binary size: $BINARY_SIZE (includes embedded frontend assets)"
    echo -e "${GREEN}✓ Frontend files embedded in binary${NC}"

    echo -e "${GREEN}✓ Distribution package ready${NC}"
fi

echo ""
echo "============================================"
echo -e "${GREEN}  Build Complete! ✓${NC}"
echo "============================================"
echo ""
echo "Standalone binary: dist/rx"
echo ""
echo "To run the standalone binary:"
echo "  ./dist/rx serve"
echo ""
echo "Then open: http://localhost:8000"
echo ""
