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

echo -e "${BLUE}[1/3] Running Backend Tests...${NC}"
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
echo -e "${BLUE}[2/3] Building Standalone Binary...${NC}"
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

# Final package info
echo -e "${BLUE}[3/3] Build Summary${NC}"
echo "--------------------------------------"

if [ -f "dist/rx" ]; then
    # Make binary executable
    chmod +x dist/rx

    BINARY_SIZE=$(du -sh dist/rx | cut -f1)
    echo "Binary size: $BINARY_SIZE"
    echo -e "${GREEN}✓ Standalone binary ready${NC}"
fi

echo ""
echo "============================================"
echo -e "${GREEN}  Build Complete! ✓${NC}"
echo "============================================"
echo ""
echo "Standalone binary: dist/rx"
echo ""
echo "Note: Frontend is now managed separately."
echo "      The server will automatically download the latest"
echo "      frontend from GitHub releases on startup."
echo ""
echo "To run the standalone binary:"
echo "  ./dist/rx serve"
echo ""
echo "Then open: http://localhost:8000"
echo ""
