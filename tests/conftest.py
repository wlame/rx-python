"""Pytest configuration and shared fixtures for RX tests.

This module provides auto-use fixtures that ensure test isolation,
particularly for cache directories.
"""

import shutil
import tempfile

import pytest


@pytest.fixture(autouse=True)
def isolate_cache_directory(monkeypatch):
    """Auto-use fixture that isolates cache directory for each test.

    This fixture:
    1. Creates a temporary directory for the test's cache
    2. Sets RX_CACHE_DIR environment variable to point to it
    3. Cleans up the directory after the test completes

    This ensures:
    - Tests don't pollute the user's ~/.cache/rx directory
    - Tests don't interfere with each other through shared cache state
    - No artifacts are left after test runs
    """
    # Create a unique temp directory for this test
    temp_cache_dir = tempfile.mkdtemp(prefix='rx_test_cache_')

    # Set the environment variable
    monkeypatch.setenv('RX_CACHE_DIR', temp_cache_dir)

    # Run the test
    yield temp_cache_dir

    # Cleanup after test
    try:
        shutil.rmtree(temp_cache_dir, ignore_errors=True)
    except Exception:
        pass  # Best effort cleanup


@pytest.fixture
def temp_cache_dir(isolate_cache_directory):
    """Fixture that provides access to the isolated cache directory path.

    Use this fixture when you need to inspect or manipulate the cache
    directory in your test.

    Returns:
        str: Path to the temporary cache directory for this test
    """
    return isolate_cache_directory
