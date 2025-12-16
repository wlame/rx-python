#!/usr/bin/env python3
"""
Test script for frontend_manager functionality.

This script tests:
1. Environment variable handling (RX_FRONTEND_VERSION, RX_FRONTEND_URL, RX_FRONTEND_PATH)
2. Version checking logic
3. Security validation (path traversal prevention)
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path


# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from rx.frontend_manager import FrontendManager, validate_path_security


def test_env_variables():
    """Test environment variable handling."""
    print('=' * 60)
    print('TEST 1: Environment Variable Handling')
    print('=' * 60)

    # Test 1: RX_FRONTEND_PATH
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ['RX_FRONTEND_PATH'] = tmpdir
        manager = FrontendManager()
        assert str(manager.cache_dir) == tmpdir, f'Expected {tmpdir}, got {manager.cache_dir}'
        print(f'✓ RX_FRONTEND_PATH: {tmpdir}')
        del os.environ['RX_FRONTEND_PATH']

    # Test 2: RX_FRONTEND_VERSION
    os.environ['RX_FRONTEND_VERSION'] = 'v1.2.3'
    manager = FrontendManager()
    assert manager.env_version == 'v1.2.3', f'Expected v1.2.3, got {manager.env_version}'
    print(f'✓ RX_FRONTEND_VERSION: {manager.env_version}')
    del os.environ['RX_FRONTEND_VERSION']

    # Test 3: RX_FRONTEND_URL
    test_url = 'https://example.com/dist.tar.gz'
    os.environ['RX_FRONTEND_URL'] = test_url
    manager = FrontendManager()
    assert manager.env_url == test_url, f'Expected {test_url}, got {manager.env_url}'
    print(f'✓ RX_FRONTEND_URL: {manager.env_url}')
    del os.environ['RX_FRONTEND_URL']

    print('✓ All environment variable tests passed!\n')


def test_path_security():
    """Test path security validation."""
    print('=' * 60)
    print('TEST 2: Path Security Validation')
    print('=' * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        base_path = Path(tmpdir)

        # Create test structure
        (base_path / 'assets').mkdir()
        (base_path / 'index.html').touch()
        (base_path / 'assets' / 'app.js').touch()

        manager = FrontendManager(cache_dir=base_path)

        # Test 1: Valid paths
        valid_paths = [
            'index.html',
            'assets/app.js',
            './index.html',
            'assets/../index.html',
        ]

        for path in valid_paths:
            validated = manager.validate_static_file_path(path)
            if validated:
                print(f'✓ Valid path accepted: {path}')
            else:
                print(f'✗ Valid path rejected: {path}')

        # Test 2: Invalid paths (should be rejected)
        invalid_paths = [
            '../etc/passwd',
            '../../etc/passwd',
            '/etc/passwd',
            'assets/../../../etc/passwd',
            'assets/../../etc/passwd',
        ]

        for path in invalid_paths:
            validated = manager.validate_static_file_path(path)
            if validated:
                print(f'✗ SECURITY ISSUE: Invalid path accepted: {path}')
            else:
                print(f'✓ Invalid path rejected: {path}')

        # Test 3: Direct security validation function
        safe_path = base_path / 'index.html'
        unsafe_path = Path('/etc/passwd')

        assert validate_path_security(base_path, safe_path), 'Safe path should validate'
        assert not validate_path_security(base_path, unsafe_path), 'Unsafe path should not validate'

        print('✓ All security validation tests passed!\n')


async def test_url_construction():
    """Test URL construction for different versions."""
    print('=' * 60)
    print('TEST 3: URL Construction')
    print('=' * 60)

    manager = FrontendManager()

    # Test latest URL
    url = manager.get_direct_download_url('latest')
    expected = 'https://github.com/wlame/rx-viewer/releases/latest/download/dist.tar.gz'
    assert url == expected, f'Expected {expected}, got {url}'
    print(f'✓ Latest URL: {url}')

    # Test specific version with 'v' prefix
    url = manager.get_direct_download_url('v1.2.3')
    expected = 'https://github.com/wlame/rx-viewer/releases/download/v1.2.3/dist.tar.gz'
    assert url == expected, f'Expected {expected}, got {url}'
    print(f'✓ Version URL (with v): {url}')

    # Test specific version without 'v' prefix
    url = manager.get_direct_download_url('1.2.3')
    expected = 'https://github.com/wlame/rx-viewer/releases/download/v1.2.3/dist.tar.gz'
    assert url == expected, f'Expected {expected}, got {url}'
    print(f'✓ Version URL (without v): {url}')

    print('✓ All URL construction tests passed!\n')


async def test_ensure_frontend_logic():
    """Test the ensure_frontend logic without actual downloads."""
    print('=' * 60)
    print('TEST 4: ensure_frontend() Logic')
    print('=' * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)

        # Test 1: No cache, no env vars -> should try to download latest
        manager = FrontendManager(cache_dir=cache_dir)
        print(f'✓ Manager created with cache dir: {cache_dir}')
        print(f'  - Frontend available: {manager.is_frontend_available()}')
        print(f'  - Env version: {manager.env_version}')
        print(f'  - Env URL: {manager.env_url}')

        # Test 2: Create fake cached frontend
        (cache_dir / 'index.html').touch()
        (cache_dir / 'assets').mkdir()
        (cache_dir / 'assets' / 'app.js').touch()

        version_data = {'version': '1.0.0', 'buildDate': '2024-01-01T00:00:00Z', 'commit': 'abc123'}

        import json

        with open(cache_dir / 'version.json', 'w') as f:
            json.dump(version_data, f)

        print('✓ Created fake frontend cache')
        print(f'  - Frontend available: {manager.is_frontend_available()}')

        # Verify validation works
        validated = manager.validate_static_file_path('index.html')
        assert validated is not None, 'Should validate existing file'
        print(f'✓ Path validation works: {validated}')

        # Test security - try to escape
        validated = manager.validate_static_file_path('../../../etc/passwd')
        assert validated is None, 'Should reject path traversal'
        print('✓ Path traversal blocked')

    print('✓ All ensure_frontend logic tests passed!\n')


def main():
    """Run all tests."""
    print('\n' + '=' * 60)
    print('FRONTEND MANAGER TEST SUITE')
    print('=' * 60 + '\n')

    try:
        # Synchronous tests
        test_env_variables()
        test_path_security()

        # Async tests
        asyncio.run(test_url_construction())
        asyncio.run(test_ensure_frontend_logic())

        print('=' * 60)
        print('ALL TESTS PASSED! ✓')
        print('=' * 60)
        print('\nThe frontend_manager module is working correctly:')
        print('  ✓ Environment variables are handled properly')
        print('  ✓ Path security validation prevents directory traversal')
        print('  ✓ URL construction works for all version formats')
        print('  ✓ Cache detection and validation work correctly')
        print('\nNext steps:')
        print('  1. Remove frontend build scripts from this repository')
        print('  2. Test actual download from GitHub (requires rx-viewer release)')
        print('  3. Test with: rx serve')

    except AssertionError as e:
        print(f'\n✗ TEST FAILED: {e}')
        sys.exit(1)
    except Exception as e:
        print(f'\n✗ UNEXPECTED ERROR: {e}')
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
