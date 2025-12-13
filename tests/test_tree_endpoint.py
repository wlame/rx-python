"""Tests for the /v1/tree endpoint"""

import os
import shutil
import tempfile

import pytest
from fastapi.testclient import TestClient

from rx.web import app


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests and set it as search root."""
    tmp_dir = tempfile.mkdtemp()
    # Resolve symlinks (e.g., /var -> /private/var on macOS) for consistent paths
    resolved_tmp_dir = os.path.realpath(tmp_dir)
    # Set environment variable so app lifespan uses our temp directory
    old_env = os.environ.get('RX_SEARCH_ROOT')
    os.environ['RX_SEARCH_ROOT'] = resolved_tmp_dir
    yield resolved_tmp_dir
    # Cleanup
    shutil.rmtree(resolved_tmp_dir, ignore_errors=True)
    # Restore original env var
    if old_env is not None:
        os.environ['RX_SEARCH_ROOT'] = old_env
    elif 'RX_SEARCH_ROOT' in os.environ:
        del os.environ['RX_SEARCH_ROOT']


@pytest.fixture
def client(temp_dir):
    """Create test client with search root set to temp directory"""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def populated_dir(temp_dir):
    """Create a populated directory structure for testing."""
    # Create subdirectories
    subdir1 = os.path.join(temp_dir, 'logs')
    subdir2 = os.path.join(temp_dir, 'data')
    os.makedirs(subdir1)
    os.makedirs(subdir2)

    # Create files
    files = {
        'readme.txt': 'This is a readme file\n',
        'app.log': 'Log line 1\nLog line 2\nLog line 3\n',
        'logs/error.log': 'Error 1\nError 2\n',
        'logs/access.log': 'Access log entry\n',
        'data/config.json': '{"key": "value"}\n',
    }

    for filename, content in files.items():
        filepath = os.path.join(temp_dir, filename)
        with open(filepath, 'w') as f:
            f.write(content)

    return temp_dir


class TestTreeEndpoint:
    """Tests for the /v1/tree endpoint"""

    def test_tree_without_path_returns_search_roots(self, client, temp_dir):
        """Test that calling /v1/tree without path returns search roots"""
        response = client.get('/v1/tree')
        assert response.status_code == 200
        data = response.json()

        assert data['is_search_root'] is True
        assert data['parent'] is None
        assert 'entries' in data
        assert len(data['entries']) == 1  # Our temp_dir is the only search root
        assert data['entries'][0]['path'] == temp_dir
        assert data['entries'][0]['type'] == 'directory'

    def test_tree_with_valid_path_returns_contents(self, client, populated_dir):
        """Test that /v1/tree with a valid path returns directory contents"""
        response = client.get(f'/v1/tree?path={populated_dir}')
        assert response.status_code == 200
        data = response.json()

        assert data['path'] == populated_dir
        assert data['is_search_root'] is True
        assert 'entries' in data

        # Should have directories first, then files
        entry_names = [e['name'] for e in data['entries']]
        assert 'data' in entry_names
        assert 'logs' in entry_names
        assert 'readme.txt' in entry_names
        assert 'app.log' in entry_names

    def test_tree_directories_come_before_files(self, client, populated_dir):
        """Test that directories are listed before files"""
        response = client.get(f'/v1/tree?path={populated_dir}')
        assert response.status_code == 200
        data = response.json()

        entries = data['entries']
        # Find index where files start (first non-directory)
        dir_count = sum(1 for e in entries if e['type'] == 'directory')

        # Verify directories come first
        for i, entry in enumerate(entries):
            if i < dir_count:
                assert entry['type'] == 'directory', f'Entry {i} should be directory'
            else:
                assert entry['type'] == 'file', f'Entry {i} should be file'

    def test_tree_entries_sorted_alphabetically(self, client, populated_dir):
        """Test that entries are sorted alphabetically within their groups"""
        response = client.get(f'/v1/tree?path={populated_dir}')
        assert response.status_code == 200
        data = response.json()

        entries = data['entries']

        # Separate directories and files
        dirs = [e['name'] for e in entries if e['type'] == 'directory']
        files = [e['name'] for e in entries if e['type'] == 'file']

        # Verify each group is sorted (case-insensitive)
        assert dirs == sorted(dirs, key=str.lower)
        assert files == sorted(files, key=str.lower)

    def test_tree_file_metadata(self, client, populated_dir):
        """Test that file entries have correct metadata"""
        response = client.get(f'/v1/tree?path={populated_dir}')
        assert response.status_code == 200
        data = response.json()

        # Find the readme.txt file
        readme = next((e for e in data['entries'] if e['name'] == 'readme.txt'), None)
        assert readme is not None

        assert readme['type'] == 'file'
        assert readme['size'] is not None
        assert readme['size'] > 0
        assert readme['size_human'] is not None
        assert readme['modified_at'] is not None
        assert readme['is_text'] is True
        assert readme['is_compressed'] is False

    def test_tree_directory_metadata(self, client, populated_dir):
        """Test that directory entries have correct metadata"""
        response = client.get(f'/v1/tree?path={populated_dir}')
        assert response.status_code == 200
        data = response.json()

        # Find the logs directory
        logs_dir = next((e for e in data['entries'] if e['name'] == 'logs'), None)
        assert logs_dir is not None

        assert logs_dir['type'] == 'directory'
        assert logs_dir['children_count'] == 2  # error.log and access.log
        assert logs_dir['modified_at'] is not None
        # Files don't have size for directories
        assert logs_dir['size'] is None

    def test_tree_subdirectory(self, client, populated_dir):
        """Test listing contents of a subdirectory"""
        logs_path = os.path.join(populated_dir, 'logs')
        response = client.get(f'/v1/tree?path={logs_path}')
        assert response.status_code == 200
        data = response.json()

        assert data['path'] == logs_path
        assert data['parent'] == populated_dir
        assert data['is_search_root'] is False

        entry_names = [e['name'] for e in data['entries']]
        assert 'error.log' in entry_names
        assert 'access.log' in entry_names

    def test_tree_path_outside_search_root_forbidden(self, client, temp_dir):
        """Test that accessing path outside search root returns 403"""
        response = client.get('/v1/tree?path=/etc')
        assert response.status_code == 403

    def test_tree_nonexistent_path_not_found(self, client, temp_dir):
        """Test that nonexistent path returns 404"""
        nonexistent = os.path.join(temp_dir, 'does_not_exist')
        response = client.get(f'/v1/tree?path={nonexistent}')
        assert response.status_code == 404

    def test_tree_file_path_returns_error(self, client, populated_dir):
        """Test that providing a file path instead of directory returns 400"""
        file_path = os.path.join(populated_dir, 'readme.txt')
        response = client.get(f'/v1/tree?path={file_path}')
        assert response.status_code == 400
        assert 'not a directory' in response.json()['detail'].lower()

    def test_tree_total_entries_count(self, client, populated_dir):
        """Test that total_entries matches actual entry count"""
        response = client.get(f'/v1/tree?path={populated_dir}')
        assert response.status_code == 200
        data = response.json()

        assert data['total_entries'] == len(data['entries'])

    def test_tree_total_size_calculation(self, client, populated_dir):
        """Test that total_size is calculated correctly"""
        response = client.get(f'/v1/tree?path={populated_dir}')
        assert response.status_code == 200
        data = response.json()

        # Calculate expected total from file entries
        expected_total = sum(e['size'] for e in data['entries'] if e['type'] == 'file')

        if expected_total > 0:
            assert data['total_size'] == expected_total
            assert data['total_size_human'] is not None
