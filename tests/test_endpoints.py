"""Tests for FastAPI endpoints"""

import os
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
    import shutil

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
def temp_test_file(temp_dir):
    """Create a temporary test file with known content"""
    content = """Line 1: Hello world
Line 2: Python is awesome
Line 3: FastAPI rocks
Line 4: Testing is important
Line 5: Hello again
"""
    temp_path = os.path.join(temp_dir, 'test_file.txt')
    with open(temp_path, 'w') as f:
        f.write(content)

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


class TestHealthEndpoint:
    """Tests for the health endpoint (at /health)"""

    def test_health_returns_ok(self, client):
        """Test health endpoint returns ok status"""
        response = client.get('/health')
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'ok'

    def test_health_includes_ripgrep_status(self, client):
        """Test health endpoint includes ripgrep availability"""
        response = client.get('/health')
        assert response.status_code == 200
        data = response.json()
        assert 'ripgrep_available' in data
        assert isinstance(data['ripgrep_available'], bool)

    def test_health_includes_app_version(self, client):
        """Test health endpoint includes app version"""
        response = client.get('/health')
        assert response.status_code == 200
        data = response.json()
        assert 'app_version' in data
        assert isinstance(data['app_version'], str)
        assert len(data['app_version']) > 0

    def test_health_includes_python_version(self, client):
        """Test health endpoint includes Python version"""
        response = client.get('/health')
        assert response.status_code == 200
        data = response.json()
        assert 'python_version' in data
        assert isinstance(data['python_version'], str)
        assert len(data['python_version']) > 0

    def test_health_includes_os_info(self, client):
        """Test health endpoint includes OS information"""
        response = client.get('/health')
        assert response.status_code == 200
        data = response.json()
        assert 'os_info' in data
        assert isinstance(data['os_info'], dict)
        assert 'system' in data['os_info']
        assert 'release' in data['os_info']
        assert 'version' in data['os_info']
        assert 'machine' in data['os_info']

    def test_health_includes_system_resources(self, client):
        """Test health endpoint includes system resources"""
        response = client.get('/health')
        assert response.status_code == 200
        data = response.json()
        assert 'system_resources' in data
        assert isinstance(data['system_resources'], dict)
        assert 'cpu_cores' in data['system_resources']
        assert 'cpu_cores_physical' in data['system_resources']
        assert 'ram_total_gb' in data['system_resources']
        assert 'ram_available_gb' in data['system_resources']
        assert 'ram_percent_used' in data['system_resources']
        assert data['system_resources']['cpu_cores'] > 0
        assert data['system_resources']['ram_total_gb'] > 0

    def test_health_includes_python_packages(self, client):
        """Test health endpoint includes Python package versions"""
        response = client.get('/health')
        assert response.status_code == 200
        data = response.json()
        assert 'python_packages' in data
        assert isinstance(data['python_packages'], dict)
        # Check for key packages
        assert 'fastapi' in data['python_packages']
        assert 'pydantic' in data['python_packages']

    def test_health_includes_constants(self, client):
        """Test health endpoint includes application constants"""
        response = client.get('/health')
        assert response.status_code == 200
        data = response.json()
        assert 'constants' in data
        assert isinstance(data['constants'], dict)
        # Check for key constants
        assert 'LOG_LEVEL' in data['constants']
        assert 'DEBUG_MODE' in data['constants']
        assert 'LINE_SIZE_ASSUMPTION_KB' in data['constants']
        assert 'MAX_SUBPROCESSES' in data['constants']
        assert 'MIN_CHUNK_SIZE_MB' in data['constants']
        assert 'MAX_FILES' in data['constants']
        # Verify types
        assert isinstance(data['constants']['LOG_LEVEL'], str)
        assert isinstance(data['constants']['DEBUG_MODE'], bool)
        assert isinstance(data['constants']['LINE_SIZE_ASSUMPTION_KB'], int)
        assert isinstance(data['constants']['MAX_SUBPROCESSES'], int)
        assert isinstance(data['constants']['MIN_CHUNK_SIZE_MB'], int)
        assert isinstance(data['constants']['MAX_FILES'], int)

    def test_health_includes_environment(self, client):
        """Test health endpoint includes environment variables"""
        response = client.get('/health')
        assert response.status_code == 200
        data = response.json()
        assert 'environment' in data
        assert isinstance(data['environment'], dict)
        # Environment dict may be empty if no app-related env vars are set

    def test_health_includes_docs_url(self, client):
        """Test health endpoint includes documentation URL"""
        response = client.get('/health')
        assert response.status_code == 200
        data = response.json()
        assert 'docs_url' in data
        assert isinstance(data['docs_url'], str)
        assert data['docs_url'].startswith('http')


class TestTraceEndpoint:
    """Tests for the trace endpoint"""

    def test_trace_requires_filename(self, client):
        """Test trace endpoint requires filename parameter"""
        response = client.get('/v1/trace?regex=test')
        assert response.status_code == 422  # Validation error

    def test_trace_requires_regex(self, client):
        """Test trace endpoint requires regex parameter"""
        response = client.get('/v1/trace?path=test.txt')
        assert response.status_code == 422  # Validation error

    def test_trace_finds_matches(self, client, temp_test_file):
        """Test trace endpoint finds matching byte offsets"""
        response = client.get('/v1/trace', params={'path': temp_test_file, 'regexp': 'Hello'})
        assert response.status_code == 200
        data = response.json()
        # Check ID-based structure
        assert 'patterns' in data
        assert 'files' in data
        assert 'matches' in data
        assert len(data['patterns']) == 1  # p1: Hello
        assert len(data['files']) == 1  # f1: temp_test_file
        assert len(data['matches']) == 2  # Two matches for "Hello"
        assert all('pattern' in m and 'file' in m and 'offset' in m for m in data['matches'])

    def test_trace_no_matches(self, client, temp_test_file):
        """Test trace endpoint returns empty list when no matches"""
        response = client.get('/v1/trace', params={'path': temp_test_file, 'regexp': 'NOTFOUND'})
        assert response.status_code == 200
        data = response.json()
        assert data['matches'] == []

    def test_trace_with_regex_pattern(self, client, temp_test_file):
        """Test trace endpoint with regex pattern"""
        response = client.get('/v1/trace', params={'path': temp_test_file, 'regexp': r'Line \d+:'})
        assert response.status_code == 200
        data = response.json()
        # All 5 lines match this pattern
        assert len(data['matches']) == 5
        assert all('pattern' in m and 'file' in m and 'offset' in m for m in data['matches'])

    def test_trace_nonexistent_file(self, client, temp_dir):
        """Test trace endpoint with nonexistent file"""
        nonexistent = os.path.join(temp_dir, 'nonexistent.txt')
        response = client.get('/v1/trace', params={'path': nonexistent, 'regexp': 'test'})
        assert response.status_code == 404  # File not found

    def test_trace_invalid_regex(self, client, temp_test_file):
        """Test trace endpoint with invalid regex"""
        response = client.get('/v1/trace', params={'path': temp_test_file, 'regexp': '[invalid'})
        assert response.status_code == 400  # Invalid regex pattern

    def test_trace_binary_file(self, client, temp_dir):
        """Test trace endpoint skips binary files"""
        # Create a temporary binary file in the temp_dir (within search root)
        binary_file = os.path.join(temp_dir, 'test.bin')
        with open(binary_file, 'wb') as f:
            f.write(b'\x00\x01\x02\x03\xff\xfe\xfd')  # Binary data with null bytes

        try:
            response = client.get('/v1/trace', params={'path': binary_file, 'regexp': 'test'})
            # Binary files are skipped, not rejected - returns 200 with empty matches
            assert response.status_code == 200
            data = response.json()
            assert len(data['matches']) == 0
            assert len(data['skipped_files']) == 1
        finally:
            if os.path.exists(binary_file):
                os.unlink(binary_file)


class TestSamplesEndpoint:
    """Tests for the /v1/samples endpoint"""

    def test_samples_with_offsets(self, client, temp_test_file):
        """Test samples endpoint with byte offsets"""
        response = client.get('/v1/samples', params={'path': temp_test_file, 'offsets': '0,20'})
        assert response.status_code == 200
        data = response.json()
        assert data['path'] == temp_test_file
        # offsets now maps offset -> line number
        assert isinstance(data['offsets'], dict)
        assert '0' in data['offsets']
        assert '20' in data['offsets']
        assert data['offsets']['0'] >= 1  # Line number is 1-based
        assert data['lines'] == {}
        assert 'samples' in data
        assert '0' in data['samples']
        assert '20' in data['samples']

    def test_samples_with_lines(self, client, temp_test_file):
        """Test samples endpoint with line numbers"""
        response = client.get('/v1/samples', params={'path': temp_test_file, 'lines': '1,3'})
        assert response.status_code == 200
        data = response.json()
        assert data['path'] == temp_test_file
        assert data['offsets'] == {}
        # lines now maps line number -> byte offset
        assert isinstance(data['lines'], dict)
        assert '1' in data['lines']
        assert '3' in data['lines']
        assert data['lines']['1'] >= 0  # Offset is 0-based
        assert 'samples' in data
        assert '1' in data['samples']
        assert '3' in data['samples']
        # Check content
        assert any('Hello world' in line for line in data['samples']['1'])
        assert any('FastAPI rocks' in line for line in data['samples']['3'])

    def test_samples_lines_single(self, client, temp_test_file):
        """Test samples endpoint with single line number"""
        response = client.get('/v1/samples', params={'path': temp_test_file, 'lines': '2'})
        assert response.status_code == 200
        data = response.json()
        # lines now is a dict
        assert isinstance(data['lines'], dict)
        assert '2' in data['lines']
        assert data['lines']['2'] >= 0
        assert '2' in data['samples']
        assert any('Python is awesome' in line for line in data['samples']['2'])

    def test_samples_mutual_exclusivity(self, client, temp_test_file):
        """Test that offsets and lines cannot be used together"""
        response = client.get('/v1/samples', params={'path': temp_test_file, 'offsets': '0', 'lines': '1'})
        assert response.status_code == 400
        assert 'cannot use both' in response.json()['detail'].lower()

    def test_samples_requires_offsets_or_lines(self, client, temp_test_file):
        """Test that either offsets or lines must be provided"""
        response = client.get('/v1/samples', params={'path': temp_test_file})
        assert response.status_code == 400
        assert 'must provide' in response.json()['detail'].lower()

    def test_samples_with_context(self, client, temp_test_file):
        """Test samples endpoint with context parameter"""
        response = client.get('/v1/samples', params={'path': temp_test_file, 'lines': '3', 'context': 1})
        assert response.status_code == 200
        data = response.json()
        assert data['before_context'] == 1
        assert data['after_context'] == 1
        # Should have line 2, 3, 4 (1 before, target, 1 after)
        assert len(data['samples']['3']) == 3

    def test_samples_with_before_after_context(self, client, temp_test_file):
        """Test samples endpoint with separate before/after context"""
        response = client.get(
            '/v1/samples', params={'path': temp_test_file, 'lines': '3', 'before_context': 1, 'after_context': 2}
        )
        assert response.status_code == 200
        data = response.json()
        assert data['before_context'] == 1
        assert data['after_context'] == 2

    def test_samples_invalid_offsets_format(self, client, temp_test_file):
        """Test samples endpoint with invalid offsets format"""
        response = client.get('/v1/samples', params={'path': temp_test_file, 'offsets': 'invalid,data'})
        assert response.status_code == 400
        assert 'invalid offsets' in response.json()['detail'].lower()

    def test_samples_invalid_lines_format(self, client, temp_test_file):
        """Test samples endpoint with invalid lines format"""
        response = client.get('/v1/samples', params={'path': temp_test_file, 'lines': 'abc,xyz'})
        assert response.status_code == 400
        assert 'invalid lines' in response.json()['detail'].lower()

    def test_samples_nonexistent_file(self, client, temp_dir):
        """Test samples endpoint with nonexistent file"""
        nonexistent = os.path.join(temp_dir, 'nonexistent.txt')
        response = client.get('/v1/samples', params={'path': nonexistent, 'offsets': '0'})
        assert response.status_code == 404

    def test_samples_line_beyond_file(self, client, temp_test_file):
        """Test samples endpoint with line number beyond file - should return 400 EOF error"""
        response = client.get('/v1/samples', params={'path': temp_test_file, 'lines': '999'})
        assert response.status_code == 400
        data = response.json()
        assert 'detail' in data
        assert 'EOF reached' in data['detail'] or 'out of bounds' in data['detail']

    def test_samples_json_structure_with_lines(self, client, temp_test_file):
        """Test complete JSON structure when using lines"""
        response = client.get('/v1/samples', params={'path': temp_test_file, 'lines': '2'})
        assert response.status_code == 200
        data = response.json()
        assert 'path' in data
        assert 'offsets' in data
        assert 'lines' in data
        assert 'before_context' in data
        assert 'after_context' in data
        assert 'samples' in data
        # offsets and lines are now dicts
        assert isinstance(data['offsets'], dict)
        assert isinstance(data['lines'], dict)
        assert isinstance(data['samples'], dict)

    def test_samples_json_structure_with_offsets(self, client, temp_test_file):
        """Test complete JSON structure when using offsets"""
        response = client.get('/v1/samples', params={'path': temp_test_file, 'offsets': '0'})
        assert response.status_code == 200
        data = response.json()
        # offsets now maps offset -> line number
        assert isinstance(data['offsets'], dict)
        assert '0' in data['offsets']
        assert data['lines'] == {}

    def test_samples_binary_file(self, client, temp_dir):
        """Test samples endpoint with binary file"""
        binary_file = os.path.join(temp_dir, 'test.bin')
        with open(binary_file, 'wb') as f:
            f.write(b'\x00\x01\x02\x03\xff\xfe\xfd')

        try:
            response = client.get('/v1/samples', params={'path': binary_file, 'offsets': '0'})
            assert response.status_code == 400  # Binary file rejected
        finally:
            if os.path.exists(binary_file):
                os.unlink(binary_file)


class TestSamplesEndpointRanges:
    """Tests for the /v1/samples endpoint with range support"""

    @pytest.fixture
    def range_test_file(self, temp_dir):
        """Create a test file with 20 numbered lines for range testing"""
        lines = [f'Line {i}: Content for line number {i}\n' for i in range(1, 21)]
        temp_path = os.path.join(temp_dir, 'range_test.txt')
        with open(temp_path, 'w') as f:
            f.writelines(lines)
        yield temp_path
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_samples_line_range_basic(self, client, range_test_file):
        """Test samples endpoint with line range"""
        response = client.get('/v1/samples', params={'path': range_test_file, 'lines': '5-10'})
        assert response.status_code == 200
        data = response.json()
        # Range key should be in samples
        assert '5-10' in data['samples']
        # Should have exactly 6 lines (5, 6, 7, 8, 9, 10)
        assert len(data['samples']['5-10']) == 6
        # Verify content
        assert 'Line 5:' in data['samples']['5-10'][0]
        assert 'Line 10:' in data['samples']['5-10'][5]

    def test_samples_line_range_ignores_context(self, client, range_test_file):
        """Test that line ranges ignore context parameters"""
        response = client.get('/v1/samples', params={'path': range_test_file, 'lines': '10-12', 'context': 5})
        assert response.status_code == 200
        data = response.json()
        # Should have exactly 3 lines (10, 11, 12), not 8 (10-5 to 12+5)
        assert '10-12' in data['samples']
        assert len(data['samples']['10-12']) == 3

    def test_samples_mixed_single_and_range(self, client, range_test_file):
        """Test mixing single line with range"""
        response = client.get('/v1/samples', params={'path': range_test_file, 'lines': '3,10-12', 'context': 1})
        assert response.status_code == 200
        data = response.json()
        # Single line 3 should have context
        assert '3' in data['samples']
        assert len(data['samples']['3']) == 3  # line 2, 3, 4
        # Range should have exact lines
        assert '10-12' in data['samples']
        assert len(data['samples']['10-12']) == 3

    def test_samples_line_range_mapping(self, client, range_test_file):
        """Test that line range returns mapping entry"""
        response = client.get('/v1/samples', params={'path': range_test_file, 'lines': '5-7'})
        assert response.status_code == 200
        data = response.json()
        # Range key should be in lines mapping
        assert '5-7' in data['lines']
        # For ranges, byte offset is -1 (not computed for performance)
        assert data['lines']['5-7'] == -1

    def test_samples_byte_range_basic(self, client, range_test_file):
        """Test samples endpoint with byte offset range"""
        # Get file size first to create a reasonable range
        response = client.get('/v1/samples', params={'path': range_test_file, 'offsets': '0-100'})
        assert response.status_code == 200
        data = response.json()
        # Range key should be in samples
        assert '0-100' in data['samples']
        # Should have some lines
        assert len(data['samples']['0-100']) > 0

    def test_samples_invalid_range_format(self, client, range_test_file):
        """Test invalid range format"""
        response = client.get('/v1/samples', params={'path': range_test_file, 'lines': '10-20-30'})
        assert response.status_code == 400
        assert 'invalid' in response.json()['detail'].lower()

    def test_samples_invalid_range_order(self, client, range_test_file):
        """Test range with start > end"""
        response = client.get('/v1/samples', params={'path': range_test_file, 'lines': '15-10'})
        assert response.status_code == 400
        assert 'start must be' in response.json()['detail'].lower()

    def test_samples_range_beyond_file(self, client, range_test_file):
        """Test range that extends beyond file end"""
        response = client.get('/v1/samples', params={'path': range_test_file, 'lines': '18-100'})
        assert response.status_code == 200
        data = response.json()
        # Should return lines 18-20 (clamped)
        assert '18-100' in data['samples']
        assert len(data['samples']['18-100']) == 3  # lines 18, 19, 20


class TestSamplesEndpointNegativeOffsets:
    """Tests for the /v1/samples endpoint with negative offset support"""

    @pytest.fixture
    def negative_test_file(self, temp_dir):
        """Create a test file with 10 numbered lines for negative offset testing"""
        lines = [f'Line {i}: Test content\n' for i in range(1, 11)]
        temp_path = os.path.join(temp_dir, 'negative_test.txt')
        with open(temp_path, 'w') as f:
            f.writelines(lines)
        yield temp_path
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_samples_negative_line_last(self, client, negative_test_file):
        """Test -1 gets last line"""
        response = client.get('/v1/samples', params={'path': negative_test_file, 'lines': '-1', 'context': 0})
        assert response.status_code == 200
        data = response.json()
        # Should have line 10 (the last line)
        assert '10' in data['samples']
        assert 'Line 10:' in data['samples']['10'][0]

    def test_samples_negative_line_fifth_from_end(self, client, negative_test_file):
        """Test -5 gets 5th line from end"""
        response = client.get('/v1/samples', params={'path': negative_test_file, 'lines': '-5', 'context': 0})
        assert response.status_code == 200
        data = response.json()
        # -5 means 5th from end = line 6 (10 - 5 + 1 = 6)
        assert '6' in data['samples']
        assert 'Line 6:' in data['samples']['6'][0]

    def test_samples_negative_line_with_context(self, client, negative_test_file):
        """Test negative line with context"""
        response = client.get('/v1/samples', params={'path': negative_test_file, 'lines': '-1', 'context': 2})
        assert response.status_code == 200
        data = response.json()
        # Should have lines 8, 9, 10 (2 before + target)
        assert '10' in data['samples']
        assert len(data['samples']['10']) == 3

    def test_samples_mixed_positive_negative(self, client, negative_test_file):
        """Test mixing positive and negative line numbers"""
        response = client.get('/v1/samples', params={'path': negative_test_file, 'lines': '1,-1', 'context': 0})
        assert response.status_code == 200
        data = response.json()
        # Should have both first and last lines
        assert '1' in data['samples']
        assert '10' in data['samples']
        assert 'Line 1:' in data['samples']['1'][0]
        assert 'Line 10:' in data['samples']['10'][0]

    def test_samples_negative_byte_offset(self, client, negative_test_file):
        """Test negative byte offset"""
        response = client.get('/v1/samples', params={'path': negative_test_file, 'offsets': '-50', 'context': 0})
        assert response.status_code == 200
        data = response.json()
        # Should have some content from near the end of file
        assert len(data['samples']) == 1

    def test_samples_negative_in_range_rejected(self, client, negative_test_file):
        """Test that negative values in ranges are rejected"""
        response = client.get('/v1/samples', params={'path': negative_test_file, 'lines': '-5-10'})
        assert response.status_code == 400
        # Should reject negative values in ranges
        assert 'invalid' in response.json()['detail'].lower()


class TestSamplesEndpointMultipleSpecifications:
    """Tests for /v1/samples endpoint with multiple line specifications.

    This tests the use case:
    `/v1/samples?path=<file>&lines=1-15,55,60-62,-1&context=5`

    Expected behavior:
    - Range 1-15: returns lines 1-15 exactly (ranges ignore context)
    - Single line 55 with context=5: returns lines 50-60
    - Range 60-62: returns lines 60-62 exactly (ranges ignore context)
    - Negative -1 with context=5: returns last 6 lines (last line + 5 before)
    """

    @pytest.fixture
    def multispec_test_file(self, temp_dir):
        """Create a test file with 100 numbered lines."""
        lines = [f'Line {i}: content for line number {i}\n' for i in range(1, 101)]
        temp_path = os.path.join(temp_dir, 'multispec_test.txt')
        with open(temp_path, 'w') as f:
            f.writelines(lines)
        yield temp_path
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_samples_multiple_specs_combined(self, client, multispec_test_file):
        """Test combining range, single line, and negative offset with context.

        URL: /v1/samples?path=<file>&lines=1-15,55,60-62,-1&context=5

        Expected samples:
        - 1-15: 15 lines (range, context ignored)
        - 55: 11 lines (50-60, single line with context 5)
        - 60-62: 3 lines (range, context ignored)
        - 100: 6 lines (95-100, resolved from -1 with context 5)
        """
        response = client.get(
            '/v1/samples', params={'path': multispec_test_file, 'lines': '1-15,55,60-62,-1', 'context': 5}
        )
        assert response.status_code == 200
        data = response.json()

        # Check all specifications are in samples
        assert '1-15' in data['samples'], 'Range 1-15 should be in samples'
        assert '55' in data['samples'], 'Line 55 should be in samples'
        assert '60-62' in data['samples'], 'Range 60-62 should be in samples'
        assert '100' in data['samples'], 'Line 100 (resolved from -1) should be in samples'

        # Verify line counts
        assert len(data['samples']['1-15']) == 15, 'Range 1-15 should have exactly 15 lines'
        assert len(data['samples']['55']) == 11, 'Line 55 with context 5 should have 11 lines (50-60)'
        assert len(data['samples']['60-62']) == 3, 'Range 60-62 should have exactly 3 lines'
        assert len(data['samples']['100']) == 6, 'Line 100 with context 5 should have 6 lines (95-100)'

        # Verify content of range 1-15
        assert 'Line 1:' in data['samples']['1-15'][0]
        assert 'Line 15:' in data['samples']['1-15'][14]

        # Verify content of line 55 with context (lines 50-60)
        assert 'Line 50:' in data['samples']['55'][0]
        assert 'Line 55:' in data['samples']['55'][5]
        assert 'Line 60:' in data['samples']['55'][10]

        # Verify content of range 60-62
        assert 'Line 60:' in data['samples']['60-62'][0]
        assert 'Line 62:' in data['samples']['60-62'][2]

        # Verify content of -1 resolved to 100 with context (lines 95-100)
        assert 'Line 95:' in data['samples']['100'][0]
        assert 'Line 100:' in data['samples']['100'][5]

    def test_samples_multiple_ranges_no_context(self, client, multispec_test_file):
        """Test multiple ranges without context."""
        response = client.get(
            '/v1/samples', params={'path': multispec_test_file, 'lines': '1-5,20-25,90-95', 'context': 0}
        )
        assert response.status_code == 200
        data = response.json()

        # All three ranges should be present
        assert '1-5' in data['samples']
        assert '20-25' in data['samples']
        assert '90-95' in data['samples']

        # Verify exact line counts
        assert len(data['samples']['1-5']) == 5
        assert len(data['samples']['20-25']) == 6
        assert len(data['samples']['90-95']) == 6

    def test_samples_multiple_single_lines_with_context(self, client, multispec_test_file):
        """Test multiple single lines with context."""
        response = client.get('/v1/samples', params={'path': multispec_test_file, 'lines': '10,50,90', 'context': 2})
        assert response.status_code == 200
        data = response.json()

        # All three single lines should be present
        assert '10' in data['samples']
        assert '50' in data['samples']
        assert '90' in data['samples']

        # Each should have 5 lines (line + 2 before + 2 after)
        assert len(data['samples']['10']) == 5  # lines 8-12
        assert len(data['samples']['50']) == 5  # lines 48-52
        assert len(data['samples']['90']) == 5  # lines 88-92

    def test_samples_multiple_negative_offsets(self, client, multispec_test_file):
        """Test multiple negative line offsets with context."""
        response = client.get('/v1/samples', params={'path': multispec_test_file, 'lines': '-1,-10,-50', 'context': 2})
        assert response.status_code == 200
        data = response.json()

        # Negative offsets should be resolved to actual line numbers
        # -1 = 100, -10 = 91, -50 = 51
        assert '100' in data['samples']  # -1 resolved
        assert '91' in data['samples']  # -10 resolved
        assert '51' in data['samples']  # -50 resolved

        # Each should have 5 lines with context 2
        assert len(data['samples']['100']) == 3  # lines 98-100 (clamped at file end)
        assert len(data['samples']['91']) == 5  # lines 89-93
        assert len(data['samples']['51']) == 5  # lines 49-53

    def test_samples_mixed_positive_negative_ranges(self, client, multispec_test_file):
        """Test mixing positive lines, negative lines, and ranges."""
        response = client.get('/v1/samples', params={'path': multispec_test_file, 'lines': '1,10-15,-5', 'context': 1})
        assert response.status_code == 200
        data = response.json()

        # Line 1 with context 1
        assert '1' in data['samples']
        assert len(data['samples']['1']) == 2  # lines 1-2 (can't go before 1)

        # Range 10-15 (no context for ranges)
        assert '10-15' in data['samples']
        assert len(data['samples']['10-15']) == 6

        # -5 = line 96 with context 1
        assert '96' in data['samples']
        assert len(data['samples']['96']) == 3  # lines 95-97

    def test_samples_url_encoded_comma_separated(self, client, multispec_test_file):
        """Test URL-encoded comma-separated values work correctly.

        This tests: /v1/samples?path=<file>&lines=1-15%2C55%2C60-62%2C-1&context=5
        Where %2C is URL-encoded comma.
        """
        # FastAPI/Starlette handles URL decoding automatically
        response = client.get(
            '/v1/samples', params={'path': multispec_test_file, 'lines': '1-15,55,60-62,-1', 'context': 5}
        )
        assert response.status_code == 200
        data = response.json()

        # Should parse all four specifications
        assert len(data['samples']) == 4

    def test_samples_lines_mapping_with_multiple_specs(self, client, multispec_test_file):
        """Test that lines mapping is correct for multiple specifications."""
        response = client.get('/v1/samples', params={'path': multispec_test_file, 'lines': '10,20-25,-1', 'context': 0})
        assert response.status_code == 200
        data = response.json()

        # Check lines mapping
        assert '10' in data['lines']
        assert '20-25' in data['lines']
        assert '100' in data['lines']  # -1 resolved

        # Single lines should have byte offset, ranges should have -1
        assert data['lines']['10'] >= 0  # actual byte offset
        assert data['lines']['20-25'] == -1  # ranges don't compute offset
        assert data['lines']['100'] >= 0  # actual byte offset

    def test_samples_overlapping_specifications(self, client, multispec_test_file):
        """Test overlapping line specifications."""
        response = client.get('/v1/samples', params={'path': multispec_test_file, 'lines': '10-20,15-25', 'context': 0})
        assert response.status_code == 200
        data = response.json()

        # Both ranges should be in output
        assert '10-20' in data['samples']
        assert '15-25' in data['samples']

        # Verify counts
        assert len(data['samples']['10-20']) == 11  # lines 10-20
        assert len(data['samples']['15-25']) == 11  # lines 15-25

    def test_samples_single_line_overlapping_with_range(self, client, multispec_test_file):
        """Test single line with context overlapping a range."""
        response = client.get('/v1/samples', params={'path': multispec_test_file, 'lines': '50,48-52', 'context': 3})
        assert response.status_code == 200
        data = response.json()

        # Line 50 with context 3 (lines 47-53)
        assert '50' in data['samples']
        assert len(data['samples']['50']) == 7

        # Range 48-52 (exactly 5 lines)
        assert '48-52' in data['samples']
        assert len(data['samples']['48-52']) == 5

    def test_samples_before_after_context_with_specs(self, client, multispec_test_file):
        """Test --before and --after context with multiple specifications."""
        response = client.get(
            '/v1/samples', params={'path': multispec_test_file, 'lines': '50,70-72', 'before': 3, 'after': 2}
        )
        assert response.status_code == 200
        data = response.json()

        # Line 50 with before=3, after=2 (lines 47-52 = 6 lines)
        # Note: implementation may use max(before, after) for symmetric context
        assert '50' in data['samples']
        # Verify that lines 47-52 are included at minimum
        sample_text = '\n'.join(data['samples']['50'])
        assert 'Line 47:' in sample_text
        assert 'Line 50:' in sample_text
        assert 'Line 52:' in sample_text

        # Range 70-72 (exactly 3 lines, context ignored for ranges)
        assert '70-72' in data['samples']
        assert len(data['samples']['70-72']) == 3


class TestSamplesEndpointCompressedRanges:
    """Tests for /v1/samples endpoint with line ranges on compressed files."""

    @pytest.fixture
    def zstd_test_file(self, temp_dir):
        """Create a seekable zstd file with numbered lines."""
        import subprocess

        # Create source file with 100 lines
        txt_path = os.path.join(temp_dir, 'test.txt')
        zst_path = os.path.join(temp_dir, 'test.txt.zst')

        # Use newline='' for consistent LF line endings on all platforms
        with open(txt_path, 'w', newline='') as f:
            for i in range(1, 101):
                f.write(f'Line {i}: content for line number {i}\n')

        # Compress with rx compress (creates seekable zstd)
        result = subprocess.run(
            ['rx', 'compress', txt_path, '-o', zst_path],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0 or not os.path.exists(zst_path):
            pytest.skip('Could not create seekable zstd file')

        yield zst_path

        for p in [txt_path, zst_path]:
            if os.path.exists(p):
                os.unlink(p)

    def test_zstd_samples_range_returns_single_chunk(self, client, zstd_test_file):
        """Test that line range on zstd file returns a single chunk with range key.

        This is the main regression test: previously ranges like 1-100 would
        return empty results because the web API only handled single lines.
        """
        response = client.get('/v1/samples', params={'path': zstd_test_file, 'lines': '1-10'})

        assert response.status_code == 200, f'Got {response.status_code}: {response.text}'
        data = response.json()

        # Critical assertion: should have '1-10' as a key, not be empty
        assert '1-10' in data['samples'], f"Expected '1-10' key in samples, got: {list(data['samples'].keys())}"

        # Should NOT have individual line keys
        assert '1' not in data['samples'], 'Should not have individual line keys'
        assert '2' not in data['samples'], 'Should not have individual line keys'

        # Should have 10 lines in the range
        assert len(data['samples']['1-10']) == 10, f'Expected 10 lines, got {len(data["samples"]["1-10"])}'

        # Verify content
        assert 'Line 1:' in data['samples']['1-10'][0]
        assert 'Line 10:' in data['samples']['1-10'][9]

    def test_zstd_samples_mixed_single_and_range(self, client, zstd_test_file):
        """Test mixing single line and range on zstd file."""
        response = client.get('/v1/samples', params={'path': zstd_test_file, 'lines': '5,20-25', 'context': 2})

        assert response.status_code == 200
        data = response.json()

        # Should have both single line and range keys
        assert '5' in data['samples'], f"Expected '5' key in samples, got: {list(data['samples'].keys())}"
        assert '20-25' in data['samples'], f"Expected '20-25' key in samples, got: {list(data['samples'].keys())}"

        # Single line should have context
        assert len(data['samples']['5']) >= 1  # At least the line itself

        # Range should have exactly 6 lines (20-25)
        assert len(data['samples']['20-25']) == 6

    def test_zstd_samples_multiple_ranges(self, client, zstd_test_file):
        """Test multiple ranges on zstd file."""
        response = client.get('/v1/samples', params={'path': zstd_test_file, 'lines': '1-5,50-55,90-95'})

        assert response.status_code == 200
        data = response.json()

        # Should have all three range keys
        assert '1-5' in data['samples']
        assert '50-55' in data['samples']
        assert '90-95' in data['samples']

        # Each range should have correct number of lines
        assert len(data['samples']['1-5']) == 5
        assert len(data['samples']['50-55']) == 6
        assert len(data['samples']['90-95']) == 6

    def test_zstd_samples_is_compressed_flag(self, client, zstd_test_file):
        """Test that is_compressed and compression_format are set correctly."""
        response = client.get('/v1/samples', params={'path': zstd_test_file, 'lines': '1-5'})

        assert response.status_code == 200
        data = response.json()

        assert data['is_compressed'] is True
        assert data['compression_format'] == 'zstd'


class TestIndexEndpoint:
    """Tests for the /v1/index endpoint."""

    @pytest.fixture
    def large_test_file(self, temp_dir):
        """Create a test file large enough to be indexed (> 1MB)."""
        # Create a file that exceeds 1MB threshold
        # Each line is about 50 bytes, so 25000 lines = ~1.25MB
        lines = [f'Line {i:06d}: content for line number {i} with padding\n' for i in range(1, 25001)]
        temp_path = os.path.join(temp_dir, 'large_test.txt')
        with open(temp_path, 'w') as f:
            f.writelines(lines)
        yield temp_path
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_index_endpoint_returns_task_response_with_string_path(self, client, large_test_file):
        """Test that POST /v1/index returns TaskResponse with string path (not PosixPath).

        This tests that the path field in the response is a proper string,
        not a PosixPath object which would cause a Pydantic validation error.
        """
        # Force indexing regardless of size
        response = client.post('/v1/index', json={'path': large_test_file, 'force': True, 'threshold': 1})

        # Should return 200 with task info
        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert 'task_id' in data
        assert 'status' in data
        assert 'path' in data
        assert 'message' in data

        # Critical: path should be a string, not cause serialization issues
        assert isinstance(data['path'], str)
        assert data['path'] == large_test_file

    def test_index_endpoint_file_not_found(self, client, temp_dir):
        """Test that POST /v1/index returns 404 for non-existent file."""
        response = client.post('/v1/index', json={'path': os.path.join(temp_dir, 'nonexistent.txt')})
        assert response.status_code == 404

    def test_index_endpoint_file_too_small(self, client, temp_test_file):
        """Test that POST /v1/index returns 400 for file below threshold."""
        # Use default threshold which should be larger than our small test file
        response = client.post('/v1/index', json={'path': temp_test_file, 'threshold': 100})
        assert response.status_code == 400
        assert 'below threshold' in response.json()['detail']

    def test_index_endpoint_with_analyze_flag(self, client, large_test_file):
        """Test that POST /v1/index accepts analyze=true parameter.

        The analyze flag enables full analysis with anomaly detection.
        """
        response = client.post(
            '/v1/index',
            json={'path': large_test_file, 'force': True, 'threshold': 1, 'analyze': True},
        )

        # Should return 200 with task info
        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert 'task_id' in data
        assert 'status' in data
        assert data['status'] in ['queued', 'running']
        assert 'path' in data
        assert isinstance(data['path'], str)

    def test_index_endpoint_analyze_false_by_default(self, client, large_test_file):
        """Test that analyze defaults to false when not specified."""
        response = client.post('/v1/index', json={'path': large_test_file, 'force': True, 'threshold': 1})

        assert response.status_code == 200
        data = response.json()
        assert 'task_id' in data

    @pytest.fixture
    def file_with_errors(self, temp_dir):
        """Create a test file with error patterns for anomaly detection."""
        lines = []
        for i in range(1, 5001):
            if i == 100:
                lines.append('ERROR: Something went wrong\n')
            elif i == 500:
                lines.append('Traceback (most recent call last):\n')
                lines.append('  File "test.py", line 10, in <module>\n')
                lines.append('    raise ValueError("test error")\n')
                lines.append('ValueError: test error\n')
            elif i == 1000:
                lines.append('WARNING: Deprecated function called\n')
            else:
                lines.append(f'Line {i:06d}: normal log content here with padding\n')
        temp_path = os.path.join(temp_dir, 'file_with_errors.txt')
        with open(temp_path, 'w') as f:
            f.writelines(lines)
        yield temp_path
        if os.path.exists(temp_path):
            os.unlink(temp_path)
