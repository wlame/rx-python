"""Prometheus metrics for RX (Regex Tracer)"""

from prometheus_client import Counter, Gauge, Histogram, Summary

# ============================================================================
# Request Metrics
# ============================================================================

# Total number of trace requests
trace_requests_total = Counter(
    'rx_trace_requests_total',
    'Total number of trace requests',
    ['status'],  # success, error, invalid_regex, file_not_found
)

# Total number of samples requests
samples_requests_total = Counter('rx_samples_requests_total', 'Total number of samples (context) requests', ['status'])

# Total number of complexity analysis requests
complexity_requests_total = Counter('rx_complexity_requests_total', 'Total number of complexity analysis requests')

# Total number of analyze requests
analyze_requests_total = Counter(
    'rx_analyze_requests_total',
    'Total number of file analysis requests',
    ['status'],  # success, error
)


# ============================================================================
# Performance Metrics
# ============================================================================

# Request duration in seconds
trace_duration_seconds = Histogram(
    'rx_trace_duration_seconds',
    'Time spent processing trace requests',
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
    # 10ms to 5 minutes - covers quick single-file searches to large directory scans
)

samples_duration_seconds = Histogram(
    'rx_samples_duration_seconds',
    'Time spent processing samples requests',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    # 1ms to 5s - samples are typically fast
)

complexity_duration_seconds = Histogram(
    'rx_complexity_duration_seconds',
    'Time spent analyzing regex complexity',
    buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
    # 0.1ms to 1s - complexity analysis is very fast
)

analyze_duration_seconds = Histogram(
    'rx_analyze_duration_seconds',
    'Time spent processing file analysis requests',
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
    # 10ms to 60s - file analysis can take time for large files or many files
)

# Ripgrep processing time (core search duration)
ripgrep_processing_seconds = Summary(
    'rx_ripgrep_processing_seconds', 'Time spent in ripgrep processing (parse_path function)'
)


# ============================================================================
# File Processing Metrics
# ============================================================================

# Number of files processed per request
files_processed_total = Counter('rx_files_processed_total', 'Total number of files processed across all requests')

files_skipped_total = Counter('rx_files_skipped_total', 'Total number of files skipped (binary files, errors, etc.)')

# File size distribution
file_size_bytes = Histogram(
    'rx_file_size_bytes',
    'Size of files being processed',
    buckets=[
        1024,  # 1KB
        10_240,  # 10KB
        102_400,  # 100KB
        1_048_576,  # 1MB
        10_485_760,  # 10MB
        104_857_600,  # 100MB
        524_288_000,  # 500MB
        1_073_741_824,  # 1GB
        5_368_709_120,  # 5GB
        10_737_418_240,  # 10GB
        26_843_545_600,  # 25GB
        53_687_091_200,  # 50GB
    ],
)

# Total bytes processed (cumulative)
bytes_processed_total = Counter('rx_bytes_processed_total', 'Total bytes processed across all files')


# ============================================================================
# Pattern & Match Metrics
# ============================================================================

# Number of patterns per request
patterns_per_request = Histogram(
    'rx_patterns_per_request', 'Number of regex patterns in a single request', buckets=[1, 2, 3, 5, 10, 20, 50, 100]
)

# Matches found per request
matches_found_total = Counter('rx_matches_found_total', 'Total number of matches found across all requests')

matches_per_request = Histogram(
    'rx_matches_per_request',
    'Number of matches found per request',
    buckets=[0, 1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000],
    # 0 to 1M matches
)


# ============================================================================
# Regex Complexity Metrics
# ============================================================================

# Regex complexity scores
regex_complexity_score = Histogram(
    'rx_regex_complexity_score',
    'Regex complexity scores from analysis',
    buckets=[1, 5, 10, 20, 30, 50, 60, 80, 100, 150, 200, 300, 500, 1000],
    # Very simple to dangerous patterns
)

# Count of patterns by complexity level
regex_complexity_level = Counter(
    'rx_regex_complexity_level_total',
    'Count of regex patterns by complexity level',
    ['level'],  # very_simple, simple, moderate, complex, very_complex, dangerous
)


# ============================================================================
# Parallel Processing Metrics
# ============================================================================

# Number of parallel tasks created per request
parallel_tasks_created = Histogram(
    'rx_parallel_tasks_created',
    'Number of parallel tasks created per request',
    buckets=[1, 2, 4, 8, 12, 16, 20, 24, 32, 48, 64, 100, 200],
)

# Active workers (gauge - current value)
active_workers = Gauge('rx_active_workers', 'Current number of active worker threads')

# Worker pool utilization (tasks completed)
worker_tasks_completed = Counter('rx_worker_tasks_completed_total', 'Total number of worker tasks completed')

worker_tasks_failed = Counter('rx_worker_tasks_failed_total', 'Total number of worker tasks that failed')


# ============================================================================
# Resource Usage Metrics
# ============================================================================

# Memory usage for large file processing (if we want to track)
large_file_threshold_mb = Gauge(
    'rx_large_file_threshold_mb', 'Threshold in MB for considering a file "large" for chunking'
)

# Max results limiting (how often is it triggered)
max_results_limited_total = Counter('rx_max_results_limited_total', 'Count of requests that hit max_results limit')


# ============================================================================
# Error Metrics
# ============================================================================

# Errors by type
errors_total = Counter(
    'rx_errors_total',
    'Total errors by type',
    ['error_type'],  # invalid_regex, file_not_found, binary_file, permission_error, internal_error
)

# HTTP status codes
http_responses_total = Counter(
    'rx_http_responses_total', 'HTTP responses by status code', ['method', 'endpoint', 'status_code']
)


# ============================================================================
# Context Extraction Metrics
# ============================================================================

# Number of offsets requested per samples call
offsets_per_samples_request = Histogram(
    'rx_offsets_per_samples_request',
    'Number of offsets requested in samples endpoint',
    buckets=[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000],
)

# Context lines requested
context_lines_before = Histogram(
    'rx_context_lines_before', 'Number of context lines before each match', buckets=[0, 1, 2, 3, 5, 10, 20, 50, 100]
)

context_lines_after = Histogram(
    'rx_context_lines_after', 'Number of context lines after each match', buckets=[0, 1, 2, 3, 5, 10, 20, 50, 100]
)


# ============================================================================
# Index Operation Metrics
# ============================================================================

# Index cache hits and misses
index_cache_hits_total = Counter('rx_index_cache_hits_total', 'Number of index cache hits')

index_cache_misses_total = Counter('rx_index_cache_misses_total', 'Number of index cache misses')

# Index load duration
index_load_duration_seconds = Histogram(
    'rx_index_load_duration_seconds',
    'Time to load index from disk',
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
)

# Index build duration
index_build_duration_seconds = Histogram(
    'rx_index_build_duration_seconds',
    'Time to build new index',
    buckets=[0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
)


# ============================================================================
# Hook Metrics
# ============================================================================

# Total hook calls by event type and status
hook_calls_total = Counter(
    'rx_hook_calls_total',
    'Total hook calls by event type and status',
    ['event_type', 'status'],  # event_type: on_file, on_match, on_complete; status: success, failed
)

# Hook call duration
hook_call_duration_seconds = Histogram(
    'rx_hook_call_duration_seconds',
    'Time spent calling hooks',
    ['event_type'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0],
    # 10ms to 5s - hooks have 3s timeout
)


# ============================================================================
# Trace Cache Metrics
# ============================================================================

# Trace cache hits and misses
trace_cache_hits_total = Counter('rx_trace_cache_hits_total', 'Number of trace cache hits')

trace_cache_misses_total = Counter('rx_trace_cache_misses_total', 'Number of trace cache misses')

trace_cache_writes_total = Counter('rx_trace_cache_writes_total', 'Number of trace cache writes')

trace_cache_skip_total = Counter(
    'rx_trace_cache_skip_total', 'Number of times trace cache was skipped (small file, max_results, etc.)'
)

# Trace cache load duration
trace_cache_load_duration_seconds = Histogram(
    'rx_trace_cache_load_duration_seconds',
    'Time to load trace cache from disk',
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
)

# Trace cache reconstruction duration
trace_cache_reconstruction_seconds = Histogram(
    'rx_trace_cache_reconstruction_seconds',
    'Time to reconstruct matches from trace cache',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0],
)


# ============================================================================
# Helper Functions
# ============================================================================


def record_trace_request(
    status: str,
    duration: float,
    num_files: int,
    num_skipped: int,
    num_patterns: int,
    num_matches: int,
    total_bytes: int,
    num_tasks: int = 0,
    hit_max_results: bool = False,
):
    """
    Record metrics for a trace request.

    Args:
        status: Request status (success, error, invalid_regex, file_not_found)
        duration: Request duration in seconds
        num_files: Number of files processed
        num_skipped: Number of files skipped
        num_patterns: Number of regex patterns
        num_matches: Number of matches found
        total_bytes: Total bytes processed
        num_tasks: Number of parallel tasks created
        hit_max_results: Whether request hit max_results limit
    """
    trace_requests_total.labels(status=status).inc()
    trace_duration_seconds.observe(duration)

    files_processed_total.inc(num_files)
    files_skipped_total.inc(num_skipped)

    patterns_per_request.observe(num_patterns)

    matches_found_total.inc(num_matches)
    matches_per_request.observe(num_matches)

    bytes_processed_total.inc(total_bytes)

    if num_tasks > 0:
        parallel_tasks_created.observe(num_tasks)

    if hit_max_results:
        max_results_limited_total.inc()


def record_samples_request(status: str, duration: float, num_offsets: int, before_ctx: int, after_ctx: int):
    """
    Record metrics for a samples request.

    Args:
        status: Request status (success, error)
        duration: Request duration in seconds
        num_offsets: Number of offsets requested
        before_ctx: Lines of before context
        after_ctx: Lines of after context
    """
    samples_requests_total.labels(status=status).inc()
    samples_duration_seconds.observe(duration)
    offsets_per_samples_request.observe(num_offsets)
    context_lines_before.observe(before_ctx)
    context_lines_after.observe(after_ctx)


def record_complexity_request(duration: float, score: int, level: str):
    """
    Record metrics for a complexity analysis request.

    Args:
        duration: Request duration in seconds
        score: Complexity score
        level: Complexity level
    """
    complexity_requests_total.inc()
    complexity_duration_seconds.observe(duration)
    regex_complexity_score.observe(score)
    regex_complexity_level.labels(level=level).inc()


def record_analyze_request(
    status: str, duration: float, num_files: int, num_skipped: int, total_bytes: int, num_workers: int
):
    """
    Record metrics for a file analysis request.

    Args:
        status: Request status (success, error)
        duration: Request duration in seconds
        num_files: Number of files analyzed
        num_skipped: Number of files skipped
        total_bytes: Total bytes analyzed
        num_workers: Number of parallel workers used
    """
    analyze_requests_total.labels(status=status).inc()
    analyze_duration_seconds.observe(duration)
    files_processed_total.inc(num_files)
    files_skipped_total.inc(num_skipped)
    bytes_processed_total.inc(total_bytes)
    if num_workers > 0:
        active_workers.set(num_workers)


def record_file_size(size_bytes: int):
    """Record the size of a file being processed."""
    file_size_bytes.observe(size_bytes)


def record_error(error_type: str):
    """
    Record an error occurrence.

    Args:
        error_type: Type of error (invalid_regex, file_not_found, binary_file, etc.)
    """
    errors_total.labels(error_type=error_type).inc()


def record_http_response(method: str, endpoint: str, status_code: int):
    """
    Record HTTP response.

    Args:
        method: HTTP method (GET, POST, etc.)
        endpoint: Endpoint path
        status_code: HTTP status code
    """
    http_responses_total.labels(method=method, endpoint=endpoint, status_code=str(status_code)).inc()


def record_hook_call(event_type: str, success: bool, duration: float):
    """
    Record metrics for a hook call.

    Args:
        event_type: Type of hook event ('on_file', 'on_match', 'on_complete')
        success: Whether the hook call succeeded
        duration: Duration of the hook call in seconds
    """
    status = 'success' if success else 'failed'
    hook_calls_total.labels(event_type=event_type, status=status).inc()
    hook_call_duration_seconds.labels(event_type=event_type).observe(duration)
