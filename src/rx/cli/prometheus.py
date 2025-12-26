"""No-op Prometheus stub for CLI usage (when --serve is not used)"""


class NoOpMetric:
    """No-op metric that accepts any method call and does nothing."""

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        return self

    def inc(self, *args, **kwargs):
        pass

    def dec(self, *args, **kwargs):
        pass

    def observe(self, *args, **kwargs):
        pass

    def labels(self, *args, **kwargs):
        return self

    def set(self, *args, **kwargs):
        pass


# Create no-op instances for all metrics
trace_requests_total = NoOpMetric()
samples_requests_total = NoOpMetric()
complexity_requests_total = NoOpMetric()
analyze_requests_total = NoOpMetric()

trace_duration_seconds = NoOpMetric()
samples_duration_seconds = NoOpMetric()
complexity_duration_seconds = NoOpMetric()
analyze_duration_seconds = NoOpMetric()
ripgrep_processing_seconds = NoOpMetric()

files_processed_total = NoOpMetric()
files_skipped_total = NoOpMetric()
file_size_bytes = NoOpMetric()
bytes_processed_total = NoOpMetric()

patterns_per_request = NoOpMetric()
matches_found_total = NoOpMetric()
matches_per_request = NoOpMetric()

regex_complexity_score = NoOpMetric()
regex_complexity_level = NoOpMetric()

parallel_tasks_created = NoOpMetric()
active_workers = NoOpMetric()
worker_tasks_completed = NoOpMetric()
worker_tasks_failed = NoOpMetric()

large_file_threshold_mb = NoOpMetric()
max_results_limited_total = NoOpMetric()

errors_total = NoOpMetric()
http_responses_total = NoOpMetric()

offsets_per_samples_request = NoOpMetric()
context_lines_before = NoOpMetric()
context_lines_after = NoOpMetric()

index_cache_hits_total = NoOpMetric()
index_cache_misses_total = NoOpMetric()
index_load_duration_seconds = NoOpMetric()
index_build_duration_seconds = NoOpMetric()

# Hook metrics
hook_calls_total = NoOpMetric()
hook_call_duration_seconds = NoOpMetric()

# Trace cache metrics
trace_cache_hits_total = NoOpMetric()
trace_cache_misses_total = NoOpMetric()
trace_cache_writes_total = NoOpMetric()
trace_cache_skip_total = NoOpMetric()
trace_cache_load_duration_seconds = NoOpMetric()
trace_cache_reconstruction_seconds = NoOpMetric()


# No-op helper functions
def record_trace_request(*args, **kwargs):
    pass


def record_samples_request(*args, **kwargs):
    pass


def record_complexity_request(*args, **kwargs):
    pass


def record_analyze_request(*args, **kwargs):
    pass


def record_file_size(*args, **kwargs):
    pass


def record_error(*args, **kwargs):
    pass


def record_http_response(*args, **kwargs):
    pass


def record_hook_call(*args, **kwargs):
    pass
