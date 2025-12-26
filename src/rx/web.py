import asyncio
import logging
import os
import pathlib
import platform
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from time import time

import anyio
import psutil
import sh
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, RedirectResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from rx import file_utils as file_utils_module

# Import real prometheus for server mode and swap it into file_utils module
from rx import prometheus as prom
from rx import trace as trace_module
from rx.__version__ import __version__
from rx.compressed_index import get_decompressed_content_at_line, get_or_build_compressed_index
from rx.compression import CompressionFormat, detect_compression, is_compressed
from rx.file_utils import get_context, get_context_by_lines, is_text_file, validate_file
from rx.frontend_manager import ensure_frontend
from rx.hooks import (
    call_hook_async,
    generate_request_id,
    get_effective_hooks,
    get_hook_env_config,
)
from rx.index import (
    calculate_exact_line_for_offset,
    calculate_exact_offset_for_line,
    create_index_file,
    get_index_path,
    get_large_file_threshold_bytes,
    is_index_valid,
    load_index,
)
from rx.indexer import FileIndexer
from rx.models import (
    ComplexityResponse,
    CompressRequest,
    IndexRequest,
    Match,
    RequestInfo,
    SamplesResponse,
    TaskResponse,
    TaskStatusResponse,
    TraceCompletePayload,
    TraceResponse,
    TreeEntry,
    TreeResponse,
)
from rx.path_security import (
    get_search_roots,
    set_search_root,
    set_search_roots,
    validate_path_within_root,
    validate_paths_within_root,
)
from rx.regex import calculate_regex_complexity
from rx.request_store import store_request, update_request
from rx.seekable_index import build_index as build_seekable_index
from rx.seekable_zstd import create_seekable_zstd, is_seekable_zstd
from rx.task_manager import TaskManager, TaskStatus
from rx.trace import HookCallbacks, parse_paths
from rx.utils import NEWLINE_SYMBOL, get_rx_cache_base


# Replace the noop prometheus in file_utils module with real one
file_utils_module.prom = prom

log_level_name = os.getenv('RX_LOG_LEVEL', 'INFO').upper()
log_level = getattr(logging, log_level_name, logging.INFO)
logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize search roots from environment variable
    # This is set by the serve CLI command, or defaults to cwd
    # Multiple roots are separated by os.pathsep (: on Unix, ; on Windows)
    search_roots_env = os.getenv('RX_SEARCH_ROOTS')
    if search_roots_env:
        roots_list = [r.strip() for r in search_roots_env.split(os.pathsep) if r.strip()]
        search_roots = set_search_roots(roots_list)
    else:
        # Fall back to single root env var for backwards compatibility
        search_root_env = os.getenv('RX_SEARCH_ROOT')
        if search_root_env:
            search_root = set_search_root(search_root_env)
            search_roots = [search_root]
        else:
            search_root = set_search_root(None)  # Defaults to cwd
            search_roots = [search_root]
    app.state.search_roots = search_roots
    if len(search_roots) == 1:
        logger.info(f'Search root: {search_roots[0]}')
    else:
        logger.info(f'Search roots ({len(search_roots)}): {", ".join(str(r) for r in search_roots)}')

    # Startup: check for ripgrep
    try:
        rg = sh.Command('rg')
        rg_path = rg._path
        logger.info(f'ripgrep found at: {rg_path}')
        app.state.rg = rg
        app.state.rg_path = rg_path
    except sh.CommandNotFound:
        logger.warning(
            'ripgrep not found. Install it:\n'
            '  macOS: brew install ripgrep\n'
            '  Ubuntu/Debian: apt install ripgrep\n'
            '  Fedora: dnf install ripgrep\n'
            '  Or use Docker image with ripgrep pre-installed'
        )
        app.state.rg = None
        app.state.rg_path = None

    # Startup: initialize task manager for background tasks
    app.state.task_manager = TaskManager()
    logger.info('Task manager initialized')

    logger.info('Checking for frontend updates...')
    try:
        frontend_available = await ensure_frontend()
        if not frontend_available:
            logger.error('Failed to ensure frontend availability - web UI may not work')
            app.state.frontend_available = False
        else:
            app.state.frontend_available = True
            logger.info('Frontend is ready')
    except Exception as e:
        logger.error(f'Error during frontend check: {e}')
        app.state.frontend_available = False

    # Startup: schedule periodic task cleanup
    async def cleanup_tasks_periodically():
        while True:
            await asyncio.sleep(300)  # Every 5 minutes
            count = await app.state.task_manager.cleanup_old_tasks()
            if count > 0:
                logger.info(f'Cleaned up {count} old tasks')

    cleanup_task = asyncio.create_task(cleanup_tasks_periodically())

    # Run app
    yield

    # Shutdown: cancel cleanup task
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

    # Shutdown: cleanup if needed
    logger.info('Shutting down rx')


app = FastAPI(
    title='RX (Regex Tracer)',
    version=__version__,
    description="""
    A high-performance file search and analysis tool powered by ripgrep.

    ## Features

    * **Fast Pattern Matching**: Search files using powerful regex patterns with parallel processing
    * **Context Extraction**: Get surrounding lines for matched patterns
    * **Complexity Analysis**: Analyze regex patterns for performance characteristics and ReDoS risks
    * **Byte Offset Results**: Get precise byte offsets for all matches

    ## Endpoints

    * `/v1/trace` - Search for regex patterns in files with optional result limits
    * `/v1/samples` - Extract context lines around specific byte offsets
    * `/v1/complexity` - Analyze regex complexity and detect ReDoS vulnerabilities
    * `/health` - Check service health and ripgrep availability

    ## Performance

    - Handles files of any size using parallel processing
    - Splits large files (>25MB) into chunks (max 20 parallel processes)
    - Early termination support with `max_results` parameter
    - Line-aligned chunk boundaries to prevent pattern splitting

    ## Use Cases

    1. **Log Analysis**: Search multi-GB log files for error patterns
    2. **Code Search**: Find patterns across large codebases
    3. **Security Auditing**: Detect sensitive data patterns in files
    4. **Regex Testing**: Analyze regex complexity before production use
    """,
    contact={
        'name': 'RxTracer API Support',
        'url': 'https://github.com/wlame/rx-tool',
    },
    license_info={'name': 'MIT'},
    lifespan=lifespan,
    docs_url='/docs',
    redoc_url='/redoc',
)


@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    # Serve the favicon (SVG format)

    favicon_path = pathlib.Path(__file__).parent / 'favicon.svg'
    if favicon_path.exists():
        return Response(content=favicon_path.read_bytes(), media_type='image/svg+xml')
    return Response(status_code=204)


def get_os_info() -> dict:
    return {
        'system': platform.system(),
        'release': platform.release(),
        'version': platform.version(),
        'machine': platform.machine(),
    }


def get_system_resources() -> dict:
    mem = psutil.virtual_memory()
    return {
        'cpu_cores': psutil.cpu_count(logical=True),
        'cpu_cores_physical': psutil.cpu_count(logical=False),
        'ram_total_gb': round(mem.total / (1024**3), 2),
        'ram_available_gb': round(mem.available / (1024**3), 2),
        'ram_percent_used': mem.percent,
    }


def get_python_packages() -> dict:
    import importlib.metadata

    python_packages = {}
    key_packages = ['fastapi', 'pydantic', 'uvicorn', 'sh', 'psutil', 'prometheus-client']
    for package in key_packages:
        try:
            version = importlib.metadata.version(package)
            python_packages[package] = version
        except importlib.metadata.PackageNotFoundError:
            pass

    return python_packages


def get_constants() -> dict:
    # Collect application constants
    return {
        'LOG_LEVEL': log_level_name,
        'DEBUG_MODE': trace_module.DEBUG_MODE,
        'LINE_SIZE_ASSUMPTION_KB': file_utils_module.LINE_SIZE_ASSUMPTION_KB,
        'MAX_SUBPROCESSES': file_utils_module.MAX_SUBPROCESSES,
        'MIN_CHUNK_SIZE_MB': file_utils_module.MIN_CHUNK_SIZE // (1024 * 1024),
        'MAX_FILES': file_utils_module.MAX_FILES,
        'NEWLINE_SYMBOL': repr(NEWLINE_SYMBOL),  # Show as repr to see escape sequences
        'CACHE_DIR': str(get_rx_cache_base()),  # Effective cache directory
    }


def get_app_env_variables() -> dict:
    env_vars = {}
    app_env_prefixes = ['RX_', 'UVICORN_', 'PROMETHEUS_', 'NEWLINE_SYMBOL']
    for key, value in os.environ.items():
        if key == 'NEWLINE_SYMBOL' or any(key.startswith(prefix) for prefix in app_env_prefixes):
            env_vars[key] = value

    return env_vars


@app.get('/health', tags=['General'])
async def health():
    """
    Health check and system introspection endpoint.

    Returns:
    - Service status
    - ripgrep availability
    - Application version
    - Operating system information
    - Application-related environment variables
    - Hook configuration (from environment variables)
    - Documentation URL
    """
    os_info = get_os_info()
    system_resources = get_system_resources()
    python_packages = get_python_packages()
    constants = get_constants()
    env_vars = get_app_env_variables()
    hooks_config = get_hook_env_config()

    # Record metrics
    prom.record_http_response('GET', '/', 200)

    # Get search roots
    search_roots = get_search_roots()

    return {
        'status': 'ok',
        'ripgrep_available': app.state.rg_path is not None,
        'search_roots': [str(r) for r in search_roots] if search_roots else None,
        'app_version': __version__,
        'python_version': platform.python_version(),
        'os_info': os_info,
        'system_resources': system_resources,
        'python_packages': python_packages,
        'constants': constants,
        'environment': env_vars,
        'hooks': hooks_config,
        'docs_url': 'https://github.com/wlame/rx-tool',
    }


@app.get('/metrics', tags=['Monitoring'], include_in_schema=True)
async def metrics():
    """
    Prometheus metrics endpoint.

    Exposes performance metrics, request counts, file processing statistics,
    and resource utilization for monitoring and observability.

    **Metrics Categories:**
    - Request metrics (counts, durations by endpoint)
    - File processing (sizes, counts, bytes processed)
    - Pattern matching (matches found, patterns per request)
    - Regex complexity (scores, levels)
    - Parallel processing (tasks, workers)
    - Errors (by type)

    Use with Prometheus to scrape and visualize RX performance.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get(
    '/v1/trace',
    tags=['Search'],
    summary='Search file for regex patterns (supports multiple patterns)',
    response_model=TraceResponse,
    responses={
        200: {'description': 'Successfully found matches'},
        400: {'description': 'Invalid regex pattern or binary file'},
        404: {'description': 'File not found'},
        503: {'description': 'ripgrep not available'},
    },
)
async def trace(
    path: list[str] = Query(
        ...,
        description='File or directory paths to search (can specify multiple)',
        examples=['/var/log/app.log', '/var/log/nginx'],
    ),
    regexp: list[str] = Query(
        ..., description='Regular expression patterns to search for (can specify multiple)', examples=['error.*failed']
    ),
    max_results: int | None = Query(
        None, description='Maximum number of results to return (optional)', ge=1, examples=[100]
    ),
    request_id: str | None = Query(
        None,
        description='Custom request ID (UUID v7 auto-generated if not provided)',
        examples=['01936c8e-7b2a-7000-8000-000000000001'],
    ),
    hook_on_file: str | None = Query(
        None,
        description='URL to call (GET) when file scan completes',
        examples=['https://example.com/hooks/file-scanned'],
    ),
    hook_on_match: str | None = Query(
        None,
        description='URL to call (GET) for each match found. Requires max_results to be set.',
        examples=['https://example.com/hooks/match-found'],
    ),
    hook_on_complete: str | None = Query(
        None,
        description='URL to call (GET) when trace completes',
        examples=['https://example.com/hooks/trace-complete'],
    ),
) -> TraceResponse:
    """
    Search files or directories for regex pattern matches and return byte offsets with ID-based structure.

    This endpoint uses parallel processing with ripgrep to efficiently search large files or multiple files in a directory.
    Supports multiple paths and multiple patterns in a single request.

    - **path**: Absolute or relative path(s) to file or directory to search - can specify multiple
    - **regexp**: Regular expression pattern(s) (ripgrep syntax) - can specify multiple
    - **max_results**: Optional limit on number of results (stops early for performance)

    Returns ID-based structure:
    - **patterns**: Dict mapping pattern IDs (p1, p2, ...) to pattern strings
    - **files**: Dict mapping file IDs (f1, f2, ...) to file paths
    - **matches**: Array of {pattern, file, offset} objects using IDs
    - **scanned_files**: List of files that were scanned (for directories)
    - **skipped_files**: List of files that were skipped (binary files)

    Examples:
    ```
    GET /v1/trace?path=/var/log/app.log&regexp=error&regexp=warning
    GET /v1/trace?path=/var/log/app.log&path=/var/log/error.log&regexp=error
    ```

    Returns matches where any of the patterns were found across all paths.
    """

    if not app.state.rg:
        prom.record_error('service_unavailable')
        prom.record_http_response('GET', '/v1/trace', 503)
        raise HTTPException(status_code=503, detail='ripgrep is not available on this system')

    # Validate paths are within search root (security check)
    try:
        validated_paths = validate_paths_within_root(path)
        # Use validated (resolved) paths for actual operations
        path = [str(p) for p in validated_paths]
    except PermissionError as e:
        prom.record_error('access_denied')
        prom.record_http_response('GET', '/v1/trace', 403)
        raise HTTPException(status_code=403, detail=str(e))

    # Get effective hook configuration (respects RX_DISABLE_CUSTOM_HOOKS)
    hooks_config = get_effective_hooks(hook_on_file, hook_on_match, hook_on_complete)

    # Validate: max_results is required when hook_on_match is configured
    if hooks_config.has_match_hook() and max_results is None:
        prom.record_error('invalid_params')
        prom.record_http_response('GET', '/v1/trace', 400)
        raise HTTPException(
            status_code=400,
            detail='max_results is required when hook_on_match is configured. '
            'This prevents accidentally triggering millions of HTTP calls.',
        )

    # Check if all paths exist
    for p in path:
        if not os.path.exists(p):
            prom.record_error('file_not_found')
            prom.record_http_response('GET', '/v1/trace', 404)
            raise HTTPException(status_code=404, detail=f'Path not found: {p}')

    # Generate or use provided request_id
    req_id = request_id or generate_request_id()

    # Store request info
    request_info = RequestInfo(
        request_id=req_id,
        paths=path,
        patterns=regexp,
        max_results=max_results,
        started_at=datetime.now(),
    )
    store_request(request_info)

    # Get the current event loop to schedule async hook calls from thread pool
    main_loop = asyncio.get_running_loop()

    # Create hook callbacks for calls during parsing (runs in thread pool)
    def on_match_callback(payload: dict) -> None:
        """Callback that schedules async hook on main event loop (non-blocking)."""
        if hooks_config.on_match_url:
            # Schedule async hook on main loop - doesn't block the thread
            main_loop.call_soon_threadsafe(
                lambda: main_loop.create_task(call_hook_async(hooks_config.on_match_url, payload, 'on_match'))
            )

    def on_file_callback(payload: dict) -> None:
        """Callback that schedules async hook on main event loop (non-blocking)."""
        if hooks_config.on_file_url:
            # Schedule async hook on main loop - doesn't block the thread
            main_loop.call_soon_threadsafe(
                lambda: main_loop.create_task(call_hook_async(hooks_config.on_file_url, payload, 'on_file'))
            )

    # Build HookCallbacks if any hooks are configured
    hook_callbacks = None
    if hooks_config.has_any_hook():
        hook_callbacks = HookCallbacks(
            on_match_found=on_match_callback if hooks_config.on_match_url else None,
            on_file_scanned=on_file_callback if hooks_config.on_file_url else None,
            request_id=req_id,
        )

    # Parse files or directories with multiple patterns
    try:
        time_before = time()
        # Offload blocking I/O to thread pool to keep event loop responsive
        result = await anyio.to_thread.run_sync(
            partial(
                parse_paths,
                paths=path,
                regexps=regexp,
                max_results=max_results,
                rg_extra_args=None,
                context_before=0,  # No context in API by default (use /v1/samples for that)
                context_after=0,
                hooks=hook_callbacks,
            )
        )
        parsing_time = time() - time_before

        # Calculate metrics
        num_files = len(result.files)
        num_skipped = len(result.skipped_files)
        num_patterns = len(result.patterns)
        num_matches = len(result.matches)

        # Calculate total bytes processed
        total_bytes = 0
        for filepath in result.files.values():
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                total_bytes += file_size
                prom.record_file_size(file_size)

        # Update request info with results
        update_request(
            req_id,
            completed_at=datetime.now(),
            total_matches=num_matches,
            total_files_scanned=num_files,
            total_files_skipped=num_skipped,
            total_time_ms=int(parsing_time * 1000),
        )

        # Call on_complete hook if configured (fire and forget - don't block response)
        if hooks_config.on_complete_url:
            complete_payload = TraceCompletePayload(
                request_id=req_id,
                paths=','.join(path),
                patterns=','.join(regexp),
                total_files_scanned=num_files,
                total_files_skipped=num_skipped,
                total_matches=num_matches,
                total_time_ms=int(parsing_time * 1000),
            ).model_dump()
            # Schedule async hook - don't await, let it run in background
            asyncio.create_task(call_hook_async(hooks_config.on_complete_url, complete_payload, 'on_complete'))

        # Record metrics
        hit_max_results = max_results is not None and num_matches >= max_results
        prom.record_trace_request(
            status='success',
            duration=parsing_time,
            num_files=num_files,
            num_skipped=num_skipped,
            num_patterns=num_patterns,
            num_matches=num_matches,
            total_bytes=total_bytes,
            hit_max_results=hit_max_results,
        )
        prom.record_http_response('GET', '/v1/trace', 200)

    except RuntimeError as e:
        prom.record_error('invalid_regex')
        prom.record_trace_request('error', 0, 0, 0, len(regexp), 0, 0)
        prom.record_http_response('GET', '/v1/trace', 400)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f'Unexpected error: {e!s}')
        prom.record_error('internal_error')
        prom.record_trace_request('error', 0, 0, 0, len(regexp), 0, 0)
        prom.record_http_response('GET', '/v1/trace', 500)
        raise HTTPException(status_code=500, detail=f'Internal error: {e!s}')

    # Build response using ID-based structure
    response = TraceResponse(
        request_id=req_id,
        path=path,  # Pass as list directly
        time=parsing_time,
        patterns=result.patterns,
        files=result.files,
        matches=[Match(**m) for m in result.matches],
        scanned_files=result.scanned_files,
        skipped_files=result.skipped_files,
        file_chunks=result.file_chunks,
        max_results=max_results,  # Include max_results in response
    )

    return response


@app.get(
    '/v1/complexity',
    tags=['Analysis'],
    summary='EXPERIMENTAL! Analyze regex complexity',
    response_model=ComplexityResponse,
    responses={
        200: {'description': 'Complexity analysis completed'},
        500: {'description': 'Internal error during analysis'},
    },
)
async def complexity(
    regex: str = Query(
        ..., description='Regular expression pattern to analyze', examples=['(a+)+', '.*.*', '^[a-z]+$']
    ),
) -> dict:
    """
    Analyze regex pattern complexity and predict performance characteristics.

    This endpoint calculates a complexity score based on various regex features that
    can impact performance, particularly patterns that may cause catastrophic backtracking
    (ReDoS vulnerabilities).

    **Use this before running expensive searches to assess potential performance issues.**

    Returns:
    - **score**: Numeric complexity score
    - **level**: Complexity level (very_simple, simple, moderate, complex, very_complex, dangerous)
    - **risk**: Risk description
    - **warnings**: List of potential performance issues
    - **details**: Breakdown of scoring components
    - **regex**: The analyzed pattern

    Score ranges:
    - 0-10: Very Simple (substring search)
    - 11-30: Simple (basic patterns)
    - 31-60: Moderate (reasonable performance)
    - 61-100: Complex (monitor performance)
    - 101-200: Very Complex (significant impact)
    - 201+: Dangerous (ReDoS risk!)
    """
    try:
        time_before = time()
        # Offload CPU-bound regex analysis to thread pool
        result = await anyio.to_thread.run_sync(calculate_regex_complexity, regex)
        duration = time() - time_before

        result['regex'] = regex  # Include the regex pattern in response

        # Record metrics
        prom.record_complexity_request(duration=duration, score=result['score'], level=result['level'])
        prom.record_http_response('GET', '/v1/complexity', 200)

        return result
    except Exception as e:
        logger.error(f'Error analyzing regex complexity: {e!s}')
        prom.record_error('internal_error')
        prom.record_http_response('GET', '/v1/complexity', 500)
        raise HTTPException(status_code=500, detail=f'Internal error: {e!s}')


@app.get(
    '/v1/index',
    tags=['Indexing'],
    summary='Index files with optional analysis',
    responses={
        200: {'description': 'File indexing completed'},
        404: {'description': 'Path not found'},
        500: {'description': 'Internal error during indexing'},
    },
)
async def index_files(
    path: str | list[str] = Query(..., description='File or directory path(s) to index', examples=['/var/log/app.log']),
    analyze: bool = Query(False, description='Run full analysis with anomaly detection (indexes ALL files)'),
    force: bool = Query(False, description='Force rebuild even if valid index exists'),
    recursive: bool = Query(True, description='Recursively process directories'),
    max_workers: int = Query(10, description='Maximum number of parallel workers', ge=1, le=50, examples=[10]),
) -> dict:
    """
    Index files with optional analysis and anomaly detection.

    This endpoint creates unified file indexes that enable efficient line-based
    access to large text files.

    **Index behavior:**
    - Without **analyze=true**: Only indexes files >= 50MB
    - With **analyze=true**: Indexes ALL files with full analysis

    **With analyze=true, the endpoint also detects:**
    - Python/Java/JavaScript/Go/Rust stack traces
    - Lines containing ERROR, FATAL, Exception keywords
    - Unusually long lines (statistical outliers)
    - High-entropy strings (potential secrets)
    - Format deviations from dominant log pattern

    **Returns:**
    - **indexed**: List of successfully indexed files with metadata
    - **skipped**: Files skipped (below threshold or not text)
    - **errors**: Files that failed to index
    - **total_time**: Total indexing time in seconds
    """
    try:
        # Convert single path to list
        paths = path if isinstance(path, list) else [path]

        # Validate paths are within search root (security check)
        try:
            validated_paths = validate_paths_within_root(paths)
            paths = [str(p) for p in validated_paths]
        except PermissionError as e:
            prom.record_error('access_denied')
            prom.record_http_response('GET', '/v1/index', 403)
            raise HTTPException(status_code=403, detail=str(e))

        # Check paths exist
        for p in paths:
            if not os.path.exists(p):
                prom.record_error('not_found')
                prom.record_http_response('GET', '/v1/index', 404)
                raise HTTPException(status_code=404, detail=f'Path not found: {p}')

        # Create indexer and run
        indexer = FileIndexer(analyze=analyze, force=force)

        # Offload blocking file indexing to thread pool
        result = await anyio.to_thread.run_sync(
            partial(
                indexer.index_paths,
                paths=paths,
                recursive=recursive,
                max_workers=max_workers,
            )
        )

        # Build response
        response = {
            'indexed': [
                {
                    'path': idx.source_path,
                    'file_type': idx.file_type.value,
                    'size_bytes': idx.source_size_bytes,
                    'line_count': idx.line_count,
                    'index_entries': len(idx.line_index),
                    'analysis_performed': idx.analysis_performed,
                    'build_time_seconds': idx.build_time_seconds,
                    'anomaly_count': len(idx.anomalies) if idx.anomalies else 0,
                    'anomaly_summary': idx.anomaly_summary,
                }
                for idx in result.indexed
            ],
            'skipped': result.skipped,
            'errors': [{'path': p, 'error': e} for p, e in result.errors],
            'total_time': result.total_time,
        }

        # Record metrics
        total_bytes = sum(idx.source_size_bytes for idx in result.indexed)
        prom.record_analyze_request(
            status='success',
            duration=result.total_time,
            num_files=len(result.indexed),
            num_skipped=len(result.skipped),
            total_bytes=total_bytes,
            num_workers=max_workers,
        )
        prom.record_http_response('GET', '/v1/index', 200)

        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error indexing files: {e!s}')
        prom.record_error('internal_error')
        prom.record_analyze_request(
            status='error', duration=0, num_files=0, num_skipped=0, total_bytes=0, num_workers=max_workers
        )
        prom.record_http_response('GET', '/v1/index', 500)
        raise HTTPException(status_code=500, detail=f'Internal error: {e!s}')


def parse_offset_or_range(value: str) -> tuple[int, int | None]:
    """Parse an offset string that can be either a single number or a range.

    Args:
        value: String like "100", "-5" (negative index), or "100-200" (range)

    Returns:
        Tuple of (start, end) where end is None for single values.
        Negative single values are allowed (will be converted later based on file size).
        Negative values in ranges are NOT allowed.

    Raises:
        ValueError: If the format is invalid
    """
    value = value.strip()

    # Try to parse as single integer first (handles both positive and negative numbers)
    try:
        num = int(value)
        # Allow negative numbers for single values (will be converted later based on file size)
        return (num, None)
    except ValueError:
        # If it contains a dash and isn't a simple negative number, try parsing as range
        if '-' in value and not value.startswith('-'):
            # Range format: "100-200"
            parts = value.split('-')
            if len(parts) != 2:
                raise ValueError(f'Invalid range format: {value}. Expected format: START-END')
            try:
                start = int(parts[0])
                end = int(parts[1])
            except ValueError:
                raise ValueError(f'Invalid range format: {value}. Both values must be integers')

            # Ranges cannot have negative values
            if start < 0 or end < 0:
                raise ValueError(f'Invalid range: {value}. Ranges cannot contain negative values')
            if start > end:
                raise ValueError(f'Invalid range: {value}. Start must be <= end')
            return (start, end)
        else:
            # Re-raise the original error for invalid integers
            raise ValueError(f'Invalid offset: {value}. Must be an integer or range (e.g., 100-200)')


def get_lines_for_byte_range(path: str, start_offset: int, end_offset: int, index) -> list[str]:
    """Get all lines that overlap with a byte offset range.

    Efficiently reads only the required portion of the file using seek.
    Returns lines from the first line containing start_offset to the last line containing end_offset.
    """
    from rx.index import calculate_line_info_for_offsets

    # Get line info for both offsets in a single pass
    line_infos = calculate_line_info_for_offsets(path, [start_offset, end_offset], index)

    start_info = line_infos.get(start_offset)
    end_info = line_infos.get(end_offset)

    if not start_info or not end_info or start_info.line_number == -1 or end_info.line_number == -1:
        return []

    # Now we can seek directly to the start line and read only what we need
    result = []
    with open(path, 'rb') as f:
        f.seek(start_info.line_start_offset)

        # Read from start line's beginning to end line's end
        bytes_to_read = end_info.line_end_offset - start_info.line_start_offset
        content = f.read(bytes_to_read)

    # Decode and split into lines
    decoded = content.decode('utf-8', errors='replace')
    # Split on newlines and strip trailing newline chars
    for line in decoded.split('\n'):
        # Handle \r\n line endings
        stripped = line.rstrip('\r')
        result.append(stripped)

    # Remove the last empty element if content ended with newline
    if result and result[-1] == '':
        result.pop()

    return result


def get_line_range(path: str, start_line: int, end_line: int, file_index: 'FileIndex | None' = None) -> list[str]:
    """Get lines from start_line to end_line (inclusive, 1-based).

    Uses the index to seek directly to the start line's byte offset for efficiency.
    Falls back to linear scan if no index is available.

    Args:
        path: File path
        start_line: Starting line number (1-based)
        end_line: Ending line number (1-based, inclusive)
        file_index: Optional pre-loaded FileIndex for efficient seeking

    Returns:
        List of lines in the range
    """
    # Validate line numbers
    if start_line < 1 or end_line < 1:
        return []

    # Try to use index for efficient seeking
    start_offset = None
    if file_index and file_index.line_index:
        # Find the closest checkpoint at or before start_line
        # line_index format: [[line_number, byte_offset], ...]
        for entry in reversed(file_index.line_index):
            checkpoint_line = entry[0]
            checkpoint_offset = entry[1]
            if checkpoint_line <= start_line:
                start_offset = checkpoint_offset
                checkpoint_start_line = checkpoint_line
                break

    result = []

    if start_offset is not None:
        # Efficient path: seek to checkpoint and scan from there
        with open(path, 'rb') as f:
            f.seek(start_offset)
            current_line = checkpoint_start_line
            for raw_line in f:
                if current_line > end_line:
                    break
                if current_line >= start_line:
                    result.append(raw_line.decode('utf-8', errors='replace').rstrip('\n\r'))
                current_line += 1
    else:
        # Fallback: linear scan from beginning (for small files without index)
        with open(path, encoding='utf-8', errors='replace') as f:
            for current_line, line in enumerate(f, 1):
                if current_line > end_line:
                    break
                if current_line >= start_line:
                    result.append(line.rstrip('\n\r'))

    return result


def get_total_line_count(path: str) -> int:
    """Get total line count for a file using cached data or by counting.

    Args:
        path: File path

    Returns:
        Total number of lines in the file
    """
    from rx.unified_index import load_index as load_unified_index

    # Try unified index cache first
    unified_idx = load_unified_index(path)
    if unified_idx and unified_idx.line_count:
        return unified_idx.line_count

    # Try old index cache
    if is_index_valid(path):
        index_path = get_index_path(path)
        file_index = load_index(index_path)
        if file_index and file_index.line_index:
            line_index = file_index.line_index
            # Get the last indexed line number and offset
            last_line, last_offset = line_index[-1]

            # Count remaining lines after the last indexed position
            remaining_lines = 0
            with open(path, 'rb') as f:
                f.seek(last_offset)
                for _ in f:
                    remaining_lines += 1

            return last_line + remaining_lines - 1

    # Fallback: count lines directly
    with open(path, 'rb') as f:
        return sum(1 for _ in f)


@app.get(
    '/v1/samples',
    tags=['Context'],
    summary='Get context lines around byte offsets or line numbers',
    response_model=SamplesResponse,
    responses={
        200: {'description': 'Successfully retrieved context lines'},
        400: {'description': 'Invalid offsets/lines format or negative context values'},
        404: {'description': 'File not found'},
        503: {'description': 'ripgrep not available'},
    },
)
async def samples(
    path: str = Query(..., description='File path to read from', examples=['/var/log/app.log']),
    offsets: str = Query(
        None,
        description='Comma-separated byte offsets or ranges. Single offset (1234) gets context. '
        'Negative offset (-100) counts from end of file. Range (1000-2000) gets exact lines covering those bytes.',
        examples=['123,456,789', '1000-2000', '-100'],
    ),
    lines: str = Query(
        None,
        description='Comma-separated line numbers or ranges (1-based). Single line (100) gets context. '
        'Negative line (-5) counts from end of file. Range (100-200) gets exact lines.',
        examples=['100,200,300', '100-200', '-1,-5'],
    ),
    context: int = Query(None, description='Number of context lines before and after (sets both)', ge=0, examples=[3]),
    before_context: int = Query(None, description='Number of lines before each offset', ge=0, examples=[2]),
    after_context: int = Query(None, description='Number of lines after each offset', ge=0, examples=[5]),
) -> dict:
    """
    Extract context lines around specific byte offsets or line numbers in a file.

    Use this endpoint to view the actual content around matches found by `/trace`.

    **Parameters:**
    - **path**: Path to the file
    - **offsets**: Comma-separated byte offsets or ranges - mutually exclusive with lines
      - Single offset: `123` - gets context lines around byte offset 123
      - Negative offset: `-100` - counts from end of file (-1 = last byte)
      - Range: `1000-2000` - gets all lines covering byte range, ignores context
    - **lines**: Comma-separated line numbers or ranges (1-based) - mutually exclusive with offsets
      - Single line: `100` - gets context lines around line 100
      - Negative line: `-1` - last line, `-5` - 5th from end
      - Range: `100-200` - gets exactly lines 100-200, ignores context
    - **context**: Set both before and after context (default: 3)
    - **before_context**: Override lines before (default: 3)
    - **after_context**: Override lines after (default: 3)

    **Notes:**
    - Ranges ignore context settings and return exact lines
    - Negative values are converted based on file size (bytes) or line count (lines)
    - Ranges cannot contain negative values

    **Examples:**
    ```
    GET /v1/samples?path=data.txt&offsets=100,200&context=2
    GET /v1/samples?path=data.txt&lines=50,100,150&context=3
    GET /v1/samples?path=data.txt&lines=100-200
    GET /v1/samples?path=data.txt&lines=-1,-5&context=2
    GET /v1/samples?path=data.txt&offsets=1000-5000
    ```
    """
    if not app.state.rg:
        prom.record_error('service_unavailable')
        prom.record_http_response('GET', '/v1/samples', 503)
        raise HTTPException(status_code=503, detail='ripgrep is not available on this system')

    # Validate path is within search root (security check)
    try:
        validated_path = validate_path_within_root(path)
        path = str(validated_path)
    except PermissionError as e:
        prom.record_error('access_denied')
        prom.record_http_response('GET', '/v1/samples', 403)
        raise HTTPException(status_code=403, detail=str(e))

    # Check if file is compressed
    file_is_compressed = await anyio.to_thread.run_sync(is_compressed, path)
    compression_format = CompressionFormat.NONE
    if file_is_compressed:
        compression_format = await anyio.to_thread.run_sync(detect_compression, path)

    # Validate mutual exclusivity of offsets and lines
    if offsets and lines:
        prom.record_error('invalid_params')
        prom.record_http_response('GET', '/v1/samples', 400)
        raise HTTPException(status_code=400, detail="Cannot use both 'offsets' and 'lines'. Provide only one.")

    if not offsets and not lines:
        prom.record_error('invalid_params')
        prom.record_http_response('GET', '/v1/samples', 400)
        raise HTTPException(status_code=400, detail="Must provide either 'offsets' or 'lines' parameter.")

    # Compressed files only support line mode
    if file_is_compressed and offsets:
        prom.record_error('invalid_params')
        prom.record_http_response('GET', '/v1/samples', 400)
        raise HTTPException(
            status_code=400,
            detail="Byte offsets are not supported for compressed files. Use 'lines' parameter instead.",
        )

    # Parse offsets or lines (supports ranges and negative values)
    parsed_values: list[tuple[int, int | None]] = []
    use_lines = False

    if offsets:
        try:
            parsed_values = [parse_offset_or_range(o) for o in offsets.split(',')]
        except ValueError as e:
            prom.record_error('invalid_offsets')
            prom.record_http_response('GET', '/v1/samples', 400)
            raise HTTPException(status_code=400, detail=f'Invalid offsets format: {e}')
    else:
        use_lines = True
        try:
            parsed_values = [parse_offset_or_range(ln) for ln in lines.split(',')]
        except ValueError as e:
            prom.record_error('invalid_lines')
            prom.record_http_response('GET', '/v1/samples', 400)
            raise HTTPException(status_code=400, detail=f'Invalid lines format: {e}')

    before = before_context if before_context is not None else context if context is not None else 3
    after = after_context if after_context is not None else context if context is not None else 3

    # Validate file (offload blocking disk I/O) - skip for compressed files
    if not file_is_compressed:
        try:
            await anyio.to_thread.run_sync(validate_file, path)
        except FileNotFoundError as e:
            prom.record_error('file_not_found')
            prom.record_http_response('GET', '/v1/samples', 404)
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            prom.record_error('binary_file')
            prom.record_http_response('GET', '/v1/samples', 400)
            raise HTTPException(status_code=400, detail=str(e))
    else:
        # For compressed files, just check existence
        if not os.path.exists(path):
            prom.record_error('file_not_found')
            prom.record_http_response('GET', '/v1/samples', 404)
            raise HTTPException(status_code=404, detail=f'File not found: {path}')

    # Get context (offload blocking file I/O)
    try:
        time_before = time()

        # Handle compressed files separately
        if file_is_compressed:
            # Build/load compressed index and get samples
            index_data = await anyio.to_thread.run_sync(get_or_build_compressed_index, path)

            context_data: dict[str, list[str]] = {}
            line_mapping: dict[str, int] = {}

            for start, end in parsed_values:
                if end is None:
                    # Single line - use context
                    lines_content = await anyio.to_thread.run_sync(
                        partial(
                            get_decompressed_content_at_line,
                            source_path=path,
                            line_number=start,
                            context_before=before,
                            context_after=after,
                            index_data=index_data,
                        )
                    )
                    context_data[str(start)] = lines_content
                    line_mapping[str(start)] = -1  # No byte offsets for compressed
                else:
                    # Range - get exact lines, no context
                    range_key = f'{start}-{end}'
                    range_lines = []
                    for line_num in range(start, end + 1):
                        lines_content = await anyio.to_thread.run_sync(
                            partial(
                                get_decompressed_content_at_line,
                                source_path=path,
                                line_number=line_num,
                                context_before=0,
                                context_after=0,
                                index_data=index_data,
                            )
                        )
                        if lines_content:
                            range_lines.extend(lines_content)
                    context_data[range_key] = range_lines
                    line_mapping[range_key] = -1

            num_items = len(parsed_values)
            duration = time() - time_before

            prom.record_samples_request(
                status='success', duration=duration, num_offsets=num_items, before_ctx=before, after_ctx=after
            )
            prom.record_http_response('GET', '/v1/samples', 200)

            return {
                'path': path,
                'offsets': {},
                'lines': line_mapping,
                'before_context': before,
                'after_context': after,
                'samples': context_data,
                'is_compressed': True,
                'compression_format': compression_format.value,
            }

        # Regular file handling
        # Load index once for mapping calculations
        index_path_val = get_index_path(path)
        file_index = await anyio.to_thread.run_sync(load_index, index_path_val)

        # Get file size for negative byte offset conversion
        file_size = os.path.getsize(path)

        # Get total line count for negative line offset conversion (only if needed)
        total_lines = None
        if use_lines:
            needs_total_lines = any(start < 0 for start, end in parsed_values if end is None)
            if needs_total_lines:
                total_lines = await anyio.to_thread.run_sync(get_total_line_count, path)

        context_data = {}
        offset_mapping = {}
        line_mapping = {}

        if use_lines:
            # Line offset mode - handle both single lines and ranges
            for start, end in parsed_values:
                # Convert negative line numbers to positive
                if end is None and start < 0:
                    # Negative single line - convert using total line count
                    # -1 means last line, -2 means second to last, etc.
                    start = total_lines + start + 1
                    if start < 1:
                        start = 1  # Clamp to first line

                if end is None:
                    # Single line - use context
                    line_context = await anyio.to_thread.run_sync(
                        partial(
                            get_context_by_lines,
                            filename=path,
                            line_numbers=[start],
                            before_context=before,
                            after_context=after,
                        )
                    )
                    context_data.update({str(k): v for k, v in line_context.items()})
                    byte_offset_val = await anyio.to_thread.run_sync(
                        partial(
                            calculate_exact_offset_for_line,
                            filename=path,
                            target_line=start,
                            index=file_index,
                        )
                    )
                    line_mapping[str(start)] = byte_offset_val
                else:
                    # Range - get exact lines, ignore context
                    range_key = f'{start}-{end}'
                    range_lines = await anyio.to_thread.run_sync(partial(get_line_range, path, start, end, file_index))
                    context_data[range_key] = range_lines
                    # For ranges, byte offset is not meaningful - use -1 to skip expensive calculation
                    line_mapping[range_key] = -1
        else:
            # Byte offset mode - handle both single offsets and ranges
            for start, end in parsed_values:
                # Convert negative offsets to positive
                if end is None and start < 0:
                    # Negative single offset - convert using file size
                    # -1 means last byte (file_size - 1), -2 means file_size - 2, etc.
                    start = file_size + start
                    if start < 0:
                        start = 0  # Clamp to start of file

                if end is None:
                    # Single offset - use context
                    offset_context = await anyio.to_thread.run_sync(
                        partial(
                            get_context,
                            filename=path,
                            offsets=[start],
                            before_context=before,
                            after_context=after,
                        )
                    )
                    context_data.update({str(k): v for k, v in offset_context.items()})
                    line_num = await anyio.to_thread.run_sync(
                        partial(
                            calculate_exact_line_for_offset,
                            filename=path,
                            target_offset=start,
                            index=file_index,
                        )
                    )
                    offset_mapping[str(start)] = line_num
                else:
                    # Range - get exact lines covering the byte range, ignore context
                    range_key = f'{start}-{end}'
                    range_lines = await anyio.to_thread.run_sync(
                        partial(get_lines_for_byte_range, path, start, end, file_index)
                    )
                    context_data[range_key] = range_lines
                    # For ranges, we store the start line number
                    start_line = await anyio.to_thread.run_sync(
                        partial(
                            calculate_exact_line_for_offset,
                            filename=path,
                            target_offset=start,
                            index=file_index,
                        )
                    )
                    offset_mapping[range_key] = start_line

        duration = time() - time_before
        num_items = len(parsed_values)

        # Record metrics
        prom.record_samples_request(
            status='success', duration=duration, num_offsets=num_items, before_ctx=before, after_ctx=after
        )
        prom.record_http_response('GET', '/v1/samples', 200)

        return {
            'path': path,
            'offsets': offset_mapping,
            'lines': line_mapping,
            'before_context': before,
            'after_context': after,
            'samples': context_data,
            'is_compressed': False,
            'compression_format': None,
        }
    except ValueError as e:
        prom.record_error('invalid_context')
        num_items = len(parsed_values)
        prom.record_samples_request('error', 0, num_items, before, after)
        prom.record_http_response('GET', '/v1/samples', 400)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f'Unexpected error: {e!s}')
        prom.record_error('internal_error')
        num_items = len(parsed_values)
        prom.record_samples_request('error', 0, num_items, before, after)
        prom.record_http_response('GET', '/v1/samples', 500)
        raise HTTPException(status_code=500, detail=f'Internal error: {e!s}')


# Helper function to parse frame size
def parse_frame_size(size_str: str) -> int:
    """Parse human-readable size string to bytes (e.g., '4M' -> 4194304)."""
    size_str = size_str.strip().upper()
    multipliers = {
        'B': 1,
        'K': 1024,
        'KB': 1024,
        'M': 1024 * 1024,
        'MB': 1024 * 1024,
        'G': 1024 * 1024 * 1024,
        'GB': 1024 * 1024 * 1024,
    }

    for suffix, mult in sorted(multipliers.items(), key=lambda x: -len(x[0])):
        if size_str.endswith(suffix):
            num_str = size_str[: -len(suffix)].strip()
            return int(float(num_str) * mult)

    return int(size_str)


# Background task runners
async def run_compress_task(
    task_id: str,
    task_manager: TaskManager,
    request: CompressRequest,
    normalized_path: str,
):
    """Run compression in background."""

    try:
        await task_manager.update_task(task_id, status=TaskStatus.RUNNING)
        logger.info(f'[Task {task_id}] Starting compression of {normalized_path}')

        # Determine output path
        output_path = request.output_path
        if not output_path:
            output_path = f'{normalized_path}.zst'

        # Parse frame size
        frame_size_bytes = parse_frame_size(request.frame_size)

        # Run compression in thread pool (blocking operation)
        start_time = time()
        result = await anyio.to_thread.run_sync(
            partial(
                create_seekable_zstd,
                input_path=normalized_path,
                output_path=output_path,
                frame_size_bytes=frame_size_bytes,
                compression_level=request.compression_level,
                threads=None,  # auto
                progress_callback=None,
            )
        )
        elapsed = time() - start_time

        # Build index if requested
        index_built = False
        total_lines = None
        if request.build_index and result:
            logger.info(f'[Task {task_id}] Building index for {output_path}')
            index_result = await anyio.to_thread.run_sync(
                partial(
                    build_seekable_index,
                    zst_path=output_path,
                    progress_callback=None,
                )
            )
            if index_result:
                index_built = True
                total_lines = index_result.total_lines

        # Build result dictionary
        task_result = {
            'success': True,
            'input_path': normalized_path,
            'output_path': output_path,
            'compressed_size': result.compressed_size if result else None,
            'decompressed_size': result.decompressed_size if result else None,
            'compression_ratio': result.compression_ratio if result else None,
            'frame_count': result.frame_count if result else None,
            'total_lines': total_lines,
            'index_built': index_built,
            'time_seconds': elapsed,
        }

        await task_manager.update_task(
            task_id,
            status=TaskStatus.COMPLETED,
            completed_at=datetime.now(UTC),
            result=task_result,
        )
        logger.info(f'[Task {task_id}] Compression completed in {elapsed:.2f}s')

    except Exception as e:
        logger.error(f'[Task {task_id}] Compression failed: {e!s}')
        await task_manager.update_task(
            task_id,
            status=TaskStatus.FAILED,
            completed_at=datetime.now(UTC),
            error=str(e),
        )


async def run_index_task(
    task_id: str,
    task_manager: TaskManager,
    request: IndexRequest,
    normalized_path: str,
):
    """Run indexing in background."""

    try:
        await task_manager.update_task(task_id, status=TaskStatus.RUNNING)
        logger.info(f'[Task {task_id}] Starting indexing of {normalized_path}')

        start_time = time()

        # Check if it's a seekable zstd file
        if is_seekable_zstd(normalized_path):
            # Build seekable zstd index
            result = await anyio.to_thread.run_sync(
                partial(
                    build_seekable_index,
                    zst_path=normalized_path,
                    progress_callback=None,
                )
            )
            index_path = get_index_path(normalized_path)  # This will need seekable version
            line_count = result.total_lines if result else None
            checkpoint_count = len(result.frames) if result else None
        else:
            # Build regular file index
            result = await anyio.to_thread.run_sync(
                partial(
                    create_index_file,
                    source_path=normalized_path,
                    force=request.force,
                )
            )
            index_path = get_index_path(normalized_path)
            line_count = result.get('analysis', {}).get('line_count') if result else None
            checkpoint_count = len(result.get('checkpoints', [])) if result else None

        elapsed = time() - start_time
        file_size = os.path.getsize(normalized_path)

        # Build result dictionary
        task_result = {
            'success': True,
            'path': normalized_path,
            'index_path': index_path,
            'line_count': line_count,
            'file_size': file_size,
            'checkpoint_count': checkpoint_count,
            'time_seconds': elapsed,
        }

        await task_manager.update_task(
            task_id,
            status=TaskStatus.COMPLETED,
            completed_at=datetime.now(UTC),
            result=task_result,
        )
        logger.info(f'[Task {task_id}] Indexing completed in {elapsed:.2f}s')

    except Exception as e:
        logger.error(f'[Task {task_id}] Indexing failed: {e!s}')
        await task_manager.update_task(
            task_id,
            status=TaskStatus.FAILED,
            completed_at=datetime.now(UTC),
            error=str(e),
        )


@app.post(
    '/v1/compress',
    tags=['Operations'],
    summary='Compress file to seekable zstd format (background task)',
    response_model=TaskResponse,
    responses={
        200: {'description': 'Compression task started'},
        400: {'description': 'Invalid parameters'},
        403: {'description': 'Path outside search roots'},
        404: {'description': 'File not found'},
        409: {'description': 'Compression already in progress for this file'},
    },
)
async def compress(request: CompressRequest):
    """
    Start background compression of a file to seekable zstd format.

    Returns immediately with a task ID. Use GET /v1/tasks/{task_id} to check progress.

    **Parameters:**
    - **path**: Input file path
    - **output_path**: Output .zst file path (default: input_path.zst)
    - **frame_size**: Target frame size (default: "4M")
    - **compression_level**: Zstd level 1-22 (default: 3)
    - **build_index**: Build line index after compression (default: true)
    - **force**: Overwrite existing output (default: false)

    **Returns:**
    Task information with task_id for status polling.
    """
    # Validate path security
    try:
        normalized_path = validate_path_within_root(request.input_path)
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))

    # Check if file exists
    if not os.path.exists(normalized_path):
        raise HTTPException(status_code=404, detail=f'File not found: {request.input_path}')

    # Check if output already exists (unless force)
    output_path = request.output_path if request.output_path else f'{normalized_path}.zst'
    if not request.force and os.path.exists(output_path):
        raise HTTPException(status_code=400, detail=f'Output file already exists: {output_path}')

    # Create task
    task, is_new = await app.state.task_manager.create_task(normalized_path, 'compress')

    if not is_new:
        # Task already running
        raise HTTPException(
            status_code=409, detail=f'Compression already in progress for {request.input_path} (task: {task.task_id})'
        )

    # Start background task
    asyncio.create_task(
        run_compress_task(
            task.task_id,
            app.state.task_manager,
            request,
            normalized_path,
        )
    )

    # Return task info
    return TaskResponse(
        task_id=task.task_id,
        status=task.status.value,
        message=f'Compression task started for {request.input_path}',
        path=normalized_path,
        started_at=task.started_at.isoformat() if task.started_at else None,
    )


@app.post(
    '/v1/index',
    tags=['Operations'],
    summary='Build line index for file (background task)',
    response_model=TaskResponse,
    responses={
        200: {'description': 'Indexing task started'},
        400: {'description': 'Invalid parameters'},
        403: {'description': 'Path outside search roots'},
        404: {'description': 'File not found'},
        409: {'description': 'Indexing already in progress for this file'},
    },
)
async def index(request: IndexRequest):
    """
    Start background indexing of a file.

    Returns immediately with a task ID. Use GET /v1/tasks/{task_id} to check progress.

    **Parameters:**
    - **path**: File path to index
    - **force**: Force rebuild even if valid index exists (default: false)
    - **threshold**: Minimum file size in MB to index (default: from RX_LARGE_TEXT_FILE_MB env)

    **Returns:**
    Task information with task_id for status polling.
    """

    # Validate path security
    try:
        normalized_path = validate_path_within_root(request.path)
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))

    # Check if file exists
    if not os.path.exists(normalized_path):
        raise HTTPException(status_code=404, detail=f'File not found: {request.path}')

    # Check file size threshold
    threshold_bytes = (request.threshold * 1024 * 1024) if request.threshold else get_large_file_threshold_bytes()
    file_size = os.path.getsize(normalized_path)
    if file_size < threshold_bytes:
        raise HTTPException(
            status_code=400, detail=f'File size {file_size} bytes is below threshold {threshold_bytes} bytes'
        )

    # Create task
    task, is_new = await app.state.task_manager.create_task(normalized_path, 'index')

    if not is_new:
        # Task already running
        raise HTTPException(
            status_code=409, detail=f'Indexing already in progress for {request.path} (task: {task.task_id})'
        )

    # Start background task
    asyncio.create_task(
        run_index_task(
            task.task_id,
            app.state.task_manager,
            request,
            normalized_path,
        )
    )

    # Return task info
    return TaskResponse(
        task_id=task.task_id,
        status=task.status.value,
        message=f'Indexing task started for {request.path}',
        path=str(normalized_path),
        started_at=task.started_at.isoformat() if task.started_at else None,
    )


@app.get(
    '/v1/tasks/{task_id}',
    tags=['Operations'],
    summary='Get status of background task',
    response_model=TaskStatusResponse,
    responses={
        200: {'description': 'Task status retrieved'},
        404: {'description': 'Task not found'},
    },
)
async def get_task_status(task_id: str):
    """
    Get the status of a background task (compress or index).

    **Parameters:**
    - **task_id**: UUID of the task

    **Returns:**
    Task status information including completion state and result.
    """
    task = await app.state.task_manager.get_task(task_id)

    if not task:
        raise HTTPException(status_code=404, detail=f'Task not found: {task_id}')

    return TaskStatusResponse(
        task_id=task.task_id,
        status=task.status.value,
        path=task.path,
        operation=task.operation,
        started_at=task.started_at.isoformat() if task.started_at else None,
        completed_at=task.completed_at.isoformat() if task.completed_at else None,
        error=task.error,
        result=task.result,
    )


# File Tree Endpoints


def get_entry_metadata(entry_path: str, entry_name: str, is_dir: bool) -> dict:
    """Get metadata for a file or directory entry.

    Args:
        entry_path: Full path to the entry
        entry_name: Name of the entry
        is_dir: Whether entry is a directory

    Returns:
        Dictionary with entry metadata
    """
    result = {
        'name': entry_name,
        'path': entry_path,
        'type': 'directory' if is_dir else 'file',
        'size': None,
        'size_human': None,
        'modified_at': None,
        'is_text': None,
        'is_compressed': None,
        'compression_format': None,
        'is_indexed': None,
        'line_count': None,
        'children_count': None,
    }

    try:
        stat_info = os.stat(entry_path)
        mtime = datetime.fromtimestamp(stat_info.st_mtime)
        result['modified_at'] = mtime.isoformat()

        if is_dir:
            # Count children for directories
            try:
                children = os.listdir(entry_path)
                result['children_count'] = len(children)
            except PermissionError:
                result['children_count'] = None
        else:
            # File-specific metadata
            file_size = stat_info.st_size
            result['size'] = file_size
            result['size_human'] = human_readable_size(file_size)

            # Check if compressed
            if is_compressed(entry_path):
                result['is_compressed'] = True
                compression_fmt = detect_compression(entry_path)
                result['compression_format'] = compression_fmt.value if compression_fmt else None
                # Compressed files are considered "text" for our purposes
                result['is_text'] = True
            else:
                result['is_compressed'] = False
                result['is_text'] = is_text_file(entry_path)

            # Check index status
            if is_index_valid(entry_path):
                result['is_indexed'] = True
                # Try to get line count from index
                try:
                    index_path = get_index_path(entry_path)
                    file_index = load_index(index_path)
                    if file_index and file_index.analysis:
                        result['line_count'] = file_index.analysis.line_count
                except Exception:
                    pass
            else:
                result['is_indexed'] = False

    except (OSError, PermissionError) as e:
        logger.debug(f'Error getting metadata for {entry_path}: {e}')

    return result


def human_readable_size(size_bytes: int) -> str:
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f'{size_bytes:.2f} {unit}'
        size_bytes /= 1024
    return f'{size_bytes:.2f} PB'


@app.get(
    '/v1/tree',
    tags=['FileTree'],
    summary='List directory contents',
    response_model=TreeResponse,
    responses={
        200: {'description': 'Directory listing retrieved'},
        403: {'description': 'Path outside search roots'},
        404: {'description': 'Path not found or not a directory'},
    },
)
async def tree(
    path: str | None = Query(
        None,
        description='Directory path to list. If not provided, lists search roots.',
        examples=['/var/log'],
    ),
) -> TreeResponse:
    """
    List contents of a directory within the search roots.

    If no path is provided, returns the list of configured search roots.

    **Parameters:**
    - **path**: Directory path to list (optional)

    **Returns:**
    - List of files and directories with metadata
    - File properties include: size, type (text/binary), compression, index status
    - Directory properties include: children count

    **Examples:**
    ```
    GET /v1/tree                    # List search roots
    GET /v1/tree?path=/var/log      # List /var/log contents
    ```
    """
    # If no path, return search roots
    if path is None:
        search_roots = get_search_roots()

        if not search_roots:
            prom.record_http_response('GET', '/v1/tree', 200)
            return TreeResponse(
                path='/',
                parent=None,
                is_search_root=True,
                entries=[],
                total_entries=0,
            )

        # Get metadata for each search root
        entries = []
        for root in search_roots:
            root_str = str(root)
            metadata = await anyio.to_thread.run_sync(get_entry_metadata, root_str, root.name, True)
            entries.append(TreeEntry(**metadata))

        prom.record_http_response('GET', '/v1/tree', 200)
        return TreeResponse(
            path='/',
            parent=None,
            is_search_root=True,
            entries=entries,
            total_entries=len(entries),
        )

    # Validate path is within search roots
    try:
        validated_path = validate_path_within_root(path)
        path = str(validated_path)
    except PermissionError as e:
        prom.record_error('access_denied')
        prom.record_http_response('GET', '/v1/tree', 403)
        raise HTTPException(status_code=403, detail=str(e))

    # Check path exists and is a directory
    if not os.path.exists(path):
        prom.record_error('not_found')
        prom.record_http_response('GET', '/v1/tree', 404)
        raise HTTPException(status_code=404, detail=f'Path not found: {path}')

    if not os.path.isdir(path):
        prom.record_error('not_a_directory')
        prom.record_http_response('GET', '/v1/tree', 400)
        raise HTTPException(status_code=400, detail=f'Path is not a directory: {path}')

    # List directory contents
    try:
        dir_entries = os.listdir(path)
    except PermissionError:
        prom.record_error('access_denied')
        prom.record_http_response('GET', '/v1/tree', 403)
        raise HTTPException(status_code=403, detail=f'Permission denied: {path}')

    # Sort entries: directories first, then files, alphabetically within each group
    dirs = []
    files = []
    for entry_name in dir_entries:
        entry_path = os.path.join(path, entry_name)
        if os.path.isdir(entry_path):
            dirs.append((entry_name, entry_path, True))
        else:
            files.append((entry_name, entry_path, False))

    dirs.sort(key=lambda x: x[0].lower())
    files.sort(key=lambda x: x[0].lower())
    sorted_entries = dirs + files

    # Get metadata for each entry (in parallel for performance)
    entries = []
    total_size = 0

    for entry_name, entry_path, is_dir in sorted_entries:
        metadata = await anyio.to_thread.run_sync(get_entry_metadata, entry_path, entry_name, is_dir)
        entries.append(TreeEntry(**metadata))
        if metadata.get('size'):
            total_size += metadata['size']

    # Determine parent path
    path_obj = Path(path)
    search_roots = get_search_roots()
    is_search_root = path_obj in search_roots

    if is_search_root:
        parent = None
    else:
        parent = str(path_obj.parent)

    prom.record_http_response('GET', '/v1/tree', 200)
    return TreeResponse(
        path=path,
        parent=parent,
        is_search_root=is_search_root,
        entries=entries,
        total_entries=len(entries),
        total_size=total_size if total_size > 0 else None,
        total_size_human=human_readable_size(total_size) if total_size > 0 else None,
    )


# Static file serving for frontend
# Use frontend_manager to get the frontend directory (from cache only)

from rx.frontend_manager import get_frontend_dir, get_frontend_manager


# Define routes that check for frontend dynamically (after startup downloads)
# Serve index.html at root
@app.get('/', include_in_schema=False)
async def serve_frontend():
    """Serve the frontend index.html or redirect to docs"""
    frontend_dir = get_frontend_dir()
    if frontend_dir and frontend_dir.exists():
        index_path = frontend_dir / 'index.html'
        if index_path.exists():
            return FileResponse(index_path)
    # Redirect to API docs if frontend not available
    return RedirectResponse(url='/docs')


# Catch-all route for SPA (must be last)
@app.get('/{full_path:path}', include_in_schema=False)
async def serve_spa(full_path: str):
    """Serve SPA for client-side routing (catch-all for non-API routes)

    Security: Validates paths to prevent directory traversal attacks like ../../etc/passwd
    """
    # Don't catch API routes or special paths
    if full_path.startswith(('v1/', 'health', 'metrics', 'docs', 'redoc', 'openapi.json')):
        raise HTTPException(status_code=404, detail=f'Not found: /{full_path}')

    # Check if frontend is available
    frontend_dir = get_frontend_dir()
    if not frontend_dir or not frontend_dir.exists():
        # No frontend available, redirect to docs
        return RedirectResponse(url='/docs')

    # Try to serve static file first with security validation
    manager = get_frontend_manager()
    validated_path = manager.validate_static_file_path(full_path)

    if validated_path:
        return FileResponse(validated_path)

    # Fall back to index.html for SPA routing (if no static file found)
    index_path = frontend_dir / 'index.html'
    if index_path.exists():
        return FileResponse(index_path)

    # Redirect to API docs if frontend not available
    return RedirectResponse(url='/docs')
