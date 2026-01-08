"""CLI command for file indexing and analysis."""

import json
import logging
import os
import sys

import click

from rx.indexer import FileIndexer
from rx.unified_index import delete_index, load_index


def _setup_logging(debug: bool = False):
    """Setup logging based on --debug flag or RX_LOG_LEVEL environment variable.

    Args:
        debug: If True, force DEBUG level logging with detailed format
    """
    if debug:
        log_level = logging.DEBUG
        log_format = '%(asctime)s.%(msecs)03d %(name)s [%(levelname)s] %(message)s'
    else:
        log_level_str = os.environ.get('RX_LOG_LEVEL', 'WARNING').upper()
        log_level = getattr(logging, log_level_str, logging.WARNING)
        log_format = '%(asctime)s %(name)s [%(levelname)s] %(message)s'

    # Configure root logger for rx modules
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt='%H:%M:%S',
        stream=sys.stderr,
    )

    # Set level for rx modules specifically
    logging.getLogger('rx').setLevel(log_level)

    # Suppress drain3 INFO messages unless DEBUG is requested
    if log_level > logging.DEBUG:
        logging.getLogger('drain3').setLevel(logging.WARNING)


def human_readable_size(size_bytes: int) -> str:
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f'{size_bytes:.2f} {unit}'
        size_bytes /= 1024
    return f'{size_bytes:.2f} PB'


@click.command('index')
@click.argument('paths', nargs=-1, required=True, type=click.Path(exists=True))
@click.option('--force', '-f', is_flag=True, help='Force rebuild even if valid index exists')
@click.option('--info', '-i', is_flag=True, help='Show index info without rebuilding')
@click.option('--delete', '-d', is_flag=True, help='Delete index for file(s)')
@click.option('--json', 'json_output', is_flag=True, help='Output in JSON format')
@click.option('--recursive', '-r', is_flag=True, help='Recursively process directories')
@click.option('--analyze', '-a', is_flag=True, help='Run full analysis with anomaly detection')
@click.option(
    '--threshold',
    type=int,
    default=None,
    help='Only index files larger than this (MB). Default: 50MB. Ignored with --analyze.',
)
@click.option('--max-workers', type=int, default=10, help='Maximum parallel workers (default: 10)')
@click.option('--debug', is_flag=True, help='Enable detailed debug logging for analysis')
def index_command(
    paths: tuple[str, ...],
    force: bool,
    info: bool,
    delete: bool,
    json_output: bool,
    recursive: bool,
    analyze: bool,
    threshold: int | None,
    max_workers: int,
    debug: bool,
):
    """Create or manage file indexes with optional analysis.

    Indexes enable efficient line-based access to large text files.
    With --analyze, runs full analysis including anomaly detection.

    \b
    Examples:
        rx index /path/to/large.log          # Create index for large files
        rx index /path/to/dir/ -r            # Index all large files in directory
        rx index /path/to/file.log --force   # Force rebuild
        rx index /path/to/file.log --info    # Show index info
        rx index /path/to/file.log --delete  # Delete index

    \b
    With analysis:
        rx index /path/to/file.log --analyze         # Full analysis + anomaly detection
        rx index /var/log/ -r --analyze              # Analyze all files recursively
        rx index /path/to/file.log --analyze --json  # JSON output (same as API)

    \b
    Index behavior:
        Without --analyze: Only indexes files >= 50MB (or threshold)
        With --analyze: Indexes ALL files with full analysis

    \b
    Environment variables:
        RX_LOG_LEVEL: Set logging level (DEBUG, INFO, WARNING, ERROR)

    \b
    Debug mode (--debug):
        Shows detailed timing for each detector, lines processed,
        anomalies found, and thread/worker statistics.
    """
    # Setup logging first
    _setup_logging(debug=debug)

    # Handle info and delete modes with existing logic
    if info or delete:
        _handle_info_or_delete(paths, info, delete, json_output, recursive)
        return

    # Use new FileIndexer for indexing
    indexer = FileIndexer(analyze=analyze, force=force)

    # Collect paths
    all_paths = list(paths)

    # Run indexing
    result = indexer.index_paths(all_paths, recursive=recursive, max_workers=max_workers)

    # Output results
    if json_output:
        _output_json(result)
    else:
        _output_human_readable(result, analyze)

    # Exit with error if any failures
    if result.errors:
        sys.exit(1)


def _handle_info_or_delete(
    paths: tuple[str, ...],
    info: bool,
    delete: bool,
    json_output: bool,
    recursive: bool,
):
    """Handle --info and --delete modes."""
    # Collect all files to process
    files_to_process = []

    for path in paths:
        if os.path.isfile(path):
            files_to_process.append(path)
        elif os.path.isdir(path):
            if recursive:
                for root, dirs, files in os.walk(path):
                    for file in files:
                        filepath = os.path.join(root, file)
                        files_to_process.append(filepath)
            else:
                for file in os.listdir(path):
                    filepath = os.path.join(path, file)
                    if os.path.isfile(filepath):
                        files_to_process.append(filepath)

    if not files_to_process:
        click.echo('No files to process.', err=True)
        sys.exit(1)

    results = []

    for filepath in files_to_process:
        result = {'path': filepath}

        if delete:
            success = delete_index(filepath)
            result['action'] = 'delete'
            result['success'] = success
            if not json_output:
                status = 'deleted' if success else 'not found'
                click.echo(f'{filepath}: index {status}')

        elif info:
            idx = load_index(filepath)
            result['action'] = 'info'

            if idx:
                result['index'] = {
                    'version': idx.version,
                    'file_type': idx.file_type.value,
                    'source_size_bytes': idx.source_size_bytes,
                    'created_at': idx.created_at,
                    'analysis_performed': idx.analysis_performed,
                    'line_count': idx.line_count,
                    'index_entries': len(idx.line_index),
                }

                if idx.anomalies:
                    result['index']['anomaly_count'] = len(idx.anomalies)
                    result['index']['anomaly_summary'] = idx.anomaly_summary

                if not json_output:
                    click.echo(f'\n{filepath}:')
                    click.echo(f'  File type: {idx.file_type.value}')
                    click.echo(f'  Source size: {human_readable_size(idx.source_size_bytes)}')
                    click.echo(f'  Created: {idx.created_at}')
                    click.echo(f'  Index entries: {len(idx.line_index)}')
                    click.echo(f'  Analysis performed: {idx.analysis_performed}')
                    if idx.line_count:
                        click.echo(f'  Lines: {idx.line_count:,}')
                    if idx.anomalies:
                        click.echo(f'  Anomalies: {len(idx.anomalies)}')
                        if idx.anomaly_summary:
                            for cat, count in idx.anomaly_summary.items():
                                click.echo(f'    {cat}: {count}')
            else:
                result['index'] = None
                if not json_output:
                    click.echo(f'\n{filepath}: no index exists')

        results.append(result)

    if json_output:
        click.echo(json.dumps({'files': results}, indent=2))


def _output_json(result):
    """Output indexing result as JSON with all collected data."""
    output = {
        'indexed': [_index_to_json(idx) for idx in result.indexed],
        'skipped': result.skipped,
        'errors': [{'path': p, 'error': e} for p, e in result.errors],
        'total_time': result.total_time,
    }
    click.echo(json.dumps(output, indent=2))


def _index_to_json(idx) -> dict:
    """Convert UnifiedFileIndex to JSON-serializable dict with all fields."""
    data = {
        'path': idx.source_path,
        'file_type': idx.file_type.value,
        'size_bytes': idx.source_size_bytes,
        'created_at': idx.created_at,
        'build_time_seconds': idx.build_time_seconds,
        'analysis_performed': idx.analysis_performed,
    }

    # Line index (offset mapping)
    if idx.line_index:
        data['line_index'] = idx.line_index
        data['index_entries'] = len(idx.line_index)
    else:
        data['line_index'] = []
        data['index_entries'] = 0

    # Line statistics (always include if available)
    if idx.line_count is not None:
        data['line_count'] = idx.line_count
    if idx.empty_line_count is not None:
        data['empty_line_count'] = idx.empty_line_count
    if idx.line_ending:
        data['line_ending'] = idx.line_ending

    # Line length statistics
    if idx.line_length_max is not None:
        data['line_length'] = {
            'max': idx.line_length_max,
            'avg': idx.line_length_avg,
            'median': idx.line_length_median,
            'p95': idx.line_length_p95,
            'p99': idx.line_length_p99,
            'stddev': idx.line_length_stddev,
        }
        if idx.line_length_max_line_number is not None:
            data['longest_line'] = {
                'line_number': idx.line_length_max_line_number,
                'byte_offset': idx.line_length_max_byte_offset,
            }

    # Compression info (if applicable)
    if idx.compression_format:
        data['compression_format'] = idx.compression_format
    if idx.decompressed_size_bytes is not None:
        data['decompressed_size_bytes'] = idx.decompressed_size_bytes
    if idx.compression_ratio is not None:
        data['compression_ratio'] = idx.compression_ratio

    # Anomaly info (if analysis was performed)
    if idx.analysis_performed:
        data['anomaly_count'] = len(idx.anomalies) if idx.anomalies else 0
        data['anomaly_summary'] = idx.anomaly_summary
        if idx.anomalies:
            data['anomalies'] = [
                {
                    'start_line': a.start_line,
                    'end_line': a.end_line,
                    'start_offset': a.start_offset,
                    'end_offset': a.end_offset,
                    'severity': a.severity,
                    'category': a.category,
                    'description': a.description,
                    'detector': a.detector,
                }
                for a in idx.anomalies
            ]

    return data


def _output_human_readable(result, analyze: bool):
    """Output indexing result in human-readable format."""
    if not result.indexed and not result.errors:
        click.echo('No files indexed.')
        return

    # Summary line
    if analyze:
        click.echo(f'Indexed and analyzed {len(result.indexed)} files in {result.total_time:.1f}s')
    else:
        click.echo(f'Indexed {len(result.indexed)} files in {result.total_time:.1f}s')

    # Details for each file
    for idx in result.indexed:
        line_info = f'{idx.line_count:,} lines' if idx.line_count else 'unknown lines'
        size_info = human_readable_size(idx.source_size_bytes)
        click.echo(f'  {idx.source_path}: {line_info}, {size_info}')

        # Show line statistics when analysis was performed
        if idx.analysis_performed:
            # Lines info with empty count
            if idx.line_count is not None and idx.empty_line_count is not None:
                click.echo(f'    Lines: {idx.line_count:,} total, {idx.empty_line_count:,} empty')

            # Line ending
            if idx.line_ending:
                click.echo(f'    Line ending: {idx.line_ending}')

            # Line length statistics
            if idx.line_length_max is not None:
                click.echo(
                    f'    Line length: max={idx.line_length_max}, '
                    f'avg={idx.line_length_avg:.1f}, '
                    f'median={idx.line_length_median:.1f}, '
                    f'p95={idx.line_length_p95:.1f}, '
                    f'p99={idx.line_length_p99:.1f}, '
                    f'stddev={idx.line_length_stddev:.1f}'
                )

                # Longest line location
                if idx.line_length_max_line_number is not None:
                    click.echo(
                        f'    Longest line: line {idx.line_length_max_line_number}, '
                        f'offset {idx.line_length_max_byte_offset}'
                    )

            # Anomalies
            if idx.anomalies:
                if idx.anomaly_summary:
                    summary_parts = [f'{count} {cat}' for cat, count in idx.anomaly_summary.items()]
                    click.echo(f'    Anomalies: {", ".join(summary_parts)}')
                else:
                    click.echo(f'    Anomalies: {len(idx.anomalies)}')
            else:
                click.echo('    Anomalies: none')

    # Show skipped files count
    if result.skipped:
        click.echo(f'Skipped {len(result.skipped)} files (below threshold or not text)')

    # Show errors
    for path, error in result.errors:
        click.echo(f'Error: {path}: {error}', err=True)
