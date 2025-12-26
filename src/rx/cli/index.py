"""CLI command for file indexing and analysis."""

import json
import os
import sys

import click

from rx.indexer import FileIndexer
from rx.unified_index import delete_index, load_index


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
    """
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
    """Output indexing result as JSON."""
    output = {
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
    click.echo(json.dumps(output, indent=2))


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

        if analyze and idx.anomalies:
            # Show anomaly summary
            if idx.anomaly_summary:
                summary_parts = [f'{count} {cat}' for cat, count in idx.anomaly_summary.items()]
                click.echo(f'    Anomalies: {", ".join(summary_parts)}')
            else:
                click.echo(f'    Anomalies: {len(idx.anomalies)}')
        elif analyze:
            click.echo('    Anomalies: none')

    # Show skipped files count
    if result.skipped:
        click.echo(f'Skipped {len(result.skipped)} files (below threshold or not text)')

    # Show errors
    for path, error in result.errors:
        click.echo(f'Error: {path}: {error}', err=True)
