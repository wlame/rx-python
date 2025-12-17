"""CLI trace command for RX"""

import json
import os
import re
import sys
import tempfile
from datetime import datetime
from time import time

import click

from rx.hooks import (
    call_hook_sync,
    generate_request_id,
    get_effective_hooks,
)
from rx.models import ContextLine, Match, ParseResult, RequestInfo, TraceCompletePayload, TraceResponse
from rx.request_store import increment_hook_counter, store_request, update_request
from rx.trace import HookCallbacks, parse_paths


def format_context_header(file_val: str, line_num: int, offset_str: str, pattern_val: str, colorize: bool) -> str:
    """Format the header line for a context block."""
    if colorize:
        return (
            click.style('=== ', fg='bright_black')
            + click.style(file_val, fg='cyan', bold=True)
            + click.style(':', fg='bright_black')
            + click.style(str(line_num), fg='yellow')
            + click.style(':', fg='bright_black')
            + click.style(offset_str, fg='white')
            + ' '
            + click.style('[', fg='bright_black')
            + click.style(pattern_val, fg='magenta', bold=True)
            + click.style(']', fg='bright_black')
            + ' '
            + click.style('===', fg='bright_black')
        )
    else:
        return f'=== {file_val}:{line_num}:{offset_str} [{pattern_val}] ==='


def find_match_for_context(
    response: TraceResponse, pattern_id: str, file_id: str, offset_int: int
) -> tuple[str | None, int | None]:
    """Find the matched line and line number for a given context block."""
    for match in response.matches:
        if match.pattern == pattern_id and match.file == file_id and match.offset == offset_int:
            # Use absolute_line_number if available, otherwise relative_line_number, or -1
            line_num = (
                match.absolute_line_number if match.absolute_line_number != -1 else (match.relative_line_number or -1)
            )
            return match.line_text, line_num
    return None, None


def build_lines_dict(ctx_lines: list, matched_line: str | None, match_line_number: int | None) -> dict[int, str]:
    """
    Build a dictionary of line numbers to line text from context lines.

    Args:
        ctx_lines: List of context line objects
        matched_line: The matched line text to include
        match_line_number: Line number of the matched line

    Returns:
        Dictionary mapping line_number -> line_text
    """
    lines_by_number = {}

    # Add context lines
    for ctx_line in ctx_lines:
        line_num = (
            ctx_line.relative_line_number if isinstance(ctx_line, ContextLine) else ctx_line.get('relative_line_number')
        )
        line_text = (
            ctx_line.line_text if isinstance(ctx_line, ContextLine) else ctx_line.get('line_text', str(ctx_line))
        )
        lines_by_number[line_num] = line_text

    # Add the matched line
    if matched_line and match_line_number:
        lines_by_number[match_line_number] = matched_line

    return lines_by_number


def highlight_pattern_in_line(line_text: str, pattern_val: str, colorize: bool) -> str:
    """
    Highlight pattern matches in a line of text.

    Args:
        line_text: The line to highlight
        pattern_val: The pattern to highlight
        colorize: Whether to apply color styling

    Returns:
        Highlighted line (or original if highlighting fails)
    """
    if not colorize:
        return line_text

    try:
        # Highlight the matched pattern in bold red
        parts = re.split(f'({pattern_val})', line_text)
        return ''.join(
            click.style(part, fg='bright_red', bold=True) if i % 2 == 1 else part for i, part in enumerate(parts)
        )
    except re.error:
        # If regex is invalid for highlighting, just return the original
        return line_text


def display_context_block(
    composite_key: str, response: TraceResponse, pattern_ids: dict[str, str], file_ids: dict[str, str], colorize: bool
) -> None:
    """
    Display a single context block (header + context lines).

    Args:
        composite_key: The composite key "pattern_id:file_id:offset"
        response: TraceResponse containing matches and context
        pattern_ids: Mapping of pattern IDs to patterns
        file_ids: Mapping of file IDs to file paths
        colorize: Whether to apply color styling
    """
    # Parse composite key
    parts = composite_key.split(':', 2)
    if len(parts) != 3:
        return

    pattern_id, file_id, offset_str = parts
    pattern_val = pattern_ids.get(pattern_id, pattern_id)
    file_val = file_ids.get(file_id, file_id)
    offset_int = int(offset_str)

    # Find the matched line
    matched_line, match_line_number = find_match_for_context(response, pattern_id, file_id, offset_int)

    # Format and display header
    header = format_context_header(file_val, match_line_number or -1, offset_str, pattern_val, colorize)
    click.echo(header)

    # Get context lines for this match
    ctx_lines = response.context_lines[composite_key]

    # Build lines dictionary (line_number -> line_text)
    lines_by_number = build_lines_dict(ctx_lines, matched_line, match_line_number)

    # Display lines in order
    for line_num in sorted(lines_by_number.keys()):
        line_text = lines_by_number[line_num]
        highlighted = highlight_pattern_in_line(line_text, pattern_val, colorize)
        click.echo(highlighted)

    click.echo()  # Blank line after each context block


def display_samples_output(
    response: TraceResponse,
    pattern_ids: dict[str, str],
    file_ids: dict[str, str],
    before_ctx: int,
    after_ctx: int,
    colorize: bool,
) -> None:
    """
    Display sample output with context lines in CLI format.

    Args:
        response: TraceResponse containing matches and context
        pattern_ids: Mapping of pattern IDs to patterns
        file_ids: Mapping of file IDs to file paths
        before_ctx: Number of context lines before
        after_ctx: Number of context lines after
        colorize: Whether to apply color styling
    """
    # Display basic match summary
    click.echo(response.to_cli(colorize=colorize))
    click.echo()
    click.echo(f'Samples (context: {before_ctx} before, {after_ctx} after):')
    click.echo()

    # Display context blocks
    if response.context_lines is not None:
        for composite_key in sorted(response.context_lines.keys()):
            display_context_block(composite_key, response, pattern_ids, file_ids, colorize)
    else:
        click.echo('No context available (context may not have been requested or no matches found)')


def handle_samples_output(
    response: TraceResponse,
    pattern_ids: dict[str, str],
    file_ids: dict[str, str],
    before_ctx: int,
    after_ctx: int,
    output_json: bool,
    no_color: bool,
) -> None:
    """
    Handle output when --samples flag is enabled.

    Args:
        response: TraceResponse containing matches and context
        pattern_ids: Mapping of pattern IDs to patterns
        file_ids: Mapping of file IDs to file paths
        before_ctx: Number of context lines before
        after_ctx: Number of context lines after
        output_json: Whether to output as JSON
        no_color: Whether to disable color
    """
    try:
        if output_json:
            # JSON output with samples
            output_data = response.model_dump()
            click.echo(json.dumps(output_data, indent=2))
        else:
            # CLI output with samples (human-readable)
            colorize = not no_color and sys.stdout.isatty()
            display_samples_output(response, pattern_ids, file_ids, before_ctx, after_ctx, colorize)
    except Exception as e:
        click.echo(f'❌ Error displaying samples: {e}', err=True)
        sys.exit(1)


@click.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click.argument('pattern_arg', type=str, required=False, metavar='PATTERN')
@click.argument('path_args', type=str, required=False, nargs=-1, metavar='[PATH ...]')
@click.option(
    '--path',
    '--file',
    type=click.Path(exists=True),
    multiple=True,
    help='File or directory path to search (can be specified multiple times)',
)
@click.option(
    '--regexp',
    '--regex',
    '-e',
    'regexp',
    type=str,
    multiple=True,
    help='Regex pattern to search (can be specified multiple times)',
)
@click.option('--max-results', type=int, help='Maximum number of results to return')
@click.option('--samples', is_flag=True, help='Show context lines around matches')
@click.option('--context', type=int, help='Number of lines before and after (for --samples)')
@click.option('--before', '-B', type=int, help='Number of lines before match (for --samples)')
@click.option('--after', '-A', type=int, help='Number of lines after match (for --samples)')
@click.option('--json', 'output_json', is_flag=True, help='Output results as JSON')
@click.option('--no-color', is_flag=True, help='Disable colored output')
@click.option('--debug', is_flag=True, help='Enable debug mode (creates .debug_* files)')
@click.option('--request-id', type=str, help='Custom request ID (auto-generated if not provided)')
@click.option('--hook-on-file', type=str, help='URL to call (GET) when file scan completes')
@click.option('--hook-on-match', type=str, help='URL to call (GET) for each match. Requires --max-results.')
@click.option('--hook-on-complete', type=str, help='URL to call (GET) when trace completes')
@click.option('--no-cache', is_flag=True, help="Disable trace cache (don't read from or write to cache)")
@click.option('--no-index', is_flag=True, help="Disable file indexing (don't use existing indexes)")
@click.pass_context
def trace_command(
    ctx,
    pattern_arg,
    path_args,
    path,
    regexp,
    max_results,
    samples,
    context,
    before,
    after,
    output_json,
    no_color,
    debug,
    request_id,
    hook_on_file,
    hook_on_match,
    hook_on_complete,
    no_cache,
    no_index,
):
    """
    Trace files and directories for regex patterns using ripgrep.

    \b
    Usage: rx [OPTIONS] PATTERN [PATH ...]

    If PATH is not specified, searches in the current directory.
    Use '-' as PATH or pipe input to search stdin.
    For multiple patterns, use -e/--regexp/--regex multiple times.
    For multiple paths, use --path/--file multiple times or list them after PATTERN.

    \b
    Basic Examples:
        rx "error.*"                                # Search in current directory
        rx "error.*" file.log                       # Search in specific file
        rx "error.*" file.log dir/                  # Search in multiple paths
        rx "error" --max-results=10                 # Limit results
        rx "error" --samples                        # Show context lines
        rx "error" --samples --context=5            # Custom context size

    \b
    Stdin Examples:
        cat file.log | rx "error"                   # Search piped input
        echo "test error" | rx "error"              # Search from echo
        rx "error" -                                # Explicit stdin (- means stdin)
        docker logs container | rx "error"          # Search container logs

    \b
    Multiple Patterns/Paths:
        rx -e "error" -e "warning" file.log         # Multiple patterns
        rx "error" --path=file1.log --path=file2.log # Multiple paths via options
        rx "error" file1.log file2.log              # Multiple paths as arguments

    \b
    Ripgrep Passthrough:
        Any unrecognized options are passed directly to ripgrep:
        rx "error" -i                               # Case-insensitive search
        rx "error" --case-sensitive                 # Case-sensitive search
        rx "error" -w                               # Match whole words only
        rx "error" -A 3                             # Show 3 lines after match

    \b
    Requirements:
        - ripgrep must be installed on your system
          macOS: brew install ripgrep
          Ubuntu/Debian: apt install ripgrep
          Fedora: dnf install ripgrep
    """

    # Resolve patterns from positional or named params
    final_regexps = []
    if regexp:
        # Named parameter --regexp/--regex/-e (tuple from multiple=True)
        final_regexps.extend(list(regexp))

    # Resolve paths from positional or named params
    # Separate actual paths from ripgrep flags that may have been captured as positional args
    final_paths = []
    rg_flags_from_positional = []

    if path:
        # Named parameter --path/--file (tuple from multiple=True)
        final_paths.extend(list(path))

    # Handle pattern_arg: it's a pattern ONLY if no -e flags were used
    # If -e flags are present, pattern_arg is actually the first path
    if pattern_arg:
        if regexp:
            # -e flags were used, so pattern_arg is actually a path
            final_paths.append(pattern_arg)
        else:
            # No -e flags, so pattern_arg is the pattern
            final_regexps.append(pattern_arg)

    if path_args:
        # Positional PATH arguments - filter out flags (starting with -)
        # Also handle flag arguments (e.g., -C 1)
        skip_next = False
        for i, arg in enumerate(path_args):
            if skip_next:
                skip_next = False
                continue

            if arg.startswith('-'):
                # This is a ripgrep flag, not a path
                rg_flags_from_positional.append(arg)
                # Check if next arg is the flag's value (not starting with -)
                if i + 1 < len(path_args) and not path_args[i + 1].startswith('-'):
                    rg_flags_from_positional.append(path_args[i + 1])
                    skip_next = True
            else:
                # This is a path
                final_paths.append(arg)

    # Handle stdin input
    # Check if we should read from stdin:
    # 1. Explicit '-' as path means stdin
    # 2. No paths provided AND stdin is not a TTY (piped input)
    stdin_temp_file = None
    use_stdin = False

    if '-' in final_paths:
        use_stdin = True
        final_paths.remove('-')
    elif not final_paths and not sys.stdin.isatty():
        use_stdin = True

    if use_stdin:
        # Read stdin content and write to temporary file
        try:
            stdin_content = sys.stdin.read()
            if stdin_content:
                # Create a temporary file with the stdin content
                fd, stdin_temp_file = tempfile.mkstemp(prefix='rx_stdin_', suffix='.txt', text=True)
                try:
                    os.write(fd, stdin_content.encode('utf-8'))
                finally:
                    os.close(fd)

                # Add the temp file to paths to search
                final_paths.append(stdin_temp_file)
            elif not final_paths:
                # No stdin content and no other paths - default to current directory
                final_paths = ['.']
        except Exception as e:
            click.echo(f'❌ Error reading stdin: {e}', err=True)
            sys.exit(1)

    # Default to current directory if no paths specified (and no stdin)
    if not final_paths:
        final_paths = ['.']

    # Extract extra ripgrep arguments from unknown options and from positional args
    rg_extra_args = list(ctx.args) if ctx.args else []
    rg_extra_args.extend(rg_flags_from_positional)

    # Apply environment variable defaults for no_cache and no_index
    if not no_cache and os.environ.get('RX_NO_CACHE', '').lower() in ('1', 'true', 'yes'):
        no_cache = True
    if not no_index and os.environ.get('RX_NO_INDEX', '').lower() in ('1', 'true', 'yes'):
        no_index = True

    # Enable debug mode if --debug flag is set
    if debug:
        os.environ['RX_DEBUG'] = '1'
        # Reload the parse_json module to pick up the new DEBUG_MODE setting
        import importlib

        from rx import parse_json

        importlib.reload(parse_json)
        click.echo('Debug mode enabled - will create .debug_* files', err=True)

    if final_paths and final_regexps:
        try:
            # Get effective hook configuration (respects RX_DISABLE_CUSTOM_HOOKS)
            hooks_config = get_effective_hooks(hook_on_file, hook_on_match, hook_on_complete)

            # Validate: max_results is required when hook_on_match is configured
            if hooks_config.has_match_hook() and max_results is None:
                click.echo(
                    '❌ Error: --max-results is required when --hook-on-match is configured.\n'
                    'This prevents accidentally triggering millions of HTTP calls.',
                    err=True,
                )
                sys.exit(1)

            # Generate or use provided request_id
            req_id = request_id or generate_request_id()

            # Store request info
            request_info = RequestInfo(
                request_id=req_id,
                paths=final_paths,
                patterns=final_regexps,
                max_results=max_results,
                started_at=datetime.now(),
            )
            store_request(request_info)

            # Calculate context parameters if --samples is requested
            if samples:
                before_ctx = before if before is not None else context if context is not None else 3
                after_ctx = after if after is not None else context if context is not None else 3
            else:
                before_ctx = 0
                after_ctx = 0

            # Create hook callbacks for synchronous calls during parsing
            def on_match_callback(payload: dict) -> None:
                """Synchronous callback for match events."""
                if hooks_config.on_match_url:
                    success = call_hook_sync(hooks_config.on_match_url, payload, 'on_match')
                    increment_hook_counter(req_id, 'on_match', success)

            def on_file_callback(payload: dict) -> None:
                """Synchronous callback for file scan events."""
                if hooks_config.on_file_url:
                    success = call_hook_sync(hooks_config.on_file_url, payload, 'on_file')
                    increment_hook_counter(req_id, 'on_file', success)

            # Build HookCallbacks if any hooks are configured
            hook_callbacks = None
            if hooks_config.has_any_hook():
                hook_callbacks = HookCallbacks(
                    on_match_found=on_match_callback if hooks_config.on_match_url else None,
                    on_file_scanned=on_file_callback if hooks_config.on_file_url else None,
                    request_id=req_id,
                )

            # Parse files or directories for matches
            try:
                time_before = time()
                parse_result: ParseResult = parse_paths(
                    final_paths,
                    final_regexps,
                    max_results=max_results,
                    rg_extra_args=rg_extra_args,
                    context_before=before_ctx,
                    context_after=after_ctx,
                    hooks=hook_callbacks,
                    use_cache=not no_cache,
                    use_index=not no_index,
                )
                parsing_time = time() - time_before

                # Update request info with results
                update_request(
                    request_id=req_id,
                    completed_at=datetime.now(),
                    total_matches=len(parse_result.matches),
                    total_files_scanned=len(parse_result.files),
                    total_files_skipped=len(parse_result.skipped_files),
                    total_time_ms=int(parsing_time * 1000),
                )

                # Call on_complete hook if configured
                if hooks_config.on_complete_url:
                    complete_payload: dict = TraceCompletePayload(
                        request_id=req_id,
                        paths=','.join(final_paths),
                        patterns=','.join(final_regexps),
                        total_files_scanned=len(parse_result.files),
                        total_files_skipped=len(parse_result.skipped_files),
                        total_matches=len(parse_result.matches),
                        total_time_ms=int(parsing_time * 1000),
                    ).model_dump()
                    success = call_hook_sync(hooks_config.on_complete_url, complete_payload, 'on_complete')
                    increment_hook_counter(req_id, 'on_complete', success)

            except FileNotFoundError as e:
                click.echo(f'❌ Error: {e}', err=True)
                sys.exit(1)
            except RuntimeError as e:
                click.echo(f'❌ Error: {e}', err=True)
                sys.exit(1)
            except Exception as e:
                click.echo(f'❌ Unexpected error: {e}', err=True)
                sys.exit(1)

            # Build response object
            # Convert context_lines to proper format if present
            converted_context = None
            if parse_result.context_lines:
                converted_context = {}
                for key, ctx_lines in parse_result.context_lines.items():
                    # Convert ContextLine objects to dict for serialization
                    converted_context[key] = ctx_lines

            response = TraceResponse(
                request_id=req_id,
                path=final_paths,  # Pass as list directly
                time=parsing_time,
                patterns=parse_result.patterns,
                files=parse_result.files,
                matches=[Match(**m) for m in parse_result.matches],
                scanned_files=parse_result.scanned_files,
                skipped_files=parse_result.skipped_files,
                file_chunks=parse_result.file_chunks,
                context_lines=converted_context,
                before_context=before_ctx if samples else None,
                after_context=after_ctx if samples else None,
                max_results=max_results,  # Include max_results in response
            )

            # Handle --samples flag
            if samples:
                handle_samples_output(
                    response=response,
                    pattern_ids=parse_result.patterns,
                    file_ids=parse_result.files,
                    before_ctx=before_ctx,
                    after_ctx=after_ctx,
                    output_json=output_json,
                    no_color=no_color,
                )
            else:
                # No samples - just show matches
                if output_json:
                    click.echo(response.model_dump_json(indent=2))
                else:
                    colorize = not no_color
                    click.echo(response.to_cli(colorize=colorize))

            sys.exit(0)
        finally:
            # Clean up temporary stdin file if it was created
            if stdin_temp_file and os.path.exists(stdin_temp_file):
                try:
                    os.unlink(stdin_temp_file)
                except Exception:
                    pass  # Ignore cleanup errors

    # No valid mode provided - show help
    ctx = click.get_current_context()
    click.echo(ctx.get_help())
    sys.exit(0)


if __name__ == '__main__':
    search_command()
