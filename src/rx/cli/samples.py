"""CLI command for getting file samples around byte offsets or line numbers."""

import json
import sys

import click

from rx.compressed_index import get_decompressed_content_at_line
from rx.compression import CompressionFormat, detect_compression, is_compressed
from rx.file_utils import get_context, get_context_by_lines, is_text_file
from rx.models import SamplesResponse, UnifiedFileIndex
from rx.unified_index import (
    LineInfo,
    calculate_exact_line_for_offset,
    calculate_exact_offset_for_line,
    calculate_line_info_for_offsets,
    get_large_file_threshold_bytes,
    load_index,
)


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


def get_lines_for_byte_range(path: str, start_offset: int, end_offset: int, index_data: dict | None) -> list[str]:
    """Get all lines that overlap with a byte offset range.

    Efficiently reads only the required portion of the file using seek.
    Returns lines from the first line containing start_offset to the last line containing end_offset.
    """

    # Get line info for both offsets in a single pass
    line_infos: dict[int, LineInfo] = calculate_line_info_for_offsets(path, [start_offset, end_offset], index_data)

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


def get_line_range(path: str, start_line: int, end_line: int, file_index: UnifiedFileIndex | None = None) -> list[str]:
    """Get lines from start_line to end_line (inclusive, 1-based).

    Uses the index to seek directly to the start line's byte offset for efficiency.
    Falls back to linear scan if no index is available.

    Args:
        path: File path
        start_line: Starting line number (1-based)
        end_line: Ending line number (1-based, inclusive)
        file_index: Optional pre-loaded UnifiedFileIndex for efficient seeking

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

    Strategy:
    1. Try to get from unified index cache (fastest, most accurate)
    2. Use index + counting remaining lines if line_count not available
    3. For large files without index: build index first
    4. For small files: run indexer with analysis
    5. Fallback: count all lines directly

    Args:
        path: File path

    Returns:
        Total number of lines in the file
    """
    import os

    from rx.indexer import FileIndexer

    # Strategy 1: Try unified index cache first
    unified_idx = load_index(path)
    if unified_idx and unified_idx.line_count:
        return unified_idx.line_count

    # Strategy 2: Use index + count remaining if line_count not available
    if unified_idx and unified_idx.line_index:
        line_index = unified_idx.line_index
        # Get the last indexed line number and offset
        last_line, last_offset = line_index[-1]

        # Count remaining lines after the last indexed position
        remaining_lines = 0
        with open(path, 'rb') as f:
            f.seek(last_offset)
            # Read from last indexed position to end
            for _ in f:
                remaining_lines += 1

        # Total = last indexed line + remaining lines - 1
        # (subtract 1 because last_line is already counted)
        return last_line + remaining_lines - 1

    # Strategy 3: Build index for file
    file_size = os.path.getsize(path)
    large_file_threshold = get_large_file_threshold_bytes()

    if file_size >= large_file_threshold:
        # Large file: build index without analysis
        click.echo('Building index for large file...', err=True)
        indexer = FileIndexer(analyze=False, force=False)
        file_index = indexer.index_file(path)
        if file_index and file_index.line_index:
            line_index = file_index.line_index
            last_line, last_offset = line_index[-1]
            remaining_lines = 0
            with open(path, 'rb') as f:
                f.seek(last_offset)
                for _ in f:
                    remaining_lines += 1
            return last_line + remaining_lines - 1
    else:
        # Small file: run indexer with analysis to get line_count
        click.echo('Indexing file...', err=True)
        indexer = FileIndexer(analyze=True)
        result = indexer.index_file(path)
        if result and result.line_count:
            return result.line_count

    # Fallback: count lines directly
    click.echo('Counting lines...', err=True)
    with open(path, 'rb') as f:
        return sum(1 for _ in f)


@click.command('samples')
@click.argument('path', type=click.Path(exists=True))
@click.option(
    '--byte-offset',
    '-b',
    multiple=True,
    type=str,
    help='Byte offset(s) or range. Single offset (1234) gets context. Negative offset (-100) counts from end. Range (1000-2000) gets exact bytes. Can be specified multiple times.',
)
@click.option(
    '--line-offset',
    '-l',
    multiple=True,
    type=str,
    help='Line number(s) or range (1-based). Single line (100) gets context. Negative line (-5) counts from end. Range (100-200) gets exact lines. Can be specified multiple times.',
)
@click.option(
    '--context',
    '-c',
    type=int,
    default=None,
    help='Number of context lines before and after (default: 3)',
)
@click.option(
    '--before',
    '-B',
    type=int,
    default=None,
    help='Number of context lines before offset',
)
@click.option(
    '--after',
    '-A',
    type=int,
    default=None,
    help='Number of context lines after offset',
)
@click.option(
    '--json',
    'json_output',
    is_flag=True,
    help='Output in JSON format',
)
@click.option(
    '--no-color',
    is_flag=True,
    help='Disable colored output',
)
@click.option(
    '--regex',
    '-r',
    type=str,
    default=None,
    help='Regex pattern to highlight in output',
)
def samples_command(
    path: str,
    byte_offset: tuple[str, ...],
    line_offset: tuple[str, ...],
    context: int | None,
    before: int | None,
    after: int | None,
    json_output: bool,
    no_color: bool,
    regex: str | None,
):
    """Get file content around specified byte offsets or line numbers.

    This command reads lines of context around one or more byte offsets
    or line numbers in a text file. Useful for examining specific locations
    in large files.

    Use -b/--byte-offset for byte offsets, or -l/--line-offset for line numbers.
    These options are mutually exclusive.

    You can specify either single values (which use context) or ranges (which get exact lines):
    - Single value: -l 100 gets line 100 with context (default 3 lines before/after)
    - Range: -l 100-200 gets exactly lines 100-200, ignoring context settings
    - Mix both: -l 50 -l 100-200 -c 5 gets line 50 with 5 lines context, plus lines 100-200

    Examples:

        rx samples /var/log/app.log -b 1234

        rx samples /var/log/app.log -b 1234 -b 5678 -c 5

        rx samples /var/log/app.log -l 100 -l 200

        rx samples /var/log/app.log -l 100 --before=2 --after=10

        rx samples /var/log/app.log -b 1234 --json

        rx samples /var/log/app.log -l 100-200

        rx samples /var/log/app.log -l 50 -l 100-200 --context=5

        rx samples /var/log/app.log -b 1000-5000
    """
    # Validate mutual exclusivity
    if byte_offset and line_offset:
        click.echo('Error: Cannot use both --byte-offset and --line-offset. Choose one.', err=True)
        sys.exit(1)

    if not byte_offset and not line_offset:
        click.echo('Error: Must provide either --byte-offset (-b) or --line-offset (-l)', err=True)
        sys.exit(1)

    # Check if file is compressed
    file_is_compressed = is_compressed(path)
    compression_format = detect_compression(path) if file_is_compressed else CompressionFormat.NONE

    # Compressed files only support line offset mode
    if file_is_compressed and byte_offset:
        click.echo(
            'Error: Byte offsets are not supported for compressed files. Use --line-offset (-l) instead.',
            err=True,
        )
        sys.exit(1)

    # Validate file is text (skip for compressed files as we can't easily check)
    if not file_is_compressed and not is_text_file(path):
        click.echo(f'Error: {path} is not a text file', err=True)
        sys.exit(1)

    # Determine context lines
    before_context = before if before is not None else context if context is not None else 3
    after_context = after if after is not None else context if context is not None else 3

    if before_context < 0 or after_context < 0:
        click.echo('Error: Context values must be non-negative', err=True)
        sys.exit(1)

    # Parse offsets/lines to separate single values from ranges
    try:
        if byte_offset:
            parsed_offsets = [parse_offset_or_range(val) for val in byte_offset]
        else:
            parsed_offsets = [parse_offset_or_range(val) for val in line_offset]
    except ValueError as e:
        click.echo(f'Error: {e}', err=True)
        sys.exit(1)

    try:
        # Handle compressed files separately
        if file_is_compressed:
            # Check if this is a seekable zstd file
            from rx.seekable_zstd import is_seekable_zstd

            if is_seekable_zstd(path):
                # Use seekable zstd index for accurate line numbers
                click.echo('Processing seekable zstd file...', err=True)
                from rx.seekable_index import get_or_build_index
                from rx.seekable_zstd import decompress_frame, read_seek_table

                index = get_or_build_index(path)
                frames = read_seek_table(path)

                # Get samples for each line/range using seekable zstd
                context_data = {}
                line_to_offset = {}

                for start, end in parsed_offsets:
                    if end is None:
                        # Single line - use context
                        line_num = start

                        # Find which frame contains this line
                        frame_idx = None
                        for frame in index.frames:
                            if frame.first_line <= line_num <= frame.last_line:
                                frame_idx = frame.index
                                first_line = frame.first_line
                                break

                        if frame_idx is None:
                            context_data[line_num] = []
                            line_to_offset[str(line_num)] = -1
                            continue

                        # Calculate byte offset for this line
                        frame_offset = frames[frame_idx].decompressed_offset

                        # Decompress the frame to calculate exact offset
                        frame_data = decompress_frame(path, frame_idx, frames)
                        frame_lines = frame_data.decode('utf-8', errors='replace').split('\n')

                        # Calculate line index within frame (0-based)
                        line_in_frame = line_num - first_line

                        # Calculate byte offset by summing lengths of lines before target
                        byte_offset = frame_offset
                        for i in range(line_in_frame):
                            byte_offset += len(frame_lines[i].encode('utf-8')) + 1  # +1 for newline

                        line_to_offset[str(line_num)] = byte_offset

                        # Get context lines
                        start_idx = max(0, line_in_frame - before_context)
                        end_idx = min(len(frame_lines), line_in_frame + after_context + 1)

                        context_data[line_num] = frame_lines[start_idx:end_idx]
                    else:
                        # Range - get exact lines, ignore context
                        range_key = f'{start}-{end}'
                        range_lines = []

                        # Collect all lines in the range, potentially across multiple frames
                        for line_num in range(start, end + 1):
                            # Find which frame contains this line
                            frame_idx = None
                            for frame in index.frames:
                                if frame.first_line <= line_num <= frame.last_line:
                                    frame_idx = frame.index
                                    first_line = frame.first_line
                                    break

                            if frame_idx is None:
                                continue

                            # Decompress the frame
                            frame_data = decompress_frame(path, frame_idx, frames)
                            frame_lines = frame_data.decode('utf-8', errors='replace').split('\n')

                            # Calculate line index within frame (0-based)
                            line_in_frame = line_num - first_line
                            if 0 <= line_in_frame < len(frame_lines):
                                range_lines.append(frame_lines[line_in_frame])

                        context_data[range_key] = range_lines
                        line_to_offset[range_key] = -1
            else:
                # Use generic compressed index for other formats
                click.echo(f'Processing compressed file ({compression_format.value})...', err=True)
                # Load from unified cache or build
                from rx.indexer import FileIndexer

                index_data = load_index(path)
                if index_data is None:
                    indexer = FileIndexer(analyze=False)
                    index_data = indexer.index_file(path)

                # Get samples for each line/range
                context_data = {}
                line_to_offset = {}
                for start, end in parsed_offsets:
                    if end is None:
                        # Single line - use context
                        lines = get_decompressed_content_at_line(
                            path,
                            start,
                            context_before=before_context,
                            context_after=after_context,
                            index=index_data,
                        )
                        context_data[start] = lines
                        line_to_offset[str(start)] = -1
                    else:
                        # Range - get exact lines, ignore context
                        range_key = f'{start}-{end}'
                        range_lines = []
                        for line_num in range(start, end + 1):
                            lines = get_decompressed_content_at_line(
                                path,
                                line_num,
                                context_before=0,
                                context_after=0,
                                index=index_data,
                            )
                            if lines:
                                range_lines.extend(lines)
                        context_data[range_key] = range_lines
                        line_to_offset[range_key] = -1

            # Use calculated offsets for seekable zstd, -1 for other formats
            if is_seekable_zstd(path):
                lines_dict = line_to_offset
            else:
                lines_dict = line_to_offset

            response = SamplesResponse(
                path=path,
                offsets={},
                lines=lines_dict,
                before_context=before_context,
                after_context=after_context,
                samples={str(k): v for k, v in context_data.items()},
                is_compressed=True,
                compression_format=compression_format.value,
            )

            if json_output:
                click.echo(json.dumps(response.model_dump(), indent=2))
            else:
                colorize = not no_color and sys.stdout.isatty()
                click.echo(response.to_cli(colorize=colorize, regex=regex))
            return

        if byte_offset:
            # Byte offset mode - handle both single offsets and ranges
            import os

            index_data = load_index(path)
            context_data = {}
            offset_to_line = {}

            # Get file size for negative offset conversion
            file_size = os.path.getsize(path)

            for start, end in parsed_offsets:
                # Convert negative offsets to positive
                if end is None and start < 0:
                    # Negative single offset - convert using file size
                    # -1 means last byte (file_size - 1), -2 means file_size - 2, etc.
                    start = file_size + start
                    if start < 0:
                        start = 0  # Clamp to start of file
                if end is None:
                    # Single offset - use context
                    offset_context = get_context(path, [start], before_context, after_context)
                    context_data.update(offset_context)
                    line_num = calculate_exact_line_for_offset(path, start, index_data)
                    offset_to_line[str(start)] = line_num
                else:
                    # Range - get exact lines covering the byte range, ignore context
                    range_key = f'{start}-{end}'
                    lines = get_lines_for_byte_range(path, start, end, index_data)
                    context_data[range_key] = lines
                    # For ranges, we store the start line number
                    start_line = calculate_exact_line_for_offset(path, start, index_data)
                    offset_to_line[range_key] = start_line

            response = SamplesResponse(
                path=path,
                offsets=offset_to_line,
                lines={},
                before_context=before_context,
                after_context=after_context,
                samples={str(k): v for k, v in context_data.items()},
            )
        else:
            # Line offset mode - handle both single lines and ranges
            index_data = load_index(path)
            context_data = {}
            line_to_offset = {}

            # Get total line count for negative offset conversion (only if needed)
            total_lines = None
            needs_total_lines = any(start < 0 for start, end in parsed_offsets if end is None)
            if needs_total_lines:
                total_lines = get_total_line_count(path)

            for start, end in parsed_offsets:
                # Convert negative line numbers to positive
                if end is None and start < 0:
                    # Negative single line - convert using total line count
                    # -1 means last line, -2 means second to last, etc.
                    start = total_lines + start + 1
                    if start < 1:
                        start = 1  # Clamp to first line
                if end is None:
                    # Single line - use context
                    line_context = get_context_by_lines(path, [start], before_context, after_context)
                    context_data.update(line_context)
                    byte_offset_val = calculate_exact_offset_for_line(path, start, index_data)
                    line_to_offset[str(start)] = byte_offset_val
                else:
                    # Range - get exact lines, ignore context
                    range_key = f'{start}-{end}'
                    lines = get_line_range(path, start, end, index_data)
                    context_data[range_key] = lines
                    # For ranges, byte offset is not meaningful - use -1 to skip expensive calculation
                    line_to_offset[range_key] = -1

            response = SamplesResponse(
                path=path,
                offsets={},
                lines=line_to_offset,
                before_context=before_context,
                after_context=after_context,
                samples={str(k): v for k, v in context_data.items()},
            )

        if json_output:
            click.echo(json.dumps(response.model_dump(), indent=2))
        else:
            colorize = not no_color and sys.stdout.isatty()
            click.echo(response.to_cli(colorize=colorize, regex=regex))

    except ValueError as e:
        click.echo(f'Error: {e}', err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f'Error: {e}', err=True)
        sys.exit(1)
