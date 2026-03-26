"""Seekable zstd file creation and reading utilities.

This module provides functionality to create and read seekable zstd files,
which enable parallel decompression and efficient random access.

Seekable zstd files consist of:
1. Multiple independently-decompressable frames
2. A seek table (in zstd skippable frame format) at the end

The seek table allows finding the compressed offset for any decompressed position,
enabling parallel processing and random access without full decompression.
"""

import structlog
import os
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from rx.compression import (
    CompressionFormat,
    detect_compression,
    get_decompressor_command,
)

logger = structlog.get_logger()

# Default frame size: MIN_CHUNK_SIZE // 5 = 4MB
# This gives roughly 20-60MB of decompressed text per frame (5-15x compression ratio)
DEFAULT_FRAME_SIZE_BYTES = 4 * 1024 * 1024  # 4 MB

# Default compression level (zstd levels 1-22, 3 is a good balance)
DEFAULT_COMPRESSION_LEVEL = 3

# Seekable zstd seek table magic number (in skippable frame)
# Skippable frame format: 0x184D2A5? where ? is 0-F
SEEKABLE_MAGIC = 0x184D2A5E  # Skippable frame with nibble 0xE

# Seek table footer magic (identifies seekable format)
SEEK_TABLE_FOOTER_MAGIC = 0x8F92EAB1


@dataclass
class FrameInfo:
    """Information about a single zstd frame."""

    index: int
    compressed_offset: int
    compressed_size: int
    decompressed_offset: int
    decompressed_size: int

    @property
    def compressed_end(self) -> int:
        """End offset of compressed data."""
        return self.compressed_offset + self.compressed_size

    @property
    def decompressed_end(self) -> int:
        """End offset of decompressed data."""
        return self.decompressed_offset + self.decompressed_size


@dataclass
class SeekableZstdInfo:
    """Information about a seekable zstd file."""

    path: str
    compressed_size: int
    decompressed_size: int
    frame_count: int
    frames: list[FrameInfo] = field(default_factory=list)
    compression_level: int = DEFAULT_COMPRESSION_LEVEL
    frame_size_target: int = DEFAULT_FRAME_SIZE_BYTES

    @property
    def is_seekable(self) -> bool:
        """Check if this is a valid seekable zstd file."""
        return self.frame_count > 0 and len(self.frames) == self.frame_count


def check_t2sz_available() -> bool:
    """Check if t2sz tool is available for creating seekable zstd."""
    return shutil.which("t2sz") is not None


def check_zstd_available() -> bool:
    """Check if zstd tool is available."""
    return shutil.which("zstd") is not None


def is_seekable_zstd(filepath: str | Path) -> bool:
    """Check if a file is a seekable zstd file with seek table.

    Args:
        filepath: Path to the file

    Returns:
        True if file is seekable zstd, False otherwise
    """
    filepath = Path(filepath)

    if not filepath.exists():
        return False

    # Check for .zst extension
    if filepath.suffix.lower() != ".zst":
        return False

    try:
        # Check for seek table at end of file
        with open(filepath, "rb") as f:
            # Seek to last 9 bytes (4 byte magic + 4 byte num_frames + 1 byte flags)
            f.seek(-9, 2)
            footer = f.read(9)

            if len(footer) < 9:
                return False

            # Check footer magic
            magic = struct.unpack("<I", footer[0:4])[0]
            return magic == SEEK_TABLE_FOOTER_MAGIC

    except (OSError, IOError, struct.error):
        return False


def read_seek_table(zst_path: str | Path) -> list[FrameInfo]:
    """Read the seek table from a seekable zstd file.

    The seek table is stored at the end of the file in a skippable frame.
    Format (per entry, 8 bytes each):
        - compressed_size: u32
        - decompressed_size: u32

    Footer (9 bytes):
        - magic: u32 (0x8F92EAB1)
        - num_frames: u32
        - flags: u8

    Args:
        zst_path: Path to seekable zstd file

    Returns:
        List of FrameInfo objects

    Raises:
        ValueError: If file is not a valid seekable zstd
    """
    zst_path = Path(zst_path)

    with open(zst_path, "rb") as f:
        # Read footer
        f.seek(-9, 2)
        footer = f.read(9)

        magic = struct.unpack("<I", footer[0:4])[0]
        if magic != SEEK_TABLE_FOOTER_MAGIC:
            raise ValueError(f"Not a seekable zstd file: {zst_path}")

        num_frames = struct.unpack("<I", footer[4:8])[0]
        flags = footer[8]

        # Check if checksums are present (flag bit 0)
        has_checksums = bool(flags & 0x01)
        entry_size = 12 if has_checksums else 8

        # Read seek table entries
        seek_table_size = num_frames * entry_size
        f.seek(-9 - seek_table_size, 2)
        seek_table_data = f.read(seek_table_size)

        # Parse entries
        frames = []
        compressed_offset = 0
        decompressed_offset = 0

        for i in range(num_frames):
            offset = i * entry_size
            compressed_size = struct.unpack("<I", seek_table_data[offset : offset + 4])[0]
            decompressed_size = struct.unpack("<I", seek_table_data[offset + 4 : offset + 8])[0]

            frames.append(
                FrameInfo(
                    index=i,
                    compressed_offset=compressed_offset,
                    compressed_size=compressed_size,
                    decompressed_offset=decompressed_offset,
                    decompressed_size=decompressed_size,
                )
            )

            compressed_offset += compressed_size
            decompressed_offset += decompressed_size

        return frames


def get_seekable_zstd_info(zst_path: str | Path) -> SeekableZstdInfo:
    """Get information about a seekable zstd file.

    Args:
        zst_path: Path to seekable zstd file

    Returns:
        SeekableZstdInfo with file details

    Raises:
        ValueError: If file is not a valid seekable zstd
    """
    zst_path = Path(zst_path)

    if not is_seekable_zstd(zst_path):
        raise ValueError(f"Not a seekable zstd file: {zst_path}")

    frames = read_seek_table(zst_path)
    file_size = zst_path.stat().st_size

    decompressed_size = sum(f.decompressed_size for f in frames)

    # Estimate frame size target from first frame (or average)
    frame_size_target = frames[0].decompressed_size if frames else DEFAULT_FRAME_SIZE_BYTES

    return SeekableZstdInfo(
        path=str(zst_path),
        compressed_size=file_size,
        decompressed_size=decompressed_size,
        frame_count=len(frames),
        frames=frames,
        frame_size_target=frame_size_target,
    )


def decompress_frame(
    zst_path: str | Path,
    frame_index: int,
    frames: Optional[list[FrameInfo]] = None,
) -> bytes:
    """Decompress a single frame from a seekable zstd file.

    Args:
        zst_path: Path to seekable zstd file
        frame_index: Index of frame to decompress (0-based)
        frames: Optional pre-loaded frame info list

    Returns:
        Decompressed frame content as bytes

    Raises:
        ValueError: If frame_index is out of range
        RuntimeError: If decompression fails
    """
    zst_path = Path(zst_path)

    if frames is None:
        frames = read_seek_table(zst_path)

    if frame_index < 0 or frame_index >= len(frames):
        raise ValueError(f"Frame index {frame_index} out of range (0-{len(frames) - 1})")

    frame = frames[frame_index]

    # Read compressed frame data
    with open(zst_path, "rb") as f:
        f.seek(frame.compressed_offset)
        compressed_data = f.read(frame.compressed_size)

    # Decompress using zstd
    try:
        import zstandard as zstd

        dctx = zstd.ZstdDecompressor()
        return dctx.decompress(compressed_data)
    except ImportError:
        # Fallback to subprocess
        proc = subprocess.run(
            ["zstd", "-d", "-c"],
            input=compressed_data,
            capture_output=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"zstd decompression failed: {proc.stderr.decode()}")
        return proc.stdout


def decompress_frames(
    zst_path: str | Path,
    frame_indices: list[int],
    frames: Optional[list[FrameInfo]] = None,
) -> dict[int, bytes]:
    """Decompress multiple frames from a seekable zstd file.

    Args:
        zst_path: Path to seekable zstd file
        frame_indices: List of frame indices to decompress
        frames: Optional pre-loaded frame info list

    Returns:
        Dictionary mapping frame_index to decompressed bytes
    """
    result = {}
    for frame_index in frame_indices:
        result[frame_index] = decompress_frame(zst_path, frame_index, frames)
    return result


def decompress_range(
    zst_path: str | Path,
    start_offset: int,
    length: int,
    frames: Optional[list[FrameInfo]] = None,
) -> bytes:
    """Decompress a range of decompressed bytes from seekable zstd.

    This finds which frames contain the requested range, decompresses them,
    and returns only the requested bytes.

    Args:
        zst_path: Path to seekable zstd file
        start_offset: Start offset in decompressed stream
        length: Number of bytes to return
        frames: Optional pre-loaded frame info list

    Returns:
        Decompressed bytes for the requested range
    """
    zst_path = Path(zst_path)

    if frames is None:
        frames = read_seek_table(zst_path)

    end_offset = start_offset + length

    # Find frames that overlap with requested range
    needed_frames = []
    for frame in frames:
        if frame.decompressed_offset < end_offset and frame.decompressed_end > start_offset:
            needed_frames.append(frame.index)

    if not needed_frames:
        return b""

    # Decompress needed frames and extract requested range
    result = b""
    for frame_idx in needed_frames:
        frame = frames[frame_idx]
        frame_data = decompress_frame(zst_path, frame_idx, frames)

        # Calculate slice within this frame
        frame_start = max(0, start_offset - frame.decompressed_offset)
        frame_end = min(len(frame_data), end_offset - frame.decompressed_offset)

        result += frame_data[frame_start:frame_end]

    return result[:length]


def create_seekable_zstd(
    input_path: str | Path,
    output_path: str | Path,
    frame_size_bytes: int = DEFAULT_FRAME_SIZE_BYTES,
    compression_level: int = DEFAULT_COMPRESSION_LEVEL,
    threads: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> SeekableZstdInfo:
    """Create a seekable zstd file from input.

    If the input is already compressed (gzip, xz, etc.), it will be decompressed first.

    Args:
        input_path: Path to input file (can be compressed)
        output_path: Path for output .zst file
        frame_size_bytes: Target size for each frame (default: 4MB)
        compression_level: Zstd compression level 1-22 (default: 3)
        threads: Number of threads for compression (default: all CPUs)
        progress_callback: Optional callback(bytes_processed, total_bytes)

    Returns:
        SeekableZstdInfo with details about created file

    Raises:
        FileNotFoundError: If input file doesn't exist
        RuntimeError: If compression fails
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Ensure output has .zst extension
    if output_path.suffix.lower() != ".zst":
        output_path = output_path.with_suffix(".zst")

    # Check if input is compressed and needs decompression first
    input_compression = detect_compression(input_path)
    temp_file = None

    try:
        if input_compression != CompressionFormat.NONE:
            logger.info("decompressing_input", compression_format=input_compression.value)
            # Create temporary file for decompressed content
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".tmp")
            temp_file.close()

            # Decompress to temp file
            decompress_cmd = get_decompressor_command(input_compression, input_path)
            with open(temp_file.name, "wb") as out:
                proc = subprocess.run(decompress_cmd, stdout=out, stderr=subprocess.PIPE)
                if proc.returncode != 0:
                    raise RuntimeError(f"Decompression failed: {proc.stderr.decode()}")

            actual_input = Path(temp_file.name)
        else:
            actual_input = input_path

        input_size = actual_input.stat().st_size

        # Create seekable zstd using t2sz if available, otherwise use pyzstd
        if check_t2sz_available():
            _create_with_t2sz(
                actual_input,
                output_path,
                frame_size_bytes,
                compression_level,
                threads,
            )
        else:
            _create_with_pyzstd(
                actual_input,
                output_path,
                frame_size_bytes,
                compression_level,
                progress_callback,
            )

        # Get info about created file
        return get_seekable_zstd_info(output_path)

    finally:
        # Clean up temp file if created
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


def _create_with_t2sz(
    input_path: Path,
    output_path: Path,
    frame_size_bytes: int,
    compression_level: int,
    threads: Optional[int],
) -> None:
    """Create seekable zstd using t2sz tool."""
    cmd = [
        "t2sz",
        "-s",
        str(frame_size_bytes),
        "-l",
        str(compression_level),
        "-o",
        str(output_path),
    ]

    if threads:
        cmd.extend(["-T", str(threads)])

    cmd.append(str(input_path))

    logger.info("running_t2sz", command=cmd)
    proc = subprocess.run(cmd, capture_output=True)

    if proc.returncode != 0:
        raise RuntimeError(f"t2sz failed: {proc.stderr.decode()}")


def _create_with_pyzstd(
    input_path: Path,
    output_path: Path,
    frame_size_bytes: int,
    compression_level: int,
    progress_callback: Optional[Callable[[int, int], None]],
) -> None:
    """Create seekable zstd using pyzstd library.

    This creates a seekable zstd file by:
    1. Reading input in ~frame_size_bytes chunks, aligned to line boundaries
    2. Compressing each chunk as an independent frame
    3. Tracking compressed/decompressed sizes
    4. Writing seek table at the end

    Frame boundaries are aligned to newlines when possible, which:
    - Makes line counting accurate and simple
    - Prevents split lines across frame boundaries
    - Doesn't significantly affect compression ratio
    """
    try:
        import zstandard as zstd
    except ImportError:
        raise RuntimeError(
            "Neither t2sz nor zstandard Python package available. Install with: pip install zstandard, or install t2sz"
        )

    input_size = input_path.stat().st_size
    frames = []
    compressed_offset = 0

    cctx = zstd.ZstdCompressor(level=compression_level)

    with open(input_path, "rb") as fin, open(output_path, "wb") as fout:
        decompressed_offset = 0
        frame_index = 0
        bytes_read = 0

        while True:
            # Read a chunk of approximately frame_size_bytes
            chunk = fin.read(frame_size_bytes)
            if not chunk:
                break

            # Try to align frame boundary to a newline
            # Read until we find a newline (read in chunks to handle very long lines)
            if chunk and not chunk.endswith(b'\n'):
                # Keep reading until we find a newline
                while True:
                    # Read in small chunks to find the next newline
                    extra_chunk = fin.read(min(4096, frame_size_bytes))
                    if not extra_chunk:
                        # End of file - no more newlines
                        break

                    newline_pos = extra_chunk.find(b'\n')
                    if newline_pos != -1:
                        # Found a newline! Include bytes up to and including it
                        chunk += extra_chunk[: newline_pos + 1]
                        # Seek back to position after the newline
                        fin.seek(fin.tell() - len(extra_chunk) + newline_pos + 1)
                        break
                    else:
                        # No newline in this chunk, include it all and continue
                        chunk += extra_chunk

            # Compress this chunk as independent frame
            compressed_chunk = cctx.compress(chunk)

            # Track frame info
            frames.append(
                FrameInfo(
                    index=frame_index,
                    compressed_offset=compressed_offset,
                    compressed_size=len(compressed_chunk),
                    decompressed_offset=decompressed_offset,
                    decompressed_size=len(chunk),
                )
            )

            # Write compressed data
            fout.write(compressed_chunk)

            compressed_offset += len(compressed_chunk)
            decompressed_offset += len(chunk)
            frame_index += 1
            bytes_read += len(chunk)

            if progress_callback:
                progress_callback(bytes_read, input_size)

        # Write seek table
        _write_seek_table(fout, frames)


def _write_seek_table(fout, frames: list[FrameInfo]) -> None:
    """Write seek table to the end of the file.

    Format:
    - Skippable frame header (8 bytes)
    - For each frame: compressed_size (4 bytes) + decompressed_size (4 bytes)
    - Footer: magic (4 bytes) + num_frames (4 bytes) + flags (1 byte)
    """
    # Build seek table entries
    seek_entries = b""
    for frame in frames:
        seek_entries += struct.pack("<II", frame.compressed_size, frame.decompressed_size)

    # Footer
    footer = struct.pack("<IIB", SEEK_TABLE_FOOTER_MAGIC, len(frames), 0)  # flags = 0

    # Full seek table (entries + footer)
    seek_table = seek_entries + footer

    # Skippable frame header
    # Format: magic (4 bytes) + frame_size (4 bytes)
    skippable_header = struct.pack("<II", SEEKABLE_MAGIC, len(seek_table))

    # Write skippable frame containing seek table
    fout.write(skippable_header)
    fout.write(seek_table)


def find_frame_for_offset(frames: list[FrameInfo], decompressed_offset: int) -> int:
    """Find which frame contains the given decompressed offset.

    Args:
        frames: List of frame info
        decompressed_offset: Offset in decompressed stream

    Returns:
        Frame index containing the offset

    Raises:
        ValueError: If offset is out of range
    """
    for frame in frames:
        if frame.decompressed_offset <= decompressed_offset < frame.decompressed_end:
            return frame.index

    raise ValueError(f"Offset {decompressed_offset} is out of range")


def find_frames_for_range(
    frames: list[FrameInfo],
    start_offset: int,
    end_offset: int,
) -> list[int]:
    """Find frames that overlap with the given decompressed range.

    Args:
        frames: List of frame info
        start_offset: Start of range in decompressed stream
        end_offset: End of range in decompressed stream

    Returns:
        List of frame indices that overlap with the range
    """
    result = []
    for frame in frames:
        if frame.decompressed_offset < end_offset and frame.decompressed_end > start_offset:
            result.append(frame.index)
    return result
