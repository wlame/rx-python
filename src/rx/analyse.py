"""Experimental features!! File analysis module"""

import logging
import os
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from time import time
from typing import Any, Callable

from rx.parse import is_text_file
from rx.regex import calculate_regex_complexity
from rx.utils import NEWLINE_SYMBOL

logger = logging.getLogger(__name__)


def human_readable_size(size_bytes: int) -> str:
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


@dataclass
class FileAnalysisResult:
    """Result of analyzing a single file."""

    file_id: str
    filepath: str
    size_bytes: int
    size_human: str
    is_text: bool

    # Metadata
    created_at: str | None = None
    modified_at: str | None = None
    permissions: str | None = None
    owner: str | None = None

    # Text file metrics (only if is_text=True)
    line_count: int | None = None
    empty_line_count: int | None = None
    max_line_length: int | None = None
    avg_line_length: float | None = None
    median_line_length: float | None = None
    line_length_stddev: float | None = None

    # Additional metrics can be added by plugins
    custom_metrics: dict[str, Any] = field(default_factory=dict)


class FileAnalyzer:
    """
    Pluggable file analysis system.

    Supports adding custom analysis functions that can:
    - Analyze file metadata
    - Process file content line by line
    - Compute custom metrics
    """

    def __init__(self):
        self.file_hooks: list[Callable] = []
        self.line_hooks: list[Callable] = []
        self.post_hooks: list[Callable] = []

    def register_file_hook(self, hook: Callable):
        """
        Register a hook that processes file metadata.

        Hook signature: hook(filepath: str, result: FileAnalysisResult) -> None
        """
        self.file_hooks.append(hook)

    def register_line_hook(self, hook: Callable):
        """
        Register a hook that processes each line.

        Hook signature: hook(line: str, line_num: int, result: FileAnalysisResult) -> None
        """
        self.line_hooks.append(hook)

    def register_post_hook(self, hook: Callable):
        """
        Register a hook that runs after file processing.

        Hook signature: hook(result: FileAnalysisResult) -> None
        """
        self.post_hooks.append(hook)

    def analyze_file(self, filepath: str, file_id: str) -> FileAnalysisResult:
        """Analyze a single file with all registered hooks."""
        try:
            stat_info = os.stat(filepath)
            size_bytes = stat_info.st_size

            # Initialize result
            result = FileAnalysisResult(
                file_id=file_id,
                filepath=filepath,
                size_bytes=size_bytes,
                size_human=human_readable_size(size_bytes),
                is_text=is_text_file(filepath),
            )

            # File metadata
            result.created_at = datetime.fromtimestamp(stat_info.st_ctime).isoformat()
            result.modified_at = datetime.fromtimestamp(stat_info.st_mtime).isoformat()
            result.permissions = oct(stat_info.st_mode)[-3:]

            try:
                import pwd

                result.owner = pwd.getpwuid(stat_info.st_uid).pw_name
            except (ImportError, KeyError):
                result.owner = str(stat_info.st_uid)

            # Run file-level hooks
            for hook in self.file_hooks:
                try:
                    hook(filepath, result)
                except Exception as e:
                    logger.warning(f"File hook failed for {filepath}: {e}")

            # Analyze text files
            if result.is_text:
                self._analyze_text_file(filepath, result)

            # Run post-processing hooks
            for hook in self.post_hooks:
                try:
                    hook(result)
                except Exception as e:
                    logger.warning(f"Post hook failed for {filepath}: {e}")

            return result

        except Exception as e:
            logger.error(f"Failed to analyze {filepath}: {e}")
            # Return minimal result for failed files
            return FileAnalysisResult(
                file_id=file_id,
                filepath=filepath,
                size_bytes=0,
                size_human="0 B",
                is_text=False,
            )

    def _analyze_text_file(self, filepath: str, result: FileAnalysisResult):
        """Analyze text file content."""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()

            # Basic line metrics
            result.line_count = len(lines)

            # Analyze line lengths (excluding empty lines)
            non_empty_lines = [line.rstrip(NEWLINE_SYMBOL + '\r') for line in lines if line.strip()]
            empty_lines = [line for line in lines if not line.strip()]

            result.empty_line_count = len(empty_lines)

            if non_empty_lines:
                line_lengths = [len(line) for line in non_empty_lines]
                result.max_line_length = max(line_lengths)
                result.avg_line_length = statistics.mean(line_lengths)
                result.median_line_length = statistics.median(line_lengths)

                if len(line_lengths) > 1:
                    result.line_length_stddev = statistics.stdev(line_lengths)
                else:
                    result.line_length_stddev = 0.0
            else:
                result.max_line_length = 0
                result.avg_line_length = 0.0
                result.median_line_length = 0.0
                result.line_length_stddev = 0.0

            # Run line-level hooks
            for line_num, line in enumerate(lines, 1):
                for hook in self.line_hooks:
                    try:
                        hook(line, line_num, result)
                    except Exception as e:
                        logger.warning(f"Line hook {hook.__name__} failed at {filepath}:{line_num}: {e}")

        except Exception as e:
            logger.error(f"Failed to analyze text content of {filepath}: {e}")


def analyse_path(paths: list[str], max_workers: int = 10) -> dict[str, Any]:
    """
    Analyze files at given paths.

    Args:
        paths: List of file or directory paths
        max_workers: Maximum number of parallel workers

    Returns:
        Dictionary with analysis results in ID-based format
    """
    start_time = time()

    # Collect all files to analyze
    files_to_analyze = []
    for path in paths:
        if os.path.isfile(path):
            files_to_analyze.append(path)
        elif os.path.isdir(path):
            # Scan directory for files
            for root, dirs, files in os.walk(path):
                for file in files:
                    filepath = os.path.join(root, file)
                    files_to_analyze.append(filepath)
        else:
            logger.warning(f"Path not found: {path}")

    # Create file IDs
    file_ids = {f"f{i + 1}": filepath for i, filepath in enumerate(files_to_analyze)}

    # Analyze files in parallel
    analyzer = FileAnalyzer()
    results = []
    scanned_files = []
    skipped_files = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(analyzer.analyze_file, filepath, file_id): (file_id, filepath)
            for file_id, filepath in file_ids.items()
        }

        for future in as_completed(future_to_file):
            file_id, filepath = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
                scanned_files.append(filepath)
            except Exception as e:
                logger.error(f"Analysis failed for {filepath}: {e}")
                skipped_files.append(filepath)

    elapsed_time = time() - start_time

    # Build response in ID-based format
    return {
        'path': ', '.join(paths) if len(paths) > 1 else paths[0],
        'time': elapsed_time,
        'files': file_ids,
        'results': [
            {
                'file': r.file_id,
                'size_bytes': r.size_bytes,
                'size_human': r.size_human,
                'is_text': r.is_text,
                'created_at': r.created_at,
                'modified_at': r.modified_at,
                'permissions': r.permissions,
                'owner': r.owner,
                'line_count': r.line_count,
                'empty_line_count': r.empty_line_count,
                'max_line_length': r.max_line_length,
                'avg_line_length': r.avg_line_length,
                'median_line_length': r.median_line_length,
                'line_length_stddev': r.line_length_stddev,
                'custom_metrics': r.custom_metrics,
            }
            for r in results
        ],
        'scanned_files': scanned_files,
        'skipped_files': skipped_files,
    }


# Re-export calculate_regex_complexity for backward compatibility
__all__ = ['analyse_path', 'calculate_regex_complexity', 'FileAnalyzer', 'FileAnalysisResult']
