"""Prefix pattern extraction using Drain3.

This module provides functionality to automatically detect dominant log line
prefix patterns from sample data. It uses the Drain3 library for log template
mining and extracts the common prefix structure.

Example usage:
    extractor = PrefixPatternExtractor()
    pattern = extractor.extract_from_file('/path/to/logfile.log')
    if pattern:
        print(f"Prefix: {pattern.pattern}")
        print(f"Regex: {pattern.regex}")
        print(f"Coverage: {pattern.coverage:.1%}")
"""

import os
import re
from collections import Counter
from dataclasses import dataclass

import structlog
from drain3 import TemplateMiner
from drain3.masking import LogMasker, RegexMaskingInstruction
from drain3.template_miner_config import TemplateMinerConfig


logger = structlog.get_logger()


@dataclass
class PrefixPattern:
    """Detected prefix pattern information."""

    pattern: str  # Masked token pattern, e.g., "<DATE> <TIME> <COMPONENT>"
    regex: str  # Compiled regex string to match the pattern
    coverage: float  # Fraction of lines matching (0.0-1.0)
    prefix_length: int  # Approximate prefix length in characters
    token_count: int  # Number of tokens in prefix


# Default masking instructions for common log elements
# Order matters - more specific patterns should come first
DEFAULT_MASKING_INSTRUCTIONS = [
    # ISO 8601 datetime: 2025-12-10T07:49:50.123Z (combined date+time)
    (r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:[.,]\d{3,6})?Z?', 'DATETIME'),
    # ISO date formats
    (r'\d{4}-\d{2}-\d{2}', 'DATE'),
    # Time with milliseconds/microseconds (with optional Z suffix)
    (r'\d{2}:\d{2}:\d{2}[.,]\d{3,6}Z?', 'TIME'),
    # Time without milliseconds
    (r'\d{2}:\d{2}:\d{2}', 'TIME'),
    # Syslog-style date: Dec 10, Jan  5
    (r'[A-Z][a-z]{2}\s+\d{1,2}', 'SYSDATE'),
    # Bracketed numeric IDs: [12345678]
    (r'\[\d+\]', 'NUM_ID'),
    # Bracketed component names: [my-component.service]
    (r'\[[\w.-]+\]', 'COMPONENT'),
    # Common log levels (to normalize)
    (r'\b(?:DEBUG|INFO|WARN(?:ING)?|ERROR|CRITICAL|FATAL|TRACE)\b', 'LEVEL'),
    # Hex strings (8+ chars): transaction IDs, hashes
    (r'\b[0-9A-Fa-f]{8,}\b', 'HEX'),
    # IP addresses
    (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', 'IP'),
    # Standalone numbers (last, to not interfere with above)
    (r'\b\d+\b', 'NUM'),
]

# Regex patterns to convert masked tokens back to matching regex
TOKEN_TO_REGEX = {
    '<DATETIME>': r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:[.,]\d{3,6})?Z?',
    '<DATE>': r'\d{4}-\d{2}-\d{2}',
    '<TIME>': r'\d{2}:\d{2}:\d{2}(?:[.,]\d{3,6})?Z?',
    '<SYSDATE>': r'[A-Z][a-z]{2}\s+\d{1,2}',
    '<NUM_ID>': r'\[\d+\]',
    '<COMPONENT>': r'\[[\w.-]+\]',
    '<LEVEL>': r'(?:DEBUG|INFO|WARN(?:ING)?|ERROR|CRITICAL|FATAL|TRACE)',
    '<HEX>': r'[0-9A-Fa-f]{8,}',
    '<IP>': r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',
    '<NUM>': r'\d+',
    '<*>': r'\S+',  # Drain3 wildcard
}


class PrefixPatternExtractor:
    """Extract dominant log line prefix patterns using Drain3.

    This class analyzes sample log lines to find the most common prefix
    structure. It's useful for detecting lines that deviate from the
    normal log format.
    """

    def __init__(
        self,
        masking_instructions: list[tuple[str, str]] | None = None,
        similarity_threshold: float = 0.3,
        coverage_threshold: float = 0.90,
        max_prefix_tokens: int = 8,
    ):
        """Initialize the extractor.

        Args:
            masking_instructions: List of (regex, name) tuples for masking.
                Defaults to DEFAULT_MASKING_INSTRUCTIONS.
            similarity_threshold: Drain3 similarity threshold (lower = more merging).
            coverage_threshold: Minimum coverage for a pattern to be considered dominant.
            max_prefix_tokens: Maximum number of tokens to consider for prefix.
        """
        self.masking_instructions = masking_instructions or DEFAULT_MASKING_INSTRUCTIONS
        self.similarity_threshold = similarity_threshold
        self.coverage_threshold = coverage_threshold
        self.max_prefix_tokens = max_prefix_tokens

        # Build masker
        instructions = [RegexMaskingInstruction(pattern, name) for pattern, name in self.masking_instructions]
        self.masker = LogMasker(instructions, mask_prefix='<', mask_suffix='>')

    def _create_miner(self) -> TemplateMiner:
        """Create a fresh TemplateMiner instance."""
        config = TemplateMinerConfig()
        config.drain_sim_th = self.similarity_threshold
        config.drain_depth = 4
        config.drain_max_children = 100
        return TemplateMiner(config=config)

    def extract_from_lines(self, lines: list[str]) -> PrefixPattern | None:
        """Extract prefix pattern from sample lines.

        Args:
            lines: List of log lines to analyze.

        Returns:
            PrefixPattern if a dominant pattern is found, None otherwise.
        """
        logger.debug("Extracting prefix pattern", line_count=len(lines))

        if not lines:
            logger.debug("No lines provided, returning None")
            return None

        miner = self._create_miner()

        # Process all lines through Drain3
        processed_count = 0
        for line in lines:
            if not line.strip():
                continue
            masked = self.masker.mask(line)
            miner.add_log_message(masked)
            processed_count += 1

        logger.debug("Processed lines through Drain3", processed_count=processed_count)

        if not miner.drain.clusters:
            logger.debug("No clusters created by Drain3, returning None")
            return None

        total_lines = sum(c.size for c in miner.drain.clusters)
        logger.debug("Drain3 clustering complete", cluster_count=len(miner.drain.clusters), total_lines=total_lines)

        if total_lines == 0:
            logger.debug("Total lines in clusters is 0, returning None")
            return None

        # Log top clusters for debugging
        sorted_clusters = sorted(miner.drain.clusters, key=lambda c: -c.size)
        for i, cluster in enumerate(sorted_clusters[:5]):
            pct = cluster.size / total_lines * 100
            template = cluster.get_template()
            logger.debug(
                "Top cluster",
                rank=i + 1,
                size=cluster.size,
                pct=round(pct, 1),
                template=template[:80],
            )

        # Extract prefix tokens from each cluster
        prefix_counts: Counter[tuple[str, ...]] = Counter()
        for cluster in miner.drain.clusters:
            template = cluster.get_template()
            tokens = tuple(template.split()[: self.max_prefix_tokens])
            prefix_counts[tokens] += cluster.size

        logger.debug("Extracted unique prefix patterns", count=len(prefix_counts), max_tokens=self.max_prefix_tokens)

        # Log top prefix patterns
        for prefix, count in prefix_counts.most_common(5):
            pct = count / total_lines * 100
            logger.debug("Top prefix pattern", count=count, pct=round(pct, 1), pattern=" ".join(prefix)[:60])

        # Find the longest common prefix that meets coverage threshold
        # Try progressively shorter prefixes
        best_prefix: tuple[str, ...] | None = None
        best_coverage = 0.0

        logger.debug(
            "Searching for prefix pattern",
            coverage_threshold=self.coverage_threshold,
            max_tokens=self.max_prefix_tokens,
        )

        for prefix_len in range(self.max_prefix_tokens, 0, -1):
            shortened: Counter[tuple[str, ...]] = Counter()
            for prefix, count in prefix_counts.items():
                short = prefix[:prefix_len]
                shortened[short] += count

            for prefix, count in shortened.most_common():
                coverage = count / total_lines
                if coverage >= self.coverage_threshold:
                    # Found a prefix that meets threshold
                    if best_prefix is None or len(prefix) > len(best_prefix):
                        best_prefix = prefix
                        best_coverage = coverage
                        logger.debug(
                            "Found prefix candidate",
                            prefix_len=prefix_len,
                            coverage=round(coverage, 3),
                            pattern=" ".join(prefix)[:60],
                        )
                    break

        if best_prefix is None:
            # No prefix meets threshold - check if we have any dominant pattern
            most_common, count = prefix_counts.most_common(1)[0]
            coverage = count / total_lines
            logger.debug(
                "No prefix met coverage threshold",
                threshold=self.coverage_threshold,
                best_coverage=round(coverage, 3),
            )
            if coverage >= 0.5:  # At least 50% for fallback
                best_prefix = most_common
                best_coverage = coverage
                logger.debug("Using fallback prefix", coverage=round(coverage, 3))
            else:
                logger.info(
                    "No dominant prefix pattern found",
                    best_coverage=round(coverage, 3),
                    fallback_threshold=0.5,
                    primary_threshold=self.coverage_threshold,
                )
                return None

        # Convert to pattern string and regex
        pattern_str = ' '.join(best_prefix)
        regex = self._prefix_to_regex(best_prefix)

        logger.debug("Generated regex", regex=regex[:100])

        # Estimate prefix length from sample lines
        prefix_length = self._estimate_prefix_length(lines, regex)

        logger.info(
            "Prefix pattern detected",
            pattern=pattern_str,
            coverage=round(best_coverage, 3),
            prefix_length=prefix_length,
        )

        return PrefixPattern(
            pattern=pattern_str,
            regex=regex,
            coverage=best_coverage,
            prefix_length=prefix_length,
            token_count=len(best_prefix),
        )

    def _prefix_to_regex(self, prefix_tokens: tuple[str, ...]) -> str:
        """Convert masked prefix tokens to a regex pattern.

        Args:
            prefix_tokens: Tuple of masked tokens.

        Returns:
            Regex pattern string.
        """
        regex_parts = ['^']

        for i, token in enumerate(prefix_tokens):
            if token in TOKEN_TO_REGEX:
                regex_parts.append(TOKEN_TO_REGEX[token])
            else:
                # Token might contain embedded masked tokens like "daemon<NUM_ID>:"
                # Replace any <TOKEN> patterns within the token
                converted = self._convert_embedded_tokens(token)
                regex_parts.append(converted)

            # Add flexible whitespace between tokens (except after last)
            if i < len(prefix_tokens) - 1:
                regex_parts.append(r'\s+')

        return ''.join(regex_parts)

    def _convert_embedded_tokens(self, token: str) -> str:
        """Convert a token that may contain embedded masked tokens.

        For example: "daemon<NUM_ID>:" -> "daemon\\[\\d+\\]:"

        Args:
            token: Token string that may contain <TOKEN> patterns.

        Returns:
            Regex pattern string.
        """
        # Pattern to find masked tokens like <DATE>, <NUM_ID>, etc.
        mask_pattern = re.compile(r'<[A-Z_*]+>')

        result = []
        last_end = 0

        for match in mask_pattern.finditer(token):
            # Add escaped literal text before this match
            if match.start() > last_end:
                result.append(re.escape(token[last_end : match.start()]))

            # Convert the masked token to regex
            mask_token = match.group()
            if mask_token in TOKEN_TO_REGEX:
                result.append(TOKEN_TO_REGEX[mask_token])
            else:
                # Unknown mask token - treat as wildcard
                result.append(r'\S+')

            last_end = match.end()

        # Add any remaining literal text
        if last_end < len(token):
            result.append(re.escape(token[last_end:]))

        return ''.join(result) if result else re.escape(token)

    def _estimate_prefix_length(self, lines: list[str], regex: str) -> int:
        """Estimate typical prefix length in characters.

        Args:
            lines: Sample lines.
            regex: Compiled regex pattern.

        Returns:
            Median prefix length in characters.
        """
        pattern = re.compile(regex)
        lengths = []

        for line in lines[:100]:  # Sample first 100
            match = pattern.match(line)
            if match:
                lengths.append(match.end())

        if not lengths:
            return 0

        lengths.sort()
        return lengths[len(lengths) // 2]  # Median

    def sample_file(
        self,
        filepath: str,
        sample_size: int = 1000,
        skip_ratio: float = 0.05,
    ) -> list[str]:
        """Sample lines from a file for pattern extraction.

        Skips the first portion of the file (often initialization logs)
        and samples from the middle/end where format is more stable.

        Args:
            filepath: Path to file.
            sample_size: Number of lines to sample.
            skip_ratio: Fraction of file to skip at start (0.0-1.0).

        Returns:
            List of sampled lines.
        """
        file_size = os.path.getsize(filepath)
        skip_bytes = int(file_size * skip_ratio)

        logger.debug(
            "Sampling file",
            filepath=filepath,
            file_size=file_size,
            skip_bytes=skip_bytes,
            skip_ratio=skip_ratio,
        )

        lines = []
        empty_count = 0
        try:
            with open(filepath, 'rb') as f:
                # Skip initial portion
                if skip_bytes > 0:
                    f.seek(skip_bytes)
                    # Skip partial line
                    partial = f.readline()
                    logger.debug("Skipped to byte offset", skip_bytes=skip_bytes, partial_line_bytes=len(partial))

                # Read lines
                for _ in range(sample_size * 2):  # Read extra in case of empty lines
                    line_bytes = f.readline()
                    if not line_bytes:
                        break

                    try:
                        line = line_bytes.decode('utf-8', errors='replace').rstrip('\r\n')
                        if line.strip():
                            lines.append(line)
                            if len(lines) >= sample_size:
                                break
                        else:
                            empty_count += 1
                    except Exception as e:
                        logger.debug("Failed to decode line", error=str(e))
                        continue

        except Exception as e:
            logger.warning("Failed to sample file", filepath=filepath, error=str(e))
            return []

        logger.debug("Sampled lines from file", line_count=len(lines), empty_skipped=empty_count)

        # Log a few sample lines for debugging
        for i, line in enumerate(lines[:3]):
            logger.debug("Sample line", index=i + 1, line=line[:100])

        return lines

    def extract_from_file(
        self,
        filepath: str,
        sample_size: int = 1000,
        skip_ratio: float = 0.05,
    ) -> PrefixPattern | None:
        """Extract prefix pattern from a file.

        Convenience method that samples the file and extracts the pattern.

        Args:
            filepath: Path to file.
            sample_size: Number of lines to sample.
            skip_ratio: Fraction of file to skip at start.

        Returns:
            PrefixPattern if found, None otherwise.
        """
        logger.debug("extract_from_file called", filepath=filepath, sample_size=sample_size, skip_ratio=skip_ratio)

        lines = self.sample_file(filepath, sample_size, skip_ratio)
        if not lines:
            logger.info("No lines sampled from file", filepath=filepath)
            return None

        result = self.extract_from_lines(lines)
        if result is None:
            logger.info("No prefix pattern found for file", filepath=filepath)
        return result
