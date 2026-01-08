"""High entropy detector for secrets and tokens."""

import logging
import math
import re
from collections import Counter

from .base import AnomalyDetector, LineContext, register_detector


logger = logging.getLogger(__name__)


@register_detector
class HighEntropyDetector(AnomalyDetector):
    """Detects high-entropy strings that might be secrets, tokens, or API keys.

    Tuned to be strict - only flags actual hashes, tokens, and keys, not normal
    text or email addresses.
    """

    def __init__(self, filepath: str | None = None):
        """Initialize the detector.

        Args:
            filepath: Path to file being analyzed (for logging context).
        """
        self._filepath = filepath
        self._detection_count = 0
        self._detection_types: dict[str, int] = {}
        logger.debug(f'[high_entropy] Initialized for file: {filepath}')

    # Minimum length of high-entropy substring to flag
    MIN_TOKEN_LENGTH = 32
    # Entropy threshold (bits per character) - raised to reduce false positives
    ENTROPY_THRESHOLD = 4.5
    # Minimum entropy for hex strings (they have max ~4 bits)
    HEX_ENTROPY_THRESHOLD = 3.5

    # Patterns that often contain secrets - must have assignment/value context
    SECRET_CONTEXT_PATTERNS = [
        re.compile(r'(?:api[_-]?key|apikey)\s*[=:]\s*["\']?[A-Za-z0-9+/=_-]{20,}', re.IGNORECASE),
        re.compile(r'(?:secret|secret[_-]?key)\s*[=:]\s*["\']?[A-Za-z0-9+/=_-]{20,}', re.IGNORECASE),
        re.compile(r'(?:password|passwd|pwd)\s*[=:]\s*["\']?[^\s"\']{8,}', re.IGNORECASE),
        re.compile(r'(?:token|auth[_-]?token|access[_-]?token)\s*[=:]\s*["\']?[A-Za-z0-9+/=_-]{20,}', re.IGNORECASE),
        re.compile(r'(?:bearer|authorization)[:\s]+[A-Za-z0-9+/=_.-]{20,}', re.IGNORECASE),
        re.compile(r'(?:private[_-]?key|credential)\s*[=:]\s*["\']?[A-Za-z0-9+/=_-]{20,}', re.IGNORECASE),
        re.compile(r'-----BEGIN\s+\w+\s+PRIVATE\s+KEY-----'),
        re.compile(r'-----BEGIN\s+RSA\s+PRIVATE\s+KEY-----'),
        re.compile(r'-----BEGIN\s+OPENSSH\s+PRIVATE\s+KEY-----'),
        # AWS access key pattern (AKIA followed by 16 uppercase alphanumeric)
        re.compile(r'(?:aws[_-]?)?(?:access[_-]?key[_-]?id|key[_-]?id)\s*[=:]\s*["\']?AKIA[0-9A-Z]{16}', re.IGNORECASE),
    ]

    # Patterns for clearly non-human strings (hashes, tokens)
    # These are checked independently without needing secret context
    HIGH_ENTROPY_PATTERNS = [
        # Hex strings (SHA256, SHA512, MD5, etc.)
        re.compile(r'(?<![A-Fa-f0-9])[a-fA-F0-9]{32}(?![A-Fa-f0-9])'),  # MD5
        re.compile(r'(?<![A-Fa-f0-9])[a-fA-F0-9]{40}(?![A-Fa-f0-9])'),  # SHA1
        re.compile(r'(?<![A-Fa-f0-9])[a-fA-F0-9]{64}(?![A-Fa-f0-9])'),  # SHA256
        re.compile(r'(?<![A-Fa-f0-9])[a-fA-F0-9]{128}(?![A-Fa-f0-9])'),  # SHA512
        # UUID-like patterns (36 chars with dashes)
        re.compile(r'[a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12}'),
        # JWT tokens (three base64url-encoded parts separated by dots)
        re.compile(r'eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}'),
        # AWS access keys (always start with AKIA)
        re.compile(r'AKIA[0-9A-Z]{16}'),
    ]

    # Patterns to exclude (common false positives)
    EXCLUDE_PATTERNS = [
        re.compile(r'@'),  # Email addresses
        re.compile(r'\.(com|org|net|io|edu|gov|co\.|info)'),  # Domain names
        re.compile(r'https?://'),  # URLs
        re.compile(r'\s'),  # Contains whitespace
    ]

    @property
    def name(self) -> str:
        return 'high_entropy'

    @property
    def category(self) -> str:
        return 'security'

    @property
    def detector_description(self) -> str:
        return 'Detects high-entropy strings that may be secrets, tokens, API keys, or cryptographic hashes'

    @property
    def severity_min(self) -> float:
        return 0.5

    @property
    def severity_max(self) -> float:
        return 0.85

    @property
    def examples(self) -> list[str]:
        return ['api_key=...', 'Bearer token', 'AWS AKIA...', 'JWT eyJ...', 'SHA256 hash', 'Private key']

    def _calculate_entropy(self, s: str) -> float:
        """Calculate Shannon entropy of a string."""
        if not s:
            return 0.0

        counts = Counter(s)
        length = len(s)
        entropy = 0.0

        for count in counts.values():
            if count > 0:
                p = count / length
                entropy -= p * math.log2(p)

        return entropy

    def _is_excluded(self, token: str) -> bool:
        """Check if token matches common false positive patterns."""
        for pattern in self.EXCLUDE_PATTERNS:
            if pattern.search(token):
                return True
        return False

    def _is_hex_string(self, s: str) -> bool:
        """Check if string is purely hex characters."""
        return all(c in '0123456789abcdefABCDEF' for c in s)

    def check_line(self, ctx: LineContext) -> float | None:
        line = ctx.line

        # First check for secret context patterns (higher severity)
        for pattern in self.SECRET_CONTEXT_PATTERNS:
            match = pattern.search(line)
            if match:
                self._detection_count += 1
                self._detection_types['secret_context'] = self._detection_types.get('secret_context', 0) + 1
                return 0.85  # High severity for secrets in context

        # Check for high-entropy strings without context
        # These patterns are specific enough (JWT, AWS keys, hashes) to flag directly
        for pattern in self.HIGH_ENTROPY_PATTERNS:
            for match in pattern.finditer(line):
                token = match.group(0)

                # Skip if matches exclusion patterns
                if self._is_excluded(token):
                    continue

                # JWT tokens are always flagged (very specific pattern)
                if token.startswith('eyJ') and '.' in token:
                    self._detection_count += 1
                    self._detection_types['jwt'] = self._detection_types.get('jwt', 0) + 1
                    return 0.7

                # AWS keys are always flagged (very specific pattern)
                if token.startswith('AKIA'):
                    self._detection_count += 1
                    self._detection_types['aws_key'] = self._detection_types.get('aws_key', 0) + 1
                    return 0.85

                # For hex hashes, check length and entropy
                if len(token) >= self.MIN_TOKEN_LENGTH:
                    entropy = self._calculate_entropy(token)

                    # Use lower threshold for hex strings (they have max ~4 bits entropy)
                    if self._is_hex_string(token):
                        if entropy >= self.HEX_ENTROPY_THRESHOLD:
                            self._detection_count += 1
                            self._detection_types['hex_hash'] = self._detection_types.get('hex_hash', 0) + 1
                            return 0.6  # Medium-high for hex hashes
                    else:
                        if entropy >= self.ENTROPY_THRESHOLD:
                            self._detection_count += 1
                            self._detection_types['high_entropy'] = self._detection_types.get('high_entropy', 0) + 1
                            return 0.5  # Medium for other high-entropy strings

        return None

    def get_description(self, lines: list[str]) -> str:
        if not lines:
            return 'Potential secret or token detected'

        first_line = lines[0]
        for pattern in self.SECRET_CONTEXT_PATTERNS:
            if pattern.search(first_line):
                return 'Potential secret in context'

        # Check for specific patterns
        if re.search(r'eyJ[A-Za-z0-9_-]+\.eyJ', first_line):
            return 'JWT token detected'
        if re.search(r'AKIA[0-9A-Z]{16}', first_line):
            return 'AWS access key detected'
        if re.search(r'[a-fA-F0-9]{64}', first_line):
            return 'SHA256 hash detected'
        if re.search(r'[a-fA-F0-9]{40}', first_line):
            return 'SHA1 hash detected'
        if re.search(r'[a-fA-F0-9]{32}', first_line):
            return 'MD5 hash detected'

        return 'High-entropy string detected'

    def get_prescan_patterns(self) -> list[tuple[re.Pattern, float]]:
        """Return patterns for ripgrep prescan."""
        return [
            (re.compile(r'api[_-]?key\s*[=:]', re.IGNORECASE), 0.6),
            (re.compile(r'secret\s*[=:]', re.IGNORECASE), 0.6),
            (re.compile(r'password\s*[=:]', re.IGNORECASE), 0.6),
            (re.compile(r'token\s*[=:]', re.IGNORECASE), 0.5),
            (re.compile(r'-----BEGIN.*KEY-----'), 0.85),
            (re.compile(r'AKIA[0-9A-Z]{16}'), 0.85),
            (re.compile(r'eyJ[A-Za-z0-9_-]+\.eyJ'), 0.7),
        ]
