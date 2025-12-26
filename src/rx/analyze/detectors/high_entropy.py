"""High entropy detector for secrets and tokens."""

import math
import re
from collections import Counter

from .base import AnomalyDetector, LineContext


class HighEntropyDetector(AnomalyDetector):
    """Detects high-entropy strings that might be secrets, tokens, or API keys."""

    # Minimum length of high-entropy substring to flag
    MIN_TOKEN_LENGTH = 40
    # Entropy threshold (bits per character)
    ENTROPY_THRESHOLD = 4.0
    # Patterns that often contain secrets
    SECRET_CONTEXT_PATTERNS = [
        re.compile(r'(?:api[_-]?key|apikey|secret|password|token|auth|credential|private[_-]?key)', re.IGNORECASE),
        re.compile(r'(?:bearer|authorization)\s*[:=]', re.IGNORECASE),
        re.compile(r'-----BEGIN\s+\w+\s+PRIVATE\s+KEY-----'),
    ]
    # Patterns for base64/hex strings
    HIGH_ENTROPY_PATTERNS = [
        re.compile(r'[A-Za-z0-9+/]{40,}={0,2}'),  # Base64
        re.compile(r'[a-fA-F0-9]{32,}'),  # Hex
        re.compile(r'[A-Za-z0-9_-]{20,}'),  # URL-safe base64 / API keys
    ]

    @property
    def name(self) -> str:
        return 'high_entropy'

    @property
    def category(self) -> str:
        return 'security'

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

    def check_line(self, ctx: LineContext) -> float | None:
        line = ctx.line

        # First check for secret context patterns (higher severity)
        for pattern in self.SECRET_CONTEXT_PATTERNS:
            if pattern.search(line):
                # Found secret context, look for high-entropy value
                for hp in self.HIGH_ENTROPY_PATTERNS:
                    match = hp.search(line)
                    if match:
                        token = match.group(0)
                        if len(token) >= self.MIN_TOKEN_LENGTH:
                            entropy = self._calculate_entropy(token)
                            if entropy >= self.ENTROPY_THRESHOLD:
                                return 0.85  # High severity for secrets in context
                return 0.6  # Medium severity for secret context without clear token

        # Check for high-entropy strings without context
        for pattern in self.HIGH_ENTROPY_PATTERNS:
            for match in pattern.finditer(line):
                token = match.group(0)
                if len(token) >= self.MIN_TOKEN_LENGTH:
                    entropy = self._calculate_entropy(token)
                    if entropy >= self.ENTROPY_THRESHOLD:
                        # Lower severity without secret context
                        return 0.5

        return None

    def get_description(self, lines: list[str]) -> str:
        if not lines:
            return 'Potential secret or token detected'

        first_line = lines[0]
        for pattern in self.SECRET_CONTEXT_PATTERNS:
            if pattern.search(first_line):
                return 'Potential secret in context'
        return 'High-entropy string detected'

    def get_prescan_patterns(self) -> list[tuple[re.Pattern, float]]:
        """Return patterns for ripgrep prescan."""
        return [
            (re.compile(r'api[_-]?key', re.IGNORECASE), 0.6),
            (re.compile(r'secret', re.IGNORECASE), 0.6),
            (re.compile(r'password', re.IGNORECASE), 0.6),
            (re.compile(r'token', re.IGNORECASE), 0.5),
            (re.compile(r'-----BEGIN.*KEY-----'), 0.85),
        ]
