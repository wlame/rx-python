"""RxTracer - FastAPI application"""

# Initialize structlog defaults (stderr, WARNING level) before any module logs.
# CLI/web entry points call configure_logging() to override.
import rx.log  # noqa: F401
