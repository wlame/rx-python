"""CLI command builder for generating equivalent CLI commands from API requests.

This module provides utilities to generate CLI command strings that are equivalent
to API endpoint calls. This helps users understand how to use the CLI for the
same operations they perform via the API.

Example:
    >>> from rx.cli_command_builder import build_cli_command
    >>> cmd = build_cli_command("trace", {"path": ["/var/log/app.log"], "regexp": ["error"]})
    >>> print(cmd)
    rx -e error /var/log/app.log
"""

import shlex
from typing import Any, Callable

import structlog

logger = structlog.get_logger()

# Registry mapping endpoint names to builder functions
CLI_BUILDERS: dict[str, Callable[[dict[str, Any]], str | None]] = {}


def shell_quote(value: str) -> str:
    """Quote a string for safe shell usage.

    Uses shlex.quote for proper POSIX shell escaping of special characters,
    spaces, quotes, and other shell metacharacters.

    Args:
        value: The string to quote.

    Returns:
        A shell-safe quoted string.

    Examples:
        >>> shell_quote("hello")
        'hello'
        >>> shell_quote("hello world")
        "'hello world'"
        >>> shell_quote("it's a test")
        "\"it's a test\""
    """
    return shlex.quote(value)


def register_cli_builder(endpoint_name: str):
    """Decorator to register a CLI command builder for an endpoint.

    Args:
        endpoint_name: The name of the endpoint (e.g., "trace", "samples").

    Returns:
        A decorator that registers the function in CLI_BUILDERS.

    Example:
        @register_cli_builder("trace")
        def build_trace_cli(params: dict) -> str:
            ...
    """

    def decorator(func: Callable[[dict[str, Any]], str | None]):
        CLI_BUILDERS[endpoint_name] = func
        return func

    return decorator


def build_cli_command(endpoint_name: str, params: dict[str, Any]) -> str | None:
    """Build CLI command for an endpoint given its parameters.

    Args:
        endpoint_name: The name of the endpoint (e.g., "trace", "samples").
        params: Dictionary of parameters from the API request.

    Returns:
        The CLI command string, or None if no CLI equivalent exists.

    Example:
        >>> build_cli_command("trace", {"path": ["/var/log"], "regexp": ["error"]})
        'rx -e error /var/log'
    """
    builder = CLI_BUILDERS.get(endpoint_name)
    if builder:
        try:
            return builder(params)
        except Exception as e:
            logger.warning("Failed to build CLI command", endpoint=endpoint_name, error=str(e))
            return None
    return None


def add_cli_command(
    response: dict[str, Any],
    endpoint_name: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Add cli_command field to a response dictionary.

    This is a helper function to be called from API endpoints to add
    the CLI equivalent command to the response.

    Args:
        response: The response dictionary to modify.
        endpoint_name: The name of the endpoint.
        params: The request parameters.

    Returns:
        The modified response dictionary with cli_command field added.
    """
    cli_cmd = build_cli_command(endpoint_name, params)
    if cli_cmd:
        response["cli_command"] = cli_cmd
        logger.debug("CLI equivalent", cli_command=cli_cmd)
    return response


# =============================================================================
# Endpoint-specific CLI command builders
# =============================================================================


@register_cli_builder("trace")
def build_trace_cli(params: dict[str, Any]) -> str:
    """Build CLI command for /v1/trace endpoint.

    Mapping:
        - regexp -> -e (multiple, short form preferred)
        - path -> positional args
        - max_results -> --max-results

    Args:
        params: Dictionary with keys: regexp, path, max_results, etc.

    Returns:
        CLI command string.

    Example:
        >>> build_trace_cli({"path": ["/var/log"], "regexp": ["error"], "max_results": 100})
        'rx -e error /var/log --max-results=100'
    """
    parts = ["rx"]

    # Patterns: use -e for each pattern
    regexps = params.get("regexp", [])
    if isinstance(regexps, str):
        regexps = [regexps]
    for pattern in regexps:
        parts.append(f"-e {shell_quote(pattern)}")

    # Paths: add as positional arguments
    paths = params.get("path", [])
    if isinstance(paths, str):
        paths = [paths]
    for path in paths:
        parts.append(shell_quote(path))

    # Optional: max_results
    if params.get("max_results"):
        parts.append(f"--max-results={params['max_results']}")

    return " ".join(parts)


@register_cli_builder("samples")
def build_samples_cli(params: dict[str, Any]) -> str:
    """Build CLI command for /v1/samples endpoint.

    Mapping:
        - path -> positional arg
        - offsets -> -b (multiple)
        - lines -> -l (multiple)
        - context -> -c
        - before_context -> -B
        - after_context -> -A

    Args:
        params: Dictionary with keys: path, offsets, lines, context, etc.

    Returns:
        CLI command string.

    Example:
        >>> build_samples_cli({"path": "/var/log/app.log", "lines": "100,200", "context": 5})
        'rx samples /var/log/app.log -l 100 -l 200 -c 5'
    """
    parts = ["rx", "samples"]

    # Path (required, positional)
    path = params.get("path", "")
    if path:
        parts.append(shell_quote(path))

    # Offsets or lines (mutually exclusive)
    if params.get("offsets"):
        offsets = params["offsets"]
        if isinstance(offsets, str):
            for offset in offsets.split(","):
                offset = offset.strip()
                if offset:
                    parts.append(f"-b {offset}")
        else:
            parts.append(f"-b {offsets}")
    elif params.get("lines"):
        lines = params["lines"]
        if isinstance(lines, str):
            for line in lines.split(","):
                line = line.strip()
                if line:
                    parts.append(f"-l {line}")
        else:
            parts.append(f"-l {lines}")

    # Context options
    if params.get("context") is not None:
        parts.append(f"-c {params['context']}")
    if params.get("before_context") is not None:
        parts.append(f"-B {params['before_context']}")
    if params.get("after_context") is not None:
        parts.append(f"-A {params['after_context']}")

    return " ".join(parts)


@register_cli_builder("complexity")
def build_complexity_cli(params: dict[str, Any]) -> str:
    """Build CLI command for /v1/complexity endpoint.

    Mapping:
        - regex -> positional arg

    Args:
        params: Dictionary with keys: regex.

    Returns:
        CLI command string.

    Example:
        >>> build_complexity_cli({"regex": "(a+)+"})
        "rx check '(a+)+'"
    """
    parts = ["rx", "check"]

    regex = params.get("regex", "")
    if regex:
        parts.append(shell_quote(regex))

    return " ".join(parts)


@register_cli_builder("index_get")
def build_index_get_cli(params: dict[str, Any]) -> str:
    """Build CLI command for GET /v1/index endpoint.

    Mapping:
        - path -> positional arg
        - Adds --info and --json flags

    Args:
        params: Dictionary with keys: path.

    Returns:
        CLI command string.

    Example:
        >>> build_index_get_cli({"path": "/var/log/app.log"})
        'rx index /var/log/app.log --info --json'
    """
    parts = ["rx", "index"]

    path = params.get("path", "")
    if path:
        parts.append(shell_quote(path))

    parts.append("--info")
    parts.append("--json")

    return " ".join(parts)


@register_cli_builder("index_post")
def build_index_post_cli(params: dict[str, Any]) -> str:
    """Build CLI command for POST /v1/index endpoint.

    Mapping:
        - path -> positional arg
        - force -> --force
        - analyze -> --analyze

    Args:
        params: Dictionary with keys: path, force, analyze.

    Returns:
        CLI command string.

    Example:
        >>> build_index_post_cli({"path": "/var/log/app.log", "analyze": True})
        'rx index /var/log/app.log --analyze'
    """
    parts = ["rx", "index"]

    path = params.get("path", "")
    if path:
        parts.append(shell_quote(path))

    if params.get("force"):
        parts.append("--force")
    if params.get("analyze"):
        parts.append("--analyze")

    return " ".join(parts)


@register_cli_builder("compress")
def build_compress_cli(params: dict[str, Any]) -> str:
    """Build CLI command for POST /v1/compress endpoint.

    Mapping:
        - input_path -> positional arg
        - output_path -> -o
        - frame_size -> -s
        - compression_level -> -l

    Args:
        params: Dictionary with keys: input_path, output_path, frame_size, compression_level.

    Returns:
        CLI command string.

    Example:
        >>> build_compress_cli({"input_path": "/var/log/app.log", "compression_level": 5})
        'rx compress /var/log/app.log -l 5'
    """
    parts = ["rx", "compress"]

    input_path = params.get("input_path", "")
    if input_path:
        parts.append(shell_quote(input_path))

    if params.get("output_path"):
        parts.append(f"-o {shell_quote(params['output_path'])}")
    if params.get("frame_size"):
        parts.append(f"-s {params['frame_size']}")
    if params.get("compression_level"):
        parts.append(f"-l {params['compression_level']}")

    return " ".join(parts)
