"""Async webhook notifications for trace events.

This module provides functionality to call external HTTP endpoints (webhooks)
when specific events occur during regex tracing operations.

Events:
- on_file_scanned: Called when a file scan is completed
- on_match_found: Called for each individual match found
- on_trace_complete: Called when the entire trace request finishes

Security Warning:
    Custom hooks can be used to make HTTP requests to arbitrary URLs.
    In untrusted environments, set RX_DISABLE_CUSTOM_HOOKS=true to only
    allow hooks configured via environment variables.
"""

import structlog
import os
import uuid
from time import time
from typing import Optional

import httpx
from pydantic import BaseModel, HttpUrl

from rx.cli import prometheus as prom

logger = structlog.get_logger()

# Environment variables for default hook URLs
HOOK_ON_FILE_URL = os.getenv('RX_HOOK_ON_FILE_URL')
HOOK_ON_MATCH_URL = os.getenv('RX_HOOK_ON_MATCH_URL')
HOOK_ON_COMPLETE_URL = os.getenv('RX_HOOK_ON_COMPLETE_URL')

# Security: disable custom hooks from request parameters
DISABLE_CUSTOM_HOOKS = os.getenv('RX_DISABLE_CUSTOM_HOOKS', '').lower() in ('true', '1', 'yes')

# Hook timeout in seconds
HOOK_TIMEOUT_SECONDS = 3.0


class HookConfig(BaseModel):
    """Configuration for hook URLs with validation."""

    on_file_url: HttpUrl | str | None = None
    on_match_url: HttpUrl | str | None = None
    on_complete_url: HttpUrl | str | None = None

    def has_any_hook(self) -> bool:
        """Check if any hook is configured."""
        return bool(self.on_file_url or self.on_match_url or self.on_complete_url)

    def has_match_hook(self) -> bool:
        """Check if match hook is configured."""
        return bool(self.on_match_url)


def get_hook_env_config() -> dict:
    """Get hook configuration from environment variables for health endpoint."""
    return {
        'on_file_url': HOOK_ON_FILE_URL,
        'on_match_url': HOOK_ON_MATCH_URL,
        'on_complete_url': HOOK_ON_COMPLETE_URL,
        'custom_hooks_disabled': DISABLE_CUSTOM_HOOKS,
    }


def get_effective_hooks(
    custom_on_file: Optional[str] = None,
    custom_on_match: Optional[str] = None,
    custom_on_complete: Optional[str] = None,
) -> HookConfig:
    """
    Get effective hook URLs, considering env vars and custom hooks disabled setting.

    Priority (when custom hooks enabled):
    1. Custom URL from request parameter
    2. Default URL from environment variable

    When RX_DISABLE_CUSTOM_HOOKS=true, only environment variable URLs are used.

    Args:
        custom_on_file: Custom URL for on_file_scanned event
        custom_on_match: Custom URL for on_match_found event
        custom_on_complete: Custom URL for on_trace_complete event

    Returns:
        HookConfig with effective URLs
    """
    if DISABLE_CUSTOM_HOOKS:
        return HookConfig(
            on_file_url=HOOK_ON_FILE_URL,
            on_match_url=HOOK_ON_MATCH_URL,
            on_complete_url=HOOK_ON_COMPLETE_URL,
        )
    return HookConfig(
        on_file_url=custom_on_file or HOOK_ON_FILE_URL,
        on_match_url=custom_on_match or HOOK_ON_MATCH_URL,
        on_complete_url=custom_on_complete or HOOK_ON_COMPLETE_URL,
    )


def generate_request_id() -> str:
    """
    Generate a UUID v7 (time-sortable) request ID.

    UUID v7 includes a timestamp component, making IDs sortable by creation time.
    Format: xxxxxxxx-xxxx-7xxx-yxxx-xxxxxxxxxxxx

    Returns:
        String representation of UUID v7
    """
    # Python 3.13+ has uuid.uuid7(), for older versions we use uuid4
    if hasattr(uuid, 'uuid7'):
        return str(uuid.uuid7())
    else:
        # Fallback to uuid4 for Python < 3.13
        return str(uuid.uuid4())


def _call_hook_internal(url: str, params: dict, timeout: float = HOOK_TIMEOUT_SECONDS) -> tuple[bool, float]:
    """
    Internal synchronous hook call implementation.

    Args:
        url: The webhook URL to call
        params: Query parameters to send
        timeout: Request timeout in seconds

    Returns:
        Tuple of (success: bool, duration: float)
    """
    request_id = params.get('request_id', 'unknown')
    start_time = time()
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(url, params=params)
            duration = time() - start_time
            success = response.is_success
            if not success:
                logger.warning("hook_call_bad_status", url=url, status_code=response.status_code, request_id=request_id)
            return success, duration
    except httpx.TimeoutException:
        duration = time() - start_time
        logger.warning("hook_call_timeout", url=url, timeout_seconds=timeout, request_id=request_id)
        return False, duration
    except Exception as e:
        duration = time() - start_time
        logger.warning("hook_call_failed", url=url, error=str(e), request_id=request_id)
        return False, duration


def call_hook_sync(url: str, payload: dict, event_type: str) -> bool:
    """
    Call a hook URL synchronously (for CLI usage).

    Args:
        url: The webhook URL to call (GET request)
        payload: Dictionary of parameters to send as query params
        event_type: Event type for metrics ('on_file', 'on_match', 'on_complete')

    Returns:
        True if hook call succeeded, False otherwise
    """
    success, duration = _call_hook_internal(url, payload)
    prom.record_hook_call(event_type, success, duration)
    return success


async def call_hook_async(url: str, payload: dict, event_type: str) -> bool:
    """
    Call a hook URL asynchronously.

    Args:
        url: The webhook URL to call (GET request)
        payload: Dictionary of parameters to send as query params
        event_type: Event type for metrics ('on_file', 'on_match', 'on_complete')

    Returns:
        True if hook call succeeded, False otherwise
    """
    request_id = payload.get('request_id', 'unknown')
    start_time = time()
    try:
        async with httpx.AsyncClient(timeout=HOOK_TIMEOUT_SECONDS) as client:
            response = await client.get(url, params=payload)
            duration = time() - start_time
            success = response.is_success
            if not success:
                logger.warning("hook_call_bad_status", url=url, status_code=response.status_code, request_id=request_id)
            prom.record_hook_call(event_type, success, duration)
            return success
    except httpx.TimeoutException:
        duration = time() - start_time
        logger.warning("hook_call_timeout", url=url, timeout_seconds=HOOK_TIMEOUT_SECONDS, request_id=request_id)
        prom.record_hook_call(event_type, False, duration)
        return False
    except Exception as e:
        duration = time() - start_time
        logger.warning("hook_call_failed", url=url, error=str(e), request_id=request_id)
        prom.record_hook_call(event_type, False, duration)
        return False
