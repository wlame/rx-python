"""Tests for rx.log — structlog configuration module."""

import json

import pytest
import structlog

from rx.log import configure_logging, get_logger, _apply_config
import logging


@pytest.fixture(autouse=True)
def restore_default_logging():
    """Restore structlog to WARNING-level stderr config after each test."""
    yield
    _apply_config(level=logging.WARNING, json_output=False)


class TestConfigureLogging:
    """Test configure_logging() sets up structlog correctly."""

    def test_configure_default(self):
        configure_logging()
        logger = structlog.get_logger()
        assert logger is not None

    def test_configure_debug_level(self):
        configure_logging(level='DEBUG')
        logger = structlog.get_logger()
        assert logger is not None

    def test_configure_json_output(self, capsys):
        configure_logging(level='INFO', json_output=True)
        logger = structlog.get_logger()
        logger.info('test_event', key='value')
        captured = capsys.readouterr()
        # JSON output goes to stderr
        parsed = json.loads(captured.err.strip())
        assert parsed['event'] == 'test_event'
        assert parsed['key'] == 'value'
        assert 'level' in parsed
        assert parsed['level'] == 'info'

    def test_configure_console_output(self, capsys):
        configure_logging(level='INFO', json_output=False)
        logger = structlog.get_logger()
        logger.info('console_test', count=42)
        captured = capsys.readouterr()
        assert 'console_test' in captured.err
        assert '42' in captured.err

    def test_level_filtering(self, capsys):
        configure_logging(level='WARNING')
        logger = structlog.get_logger()
        logger.info('should_not_appear')
        logger.warning('should_appear')
        captured = capsys.readouterr()
        assert 'should_not_appear' not in captured.err
        assert 'should_appear' in captured.err

    def test_case_insensitive_level(self):
        configure_logging(level='debug')
        logger = structlog.get_logger()
        assert logger is not None


class TestGetLogger:
    """Test get_logger() helper."""

    def test_returns_bound_logger(self):
        configure_logging()
        logger = get_logger()
        assert logger is not None

    def test_with_initial_values(self, capsys):
        configure_logging(level='INFO', json_output=True)
        logger = get_logger(component='test')
        logger.info('bound_test')
        captured = capsys.readouterr()
        parsed = json.loads(captured.err.strip())
        assert parsed['component'] == 'test'
        assert parsed['event'] == 'bound_test'

    def test_timestamps_present(self, capsys):
        configure_logging(level='INFO', json_output=True)
        logger = get_logger()
        logger.info('timestamp_test')
        captured = capsys.readouterr()
        parsed = json.loads(captured.err.strip())
        assert 'timestamp' in parsed
