"""Unit tests for baize.core.logging.setup_logging()."""

import io
import json
from unittest.mock import patch

import structlog

from baize.core.logging import setup_logging


class TestSetupLogging:
    """Verify setup_logging() produces valid JSON structured logs."""

    def _capture_log(self, level: str, log_fn, *args, **kwargs) -> dict:
        """Call setup_logging, emit one log, return parsed JSON output."""
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            setup_logging(log_level=level)
            logger = structlog.get_logger("test")
            log_fn(logger, *args, **kwargs)
        output = buf.getvalue().strip()
        # Parse the first (and only) JSON line
        return json.loads(output)

    def test_output_is_valid_json(self):
        setup_logging()
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            setup_logging()
            structlog.get_logger("test").info("hello")
        raw = buf.getvalue().strip()
        # Should be parseable JSON
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)

    def test_json_contains_timestamp_field(self):
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            setup_logging()
            structlog.get_logger("test").info("ts_test")
        parsed = json.loads(buf.getvalue().strip())
        assert "timestamp" in parsed

    def test_timestamp_is_iso8601(self):
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            setup_logging()
            structlog.get_logger("test").info("iso_test")
        parsed = json.loads(buf.getvalue().strip())
        ts = parsed["timestamp"]
        # ISO 8601 UTC ends with 'Z' or '+00:00'
        assert "T" in ts, f"Expected ISO 8601 timestamp, got: {ts}"

    def test_json_contains_level_field(self):
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            setup_logging()
            structlog.get_logger("test").info("level_test")
        parsed = json.loads(buf.getvalue().strip())
        assert "level" in parsed

    def test_level_is_lowercase(self):
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            setup_logging()
            structlog.get_logger("test").info("lower_test")
        parsed = json.loads(buf.getvalue().strip())
        assert parsed["level"] == parsed["level"].lower(), f"level not lowercase: {parsed['level']}"

    def test_json_contains_event_field(self):
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            setup_logging()
            structlog.get_logger("test").info("my_event")
        parsed = json.loads(buf.getvalue().strip())
        assert "event" in parsed
        assert parsed["event"] == "my_event"

    def test_extra_kwargs_included_in_output(self):
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            setup_logging()
            structlog.get_logger("test").info("kv_test", foo="bar", count=42)
        parsed = json.loads(buf.getvalue().strip())
        assert parsed["foo"] == "bar"
        assert parsed["count"] == 42

    def test_debug_visible_when_log_level_debug(self):
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            setup_logging(log_level="DEBUG")
            structlog.get_logger("test").debug("debug_msg")
        raw = buf.getvalue().strip()
        assert raw, "Expected DEBUG log to appear when LOG_LEVEL=DEBUG"
        parsed = json.loads(raw)
        assert parsed["level"] == "debug"

    def test_debug_suppressed_when_log_level_info(self):
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            setup_logging(log_level="INFO")
            structlog.get_logger("test").debug("suppressed_debug")
        raw = buf.getvalue().strip()
        assert not raw, "Expected DEBUG log to be suppressed when LOG_LEVEL=INFO"

    def test_idempotent_multiple_calls_no_error(self):
        """Calling setup_logging() multiple times must not raise."""
        setup_logging()
        setup_logging()
        setup_logging(log_level="DEBUG")
