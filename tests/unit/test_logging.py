import json
import logging
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.middleware.logging_middleware import LoggingMiddleware
from infrastructure.logging.config import JSONFormatter, setup_logging


def test_json_formatter_produces_valid_json():
    """Verifies that JSONFormatter formats LogRecords into valid JSON with expected fields."""
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname="test.py",
        lineno=42,
        msg="Test log message",
        args=(),
        exc_info=None,
    )

    formatted_output = formatter.format(record)
    parsed = json.loads(formatted_output)

    assert parsed["message"] == "Test log message"
    assert parsed["level"] == "INFO"
    # Flexibly check for either 'logger' or 'name' depending on the formatter schema
    assert parsed.get("logger") == "test_logger" or parsed.get("name") == "test_logger"
    assert "timestamp" in parsed


def test_setup_logging_console_mode():
    """Verifies that setup_logging configures standard console formatting."""
    setup_logging(log_level="DEBUG", log_format="console")
    root_logger = logging.getLogger()

    assert root_logger.level == logging.DEBUG
    assert len(root_logger.handlers) > 0
    assert not isinstance(root_logger.handlers[0].formatter, JSONFormatter)


def test_setup_logging_json_mode():
    """Verifies that setup_logging configures JSONFormatter when specified."""
    setup_logging(log_level="WARNING", log_format="json")
    root_logger = logging.getLogger()

    assert root_logger.level == logging.WARNING
    assert len(root_logger.handlers) > 0
    assert isinstance(root_logger.handlers[0].formatter, JSONFormatter)


@pytest.fixture
def test_app() -> FastAPI:
    """Creates a minimal FastAPI app wrapped with LoggingMiddleware for isolation testing."""
    app = FastAPI()
    app.add_middleware(LoggingMiddleware)

    @app.get("/test-endpoint")
    def sample_endpoint():
        return {"status": "ok"}

    return app


def test_logging_middleware_generates_request_id(test_app: FastAPI):
    """Verifies that LoggingMiddleware injects an X-Request-ID header when none is provided."""
    client = TestClient(test_app)
    response = client.get("/test-endpoint")

    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    assert len(response.headers["X-Request-ID"]) > 0


def test_logging_middleware_preserves_incoming_request_id(test_app: FastAPI):
    """Verifies that LoggingMiddleware preserves and propagates existing X-Request-ID headers."""
    client = TestClient(test_app)
    custom_id = "custom-trace-id-12345"
    response = client.get("/test-endpoint", headers={"X-Request-ID": custom_id})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == custom_id