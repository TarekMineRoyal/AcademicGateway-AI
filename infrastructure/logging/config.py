import json
import logging
import sys
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured observability logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Include custom dynamic fields attached to the log record
        standard_fields = {
            "args", "asctime", "created", "exc_info", "exc_text",
            "filename", "funcName", "levelname", "levelno",
            "lineno", "module", "msecs", "message", "msg",
            "name", "pathname", "process", "processName",
            "relativeCreated", "stack_info", "thread", "threadName"
        }
        for key, val in record.__dict__.items():
            if key not in standard_fields:
                log_data[key] = val

        return json.dumps(log_data)


def setup_logging(log_level: str = "INFO", log_format: str = "console") -> None:
    """Centralized function to initialize application-wide logging."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear pre-existing handlers to prevent duplicate entries
    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric_level)

    if log_format.lower() == "json":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)