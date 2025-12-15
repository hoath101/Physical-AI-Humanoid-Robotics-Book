import logging
from datetime import datetime
import json
from typing import Dict, Any

class CustomFormatter(logging.Formatter):
    """
    Custom formatter to output logs in JSON format with additional context.
    """
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # Add any extra fields that were passed to the log record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info']:
                log_entry[key] = value

        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_entry)

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with custom formatting.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent adding multiple handlers if logger already exists
    if logger.handlers:
        return logger

    # Create handler
    handler = logging.StreamHandler()

    # Create formatter and add it to the handler
    formatter = CustomFormatter()
    handler.setFormatter(formatter)

    # Add handler to the logger
    logger.addHandler(handler)

    return logger

# Create a global logger instance
app_logger = setup_logger("rag_chatbot_api", logging.INFO)

def log_api_call(endpoint: str, method: str, duration: float, user_id: str = None, book_id: str = None):
    """
    Log an API call with performance metrics.
    """
    app_logger.info(
        f"API call completed: {method} {endpoint}",
        extra={
            "endpoint": endpoint,
            "method": method,
            "duration_ms": round(duration * 1000, 2),
            "user_id": user_id,
            "book_id": book_id
        }
    )

def log_error(error_msg: str, error_type: str, endpoint: str = None, user_id: str = None):
    """
    Log an error with context.
    """
    app_logger.error(
        error_msg,
        extra={
            "error_type": error_type,
            "endpoint": endpoint,
            "user_id": user_id
        }
    )

def log_performance(metric_name: str, value: float, unit: str = "ms", context: Dict[str, Any] = None):
    """
    Log performance metrics.
    """
    log_data = {
        "metric_name": metric_name,
        "value": value,
        "unit": unit
    }

    if context:
        log_data.update(context)

    app_logger.info(f"Performance metric: {metric_name} = {value} {unit}", extra=log_data)