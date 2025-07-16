import logging
from contextlib import contextmanager
from io import StringIO
import sys


@contextmanager
def capture_logs(logger_name=None):
    logger = logging.getLogger(logger_name)
    # Save original handlers
    original_handlers = logger.handlers.copy()
    # Create capture handler
    log_capture = StringIO()
    capture_handler = logging.StreamHandler(log_capture)
    logger.handlers = [capture_handler]

    try:
        yield log_capture
    finally:
        # Restore original handlers
        logger.handlers = original_handlers

@contextmanager
def capture_stdout(logger_name=None):
    # Create string buffer to capture output
    output = StringIO()
    # Store original stdout
    original_stdout = sys.stdout
    # Redirect stdout to our buffer
    sys.stdout = output

    try:
        yield output
    finally:
        # Restore original stdout
        sys.stdout = original_stdout