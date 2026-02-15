import logging
import os
import sys
from contextlib import contextmanager
from io import StringIO


def load_nvidia_api_key() -> str:
    """Load NVIDIA API key from environment or file.

    Checks in order:
    1. NVIDIA_API_KEY environment variable
    2. NVIDIA_API_KEY_FILE environment variable (path to key file)
    3. ~/.nvidia_api_key
    4. ~/.config/nvrx/nvidia_api_key

    Returns:
        API key string, or empty string if not found.
    """
    # Check direct env var first
    api_key = os.getenv("NVIDIA_API_KEY")
    if api_key:
        return api_key.strip()

    # Check file path from env var
    key_file = os.getenv("NVIDIA_API_KEY_FILE")
    if key_file and os.path.isfile(key_file):
        with open(key_file) as f:
            return f.read().strip()

    # Check common file locations
    home = os.path.expanduser("~")
    common_locations = [
        os.path.join(home, ".nvidia_api_key"),
        os.path.join(home, ".config", "nvrx", "nvidia_api_key"),
    ]
    for path in common_locations:
        if os.path.isfile(path):
            with open(path) as f:
                return f.read().strip()

    return ""


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
