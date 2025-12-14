import logging
import sys


def setup_logger():
    """
    Configures the logger to write to standard output (stdout).
    This is essential because Docker captures stdout/stderr and redirects it to the log file.
    """
    logger = logging.getLogger("LegalTextDecoder")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if the logger is already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger