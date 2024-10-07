import contextlib
import logging


def get_logger(log_context: str) -> logging.Logger:
    """Get a logger for a context. Avoid making it too granular!"""
    if log_context is None:
        log_context = "otclim"
    else:
        log_context = f"otclim.{log_context}"

    logger = logging.getLogger(log_context)
    return logger


@contextlib.contextmanager
def warnings_to_logger(logger: logging.Logger):
    """Redirect warnings to the logger.

    See Also
    --------
    https://docs.python.org/3/library/warnings.html#warnings.catch_warnings

    """
    import warnings

    def warning_to_logger(message, category, filename, lineno, file=None, line=None):
        """Redirect warnings to logger with standard formatting."""
        warn_str = warnings.formatwarning(message, category, filename, lineno, line)
        logger.warning(warn_str.strip())

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = warning_to_logger
        yield
