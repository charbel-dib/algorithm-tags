"""Logging setup shared by the command line and the library.

Every command writes to the console and to ``logs/algo_tags.log`` (rotating file), with
timestamps, so that a run can be audited afterwards: which data, which model, how long each
step took and how fast predictions were.
"""

import logging
import time
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOGGER_NAME = "algo_tags"
FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def get_logger(name: str = "") -> logging.Logger:
    """Return the package logger, or one of its children (``get_logger("model")``)."""
    return logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)


def setup_logging(log_file: Path | None = Path("logs/algo_tags.log"), level: int = logging.INFO) -> None:
    """Send package logs to the console and, unless ``log_file`` is None, to a rotating file.

    If the file cannot be created (read-only folder), a warning is logged and the console is kept,
    so that a logging problem never stops a prediction.
    """
    logger = get_logger()
    logger.setLevel(level)
    for handler in logger.handlers:  # a second call (tests, notebooks) must not leak open files
        handler.close()
    logger.handlers.clear()
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
    logger.addHandler(console)
    logger.propagate = False
    if log_file is not None:
        try:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            handler = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        except OSError as error:  # read-only folder, container without volume: keep the console logs
            logger.warning("cannot write the log file %s (%s), logging to the console only", log_file, error)
            return
        handler.setFormatter(logging.Formatter(FORMAT))
        logger.addHandler(handler)


@contextmanager
def timed(logger: logging.Logger, step: str):
    """Log how long the wrapped block took: ``with timed(log, "fit"): ...``.

    A block that raises is logged as failed, with its duration, and the error is passed on.
    """
    start = time.perf_counter()
    try:
        yield
    except Exception:
        logger.error("%s failed after %.2f s", step, time.perf_counter() - start)
        raise
    logger.info("%s done in %.2f s", step, time.perf_counter() - start)
