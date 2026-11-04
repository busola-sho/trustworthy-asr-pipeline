"""Small process-environment helpers for model loading and plotting libraries."""

from contextlib import contextmanager
import os


@contextmanager
def temporary_environment(settings):
    """Restore the caller's environment even when a stage fails."""
    previous = {name: os.environ.get(name) for name in settings}
    os.environ.update(settings)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
