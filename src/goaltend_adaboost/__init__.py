"""Goaltend vs legal: accelerometer ingestion, fusion features, AdaBoost CV pipeline."""

__version__ = "0.1.0"
__all__ = ["run", "__version__"]


def __getattr__(name: str):
    if name == "run":
        from .model import run as _run

        return _run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
