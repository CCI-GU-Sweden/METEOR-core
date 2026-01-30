try:
    from importlib.metadata import version

    __version__ = version("METEOR-core")
except ImportError:
    __version__ = "unknown"

__all__ = []
