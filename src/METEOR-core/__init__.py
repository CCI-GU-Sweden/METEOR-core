try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from .metrology import metrology_splitter

__all__ = ["metrology_splitter"]
