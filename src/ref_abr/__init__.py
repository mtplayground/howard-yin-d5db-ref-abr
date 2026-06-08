"""Python package for the reference ABR experiment toolkit."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("howard-yin-d5db-ref-abr")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__"]
