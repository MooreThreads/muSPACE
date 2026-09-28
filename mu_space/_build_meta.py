"""Stable interface for build-generated package metadata."""

try:
    from ._version import __git_version__, __version__
except ModuleNotFoundError:
    __version__ = "0.0.0+unknown"
    __git_version__ = "unknown"
