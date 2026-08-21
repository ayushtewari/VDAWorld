"""VDAWorld package."""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("vdaworld")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"
