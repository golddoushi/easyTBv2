"""Electron-phonon coupling between independent physical models."""

from .base import EPCModel
from .ssh import SSHModelEPC, SSHTimeOperators

__all__ = ["EPCModel", "SSHModelEPC", "SSHTimeOperators"]
