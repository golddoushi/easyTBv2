"""Prescribed atom-resolved displacement trajectories."""

from dataclasses import dataclass
import numpy as np

@dataclass
class HarmonicTrajectory:
    """u(t)=cosine*cos(energy*t)+sine*sin(energy*t), in Angstrom.

    ``sine*energies`` is the initial velocity in Angstrom/(hbar/eV).
    """
    cosine: np.ndarray
    sine: np.ndarray
    energies: np.ndarray

    def displacements(self, time):
        phase = self.energies*float(time)
        return self.cosine*np.cos(phase) + self.sine*np.sin(phase)

    def velocities(self, time):
        phase = self.energies*float(time)
        return self.energies*(-self.cosine*np.sin(phase) + self.sine*np.cos(phase))

@dataclass
class DampedPeriodicTrajectory:
    """Stationary Gaussian damped-oscillator path represented by a Fourier series."""
    cosine: np.ndarray
    sine: np.ndarray
    base_energy: float
    energies: np.ndarray
    damping: float
    period: float
    spectrum_weights: np.ndarray

    def displacements(self, time):
        harmonics = np.arange(len(self.cosine))*self.base_energy
        phase = harmonics*float(time)
        return np.einsum('n,nac->ac', np.cos(phase), self.cosine)+np.einsum(
            'n,nac->ac', np.sin(phase), self.sine)

    def velocities(self, time):
        harmonics = np.arange(len(self.cosine))*self.base_energy
        phase = harmonics*float(time)
        return np.einsum('n,nac->ac', -harmonics*np.sin(phase), self.cosine)+np.einsum(
            'n,nac->ac', harmonics*np.cos(phase), self.sine)

