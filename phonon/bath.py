"""Calculation-specific phonon occupation and sampling, separate from a model."""

from dataclasses import dataclass
import numpy as np
from .selection import PhononModeSelection


@dataclass(frozen=True)
class PhononBath:
    """Shared bath for collision, self-energy, snapshots and trajectories.

    Temperature is in kelvin. Selection weights retain their absolute values;
    a selection supersedes the auxiliary uniform qgrid. Quantum trajectories
    are not implemented: use classical or Wigner statistics for real motion.
    """

    temperature: float = 300.
    statistics: str = 'quantum'
    selection: PhononModeSelection | None = None
    qgrid: tuple | None = None

    def __post_init__(self):
        if not np.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError('Phonon temperature must be finite and nonnegative, in kelvin')
        if self.statistics not in ('quantum', 'classical', 'wigner'):
            raise ValueError('Phonon statistics must be quantum, classical or wigner')
        if self.selection is not None and not isinstance(self.selection, PhononModeSelection):
            raise TypeError('selection must be PhononModeSelection')
        if self.qgrid is not None:
            grid = np.asarray(self.qgrid)
            if grid.shape != (3,) or not np.isfinite(grid).all() or np.any(grid < 1) or np.any(grid != grid.astype(int)):
                raise ValueError('qgrid must contain three positive integers')
            object.__setattr__(self, 'qgrid', tuple(grid.astype(int)))

    def displacementFactors(self, phonons):
        return phonons.displacementFactors(self.temperature,
            quantum=self.statistics != 'classical', selection=self.selection)

    def meanSquareDisplacements(self, phonons):
        return phonons.meanSquareDisplacements(self.temperature,
            quantum=self.statistics != 'classical', selection=self.selection)

    def sampleDisplacements(self, phonons, nSamples=None, seed=None):
        return phonons.sampleDisplacements(self.temperature, nSamples=nSamples,
            seed=seed, quantum=self.statistics != 'classical', selection=self.selection)

    def sampleTrajectory(self, phonons, seed=None):
        return phonons.sampleTrajectory(self.temperature, seed=seed,
            statistics=self.statistics, selection=self.selection)

    def sampleDampedTrajectory(self, phonons, damping, period, phase_points, seed=None):
        return phonons.sampleDampedTrajectory(self.temperature, damping, period,
            phase_points, seed=seed, statistics=self.statistics, selection=self.selection)
