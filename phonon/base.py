"""Atom-resolved harmonic modes, independent of the electronic system."""

from abc import ABC, abstractmethod
import numpy as np
from scipy.constants import atomic_mass, electron_volt, hbar


def _wavevector(value, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain three finite fractional coordinates")
    return value


class Phonon(ABC):
    """A phonon model with one mass (amu) per atom in a defined atom order.

    Atom indices must match the electronic atoms when constructing an EPC
    model. Fractional q uses that system's reciprocal lattice. Polarizations
    are dimensionless eigenvectors in mass-weighted coordinates, normalized
    by sum_atom,xyz |e|**2 = 1. The physical displacement uses
    e[atom,xyz]/sqrt(mass[atom]) and exp(2*pi*i*q.L); basis-position phases
    belong in e. There is no 1/sqrt(Ncells) factor in this interface.

    Sampling trajectories and constructing supercells are model-specific
    capabilities, not requirements of this mode interface.
    """

    masses: np.ndarray

    def __init__(self, masses):
        self.masses = np.atleast_1d(np.asarray(masses, dtype=float)).copy()
        if (self.masses.ndim != 1 or not len(self.masses)
                or np.any(self.masses <= 0) or not np.all(np.isfinite(self.masses))):
            raise ValueError("masses must be a positive finite array in amu")

    @abstractmethod
    def modes(self, q=(0, 0, 0)):
        """Return energies (nmodes,) in eV and e (nmodes,natoms,3)."""

    def activeModeMask(self, q=(0, 0, 0)):
        """Keep branch labels fixed; providers may explicitly mask translations.

        The default retains the historical strictly-positive mode contract.
        A force-constant provider can override this with a documented acoustic
        policy. An inactive mode is never assigned an artificial frequency.
        """
        energies, _ = self.modes(q)
        energies = np.asarray(energies)
        if np.any(energies <= 0) or not np.all(np.isfinite(energies)):
            raise ValueError('EPC requires positive finite active mode energies; zero acoustic modes need an explicit limiting prescription')
        return np.ones(energies.shape, dtype=bool)

    def modeDisplacements(self, q=(0, 0, 0)):
        """Zero-point displacement amplitudes in Angstrom, without Bose factors.

        Zero/unstable modes have no finite harmonic zero-point normalization;
        handling them requires a separate prescription in the phonon model.
        """
        energies, vectors = self.modes(q)
        energies = np.asarray(energies)
        if np.any(energies <= 0) or not np.all(np.isfinite(energies)):
            raise ValueError("Zero-point displacements require positive finite phonon energies")
        lengths = np.sqrt(hbar**2 / (2 * self.masses[None, :, None] * atomic_mass
                                    * energies[:, None, None] * electron_volt)) * 1e10
        return vectors * lengths


def active_mode_indices(phonons, q, selection=None):
    """Intersect a selection with an explicit mask without renumbering branches."""
    energies, _ = phonons.modes(q)
    energies = np.asarray(energies)
    mask = np.asarray(phonons.activeModeMask(q))
    if energies.ndim != 1 or mask.dtype != np.dtype(bool) or mask.shape != energies.shape:
        raise ValueError('Phonon active mask must be boolean with one entry per branch')
    if not np.all(np.isfinite(energies)) or np.any(energies[mask] <= 0):
        raise ValueError('EPC requires positive finite active mode energies')
    branches = np.arange(len(energies)) if selection is None else selection.branch_indices(len(energies))
    return branches[mask[branches]]
