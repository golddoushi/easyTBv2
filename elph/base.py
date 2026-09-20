"""Physical electron-phonon coupling, without response or collision state."""

from abc import ABC, abstractmethod
import numpy as np
from phonon import Phonon


class EPCModel(ABC):
    """Coupling between explicitly supplied electronic and phonon models.

    Phonon atom order and fractional-q lattice must match the electronic
    system. This correspondence is supplied by the caller; atom counts alone
    cannot establish it. Occupations, broadening and collision approximations
    belong to calculators.
    """

    def __init__(self, electronic_model, phonon_model):
        if not isinstance(phonon_model, Phonon):
            raise TypeError("phonon_model must implement Phonon")
        if len(phonon_model.masses) != len(electronic_model.atomicOrbInfoList):
            raise ValueError("Phonon and electronic atom counts differ")
        self.electronic_model = electronic_model
        self.phonon_model = phonon_model

    @abstractmethod
    def validate(self):
        """Check the concrete coupling's structural requirements."""

    @abstractmethod
    def displacementVertices(self, kpoints, q, displacements):
        """Return <k+q|delta H|k>, axes (k,mode,orbital,orbital), in eV.

        Displacements have axes (mode,atom,xyz), in Angstrom, and carry
        exp(2*pi*i*q.L). Neither thermal nor 1/sqrt(Ncells) factors enter.
        """

    @property
    @abstractmethod
    def has_second_order(self):
        """Whether this model includes a nonzero quadratic displacement term."""

    @abstractmethod
    def secondDisplacementVertices(self, kpoints, q, displacements):
        """Return H''(k;q,-q), axes (k,mode,orbital,orbital), in eV.

        Include neither the Taylor factor 1/2 nor thermal factors. Called
        by self-energy calculations only when has_second_order is true.
        """

    def matrixElementsBatch(self, kpoints, q, vectors_k, vectors_kq):
        """Return g[k,mode,final_band,initial_band] in eV.

        Electronic eigenvectors are columns of (nk,norbs,norbs) arrays.
        Sampling and diagonalizing the electronic system belong to callers.
        """
        kpoints = np.asarray(kpoints, dtype=float)
        vectors_k, vectors_kq = np.asarray(vectors_k), np.asarray(vectors_kq)
        expected = (len(kpoints), self.electronic_model.norbs, self.electronic_model.norbs)
        if vectors_k.shape != expected or vectors_kq.shape != expected:
            raise ValueError(f"batched eigenvector matrices must have shape {expected}")
        vertices = self.displacementVertices(
            kpoints, q, self.phonon_model.modeDisplacements(q))
        return (vectors_kq.conj().swapaxes(-2, -1)[:, None]
                @ vertices @ vectors_k[:, None])
