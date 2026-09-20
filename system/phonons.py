"""Independent phonon models and prescribed displacement trajectories."""

from dataclasses import dataclass
import numpy as np
from scipy.constants import atomic_mass, Boltzmann, electron_volt, hbar

def _wavevector(value, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain three finite fractional coordinates")
    return value

@dataclass
class EinsteinPhonons:
    """Independent pinned ionic oscillators, with a flat branch for each DOF.

    Degenerate branches use real atom/Cartesian unit polarizations. The mode
    energies are independent of q, but the physical displacement still carries
    exp(2 pi i q.L). No electronic backreaction or acoustic sum rule is imposed
    on this prescribed bath. The SSH interaction itself is translation invariant.
    """

    masses: np.ndarray
    energies: np.ndarray
    directions: tuple = (0, 1, 2)
    modeSelection: object = None

    def __post_init__(self):
        self.masses = np.atleast_1d(np.asarray(self.masses, dtype=float)).copy()
        self.directions = tuple(self.directions)
        if (self.masses.ndim != 1 or not len(self.masses)
                or np.any(self.masses <= 0) or not np.all(np.isfinite(self.masses))):
            raise ValueError("masses must be a positive finite array in amu")
        if (not self.directions or len(set(self.directions)) != len(self.directions)
                or any(d not in (0, 1, 2) for d in self.directions)):
            raise ValueError("directions must be distinct Cartesian indices from (0,1,2)")
        values = np.asarray(self.energies, dtype=float)
        if values.shape == (len(self.masses),):
            values = values[:, None]
        self.energies = np.broadcast_to(values, (len(self.masses), len(self.directions))).copy()
        if np.any(self.energies <= 0) or not np.all(np.isfinite(self.energies)):
            raise ValueError("phonon energies hbar*Omega must be positive and finite, in eV")
        self._primitive_natoms = len(self.masses)
        self._cell_origins = np.zeros((1, 3), int)
        self._supercell_matrix = np.eye(3, dtype=int)

    def modes(self, q=(0, 0, 0)):
        """Return (energies[nu], polarizations[nu,atom,xyz]); no 1/sqrt(N)."""
        _wavevector(q, "q")
        vectors = np.zeros((self.energies.size, len(self.masses), 3))
        for nu, (atom, direction) in enumerate(
                (a, d) for a in range(len(self.masses)) for d in self.directions):
            vectors[nu, atom, direction] = 1.0
        return self.energies.ravel().copy(), vectors

    def modeDisplacements(self, q=(0, 0, 0)):
        """Zero-point length times polarization, in Angstrom (not thermal RMS)."""
        energies, vectors = self.modes(q)
        lengths = np.sqrt(hbar**2 / (2 * self.masses[None, :, None] * atomic_mass
                                    * energies[:, None, None] * electron_volt)) * 1e10
        return vectors * lengths

    def _bareMeanSquareDisplacements(self, temperature, quantum=True):
        """Return <u_atom,xyz^2> in Angstrom^2; inactive directions are zero."""
        temperature = float(temperature)
        if not np.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative, in kelvin")
        zero_point = (hbar**2 * 1e20 / (2 * self.masses[:, None] * atomic_mass
                                      * self.energies * electron_volt))
        kBT = Boltzmann * temperature / electron_volt
        if quantum:
            factor = 1.0 if temperature == 0 else 1 / np.tanh(self.energies / (2*kBT))
        else:
            factor = 2*kBT / self.energies
        result = np.zeros((len(self.masses), 3))
        result[:, self.directions] = zero_point * factor
        return result

    def displacementFactors(self, temperature, quantum=True):
        """Gaussian covariance factors, including correlations of selected modes."""
        variance = self._bareMeanSquareDisplacements(temperature, quantum)
        if self.modeSelection is None:
            active = np.flatnonzero(variance.ravel())
            factors = np.zeros((len(active), variance.size))
            factors[np.arange(len(active)), active] = np.sqrt(variance.ravel()[active])
            return factors.reshape((-1,)+variance.shape)
        self.modeSelection.check_supercell(self._supercell_matrix)
        return self.modeSelection.real_basis(self._cell_origins, self._primitive_natoms,
                                              self.directions)*np.sqrt(variance)[None]

    def meanSquareDisplacements(self, temperature, quantum=True):
        if self.modeSelection is None:
            return self._bareMeanSquareDisplacements(temperature, quantum)
        # The diagonal covariance is translation invariant, including on the
        # primitive model where the selected finite-q field cannot be sampled.
        variance = self._bareMeanSquareDisplacements(temperature, quantum)
        mask = np.zeros((self._primitive_natoms, 3))
        for branch in self.modeSelection.branch_indices(self._primitive_natoms*len(self.directions)):
            atom, axis = divmod(branch, len(self.directions))
            mask[atom, self.directions[axis]] = self.modeSelection.weights.sum()
        return variance*np.tile(mask, (len(self._cell_origins), 1))

    def _projectSelection(self, values):
        if self.modeSelection is None:
            return values
        self.modeSelection.check_supercell(self._supercell_matrix)
        return self.modeSelection.project(values, self._cell_origins,
                                          self._primitive_natoms, self.directions)

    def sampleDisplacements(self, temperature, nSamples=None, seed=None, quantum=True):
        """Independent Gaussian snapshots; quantum=True samples the position marginal.

        Quantum zero-point snapshots are static disorder, not an inelastic quantum
        phonon bath. nSamples=None returns one (natom,3) configuration.
        """
        variance = self._bareMeanSquareDisplacements(temperature, quantum=quantum)
        if nSamples is not None and (not isinstance(nSamples, (int, np.integer)) or nSamples < 1):
            raise ValueError("nSamples must be a positive integer or None")
        shape = variance.shape if nSamples is None else (nSamples,) + variance.shape
        return self._projectSelection(np.random.default_rng(seed).normal(size=shape) * np.sqrt(variance))

    def supercell(self, ncells, internal_cells=None, matrix=None):
        result = EinsteinPhonons(np.tile(self.masses, ncells),
                                 np.tile(self.energies, (ncells, 1)), self.directions,
                                 self.modeSelection)
        if self.modeSelection is not None:
            if internal_cells is None or matrix is None:
                raise ValueError('selected phonons require supercell origins and matrix')
            if len(self._cell_origins) != 1:
                raise NotImplementedError('build a selected-mode supercell directly from the primitive model')
            self.modeSelection.check_supercell(matrix)
            result._primitive_natoms = self._primitive_natoms
            result._cell_origins = np.asarray(internal_cells, int).copy()
            result._supercell_matrix = np.asarray(matrix, int).copy()
        return result

    def sampleTrajectory(self, temperature, seed=None, statistics="classical"):
        """Stationary harmonic trajectory with sampled positions AND momenta.

        Time is in hbar/eV. Classical statistics obey equipartition; Wigner
        statistics reproduce the symmetrized quantum harmonic correlator,
        but neither gives the asymmetric quantum emission/absorption noise.
        The bath is prescribed: no electronic force or thermostat is applied.
        """
        if statistics not in ("classical", "wigner"):
            raise ValueError("trajectory statistics must be classical or wigner")
        variance = self._bareMeanSquareDisplacements(temperature, quantum=statistics == "wigner")
        rng = np.random.default_rng(seed)
        quadratures = rng.normal(size=(2,) + variance.shape)*np.sqrt(variance)
        quadratures = self._projectSelection(quadratures)
        frequencies = np.zeros_like(variance)
        frequencies[:, self.directions] = self.energies
        return HarmonicTrajectory(quadratures[0], quadratures[1], frequencies)

    def sampleDampedTrajectory(self, temperature, damping, period, phase_points,
                                seed=None, statistics="classical"):
        """Sample a stationary periodic approximation to a damped Langevin bath.

        ``damping`` is the amplitude-decay rate in eV for
        u''+2*damping*u'+omega**2*u=noise.  The Fourier coefficients sample the
        corresponding classical displacement spectrum and are normalized to
        the requested classical or Wigner equal-time variance.  The Wigner
        option remains symmetrized noise, not a quantum detailed-balance bath.
        """
        if statistics not in ("classical", "wigner"):
            raise ValueError("trajectory statistics must be classical or wigner")
        damping, period = float(damping), float(period)
        if not np.isfinite(damping) or damping <= 0:
            raise ValueError("damping must be positive and finite, in eV")
        if not np.isfinite(period) or period <= 0:
            raise ValueError("period must be positive and finite, in hbar/eV")
        if (not isinstance(phase_points, (int, np.integer)) or phase_points < 5
                or phase_points % 2 == 0):
            raise ValueError("phase_points must be an odd integer >= 5")

        base_energy = 2*np.pi/period
        harmonics = np.arange((phase_points-1)//2+1, dtype=float)
        sampled_energies = harmonics[:, None, None]*base_energy
        mode_energies = self.energies[None]
        denominator = ((mode_energies**2-sampled_energies**2)**2
                       +(2*damping*sampled_energies)**2)
        weights = 1/denominator
        weights[1:] *= 2
        weights /= np.sum(weights, axis=0, keepdims=True)

        variance = self._bareMeanSquareDisplacements(temperature, quantum=statistics == "wigner")
        rng = np.random.default_rng(seed)
        shape = (len(harmonics), len(self.masses), 3)
        cosine, sine = np.zeros(shape), np.zeros(shape)
        active_variance = variance[:, self.directions]
        coefficient_scale = np.sqrt(weights*active_variance[None])
        cosine[:, :, self.directions] = rng.normal(size=weights.shape)*coefficient_scale
        sine[1:, :, self.directions] = rng.normal(size=weights[1:].shape)*coefficient_scale[1:]
        cosine, sine = self._projectSelection(cosine), self._projectSelection(sine)
        mode_energy_grid = np.zeros_like(variance)
        mode_energy_grid[:, self.directions] = self.energies
        return DampedPeriodicTrajectory(cosine, sine, base_energy, mode_energy_grid,
                                        damping, period, weights)

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

