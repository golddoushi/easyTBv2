"""Independent phonon models and prescribed displacement trajectories."""

from dataclasses import dataclass
import numpy as np
from scipy.constants import atomic_mass, Boltzmann, electron_volt, hbar

def _wavevector(value, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain three finite fractional coordinates")
    return value

class PhononModeSelection:
    """Phonon quadrature: fractional q points, branches and weights per q.

    This selector does not define frequencies, polarizations or a dispersion
    relation. The same branch list is applied at every selected q. Physical
    displacement construction and sampling belong to the phonon model.

    Conjugate partners are inserted with equal weight; an explicitly supplied
    partner is counted once. Weights are never normalized. ``branches=None``
    means all branches, with the same q weight for each branch. Branch labels
    at -q must correspond to conjugate polarizations at q; the selector
    does not infer this correspondence from eigenvectors.
    """

    def __init__(self, qpoints, branches=None, weights=None):
        points = np.atleast_2d(np.asarray(qpoints, float))
        if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
            raise ValueError('qpoints must be a nonempty finite (nq,3) array')
        values = np.ones(len(points)) if weights is None else np.asarray(weights, float)
        if values.shape != (len(points),) or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError('one positive finite weight is required per input q point')
        if branches is not None:
            b = np.asarray(branches)
            if b.ndim != 1 or not len(b) or not np.isfinite(b).all() or np.any(b < 0) or np.any(b != b.astype(int)):
                raise ValueError('branches must be nonnegative integer indices')
            if len(np.unique(b)) != len(b):
                raise ValueError('branches must be distinct')
            branches = tuple(sorted(b.astype(int).tolist()))
        self.branches = branches
        entries = {}
        for q, weight in zip(points, values):
            for sign in (1, -1):
                key = tuple(np.round(np.mod(sign*q, 1), 14) % 1)
                if key in entries and not np.isclose(entries[key], weight, rtol=1e-12, atol=0):
                    raise ValueError('duplicate/conjugate q points must have equal weights')
                entries[key] = float(weight)
        keys = sorted(entries)
        self.qpoints = np.array(keys)
        self.weights = np.array([entries[key] for key in keys])
        self.qpoints.setflags(write=False)
        self.weights.setflags(write=False)

    @property
    def signature(self):
        return (self.qpoints.tobytes(), self.weights.tobytes(), self.branches)

    def branch_indices(self, nmodes):
        indices = np.arange(nmodes) if self.branches is None else np.array(self.branches, int)
        if np.any(indices >= nmodes):
            raise ValueError(f'phonon branch index exceeds available range 0..{nmodes-1}')
        return indices

    def mesh_shifts(self, grid):
        shifts = self.qpoints*np.asarray(grid)
        if not np.allclose(shifts, np.rint(shifts), rtol=0, atol=1e-9):
            raise ValueError('selected q points must map the electronic k grid onto itself')
        return np.rint(shifts).astype(int)

    def check_supercell(self, matrix):
        phases = self.qpoints@np.asarray(matrix).T
        if not np.allclose(phases, np.rint(phases), rtol=0, atol=1e-9):
            raise ValueError('selected q points are not commensurate with the supercell')

    def as_dict(self):
        return dict(qpoints=self.qpoints.tolist(), weights=self.weights.tolist(),
                    branches=None if self.branches is None else list(self.branches),
                    total_q_weight=float(self.weights.sum()), weight_normalization='none')


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

    def _selectionBasis(self):
        """Einstein-mode covariance factors B, with C = B.T B for unit local variance.

        Branches are local atom/Cartesian oscillators, not general phonon
        polarizations. Cell-major atom ordering matches System.make_supercell.
        A generic pair has independent cosine/sine coordinates; a self-inverse
        point has one.
        Multiplying B by local thermal RMS gives physical displacement factors.
        """
        selection = self.modeSelection
        cells, natoms, directions = self._cell_origins, self._primitive_natoms, self.directions
        branches = selection.branch_indices(natoms*len(directions))
        factors, visited = [], set()
        for q, weight in zip(selection.qpoints, selection.weights):
            key = tuple(q)
            if key in visited:
                continue
            opposite = tuple(np.round(np.mod(-q, 1), 14) % 1)
            visited.update((key, opposite))
            phase = 2*np.pi*cells@q
            patterns = ([np.sqrt(weight)*np.cos(phase)] if key == opposite else
                        [np.sqrt(2*weight)*np.cos(phase), np.sqrt(2*weight)*np.sin(phase)])
            for branch in branches:
                atom, axis = divmod(branch, len(directions))
                for pattern in patterns:
                    factor = np.zeros((len(cells), natoms, 3))
                    factor[:, atom, directions[axis]] = pattern
                    factors.append(factor.reshape(-1, 3))
        return np.array(factors)

    def displacementFactors(self, temperature, quantum=True):
        """Gaussian covariance factors, including correlations of selected modes."""
        variance = self._bareMeanSquareDisplacements(temperature, quantum)
        if self.modeSelection is None:
            active = np.flatnonzero(variance.ravel())
            factors = np.zeros((len(active), variance.size))
            factors[np.arange(len(active)), active] = np.sqrt(variance.ravel()[active])
            return factors.reshape((-1,)+variance.shape)
        self.modeSelection.check_supercell(self._supercell_matrix)
        return self._selectionBasis()*np.sqrt(variance)[None]

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
        """Filter independent local Gaussian fields, scaling by sqrt(N*w_q).

        Supports arbitrary leading sample/time-Fourier axes. Since Einstein
        energies are cell independent, projection commutes with propagation.
        """
        if self.modeSelection is None:
            return values
        selection = self.modeSelection
        selection.check_supercell(self._supercell_matrix)
        cells, natoms, directions = self._cell_origins, self._primitive_natoms, self.directions
        values = np.asarray(values)
        ncells = len(cells)
        fields = values.reshape(values.shape[:-2]+(ncells, natoms, 3))
        result = np.zeros_like(fields, dtype=complex)
        mask = np.zeros((natoms, 3))
        for branch in selection.branch_indices(natoms*len(directions)):
            atom, axis = divmod(branch, len(directions))
            mask[atom, directions[axis]] = 1
        for q, weight in zip(selection.qpoints, selection.weights):
            phase = np.exp(2j*np.pi*np.asarray(cells)@q)
            coefficient = np.einsum('l,...lac->...ac', phase.conj(), fields)/ncells
            result += np.sqrt(ncells*weight)*np.einsum('l,...ac->...lac', phase, coefficient*mask)
        if np.max(np.abs(result.imag), initial=0) > 1e-9*max(1., np.max(np.abs(result.real), initial=0)):
            raise ValueError('selected displacement failed the realness check')
        return result.real.reshape(values.shape)

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

