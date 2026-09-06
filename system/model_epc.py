"""
Phonon system and model electron-phonon-couplings
By now, only Einstein phonon model is implemented, in which the phonon frequency is constant

The SSH-EPC is implemented for EPC model.
Only the hopping amplitude t_0 is coupled to the bond length and direction
"""

from dataclasses import dataclass
import numpy as np
from scipy.constants import atomic_mass, Boltzmann, electron_volt, hbar
from auxiliary import uniform_qmesh

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

    def meanSquareDisplacements(self, temperature, quantum=True):
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

    def sampleDisplacements(self, temperature, nSamples=None, seed=None, quantum=True):
        """Independent Gaussian snapshots; quantum=True samples the position marginal.

        Quantum zero-point snapshots are static disorder, not an inelastic quantum
        phonon bath. nSamples=None returns one (natom,3) configuration.
        """
        variance = self.meanSquareDisplacements(temperature, quantum=quantum)
        if nSamples is not None and (not isinstance(nSamples, (int, np.integer)) or nSamples < 1):
            raise ValueError("nSamples must be a positive integer or None")
        shape = variance.shape if nSamples is None else (nSamples,) + variance.shape
        return np.random.default_rng(seed).normal(size=shape) * np.sqrt(variance)

    def supercell(self, ncells):
        return EinsteinPhonons(np.tile(self.masses, ncells),
                               np.tile(self.energies, (ncells, 1)), self.directions)

    def sampleTrajectory(self, temperature, seed=None, statistics="classical"):
        """Stationary harmonic trajectory with sampled positions AND momenta.

        Time is in hbar/eV. Classical statistics obey equipartition; Wigner
        statistics reproduce the symmetrized quantum harmonic correlator,
        but neither gives the asymmetric quantum emission/absorption noise.
        The bath is prescribed: no electronic force or thermostat is applied.
        """
        if statistics not in ("classical", "wigner"):
            raise ValueError("trajectory statistics must be classical or wigner")
        variance = self.meanSquareDisplacements(temperature, quantum=statistics == "wigner")
        rng = np.random.default_rng(seed)
        quadratures = rng.normal(size=(2,) + variance.shape)*np.sqrt(variance)
        frequencies = np.zeros_like(variance)
        frequencies[:, self.directions] = self.energies
        return HarmonicTrajectory(quadratures[0], quadratures[1], frequencies)

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

class SSHModelEPC:
    """Scalar bond coupling with frozen equilibrium Rashba/exchange.

    alpha_R = d H(R)/d bond_length, in eV/Angstrom, duplicated in the two spin
    blocks. For H_ij=-t_ij and t=t0-alpha*stretch this is the usual positive alpha.
    The directions and lattice remain those of the equilibrium structure.
    An optional beta_R (eV/Angstrom^2) adds H = H0 + alpha*s + beta*s^2/2,
    where s = d_hat^0.(u_j-u_i). This is curvature in PROJECTED stretch,
    not the geometrical transverse curvature of an exact bond-length model.
    """

    def __init__(self, model, alpha_R, beta_R=None):
        if model.overlap:
            raise NotImplementedError("Model EPC currently requires an orthogonal basis")
        mapping = np.asarray(model.iorb2iatomMap)
        if (mapping.shape != (model.norbs,) or not np.issubdtype(mapping.dtype, np.integer)
                or np.any(mapping < 0) or np.any(mapping >= len(model.atomicOrbInfoList))):
            raise ValueError("System.iorb2iatomMap must contain one valid atom index per orbital")
        # Retain the equilibrium assignment alongside the other EPC references.
        self.orbital_atoms = mapping.copy()
        self.iRvec = model.iRvec.copy()
        self.lattice = model.lattice.copy()
        self.centers = model.rList.copy()
        self.alpha_R = np.array(alpha_R, dtype=float, copy=True)
        self.beta_R = (np.zeros_like(self.alpha_R) if beta_R is None
                       else np.array(beta_R, dtype=float, copy=True))
        if self.alpha_R.shape != model.ham_R.shape or not np.all(np.isfinite(self.alpha_R)):
            raise ValueError("SSH coefficients must match the Hamiltonian R grid")
        if self.beta_R.shape != model.ham_R.shape or not np.all(np.isfinite(self.beta_R)):
            raise ValueError("Quadratic SSH coefficients must match the Hamiltonian R grid")
        self.natoms = len(model.atomicOrbInfoList)
        lengths, self.unit_bonds = model.bondDirection(self.iRvec)
        if np.any((lengths <= 1e-12) & ((self.alpha_R != 0) | (self.beta_R != 0))):
            raise ValueError("SSH coefficients on zero-length bonds must be zero")

    def validate(self, model):
        if (model.ham_R.shape != self.alpha_R.shape
                or not np.array_equal(model.iRvec, self.iRvec)
                or not np.array_equal(model.iorb2iatomMap, self.orbital_atoms)
                or not np.allclose(model.lattice, self.lattice, rtol=0, atol=1e-10)
                or not np.allclose(model.rList, self.centers, rtol=0, atol=1e-10)):
            raise ValueError("SSH reference geometry/grid/atom mapping changed; attach EPC to the equilibrium model again")

    def deltaHamiltonian(self, model, displacements):
        """Delta H(R) for a real periodic displacement of this model's atoms."""
        self.validate(model)
        u = np.asarray(displacements)
        if u.shape != (self.natoms, 3) or np.iscomplexobj(u) or not np.all(np.isfinite(u)):
            raise ValueError(f"displacements must be a real finite ({self.natoms},3) array in Angstrom")
        orbital_u = u[self.orbital_atoms].T
        difference = orbital_u[None, :, None, :] - orbital_u[None, :, :, None]
        stretch = np.sum(self.unit_bonds * difference, axis=1)
        return self.alpha_R * stretch + 0.5*self.beta_R * stretch**2

    def secondDisplacementVertex(self, model, k, q, displacement):
        """Return H''(k;q,-q) in eV for a mode displacement in Angstrom.

        The second derivative is beta_R * |d_hat.(e_j exp(iq.R)-e_i)|^2.
        No 1/2 or thermal factor is included here. The Debye-Waller contribution
        is one half of this vertex times (2*N_qnu+1), averaged over q and modes.
        It is identically zero for the default strictly linear SSH model.
        """
        k = _wavevector(k, "k")
        u = np.asarray(displacement, dtype=complex)
        if u.shape != (self.natoms, 3) or not np.all(np.isfinite(u)):
            raise ValueError(f"displacement must have shape ({self.natoms},3) and finite entries")
        return self.secondDisplacementVertices(model, k[None], q, u[None])[0, 0]

    def secondDisplacementVertices(self, model, kpoints, q, displacements):
        """Batched ``H''(k;q,-q)`` for many k points and phonon modes.

        The result has shape ``(nk, nmodes, norbs, norbs)``.  Geometry and
        array consistency are checked once per batch instead of once per
        ``(k,q,mode)`` tuple.
        """
        self.validate(model)
        q = _wavevector(q, "q")
        kpoints = np.asarray(kpoints, dtype=float)
        if (kpoints.ndim != 2 or kpoints.shape[1] != 3
                or not len(kpoints) or not np.all(np.isfinite(kpoints))):
            raise ValueError("kpoints must have shape (nk,3) and finite entries")
        u = np.asarray(displacements, dtype=complex)
        if (u.ndim != 3 or u.shape[1:] != (self.natoms, 3)
                or not len(u) or not np.all(np.isfinite(u))):
            raise ValueError(f"displacements must have shape (nmodes,{self.natoms},3) and finite entries")
        orbital_u = u[:, self.orbital_atoms, :].transpose(0, 2, 1)
        phase_q = np.exp(2j*np.pi*(self.iRvec @ q))
        difference = (phase_q[None, :, None, None, None]
                      * orbital_u[:, None, :, None, :]
                      - orbital_u[:, None, :, :, None])
        stretch = np.sum(self.unit_bonds[None] * difference, axis=2)
        vertex_R = self.beta_R[None] * np.abs(stretch)**2
        phase_k = np.exp(2j*np.pi*(kpoints @ self.iRvec.T))
        return np.einsum("kr,vrij->kvij", phase_k, vertex_R, optimize=True)

    def displacementVertex(self, model, k, q, displacement):
        """<k+q|delta H|k> for u_La = displacement[a] exp(2 pi i q.L).

        displacement may be complex and has shape (natom,3), in Angstrom.
        The returned matrix is in the localized orbital basis, in eV. There is no
        1/sqrt(Ncells) factor. This vertex includes all finite-q bond phases.
        """
        k = _wavevector(k, "k")
        u = np.asarray(displacement, dtype=complex)
        if u.shape != (self.natoms, 3) or not np.all(np.isfinite(u)):
            raise ValueError(f"displacement must have shape ({self.natoms},3) and finite entries")
        return self.displacementVertices(model, k[None], q, u[None])[0, 0]

    def displacementVertices(self, model, kpoints, q, displacements):
        """Batched localized-orbital vertices for many k points and modes.

        Returns ``vertices[k, mode, orbital_i, orbital_j]`` in eV.
        """
        self.validate(model)
        q = _wavevector(q, "q")
        kpoints = np.asarray(kpoints, dtype=float)
        if (kpoints.ndim != 2 or kpoints.shape[1] != 3
                or not len(kpoints) or not np.all(np.isfinite(kpoints))):
            raise ValueError("kpoints must have shape (nk,3) and finite entries")
        u = np.asarray(displacements, dtype=complex)
        if (u.ndim != 3 or u.shape[1:] != (self.natoms, 3)
                or not len(u) or not np.all(np.isfinite(u))):
            raise ValueError(f"displacements must have shape (nmodes,{self.natoms},3) and finite entries")
        orbital_u = u[:, self.orbital_atoms, :].transpose(0, 2, 1)
        phase_q = np.exp(2j*np.pi*(self.iRvec @ q))
        difference = (phase_q[None, :, None, None, None]
                      * orbital_u[:, None, :, None, :]
                      - orbital_u[:, None, :, :, None])
        vertex_R = self.alpha_R[None] * np.sum(
            self.unit_bonds[None] * difference, axis=2)
        phase_k = np.exp(2j*np.pi*(kpoints @ self.iRvec.T))
        return np.einsum("kr,vrij->kvij", phase_k, vertex_R, optimize=True)

    def matrixElements(self, model, k, q, bandBasis=True, vectors_k=None, vectors_kq=None):
        """Complex g[nu,m,n](k,q), in eV; final m at k+q, initial n at k.

        H_ep = Ncells^(-1/2) sum g c^dagger_(k+q) c_k (b_q + b^dagger_(-q)).
        Thermal Bose factors belong in the collision kernel, not in g.
        """
        k, q = _wavevector(k, "k"), _wavevector(q, "q")
        phonons = getattr(model, "modelPhonons", None)
        if phonons is None or len(phonons.masses) != self.natoms:
            raise ValueError("Call setModelPhonons with one mass per atom before computing g")
        if not bandBasis:
            if vectors_k is not None or vectors_kq is not None:
                raise ValueError("Eigenvectors require bandBasis=True")
            return self.displacementVertices(
                model, k[None], q, phonons.modeDisplacements(q))[0]
        if vectors_k is None or vectors_kq is None:
            from calculator.Path import EnergyOnPath
            points = [point for point, vectors in ((k, vectors_k), (k+q, vectors_kq)) if vectors is None]
            states = iter(EnergyOnPath(model=model, kpath=np.array(points), syncModelVerbose=False).vectors)
            if vectors_k is None:
                vectors_k = next(states)
            if vectors_kq is None:
                vectors_kq = next(states)
        vectors_k, vectors_kq = np.asarray(vectors_k), np.asarray(vectors_kq)
        expected = (model.norbs, model.norbs)
        if vectors_k.shape != expected or vectors_kq.shape != expected:
            raise ValueError(f"Eigenvector matrices must have shape {expected}, with eigenvectors in columns")
        return self.matrixElementsBatch(
            model, k[None], q, vectors_k[None], vectors_kq[None])[0]

    def matrixElementsBatch(self, model, kpoints, q, vectors_k, vectors_kq):
        """Band-basis ``g`` for a batch of k points at one q point.

        ``vectors_k`` and ``vectors_kq`` contain eigenvectors in columns and
        both have shape ``(nk,norbs,norbs)``.  The returned axes are
        ``(k, mode, final_band, initial_band)``.
        """
        phonons = getattr(model, "modelPhonons", None)
        if phonons is None or len(phonons.masses) != self.natoms:
            raise ValueError("Call setModelPhonons with one mass per atom before computing g")
        kpoints = np.asarray(kpoints, dtype=float)
        vectors_k, vectors_kq = np.asarray(vectors_k), np.asarray(vectors_kq)
        expected = (len(kpoints), model.norbs, model.norbs)
        if vectors_k.shape != expected or vectors_kq.shape != expected:
            raise ValueError(f"batched eigenvector matrices must have shape {expected}")
        vertices = self.displacementVertices(
            model, kpoints, q, phonons.modeDisplacements(q))
        return (vectors_kq.conj().swapaxes(-2, -1)[:, None]
                @ vertices @ vectors_k[:, None])

class SSHTimeOperators:
    """H[u(t)], dH/dA and d2H/dA2 with equilibrium point-orbital centers.

    The fixed orbital basis defines precisely the same SSH hopping-only EPC
    as the primitive-cell perturbation calculation. Translating orbital
    centers would require extra moving-basis and convective-current terms.
    """
    def __init__(self, owner):
        model = owner.model
        if not isinstance(model.modelEPC, SSHModelEPC) or not isinstance(model.modelPhonons, EinsteinPhonons):
            raise NotImplementedError("Phonon TDSE currently requires SSHModelEPC and EinsteinPhonons")
        epc = model.modelEPC
        epc.validate(model)
        expected = np.zeros_like(model.AA_R_W)
        indices = np.arange(model.norbs)
        expected[model.iRvecIndex[(0, 0, 0)]][:, indices, indices] = model.rList
        if not np.allclose(model.AA_R_W, expected, atol=1e-10, rtol=0):
            raise NotImplementedError("Phonon TDSE requires diagonal point-orbital positions")
        active = (model.ham_R != 0) | (epc.alpha_R != 0) | (epc.beta_R != 0)
        ir, i, j = np.nonzero(active)
        self.i, self.j = i, j
        self.initial_atoms, self.final_atoms = epc.orbital_atoms[i], epc.orbital_atoms[j]
        self.h0, self.alpha, self.beta = model.ham_R[active], epc.alpha_R[active], epc.beta_R[active]
        self.unit_bonds = epc.unit_bonds[ir, :, i, j]
        self.bonds = (model.iRvec[ir] @ model.lattice + model.rList[:, j].T-model.rList[:, i].T)
        kpoints = uniform_qmesh(owner.kgrid) + owner.kOffset
        self.phase = np.exp(2j*np.pi*(kpoints @ model.iRvec[ir].T))
        self.nk, self.nb = len(kpoints), model.norbs

    def at_displacement(self, u):
        stretch = np.sum(self.unit_bonds*(u[self.final_atoms]-u[self.initial_atoms]), axis=1)
        hopping = self.phase*(self.h0 + self.alpha*stretch + .5*self.beta*stretch**2)[None]
        h = np.zeros((self.nk, self.nb, self.nb), complex)
        v = np.zeros((self.nk, 3, self.nb, self.nb), complex)
        w = np.zeros((self.nk, 3, 3, self.nb, self.nb), complex)
        # Different R bonds can share an orbital pair at a supercell Bloch k.
        for k in range(self.nk):
            np.add.at(h[k], (self.i, self.j), hopping[k])
            for a in range(3):
                np.add.at(v[k, a], (self.i, self.j), 1j*self.bonds[:, a]*hopping[k])
                for b in range(3):
                    np.add.at(w[k, b, a], (self.i, self.j), -self.bonds[:, b]*self.bonds[:, a]*hopping[k])
        return h, v, w