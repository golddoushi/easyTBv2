"""Electron-phonon coupling between explicit electronic and phonon models."""

import copy
import numpy as np
from phonon.base import _wavevector
from .base import EPCModel

class SSHModelEPC(EPCModel):
    """Scalar bond coupling with frozen equilibrium Rashba/exchange.

    electronic_model and phonon_model are explicit, independent physical
    models. This object holds their coupling; neither model owns the other.
    Electronic response state and collision settings are supplied separately.

    alpha_R = d H(R)/d bond_length, in eV/Angstrom, duplicated in the two spin
    blocks. For H_ij=-t_ij and t=t0-alpha*stretch this is the usual positive alpha.
    The directions and lattice remain those of the equilibrium structure.
    An optional beta_R (eV/Angstrom^2) adds H = H0 + alpha*s + beta*s^2/2,
    where s = d_hat^0.(u_j-u_i). This is curvature in PROJECTED stretch,
    not the geometrical transverse curvature of an exact bond-length model.
    """

    def __init__(self, electronic_model, phonon_model, alpha_R, beta_R=None):
        super().__init__(electronic_model, phonon_model)
        model = electronic_model
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

    @property
    def has_second_order(self):
        return bool(np.any(self.beta_R != 0))

    def validate(self):
        model = self.electronic_model
        if (model.ham_R.shape != self.alpha_R.shape
                or not np.array_equal(model.iRvec, self.iRvec)
                or not np.array_equal(model.iorb2iatomMap, self.orbital_atoms)
                or not np.allclose(model.lattice, self.lattice, rtol=0, atol=1e-10)
                or not np.allclose(model.rList, self.centers, rtol=0, atol=1e-10)):
            raise ValueError("SSH reference geometry/grid/atom mapping changed; construct EPC from the equilibrium models again")

    def deltaHamiltonian(self, displacements):
        """Delta H(R) for a real periodic displacement of this model's atoms."""
        self.validate()
        u = np.asarray(displacements)
        if u.shape != (self.natoms, 3) or np.iscomplexobj(u) or not np.all(np.isfinite(u)):
            raise ValueError(f"displacements must be a real finite ({self.natoms},3) array in Angstrom")
        orbital_u = u[self.orbital_atoms].T
        difference = orbital_u[None, :, None, :] - orbital_u[None, :, :, None]
        stretch = np.sum(self.unit_bonds * difference, axis=1)
        return self.alpha_R * stretch + 0.5*self.beta_R * stretch**2

    def secondDisplacementVertex(self, k, q, displacement):
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
        return self.secondDisplacementVertices(k[None], q, u[None])[0, 0]

    def secondDisplacementVertices(self, kpoints, q, displacements):
        """Batched ``H''(k;q,-q)`` for many k points and phonon modes.

        The result has shape ``(nk, nmodes, norbs, norbs)``.  Geometry and
        array consistency are checked once per batch instead of once per
        ``(k,q,mode)`` tuple.
        """
        self.validate()
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

    def displacementVertex(self, k, q, displacement):
        """<k+q|delta H|k> for u_La = displacement[a] exp(2 pi i q.L).

        displacement may be complex and has shape (natom,3), in Angstrom.
        The returned matrix is in the localized orbital basis, in eV. There is no
        1/sqrt(Ncells) factor. This vertex includes all finite-q bond phases.
        """
        k = _wavevector(k, "k")
        u = np.asarray(displacement, dtype=complex)
        if u.shape != (self.natoms, 3) or not np.all(np.isfinite(u)):
            raise ValueError(f"displacement must have shape ({self.natoms},3) and finite entries")
        return self.displacementVertices(k[None], q, u[None])[0, 0]

    def displacementVertices(self, kpoints, q, displacements):
        """Batched localized-orbital vertices for many k points and modes.

        Returns ``vertices[k, mode, orbital_i, orbital_j]`` in eV.
        """
        self.validate()
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

    def matrixElements(self, k, q, bandBasis=True, vectors_k=None, vectors_kq=None):
        """Complex g[nu,m,n](k,q), in eV; final m at k+q, initial n at k.

        H_ep = Ncells^(-1/2) sum g c^dagger_(k+q) c_k (b_q + b^dagger_(-q)).
        Thermal Bose factors belong in the collision kernel, not in g.
        """
        model = self.electronic_model
        k, q = _wavevector(k, "k"), _wavevector(q, "q")
        phonons = self.phonon_model
        if phonons is None or len(phonons.masses) != self.natoms:
            raise ValueError("The phonon model must have one mass per electronic atom")
        if not bandBasis:
            if vectors_k is not None or vectors_kq is not None:
                raise ValueError("Eigenvectors require bandBasis=True")
            return self.displacementVertices(
                k[None], q, phonons.modeDisplacements(q))[0]
        if vectors_k is None or vectors_kq is None:
            points = [point for point, vectors in ((k, vectors_k), (k+q, vectors_kq)) if vectors is None]
            phase = np.exp(2j*np.pi*(np.asarray(points) @ model.iRvec.T))
            hamiltonians = np.einsum("kr,rij->kij", phase, model.ham_R)
            states = iter(np.linalg.eigh(hamiltonians)[1])
            if vectors_k is None:
                vectors_k = next(states)
            if vectors_kq is None:
                vectors_kq = next(states)
        vectors_k, vectors_kq = np.asarray(vectors_k), np.asarray(vectors_kq)
        expected = (model.norbs, model.norbs)
        if vectors_k.shape != expected or vectors_kq.shape != expected:
            raise ValueError(f"Eigenvector matrices must have shape {expected}, with eigenvectors in columns")
        return self.matrixElementsBatch(
            k[None], q, vectors_k[None], vectors_kq[None])[0]

    @classmethod
    def from_files(cls, electronic_model, phonon_model, alphaR="epc", betaR=None):
        """Read SSH derivatives, optionally including quadratic bond curvature.

        ``alpha_R`` is dH_ij/d bond_length in eV/Angstrom. Thus for the usual
        H_ij=-t_ij and t_ij=t0_ij-alpha_ij*stretch, the file stores +alpha_ij.
        Delta H_ij = alpha_ij d_hat^0_ij.(u_j-u_i), using equilibrium unit
        bond vectors and Cartesian displacements. The file layout matches
        addRashbaSOC, with spatial indices and real reciprocal coefficients.
        All R points must exist in iRvec, and nonzero coefficients must act on
        existing Hamiltonian bonds. Atoms are assigned through iorb2iatomMap;
        orbital positions come from rList, with any number of s orbitals per atom.
        Rashba and exchange are held fixed when atoms are displaced.
        betaR optionally names a second scalar file in eV/Angstrom^2, defining
        Delta H = alpha*s + beta*s^2/2 with s=d_hat^0.(u_j-u_i). betaR=None
        retains the original linear model and has exactly zero Debye-Waller
        vertex. beta is independent information; it cannot be inferred from g.
        """
        model = electronic_model
        iRvec, strengths, _ = model.read_scalar_bonds(alphaR)
        quadratic = None if betaR is None else model.read_scalar_bonds(betaR)
        n = model.norbs // 2
        alpha = np.zeros_like(model.ham_R.real)
        beta = np.zeros_like(alpha)
        coupling_sets = [(iRvec, strengths)]
        if quadratic is not None:
            coupling_sets.append(quadratic[:2])
        for vectors, coefficients in coupling_sets:
            for R, value in zip(vectors, coefficients):
                block = model.ham_R[model.iRvecIndex[tuple(R)]].reshape(2, n, 2, n)
                existing = np.any(block != 0, axis=(0, 2))
                if np.any((value != 0) & ~existing):
                    raise ValueError(f"EPC coefficients at R={tuple(R)} must couple existing Hamiltonian bonds")
        for R, value in zip(iRvec, strengths):
            ir = model.iRvecIndex[tuple(R)]
            alpha[ir, :n, :n] = value
            alpha[ir, n:, n:] = value
        if quadratic is not None:
            for R, value in zip(quadratic[0], quadratic[1]):
                ir = model.iRvecIndex[tuple(R)]
                beta[ir, :n, :n] = beta[ir, n:, n:] = value
        return cls(model, phonon_model, alpha, beta)

    def makeDisplacedModel(self, displacements):
        """Return an independent frozen snapshot of an equilibrium SSH model.

        Build the desired supercell FIRST, then provide one real displacement
        (Angstrom) per supercell atom. Scalar hopping and orbital centers move;
        Rashba/exchange matrices remain fixed. The equilibrium SSH reference is not
        updated or accumulated. The returned snapshot is an electronic model; sample each new
        snapshot from this equilibrium EPC model.
        """
        model = self.electronic_model
        delta = self.deltaHamiltonian(displacements)
        orbital_u = np.asarray(displacements)[model.iorb2iatomMap].T
        # A point-orbital position operator is needed to move centers consistently.
        iR0 = model.iRvecIndex.get((0, 0, 0))
        indices = np.arange(model.norbs)
        if iR0 is None or model.AA_R_W is None:
            raise ValueError("Displaced model requires an onsite point-orbital position operator")
        expected = np.zeros_like(model.AA_R_W)
        expected[iR0][:, indices, indices] = model.rList
        if not np.allclose(model.AA_R_W, expected, atol=1e-10, rtol=0):
            raise NotImplementedError("Displacements currently require a diagonal point-orbital position operator")
        result = copy.deepcopy(model)
        result.ham_R = result.ham_R + delta
        if result.nonexchange_R is not None:
            result.nonexchange_R = result.nonexchange_R + delta
        result.rList = result.rList + orbital_u
        result.posinfo['pos ions (cart.)'] = np.asarray(result.posinfo['pos ions (cart.)']) + displacements
        result.posinfo['pos ions (frac.)'] = result.posinfo['pos ions (cart.)'] @ np.linalg.inv(result.lattice)
        result.AA_R_W[iR0][:, indices, indices] += orbital_u
        result._refreshHamiltonianDerivedMatrices()
        return result

    def supercell(self, matrix, omitRHam=False):
        """Return an independent EPC model with both physical models expanded.

        Coupling derivatives use exactly the electronic orbital/cell mapping.
        The primitive models and this coupling are left unchanged.
        """
        self.validate()
        electronic = copy.deepcopy(self.electronic_model)
        arrays, origins = electronic.make_supercell(
            matrix, omitRHam=omitRHam,
            additional_operators={"alpha": self.alpha_R, "beta": self.beta_R})
        phonons = self.phonon_model.supercell(len(origins), origins, np.asarray(matrix, int))
        return type(self)(electronic, phonons, arrays["alpha"], arrays["beta"])

class SSHTimeOperators:
    """H[u(t)], dH/dA and d2H/dA2 with equilibrium point-orbital centers.

    The fixed orbital basis defines precisely the same SSH hopping-only EPC
    as the primitive-cell perturbation calculation. Translating orbital
    centers would require extra moving-basis and convective-current terms.
    kpoints is an explicit (nk,3) array in fractional reciprocal coordinates.
    No phonon sampling or response-calculator state is read here.
    """
    def __init__(self, epc_model, kpoints):
        model = epc_model.electronic_model
        epc = epc_model
        if not isinstance(epc, SSHModelEPC):
            raise TypeError("SSHTimeOperators requires SSHModelEPC")
        epc.validate()
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
        kpoints = np.asarray(kpoints, dtype=float)
        if kpoints.ndim != 2 or kpoints.shape[1] != 3 or not np.isfinite(kpoints).all():
            raise ValueError("kpoints must have shape (nk,3) and finite entries")
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

    def displacement_coefficients(self, u, kslice=slice(None), moving_centers=False):
        """Exact coefficients through x^2 for H(xu), v(xu), w(xu).

        Optical vertices are differentiated along with the SSH Hamiltonian.
        Moving centers implement the static makeDisplacedModel convention;
        fixed centers implement the hopping-only trajectory convention.
        The quadratic coefficient includes the 1/2 multiplying beta.
        """
        u = np.asarray(u, float)
        difference = u[self.final_atoms]-u[self.initial_atoms]
        stretch = np.sum(self.unit_bonds*difference, axis=1)
        dr = difference if moving_centers else np.zeros_like(difference)
        hs = (self.h0, self.alpha*stretch, .5*self.beta*stretch**2)
        phase = self.phase[kslice]
        nk = len(phase)
        h = np.zeros((3, nk, self.nb, self.nb), complex)
        v = np.zeros((3, nk, 3, self.nb, self.nb), complex)
        w = np.zeros((3, nk, 3, 3, self.nb, self.nb), complex)
        for order in range(3):
            hopping = phase*hs[order]
            previous = phase*hs[order-1] if order >= 1 else np.zeros_like(hopping)
            previous2 = phase*hs[order-2] if order >= 2 else np.zeros_like(hopping)
            for k in range(nk):
                np.add.at(h[order, k], (self.i, self.j), hopping[k])
            for a in range(3):
                velocity = 1j*(self.bonds[:, a]*hopping+dr[:, a]*previous)
                for k in range(nk):
                    np.add.at(v[order, k, a], (self.i, self.j), velocity[k])
                for b in range(3):
                    contact = -(self.bonds[:, a]*self.bonds[:, b]*hopping
                        +(dr[:, a]*self.bonds[:, b]+self.bonds[:, a]*dr[:, b])*previous
                        +dr[:, a]*dr[:, b]*previous2)
                    for k in range(nk):
                        np.add.at(w[order, k, b, a], (self.i, self.j), contact[k])
        return h, v, w
