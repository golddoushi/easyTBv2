"""Lowest-order retarded Fan-Migdal and Debye-Waller self-energies for model EPC.

Conventions follow https://vasp.at/wiki/Electron-phonon_interactions_theory:
phonon energies are hbar*Omega in eV, and retarded denominators have +i*eta.
The static comparison is Fan(adiabatic) + DW, not DW alone. No rigid-ion
reconstruction of the second derivative is assumed for a tight-binding model.
"""

import numpy as np
from scipy.constants import Boltzmann, electron_volt
from scipy.special import expit

from .Grid import Energy
from .Path import EnergyOnPath


def uniform_qmesh(grid):
    """Full Gamma-centered mesh in fractional reciprocal coordinates."""
    values = np.asarray(grid)
    if (values.shape != (3,) or not np.all(np.isfinite(values))
            or np.any(values < 1) or not np.all(values == values.astype(int))):
        raise ValueError("qgrid must contain three positive integers")
    grid = values.astype(int)
    return np.indices(tuple(grid)).reshape(3, -1).T / grid


def _points(value, name):
    points = np.atleast_2d(np.asarray(value, dtype=float))
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must have shape (npoints,3), with finite fractional coordinates")
    return points

class ModelElectronSelfEnergy:
    """One-shot model self-energy, evaluated immediately on the supplied grids.

    Parameters
    ----------
    model : System
        An equilibrium model with modelEPC and modelPhonons attached.
    kpoints : array (nk,3), or one 3-vector
        Fractional reciprocal coordinates of the external electronic states.
    qgrid : tuple, optional
        Full mesh, default (8,8,1). Mutually exclusive with qpoints.
        Electronic states are computed by Energy once per distinct shifted
        mesh, then reused for all equivalent k and k+q points.
    qpoints, qweights : arrays, optional
        Explicit phonon quadrature; weights must be nonnegative and are
        normalized to sum to one. Uniform weights are used if omitted.
        For explicit quadrature, EnergyOnPath solves the unique electronic
        points together. The underlying solvers are in energy_calculators.
    temperature : float
        Common electron/phonon temperature in kelvin, including exactly zero.
    fermi : float
        Chemical potential in eV. It sets intermediate-state Fermi factors.
    delta : float
        Positive retarded denominator regulator in eV, not a fitted lifetime.
    adiabatic : bool
        False retains +/- phonon energies and Pauli factors. True neglects
        phonon energies only in denominators; Bose factors and zero-point
        displacement normalization are retained. Use True for frozen snapshots.
    energyList : 1D array, optional
        Common real evaluation energies in eV. None evaluates each band's
        on-shell energy. Full matrices are stored even for on-shell evaluation,
        so a degenerate subspace can be treated at a common energy explicitly.

    Outputs
    -------
    fan_migdal_matrix : complex (nk,ne,nb,nb)
        Full band-space retarded Fan matrix at each evaluation energy.
    debye_waller_matrix : complex Hermitian (nk,nb,nb)
        Frequency-independent thermal average of the quadratic Hamiltonian.
    fan_migdal, total : complex (nk,nb) or (nk,ne,nb)
        Diagonal self-energies; the first shape applies to on-shell evaluation.
    debye_waller : real (nk,nb)
        Diagonal DW energy shift, identically zero for strictly linear SSH.

    On-shell energy shifts use Z=1 and neglect off-diagonal mixing. They are
    reliable for isolated nondegenerate bands at weak coupling. For degenerate
    manifolds use total_matrix at a common energy and diagonalize the block.
    Static Im(Sigma) depends on delta and is not a quantum EPC transport rate.
    """

    def __init__(self, model, kpoints, qgrid=None, qpoints=None, qweights=None,
                 temperature=0.0, fermi=0.0, delta=1e-3, adiabatic=False, energyList=None):
        if model.modelEPC is None or model.modelPhonons is None:
            raise ValueError("Attach model EPC and model phonons before computing self-energy")
        model.modelEPC.validate(model)
        if len(model.modelPhonons.masses) != model.modelEPC.natoms:
            raise ValueError("Phonon and EPC atom counts differ")
        self.kpoints = _points(kpoints, "kpoints")
        if qpoints is not None and qgrid is not None:
            raise ValueError("Specify qgrid or qpoints, not both")
        self.qpoints = (uniform_qmesh((8, 8, 1) if qgrid is None else qgrid)
                        if qpoints is None else _points(qpoints, "qpoints"))
        self.qgrid = (tuple(int(n) for n in ((8, 8, 1) if qgrid is None else qgrid))
                      if qpoints is None else None)
        weights = np.ones(len(self.qpoints)) if qweights is None else np.asarray(qweights, dtype=float)
        if (weights.shape != (len(self.qpoints),) or not np.all(np.isfinite(weights))
                or np.any(weights < 0) or weights.sum() <= 0):
            raise ValueError("qweights must be finite, nonnegative, and have positive sum")
        self.qweights = weights / weights.sum()
        self.temperature, self.fermi, self.delta = float(temperature), float(fermi), float(delta)
        self.adiabatic = bool(adiabatic)
        self.on_shell = energyList is None
        self.model = model
        self._prepare_electronic_states()
        if self.on_shell:
            self.evaluation_energies = self.energies.copy()
        else:
            energyList = np.asarray(energyList, dtype=float)
            if energyList.ndim != 1 or not len(energyList) or not np.all(np.isfinite(energyList)):
                raise ValueError("energyList must be a nonempty finite 1D array in eV")
            self.evaluation_energies = np.broadcast_to(energyList, (len(self.kpoints), len(energyList))).copy()
        nk, ne = self.evaluation_energies.shape
        nb = model.norbs
        self.fan_migdal_matrix = np.zeros((nk, ne, nb, nb), dtype=complex)
        self.debye_waller_matrix = np.zeros((nk, nb, nb), dtype=complex)
        kBT = Boltzmann * self.temperature / electron_volt
        epc, phonons = model.modelEPC, model.modelPhonons
        has_quadratic = np.any(epc.beta_R != 0)

        # q is the outer loop so all electronic k points share one vectorized
        # EPC construction and one set of medium-sized contractions.  This is
        # much faster than dispatching tiny NumPy operations for every k/q/mode.
        for iq, (q, weight) in enumerate(zip(self.qpoints, self.qweights)):
            if weight == 0:
                continue
            intermediate = self.intermediate_energies[:, iq]
            vectors_kq = self.intermediate_vectors[:, iq]
            phonon_energies, _ = phonons.modes(q)
            if self.temperature == 0:
                bose = np.zeros_like(phonon_energies)
                occupation = np.where(intermediate < self.fermi, 1.0,
                                      np.where(intermediate > self.fermi, 0.0, 0.5))
            else:
                # exp(-x)/(-expm1(-x)) avoids overflow for cold/high-energy modes.
                x = phonon_energies / kBT
                bose = np.exp(-x) / (-np.expm1(-x))
                occupation = expit((self.fermi-intermediate) / kBT)
            g = model.getModelEPCBatch(
                self.kpoints, q, self.vectors, vectors_kq)
            denominator = (self.evaluation_energies[:, :, None, None]
                           - intermediate[:, None, None, :] + 1j*self.delta)
            if self.adiabatic:
                propagator = (2*bose[None, None, :, None]+1) / denominator
            else:
                propagator = ((bose[None, None, :, None]+1
                               - occupation[:, None, None, :])
                              / (denominator-phonon_energies[None, None, :, None])
                              + (bose[None, None, :, None]
                                 + occupation[:, None, None, :])
                              / (denominator+phonon_energies[None, None, :, None]))
            # g[k,nu,m,n] maps external n to intermediate m; retain complex
            # products for off-diagonal self-energies, not just |g|^2.
            self.fan_migdal_matrix += weight*np.einsum(
                "kvmi,kavm,kvmj->kaij", g.conj(), propagator, g, optimize=True)
            if has_quadratic:
                displacements = phonons.modeDisplacements(q)
                if hasattr(epc, "secondDisplacementVertices"):
                    hessian = epc.secondDisplacementVertices(
                        model, self.kpoints, q, displacements)
                else:
                    hessian = np.array([
                        [epc.secondDisplacementVertex(model, k, q, u)
                         for u in displacements]
                        for k in self.kpoints])
                hessian_band = (self.vectors.conj().swapaxes(-2, -1)[:, None]
                                @ hessian @ self.vectors[:, None])
                self.debye_waller_matrix += 0.5*weight*np.einsum(
                    "v,kvij->kij", 2*bose+1, hessian_band, optimize=True)

    def _prepare_electronic_states(self):
        """Index established calculator results; no separate eigensolver.

        Cell-periodic eigenstates are periodic under reciprocal lattice shifts.
        On a uniform q mesh, external k points differing by a mesh vector share
        one Energy FFT. Explicit q quadratures use a single batched path solve.
        """
        nk, nq, nb = len(self.kpoints), len(self.qpoints), self.model.norbs
        if self.qgrid is not None:
            grid = np.array(self.qgrid)
            # Rounding is only used to identify equivalent mesh offsets. The
            # FFT phase is evaluated at the original representative k point.
            offsets = np.round(np.mod(self.kpoints*grid, 1), 12) % 1
            _, representatives, self._calculator_index = np.unique(
                offsets, axis=0, return_index=True, return_inverse=True)
            solvers = []
            self._external_indices = np.empty(nk, dtype=int)
            self._intermediate_indices = np.empty((nk, nq), dtype=int)
            qindices = np.rint(self.qpoints*grid).astype(int)
            for group, representative in enumerate(representatives):
                shift = self.kpoints[representative]
                solvers.append(Energy(
                    model=self.model, kgrid=self.qgrid, kOffset=shift))
                members = np.flatnonzero(self._calculator_index == group)
                indices = np.rint((self.kpoints[members]-shift)*grid).astype(int) % grid
                self._external_indices[members] = np.ravel_multi_index(indices.T, self.qgrid)
                intermediate = (indices[:, None, :] + qindices[None, :, :]) % grid
                self._intermediate_indices[members] = np.ravel_multi_index(
                    intermediate.reshape(-1, 3).T, self.qgrid).reshape(len(members), nq)
            self.energy_calculators = tuple(solvers)
        else:
            points = np.concatenate((self.kpoints,
                                     (self.kpoints[:, None, :] + self.qpoints).reshape(-1, 3)))
            reduced = np.mod(points, 1)
            _, representatives, inverse = np.unique(
                np.round(reduced, 12) % 1, axis=0, return_index=True, return_inverse=True)
            self.energy_calculators = (EnergyOnPath(
                model=self.model, kpath=reduced[representatives], syncModelVerbose=False),)
            self._calculator_index = np.zeros(nk, dtype=int)
            self._external_indices = inverse[:nk]
            self._intermediate_indices = inverse[nk:].reshape(nk, nq)
        self.energies = np.array([
            self.energy_calculators[group].energies.reshape(-1, nb)[index]
            for group, index in zip(self._calculator_index, self._external_indices)])
        self.vectors = np.array([
            self.energy_calculators[group].vectors.reshape(-1, nb, nb)[index]
            for group, index in zip(self._calculator_index, self._external_indices)])
        # Materialize k+q states in contiguous arrays once.  Downstream Fan
        # contractions can then batch the entire external k mesh for each q.
        self.intermediate_energies = np.empty((nk, nq, nb), dtype=float)
        self.intermediate_vectors = np.empty((nk, nq, nb, nb), dtype=complex)
        for group, states in enumerate(self.energy_calculators):
            members = np.flatnonzero(self._calculator_index == group)
            if not len(members):
                continue
            energies = states.energies.reshape(-1, nb)
            vectors = states.vectors.reshape(-1, nb, nb)
            indices = self._intermediate_indices[members]
            self.intermediate_energies[members] = energies[indices]
            self.intermediate_vectors[members] = vectors[indices]

    @property
    def fan_migdal(self):
        diagonal = np.diagonal(self.fan_migdal_matrix, axis1=-2, axis2=-1)
        return np.diagonal(diagonal, axis1=1, axis2=2) if self.on_shell else diagonal

    @property
    def debye_waller(self):
        return np.diagonal(self.debye_waller_matrix, axis1=-2, axis2=-1).real

    @property
    def total_matrix(self):
        return self.fan_migdal_matrix + self.debye_waller_matrix[:, None]

    @property
    def total(self):
        dw = self.debye_waller if self.on_shell else self.debye_waller[:, None]
        return self.fan_migdal + dw

    @property
    def energy_shifts(self):
        """Real diagonal shift in eV, using the specified evaluation energies."""
        return self.total.real

    @property
    def linewidths(self):
        """Retarded linewidth -2 Im(Sigma), in eV (hbar/tau when applicable)."""
        return -2*self.total.imag

    @property
    def renormalized_energies(self):
        if not self.on_shell:
            raise ValueError("Renormalized band energies require on-shell evaluation (energyList=None)")
        return self.energies + self.energy_shifts

    def bandGap(self, occupiedBands):
        """Band-edge shifts on the supplied k set, not an unsampled BZ search.

        occupiedBands is the number of occupied bands per primitive cell.
        Results are scalar eV; momentum/band indices refer to kpoints.
        """
        nb = self.energies.shape[1]
        if not isinstance(occupiedBands, (int, np.integer)) or not 0 < occupiedBands < nb:
            raise ValueError("occupiedBands must be between one and nbands-1")
        renormalized = self.renormalized_energies
        vbm = float(np.max(self.energies[:, :occupiedBands]))
        cbm = float(np.min(self.energies[:, occupiedBands:]))
        new_vbm = float(np.max(renormalized[:, :occupiedBands]))
        new_cbm = float(np.min(renormalized[:, occupiedBands:]))
        if cbm <= vbm:
            raise ValueError("The sampled bare bands do not define an insulating gap")
        return dict(bare_gap=cbm-vbm, renormalized_gap=new_cbm-new_vbm,
                    gap_shift=(new_cbm-new_vbm)-(cbm-vbm),
                    bare_vbm=vbm, bare_cbm=cbm, vbm_shift=new_vbm-vbm, cbm_shift=new_cbm-cbm)

class ModelEPCLinewidth:
    """On-shell toy-EPC linewidth source for ``CollisionSpecification('serta')``.

    ``adiabatic=False`` is the dynamical Fan-Migdal/SERTA rate with phonon
    absorption, emission, and Pauli factors. ``adiabatic=True`` is the elastic
    frozen-phonon Born rate obtained from the static Fan propagator. In that
    case ``eta`` represents the numerical delta-function broadening and the
    result must be converged jointly with the q mesh.

    The calculator's ``kOffset`` mesh is evaluated in one
    ``ModelElectronSelfEnergy`` call. Results are cached and reused by
    LG, VG, and TDSE calculators having identical energies. The source supplies
    linewidths only: it does not add Re(Sigma), Debye-Waller shifts, or vertex
    corrections to optical matrix elements.
    """

    def __init__(self, qgrid=None, temperature=300., fermi=0., delta=.02,
                 adiabatic=False):
        self.qgrid=tuple(qgrid) if qgrid is not None else None
        self.temperature=float(temperature)
        self.fermi=float(fermi)
        self.delta=float(delta)
        self.adiabatic=bool(adiabatic)
        self.last_self_energy=None
        self._cache_signature=None
        self._cache_energies=None
        self._cache_linewidth=None

    def populationLinewidth(self, owner):
        energies=np.asarray(owner.energies)
        signature=(id(owner.model),tuple(owner.kgrid),tuple(np.asarray(owner.kOffset)))
        if (self._cache_signature==signature and self._cache_energies is not None
                and self._cache_energies.shape==energies.shape
                and np.allclose(self._cache_energies,energies,atol=1e-12,rtol=1e-12)):
            return self._cache_linewidth.copy()
        grid=tuple(owner.kgrid)
        qgrid=grid if self.qgrid is None else self.qgrid
        kpoints=np.mod(uniform_qmesh(grid)+owner.kOffset,1.)
        result=ModelElectronSelfEnergy(owner.model,kpoints,qgrid=qgrid,
                                       temperature=self.temperature,fermi=self.fermi,
                                       delta=self.delta,adiabatic=self.adiabatic)
        reference=result.energies.reshape(energies.shape)
        if not np.allclose(reference,energies,atol=2e-10,rtol=1e-10):
            raise ValueError("Model EPC linewidth states do not match the conductivity eigensystem")
        linewidth=result.linewidths.reshape(energies.shape)
        if np.any(linewidth < -1e-10) or not np.all(np.isfinite(linewidth)):
            raise ValueError("Model EPC retarded self-energy produced invalid linewidths")
        linewidth=np.maximum(linewidth,0.)
        self.last_self_energy=result
        self._cache_signature=signature
        self._cache_energies=energies.copy()
        self._cache_linewidth=linewidth.copy()
        return linewidth
