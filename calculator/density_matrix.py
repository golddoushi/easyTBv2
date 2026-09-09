"""Reusable first- and second-order density-matrix kernels.

The kernels in this module only build density matrices. Observable vertices,
Brillouin-zone integration, projections and output storage remain the
responsibility of the calculator classes that consume these blocks.
"""

from dataclasses import dataclass
from collections import OrderedDict
from types import SimpleNamespace
import numpy as np
from scipy.constants import Boltzmann, electron_volt
from scipy.sparse import bmat, csc_matrix, diags, eye
from scipy.sparse.linalg import LinearOperator, gmres, splu
from scipy.special import expit

from auxiliary import uniform_qmesh

DC_CHANNEL_NAMES = ("I", "E")
DC_CHANNEL_DESCRIPTIONS = {
    "I": "first-order intraband displaced population",
    "E": "first-order interband coherence",
}

AC_CHANNEL_NAMES = ("II", "EI", "IE", "EE")
AC_CHANNEL_DESCRIPTIONS = {
    "II": "second-order intraband from first-order intraband",
    "EI": "second-order interband from first-order intraband",
    "IE": "second-order intraband from first-order interband",
    "EE": "second-order interband from first-order interband",
}

class CollisionSpecification:
    """Type of the collision model used by density-matrix kernels.

    ``crta`` and ``serta`` construct a linewidth-only relaxation operator.
    ``epc`` constructs a mesh-global Born-Markov electron-phonon population
    operator, including scattering-in and Pauli blocking.  Its coherences use
    the corresponding secular out-scattering damping.  A bound collision
    object is created for each calculator k grid.

    Parameters
    ----------
    mode : str
        Collision mode: ``'crta'``, ``'serta'``, or ``'epc'``.  ``'full'``
        and ``'epc_markov'`` are accepted aliases for ``'epc'``.
    constantLinewidth : float, optional
        In eV.  The CRTA linewidth. For ``epc`` it is a number-conserving BGK
        background on the fixed-particle-number manifold and a coherence
        linewidth. It may be zero when the EPC graph itself relaxes every
        trace-free population mode.
    linewidthFloor : float, optional
        in eV. In present implementation, it behaves like some criterial,
        for the expressions with reverse, once the denominator < this, the
        expression is forced to be zero. In some TDSE calculators, once the
        linewidth < this, the calculator threw errors.
    linewidthSource
        Band/k-resolved linewidth source used only by ``serta``.
    secondOrderLinewidthDerivative : ``'frozen'`` | ``'covariant'``
        Second-order CRTA/SERTA only. The default retains the historical
        frozen-linewidth derivative. ``covariant`` differentiates the damping
        matrix in the complete orthogonal orbital basis, including its band
        projectors, and uses the same kinetic equation in LG and VG. Requires
        fullCovariantDerivative=True and a converged electronic mesh.
    qgrid : tuple(3), optional
        Uniform phonon mesh for ``epc``.  It defaults to the calculator k mesh
        and each q-grid dimension must divide the corresponding k-grid one.
    phononSelection : PhononModeSelection, optional
        Restricted q/branch quadrature for ``epc``, also read from
        modelPhonons.modeSelection. Supersedes qgrid, preserves absolute weights
        and requires the selected q shifts to close on the electronic mesh.
    phononTemperature : float
        Bath temperature in kelvin used by ``epc``.
    delta : float
        Positive Lorentzian energy-conservation broadening in eV used while
        building the ``epc`` transition rates.  It is not a relaxation time.
    bathStatistics : ``'quantum'`` | ``'classical'`` | ``'wigner'``
        Phonon population used by ``epc``.  The latter two reproduce the real
        harmonic-noise conventions used by the explicit phonon benchmarks.
    enforceDetailedBalance : bool
        Pair-balance the broadened transition rates at ``phononTemperature``.
        This makes the finite-temperature Fermi distribution an exact
        stationary state even at finite ``delta``. With classical or Wigner
        activities this is a thermally balanced surrogate, not the spectrum
        of an undamped classical trajectory.
    populationSolver : ``'auto'`` | ``'direct'`` | ``'iterative'``
        EPC population solves use cached sparse LU for <=512 states in auto
        mode, and diagonally preconditioned GMRES for larger systems.
    solverTolerance, stationaryTolerance : float
        Relative linear-solve tolerance and absolute steady-state residual
        tolerance in eV, respectively. EPC population solves are constrained
        to fixed particle number. Response clients require
        fullCovariantDerivative=True and a converged electronic mesh.

    """

    def __init__(self, mode:str,
                 constantLinewidth:float=0.06582, linewidthFloor:float=1e-12,
                 linewidthSource=None, qgrid=None, phononTemperature=300.,
                 delta=.03, bathStatistics="quantum", populationSolver="auto",
                 solverTolerance=1e-10, stationaryTolerance=1e-12,
                 enforceDetailedBalance=True, secondOrderLinewidthDerivative="frozen",
                 phononSelection=None):
        mode=str(mode).lower()
        if mode in ("full", "epc_markov"):
            mode="epc"
        if mode not in ("crta","serta","epc"):
            raise NotImplementedError("Available collision modes are 'crta', 'serta', and 'epc'")
        self.mode=mode
        self.constantLinewidth= float(constantLinewidth)
        self.linewidthFloor=float(linewidthFloor)
        self.linewidthSource=linewidthSource
        self.qgrid=None if qgrid is None else tuple(qgrid)
        self.phononSelection=phononSelection
        self.phononTemperature=float(phononTemperature)
        self.delta=float(delta)
        self.bathStatistics=str(bathStatistics).lower()
        self.populationSolver=str(populationSolver).lower()
        self.solverTolerance=float(solverTolerance)
        self.stationaryTolerance=float(stationaryTolerance)
        self.enforceDetailedBalance=bool(enforceDetailedBalance)
        if secondOrderLinewidthDerivative not in ("frozen", "covariant"):
            raise ValueError("secondOrderLinewidthDerivative must be frozen or covariant")
        self.secondOrderLinewidthDerivative=secondOrderLinewidthDerivative
        if (not np.isfinite(self.constantLinewidth) or self.constantLinewidth < 0
                or not np.isfinite(self.linewidthFloor) or self.linewidthFloor < 0):
            raise ValueError("constantLinewidth and linewidthFloor must be finite and nonnegative")
        if self.mode != "serta" and linewidthSource is not None:
            raise ValueError("linewidthSource is only meaningful for SERTA")
        if self.mode == "epc":
            if self.populationSolver not in ("auto", "direct", "iterative"):
                raise ValueError("populationSolver must be auto, direct, or iterative")
            if (not np.isfinite(self.solverTolerance) or self.solverTolerance <= 0
                    or not np.isfinite(self.stationaryTolerance) or self.stationaryTolerance <= 0):
                raise ValueError("collision solver tolerances must be positive and finite")
            if (not np.isfinite(self.phononTemperature) or self.phononTemperature < 0
                    or not np.isfinite(self.delta) or self.delta <= 0):
                raise ValueError("EPC collision requires nonnegative temperature and positive finite delta")
            if self.bathStatistics not in ("quantum", "classical", "wigner"):
                raise ValueError("bathStatistics must be 'quantum', 'classical', or 'wigner'")

    def bind(self, owner):
        if self.mode == "epc":
            return CollisionOperator_full(self,owner)
        return CollisionOperator_linewidth(self,owner)

    @classmethod
    def fromOwner(cls, owner):
        specification=getattr(owner,"collisionSpecification",None)
        if specification is None:
            specification=cls("crta",constantLinewidth=owner.eta)
        elif isinstance(specification,str):
            specification=cls(specification,constantLinewidth=owner.eta)
        elif not isinstance(specification,cls):
            raise TypeError("collisionSpecification must be None, a mode string, or CollisionSpecification")
        return specification.bind(owner)


class CollisionOperator:
    """Bound collision interface used by density-matrix kernels.

    ``solve`` inverts the collision-dressed electronic Liouvillian for a
    matrix source.  ``apply`` applies its positive damping operator.  The
    linewidth-only implementation is local in k; the EPC implementation has a
    mesh-global population block and returns a Fermi-level-specific view from
    :meth:`forFermi`.
    """

    mesh_coupled=False

    def mesh_derivative(self, matrix, direction):
        """Covariant spectral derivative in the complete orthogonal basis.

        Even-grid Nyquist derivatives are zero to preserve Hermiticity.
        The orbital connection is shared by linewidth and full collisions.
        """
        owner = self.owner
        if owner.model.overlap or owner.energies.shape[-1] != owner.model.norbs:
            raise NotImplementedError("Covariant collision derivatives require the complete orthogonal model basis")
        if not hasattr(self, "_mesh_connection"):
            buffer = np.zeros(owner.kgrid + (3, owner.model.norbs, owner.model.norbs), complex)
            for ir, rv in enumerate(owner.model.iRvec):
                buffer[tuple(np.asarray(rv) % owner.kgrid)] += owner.model.AA_R_W[ir]*owner.phaseFactorR[ir]
            connection = np.fft.ifftn(buffer, axes=(0, 1, 2))*np.prod(owner.kgrid)
            self._mesh_connection = .5*(connection + connection.conj().swapaxes(-1, -2))
        u = owner.vectors
        adj = u.conj().swapaxes(-1, -2)
        orbital = u @ matrix @ adj
        coefficients = np.fft.fftn(orbital, axes=(0, 1, 2))
        moment = np.zeros(owner.kgrid)
        for axis, n in enumerate(owner.kgrid):
            r = np.fft.fftfreq(n)*n
            if n % 2 == 0:
                r[n//2] = 0.
            shape = [1, 1, 1]
            shape[axis] = n
            moment += r.reshape(shape)*owner.model.lattice[axis, direction]
        derivative = np.fft.ifftn(1j*moment[..., None, None]*coefficients, axes=(0, 1, 2))
        connection = self._mesh_connection[..., direction, :, :]
        derivative -= 1j*(connection @ orbital - orbital @ connection)
        return adj @ derivative @ u

    def forFermi(self, fermi):
        """Return the collision view linearized around one chemical potential."""
        return self

    def second_variation(self, left, right):
        """D^2 C[left,right] in eV; linear CRTA/SERTA have no such source.

        This is a symmetric bilinear map, without complex conjugation. The
        linear damping exposed by apply is -D C, whereas this method uses
        the gain-minus-loss sign of the physical collision integral.
        """
        return np.zeros_like(np.asarray(left), dtype=complex)


class CollisionOperator_linewidth(CollisionOperator):
    """CRTA/SERTA secular linewidth operator, local in k and band indices.

    It computes linewidths without scattering-in or k-to-k+q population coupling.
    """

    def __init__(self, specification:CollisionSpecification, owner):
        self.specification = specification
        self.owner = owner
        self.constant_eta = None
        # CASE crta, return the constant linewidth directly
        if specification.mode == "crta":
            self.constant_eta = specification.constantLinewidth
            self.population_linewidth=np.full(owner.energies.shape,self.constant_eta,dtype=float)
        # CASE serta, the linewidths is to be computed further
        elif specification.mode == "serta":
            self.population_linewidth=self._sertaLinewidth()
        else:
            raise NotImplementedError("linewidth collision supports only CRTA and SERTA")
        # For now, the interband coherence linewidth is just the average of population linewidth
        # This is suitable for crta and covariant serta
        self.coherence_linewidth=0.5*(self.population_linewidth[..., :, None]
                                      + self.population_linewidth[..., None, :])

    def _sertaLinewidth(self):
        model = self.owner.model
        source= self.specification.linewidthSource
        # CASE1, linewidth is set externally, e.g., using model EPC-linewidth
        #        So, it is assumed to be a quantity per band per k due to the serta-form.
        #        Or, it provides populationLinewidth method to compute the linewdith per band per k
        if source is not None:
            linewidth = (source.populationLinewidth(self.owner) if hasattr(source, "populationLinewidth")
                         else np.asarray(source, dtype=float))
            if linewidth.shape != self.owner.energies.shape:
                raise ValueError(f"SERTA linewidth source has shape {linewidth.shape}, expected {self.owner.energies.shape}")
            if not np.all(np.isfinite(linewidth)) or np.any(linewidth < -1e-10):
                raise ValueError("SERTA linewidth source must be finite and nonnegative")
            return np.maximum(linewidth, 0.0)

        # CASE2, linewidth is computed by VASP, then transformed to real-grid to be electron_linewidth_sqrt_R (squared linewidth)
        #        Because the electron_linewidth_sqrt_R can be defined in coarsed grid, it is interpolated to the kgrid of the 
        #        calculator first, then transformed to the eigen-space. The diagonal term is used to be the linewidth per band per k.
        linewidth_sqrt_R = getattr(model,"electronLinewidthSqrt_R",None)
        if linewidth_sqrt_R is None:
            linewidth_sqrt_R = getattr(model,"electron_linewidth_sqrt_R",None)
        if linewidth_sqrt_R is None:
            raise ValueError("SERTA requires model.includeElectronSelfEnergy(...) before constructing the calculator")
        if model.overlap:
            raise NotImplementedError("SERTA linewidth interpolation is currently implemented for orthogonal Wannier models")

        linewidth_sqrt_R = np.asarray(linewidth_sqrt_R) * self.owner.phaseFactorR[:,None,None]
        linewidth_grid = np.zeros(self.owner.kgrid + (model.norbs, model.norbs), dtype=complex)
        for iR,Rvec in enumerate(model.iRvec):
            linewidth_grid[tuple(np.asarray(Rvec) % self.owner.kgrid)] += linewidth_sqrt_R[iR]
        linewidth_sqrt_W = np.fft.ifftn(linewidth_grid, axes=(0,1,2)) * np.prod(self.owner.kgrid)
        linewidth_W = linewidth_sqrt_W @ linewidth_sqrt_W.conj().swapaxes(-2,-1)
        vectors = self.owner.vectors
        linewidth_H = vectors.conj().swapaxes(-2,-1) @ linewidth_W @ vectors
        linewidth = np.diagonal(linewidth_H, axis1=-2, axis2=-1).real
        return np.maximum(linewidth, 0.0)

    @property
    def population_inverse(self):
        inverse=np.zeros_like(self.population_linewidth)
        np.divide(1.0,self.population_linewidth,out=inverse,
                  where=self.population_linewidth > self.specification.linewidthFloor)
        return inverse

    def population_resolvent(self, frequency=0.0):
        """Return the diagonal-population resolvent 1/(-Omega-i Gamma_n)."""
        denominator=-frequency-1j*self.population_linewidth
        result=np.zeros_like(denominator,dtype=complex)
        np.divide(1.0,denominator,out=result,
                  where=np.abs(denominator) > self.specification.linewidthFloor)
        return result

    def solve_population(self, source, frequency=0.0):
        """Solve ``(-Omega-i Gamma) x = source`` for band populations."""
        return np.asarray(source)*self.population_resolvent(frequency)

    def solve_steady_population(self, source):
        """Solve ``Gamma x = source`` for a dc population displacement."""
        return np.asarray(source)*self.population_inverse

    def apply_population(self, values):
        return self.population_linewidth*np.asarray(values)

    def apply(self, matrix):
        """Apply the secular damping superoperator to a band-space matrix."""
        return self.coherence_linewidth*np.asarray(matrix)

    def solve(self, source, delta_energy, frequency=0.0):
        """Solve the local, secular collision-dressed Liouvillian."""
        return np.asarray(source)*self.resolvent(delta_energy,frequency)

    def resolvent(self, delta_energy, frequency=0.0):
        denominator=np.asarray(delta_energy)-frequency-1j*self.coherence_linewidth
        result=np.zeros_like(denominator,dtype=complex)
        np.divide(1.0,denominator,out=result,
                  where=np.abs(denominator) > self.specification.linewidthFloor)
        return result

    def requireConstant(self, context):
        if self.constant_eta is None:
            raise NotImplementedError(f"{context} does not yet support a state-dependent SERTA collision source")
        return self.constant_eta


class CollisionOperator_full(CollisionOperator):
    """Fixed-number Markov EPC collision with scattering-in and Pauli blocking.

    The q-resolved golden-rule transition matrix is built once from the
    calculator eigensystem, ``System.modelEPC``, and ``System.modelPhonons``.
    Finite-width rates are pair-balanced by default so the Fermi state obeys
    detailed balance exactly. Population response solves remain in the
    trace-free tangent space. Coherences use secular half-sum damping; its
    population dependence enters D^2 C. This is a population-complete Markov
    collision, not an exact non-Markov phonon memory kernel.
    """

    mesh_coupled=True

    def __init__(self, specification, owner):
        self.specification=specification
        self.owner=owner
        model=owner.model
        if model.modelEPC is None or model.modelPhonons is None:
            raise ValueError("EPC collision requires modelEPC and modelPhonons")
        if model.overlap:
            raise NotImplementedError("mesh-global EPC collision currently requires an orthogonal model")
        if owner.energies.shape[-1] != model.norbs:
            raise NotImplementedError("mesh-global EPC collision requires the complete model basis")
        model.modelEPC.validate(model)
        self.kgrid=np.asarray(owner.kgrid,dtype=int)
        self.phononSelection = (getattr(model.modelPhonons, 'modeSelection', None)
                                if specification.phononSelection is None else specification.phononSelection)
        qgrid=self.kgrid if specification.qgrid is None else np.asarray(specification.qgrid)
        if (qgrid.shape != (3,) or not np.all(np.isfinite(qgrid))
                or np.any(qgrid < 1) or not np.all(qgrid == qgrid.astype(int))):
            raise ValueError("EPC qgrid must contain three positive integers")
        self.qgrid=qgrid.astype(int)
        if self.phononSelection is not None:
            self.phononSelection.mesh_shifts(self.kgrid)
            if len(getattr(model.modelPhonons, '_cell_origins', [None])) != 1:
                raise ValueError('evaluate selected EPC collisions on the primitive model; the selection uses primitive q and branch labels')
            self.phononSelection.branch_indices(len(model.modelPhonons.modes(self.phononSelection.qpoints[0])[0]))
        elif np.any(self.kgrid % self.qgrid):
            raise ValueError("each EPC qgrid dimension must divide the calculator kgrid")
        bath_kBT=Boltzmann*specification.phononTemperature/electron_volt
        if specification.enforceDetailedBalance:
            if bath_kBT <= 0:
                raise ValueError("detailed-balance EPC rates require a positive phonon temperature")
            if not np.isclose(owner.kBT,bath_kBT,rtol=2e-6,atol=1e-10):
                raise ValueError("detailed-balance EPC rates require electronic kBT to match phononTemperature")
        self.bath_kBT=bath_kBT
        self.qgrid=tuple(self.qgrid)
        self.nk=int(np.prod(self.kgrid))
        self.nbands=owner.energies.shape[-1]
        self.size=self.nk*self.nbands
        self.transition_rates=self._transition_rates()
        self.reverse_rates=self.transition_rates.T.tocsr()
        self._views={}
        self.population_linewidth=None
        self.coherence_linewidth=None

    @staticmethod
    def _lorentzian(value, delta):
        return delta/np.pi/(value*value+delta*delta)

    def _phonon_population(self, energies):
        specification=self.specification
        kBT=Boltzmann*specification.phononTemperature/electron_volt
        if specification.bathStatistics == "classical":
            return np.zeros_like(energies) if kBT == 0 else kBT/energies
        if specification.bathStatistics == "wigner":
            if kBT == 0:
                return np.full_like(energies,.5)
            return .5/np.tanh(energies/(2*kBT))
        if kBT == 0:
            return np.zeros_like(energies)
        x=energies/kBT
        return np.exp(-x)/(-np.expm1(-x))

    def _transition_rates(self):
        """Return R[final,initial] in eV for all mesh-closed EPC processes."""
        owner, model=self.owner, self.owner.model
        self.rate_balance_relative_change=0.
        if not np.any(model.modelEPC.alpha_R):
            return csc_matrix((self.size,self.size),dtype=float)
        grid=self.kgrid
        kindices=np.indices(tuple(grid)).reshape(3,-1).T
        kpoints=uniform_qmesh(grid)+owner.kOffset
        energies=owner.energies.reshape(self.nk,self.nbands)
        vectors=owner.vectors.reshape(self.nk,self.nbands,self.nbands)
        if self.phononSelection is None:
            qpoints=uniform_qmesh(self.qgrid)
            qshifts=np.indices(self.qgrid).reshape(3,-1).T*(grid//np.asarray(self.qgrid))
            qweights=np.full(len(qpoints), 1/len(qpoints))
        else:
            qpoints=self.phononSelection.qpoints
            qshifts=self.phononSelection.mesh_shifts(grid)
            qweights=self.phononSelection.weights
        # q maps each initial state to a distinct final k block. Sum modes
        # before storage and fill CSC directly, avoiding mode-resolved COO
        # triplets and their concatenation copies. No rate cutoff is applied.
        per_column=len(qpoints)*self.nbands
        index_dtype=np.int32 if self.size*per_column < np.iinfo(np.int32).max else np.int64
        values=np.empty((self.size,per_column),dtype=float)
        rows=np.empty(values.shape,dtype=index_dtype)
        source_k=np.arange(self.nk)
        modes=owner.model.modelPhonons
        for iq, (qshift, q, qweight) in enumerate(zip(qshifts,qpoints,qweights)):
            destination_indices=(kindices+qshift) % grid
            destination_k=np.ravel_multi_index(destination_indices.T,tuple(grid))
            phonon_energies,_=modes.modes(q)
            branches=(np.arange(len(phonon_energies)) if self.phononSelection is None
                      else self.phononSelection.branch_indices(len(phonon_energies)))
            phonon_energies=phonon_energies[branches]
            if np.any(phonon_energies <= 0) or not np.all(np.isfinite(phonon_energies)):
                raise ValueError("EPC rates require positive finite mode energies; zero acoustic modes need an explicit limiting prescription")
            population=self._phonon_population(phonon_energies)
            g=model.getModelEPCBatch(kpoints,q,vectors,vectors[destination_k])[:,branches]
            initial=energies[:,None,None,:]
            final=energies[destination_k][:,None,:,None]
            difference=initial-final
            emission=self._lorentzian(difference-phonon_energies[None,:,None,None],self.specification.delta)
            absorption=self._lorentzian(difference+phonon_energies[None,:,None,None],self.specification.delta)
            if self.specification.bathStatistics == "quantum":
                weight=(population[None,:,None,None]+1)*emission+population[None,:,None,None]*absorption
            else:
                weight=population[None,:,None,None]*(emission+absorption)
            rates=(2*np.pi*qweight*np.sum(np.abs(g)**2*weight,axis=1)).swapaxes(-1,-2)
            final_ids=np.broadcast_to(destination_k[:,None,None]*self.nbands
                                     +np.arange(self.nbands)[None,None,:],rates.shape)
            initial_ids=source_k[:,None,None]*self.nbands+np.arange(self.nbands)[None,:,None]
            rates=np.where(final_ids == initial_ids,0.,rates)
            slot=slice(iq*self.nbands,(iq+1)*self.nbands)
            values[:,slot]=rates.reshape(self.size,self.nbands)
            rows[:,slot]=final_ids.reshape(self.size,self.nbands)
        result=csc_matrix((values.ravel(),rows.ravel(),
                           np.arange(self.size+1,dtype=index_dtype)*per_column),shape=(self.size,self.size))
        result.eliminate_zeros()
        result.sort_indices()
        if self.specification.enforceDetailedBalance:
            raw_norm=np.linalg.norm(result.data)
            activity=(result+result.T).tocoo()
            energies=energies.reshape(-1)
            thermal_fraction=expit(-(energies[activity.row]-energies[activity.col])/self.bath_kBT)
            balanced=csc_matrix((activity.data*thermal_fraction,(activity.row,activity.col)),shape=activity.shape)
            balanced.eliminate_zeros()
            balanced.sort_indices()
            difference=balanced-result
            self.rate_balance_relative_change=(np.linalg.norm(difference.data)/raw_norm
                                               if raw_norm else 0.)
            result=balanced
        return result

    def population_collision(self, occupations):
        """Evaluate the nonlinear Pauli gain-minus-loss collision integral.

        ``transition_rates[final, initial]`` is the Golden-rule transition
        rate.  This routine is primarily useful for diagnostics; response
        kernels use the Fermi-level linearization returned by
        :meth:`forFermi`.
        """
        values=np.asarray(occupations)
        if values.shape != self.owner.energies.shape:
            raise ValueError(f"occupations have shape {values.shape}, expected {self.owner.energies.shape}")
        flat=values.reshape(-1)
        gain=(1-flat)*(self.transition_rates@flat)
        loss=flat*(self.reverse_rates@(1-flat))
        return np.asarray(gain-loss).reshape(values.shape)

    def population_jacobian(self, occupations):
        """Derivative of the EPC gain-minus-loss integral (no reservoir)."""
        p=np.asarray(occupations).reshape(-1)
        r, rt=self.transition_rates,self.reverse_rates
        return (diags(1-p)@r+diags(p)@rt-diags(r@p+rt@(1-p))).tocsc()

    def forFermi(self, fermi):
        key=float(fermi)
        if key not in self._views:
            self._views[key]=_CollisionOperatorFullFermi(self,key)
        return self._views[key]


class _CollisionOperatorFullFermi(CollisionOperator):
    """Fixed-number stationary state and tangent-space Pauli linearization.

    The optional constant linewidth is a number-conserving BGK background,
    ``-gamma*(p-f_mu[p])``, where ``mu[p]`` gives the same total population as
    ``p``. Population response vectors are solved with an explicit zero-trace
    constraint. Pair-balanced rates make the requested Fermi state stationary
    without the BGK term; raw-rate diagnostics may still use a constrained
    nonlinear stationary solve.
    """

    mesh_coupled=True

    def __init__(self, parent, fermi):
        self.parent=parent
        self.specification=parent.specification
        self.owner=parent.owner
        self.fermi=fermi
        energies=parent.owner.energies.reshape(-1)
        kBT=parent.owner.kBT
        if kBT <= 0:
            raise ValueError("mesh-global EPC collision requires kBT > 0 for Pauli linearization")
        self._energies=energies
        self.equilibrium_occupations=expit((fermi-energies)/kBT).reshape(self.owner.energies.shape)
        # Compatibility alias. This is a closed-system reference, not a reservoir.
        self.reservoir_occupations=self.equilibrium_occupations
        sensitivity=(self.equilibrium_occupations*(1-self.equilibrium_occupations)/kBT).reshape(-1)
        if sensitivity.sum() <= np.finfo(float).tiny:
            raise ValueError("fixed-number EPC response requires thermally active occupations")
        self.number_mode=sensitivity/sensitivity.sum()
        curvature=(self.equilibrium_occupations*(1-self.equilibrium_occupations)
                   *(1-2*self.equilibrium_occupations)/kBT**2).reshape(-1)
        self.number_mode_second=(curvature*sensitivity.sum()-sensitivity*curvature.sum())/sensitivity.sum()**3
        self.target_particle_number=float(self.equilibrium_occupations.sum())
        self._factors=OrderedDict()
        self.max_projected_source=0.
        self.stationary_occupations,self.stationary_info=self._stationary_state()
        self.population_damping=self._damping(self.stationary_occupations)
        linewidth=np.asarray(self.population_damping.diagonal()).real.reshape(parent.owner.energies.shape)
        self.population_linewidth=np.maximum(linewidth,0.)
        p=self.stationary_occupations.reshape(-1)
        outgoing=(self.specification.constantLinewidth+self.parent.transition_rates@p
                  +self.parent.reverse_rates@(1-p)).reshape(self.owner.energies.shape)
        self.coherence_linewidth=.5*(outgoing[..., :, None]+outgoing[..., None, :])
        self.constant_eta=None

    def _damping(self, populations):
        return (-self.parent.population_jacobian(populations)
                +self.specification.constantLinewidth*eye(self.parent.size,format="csc"))

    def _fermi_for_number(self, particle_number):
        value=np.asarray(particle_number)
        if abs(value.imag) > 1e-11*max(1.,abs(value.real)):
            raise ValueError("a physical population must have a real particle number")
        target=float(value.real)
        if target < -1e-10 or target > self.parent.size+1e-10:
            raise ValueError("particle number lies outside the Pauli interval")
        if abs(target-self.target_particle_number) <= 1e-12*max(1.,self.parent.size):
            return self.equilibrium_occupations
        if target <= 0:
            return np.zeros_like(self.equilibrium_occupations)
        if target >= self.parent.size:
            return np.ones_like(self.equilibrium_occupations)
        kBT=self.owner.kBT
        low=self._energies.min()-64*kBT
        high=self._energies.max()+64*kBT
        for _ in range(80):
            chemical_potential=.5*(low+high)
            occupations=expit((chemical_potential-self._energies)/kBT)
            if occupations.sum() < target:
                low=chemical_potential
            else:
                high=chemical_potential
        return occupations.reshape(self.owner.energies.shape)

    def _linear_solve(self, operator, source, cache_key=None, constrained=True):
        source=np.asarray(source).reshape(-1)
        dtype=np.result_type(source,operator.dtype,complex)
        if constrained:
            removed=source.sum()
            self.max_projected_source=max(self.max_projected_source,float(abs(removed)))
            source=source-self.number_mode*removed
            n=self.parent.size
            matrix=bmat([[operator.astype(complex),csc_matrix(self.number_mode[:,None])],
                         [csc_matrix(np.full((1,n),1/n)),csc_matrix((1,1))]],format="csc")
            rhs=np.concatenate((source.astype(complex),[0.]))
            factor_key=None if cache_key is None else ("fixed-number",cache_key)
        else:
            matrix=operator.astype(complex).tocsc()
            rhs=source.astype(complex)
            factor_key=cache_key
        if not np.any(rhs):
            return np.zeros(source.shape,dtype=dtype)
        direct=(self.specification.populationSolver == "direct" or
                (self.specification.populationSolver == "auto" and self.parent.size <= 512))
        if direct:
            factor=self._factors.get(factor_key) if factor_key is not None else None
            if factor is None:
                factor=splu(matrix)
                if factor_key is not None:
                    self._factors[factor_key]=factor
                    while len(self._factors)>2:
                        self._factors.popitem(last=False)
            elif factor_key is not None:
                self._factors.move_to_end(factor_key)
            result=factor.solve(rhs)
            for _ in range(3):
                correction=rhs-matrix@result
                if np.linalg.norm(correction) <= self.specification.solverTolerance*np.linalg.norm(rhs):
                    break
                result+=factor.solve(correction)
        else:
            diagonal=matrix.diagonal().astype(complex)
            diagonal[np.abs(diagonal) <= self.specification.linewidthFloor]=1.
            preconditioner=LinearOperator(matrix.shape,dtype=complex,matvec=lambda x:x/diagonal)
            result,info=gmres(matrix,rhs,M=preconditioner,
                rtol=self.specification.solverTolerance,atol=0.,restart=50,maxiter=200)
            if info != 0:
                raise RuntimeError(f"EPC population GMRES did not converge (info={info})")
        residual=np.linalg.norm(matrix@result-rhs)/np.linalg.norm(rhs)
        if not np.isfinite(residual) or residual > max(10*self.specification.solverTolerance,1e-12):
            raise RuntimeError(f"EPC population solve residual {residual:.3g} exceeds tolerance")
        if constrained:
            result=result[:-1]
            result=result-self.number_mode*result.sum()
        return result

    def _stationary_state(self):
        target=self.equilibrium_occupations
        gamma=self.specification.constantLinewidth
        def residual(p):
            reference=self._fermi_for_number(p.sum())
            return self.parent.population_collision(p)-gamma*(p-reference)
        p=target.copy()
        initial=float(np.max(np.abs(residual(p))))
        if self.specification.enforceDetailedBalance:
            scale=max(float(np.max(self.parent.transition_rates.data))
                      if self.parent.transition_rates.nnz else 0.,gamma,1.)
            if initial > max(100*np.finfo(float).eps*scale,10*self.specification.stationaryTolerance):
                raise RuntimeError(f"pair-balanced EPC rates violate their Fermi stationary state by {initial:.3g} eV")
            return p,dict(iterations=0,residual_eV=initial,detailed_balance=True,
                detailed_balance_rate_change=self.parent.rate_balance_relative_change,
                particle_number=self.target_particle_number,max_population_shift=0.)
        for iteration in range(80):
            rhs=residual(p)
            norm=float(np.max(np.abs(rhs)))
            if norm <= self.specification.stationaryTolerance:
                return p,dict(iterations=iteration,residual_eV=norm,detailed_balance=False,
                    raw_rate_initial_residual_eV=initial,particle_number=float(p.sum()),
                    max_population_shift=float(np.max(np.abs(p-target))))
            step=self._linear_solve(self._damping(p),rhs.ravel(),constrained=True).real.reshape(p.shape)
            step=np.where(((p <= 0)&(step < 0))|((p >= 1)&(step > 0)),0.,step)
            length=1.
            positive,negative=step>0,step<0
            if np.any(positive):
                length=min(length,float(np.min((1-p[positive])/step[positive])))
            if np.any(negative):
                length=min(length,float(np.min(-p[negative]/step[negative])))
            for _ in range(50):
                trial=p+length*step
                trial+=self.number_mode.reshape(p.shape)*(self.target_particle_number-trial.sum())
                if np.max(np.abs(residual(trial))) < norm:
                    p=trial
                    break
                length*=.5
            else:
                raise RuntimeError("EPC stationary-state Newton iteration stalled")
        raise RuntimeError("EPC stationary-state solve did not converge")

    def collision_integral(self, matrix):
        """Nonlinear fixed-number Pauli/BGK equation, excluding -i[H,rho].

        Coherences have half-sum population-dependent gain/loss damping.
        This closure omits coherence transfer and pure-dephasing vertices.
        """
        array=np.asarray(matrix)
        p=np.diagonal(array,axis1=-2,axis2=-1)
        r,rt=self.parent.transition_rates,self.parent.reverse_rates
        linewidth=(self.specification.constantLinewidth+r@p.ravel()+rt@(1-p.ravel())).reshape(p.shape)
        result=-.5*(linewidth[..., :,None]+linewidth[...,None,:])*array
        indices=np.arange(p.shape[-1])
        reference=self._fermi_for_number(p.sum())
        result[...,indices,indices]=(self.parent.population_collision(p)
            -self.specification.constantLinewidth*(p-reference))
        return result

    def second_variation(self, left, right):
        x,y=np.asarray(left),np.asarray(right)
        px,py=np.diagonal(x,axis1=-2,axis2=-1),np.diagonal(y,axis1=-2,axis2=-1)
        r,rt=self.parent.transition_rates,self.parent.reverse_rates
        gx=((r@px.ravel())-(rt@px.ravel())).reshape(px.shape)
        gy=((r@py.ravel())-(rt@py.ravel())).reshape(py.shape)
        result=-.5*((gx[..., :,None]+gx[...,None,:])*y
                     +(gy[..., :,None]+gy[...,None,:])*x)
        indices=np.arange(px.shape[-1])
        bgk=self.specification.constantLinewidth*self.number_mode_second.reshape(px.shape)*px.sum()*py.sum()
        result[...,indices,indices]=-px*gy-py*gx+bgk
        return result

    def stationary_derivative(self, direction, drive_velocity=None):
        """D_k rho0, using exact Fermi derivatives plus a mesh correction."""
        if drive_velocity is not None and not np.array_equal(drive_velocity,self.owner.velocity_K):
            raise NotImplementedError("Stationary EPC response currently requires the total electric drive vertex")
        e=self.owner.energies
        result=fermi_divided_difference(e[..., :,None],e[...,None,:],self.fermi,self.owner.kBT)
        result=result*self.owner.velocity_K[...,direction,:,:]
        correction=self.stationary_occupations-self.equilibrium_occupations
        if np.any(correction):
            result=result+self.mesh_derivative(diagonal_matrix(correction),direction)
        return result

    def apply_population(self, values):
        shape=self.owner.energies.shape
        array=np.asarray(values)
        if array.shape != shape:
            raise ValueError(f"population source has shape {array.shape}, expected {shape}")
        flat=array.reshape(-1)
        result=self.population_damping@flat
        result-=self.specification.constantLinewidth*self.number_mode*flat.sum()
        return np.asarray(result).reshape(shape)

    def solve_population(self, source, frequency=0.0):
        shape=self.owner.energies.shape
        array=np.asarray(source)
        if array.shape != shape:
            raise ValueError(f"population source has shape {array.shape}, expected {shape}")
        frequency=float(frequency)
        if not np.isfinite(frequency):
            raise ValueError("Population solve frequency must be finite")
        operator=self.population_damping-1j*frequency*eye(self.parent.size,format="csc")
        return (1j*self._linear_solve(operator,array.reshape(-1),frequency)).reshape(shape)

    def solve_steady_population(self, source):
        shape=self.owner.energies.shape
        array=np.asarray(source)
        if array.shape != shape:
            raise ValueError(f"population source has shape {array.shape}, expected {shape}")
        return self._linear_solve(self.population_damping,array.reshape(-1),0.).reshape(shape)

    @property
    def population_inverse(self):
        raise NotImplementedError("mesh-global EPC populations require solve_steady_population(source)")

    def population_resolvent(self, frequency=0.0):
        raise NotImplementedError("mesh-global EPC populations require solve_population(source, frequency)")

    def apply(self, matrix):
        array=np.asarray(matrix)
        expected=self.owner.energies.shape+(self.owner.energies.shape[-1],)
        if array.shape != expected:
            raise ValueError(f"density-matrix source has shape {array.shape}, expected {expected}")
        output=self.coherence_linewidth*array
        indices=np.arange(self.owner.energies.shape[-1])
        diagonal=np.diagonal(array,axis1=-2,axis2=-1)
        output[...,indices,indices]=self.apply_population(diagonal)
        return output

    def resolvent(self, delta_energy, frequency=0.0):
        denominator=np.asarray(delta_energy)-frequency-1j*self.coherence_linewidth
        result=np.zeros_like(denominator,dtype=complex)
        np.divide(1.,denominator,out=result,where=np.abs(denominator)>self.specification.linewidthFloor)
        return result

    def solve(self, source, delta_energy, frequency=0.0):
        array=np.asarray(source)
        result=array*self.resolvent(delta_energy,frequency)
        indices=np.arange(self.owner.energies.shape[-1])
        diagonal=np.diagonal(array,axis1=-2,axis2=-1)
        result[...,indices,indices]=self.solve_population(diagonal,frequency)
        return result

    def requireConstant(self, context):
        raise NotImplementedError(f"{context} does not support a mesh-global EPC collision")

def commutator(left, right):
    return left @ right - right @ left

def diagonal_matrix(values):
    """Embed band populations in a diagonal density matrix."""
    nbands = values.shape[-1]
    matrix = np.zeros(values.shape + (nbands,), dtype=complex)
    indices = np.arange(nbands)
    matrix[..., indices, indices] = values
    return matrix

def fermi_divided_difference(x, y, fermi, kBT):
    """Stable first divided difference of the Fermi function, including x=y."""
    if kBT <= 0:
        raise ValueError("Fermi divided differences require kBT > 0")
    low = (np.minimum(x, y) - fermi) / kBT
    high = (np.maximum(x, y) - fermi) / kBT
    distance = np.abs(np.asarray(x) - y) / kBT
    ratio = np.ones_like(distance, dtype=float)
    np.divide(-np.expm1(-distance), distance, out=ratio, where=distance != 0)
    return -expit(-low) * expit(high) * ratio / kBT

def fermi_second_divided_difference(x, y, z, fermi, kBT):
    """Stable symmetric f[x,y,z]; the confluent limit is f''(x)/2.

    Separated nodes use the divided-difference recurrence. Nearby nodes use
    a Taylor series and complete homogeneous polynomials, avoiding subtraction
    of almost equal first divided differences. All arguments may broadcast.
    """
    if kBT <= 0:
        raise ValueError("Fermi divided differences require kBT > 0")
    x, y, z = np.broadcast_arrays(x, y, z)
    low = np.minimum(np.minimum(x, y), z)
    high = np.maximum(np.maximum(x, y), z)
    middle = np.maximum(np.minimum(x, y), np.minimum(np.maximum(x, y), z))
    span = high - low
    nearby = span <= 0.5 * kBT
    result = np.zeros_like(span, dtype=float)
    numerator = (fermi_divided_difference(middle, high, fermi, kBT)
                 - fermi_divided_difference(low, middle, fermi, kBT))
    np.divide(numerator, span, out=result, where=~nearby)
    if np.any(nearby):
        center = low[nearby] + 0.5 * span[nearby]
        dx = (x[nearby] - center) / kBT
        dy = (y[nearby] - center) / kBT
        dz = (z[nearby] - center) / kBT
        e1, e2, e3 = dx + dy + dz, dx * dy + dx * dz + dy * dz, dx * dy * dz
        t = (center - fermi) / kBT
        # Expand the smaller particle/hole occupation to retain filled-band tails.
        occupation = expit(-np.abs(t))
        coefficients = [occupation, -occupation * (1.0 - occupation)]
        for order in range(1, 20):
            convolution = sum(coefficients[j] * coefficients[order-j] for j in range(1, order))
            coefficients.append(((2.0 * occupation - 1.0) * coefficients[order] + convolution) / (order + 1))
        h0, h1, h2 = np.ones_like(t), np.zeros_like(t), np.zeros_like(t)
        series = np.zeros_like(t)
        for order in range(19):
            coefficient = coefficients[order + 2]
            if order % 2 == 0:
                coefficient = np.where(t < 0, -coefficient, coefficient)
            series += coefficient * h0
            h0, h1, h2 = e1 * h0 - e2 * h1 + e3 * h2, h0, h1
        result[nearby] = series / kBT**2
    return result

@dataclass
class SecondOrderDensityMatrixBlock:
    """Density matrices for one ordered optical tensor component."""

    rho0: np.ndarray
    rho1_minus: dict
    rho1_plus: dict
    rho2: dict
    rho2_total: np.ndarray = None

    @property
    def total_second_order(self):
        """Prefer the direct total over cancellation between reporting channels."""
        return self.rho2_total if self.rho2_total is not None else sum(self.rho2.values())

@dataclass
class FirstOrderDensityMatrixBlock:
    """First-order dc density matrices for one electric-field direction."""

    rho0: np.ndarray
    rho1: dict

class LengthGaugeDCFirstOrderFermiState:
    def __init__(self, kernel, f, fp=None, fermi=None):
        self.kernel = kernel
        self.owner = kernel.owner
        self.collision = kernel.collision.forFermi(fermi)
        self.f = self.collision.stationary_occupations if self.collision.mesh_coupled else f
        self.fp = fp
        self.rho0 = diagonal_matrix(self.f)
        self.f_nm = self.f[..., None, :] - self.f[..., :, None]

    def block(self, direction, drive_velocity=None):
        """Return intraband and interband dc density matrices.

        ``drive_velocity`` may be a projected velocity tensor with the same
        layout as ``owner.velocity_K``. Its diagonal defines the projected
        intraband drive, preserving the projection convention used by the
        Hall calculators.
        """
        kernel = self.kernel
        velocity = self.owner.velocity_K if drive_velocity is None else drive_velocity
        velocity_direction = velocity[..., direction, :, :]
        if self.collision.mesh_coupled:
            derivative=self.collision.stationary_derivative(direction,drive_velocity)
            rho=self.collision.solve(1j*derivative,self.owner.deltaE)
            return FirstOrderDensityMatrixBlock(self.rho0,
                {"I":rho*kernel.diag if kernel.full_covariant_derivative else np.zeros_like(rho),
                 "E":rho*kernel.off_diag})
        rho_e = self.collision.solve(
            -1j*self.f_nm*kernel.deltaE_inverse_pure*velocity_direction*kernel.off_diag,
            self.owner.deltaE,
        )*kernel.off_diag
        rho_i = np.zeros_like(rho_e)
        if kernel.full_covariant_derivative:
            velocity_diag = np.diagonal(velocity_direction, axis1=-2, axis2=-1).real
            population = self.collision.solve_steady_population(-self.fp*velocity_diag)
            rho_i[..., kernel.indices, kernel.indices] = population
        return FirstOrderDensityMatrixBlock(
            rho0=self.rho0,
            rho1={"I": rho_i, "E": rho_e},
        )

class LengthGaugeDCFirstOrderDMKernel:
    """Reusable dc length-gauge first-order density-matrix kernel.

    The ``E`` channel is the interband coherence contributor. 
    ``fullCovariantDerivative=True`` additionally enables the diagonal ``I`` channel, 
    which requires finite temperature and finite relaxation energy. 
    In the conventions of the calculator module,

        rho_E[m,n] = -i (f_n-f_m) v_a[m,n]
                     / (DeltaE_mn (DeltaE_mn-i eta)),
        rho_I[n,n] = -(df_n/dE) v_a[n,n] / eta.

    These blocks include the electron-current minus sign: they are minus the
    physical density response for H(k+A), E=-dA/dt, and contract with +v.
    """

    channel_names = DC_CHANNEL_NAMES
    channel_descriptions = DC_CHANNEL_DESCRIPTIONS

    def __init__(self, owner, full_covariant_derivative=False, finite_temperature_occupations=False):
        self.owner = owner
        self.full_covariant_derivative = bool(full_covariant_derivative)
        self.finite_temperature_occupations = bool(finite_temperature_occupations)
        self.nbands = owner.energies.shape[-1]
        self.diag = np.eye(self.nbands)
        self.off_diag = 1.0 - self.diag
        self.indices = np.arange(self.nbands)
        self.deltaE_inverse_pure = owner.deltaEInverse_pure
        self.collision = CollisionSpecification.fromOwner(owner)
        self.g0_mn = (None if self.collision.mesh_coupled
                      else self.collision.resolvent(owner.deltaE))

        if self.collision.mesh_coupled and not self.full_covariant_derivative:
            raise ValueError("Stationary EPC response requires fullCovariantDerivative=True")

        if self.full_covariant_derivative and owner.kBT <= 0:
            raise ValueError("fullCovariantDerivative requires kBT > 0")
        if (self.full_covariant_derivative and not self.collision.mesh_coupled
                and not np.any(self.collision.population_linewidth > self.collision.specification.linewidthFloor)):
            raise ValueError("fullCovariantDerivative requires finite population relaxation")

    def fermi_state(self, ifermi, fermi):
        f, fp = self.occupations(ifermi, fermi)
        return LengthGaugeDCFirstOrderFermiState(self, f, fp, fermi)

    def occupations(self, ifermi, fermi):
        """Return only the band weights needed by a contracted response.

        This avoids constructing diagonal density matrices when an observable
        can be reduced to exact band-resolved response weights first.
        """
        if self.full_covariant_derivative or self.finite_temperature_occupations:
            f = self.owner.fdOccupation(fermi)
        else:
            f = np.int8(self.owner.energies < fermi)
        fp = self.owner.fdEnergyDerivatives(f)[0] if self.full_covariant_derivative else None
        return f, fp

class LengthGaugeFirstOrderDMKernel:
    """Linear conductivity, with exp(-i*Omega*t) and Omega in energy units.

    The returned matrices include the charge-current minus sign and contract
    with +v, as in the dc kernel. Full mode uses the Frechet derivative of
    f(H), including its finite degenerate limit, with no inverse-gap cutoff.
    I/E label diagonal/off-diagonal matrices (only their sum is invariant
    under rotations inside an exactly degenerate subspace).
    """

    channel_names = DC_CHANNEL_NAMES
    channel_descriptions = DC_CHANNEL_DESCRIPTIONS

    def __init__(self, owner, full_covariant_derivative=True, finite_temperature_occupations=True):
        self.owner = owner
        self.full_covariant_derivative = bool(full_covariant_derivative)
        self.finite_temperature_occupations = bool(finite_temperature_occupations)
        self.collision = CollisionSpecification.fromOwner(owner)
        self.diag = np.eye(owner.energies.shape[-1], dtype=bool)
        if self.collision.mesh_coupled and not self.full_covariant_derivative:
            raise ValueError("Stationary EPC response requires fullCovariantDerivative=True")
        if self.full_covariant_derivative and owner.kBT <= 0:
            raise ValueError("fullCovariantDerivative requires kBT > 0")

    def fermi_state(self, ifermi, fermi):
        return FirstOrderFermiState(self, fermi)

class FirstOrderFermiState:
    def __init__(self, kernel, fermi):
        self.kernel = kernel
        owner = kernel.owner
        self.collision = kernel.collision.forFermi(fermi)
        self.f = (owner.fdOccupation(fermi) if kernel.full_covariant_derivative
                  or kernel.finite_temperature_occupations else (owner.energies < fermi).astype(float))
        if self.collision.mesh_coupled:
            self.f=self.collision.stationary_occupations
        self.rho0 = diagonal_matrix(self.f)
        if kernel.full_covariant_derivative:
            e = owner.energies
            self.divided_difference = fermi_divided_difference(
                e[..., :, None], e[..., None, :], fermi, owner.kBT)
        else:
            self.divided_difference = (self.f[..., :, None]-self.f[..., None, :]) * owner.deltaEInverse_pure

    def derivative(self, direction, drive_velocity=None):
        if self.collision.mesh_coupled:
            return self.collision.stationary_derivative(direction,drive_velocity)
        velocity = self.kernel.owner.velocity_K if drive_velocity is None else drive_velocity
        return self.divided_difference * velocity[..., direction, :, :]

    def at_frequency(self, omega):
        return FirstOrderFrequencyState(self, omega)

class FirstOrderFrequencyState:
    def __init__(self, state, omega):
        self.state = state
        self.kernel = state.kernel
        self.omega = float(omega)
        if not np.isfinite(self.omega):
            raise ValueError("Photon energies must be finite")
        collision = self.state.collision
        if self.omega == 0 and self.kernel.full_covariant_derivative and np.any(
                collision.population_linewidth <= collision.specification.linewidthFloor):
            raise ValueError("dc conductivity requires finite population relaxation")
        self.collision = self.state.collision

    def block(self, direction, drive_velocity=None):
        rho = self.collision.solve(1j*self.state.derivative(direction, drive_velocity),
                                   self.kernel.owner.deltaE,self.omega)
        return FirstOrderDensityMatrixBlock(self.state.rho0,
                                            {"I": rho*self.kernel.diag, "E": rho*~self.kernel.diag})

class VelocityGaugeFirstOrderDMKernel(LengthGaugeFirstOrderDMKernel):
    """H(A)=H0+A*v, j(A)=-(v+A*w), rho_eq(A)=f(H(A)).

    In physical density conventions, the coefficient of A is
      R(Omega) = -([v,f]+i*G[D f])/(Delta-Omega-i*Gamma).
    Conductivity is i/Omega * (Tr[R v] + Tr[f w]). The equilibrium
    pole has residue Tr[(D f)v+f w], a BZ total derivative. At zero
    frequency the pole is removed analytically using the periodic BZ sum
    rule. The finite part is i*R'(0). At nonzero frequency the calculator
    retains the residue so mesh/basis errors remain visible.
    """

    def __init__(self, owner, full_covariant_derivative=True, finite_temperature_occupations=True):
        if not full_covariant_derivative:
            raise ValueError("VG conductivity requires fullCovariantDerivative=True for covariant relaxation")
        super().__init__(owner, full_covariant_derivative, finite_temperature_occupations)

    def fermi_state(self, ifermi, fermi):
        return VelocityGaugeFirstOrderFermiState(self, fermi)


class VelocityGaugeFirstOrderFermiState(FirstOrderFermiState):
    def at_frequency(self, omega):
        return VelocityGaugeFirstOrderFrequencyState(self, omega)


class VelocityGaugeFirstOrderFrequencyState(FirstOrderFrequencyState):
    def vector_potential_density(self, direction):
        """Physical delta-rho/A, before current contraction and E conversion."""
        df = self.state.derivative(direction)
        if self.collision.mesh_coupled:
            # Gauge transform of the LG response. This also transforms the
            # mesh-global collision and its nonthermal stationary reference.
            rho=self.collision.solve(1j*df,self.kernel.owner.deltaE,self.omega)
            return df-1j*self.omega*rho
        source = -commutator(self.kernel.owner.velocity_K[..., direction, :, :], self.state.rho0)
        source -= 1j*self.collision.apply(df)
        return self.collision.solve(source,self.kernel.owner.deltaE,self.omega)

    def block(self, direction, drive_velocity=None):
        # (R(Omega)-R(0))/Omega = Df/(Delta-Omega-i*Gamma).
        # Evaluate this identity before subtraction, retaining precision at dc.
        # The omitted R(0)=Df pole is restored with f*w by the calculator.
        return super().block(direction, drive_velocity)


class _SecondOrderDMFrequencyState:
    '''Abstract of the 2nd order density matrix at specific real frequency'''
    def block(self) -> SecondOrderDensityMatrixBlock:
        raise NotImplementedError("The block function for this kernel is not implemented in the frequency state.")

class _SecondOrderDMFermiState:
    '''Abstract of the 2nd order density matrix at specific Fermi energy'''
    def __init__(self, kernel, f, dfdk=None, d2fdk2=None, fermi=None):
        self.kernel = kernel
        self.owner = kernel.owner
        self.collision = kernel.collision.forFermi(fermi)
        self.f = f
        self.dfdk = dfdk
        self.d2fdk2 = d2fdk2
        self.rho0 = diagonal_matrix(f)
        self.f_nm = f[..., None, :] - f[..., :, None]

    def at_frequency(self) -> _SecondOrderDMFrequencyState:
        raise NotImplementedError("The at_frequency function for this kernel is not implemented in the fermi state.")

class _SecondOrderDMKernel:
    channel_names = AC_CHANNEL_NAMES
    channel_descriptions = AC_CHANNEL_DESCRIPTIONS

    def __init__(self, owner, full_covariant_derivative=False, finite_temperature_occupations=False):
        self.owner = owner
        self.full_covariant_derivative = bool(full_covariant_derivative)
        self.finite_temperature_occupations = bool(finite_temperature_occupations)
        self.nbands = owner.energies.shape[-1]
        self.diag = np.eye(self.nbands)
        self.off_diag = 1.0 - self.diag
        self.indices = np.arange(self.nbands)
        self.deltaE_mn = owner.deltaE
        self.collision = CollisionSpecification.fromOwner(owner)
        self.covariant_linewidth = (
            not self.collision.mesh_coupled
            and self.collision.specification.secondOrderLinewidthDerivative == "covariant")
        if self.covariant_linewidth:
            if not self.full_covariant_derivative:
                raise ValueError("Covariant linewidth derivatives require fullCovariantDerivative=True")
            self.intra_mask = self.diag.astype(bool)
            self.inter_mask = ~self.intra_mask
            self.inter_inverse = owner.deltaEInverse_pure*self.off_diag
        self.g0_mn = (None if self.collision.mesh_coupled
                      else self.collision.resolvent(self.deltaE_mn))

        if self.full_covariant_derivative and owner.kBT <= 0:
            raise ValueError("fullCovariantDerivative requires kBT > 0")
        if (self.full_covariant_derivative and not self.collision.mesh_coupled
                and not np.any(self.collision.population_linewidth > self.collision.specification.linewidthFloor)):
            raise ValueError("fullCovariantDerivative requires finite population relaxation")

        self.state_class : _SecondOrderDMFermiState

    @property
    def active_rho2_channels(self):
        """Density-matrix channels that can be nonzero for this mode."""
        return self.channel_names if self.full_covariant_derivative else ("IE", "EE")

    def fermi_state(self, ifermi, fermi) -> _SecondOrderDMFermiState:
        if self.covariant_linewidth:
            return CovariantLinewidthSecondOrderFermiState(self, fermi,
                velocity_gauge=isinstance(self, VelocityGaugeSecondOrderDMKernel))
        if self.collision.mesh_coupled:
            if not self.full_covariant_derivative:
                raise ValueError("Nonlinear EPC response requires fullCovariantDerivative=True")
            return MarkovSecondOrderFermiState(self,fermi,
                velocity_gauge=isinstance(self,VelocityGaugeSecondOrderDMKernel))
        if self.full_covariant_derivative:     
            # the df/dk and d2f/dk2 are to be computed only for full covariant derivate case
            # NOTE: the \rho(0) becomes smoothed f-d, too. Because the df/dt and d2f/dk2 are only meaningful under finite temp.
            f, dfdk, d2fdk2 = self.owner.fermiKDerivatives(ifermi)
        elif self.finite_temperature_occupations:
            # switch on the finite temperature with smoothed f-d distribution
            f = self.owner.fdOccupation(fermi)
            dfdk = d2fdk2 = None
        else:
            # step function, the 0K f-d distribution
            f = np.int8(self.owner.energies < fermi)
            dfdk = d2fdk2 = None
        return self.state_class(self, f, dfdk, d2fdk2, fermi=fermi)

class LengthGaugeFrequencyState(_SecondOrderDMFrequencyState):
    """Historical interband-dipole-only response (full mode uses the class below)."""
    def __init__(self, fermi_state, omega):
        self.fermi_state = fermi_state
        self.kernel = fermi_state.kernel
        self.owner = self.kernel.owner
        self.omega = omega
        self.rho1_e = {}     # inter-band rho(1)
        self.rho1_i = {}     # intra-band rho(1)
        self.factor = {}

        for direction in range(3):
            r_direction = self.owner.r_K[..., direction, :, :]
            for sign in (-1, 1):
                frequency = sign * omega
                source=fermi_state.f_nm*r_direction
                self.rho1_e[(direction, sign)] = fermi_state.collision.solve(
                    source,self.kernel.deltaE_mn,frequency
                )*self.kernel.off_diag

    def block(self, a, b):
        kernel = self.kernel
        owner = self.owner
        state = self.fermi_state
        r_a = owner.r_K[..., a, :, :]
        r_b = owner.r_K[..., b, :, :]
        rho1_e_a = self.rho1_e[(a, -1)]
        rho1_e_b = self.rho1_e[(b, 1)]
        comm_ba = commutator(r_b, rho1_e_a)
        comm_ab = commutator(r_a, rho1_e_b)
        zero = np.zeros_like(kernel.deltaE_mn, dtype=complex)
        total=state.collision.solve(comm_ba+comm_ab,kernel.deltaE_mn)
        rho2 = {
            "II": zero,
            "EI": zero,
            "IE": total*kernel.diag,
            "EE": total*kernel.off_diag,
        }

        return SecondOrderDensityMatrixBlock(
            rho0=state.rho0,
            rho1_minus={"I": zero, "E": rho1_e_a},
            rho1_plus={"I": zero, "E": rho1_e_b},
            rho2=rho2,
        )

class LengthGaugeFermiState(_SecondOrderDMFermiState):
    def at_frequency(self, omega): 
        return LengthGaugeFrequencyState(self, omega)

class DividedDifferenceLGFermiState(_SecondOrderDMFermiState):
    """Full LG equilibrium derivatives without band-connection denominators."""

    def __init__(self, kernel, fermi):
        owner = kernel.owner
        super().__init__(kernel, owner.fdOccupation(fermi), fermi=fermi)
        energies = owner.energies
        v = owner.velocity_K
        self.f_div1 = fermi_divided_difference(energies[..., :, None], energies[..., None, :], fermi, owner.kBT)
        self.df = self.f_div1[..., None, :, :] * v
        # w[a,b] = D_b V_a. Only an N x N intermediate is made for each band l.
        self.d2f = self.f_div1[..., None, None, :, :] * owner.w_K
        for l in range(kernel.nbands):
            div2 = fermi_second_divided_difference(
                energies[..., :, None], energies[..., l, None, None], energies[..., None, :], fermi, owner.kBT
            )
            for a in range(3):
                for b in range(a, 3):
                    term = div2 * (v[..., b, :, l, None] * v[..., a, l, None, :]
                                   + v[..., a, :, l, None] * v[..., b, l, None, :])
                    self.d2f[..., a, b, :, :] += term
                    if a != b:
                        self.d2f[..., b, a, :, :] += term

        self.df_i = self.df * kernel.intra_mask[..., None, :, :]
        self.d2f_i = np.empty_like(self.d2f)
        # Derivative of a spectral-block projection. These inverse gaps affect
        # channel bookkeeping only, never the direct total density matrix.
        for b in range(3):
            connection = kernel.inter_inverse * v[..., b, :, :]
            for a in range(3):
                self.d2f_i[..., a, b, :, :] = (
                    kernel.intra_mask * (self.d2f[..., a, b, :, :] + commutator(connection, self.df[..., a, :, :]))
                    - commutator(connection, self.df_i[..., a, :, :])
                )

    def at_frequency(self, omega):
        return DividedDifferenceLGFrequencyState(self, omega)

class DividedDifferenceLGFrequencyState(_SecondOrderDMFrequencyState):
    """Differentiate the Liouville equation with frozen equilibrium linewidths."""

    def __init__(self, fermi_state, omega):
        self.fermi_state = fermi_state
        self.kernel = fermi_state.kernel
        self.owner = self.kernel.owner
        self.omega = omega
        self.resolvent = {}
        self.rho1 = {}
        self.rho1_i = {}
        self.rho1_e = {}
        for sign in (-1, 1):
            for a in range(3):
                rho = fermi_state.collision.solve(
                    1j*fermi_state.df[..., a, :, :],self.kernel.deltaE_mn,sign*omega)
                self.rho1[(a, sign)] = rho
                self.rho1_i[(a, sign)] = rho * self.kernel.intra_mask
                self.rho1_e[(a, sign)] = rho * self.kernel.inter_mask

    def block(self, a, b):
        kernel, state = self.kernel, self.fermi_state
        source = np.zeros_like(self.rho1[(a, -1)])
        source_i = np.zeros_like(source)
        for drive, derivative, sign in ((a, b, -1), (b, a, 1)):
            v = self.owner.velocity_K[..., derivative, :, :]
            source += state.collision.solve(
                -state.d2f[..., drive, derivative, :, :] - 1j*commutator(v,self.rho1[(drive,sign)]),
                kernel.deltaE_mn,sign*self.omega)
            source_i += state.collision.solve(
                -state.d2f_i[..., drive, derivative, :, :] - 1j*commutator(v,self.rho1_i[(drive,sign)]),
                kernel.deltaE_mn,sign*self.omega)
        total = state.collision.solve(source,kernel.deltaE_mn)
        from_i = state.collision.solve(source_i,kernel.deltaE_mn)
        from_e = total - from_i
        return SecondOrderDensityMatrixBlock(
            rho0=state.rho0,
            rho1_minus={"I": self.rho1_i[(a, -1)], "E": self.rho1_e[(a, -1)]},
            rho1_plus={"I": self.rho1_i[(b, 1)], "E": self.rho1_e[(b, 1)]},
            rho2={"II": from_i * kernel.intra_mask, "EI": from_i * kernel.inter_mask,
                  "IE": from_e * kernel.intra_mask, "EE": from_e * kernel.inter_mask},
            rho2_total=total,
        )

class LengthGaugeSecondOrderDMKernel(_SecondOrderDMKernel):
    """Full LG uses divided differences; the interband-only path is unchanged.

    In full mode degen groups adjacent eigenvalues into spectral blocks for
    I/E reporting only. No energies are flattened and no gaps are masked in
    the total. I denotes within-block matrices, E between-block matrices.
    For isolated bands these are the usual diagonal/off-diagonal channels.
    """

    state_class = LengthGaugeFermiState

    def __init__(self, owner, full_covariant_derivative=False, finite_temperature_occupations=False):
        super().__init__(owner, full_covariant_derivative, finite_temperature_occupations)
        if self.full_covariant_derivative:
            sectors = {"I": "within-block", "E": "between-block"}
            self.channel_descriptions = {
                name: f"second-order {sectors[name[0]]} from first-order {sectors[name[1]]}"
                for name in self.channel_names
            }
            if not hasattr(owner, "w_K"):
                raise ValueError("Full LG requires w_K[a,b] = D_b velocity_K[a]")
            energies = owner.energies
            scale = np.maximum(1.0, np.max(np.abs(energies), axis=-1, keepdims=True))
            tolerance = np.maximum(abs(getattr(owner, "degen", 0.0)), 64 * np.finfo(float).eps * scale)
            labels = np.concatenate((np.zeros_like(energies[..., :1], dtype=int),
                                     np.cumsum(np.diff(energies, axis=-1) > tolerance, axis=-1)), axis=-1)
            self.intra_mask = labels[..., :, None] == labels[..., None, :]
            self.inter_mask = ~self.intra_mask
            self.inter_inverse = np.zeros_like(self.deltaE_mn)
            np.divide(1.0, self.deltaE_mn, out=self.inter_inverse, where=self.inter_mask)

    def fermi_state(self, ifermi, fermi):
        if self.collision.mesh_coupled or self.covariant_linewidth:
            return _SecondOrderDMKernel.fermi_state(self,ifermi,fermi)
        if self.full_covariant_derivative:
            return DividedDifferenceLGFermiState(self, fermi)
        return super().fermi_state(ifermi, fermi)

class VelocityGaugeFrequencyState(_SecondOrderDMFrequencyState):
    def __init__(self, fermi_state, omega):
        self.fermi_state = fermi_state
        self.kernel = fermi_state.kernel
        self.owner = self.kernel.owner
        self.omega = omega
        if abs(omega) < 1e-15:
            raise ValueError("velocity-gauge response is singular at exactly Omega=0")

        self.rho1_e = {}
        self.rho1_i = {}
        for direction in range(3):
            source_e = self.owner.velocity_K[..., direction, :, :] * fermi_state.f_nm
            if self.kernel.full_covariant_derivative:
                source_e += fermi_state.collision.apply(
                    self.owner.r_K[..., direction, :, :] * fermi_state.f_nm * self.kernel.off_diag
                )
            for sign in (-1, 1):
                frequency = sign * omega
                self.rho1_e[(direction, sign)] = fermi_state.collision.solve(
                    source_e,self.kernel.deltaE_mn,frequency
                )*self.kernel.off_diag
                if self.kernel.full_covariant_derivative:
                    population=fermi_state.collision.solve_population(
                        1j*fermi_state.collision.apply_population(fermi_state.dfdk[...,direction,:]),frequency)
                    self.rho1_i[(direction, sign)] = diagonal_matrix(population)

    def block(self, a, b):
        kernel = self.kernel
        owner = self.owner
        state = self.fermi_state
        v_a = owner.velocity_K[..., a, :, :]
        v_b = owner.velocity_K[..., b, :, :]
        w_ab = 0.5 * (
            owner.w_K[..., a, b, :, :] + owner.w_K[..., b, a, :, :]
        )
        rho1_e_a = self.rho1_e[(a, -1)]
        rho1_e_b = self.rho1_e[(b, 1)]
        source_e = commutator(v_b, rho1_e_a) + commutator(v_a, rho1_e_b)
        source_contact = -w_ab * state.f_nm
        zero = np.zeros_like(kernel.deltaE_mn, dtype=complex)
        resolved_e=state.collision.solve(source_e,kernel.deltaE_mn)
        rho2 = {
            "II": zero,
            "IE": resolved_e*kernel.diag,
            "EE": state.collision.solve(source_e+source_contact,kernel.deltaE_mn)*kernel.off_diag,
        }
        if kernel.full_covariant_derivative:
            rho1_i_a = self.rho1_i[(a, -1)]
            rho1_i_b = self.rho1_i[(b, 1)]
            population_a = np.diagonal(rho1_i_a, axis1=-2, axis2=-1)
            population_b = np.diagonal(rho1_i_b, axis1=-2, axis2=-1)
            source_i = (
                v_b * (population_a[..., None, :] - population_a[..., :, None])
                + v_a * (population_b[..., None, :] - population_b[..., :, None])
            )
            rho2["EI"] = state.collision.solve(source_i,kernel.deltaE_mn)*kernel.off_diag
        else:
            rho1_i_a = rho1_i_b = zero
            rho2["EI"] = zero

        return SecondOrderDensityMatrixBlock(
            rho0=state.rho0,
            rho1_minus={"I": rho1_i_a, "E": rho1_e_a},
            rho1_plus={"I": rho1_i_b, "E": rho1_e_b},
            rho2=rho2,
        )

class VelocityGaugeFermiState(_SecondOrderDMFermiState):
    def at_frequency(self, omega):
        return VelocityGaugeFrequencyState(self, omega)

    def relaxation(self, a, b):
        kernel = self.kernel
        zero = np.zeros_like(kernel.deltaE_mn, dtype=complex)
        if not kernel.full_covariant_derivative:
            return {name: zero.copy() for name in AC_CHANNEL_NAMES}

        c2_ab = self.owner.fermiCovariantSecondDerivativeChannels(
            self.f, self.dfdk, self.d2fdk2, a, b
        )
        c2_ba = c2_ab if a == b else self.owner.fermiCovariantSecondDerivativeChannels(
            self.f, self.dfdk, self.d2fdk2, b, a
        )
        result = {}
        for name in AC_CHANNEL_NAMES:
            source = 0.5j * self.collision.apply(c2_ab[name] + c2_ba[name])
            sector = kernel.diag if name in ("II", "IE") else kernel.off_diag
            result[name] = self.collision.solve(source,kernel.deltaE_mn)*sector
        return result

class VelocityGaugeSecondOrderDMKernel(_SecondOrderDMKernel):
    """Velocity-gauge Liouville kernel with optional covariant relaxation."""

    state_class = VelocityGaugeFermiState

class MarkovSecondOrderFermiState(_SecondOrderDMFermiState):
    """Stationary nonlinear Markov response with a covariant mesh derivative.

    Both gauges use the same LG kinetic equation. The VG state is its gauge
    transform, including the collision/reference transformation; observable
    contact vertices remain in the established Grid calculators.
    """
    def __init__(self,kernel,fermi,velocity_gauge=False):
        collision=kernel.collision.forFermi(fermi)
        super().__init__(kernel,collision.stationary_occupations,fermi=fermi)
        self.velocity_gauge=velocity_gauge
        self.df=np.stack([collision.stationary_derivative(a) for a in range(3)],axis=-3)
        # Subtract an analytic CRTA reference before taking numerical mesh
        # derivatives. Thus the weak-EPC limit retains the established exact
        # Fermi derivatives even on a coarse mesh.
        reference_kernel=SimpleNamespace(owner=self.owner,nbands=kernel.nbands,
            collision=CollisionOperator_linewidth(CollisionSpecification('crta',
                constantLinewidth=collision.specification.constantLinewidth),self.owner),
            intra_mask=kernel.diag.astype(bool),inter_mask=kernel.off_diag.astype(bool),
            inter_inverse=self.owner.deltaEInverse_pure*kernel.off_diag)
        self.reference=DividedDifferenceLGFermiState(reference_kernel,fermi)
        self.d2f=self.reference.d2f.copy()
        for a in range(3):
            for b in range(3):
                self.d2f[...,a,b,:,:]+=.5*(
                    collision.mesh_derivative(self.df[...,a,:,:]-self.reference.df[...,a,:,:],b)
                    +collision.mesh_derivative(self.df[...,b,:,:]-self.reference.df[...,b,:,:],a))

    def at_frequency(self,omega):
        return MarkovSecondOrderFrequencyState(self,omega)

    def relaxation(self,a,b):
        # The VG gauge transformation already contains the complete D^2 rho0.
        return {name:np.zeros_like(self.rho0) for name in self.kernel.channel_names}


class MarkovSecondOrderFrequencyState(_SecondOrderDMFrequencyState):
    """Second-order response of the explicit nonlinear secular Pauli model."""
    def __init__(self,state,omega):
        self.state=state
        self.kernel=state.kernel
        self.omega=float(omega)
        if self.omega == 0:
            raise ValueError("Second-order optical EPC response requires nonzero frequency")
        self.rho1={}
        self.reference_rho1={}
        self.derivatives={}
        for a in range(3):
            for sign in (-1,1):
                self.rho1[a,sign]=state.collision.solve(1j*state.df[...,a,:,:],
                    self.kernel.deltaE_mn,sign*self.omega)
                self.reference_rho1[a,sign]=state.reference.collision.solve(
                    1j*state.reference.df[...,a,:,:],self.kernel.deltaE_mn,sign*self.omega)

    def derivative(self,drive,sign,direction,intraband=False):
        key=(drive,sign,direction,intraband)
        if key not in self.derivatives:
            state,kernel=self.state,self.kernel
            mask=kernel.diag if intraband else 1.
            reference=self.reference_rho1[drive,sign]*mask
            d2f=state.reference.d2f_i if intraband else state.reference.d2f
            analytic=state.reference.collision.solve(
                1j*d2f[...,drive,direction,:,:]
                -commutator(state.owner.velocity_K[...,direction,:,:],reference),
                kernel.deltaE_mn,sign*self.omega)
            correction=self.rho1[drive,sign]*mask-reference
            self.derivatives[key]=analytic+state.collision.mesh_derivative(correction,direction)
        return self.derivatives[key]

    def block(self,a,b):
        state,kernel=self.state,self.kernel
        left,right=self.rho1[a,-1],self.rho1[b,1]
        db_left,da_right=self.derivative(a,-1,b),self.derivative(b,1,a)
        db_left_i,da_right_i=self.derivative(a,-1,b,True),self.derivative(b,1,a,True)
        nonlinear=state.collision.second_variation(left,right)
        total=state.collision.solve(1j*(db_left+da_right-nonlinear),kernel.deltaE_mn)
        # Nonlinear terms contain at least one population perturbation and are
        # reported with the intraband-origin channels; totals are independent
        # of this reporting choice.
        from_i=state.collision.solve(1j*(db_left_i+da_right_i-nonlinear),kernel.deltaE_mn)
        from_e=total-from_i
        channels={"II":from_i*kernel.diag,"EI":from_i*kernel.off_diag,
                  "IE":from_e*kernel.diag,"EE":from_e*kernel.off_diag}
        if state.velocity_gauge:
            for name in channels:
                channels[name]*=self.omega**2
            cross_i=1j*self.omega*(db_left_i-da_right_i)+state.d2f[...,a,b,:,:]
            cross_e=1j*self.omega*(db_left-da_right)-1j*self.omega*(db_left_i-da_right_i)
            channels['II']+=cross_i*kernel.diag
            channels['EI']+=cross_i*kernel.off_diag
            channels['IE']+=cross_e*kernel.diag
            channels['EE']+=cross_e*kernel.off_diag
            total=sum(channels.values())
            # Existing VG observable contractions expect minus delta-rho/A.
            left=-state.df[...,a,:,:]-1j*self.omega*left
            right=-state.df[...,b,:,:]+1j*self.omega*right
        return SecondOrderDensityMatrixBlock(state.rho0,
            {"I":left*kernel.diag,"E":left*kernel.off_diag},
            {"I":right*kernel.diag,"E":right*kernel.off_diag},channels,total)

class CovariantLinewidthSecondOrderFermiState(DividedDifferenceLGFermiState):
    """Linear RTA collision with its full k derivative, in either gauge.

    Half-sum coherence damping is C[X]={G,X}/2 with G=diag(Gamma_n).
    Thus (D C)[X]={D G,X}/2. Differentiating G in orbital space includes
    both linewidth gradients and derivatives of the band projectors.
    This does not add scattering-in or occupation-dependent collision rates.
    """

    def __init__(self, kernel, fermi, velocity_gauge=False):
        super().__init__(kernel, fermi)
        self.velocity_gauge = velocity_gauge
        damping = diagonal_matrix(self.collision.population_linewidth)
        # Remove a constant identity before FFT to retain the exact CRTA limit.
        damping -= np.eye(kernel.nbands)*np.mean(self.collision.population_linewidth)
        self.damping_derivative = np.stack([
            self.collision.mesh_derivative(damping, a) for a in range(3)], axis=-3)

    def at_frequency(self, omega):
        return CovariantLinewidthSecondOrderFrequencyState(self, omega)

    def relaxation(self, a, b):
        # The common VG transformation already includes the target vertices.
        return {name: np.zeros_like(self.rho0) for name in self.kernel.channel_names}


class CovariantLinewidthSecondOrderFrequencyState(MarkovSecondOrderFrequencyState):
    """Use the common LG/VG assembly with an analytic Liouvillian derivative."""

    def __init__(self, state, omega):
        if not np.isfinite(omega) or omega == 0:
            raise ValueError("Second-order optical response requires nonzero finite frequency")
        self.state, self.kernel, self.omega = state, state.kernel, float(omega)
        self.derivatives = {}
        self.rho1 = {(a, sign): state.collision.solve(
            1j*state.df[..., a, :, :], self.kernel.deltaE_mn, sign*self.omega)
            for a in range(3) for sign in (-1, 1)}

    def derivative(self, drive, sign, direction, intraband=False):
        key = (drive, sign, direction, intraband)
        if key not in self.derivatives:
            state, kernel = self.state, self.kernel
            rho = self.rho1[drive, sign]*(kernel.diag if intraband else 1.)
            d2f = state.d2f_i if intraband else state.d2f
            dg = state.damping_derivative[..., direction, :, :]
            # A=ad_H-Omega-i C: D rho=A^-1(i D^2 f-[v,rho]+i (D C)[rho]).
            source = (1j*d2f[..., drive, direction, :, :]
                      - commutator(state.owner.velocity_K[..., direction, :, :], rho)
                      + .5j*(dg @ rho + rho @ dg))
            self.derivatives[key] = state.collision.solve(source, kernel.deltaE_mn, sign*self.omega)
        return self.derivatives[key]
