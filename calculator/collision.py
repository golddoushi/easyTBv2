"""Collision models and explicit, reusable preparation for response calculations."""

from phonon import PhononBath
from .requirements import ComputationRequirements
from collections import OrderedDict
from dataclasses import dataclass
import numpy as np
from scipy.constants import Boltzmann, electron_volt
from scipy.sparse import bmat, csc_matrix, diags, eye
from scipy.sparse.linalg import LinearOperator, gmres, splu
from scipy.special import expit
from auxiliary import uniform_qmesh
from .electronic import ElectronicVertices

def collision_mode(specification=None, prepared=None):
    """Read mode metadata from exactly one collision input, without building rates."""
    if specification is not None and prepared is not None:
        raise ValueError("Supply collisionSpecification or preparedCollision, not both")
    if prepared is not None:
        if not isinstance(prepared, PreparedCollision):
            raise TypeError("preparedCollision must be a PreparedCollision")
        return prepared.specification.mode
    if specification is None:
        raise TypeError("Supply either CollisionSpecification or PreparedCollision")
    if not isinstance(specification, CollisionSpecification):
        raise TypeError("collisionSpecification must be a CollisionSpecification")
    return specification.mode

def prepare_collision(electronic, specification, *, kBT, epc_model=None):
    """Prepare after Velocity, then pass the result as preparedCollision=.

    specification must be a CollisionSpecification; no mode strings or eta
    defaults are translated here. This explicitly prepares fresh rates.
    DensityMatrixResponse composes this preparation once per response.
    Reuse assumes the model, providers and specification remain unchanged.
    epc_model explicitly pairs electronic and phonon models. It is never
    inferred from inputs. CRTA needs no EPC model.
    """
    return PreparedCollision(CollisionInputs.from_electronic(electronic, kBT=kBT, epc_model=epc_model), specification)

def resolve_collision(electronic, specification=None, *, kBT, prepared=None, epc_model=None):
    """Return preparation from explicit arguments, without modifying electronic."""
    collision_mode(specification, prepared)
    if prepared is None:
        prepared = prepare_collision(electronic, specification, kBT=kBT, epc_model=epc_model)
    else:
        prepared.bind(electronic, kBT=kBT, epc_model=epc_model)
    return prepared

def prepare_phonon_background(electronic, specification, *, kBT, epc_model):
    """CRTA-only path for explicit/static phonons, independent of EPC rates."""
    if not isinstance(specification, CollisionSpecification):
        raise TypeError("Explicit/static phonon calculations require a CRTA CollisionSpecification")
    if specification.mode != "crta":
        raise ValueError("Explicit/static phonon calculations require CRTA background only; an EPC/SERTA collision would double count scattering")
    return CollisionOperator_linewidth(specification, CollisionInputs.from_electronic(electronic, kBT=kBT, epc_model=epc_model))

@dataclass(frozen=True)
class CollisionInputs:
    """Electronic state and optional EPC model, without response outputs.

    The EPC model explicitly identifies the electronic and phonon model pair;
    electronic arrays are shared with ElectronicVertices, avoiding copies.

    The model and arrays must stay unchanged during use; their contents are not
    monitored. Velocity supplies stationary-state response derivatives.
    """
    electronic: ElectronicVertices
    kBT: float
    epc_model: object = None

    @classmethod
    def from_electronic(cls, electronic, *, kBT, epc_model=None):
        if not isinstance(electronic, ElectronicVertices):
            raise TypeError('electronic must be ElectronicVertices')
        if epc_model is not None and epc_model.electronic_model is not electronic.states.model:
            raise ValueError("epcModel must refer to the calculator electronic model")
        return cls(electronic, float(kBT), epc_model)

    @property
    def states(self):
        return self.electronic.states

    @property
    def model(self):
        return self.states.model

    @property
    def kgrid(self):
        return self.states.kgrid

    @property
    def kOffset(self):
        return self.states.kOffset

    @property
    def energies(self):
        return self.states.energies

    @property
    def vectors(self):
        return self.states.vectors

    @property
    def phaseFactorR(self):
        return self.states.phaseFactorR

    @property
    def velocity_K(self):
        return self.electronic.velocity_K

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
    phononBath : PhononBath
        Shared phonon temperature (kelvin), statistics and q/branch selection.
        Its uniform qgrid defaults to the electronic mesh; each dimension
        must divide the corresponding electronic dimension. A selection
        supersedes that grid, preserves absolute weights and must close on
        the electronic mesh. Nothing is read from the physical phonon model
        except its modes and displacement patterns.
    delta : float
        Positive Lorentzian energy-conservation broadening in eV used while
        building EPC transition rates. It is not a relaxation time.
    enforceDetailedBalance : bool
        Pair-balance the broadened transition rates at ``phononBath.temperature``.
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
    transitionStorage : ``'auto'`` | ``'sparse'`` | ``'matrix_free'``
        Full EPC storage. Auto uses a conservative sparse workspace estimate;
        matrix_free recomputes bounded rate blocks and uses iterative solves.
    workspaceBytes, kBatchSize
        Full EPC numerical workspace estimate and optional electronic batch cap.
        Resident models/prepared electronic arrays are separate from this budget.

    """

    def __init__(self, mode:str,
                 constantLinewidth:float=0.06582, linewidthFloor:float=1e-12,
                 linewidthSource=None, phononBath=PhononBath(),
                 delta=.03, populationSolver="auto",
                 solverTolerance=1e-10, stationaryTolerance=1e-12,
                 enforceDetailedBalance=True, secondOrderLinewidthDerivative="frozen",
                 transitionStorage='auto', workspaceBytes=512*1024**2, kBatchSize=None):
        mode=str(mode).lower()
        if mode in ("full", "epc_markov"):
            mode="epc"
        if mode not in ("crta","serta","epc"):
            raise NotImplementedError("Available collision modes are 'crta', 'serta', and 'epc'")
        self.mode=mode
        self.constantLinewidth= float(constantLinewidth)
        self.linewidthFloor=float(linewidthFloor)
        self.linewidthSource=linewidthSource
        if not isinstance(phononBath, PhononBath):
            raise TypeError('phononBath must be PhononBath')
        self.phononBath = phononBath
        self.delta=float(delta)
        self.populationSolver=str(populationSolver).lower()
        self.solverTolerance=float(solverTolerance)
        self.stationaryTolerance=float(stationaryTolerance)
        self.enforceDetailedBalance=bool(enforceDetailedBalance)
        if secondOrderLinewidthDerivative not in ("frozen", "covariant"):
            raise ValueError("secondOrderLinewidthDerivative must be frozen or covariant")
        self.secondOrderLinewidthDerivative=secondOrderLinewidthDerivative
        if transitionStorage not in ('auto', 'sparse', 'matrix_free'):
            raise ValueError('transitionStorage must be auto, sparse or matrix_free')
        self.transitionStorage = transitionStorage
        self.workspaceBytes = workspaceBytes
        self.kBatchSize = kBatchSize
        if self.mode != "serta" and linewidthSource is not None:
            raise ValueError("linewidthSource is only meaningful for SERTA")
        if self.mode == "epc":
            if transitionStorage == 'matrix_free' and populationSolver == 'direct':
                raise ValueError('matrix_free collisions require an iterative population solver')
            if self.populationSolver not in ("auto", "direct", "iterative"):
                raise ValueError("populationSolver must be auto, direct, or iterative")

    @property
    def requirements(self):
        return ComputationRequirements(
            requires_full_mesh=self.mode == 'epc' or self.secondOrderLinewidthDerivative == 'covariant')

    def bind(self, inputs):
        if self.mode == "epc":
            return CollisionOperator_full(self,inputs)
        return CollisionOperator_linewidth(self,inputs)

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
        inputs = self.inputs
        if inputs.model.overlap or inputs.energies.shape[-1] != inputs.model.norbs:
            raise NotImplementedError("Covariant collision derivatives require the complete orthogonal model basis")
        if not hasattr(self, "_mesh_connection"):
            buffer = np.zeros(inputs.kgrid + (3, inputs.model.norbs, inputs.model.norbs), complex)
            for ir, rv in enumerate(inputs.model.iRvec):
                buffer[tuple(np.asarray(rv) % inputs.kgrid)] += inputs.model.AA_R_W[ir]*inputs.phaseFactorR[ir]
            connection = np.fft.ifftn(buffer, axes=(0, 1, 2))*np.prod(inputs.kgrid)
            self._mesh_connection = .5*(connection + connection.conj().swapaxes(-1, -2))
        u = inputs.vectors
        adj = u.conj().swapaxes(-1, -2)
        orbital = u @ matrix @ adj
        coefficients = np.fft.fftn(orbital, axes=(0, 1, 2))
        moment = np.zeros(inputs.kgrid)
        for axis, n in enumerate(inputs.kgrid):
            r = np.fft.fftfreq(n)*n
            if n % 2 == 0:
                r[n//2] = 0.
            shape = [1, 1, 1]
            shape[axis] = n
            moment += r.reshape(shape)*inputs.model.lattice[axis, direction]
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

    def __init__(self, specification:CollisionSpecification, inputs):
        self.specification = specification
        self.inputs = inputs
        self.constant_eta = None
        # CASE crta, return the constant linewidth directly
        if specification.mode == "crta":
            self.constant_eta = specification.constantLinewidth
            self.population_linewidth=np.full(inputs.energies.shape,self.constant_eta,dtype=float)
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
        model = self.inputs.model
        source= self.specification.linewidthSource
        # CASE1, linewidth is set externally, e.g., using model EPC-linewidth
        #        So, it is assumed to be a quantity per band per k due to the serta-form.
        #        Or, it provides populationLinewidth method to compute the linewdith per band per k
        if source is not None:
            linewidth = (source.populationLinewidth(self.inputs) if hasattr(source, "populationLinewidth")
                         else np.asarray(source, dtype=float))
            if linewidth.shape != self.inputs.energies.shape:
                raise ValueError(f"SERTA linewidth source has shape {linewidth.shape}, expected {self.inputs.energies.shape}")
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

        linewidth_sqrt_R = np.asarray(linewidth_sqrt_R) * self.inputs.phaseFactorR[:,None,None]
        linewidth_grid = np.zeros(self.inputs.kgrid + (model.norbs, model.norbs), dtype=complex)
        for iR,Rvec in enumerate(model.iRvec):
            linewidth_grid[tuple(np.asarray(Rvec) % self.inputs.kgrid)] += linewidth_sqrt_R[iR]
        linewidth_sqrt_W = np.fft.ifftn(linewidth_grid, axes=(0,1,2)) * np.prod(self.inputs.kgrid)
        linewidth_W = linewidth_sqrt_W @ linewidth_sqrt_W.conj().swapaxes(-2,-1)
        vectors = self.inputs.vectors
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
    calculator eigensystem, the explicit EPC model, and its phonon model.
    Finite-width rates are pair-balanced by default so the Fermi state obeys
    detailed balance exactly. Population response solves remain in the
    trace-free tangent space. Coherences use secular half-sum damping; its
    population dependence enters D^2 C. This is a population-complete Markov
    collision, not an exact non-Markov phonon memory kernel.
    """

    mesh_coupled=True

    def __init__(self, specification, inputs):
        self.specification=specification
        self.inputs=inputs
        model=inputs.model
        self.epc_model = inputs.epc_model
        if self.epc_model is None:
            raise ValueError("EPC collision requires an explicit epcModel")
        if model.overlap:
            raise NotImplementedError("mesh-global EPC collision currently requires an orthogonal model")
        if inputs.energies.shape[-1] != model.norbs:
            raise NotImplementedError("mesh-global EPC collision requires the complete model basis")
        self.epc_model.validate()
        self.kgrid=np.asarray(inputs.kgrid,dtype=int)
        self.phononSelection = specification.phononBath.selection
        qgrid=self.kgrid if specification.phononBath.qgrid is None else np.asarray(specification.phononBath.qgrid)
        if (qgrid.shape != (3,) or not np.all(np.isfinite(qgrid))
                or np.any(qgrid < 1) or not np.all(qgrid == qgrid.astype(int))):
            raise ValueError("EPC qgrid must contain three positive integers")
        self.qgrid=qgrid.astype(int)
        if self.phononSelection is not None:
            self.phononSelection.mesh_shifts(self.kgrid)
            if len(getattr(self.epc_model.phonon_model, '_cell_origins', [None])) != 1:
                raise ValueError('evaluate selected EPC collisions on the primitive model; the selection uses primitive q and branch labels')
            self.phononSelection.branch_indices(len(self.epc_model.phonon_model.modes(self.phononSelection.qpoints[0])[0]))
        elif np.any(self.kgrid % self.qgrid):
            raise ValueError("each EPC qgrid dimension must divide the calculator kgrid")
        bath_kBT=Boltzmann*specification.phononBath.temperature/electron_volt
        if specification.enforceDetailedBalance:
            if bath_kBT <= 0:
                raise ValueError("detailed-balance EPC rates require a positive phonon temperature")
            if not np.isclose(inputs.kBT,bath_kBT,rtol=2e-6,atol=1e-10):
                raise ValueError("detailed-balance EPC rates require electronic kBT to match phononBath.temperature")
        self.bath_kBT=bath_kBT
        self.qgrid=tuple(self.qgrid)
        self.nk=int(np.prod(self.kgrid))
        self.nbands=inputs.energies.shape[-1]
        self.size=self.nk*self.nbands
        self.transition_rates=self._transition_rates()
        self.reverse_rates=(self.transition_rates.T if isinstance(self.transition_rates, LinearOperator)
                            else self.transition_rates.T.tocsr())
        self._views={}
        self.population_linewidth=None
        self.coherence_linewidth=None

    @staticmethod
    def _lorentzian(value, delta):
        return delta/np.pi/(value*value+delta*delta)

    def _phonon_population(self, energies):
        from .epc_rates import phonon_population
        return phonon_population(energies, self.specification.phononBath)

    def _transition_rates(self):
        """Return R[final,initial] in eV for all mesh-closed EPC processes."""
        from .epc_rates import MeshEPCRates, EPCRateOperator
        inputs=self.inputs
        self.rate_balance_relative_change=0.
        grid=self.kgrid
        if self.phononSelection is None:
            qpoints=uniform_qmesh(self.qgrid)
            qshifts=np.indices(self.qgrid).reshape(3,-1).T*(grid//np.asarray(self.qgrid))
            qweights=np.full(len(qpoints), 1/len(qpoints))
        else:
            qpoints=self.phononSelection.qpoints
            qshifts=self.phononSelection.mesh_shifts(grid)
            qweights=self.phononSelection.weights
        # Include forward/reverse and balancing temporaries in the estimate.
        self.estimatedTransitionBytes = 96*self.size*len(qpoints)*self.nbands+16*(self.size+1)
        requested = self.specification.transitionStorage
        self.transitionStorage = ('sparse' if self.estimatedTransitionBytes <= self.specification.workspaceBytes//2
                                  else 'matrix_free') if requested == 'auto' else requested
        if self.transitionStorage == 'sparse' and self.estimatedTransitionBytes >= self.specification.workspaceBytes:
            raise MemoryError('Sparse EPC rates exceed workspaceBytes; use matrix_free or a larger budget')
        if self.transitionStorage == 'matrix_free' and self.specification.populationSolver == 'direct':
            raise ValueError('matrix_free collisions require an iterative population solver')
        workspace = self.specification.workspaceBytes-(self.estimatedTransitionBytes if self.transitionStorage == 'sparse' else 0)
        self.rate_sampler = MeshEPCRates(inputs, self.specification, qpoints, qshifts, qweights,
                                        workspace=workspace)
        if self.transitionStorage == 'matrix_free':
            result = EPCRateOperator(self.rate_sampler)
            self.rate_scale = result.max_rate_bound
            # Computing the sparse Frobenius diagnostic would defeat bounded
            # storage. Stationarity and iterative residuals remain checked.
            self.rate_balance_relative_change = None if self.specification.enforceDetailedBalance else 0.
            return result
        result=self.rate_sampler.sparse()
        if self.specification.enforceDetailedBalance:
            raw_norm=np.linalg.norm(result.data)
            activity=(result+result.T).tocoo()
            energies=inputs.energies.reshape(-1)
            thermal_fraction=expit(-(energies[activity.row]-energies[activity.col])/self.bath_kBT)
            balanced=csc_matrix((activity.data*thermal_fraction,(activity.row,activity.col)),shape=activity.shape)
            balanced.eliminate_zeros()
            balanced.sort_indices()
            difference=balanced-result
            self.rate_balance_relative_change=(np.linalg.norm(difference.data)/raw_norm
                                               if raw_norm else 0.)
            result=balanced
        self.rate_scale = float(np.max(result.data)) if result.nnz else 0.
        return result

    def population_collision(self, occupations):
        """Evaluate the nonlinear Pauli gain-minus-loss collision integral.

        ``transition_rates[final, initial]`` is the Golden-rule transition
        rate.  This routine is primarily useful for diagnostics; response
        kernels use the Fermi-level linearization returned by
        :meth:`forFermi`.
        """
        values=np.asarray(occupations)
        if values.shape != self.inputs.energies.shape:
            raise ValueError(f"occupations have shape {values.shape}, expected {self.inputs.energies.shape}")
        flat=values.reshape(-1)
        gain=(1-flat)*(self.transition_rates@flat)
        loss=flat*(self.reverse_rates@(1-flat))
        return np.asarray(gain-loss).reshape(values.shape)

    def population_jacobian(self, occupations):
        """Derivative of the EPC gain-minus-loss integral (no reservoir)."""
        p=np.asarray(occupations).reshape(-1)
        r, rt=self.transition_rates,self.reverse_rates
        if isinstance(r, LinearOperator):
            damping = r@p+rt@(1-p)
            result = LinearOperator(r.shape, dtype=np.result_type(p, float),
                matvec=lambda x: (1-p)*(r@x)+p*(rt@x)-damping*x)
            result.population_diagonal = -damping
            return result
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
        self.inputs=parent.inputs
        self.fermi=fermi
        energies=parent.inputs.energies.reshape(-1)
        kBT=parent.inputs.kBT
        if kBT <= 0:
            raise ValueError("mesh-global EPC collision requires kBT > 0 for Pauli linearization")
        self._energies=energies
        self.equilibrium_occupations=expit((fermi-energies)/kBT).reshape(self.inputs.energies.shape)
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
        linewidth=np.asarray(self.population_damping.diagonal()).real.reshape(parent.inputs.energies.shape)
        self.population_linewidth=np.maximum(linewidth,0.)
        p=self.stationary_occupations.reshape(-1)
        outgoing=(self.specification.constantLinewidth+self.parent.transition_rates@p
                  +self.parent.reverse_rates@(1-p)).reshape(self.inputs.energies.shape)
        self.coherence_linewidth=.5*(outgoing[..., :, None]+outgoing[..., None, :])
        self.constant_eta=None

    def _damping(self, populations):
        from .epc_rates import shifted_operator
        jacobian = self.parent.population_jacobian(populations)
        if isinstance(jacobian, LinearOperator):
            operator = LinearOperator(jacobian.shape, dtype=jacobian.dtype, matvec=lambda x: -(jacobian@x))
            return shifted_operator(operator, -jacobian.population_diagonal, self.specification.constantLinewidth)
        return (-jacobian
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
        kBT=self.inputs.kBT
        low=self._energies.min()-64*kBT
        high=self._energies.max()+64*kBT
        for _ in range(80):
            chemical_potential=.5*(low+high)
            occupations=expit((chemical_potential-self._energies)/kBT)
            if occupations.sum() < target:
                low=chemical_potential
            else:
                high=chemical_potential
        return occupations.reshape(self.inputs.energies.shape)

    def _linear_solve(self, operator, source, cache_key=None, constrained=True):
        source=np.asarray(source).reshape(-1)
        dtype=np.result_type(source,operator.dtype,complex)
        if constrained:
            removed=source.sum()
            self.max_projected_source=max(self.max_projected_source,float(abs(removed)))
            source=source-self.number_mode*removed
            n=self.parent.size
            if isinstance(operator, LinearOperator):
                matrix = LinearOperator((n+1, n+1), dtype=complex,
                    matvec=lambda x: np.concatenate((operator@x[:-1]+self.number_mode*x[-1], [x[:-1].sum()/n])))
                diagonal = np.concatenate((operator.population_diagonal, [0.])).astype(complex)
            else:
                matrix=bmat([[operator.astype(complex),csc_matrix(self.number_mode[:,None])],
                             [csc_matrix(np.full((1,n),1/n)),csc_matrix((1,1))]],format="csc")
                diagonal = matrix.diagonal().astype(complex)
            rhs=np.concatenate((source.astype(complex),[0.]))
            factor_key=None if cache_key is None else ("fixed-number",cache_key)
        else:
            matrix=operator if isinstance(operator, LinearOperator) else operator.astype(complex).tocsc()
            diagonal=(operator.population_diagonal.copy() if isinstance(operator, LinearOperator)
                      else matrix.diagonal()).astype(complex)
            rhs=source.astype(complex)
            factor_key=cache_key
        if not np.any(rhs):
            return np.zeros(source.shape,dtype=dtype)
        direct=(not isinstance(matrix, LinearOperator) and
                (self.specification.populationSolver == "direct" or
                 (self.specification.populationSolver == "auto" and self.parent.size <= 512)))
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
            scale=max(self.parent.rate_scale,gamma,1.)
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
        if drive_velocity is not None and not np.array_equal(drive_velocity,self.inputs.velocity_K):
            raise NotImplementedError("Stationary EPC response currently requires the total electric drive vertex")
        from .density_matrix import fermi_divided_difference, diagonal_matrix
        e=self.inputs.energies
        result=fermi_divided_difference(e[..., :,None],e[...,None,:],self.fermi,self.inputs.kBT)
        result=result*self.inputs.velocity_K[...,direction,:,:]
        correction=self.stationary_occupations-self.equilibrium_occupations
        if np.any(correction):
            result=result+self.mesh_derivative(diagonal_matrix(correction),direction)
        return result

    def apply_population(self, values):
        shape=self.inputs.energies.shape
        array=np.asarray(values)
        if array.shape != shape:
            raise ValueError(f"population source has shape {array.shape}, expected {shape}")
        flat=array.reshape(-1)
        result=self.population_damping@flat
        result-=self.specification.constantLinewidth*self.number_mode*flat.sum()
        return np.asarray(result).reshape(shape)

    def solve_population(self, source, frequency=0.0):
        shape=self.inputs.energies.shape
        array=np.asarray(source)
        if array.shape != shape:
            raise ValueError(f"population source has shape {array.shape}, expected {shape}")
        frequency=float(frequency)
        if not np.isfinite(frequency):
            raise ValueError("Population solve frequency must be finite")
        from .epc_rates import shifted_operator
        operator=shifted_operator(self.population_damping, self.population_damping.diagonal()
                                 if not isinstance(self.population_damping, LinearOperator)
                                 else self.population_damping.population_diagonal, -1j*frequency)
        return (1j*self._linear_solve(operator,array.reshape(-1),frequency)).reshape(shape)

    def solve_steady_population(self, source):
        shape=self.inputs.energies.shape
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
        expected=self.inputs.energies.shape+(self.inputs.energies.shape[-1],)
        if array.shape != expected:
            raise ValueError(f"density-matrix source has shape {array.shape}, expected {expected}")
        output=self.coherence_linewidth*array
        indices=np.arange(self.inputs.energies.shape[-1])
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
        indices=np.arange(self.inputs.energies.shape[-1])
        diagonal=np.diagonal(array,axis1=-2,axis2=-1)
        result[...,indices,indices]=self.solve_population(diagonal,frequency)
        return result

    def requireConstant(self, context):
        raise NotImplementedError(f"{context} does not support a mesh-global EPC collision")

class PreparedCollision:
    """Reusable preparation, with cached Fermi views and frequency solves.

    Sequential reuse requires the same unmodified model and exact band gauge
    and mesh. Fermi levels, frequencies and observable projections may change.
    Use a separate prepared object for each concurrent response calculation.
    """
    def __init__(self, inputs, specification:CollisionSpecification):
        self.inputs = inputs
        self.specification = specification
        if specification.mode == "epc" and inputs.velocity_K is None:
            raise ValueError("Prepare EPC response collisions from Velocity, not Energy alone")
        self.operator = specification.bind(inputs)

    @property
    def mesh_coupled(self):
        return self.operator.mesh_coupled

    def bind(self, electronic, *, kBT, epc_model=None):
        """Validate a consumer without rebinding the shared operator's inputs."""
        if epc_model is not None and epc_model is not self.inputs.epc_model:
            raise ValueError("Prepared collision conflicts with epcModel")
        if electronic.states.model is not self.inputs.model:
            raise ValueError("Prepared collision requires the same model object")
        if electronic.states is not self.inputs.states:
            raise ValueError("Prepared collision requires the same ElectronicStates and band basis")
        if kBT != self.inputs.kBT:
            raise ValueError("Prepared collision does not match the electronic temperature")
        return self.operator
