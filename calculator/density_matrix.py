"""Reusable first- and second-order density-matrix kernels.

The kernels in this module only build density matrices. Observable vertices,
Brillouin-zone integration, projections and output storage remain the
responsibility of the calculator classes that consume these blocks.
"""

from dataclasses import dataclass
import numpy as np
from scipy.special import expit

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

class CollisionOperator:
    """Collision model used by density-matrix kernels.

    ``crta`` uses one relaxation energy ``eta=hbar/tau``.  ``serta`` reads the
    Wannier-interpolated electron-phonon linewidth attached to ``System_w90``.
    A bound collision object is created separately for each calculator k grid,
    which keeps this specification safe to share between BZ worker processes.
    """

    def __init__(self, mode="crta", eta=None, linewidthFloor=1e-12):
        mode=str(mode).lower()
        if mode not in ("crta","serta"):
            raise NotImplementedError("By now, only 'crta', 'serta' have been built")
        self.mode=mode
        self.eta=None if eta is None else float(eta)
        self.linewidthFloor=float(linewidthFloor)
        assert self.linewidthFloor >= 0, f"{linewidthFloor=} must be non-negative instead"

    def bind(self, owner):
        return BoundCollisionOperator(self,owner)

    @classmethod
    def fromOwner(cls, owner):
        specification=getattr(owner,"collisionOperator",None)
        if specification is None:
            specification=cls("crta",eta=owner.eta)
        elif isinstance(specification,str):
            specification=cls(specification)
        elif not isinstance(specification,cls):
            raise TypeError("collisionOperator must be None, a mode string, or CollisionOperator")
        return specification.bind(owner)


class BoundCollisionOperator:
    """Collision linewidths represented on one calculator's eigenstate grid."""

    def __init__(self, specification, owner):
        self.specification=specification
        self.owner=owner
        self.mode=specification.mode
        self.constant_eta=None
        if self.mode == "crta":
            eta=owner.eta if specification.eta is None else specification.eta
            if eta < 0: raise ValueError(f"CRTA {eta=} must be non-negative instead")
            self.constant_eta=float(eta)
            self.population_linewidth=np.full(owner.energies.shape,self.constant_eta,dtype=float)
        elif self.mode == "serta":
            self.population_linewidth=self._sertaLinewidth()
        else:
            raise NotImplementedError("By now, only 'crta', 'serta' have been built")
        self.coherence_linewidth=0.5*(self.population_linewidth[..., :, None]
                                      + self.population_linewidth[..., None, :])

    def _sertaLinewidth(self):
        model=self.owner.model
        linewidth_sqrt_R=getattr(model,"electronLinewidthSqrt_R",None)
        if linewidth_sqrt_R is None:
            linewidth_sqrt_R=getattr(model,"electron_linewidth_sqrt_R",None)
        if linewidth_sqrt_R is None:
            raise ValueError("SERTA requires model.includeElectronSelfEnergy(...) before constructing the calculator")
        if model.overlap:
            raise NotImplementedError("SERTA linewidth interpolation is currently implemented for orthogonal Wannier models")

        linewidth_sqrt_R=np.asarray(linewidth_sqrt_R)
        if self.owner.phase:
            linewidth_sqrt_R=linewidth_sqrt_R*self.owner.phaseshift[:,None,None]
        linewidth_grid=np.zeros(self.owner.kgrid+(model.norbs,model.norbs),dtype=complex)
        for iR,Rvec in enumerate(model.iRvec):
            # Wigner-Seitz boundary vectors can share the same FFT-grid index.
            # Their 1/Ndegen contributions must be accumulated, not overwritten.
            linewidth_grid[tuple(Rvec)]+=linewidth_sqrt_R[iR]
        linewidth_sqrt_W=np.fft.ifftn(linewidth_grid,axes=(0,1,2))*np.prod(self.owner.kgrid)
        linewidth_W=linewidth_sqrt_W@linewidth_sqrt_W.conj().swapaxes(-2,-1)
        vectors=self.owner.vectors
        linewidth_H=vectors.conj().swapaxes(-2,-1)@linewidth_W@vectors
        linewidth=np.diagonal(linewidth_H,axis1=-2,axis2=-1).real
        return np.maximum(linewidth,0.0)

    @property
    def population_inverse(self):
        inverse=np.zeros_like(self.population_linewidth)
        np.divide(1.0,self.population_linewidth,out=inverse,
                  where=self.population_linewidth > self.specification.linewidthFloor)
        return inverse

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
    def __init__(self, kernel, f, fp=None):
        self.kernel = kernel
        self.owner = kernel.owner
        self.f = f
        self.fp = fp
        self.rho0 = diagonal_matrix(f)
        self.f_nm = f[..., None, :] - f[..., :, None]

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
        rho_e = (
            1j * self.f_nm * kernel.deltaE_inverse_pure * kernel.g0_mn
            * velocity_direction * kernel.off_diag
        )
        rho_i = np.zeros_like(rho_e)
        if kernel.full_covariant_derivative:
            velocity_diag = np.diagonal(velocity_direction, axis1=-2, axis2=-1).real
            population = -self.fp * velocity_diag * kernel.collision.population_inverse
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

        rho_E[m,n] = i (f_n-f_m) v_a[m,n]
                     / (DeltaE_mn (DeltaE_mn-i eta)),
        rho_I[n,n] = -(df_n/dE) v_a[n,n] / eta.
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
        self.collision = CollisionOperator.fromOwner(owner)
        self.g0_mn = self.collision.resolvent(owner.deltaE)

        if self.full_covariant_derivative and owner.kBT <= 0:
            raise ValueError("fullCovariantDerivative requires kBT > 0")
        if self.full_covariant_derivative and not np.any(
                self.collision.population_linewidth > self.collision.specification.linewidthFloor):
            raise ValueError("fullCovariantDerivative requires finite population relaxation")

    def fermi_state(self, ifermi, fermi):
        f, fp = self.occupations(ifermi, fermi)
        return LengthGaugeDCFirstOrderFermiState(self, f, fp)

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

class _SecondOrderDMFrequencyState:
    '''Abstract of the 2nd order density matrix at specific real frequency'''
    def block(self) -> SecondOrderDensityMatrixBlock:
        raise NotImplementedError("The block function for this kernel is not implemented in the frequency state.")

class _SecondOrderDMFermiState:
    '''Abstract of the 2nd order density matrix at specific Fermi energy'''
    def __init__(self, kernel, f, dfdk=None, d2fdk2=None):
        self.kernel = kernel
        self.owner = kernel.owner
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
        self.collision = CollisionOperator.fromOwner(owner)
        self.eta = self.collision.requireConstant("second-order density-matrix kernels")
        self.g0_mn = self.collision.resolvent(self.deltaE_mn)

        if self.full_covariant_derivative and owner.kBT <= 0:
            raise ValueError("fullCovariantDerivative requires kBT > 0")
        if self.full_covariant_derivative and abs(self.eta) < 1e-15:
            raise ValueError("fullCovariantDerivative requires finite eta for the dc density matrix")

        self.state_class : _SecondOrderDMFermiState

    @property
    def active_rho2_channels(self):
        """Density-matrix channels that can be nonzero for this mode."""
        return self.channel_names if self.full_covariant_derivative else ("IE", "EE")

    def fermi_state(self, ifermi, fermi) -> _SecondOrderDMFermiState:
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
        return self.state_class(self, f, dfdk, d2fdk2)

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
                self.factor[sign] = fermi_state.f_nm * self.kernel.collision.resolvent(
                    self.kernel.deltaE_mn,frequency
                )
                self.rho1_e[(direction, sign)] = r_direction * self.factor[sign] * self.kernel.off_diag

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
        rho2 = {
            "II": zero,
            "EI": zero,
            "IE": kernel.g0_mn * (comm_ba + comm_ab) * kernel.diag,
            "EE": kernel.g0_mn * (comm_ba + comm_ab) * kernel.off_diag,
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
        super().__init__(kernel, owner.fdOccupation(fermi))
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
    """Differentiate (ad_H - s*Omega - i*eta) rho1 = i D f(H)."""

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
            self.resolvent[sign] = self.kernel.collision.resolvent(self.kernel.deltaE_mn,sign*omega)
            for a in range(3):
                rho = 1j * self.resolvent[sign] * fermi_state.df[..., a, :, :]
                self.rho1[(a, sign)] = rho
                self.rho1_i[(a, sign)] = rho * self.kernel.intra_mask
                self.rho1_e[(a, sign)] = rho * self.kernel.inter_mask

    def block(self, a, b):
        kernel, state = self.kernel, self.fermi_state
        source = np.zeros_like(self.rho1[(a, -1)])
        source_i = np.zeros_like(source)
        for drive, derivative, sign in ((a, b, -1), (b, a, 1)):
            v = self.owner.velocity_K[..., derivative, :, :]
            source += self.resolvent[sign] * (
                -state.d2f[..., drive, derivative, :, :] - 1j * commutator(v, self.rho1[(drive, sign)])
            )
            source_i += self.resolvent[sign] * (
                -state.d2f_i[..., drive, derivative, :, :] - 1j * commutator(v, self.rho1_i[(drive, sign)])
            )
        total = kernel.g0_mn * source
        from_i = kernel.g0_mn * source_i
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
                source_e += (
                    self.kernel.eta * self.owner.r_K[..., direction, :, :]
                    * fermi_state.f_nm * self.kernel.off_diag
                )
            for sign in (-1, 1):
                frequency = sign * omega
                self.rho1_e[(direction, sign)] = (
                    source_e
                    * self.kernel.collision.resolvent(self.kernel.deltaE_mn,frequency)
                    * self.kernel.off_diag
                )
                if self.kernel.full_covariant_derivative:
                    population = (
                        1j * self.kernel.eta * fermi_state.dfdk[..., direction, :]
                        / (-frequency - 1j * self.kernel.eta)
                    )
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
        rho2 = {
            "II": zero,
            "IE": kernel.g0_mn * source_e * kernel.diag,
            "EE": kernel.g0_mn * (source_e + source_contact) * kernel.off_diag,
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
            rho2["EI"] = kernel.g0_mn * source_i * kernel.off_diag
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
            source = 0.5j * kernel.eta * (c2_ab[name] + c2_ba[name])
            sector = kernel.diag if name in ("II", "IE") else kernel.off_diag
            result[name] = kernel.g0_mn * source * sector
        return result

class VelocityGaugeSecondOrderDMKernel(_SecondOrderDMKernel):
    """Velocity-gauge Liouville kernel with optional covariant relaxation."""

    state_class = VelocityGaugeFermiState
