"""Reusable first- and second-order density-matrix kernels.

The kernels in this module only build density matrices. Observable vertices,
Brillouin-zone integration, projections and output storage remain the
responsibility of the calculator classes that consume these blocks.
"""

from dataclasses import dataclass
import numpy as np

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

def commutator(left, right):
    return left @ right - right @ left

def diagonal_matrix(values):
    """Embed band populations in a diagonal density matrix."""
    nbands = values.shape[-1]
    matrix = np.zeros(values.shape + (nbands,), dtype=complex)
    indices = np.arange(nbands)
    matrix[..., indices, indices] = values
    return matrix

@dataclass
class SecondOrderDensityMatrixBlock:
    """Density matrices for one ordered optical tensor component."""

    rho0: np.ndarray
    rho1_minus: dict
    rho1_plus: dict
    rho2: dict

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
            population = -self.fp * velocity_diag / self.owner.eta
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
        self.g0_mn = owner.deltaEInverse

        if self.full_covariant_derivative and owner.kBT <= 0:
            raise ValueError("fullCovariantDerivative requires kBT > 0")
        if self.full_covariant_derivative and abs(owner.eta) < 1e-15:
            raise ValueError("fullCovariantDerivative requires finite eta for the dc intraband density matrix")

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
        self.g0_mn = 1.0 / (self.deltaE_mn - 1j * owner.eta)

        if self.full_covariant_derivative and owner.kBT <= 0:
            raise ValueError("fullCovariantDerivative requires kBT > 0")
        if self.full_covariant_derivative and abs(owner.eta) < 1e-15:
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
                denominator = self.kernel.deltaE_mn - frequency - 1j * self.owner.eta
                self.factor[sign] = fermi_state.f_nm / denominator
                self.rho1_e[(direction, sign)] = r_direction * self.factor[sign] * self.kernel.off_diag
                if self.kernel.full_covariant_derivative:
                    population = 1j * fermi_state.dfdk[..., direction, :] / (-frequency - 1j * self.owner.eta)
                    self.rho1_i[(direction, sign)] = diagonal_matrix(population)

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
        }

        if kernel.full_covariant_derivative:
            rho1_i_a = self.rho1_i[(a, -1)]
            rho1_i_b = self.rho1_i[(b, 1)]
            source_ii = (
                -state.d2fdk2[..., b, a, :] / (self.omega - 1j * owner.eta)
                -state.d2fdk2[..., a, b, :] / (-self.omega - 1j * owner.eta)
            )
            rho2["II"][..., kernel.indices, kernel.indices] = source_ii / (-1j * owner.eta)

            population_a = np.diagonal(rho1_i_a, axis1=-2, axis2=-1)
            population_b = np.diagonal(rho1_i_b, axis1=-2, axis2=-1)
            source_ei = (
                  r_b * (population_a[..., None, :] - population_a[..., :, None])
                + r_a * (population_b[..., None, :] - population_b[..., :, None])
            )
            rho2["EI"] = kernel.g0_mn * source_ei * kernel.off_diag

            denominator_minus = kernel.deltaE_mn + self.omega - 1j * owner.eta
            denominator_plus = kernel.deltaE_mn - self.omega - 1j * owner.eta
            delta_f_b = state.dfdk[..., b, None, :] - state.dfdk[..., b, :, None]
            delta_f_a = state.dfdk[..., a, None, :] - state.dfdk[..., a, :, None]
            delta_v_b = owner.energy_deriv1_K[..., b, :, None] - owner.energy_deriv1_K[..., b, None, :]
            delta_v_a = owner.energy_deriv1_K[..., a, :, None] - owner.energy_deriv1_K[..., a, None, :]
            factor_deriv_ba = delta_f_b / denominator_minus - state.f_nm * delta_v_b / denominator_minus**2
            factor_deriv_ab = delta_f_a / denominator_plus - state.f_nm * delta_v_a / denominator_plus**2
            rho1_e_a_deriv_b = (owner.r_deriv_K[..., b, a, :, :] * self.factor[-1] + r_a * factor_deriv_ba) * kernel.off_diag
            rho1_e_b_deriv_a = (owner.r_deriv_K[..., a, b, :, :] * self.factor[ 1] + r_b * factor_deriv_ab) * kernel.off_diag
            source_ee = 1j * rho1_e_a_deriv_b + comm_ba + 1j * rho1_e_b_deriv_a + comm_ab
            rho2["EE"] = kernel.g0_mn * source_ee * kernel.off_diag
        else:
            rho1_i_a = rho1_i_b = zero
            rho2["EE"] = kernel.g0_mn * (comm_ba + comm_ab) * kernel.off_diag

        return SecondOrderDensityMatrixBlock(
            rho0=state.rho0,
            rho1_minus={"I": rho1_i_a, "E": rho1_e_a},
            rho1_plus={"I": rho1_i_b, "E": rho1_e_b},
            rho2=rho2,
        )

class LengthGaugeFermiState(_SecondOrderDMFermiState):
    def at_frequency(self, omega): 
        return LengthGaugeFrequencyState(self, omega)

class LengthGaugeSecondOrderDMKernel(_SecondOrderDMKernel):
    """Length-gauge Liouville kernel based on the covariant derivative."""

    state_class = LengthGaugeFermiState

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
                    self.owner.eta * self.owner.r_K[..., direction, :, :]
                    * fermi_state.f_nm * self.kernel.off_diag
                )
            for sign in (-1, 1):
                frequency = sign * omega
                self.rho1_e[(direction, sign)] = (
                    source_e
                    / (self.kernel.deltaE_mn - frequency - 1j * self.owner.eta)
                    * self.kernel.off_diag
                )
                if self.kernel.full_covariant_derivative:
                    population = (
                        1j * self.owner.eta * fermi_state.dfdk[..., direction, :]
                        / (-frequency - 1j * self.owner.eta)
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
            source = 0.5j * self.owner.eta * (c2_ab[name] + c2_ba[name])
            sector = kernel.diag if name in ("II", "IE") else kernel.off_diag
            result[name] = kernel.g0_mn * source * sector
        return result

class VelocityGaugeSecondOrderDMKernel(_SecondOrderDMKernel):
    """Velocity-gauge Liouville kernel with optional covariant relaxation."""

    state_class = VelocityGaugeFermiState
