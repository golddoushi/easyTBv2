"""Reusable first- and second-order density-matrix kernels.

The kernels in this module only build density matrices. Observable vertices,
Brillouin-zone integration, projections and output storage remain the
responsibility of the calculator classes that consume these blocks.

All response orders are physical densities, with H_V=H(k+A), E=-dA/dt,
H_L=H0+E*r and exp(-i*omega*t), in hbar=e=1 units. In particular the
electric-field first-order equation is (Gamma-i*omega+i ad_H)rho1=D f.
Only a charge-current observable supplies j=-dH/dA; density kernels never
absorb that minus sign. A second-order block has no extra electron charge.
"""

from dataclasses import dataclass
from types import SimpleNamespace
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

# Compatibility exports; collision preparation lives outside response kernels.
from .collision import (CollisionSpecification, CollisionOperator,
                        CollisionOperator_linewidth, CollisionOperator_full)

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

def fermi_third_divided_difference(x, y, z, w, fermi, kBT):
    """Symmetric f[x,y,z,w], with analytic confluent limits at degeneracy.

    Nearby nodes use the Fermi Taylor series and complete homogeneous
    polynomials. No inverse electronic-gap cutoff or k-neighbor data is used.
    """
    if kBT <= 0:
        raise ValueError('Fermi divided differences require kBT > 0')
    nodes = np.sort(np.stack(np.broadcast_arrays(x, y, z, w)), axis=0)
    low, middle1, middle2, high = nodes
    span = high-low
    nearby = span <= .5*kBT
    result = np.zeros_like(span)
    far = ~nearby
    if np.any(far):
        result[far] = (fermi_second_divided_difference(middle1[far], middle2[far], high[far], fermi, kBT)
                       -fermi_second_divided_difference(low[far], middle1[far], middle2[far], fermi, kBT))/span[far]
    if np.any(nearby):
        center = low[nearby]+.5*span[nearby]
        dx = (nodes[:, nearby]-center)/kBT
        e1 = np.sum(dx, axis=0)
        e2 = sum(dx[i]*dx[j] for i in range(4) for j in range(i+1, 4))
        e3 = sum(dx[i]*dx[j]*dx[l] for i in range(4) for j in range(i+1, 4) for l in range(j+1, 4))
        e4 = np.prod(dx, axis=0)
        t = (center-fermi)/kBT
        f = expit(-np.abs(t))
        coefficients = [f, -f*(1-f)]
        for order in range(1, 22):
            convolution = sum(coefficients[j]*coefficients[order-j] for j in range(1, order))
            coefficients.append(((2*f-1)*coefficients[order]+convolution)/(order+1))
        h0 = np.ones_like(t)
        h1, h2, h3 = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
        value = np.zeros_like(t)
        for order in range(20):
            c = coefficients[order+3]
            if order % 2:
                c = np.where(t < 0, -c, c)
            value += c*h0
            h0, h1, h2, h3 = e1*h0-e2*h1+e3*h2-e4*h3, h0, h1, h2
        result[nearby] = value/kBT**3
    return result


class StaticDisplacementDMKernel:
    """Complete static Gaussian expansion of the CRTA electric response.

    H=H0+x*H1+x^2*H2, v=v0+x*v1+x^2*v2, <x^2>=1. All matrices
    use the equilibrium eigenbasis; H2 is the quadratic coefficient, not the
    second derivative. This expands the same covariant instantaneous-Fermi
    background as the first-order VG kernel. No extra EPC linewidth is added.

    Returned r matrices are physical electric-field density coefficients.
    They obey
      (gamma-i*omega+i ad_H0) r_j = d_j-i[H1,r_(j-1)]-i[H2,r_(j-2)],
    where d_j are coefficients of D f(H(x))[v(x)]. The VG equilibrium pole
    and observable derivatives are contracted by the consuming calculator.
    """

    def __init__(self, energies, fermi, kBT, linewidth, thirdBlockSize=8):
        self.e = np.asarray(energies)
        if self.e.ndim != 2 or not np.isfinite(self.e).all():
            raise ValueError('energies must be finite (nk,nb)')
        if not np.isfinite(kBT) or kBT <= 0 or not np.isfinite(linewidth) or linewidth <= 0:
            raise ValueError('static DM expansion requires positive temperature and background linewidth')
        if not isinstance(thirdBlockSize, (int, np.integer)) or thirdBlockSize < 1:
            raise ValueError('thirdBlockSize must be a positive integer')
        self.fermi, self.kBT, self.gamma = float(fermi), float(kBT), float(linewidth)
        self.f = expit((fermi-self.e)/kBT)
        self.gaps = self.e[:, :, None]-self.e[:, None, :]
        self.dd1 = fermi_divided_difference(self.e[:, :, None], self.e[:, None, :], fermi, kBT)
        self.dd2 = fermi_second_divided_difference(self.e[:, :, None, None],
            self.e[:, None, :, None], self.e[:, None, None, :], fermi, kBT)
        self.third_block_size = thirdBlockSize
        # Cache only small fourth-rank tensors. Large supercells stream the
        # last band index, bounding scratch memory rather than allocating nb^4.
        self._cache_third = len(self.e)*self.e.shape[1]**4 <= 1_000_000
        self.dd3 = None

    def _third(self, columns):
        e = self.e
        return fermi_third_divided_difference(e[:, :, None, None, None],
            e[:, None, :, None, None], e[:, None, None, :, None],
            e[:, None, None, None, columns], self.fermi, self.kBT)

    def second_product(self, a, b):
        return np.einsum('kilj,kil,klj->kij', self.dd2, a, b, optimize=True)

    def coefficients(self, h1, h2, v0, v1, v2):
        """Return equilibrium f0/f1/f2 and electric-source d0/d1/d2.

        v arrays have shape (nk,drive,nb,nb). No displaced eigensolve or
        displacement finite difference occurs in this kernel.
        """
        f0 = diagonal_matrix(self.f)
        f1 = self.dd1*h1
        f2 = self.dd1*h2+self.second_product(h1, h1)
        d0 = self.dd1[:, None]*v0
        d1 = self.dd1[:, None]*v1
        d2 = self.dd1[:, None]*v2
        for a in range(v0.shape[1]):
            d1[:, a] += self.second_product(h1, v0[:, a])+self.second_product(v0[:, a], h1)
            d2[:, a] += (self.second_product(h1, v1[:, a])+self.second_product(v1[:, a], h1)
                          +self.second_product(h2, v0[:, a])+self.second_product(v0[:, a], h2))
        active = [a for a in range(v0.shape[1]) if np.any(v0[:, a])]
        if not np.any(h1) or not active:
            return (f0, f1, f2), (d0, d1, d2)
        if self._cache_third and self.dd3 is None:
            self.dd3 = self._third(slice(None))
        nb = self.e.shape[1]
        block = nb if self.dd3 is not None else self.third_block_size
        for start in range(0, nb, block):
            columns = slice(start, min(start+block, nb))
            dd = self.dd3 if self.dd3 is not None else self._third(columns)
            for a in active:
                v = v0[:, a]
                # D^3 f[H1,H1,v]/2 has three distinct ordered products.
                for left, middle, right in ((h1, h1, v), (h1, v, h1), (v, h1, h1)):
                    d2[:, a, :, columns] += np.einsum('kilpj,kil,klp,kpj->kij',
                        dd, left, middle, right[:, :, columns], optimize=True)
        return (f0, f1, f2), (d0, d1, d2)

    def response(self, sources, h1, h2, omega):
        if not np.isfinite(omega):
            raise ValueError('omega must be finite')
        denominator = self.gamma+1j*(self.gaps-float(omega))
        d0, d1, d2 = sources
        r0 = d0/denominator[:, None]
        r1 = (d1-1j*commutator(h1[:, None], r0))/denominator[:, None]
        r2 = (d2-1j*commutator(h1[:, None], r1)
              -1j*commutator(h2[:, None], r0))/denominator[:, None]
        return r0, r1, r2


@dataclass
class SecondOrderDensityMatrixBlock:
    """Physical densities for one ordered optical tensor component.

    LG blocks multiply electric-field amplitudes; VG second-order blocks
    multiply vector-potential amplitudes. No observable charge is included.
    rho2 is the mixed coefficient for the ordered (-omega,+omega) fields,
    including both drive orderings (not divided by 2!).
    """

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
    """Physical first-order electric-field density, without observable charge.

    The VG linear kernel exposes the pole-subtracted electric-field response;
    vector_potential_density() returns its full physical delta-rho/A instead.
    """

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
            rho=self.collision.solve(-1j*derivative,self.owner.deltaE)
            return FirstOrderDensityMatrixBlock(self.rho0,
                {"I":rho*kernel.diag if kernel.full_covariant_derivative else np.zeros_like(rho),
                 "E":rho*kernel.off_diag})
        rho_e = self.collision.solve(
            1j*self.f_nm*kernel.deltaE_inverse_pure*velocity_direction*kernel.off_diag,
            self.owner.deltaE,
        )*kernel.off_diag
        rho_i = np.zeros_like(rho_e)
        if kernel.full_covariant_derivative:
            velocity_diag = np.diagonal(velocity_direction, axis1=-2, axis2=-1).real
            population = self.collision.solve_steady_population(self.fp*velocity_diag)
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

        rho_E[m,n] = +i (f_n-f_m) v_a[m,n]
                     / (DeltaE_mn (DeltaE_mn-i eta)),
        rho_I[n,n] = (df_n/dE) v_a[n,n] / eta.

    These are physical density responses for H(k+A), E=-dA/dt and H_L=H0+E*r.
    Observables contract as Tr(rho O); charge-current calculators supply j=-v.
    """

    channel_names = DC_CHANNEL_NAMES
    channel_descriptions = DC_CHANNEL_DESCRIPTIONS

    def __init__(self, owner, full_covariant_derivative=False, finite_temperature_occupations=False, *, collision):
        self.owner = owner
        self.full_covariant_derivative = bool(full_covariant_derivative)
        self.finite_temperature_occupations = bool(finite_temperature_occupations)
        self.nbands = owner.energies.shape[-1]
        self.diag = np.eye(self.nbands)
        self.off_diag = 1.0 - self.diag
        self.indices = np.arange(self.nbands)
        self.deltaE_inverse_pure = owner.deltaEInverse_pure
        self.collision = collision
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

    The returned matrices are physical delta-rho/E for H_L=H0+E*r and contain
    no charge-current sign. Full mode uses the Frechet derivative of
    f(H), including its finite degenerate limit, with no inverse-gap cutoff.
    I/E label diagonal/off-diagonal matrices (only their sum is invariant
    under rotations inside an exactly degenerate subspace).
    """

    channel_names = DC_CHANNEL_NAMES
    channel_descriptions = DC_CHANNEL_DESCRIPTIONS

    def __init__(self, owner, full_covariant_derivative=True, finite_temperature_occupations=True, *, collision):
        self.owner = owner
        self.full_covariant_derivative = bool(full_covariant_derivative)
        self.finite_temperature_occupations = bool(finite_temperature_occupations)
        self.collision = collision
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
        self._rho0 = None
        if kernel.full_covariant_derivative:
            e = owner.energies
            self.divided_difference = fermi_divided_difference(
                e[..., :, None], e[..., None, :], fermi, owner.kBT)
        else:
            self.divided_difference = (self.f[..., :, None]-self.f[..., None, :]) * owner.deltaEInverse_pure

    @property
    def rho0(self):
        """Build the equilibrium density matrix only for clients that need it."""
        if self._rho0 is None:
            self._rho0 = diagonal_matrix(self.f)
        return self._rho0

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
        rho = self.collision.solve(-1j*self.state.derivative(direction, drive_velocity),
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

    def __init__(self, owner, full_covariant_derivative=True, finite_temperature_occupations=True, *, collision):
        if not full_covariant_derivative:
            raise ValueError("VG conductivity requires fullCovariantDerivative=True for covariant relaxation")
        super().__init__(owner, full_covariant_derivative, finite_temperature_occupations, collision=collision)

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
            rho=self.collision.solve(-1j*df,self.kernel.owner.deltaE,self.omega)
            return df+1j*self.omega*rho
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

    def __init__(self, owner, full_covariant_derivative=False, finite_temperature_occupations=False, *, collision):
        self.owner = owner
        self.full_covariant_derivative = bool(full_covariant_derivative)
        self.finite_temperature_occupations = bool(finite_temperature_occupations)
        self.nbands = owner.energies.shape[-1]
        self.diag = np.eye(self.nbands)
        self.off_diag = 1.0 - self.diag
        self.indices = np.arange(self.nbands)
        self.deltaE_mn = owner.deltaE
        self.collision = collision
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
                source=-fermi_state.f_nm*r_direction
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
        total=state.collision.solve(-comm_ba-comm_ab,kernel.deltaE_mn)
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
                    -1j*fermi_state.df[..., a, :, :],self.kernel.deltaE_mn,sign*omega)
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
                -state.d2f[..., drive, derivative, :, :] + 1j*commutator(v,self.rho1[(drive,sign)]),
                kernel.deltaE_mn,sign*self.omega)
            source_i += state.collision.solve(
                -state.d2f_i[..., drive, derivative, :, :] + 1j*commutator(v,self.rho1_i[(drive,sign)]),
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

    def __init__(self, owner, full_covariant_derivative=False, finite_temperature_occupations=False, *, collision):
        super().__init__(owner, full_covariant_derivative, finite_temperature_occupations, collision=collision)
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
                    -source_e,self.kernel.deltaE_mn,frequency
                )*self.kernel.off_diag
                if self.kernel.full_covariant_derivative:
                    population=fermi_state.collision.solve_population(
                        -1j*fermi_state.collision.apply_population(fermi_state.dfdk[...,direction,:]),frequency)
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
        source_e = -commutator(v_b, rho1_e_a) - commutator(v_a, rho1_e_b)
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
            source_i = -(
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
                self.rho1[a,sign]=state.collision.solve(-1j*state.df[...,a,:,:],
                    self.kernel.deltaE_mn,sign*self.omega)
                self.reference_rho1[a,sign]=state.reference.collision.solve(
                    -1j*state.reference.df[...,a,:,:],self.kernel.deltaE_mn,sign*self.omega)

    def derivative(self,drive,sign,direction,intraband=False):
        key=(drive,sign,direction,intraband)
        if key not in self.derivatives:
            state,kernel=self.state,self.kernel
            mask=kernel.diag if intraband else 1.
            reference=self.reference_rho1[drive,sign]*mask
            d2f=state.reference.d2f_i if intraband else state.reference.d2f
            analytic=state.reference.collision.solve(
                -1j*d2f[...,drive,direction,:,:]
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
        total=state.collision.solve(-1j*(db_left+da_right+nonlinear),kernel.deltaE_mn)
        # Nonlinear terms contain at least one population perturbation and are
        # reported with the intraband-origin channels; totals are independent
        # of this reporting choice.
        from_i=state.collision.solve(-1j*(db_left_i+da_right_i+nonlinear),kernel.deltaE_mn)
        from_e=total-from_i
        channels={"II":from_i*kernel.diag,"EI":from_i*kernel.off_diag,
                  "IE":from_e*kernel.diag,"EE":from_e*kernel.off_diag}
        if state.velocity_gauge:
            for name in channels:
                channels[name]*=self.omega**2
            cross_i=-1j*self.omega*(db_left_i-da_right_i)+state.d2f[...,a,b,:,:]
            cross_e=-1j*self.omega*(db_left-da_right)+1j*self.omega*(db_left_i-da_right_i)
            channels['II']+=cross_i*kernel.diag
            channels['EI']+=cross_i*kernel.off_diag
            channels['IE']+=cross_e*kernel.diag
            channels['EE']+=cross_e*kernel.off_diag
            total=sum(channels.values())
            # Physical VG density: delta-rho/A = D rho0 + i*frequency*rho_LG.
            left=state.df[...,a,:,:]-1j*self.omega*left
            right=state.df[...,b,:,:]+1j*self.omega*right
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
            -1j*state.df[..., a, :, :], self.kernel.deltaE_mn, sign*self.omega)
            for a in range(3) for sign in (-1, 1)}

    def derivative(self, drive, sign, direction, intraband=False):
        key = (drive, sign, direction, intraband)
        if key not in self.derivatives:
            state, kernel = self.state, self.kernel
            rho = self.rho1[drive, sign]*(kernel.diag if intraband else 1.)
            d2f = state.d2f_i if intraband else state.d2f
            dg = state.damping_derivative[..., direction, :, :]
            # A=ad_H-Omega-i C: D rho=A^-1(-i D^2 f-[v,rho]+i (D C)[rho]).
            source = (-1j*d2f[..., drive, direction, :, :]
                      - commutator(state.owner.velocity_K[..., direction, :, :], rho)
                      + .5j*(dg @ rho + rho @ dg))
            self.derivatives[key] = state.collision.solve(source, kernel.deltaE_mn, sign*self.omega)
        return self.derivatives[key]
