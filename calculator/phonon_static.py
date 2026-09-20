"""Static Gaussian EPC correction from equilibrium-basis DM coefficients.

The only eigenstates are the owner's established Energy/Velocity result.
Hamiltonian, Fermi-target, Liouvillian, drive, current and contact vertices
are expanded consistently to second order in Gaussian displacement. This is
a response expansion, not an additional lifetime on top of SERTA.
"""
import numpy as np
from scipy.constants import elementary_charge, hbar, angstrom
from .density_matrix import StaticDisplacementDMKernel
from system.model_epc import SSHTimeOperators


def solve_static_epc_conductivity(owner, temperature, statistics, moving_centers,
                                  factors, batch_size, *, epc_model, collision):
    specification = collision.specification
    if specification.mode != 'crta':
        raise ValueError('Static EPC expansion requires CRTA background only; adding SERTA/full collision would double count EPC')
    gamma = specification.constantLinewidth
    if not np.isfinite(gamma) or gamma <= 0 or not np.isfinite(owner.kBT) or owner.kBT <= 0:
        raise ValueError('Static EPC expansion requires positive finite background linewidth and electronic kBT')
    if statistics not in ('quantum', 'classical', 'wigner'):
        raise ValueError('static phononStatistics must be quantum, wigner or classical')
    if not np.isfinite(temperature) or temperature < 0:
        raise ValueError('phononTemperature must be finite and nonnegative')
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError('kBatchSize must be a positive integer')
    operators = SSHTimeOperators(owner, epc_model)
    phonons = epc_model.phonon_model
    covariance_source = 'model phonons' if factors is None else 'provided covariance factors'
    factors = (phonons.displacementFactors(temperature, quantum=statistics != 'classical')
               if factors is None else np.asarray(factors))
    if np.iscomplexobj(factors) and np.any(factors.imag != 0):
        raise ValueError('displacementFactors must be real')
    factors = np.asarray(factors.real, float)
    if (factors.ndim != 3 or factors.shape[1:] != (len(phonons.masses), 3)
            or not np.isfinite(factors).all()):
        raise ValueError('displacementFactors must be finite real (nmode,natom,3) covariance factors in Angstrom')
    factors = factors[np.any(factors != 0, axis=(1, 2))]
    owner.displacement_factors = factors.copy()
    nk, nb = operators.nk, operators.nb
    energies = owner.energies.reshape(nk, nb)
    vectors = owner.vectors.reshape(nk, nb, nb)
    shape = (len(owner.fermiList), len(owner.omegaList), 3, 3)
    owner.sigma_bare = np.zeros(shape, complex)
    owner.sigma_correction_channels = {name: np.zeros(shape, complex) for name in
        ('density', 'mixed_current', 'quadratic_current', 'contact')}
    owner.sigma_correction_by_mode = np.zeros((len(factors),)+shape, complex)
    owner.sum_rule_residual = np.zeros((len(owner.fermiList), 3, 3), complex)
    owner.correction_sum_rule_residual = np.zeros_like(owner.sum_rule_residual)
    scale = elementary_charge**2/(hbar*angstrom*abs(owner.model.Vcell)*nk)
    batch_size = min(batch_size, max(1, 1_000_000//max(1, nb**4)))

    def velocity_trace(r, v):
        return np.einsum('kaij,kbji->ab', r, v, optimize=True)*scale

    def current(r, v):
        return -velocity_trace(r, v)  # electron current j=-v

    def contact(f, w):
        return np.einsum('kij,kbaji->ab', f, w, optimize=True)*scale

    for start in range(0, nk, batch_size):
        ks = slice(start, min(start+batch_size, nk))
        z = vectors[ks]
        zadj = z.conj().swapaxes(-2, -1)

        def rotate(items):
            h, v, w = items
            return (zadj[None]@h@z[None], zadj[None, :, None]@v@z[None, :, None],
                    zadj[None, :, None, None]@w@z[None, :, None, None])

        # Evaluate the exact equilibrium Peierls vertices once per chunk.
        hzero, vzero, wzero = rotate(operators.displacement_coefficients(
            np.zeros((len(phonons.masses), 3)), ks, moving_centers))
        for fi, fermi in enumerate(owner.fermiList):
            kernel = StaticDisplacementDMKernel(energies[ks], fermi, owner.kBT, gamma)
            f0 = np.zeros_like(hzero[0])
            indices = np.arange(nb)
            f0[:, indices, indices] = kernel.f
            d0 = kernel.dd1[:, None]*vzero[0]
            # The VG pole i/omega*(Tr[Df*v]+Tr[f*w]) already combines
            # the electron-current minus with E=i*omega*A.
            residue0 = velocity_trace(d0, vzero[0])+contact(f0, wzero[0])
            owner.sum_rule_residual[fi] += residue0
            for wi, omega in enumerate(owner.omegaList):
                r0 = d0/(gamma+1j*(kernel.gaps-omega))[:, None]
                owner.sigma_bare[fi, wi] += current(r0, vzero[0])
                if omega != 0:
                    owner.sigma_bare[fi, wi] += 1j*residue0/omega
            for mode, factor in enumerate(factors):
                h, v, w = rotate(operators.displacement_coefficients(factor, ks, moving_centers))
                fs, ds = kernel.coefficients(h[1], h[2], v[0], v[1], v[2])
                residue2 = (velocity_trace(ds[2], v[0])+velocity_trace(ds[1], v[1])+velocity_trace(ds[0], v[2])
                            +contact(fs[2], w[0])+contact(fs[1], w[1])+contact(fs[0], w[2]))
                owner.correction_sum_rule_residual[fi] += residue2
                for wi, omega in enumerate(owner.omegaList):
                    r0, r1, r2 = kernel.response(ds, h[1], h[2], omega)
                    terms = dict(density=current(r2, v[0]), mixed_current=current(r1, v[1]),
                                 quadratic_current=current(r0, v[2]),
                                 contact=1j*residue2/omega if omega != 0 else np.zeros((3, 3), complex))
                    for name, value in terms.items():
                        owner.sigma_correction_channels[name][fi, wi] += value
                        owner.sigma_correction_by_mode[mode, fi, wi] += value
    owner.sigma_correction = sum(owner.sigma_correction_channels.values())
    owner.sigma = owner.sigma_bare+owner.sigma_correction
    owner.projected_sigma = owner.sigma[None, None]
    owner.static_epc_info = dict(solver='static Gaussian density-matrix expansion',
        displacement_order=2, electric_field_order=1, phonon_temperature_K=float(temperature),
        statistics=statistics, moving_orbital_centers=bool(moving_centers),
        covariance_factor_count=len(factors), covariance_source=covariance_source,
        background_linewidth_eV=float(gamma),
        additional_epc_linewidth=False, displacement_finite_differences=False,
        displaced_eigensolves=False, electronic_backreaction=False,
        k_batch_size=batch_size, primitive_phonon_selection=(None if phonons.modeSelection is None
                                                           else phonons.modeSelection.as_dict()))
