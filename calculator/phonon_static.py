"""Static Gaussian EPC correction from equilibrium-basis DM coefficients.

The only eigenstates are the explicitly supplied electronic preparation.
Hamiltonian, Fermi-target, Liouvillian, drive, current and contact vertices
are expanded consistently to second order in Gaussian displacement. This is
a response expansion, not an additional lifetime on top of SERTA.
"""
import numpy as np
from .response_inputs import ResponseInputs
from .response_results import ConductivityResult
from scipy.constants import elementary_charge, hbar, angstrom
from .density_matrix import StaticDisplacementDMKernel
from elph import SSHTimeOperators
from phonon import EinsteinPhonon
from auxiliary import uniform_qmesh


def solve_static_epc_conductivity(inputs: ResponseInputs, temperature, statistics, moving_centers,
                                  factors, batch_size, *, epc_model, collision):
    result = ConductivityResult(np.zeros((len(inputs.fermiList), len(inputs.omegaList), 3, 3), complex))
    specification = collision.specification
    if specification.mode != 'crta':
        raise ValueError('Static EPC expansion requires CRTA background only; adding SERTA/full collision would double count EPC')
    gamma = specification.constantLinewidth
    if statistics not in ('quantum', 'classical', 'wigner'):
        raise ValueError('static phononStatistics must be quantum, wigner or classical')
    phonons = epc_model.phonon_model
    if not isinstance(phonons, EinsteinPhonon):
        raise NotImplementedError("This displacement solver currently requires EinsteinPhonon")
    operators = SSHTimeOperators(epc_model, uniform_qmesh(inputs.kgrid) + inputs.kOffset)
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
    result.displacement_factors = factors.copy()
    nk, nb = operators.nk, operators.nb
    energies = inputs.energies.reshape(nk, nb)
    vectors = inputs.vectors.reshape(nk, nb, nb)
    shape = (len(inputs.fermiList), len(inputs.omegaList), 3, 3)
    result.sigma_bare = np.zeros(shape, complex)
    result.sigma_correction_channels = {name: np.zeros(shape, complex) for name in
        ('density', 'mixed_current', 'quadratic_current', 'contact')}
    result.sigma_correction_by_mode = np.zeros((len(factors),)+shape, complex)
    result.sum_rule_residual = np.zeros((len(inputs.fermiList), 3, 3), complex)
    result.correction_sum_rule_residual = np.zeros_like(result.sum_rule_residual)
    scale = elementary_charge**2/(hbar*angstrom*abs(inputs.model.Vcell)*nk)
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
        for fi, fermi in enumerate(inputs.fermiList):
            kernel = StaticDisplacementDMKernel(energies[ks], fermi, inputs.kBT, gamma)
            f0 = np.zeros_like(hzero[0])
            indices = np.arange(nb)
            f0[:, indices, indices] = kernel.f
            d0 = kernel.dd1[:, None]*vzero[0]
            # The VG pole i/omega*(Tr[Df*v]+Tr[f*w]) already combines
            # the electron-current minus with E=i*omega*A.
            residue0 = velocity_trace(d0, vzero[0])+contact(f0, wzero[0])
            result.sum_rule_residual[fi] += residue0
            for wi, omega in enumerate(inputs.omegaList):
                r0 = d0/(gamma+1j*(kernel.gaps-omega))[:, None]
                result.sigma_bare[fi, wi] += current(r0, vzero[0])
                if omega != 0:
                    result.sigma_bare[fi, wi] += 1j*residue0/omega
            for mode, factor in enumerate(factors):
                h, v, w = rotate(operators.displacement_coefficients(factor, ks, moving_centers))
                fs, ds = kernel.coefficients(h[1], h[2], v[0], v[1], v[2])
                residue2 = (velocity_trace(ds[2], v[0])+velocity_trace(ds[1], v[1])+velocity_trace(ds[0], v[2])
                            +contact(fs[2], w[0])+contact(fs[1], w[1])+contact(fs[0], w[2]))
                result.correction_sum_rule_residual[fi] += residue2
                for wi, omega in enumerate(inputs.omegaList):
                    r0, r1, r2 = kernel.response(ds, h[1], h[2], omega)
                    terms = dict(density=current(r2, v[0]), mixed_current=current(r1, v[1]),
                                 quadratic_current=current(r0, v[2]),
                                 contact=1j*residue2/omega if omega != 0 else np.zeros((3, 3), complex))
                    for name, value in terms.items():
                        result.sigma_correction_channels[name][fi, wi] += value
                        result.sigma_correction_by_mode[mode, fi, wi] += value
    result.sigma_correction = sum(result.sigma_correction_channels.values())
    result.sigma = result.sigma_bare+result.sigma_correction
    result.static_epc_info = dict(solver='static Gaussian density-matrix expansion',
        displacement_order=2, electric_field_order=1, phonon_temperature_K=float(temperature),
        statistics=statistics, moving_orbital_centers=bool(moving_centers),
        covariance_factor_count=len(factors), covariance_source=covariance_source,
        background_linewidth_eV=float(gamma),
        additional_epc_linewidth=False, displacement_finite_differences=False,
        displaced_eigensolves=False, electronic_backreaction=False,
        k_batch_size=batch_size, primitive_phonon_selection=(None if phonons.modeSelection is None
                                                           else phonons.modeSelection.as_dict()))
    return result
