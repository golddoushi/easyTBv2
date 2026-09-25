"""Frequency-domain stochastic Liouville response for a periodic SSH bath.

For one sampled equal-frequency trajectory let theta=nu*t. Solve
  (nu*d_theta + gamma + i[H(theta),.]) rho0 = gamma*f(H(theta)),
  (nu*d_theta + gamma - i*omega + i[H(theta),.]) R_a
      = -i[v_a(theta),rho0] + gamma*D f(H(theta))[v_a(theta)].
R is the coefficient of vector potential, including all resolved phonon
sidebands. The conductivity envelope is i/omega*Tr(R_a*v_b+rho0*w_ba).
Eliminating nonzero harmonics gives a frequency-dependent, nonsecular memory
operator and dressed drive/current vertices, not independent band lifetimes.
"""
import numpy as np
from .response_inputs import ResponseInputs
from .response_results import ConductivityResult
from scipy.special import expit
from scipy.sparse.linalg import LinearOperator, gmres
from scipy.constants import elementary_charge, hbar, angstrom

from .density_matrix import fermi_divided_difference
from elph import SSHTimeOperators
from phonon import EinsteinPhonon
from auxiliary import uniform_qmesh


class HarmonicPhononLiouvillian:
    """Matrix-free Fourier collocation of the conditional electronic DM.

    Matrices are represented in the fixed equilibrium band basis, and theta
    derivatives are spectral. The preconditioner inverts the bare electronic
    Liouvillian at every harmonic; it does not approximate the solved operator.
    """
    def __init__(self, hamiltonian, bare_energies, base_energy, gamma,
                 tolerance=1e-9, maxiter=100, restart=30):
        self.h = np.asarray(hamiltonian)
        self.shape = self.h.shape
        self.size = self.h.size
        self.harmonics = np.fft.fftfreq(len(self.h))*len(self.h)
        self.derivative = 1j*base_energy*self.harmonics[:, None, None]
        self.gaps = bare_energies[:, None]-bare_energies[None, :]
        self.gamma = gamma
        self.tolerance, self.maxiter, self.restart = tolerance, maxiter, restart

    def apply(self, matrix, omega):
        derivative = np.fft.ifft(self.derivative*np.fft.fft(matrix, axis=0), axis=0)
        return derivative + (self.gamma-1j*omega)*matrix + 1j*(self.h @ matrix-matrix @ self.h)

    def solve(self, source, omega=0.):
        norm = np.linalg.norm(source)
        if norm == 0:
            return np.zeros_like(source), dict(omega_eV=float(omega), iterations=0, relative_residual=0.)
        diagonal = self.gamma-1j*omega+self.derivative+1j*self.gaps[None]
        def precondition(flat):
            harmonics = np.fft.fft(flat.reshape(self.shape), axis=0)
            return np.fft.ifft(harmonics/diagonal, axis=0).ravel()
        operator = LinearOperator((self.size, self.size), dtype=complex,
            matvec=lambda x: self.apply(x.reshape(self.shape), omega).ravel())
        preconditioner = LinearOperator(operator.shape, dtype=complex, matvec=precondition)
        iterations = []
        solution, info = gmres(operator, source.ravel(), M=preconditioner,
            x0=precondition(source.ravel()), rtol=self.tolerance, atol=0.,
            restart=self.restart, maxiter=self.maxiter,
            callback=lambda residual: iterations.append(float(residual)), callback_type='pr_norm')
        solution = solution.reshape(self.shape)
        residual = float(np.linalg.norm(self.apply(solution, omega)-source)/norm)
        if info != 0 or residual > max(5*self.tolerance, 1e-12):
            raise RuntimeError(f"Phonon Liouville solve failed: GMRES info={info}, residual={residual:.3g}")
        return solution, dict(omega_eV=float(omega), iterations=len(iterations), relative_residual=residual)


def solve_phonon_conductivity(inputs: ResponseInputs, phononBath, trajectories, seed,
                             phase_points, tolerance, maxiter, restart,
                             damping=0., bath_cycles=8., *, epc_model, collision):
    temperature, statistics = phononBath.temperature, phononBath.statistics
    result = ConductivityResult(np.zeros((len(inputs.fermiList), len(inputs.omegaList), 3, 3), complex))
    if collision.specification.mode != 'crta':
        raise ValueError("Explicit harmonic Liouville dynamics requires CRTA background; adding SERTA would double count EPC")
    gamma = collision.specification.constantLinewidth
    for name, value, minimum in (('phase_points', phase_points, 5), ('trajectories', trajectories, 1),
                                 ('maxiter', maxiter, 1), ('restart', restart, 1)):
        if not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if phase_points % 2 == 0:
        raise ValueError("Use an odd phononPhasePoints to avoid an unpaired Nyquist harmonic")
    damping, bath_cycles = float(damping), float(bath_cycles)
    phonons = epc_model.phonon_model
    if not isinstance(phonons, EinsteinPhonon):
        raise NotImplementedError("This displacement solver currently requires EinsteinPhonon")
    operators = SSHTimeOperators(epc_model, uniform_qmesh(inputs.kgrid) + inputs.kOffset)
    mode_energy = float(phonons.energies.flat[0])
    spawned_seeds = np.random.SeedSequence(seed).spawn(trajectories)
    if damping == 0:
        if not np.allclose(phonons.energies, mode_energy, rtol=1e-12, atol=0):
            raise NotImplementedError("Undamped one-phase Liouville solver requires equal Einstein mode energies")
        base_energy, bath_period = mode_energy, 2*np.pi/mode_energy
        samples = [phononBath.sampleTrajectory(phonons, seed=s)
                   for s in spawned_seeds]
        bath_name = 'prescribed harmonic'
    else:
        bath_period = bath_cycles*2*np.pi/np.min(phonons.energies)
        base_energy = 2*np.pi/bath_period
        max_resolved = ((phase_points-1)//2)*base_energy
        if max_resolved < np.max(phonons.energies):
            required = 2*int(np.ceil(np.max(phonons.energies)/base_energy))+1
            raise ValueError(f"phononPhasePoints must be at least {required} for the damped bath spectrum")
        samples = [phononBath.sampleDampedTrajectory(
            phonons, damping, bath_period, phase_points, seed=s)
            for s in spawned_seeds]
        bath_name = 'stationary periodic damped-Langevin spectrum'
    result.phonon_initial_positions = np.array([s.displacements(0) for s in samples])
    result.phonon_initial_velocities = np.array([s.velocities(0) for s in samples])
    result.phonon_trajectory_cosine = np.array([s.cosine for s in samples])
    result.phonon_trajectory_sine = np.array([s.sine for s in samples])
    result.phonon_energies = samples[0].energies.copy()
    result.phonon_frequency = base_energy
    result.phonon_harmonics = np.fft.fftfreq(phase_points)*phase_points
    result.phonon_dynamics_info = dict(bath=bath_name, statistics=statistics,
        temperature_K=float(temperature), trajectories=int(trajectories), seed=seed,
        electronic_backreaction=False, orbital_centers='fixed equilibrium SSH basis',
        background_linewidth_eV=float(gamma), solver='frequency-domain harmonic Liouville',
        phase_points=int(phase_points), tolerance=float(tolerance),
        phonon_damping_eV=damping, bath_period_hbar_per_eV=float(bath_period),
        bath_cycles=float(1. if damping == 0 else bath_cycles),
        fundamental_energy_eV=float(base_energy))
    result.solver_info = []
    result.sigma_samples = np.zeros((trajectories,)+result.sigma.shape, complex)
    # Phase-resolved conductivity envelopes permit finite-window comparison
    # with TDSE, including real-drive image terms when frequencies coincide.
    result.conductivity_harmonics = np.zeros(
        (trajectories, len(inputs.fermiList), len(inputs.omegaList), phase_points, 3, 3), complex)
    phases = 2*np.pi*np.arange(phase_points)/phase_points
    scale = elementary_charge**2/(hbar*angstrom*abs(inputs.model.Vcell)*operators.nk)
    vectors = inputs.vectors.reshape(operators.nk, operators.nb, operators.nb)
    energies = inputs.energies.reshape(operators.nk, operators.nb)
    for sample_index, trajectory in enumerate(samples):
        sampled = [operators.at_displacement(trajectory.displacements(theta/base_energy)) for theta in phases]
        h_all, v_all, w_all = [np.array(items) for items in zip(*sampled)]
        envelopes = np.zeros((len(inputs.fermiList), len(inputs.omegaList), phase_points, 3, 3), complex)
        for k in range(operators.nk):
            u, udag = vectors[k], vectors[k].conj().T
            h = udag @ h_all[:, k] @ u
            v = udag @ v_all[:, k] @ u
            w = udag @ w_all[:, k] @ u
            liouvillian = HarmonicPhononLiouvillian(
                h, energies[k], base_energy, gamma, tolerance, maxiter, restart)
            e, z = np.linalg.eigh(h)
            zadj = z.conj().swapaxes(-2, -1)
            for fi, fermi in enumerate(inputs.fermiList):
                equilibrium = (z*expit((fermi-e)/inputs.kBT)[:, None, :]) @ zadj
                rho0, info = liouvillian.solve(gamma*equilibrium)
                result.solver_info.append(dict(info, sample=sample_index, k=k, fermi=fi, drive=None))
                df = fermi_divided_difference(e[:, :, None], e[:, None, :], fermi, inputs.kBT)
                for a in range(3):
                    va = v[:, a]
                    equilibrium_derivative = z @ (df*(zadj @ va @ z)) @ zadj
                    source = -1j*(va @ rho0-rho0 @ va) + gamma*equilibrium_derivative
                    contact = np.einsum('pmn,pbnm->pb', rho0, w[:, :, a])
                    for iw, omega in enumerate(inputs.omegaList):
                        response, info = liouvillian.solve(source, float(omega))
                        result.solver_info.append(dict(info, sample=sample_index, k=k, fermi=fi, drive=a))
                        envelope = np.einsum('pmn,pbnm->pb', response, v)+contact
                        envelopes[fi, iw, :, a] += (1j/omega)*envelope*scale
        harmonics = np.fft.fft(envelopes, axis=2)/phase_points
        result.conductivity_harmonics[sample_index] = harmonics
        result.sigma_samples[sample_index] = harmonics[:, :, 0]
    result.sigma[:] = result.sigma_samples.mean(axis=0)
    if trajectories > 1:
        result.sigma_stderr_real = result.sigma_samples.real.std(axis=0, ddof=1)/np.sqrt(trajectories)
        result.sigma_stderr_imag = result.sigma_samples.imag.std(axis=0, ddof=1)/np.sqrt(trajectories)
    else:
        result.sigma_stderr_real = np.full(result.sigma.shape, np.nan)
        result.sigma_stderr_imag = np.full(result.sigma.shape, np.nan)
    return result


def lockin_conductivity(result: ConductivityResult, times, omega, sample=0, ifermi=0, iomega=0):
    """Stationary spectral solution measured with the TDSE real-drive window.

    Includes phonon sideband leakage and the conjugate optical image. No time
    propagation occurs: reconstruct only the nine scalar current envelopes.
    """
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or not len(times) or not np.all(np.isfinite(times)):
        raise ValueError("times must be a nonempty finite 1D array in hbar/eV")
    coefficients = result.conductivity_harmonics[sample, ifermi, iomega]
    output = np.zeros((3, 3), complex)
    for start in range(0, len(times), 4096):
        t = times[start:start+4096]
        basis = np.exp(1j*result.phonon_frequency*t[:, None]*result.phonon_harmonics[None])
        envelope = np.einsum('tp,pab->tab', basis, coefficients)
        current = (envelope*np.exp(-1j*omega*t[:, None, None])).real
        output += np.sum(2*current*np.exp(1j*omega*t[:, None, None]), axis=0)
    return output/len(times)
