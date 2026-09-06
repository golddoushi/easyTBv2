"""Frequency-domain stochastic Liouville response for a harmonic SSH bath.

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
from scipy.special import expit
from scipy.sparse.linalg import LinearOperator, gmres
from scipy.constants import elementary_charge, hbar, angstrom

from .density_matrix import fermi_divided_difference
from system.model_epc import SSHTimeOperators


class HarmonicPhononLiouvillian:
    """Matrix-free Fourier collocation of the conditional electronic DM.

    Matrices are represented in the fixed equilibrium band basis, and theta
    derivatives are spectral. The preconditioner inverts the bare electronic
    Liouvillian at every harmonic; it does not approximate the solved operator.
    """
    def __init__(self, hamiltonian, bare_energies, phonon_energy, gamma,
                 tolerance=1e-9, maxiter=100, restart=30):
        self.h = np.asarray(hamiltonian)
        self.shape = self.h.shape
        self.size = self.h.size
        self.harmonics = np.fft.fftfreq(len(self.h))*len(self.h)
        self.derivative = 1j*phonon_energy*self.harmonics[:, None, None]
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


def solve_phonon_conductivity(owner, temperature, statistics, trajectories, seed,
                             phase_points, tolerance, maxiter, restart):
    if owner.collision.specification.mode != 'crta':
        raise ValueError("Explicit harmonic Liouville dynamics requires CRTA background; adding SERTA would double count EPC")
    gamma = owner.collision.specification.constantLinewidth
    if (not np.isfinite(gamma) or gamma <= 0 or not np.isfinite(tolerance) or tolerance <= 0
            or not np.isfinite(temperature) or temperature < 0):
        raise ValueError("Require positive finite background linewidth/tolerance and nonnegative temperature")
    for name, value, minimum in (('phase_points', phase_points, 5), ('trajectories', trajectories, 1),
                                 ('maxiter', maxiter, 1), ('restart', restart, 1)):
        if not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if phase_points % 2 == 0:
        raise ValueError("Use an odd phononPhasePoints to avoid an unpaired Nyquist harmonic")
    operators = SSHTimeOperators(owner)
    phonons = owner.model.modelPhonons
    nu = float(phonons.energies.flat[0])
    if not np.allclose(phonons.energies, nu, rtol=1e-12, atol=0):
        raise NotImplementedError("One-phase Liouville solver requires equal Einstein mode energies; dispersive modes need multiple phases")
    samples = [phonons.sampleTrajectory(temperature, seed=s, statistics=statistics)
               for s in np.random.SeedSequence(seed).spawn(trajectories)]
    owner.phonon_initial_positions = np.array([s.cosine for s in samples])
    owner.phonon_initial_velocities = np.array([s.velocities(0) for s in samples])
    owner.phonon_energies = samples[0].energies.copy()
    owner.phonon_frequency = nu
    owner.phonon_harmonics = np.fft.fftfreq(phase_points)*phase_points
    owner.phonon_dynamics_info = dict(bath='prescribed harmonic', statistics=statistics,
        temperature_K=float(temperature), trajectories=int(trajectories), seed=seed,
        electronic_backreaction=False, orbital_centers='fixed equilibrium SSH basis',
        background_linewidth_eV=float(gamma), solver='frequency-domain harmonic Liouville',
        phase_points=int(phase_points), tolerance=float(tolerance))
    owner.solver_info = []
    owner.sigma_samples = np.zeros((trajectories,)+owner.sigma.shape, complex)
    # Phase-resolved conductivity envelopes permit finite-window comparison
    # with TDSE, including real-drive image terms when frequencies coincide.
    owner.conductivity_harmonics = np.zeros(
        (trajectories, len(owner.fermiList), len(owner.omegaList), phase_points, 3, 3), complex)
    phases = 2*np.pi*np.arange(phase_points)/phase_points
    scale = elementary_charge**2/(hbar*angstrom*abs(owner.model.Vcell)*operators.nk)
    vectors = owner.vectors.reshape(operators.nk, operators.nb, operators.nb)
    energies = owner.energies.reshape(operators.nk, operators.nb)
    for sample_index, trajectory in enumerate(samples):
        sampled = [operators.at_displacement(trajectory.displacements(theta/nu)) for theta in phases]
        h_all, v_all, w_all = [np.array(items) for items in zip(*sampled)]
        envelopes = np.zeros((len(owner.fermiList), len(owner.omegaList), phase_points, 3, 3), complex)
        for k in range(operators.nk):
            u, udag = vectors[k], vectors[k].conj().T
            h = udag @ h_all[:, k] @ u
            v = udag @ v_all[:, k] @ u
            w = udag @ w_all[:, k] @ u
            liouvillian = HarmonicPhononLiouvillian(h, energies[k], nu, gamma, tolerance, maxiter, restart)
            e, z = np.linalg.eigh(h)
            zadj = z.conj().swapaxes(-2, -1)
            for fi, fermi in enumerate(owner.fermiList):
                equilibrium = (z*expit((fermi-e)/owner.kBT)[:, None, :]) @ zadj
                rho0, info = liouvillian.solve(gamma*equilibrium)
                owner.solver_info.append(dict(info, sample=sample_index, k=k, fermi=fi, drive=None))
                df = fermi_divided_difference(e[:, :, None], e[:, None, :], fermi, owner.kBT)
                for a in range(3):
                    va = v[:, a]
                    equilibrium_derivative = z @ (df*(zadj @ va @ z)) @ zadj
                    source = -1j*(va @ rho0-rho0 @ va) + gamma*equilibrium_derivative
                    contact = np.einsum('pmn,pbnm->pb', rho0, w[:, :, a])
                    for iw, omega in enumerate(owner.omegaList):
                        response, info = liouvillian.solve(source, float(omega))
                        owner.solver_info.append(dict(info, sample=sample_index, k=k, fermi=fi, drive=a))
                        envelope = np.einsum('pmn,pbnm->pb', response, v)+contact
                        envelopes[fi, iw, :, a] += (1j/omega)*envelope*scale
        harmonics = np.fft.fft(envelopes, axis=2)/phase_points
        owner.conductivity_harmonics[sample_index] = harmonics
        owner.sigma_samples[sample_index] = harmonics[:, :, 0]
    owner.sigma[:] = owner.sigma_samples.mean(axis=0)
    if trajectories > 1:
        owner.sigma_stderr_real = owner.sigma_samples.real.std(axis=0, ddof=1)/np.sqrt(trajectories)
        owner.sigma_stderr_imag = owner.sigma_samples.imag.std(axis=0, ddof=1)/np.sqrt(trajectories)
    else:
        owner.sigma_stderr_real = np.full(owner.sigma.shape, np.nan)
        owner.sigma_stderr_imag = np.full(owner.sigma.shape, np.nan)


def lockin_conductivity(owner, times, sample=0, ifermi=0, iomega=0):
    """Stationary spectral solution measured with the TDSE real-drive window.

    Includes phonon sideband leakage and the conjugate optical image. No time
    propagation occurs: reconstruct only the nine scalar current envelopes.
    """
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or not len(times) or not np.all(np.isfinite(times)):
        raise ValueError("times must be a nonempty finite 1D array in hbar/eV")
    coefficients = owner.conductivity_harmonics[sample, ifermi, iomega]
    omega = owner.omegaList[iomega]
    output = np.zeros((3, 3), complex)
    for start in range(0, len(times), 4096):
        t = times[start:start+4096]
        basis = np.exp(1j*owner.phonon_frequency*t[:, None]*owner.phonon_harmonics[None])
        envelope = np.einsum('tp,pab->tab', basis, coefficients)
        current = (envelope*np.exp(-1j*omega*t[:, None, None])).real
        output += np.sum(2*current*np.exp(1j*omega*t[:, None, None]), axis=0)
    return output/len(times)
