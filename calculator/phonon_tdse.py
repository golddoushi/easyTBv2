"""Real-time linear optical response in a prescribed harmonic SSH bath.

The electronic step is the derivative of an exact frozen-H TDSE/CRTA step.
It propagates the zero-field density and its field derivative together; it
does not use a frequency-domain response kernel or insert an EPC linewidth.
"""
import numpy as np
from scipy.special import expit
from scipy.constants import elementary_charge, hbar, angstrom

from .density_matrix import fermi_divided_difference
from system.model_epc import SSHTimeOperators

def linear_tdse_step(rho, response, hamiltonian, velocity, vector_potential,
                     dt, linewidth, fermi, kBT):
    """Exact linearization of a frozen midpoint step, with scalar CRTA.

    rho[k,i,j], response[a,k,i,j], velocity[k,a,i,j]; response is per unit
    peak electric field and vector_potential is A/E0.  All directions share
    one eigensolve. Degenerate exponential/Fermi derivatives are finite.
    """
    energies, vectors = np.linalg.eigh(hamiltonian)
    adjoint = vectors.conj().swapaxes(-2, -1)
    r = adjoint @ rho @ vectors
    s = adjoint[None] @ response @ vectors[None]
    v = adjoint[None] @ velocity.swapaxes(0, 1) @ vectors[None]
    phases = np.exp(-1j*energies*dt)
    evolution = phases[..., :, None]*phases.conj()[..., None, :]
    # Frechet derivative of exp(-i H dt), evaluated without inverse-gap cutoffs.
    gaps = energies[..., :, None]-energies[..., None, :]
    midpoint = .5*(energies[..., :, None]+energies[..., None, :])
    derivative_u = (-1j*dt*np.exp(-1j*midpoint*dt)
                    * np.sinc(gaps*dt/(2*np.pi)))[None]*v
    first = (derivative_u @ r[None])*phases.conj()[None, ..., None, :]
    derivative_evolution = first + first.conj().swapaxes(-2, -1)
    f = expit((fermi-energies)/kBT)
    df = fermi_divided_difference(
        energies[..., :, None], energies[..., None, :], fermi, kBT)
    decay = np.exp(-linewidth*dt)
    s = decay*(s*evolution[None] + vector_potential*derivative_evolution)
    s += (-np.expm1(-linewidth*dt))*vector_potential*df[None]*v
    r = decay*r*evolution
    indices = np.arange(energies.shape[-1])
    r[..., indices, indices] += (-np.expm1(-linewidth*dt))*f
    return vectors @ r @ adjoint, vectors[None] @ s @ adjoint[None]

def propagate_phonon_conductivity(owner, *, n_steps_per_cycle, n_cycles_measure,
                                 transient_lifetimes, max_phase_step, recordCurrent,
                                 temperature, statistics, trajectories, seed,
                                 phonon_cycles, phonon_damping=0., bath_cycles=8.,
                                 phase_points=33):
    """Populate the existing Opticalconductivity_TDSE output interface."""
    if owner.collision.specification.mode != 'crta':
        raise ValueError("Explicit phonon trajectories require CRTA background only; EPC SERTA would double count scattering")
    gamma = owner.collision.specification.constantLinewidth
    if (not np.isfinite(gamma) or gamma <= 0 or trajectories < 1 or int(trajectories) != trajectories
            or n_steps_per_cycle < 1 or int(n_steps_per_cycle) != n_steps_per_cycle
            or n_cycles_measure < 1 or int(n_cycles_measure) != n_cycles_measure
            or not np.isfinite(max_phase_step) or max_phase_step <= 0
            or not np.isfinite(transient_lifetimes) or transient_lifetimes < 0
            or not np.isfinite(phonon_cycles) or phonon_cycles < 0):
        raise ValueError("Require positive background linewidth, integer steps/cycles/trajectories, and valid time controls")
    phonon_damping, bath_cycles = float(phonon_damping), float(bath_cycles)
    if not np.isfinite(phonon_damping) or phonon_damping < 0:
        raise ValueError("phononDamping must be nonnegative and finite, in eV")
    if not np.isfinite(bath_cycles) or bath_cycles <= 0:
        raise ValueError("phononBathCycles must be positive and finite")
    operators = SSHTimeOperators(owner)
    phonons = owner.model.modelPhonons
    spawned_seeds = np.random.SeedSequence(seed).spawn(int(trajectories))
    if phonon_damping == 0:
        samples = [phonons.sampleTrajectory(temperature, seed=s, statistics=statistics)
                   for s in spawned_seeds]
        bath_period = 2*np.pi/np.min(phonons.energies)
        base_energy = np.min(phonons.energies)
        bath_name = 'prescribed harmonic'
    else:
        bath_period = bath_cycles*2*np.pi/np.min(phonons.energies)
        base_energy = 2*np.pi/bath_period
        if (not isinstance(phase_points, (int, np.integer)) or phase_points < 5
                or phase_points % 2 == 0):
            raise ValueError("phononPhasePoints must be an odd integer >= 5")
        max_resolved = ((phase_points-1)//2)*base_energy
        if max_resolved < np.max(phonons.energies):
            required = 2*int(np.ceil(np.max(phonons.energies)/base_energy))+1
            raise ValueError(f"phononPhasePoints must be at least {required} for the damped bath spectrum")
        samples = [phonons.sampleDampedTrajectory(
            temperature, phonon_damping, bath_period, phase_points, seed=s, statistics=statistics)
            for s in spawned_seeds]
        bath_name = 'stationary periodic damped-Langevin spectrum'
    owner.phonon_initial_positions = np.array([s.displacements(0) for s in samples])
    owner.phonon_initial_velocities = np.array([s.velocities(0) for s in samples])
    owner.phonon_trajectory_cosine = np.array([s.cosine for s in samples])
    owner.phonon_trajectory_sine = np.array([s.sine for s in samples])
    owner.phonon_energies = samples[0].energies.copy()
    owner.phonon_frequency = base_energy
    owner.phonon_dynamics_info = dict(
        bath=bath_name, statistics=statistics, temperature_K=float(temperature),
        trajectories=int(trajectories), seed=seed, electronic_backreaction=False,
        orbital_centers='fixed equilibrium SSH basis', background_linewidth_eV=float(gamma),
        integrator='linearized exact midpoint TDSE/CRTA', phonon_cycles_min=float(phonon_cycles),
        phonon_damping_eV=phonon_damping, bath_period_hbar_per_eV=float(bath_period),
        bath_cycles=float(1. if phonon_damping == 0 else bath_cycles),
        phase_points=(None if phonon_damping == 0 else int(phase_points)),
        fundamental_energy_eV=float(base_energy))
    owner.sigma_samples = np.zeros((int(trajectories),)+owner.sigma.shape, complex)
    scale = elementary_charge**2/(hbar*angstrom*abs(owner.model.Vcell))
    longest_phonon_period = 2*np.pi/np.min(phonons.energies)
    bath_bandwidth = (np.max(phonons.energies) if phonon_damping == 0
                      else ((phase_points-1)//2)*base_energy)
    span = np.max(np.ptp(owner.energies, axis=-1))
    # Frame remains the localized basis for the entire moving-ion trajectory.
    vectors = owner.vectors.reshape(operators.nk, operators.nb, operators.nb)
    adjoint = vectors.conj().swapaxes(-2, -1)
    for iw, omega in enumerate(owner.omegaList):
        period = 2*np.pi/omega
        steps = max(int(n_steps_per_cycle), int(np.ceil(
            period*max(span, gamma, bath_bandwidth)/max_phase_step)))
        dt = period/steps
        ntrans = int(np.ceil(transient_lifetimes/(gamma*period)))*steps
        needed_cycles = phonon_cycles*longest_phonon_period/period
        # An exactly commensurate period can land just above an integer in
        # floating point (1.3/.02 -> 65). Adding a whole extra optical cycle
        # then spoils the phonon-period average and leaks sidebands into sigma.
        nearest = np.rint(needed_cycles)
        if np.isclose(needed_cycles, nearest, rtol=1e-13, atol=1e-13):
            needed_cycles = nearest
        cycles = max(int(n_cycles_measure), int(np.ceil(needed_cycles)))
        nmeasure = cycles*steps
        owner.integration_info.append(dict(steps_per_cycle=steps, dt=dt,
            transient_steps=ntrans, measurement_steps=nmeasure, measurement_cycles=cycles,
            measurement_phonon_periods=nmeasure*dt/longest_phonon_period))
        times = (np.arange(nmeasure)+ntrans+1)*dt
        for fi, fermi in enumerate(owner.fermiList):
            for sample_index, trajectory in enumerate(samples):
                f = owner.fdOccupation(fermi).reshape(operators.nk, operators.nb)
                rho = (vectors*f[:, None, :]) @ adjoint
                response = np.zeros((3,)+rho.shape, complex)
                current = np.empty((nmeasure, 3, 3))
                for step in range(ntrans+nmeasure):
                    time = (step+.5)*dt
                    h, v, _ = operators.at_displacement(trajectory.displacements(time))
                    rho, response = linear_tdse_step(
                        rho, response, h, v, -np.sin(omega*time)/omega,
                        dt, gamma, fermi, owner.kBT)
                    if step >= ntrans:
                        time = (step+1)*dt
                        _, v, w = operators.at_displacement(trajectory.displacements(time))
                        a_field = -np.sin(omega*time)/omega
                        j = np.einsum('akmn,kbnm->ab', response, v)
                        j += a_field*np.einsum('kmn,kbanm->ab', rho, w)
                        current[step-ntrans] = -j.real*scale/operators.nk
                owner.sigma_samples[sample_index, fi, iw] = 2*np.mean(
                    current*np.exp(1j*omega*times[:, None, None]), axis=0)
                if recordCurrent:
                    for a in range(3):
                        owner.time_traces[sample_index, fi, iw, a] = dict(
                            time=times, normalized_current=current[:, a],
                            electric_field=owner.E0*np.cos(omega*times))
    owner.sigma[:] = owner.sigma_samples.mean(axis=0)
    if trajectories > 1:
        owner.sigma_stderr_real = owner.sigma_samples.real.std(axis=0, ddof=1)/np.sqrt(trajectories)
        owner.sigma_stderr_imag = owner.sigma_samples.imag.std(axis=0, ddof=1)/np.sqrt(trajectories)
    else:
        owner.sigma_stderr_real = np.full(owner.sigma.shape, np.nan)
        owner.sigma_stderr_imag = np.full(owner.sigma.shape, np.nan)
