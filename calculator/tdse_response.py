"""Electronic TDSE conductivity from explicit electronic inputs and collision."""

import numpy as np
from scipy.constants import elementary_charge, hbar, angstrom
from auxiliary import fermiDiracOccupations
from .density_matrix import diagonal_matrix
from .response_inputs import ResponseInputs
from .response_results import ConductivityResult


def solve_electronic_conductivity(inputs: ResponseInputs, *, collision, E0=1e-4,
                                 n_steps_per_cycle=128, n_cycles_measure=2,
                                 transient_lifetimes=12., max_phase_step=.3,
                                 recordCurrent=False):
    result = ConductivityResult(np.zeros((len(inputs.fermiList), len(inputs.omegaList), 3, 3), complex))
    gamma = collision.coherence_linewidth
    def contract(rho, observable):
        return np.einsum("...mn,...cmn->...c", rho, observable.swapaxes(-1, -2), optimize=True)
    h0 = diagonal_matrix(inputs.energies)
    scale = elementary_charge**2 / (hbar*angstrom*abs(inputs.model.Vcell))
    # ± fields share a batched evolution. No response kernel enters it.
    signs = np.array([1.,-1.]).reshape((2,)+(1,)*h0.ndim)
    for iw, omega in enumerate(inputs.omegaList):
        period = 2*np.pi/omega
        span = np.max(np.ptp(inputs.energies, axis=-1))
        steps = max(int(n_steps_per_cycle), int(np.ceil(period*max(span,float(np.max(gamma)))/max_phase_step)))
        dt = period/steps
        ntrans = int(np.ceil(transient_lifetimes/(np.min(gamma)*period)))*steps
        nmeasure = int(n_cycles_measure)*steps
        result.integration_info.append(dict(steps_per_cycle=steps, dt=dt,
                                          transient_steps=ntrans, measurement_steps=nmeasure))
        half_decay = np.exp(-gamma*dt/2)
        for ifermi, fermi in enumerate(inputs.fermiList):
            rho0 = diagonal_matrix(inputs.fdOccupation(fermi))
            for a in range(3):
                va, waa = inputs.velocity_K[...,a,:,:], inputs.w_K[...,a,a,:,:]
                rho = np.broadcast_to(rho0, (2,)+rho0.shape).copy()
                times, currents = [], []

                def field_operators(time):
                    A = -E0/omega*np.sin(omega*time)*signs
                    H = h0 + A*va + .5*A*A*waa
                    return A, H

                for step in range(ntrans+nmeasure):
                    _, Hmid = field_operators((step+.5)*dt)
                    eig, vec = np.linalg.eigh(Hmid)
                    U = (vec*np.exp(-1j*eig*dt)[...,None,:]) @ vec.conj().swapaxes(-1,-2)
                    f = fermiDiracOccupations(eig, np.array([fermi]), inputs.kBT)[0]
                    equilibrium = (vec*f[...,None,:]) @ vec.conj().swapaxes(-1,-2)
                    rho = equilibrium + half_decay*(rho-equilibrium)
                    rho = U @ rho @ U.conj().swapaxes(-1,-2)
                    rho = equilibrium + half_decay*(rho-equilibrium)
                    if step >= ntrans:
                        time = (step+1)*dt
                        A, _ = field_operators(time)
                        current_vertex = inputs.velocity_K + A[...,None,:,:]*inputs.w_K[..., :,a,:,:]
                        current = -contract(rho, current_vertex).real
                        odd = .5*(current[0]-current[1]).mean(axis=(0,1,2))
                        times.append(time)
                        currents.append(odd*scale/E0)
                times, currents = np.asarray(times), np.asarray(currents)
                result.sigma[ifermi,iw,a] = 2*np.mean(currents*np.exp(1j*omega*times[:,None]), axis=0)
                if recordCurrent:
                    result.time_traces[ifermi,iw,a] = dict(time=times, normalized_current=currents,
                                                        electric_field=E0*np.cos(omega*times))
    return result
