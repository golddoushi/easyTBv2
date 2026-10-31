"""Streamed on-shell EPC linewidths from the main workflow's electronic states."""
import numpy as np
import weakref
from scipy.constants import Boltzmann, electron_volt
from scipy.special import expit
from phonon import PhononBath
from phonon.base import active_mode_indices
from auxiliary import uniform_qmesh
from .epc_rates import electronic_batch_size, phonon_population, band_vertices

class EPCLinewidth:
    """Bounded SERTA source for CollisionSpecification('serta').

    The returned relaxation energy is -2 Im Sigma in eV, never a rate in s^-1.
    No real self-energy or Debye-Waller term is inserted into the Hamiltonian.
    Linewidths are frozen at the explicitly supplied Fermi level and bath.
    ``block_trace`` averages diagonal rates within complete near-degenerate
    blocks, matching epc.transport's secular scalar-lifetime approximation.
    ``band`` preserves band-diagonal rates; ``reject`` rejects such blocks.
    """
    def __init__(self, epc_model, fermi=0., delta=.03, *, phononBath=PhononBath(),
                 degeneracyPolicy='block_trace', degeneracyTolerance=1e-8,
                 workspaceBytes=256*1024**2, kBatchSize=None):
        if not isinstance(phononBath, PhononBath):
            raise TypeError('phononBath must be PhononBath')
        if degeneracyPolicy not in ('block_trace', 'band', 'reject'):
            raise ValueError('degeneracyPolicy must be block_trace, band or reject')
        if not np.isfinite([fermi, delta, degeneracyTolerance]).all() or delta <= 0 or degeneracyTolerance < 0:
            raise ValueError('Finite Fermi energy, positive delta and nonnegative degeneracy tolerance required')
        if type(workspaceBytes) is not int or workspaceBytes <= 0:
            raise ValueError('workspaceBytes must be a positive integer')
        if kBatchSize is not None and (type(kBatchSize) is not int or kBatchSize < 1):
            raise ValueError('kBatchSize must be positive')
        self.epc_model, self.fermi, self.delta = epc_model, float(fermi), float(delta)
        self.phononBath = phononBath
        self.degeneracyPolicy, self.degeneracyTolerance = degeneracyPolicy, float(degeneracyTolerance)
        self.workspaceBytes, self.kBatchSize = workspaceBytes, kBatchSize
        self._cache_signature = None; self._cache_linewidth = None
        self._cache_states_ref = None
        self.info = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_cache_states_ref'] = None
        state['_cache_signature'] = None
        state['_cache_linewidth'] = None
        return state

    def populationLinewidth(self, inputs):
        from .Path import EnergyOnPath
        if inputs.model is not self.epc_model.electronic_model:
            raise ValueError('Linewidth EPC model must refer to the calculator electronic model')
        if inputs.states.skip_low or inputs.energies.shape[-1] != inputs.model.norbs:
            raise ValueError('Streamed EPC linewidth requires the complete electronic model basis')
        self.epc_model.validate()
        bath = self.phononBath
        signature = (id(inputs.states), bath.temperature, bath.statistics, bath.qgrid,
            None if bath.selection is None else bath.selection.signature, self.fermi,
            self.delta, self.degeneracyPolicy, self.degeneracyTolerance)
        if (signature == self._cache_signature and self._cache_states_ref is not None
                and self._cache_states_ref() is inputs.states):
            return self._cache_linewidth.copy()
        qgrid = inputs.kgrid if bath.qgrid is None else bath.qgrid
        qpoints = uniform_qmesh(qgrid) if bath.selection is None else bath.selection.qpoints
        weights = np.full(len(qpoints), 1/len(qpoints)) if bath.selection is None else bath.selection.weights
        modes = self.epc_model.phonon_model
        branches = [active_mode_indices(modes, q, bath.selection) for q in qpoints]
        nb = inputs.model.norbs
        energies = inputs.energies.reshape(-1, nb)
        vectors = inputs.vectors.reshape(-1, nb, nb)
        blocks = []
        external_energies = energies.copy()
        for ik, energy in enumerate(energies):
            row = []; start = 0
            for stop in range(1, nb+1):
                if stop == nb or energy[stop]-energy[start] > self.degeneracyTolerance:
                    block = np.arange(start, stop); row.append(block)
                    if len(block) > 1:
                        if self.degeneracyPolicy == 'reject':
                            raise ValueError('Degenerate EPC linewidth block rejected by policy')
                        if self.degeneracyPolicy == 'block_trace':
                            external_energies[ik, block] = np.mean(energy[block])
                    start = stop
            blocks.append(row)
        kpoints = inputs.states.kpoints
        nk = len(kpoints)
        required = 32*nk*nb  # output, cache, external energies and return copy
        if required >= self.workspaceBytes:
            raise MemoryError('EPC linewidth output exceeds workspaceBytes')
        batch = min(nk, electronic_batch_size(self.epc_model,
            max((len(modes.modes(q)[0]) for q in qpoints), default=0), nb,
            self.workspaceBytes-required, self.kBatchSize))
        linewidth = np.zeros_like(energies)
        grid = np.asarray(inputs.kgrid)
        kindices = np.indices(tuple(grid)).reshape(3, -1).T
        kbt = Boltzmann*bath.temperature/electron_volt
        qdata = []
        for q, weight, selected in zip(qpoints, weights, branches):
            if not len(selected) or weight == 0:
                continue
            omega = np.asarray(modes.modes(q)[0])[selected]
            population = phonon_population(omega, bath)
            shifted = q*grid
            closed = np.allclose(shifted, np.rint(shifted), atol=1e-9, rtol=0)
            shift = np.rint(shifted).astype(int)
            qdata.append((q, weight, selected, omega, population, closed, shift))
        for start in range(0, nk, batch):
            stop = min(nk, start+batch)
            for q, weight, selected, omega, population, closed, shift in qdata:
                if closed:
                    destination = np.ravel_multi_index(((kindices[start:stop]+shift) % grid).T, tuple(grid))
                    final_e, final_v = energies[destination], vectors[destination]
                else:
                    states = EnergyOnPath(model=inputs.model, kpath=np.mod(kpoints[start:stop]+q, 1.),
                                          syncModelVerbose=False)
                    final_e, final_v = states.energies, states.vectors
                g = band_vertices(self.epc_model, kpoints[start:stop], q, vectors[start:stop],
                                  final_v, (self.workspaceBytes-required)//2)[:, selected]
                occupations = (np.where(final_e < self.fermi, 1., np.where(final_e > self.fermi, 0., .5))
                               if kbt == 0 else expit((self.fermi-final_e)/kbt))
                difference = external_energies[start:stop, None, None, :]-final_e[:, None, :, None]
                emission = self.delta/np.pi/((difference-omega[None, :, None, None])**2+self.delta**2)
                absorption = self.delta/np.pi/((difference+omega[None, :, None, None])**2+self.delta**2)
                if bath.statistics == 'quantum':
                    factors = ((population[None, :, None, None]+1-occupations[:, None, :, None])*emission
                              +(population[None, :, None, None]+occupations[:, None, :, None])*absorption)
                else:
                    factors = population[None, :, None, None]*(emission+absorption)
                linewidth[start:stop] += 2*np.pi*weight*np.sum(np.abs(g)**2*factors, axis=(1, 2))
        max_block_spread = 0.
        if self.degeneracyPolicy != 'band':
            for ik, row in enumerate(blocks):
                for block in row:
                    if len(block) > 1:
                        if self.degeneracyPolicy == 'reject':
                            raise ValueError('Degenerate EPC linewidth block rejected by policy')
                        linewidth[ik, block] = np.mean(linewidth[ik, block])
                        max_block_spread = max(max_block_spread, float(np.ptp(energies[ik, block])))
        if not np.isfinite(linewidth).all() or np.any(linewidth < 0):
            raise ValueError('Invalid on-shell EPC linewidth')
        self.info = dict(units='eV', definition='-2 Im Sigma = hbar/tau',
                         k_batch_size=batch, degeneracy_policy=self.degeneracyPolicy,
                         max_block_energy_spread_ev=max_block_spread,
                         general_hessian='unknown' if getattr(self.epc_model, 'capabilities', {}).get('general_hessian', {}).get('status') == 'unknown' else 'provider_defined',
                         real_self_energy_included=False, debye_waller_included=False)
        self._cache_signature = signature
        # Object ids can be recycled between short-lived shifted BZ tasks.
        # Verify the actual live basis without retaining a previous mesh's arrays.
        self._cache_states_ref = weakref.ref(inputs.states)
        self._cache_linewidth = linewidth.reshape(inputs.energies.shape).copy()
        return self._cache_linewidth.copy()
