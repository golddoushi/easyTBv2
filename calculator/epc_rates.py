"""Bounded electronic batches for sparse or matrix-free mesh EPC rates."""
import numpy as np
from scipy.special import expit
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator
from scipy.constants import Boltzmann, electron_volt
from phonon.base import active_mode_indices
from auxiliary import uniform_qmesh


def electronic_batch_size(epc, modes, nbands, workspace, requested=None):
    """Conservative transient-array budget; resident models are caller-owned."""
    available = min(workspace, getattr(epc, 'workspaceBytes', workspace))
    per_k = 128*max(1, modes)*nbands*nbands+128*nbands*nbands+1024
    count = available//(2*per_k)
    if count < 1:
        raise MemoryError('EPC workspace cannot hold one electronic batch')
    if requested is not None:
        if type(requested) is not int or requested < 1:
            raise ValueError('kBatchSize must be a positive integer')
        count = min(count, requested)
    elif hasattr(epc, 'preferredKBatchSize'):
        count = min(count, epc.preferredKBatchSize)
    return int(count)


def phonon_population(energies, bath):
    kbt = Boltzmann*bath.temperature/electron_volt
    if bath.statistics == 'classical':
        return np.zeros_like(energies) if kbt == 0 else kbt/energies
    if bath.statistics == 'wigner':
        return np.full_like(energies, .5) if kbt == 0 else .5/np.tanh(energies/(2*kbt))
    if kbt == 0:
        return np.zeros_like(energies)
    x = energies/kbt
    return np.exp(-x)/(-np.expm1(-x))


def band_vertices(epc, kpoints, q, incoming, outgoing, workspace):
    bounded = getattr(epc, 'matrixElementsBatchBounded', None)
    if bounded is not None:
        return bounded(kpoints, q, incoming, outgoing, workspaceBytes=workspace)
    return epc.matrixElementsBatch(kpoints, q, incoming, outgoing)


class MeshEPCRates:
    """Golden-rule R[final,initial] in eV, streamed by q and electronic batch."""
    def __init__(self, inputs, specification, qpoints, qshifts, qweights, *, workspace):
        self.inputs, self.specification = inputs, specification
        self.qpoints, self.qshifts, self.qweights = qpoints, qshifts, qweights
        self.epc = inputs.epc_model
        self.grid = np.asarray(inputs.kgrid)
        self.nk = int(np.prod(self.grid)); self.nb = inputs.energies.shape[-1]
        self.size = self.nk*self.nb
        self.kindices = np.indices(tuple(self.grid)).reshape(3, -1).T
        self.kpoints = uniform_qmesh(self.grid)+inputs.kOffset
        self.energies = inputs.energies.reshape(self.nk, self.nb)
        self.vectors = inputs.vectors.reshape(self.nk, self.nb, self.nb)
        self.workspace = workspace
        self.mode_indices = []
        for q in qpoints:
            self.mode_indices.append(active_mode_indices(self.epc.phonon_model, q,
                                                       specification.phononBath.selection))
        modes = max((len(self.epc.phonon_model.modes(q)[0]) for q in qpoints), default=0)
        self.batch_size = min(self.nk, electronic_batch_size(self.epc, modes, self.nb,
                                    workspace, specification.kBatchSize))

    def blocks(self):
        spec = self.specification
        bands = np.arange(self.nb)
        for iq, (q, shift, weight) in enumerate(zip(self.qpoints, self.qshifts, self.qweights)):
            branches = self.mode_indices[iq]
            if not len(branches) or weight == 0:
                continue
            energies, _ = self.epc.phonon_model.modes(q)
            omega = np.asarray(energies)[branches]
            population = phonon_population(omega, spec.phononBath)
            for start in range(0, self.nk, self.batch_size):
                stop = min(self.nk, start+self.batch_size)
                destination = np.ravel_multi_index(((self.kindices[start:stop]+shift) % self.grid).T,
                                                   tuple(self.grid))
                g = band_vertices(self.epc, self.kpoints[start:stop], q,
                                  self.vectors[start:stop], self.vectors[destination], self.workspace//2)[:, branches]
                difference = self.energies[start:stop, None, None, :]-self.energies[destination, None, :, None]
                emission = spec.delta/np.pi/((difference-omega[None, :, None, None])**2+spec.delta**2)
                absorption = spec.delta/np.pi/((difference+omega[None, :, None, None])**2+spec.delta**2)
                if spec.phononBath.statistics == 'quantum':
                    factors = (population[None, :, None, None]+1)*emission+population[None, :, None, None]*absorption
                else:
                    factors = population[None, :, None, None]*(emission+absorption)
                rates = 2*np.pi*weight*np.sum(np.abs(g)**2*factors, axis=1)
                # Exclude transitions from a state into itself, preserving the
                # full (not flattened) band energies and original branch labels.
                same_k = destination == np.arange(start, stop)
                rates[:, bands, bands] = np.where(same_k[:, None], 0., rates[:, bands, bands])
                yield iq, start, stop, destination, rates

    def sparse(self):
        columns = len(self.qpoints)*self.nb
        values = np.zeros((self.size, columns), float)
        rows = np.zeros(values.shape, dtype=np.int64)
        for iq, start, stop, destination, rates in self.blocks():
            slot = slice(iq*self.nb, (iq+1)*self.nb)
            values[start*self.nb:stop*self.nb, slot] = rates.swapaxes(-1, -2).reshape(-1, self.nb)
            final_ids = destination[:, None]*self.nb+np.arange(self.nb)
            rows[start*self.nb:stop*self.nb, slot] = np.repeat(final_ids, self.nb, axis=0)
        result = csc_matrix((values.ravel(), rows.ravel(), np.arange(self.size+1, dtype=np.int64)*columns),
                            shape=(self.size, self.size))
        result.eliminate_zeros(); result.sort_indices()
        return result


class EPCRateOperator(LinearOperator):
    """Recompute bounded rate blocks instead of retaining Nk*Nq*Nb^2 entries.

    Pair balance uses R_balanced=(R+R.T)*thermal_fraction exactly, including
    non-inversion-symmetric selected quadratures. Every raw edge contributes
    both balanced directions. Transpose actions never assume EPC reciprocity.
    """
    def __init__(self, sampler):
        self.sampler = sampler
        self.balanced = sampler.specification.enforceDetailedBalance
        self.bath_kbt = Boltzmann*sampler.specification.phononBath.temperature/electron_volt
        maximum = 0.
        for _, _, _, _, rates in sampler.blocks():
            maximum = max(maximum, float(np.max(rates)))
        # A conservative bound also accommodates duplicated selected q points.
        self.max_rate_bound = 2*len(sampler.qpoints)*maximum if self.balanced else len(sampler.qpoints)*maximum
        super().__init__(dtype=np.dtype(float), shape=(sampler.size, sampler.size))

    def _action(self, vector, transpose):
        sampler = self.sampler
        x = np.asarray(vector).reshape(sampler.nk, sampler.nb)
        result = np.zeros_like(x, dtype=np.result_type(x, float))
        for _, start, stop, destination, rates in sampler.blocks():
            if self.balanced:
                difference = sampler.energies[destination, :, None]-sampler.energies[start:stop, None, :]
                fraction = expit(-difference/self.bath_kbt)
                forward, reverse = rates*fraction, rates*(1-fraction)
            else:
                forward, reverse = rates, None
            if transpose:
                result[start:stop] += np.einsum('bmn,bm->bn', forward, x[destination])
                if reverse is not None:
                    np.add.at(result, destination, np.einsum('bmn,bn->bm', reverse, x[start:stop]))
            else:
                np.add.at(result, destination, np.einsum('bmn,bn->bm', forward, x[start:stop]))
                if reverse is not None:
                    result[start:stop] += np.einsum('bmn,bm->bn', reverse, x[destination])
        return result.ravel()

    def _matvec(self, vector):
        return self._action(vector, False)

    def _rmatvec(self, vector):
        return self._action(vector, True)

    def diagonal(self):
        return np.zeros(self.shape[0])


def shifted_operator(operator, diagonal, shift):
    """Shift a population operator while retaining an explicit preconditioner."""
    if not isinstance(operator, LinearOperator):
        from scipy.sparse import eye
        return operator+shift*eye(operator.shape[0], format='csc')
    result = LinearOperator(operator.shape, dtype=np.result_type(operator.dtype, shift),
                            matvec=lambda x: operator@x+shift*x)
    result.population_diagonal = np.asarray(diagonal)+shift
    result.diagonal = lambda: result.population_diagonal.copy()
    return result
