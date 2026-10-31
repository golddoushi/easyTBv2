"""VASP linear EPC in the main workflow's explicitly shared Wannier basis."""
import numpy as np
from collections import OrderedDict
import threading
from .base import EPCModel
from phonon import ForceConstantPhonon


def validate_wannier_frame(system, source):
    """Compare numerical checkpoint frames, not just paths or spectra."""
    checkpoints = getattr(system, '_w90_chks', ())
    if len(checkpoints) != 1 or system.format != 'VASP' or system.overlap:
        raise ValueError('VASP EPC requires an unreordered orthogonal single-frame System_w90')
    chk = checkpoints[0]
    u = np.zeros((chk.num_kpts, chk.num_bands, chk.num_wann), complex)
    for ik, matrix in enumerate(chk.v_matrix):
        u[ik, chk.win_min[ik]:chk.win_max[ik]] = np.asarray(matrix).T
    if (u.shape != source.reference.u.shape
            or not np.allclose(u, source.reference.u, atol=1e-9, rtol=0)
            or not np.allclose(chk.kpt_list, source.reference.kpoints_frac, atol=1e-10, rtol=0)):
        raise ValueError('Electronic and EPC equilibrium Wannier gauges differ')
    if system.norbs != source.epc.shape[-1] or not np.allclose(system.lattice, source.lattice_ang, atol=1e-7, rtol=0):
        raise ValueError('Electronic and EPC orbital counts or lattices differ')
    positions = np.asarray(system.posinfo['pos ions (frac.)'])
    if positions.shape != source.positions_frac.shape:
        raise ValueError('Electronic and EPC atom counts differ')
    difference = positions-source.positions_frac
    if not np.allclose(difference, np.rint(difference), atol=1e-7, rtol=0):
        raise ValueError('Electronic and EPC atom order or origin differs')
    # POSCAR's ordered species must also match, including repeated species groups.
    species = np.repeat(system.posinfo['label list'], system.posinfo['nions'])
    if not np.array_equal(species, source.species):
        raise ValueError('Electronic and EPC species differ')


class VaspWannierEPC(EPCModel):
    """Use the reviewed source-to-RAM loader and interpolation through EPCModel.

    Initialization does not run external jobs. The source includes wannelph,
    PAW bridge, equilibrium chk/eig/hr/wsvec and independent force constants.
    ``source`` may be reused across responses. The electronic System must use
    its exact wsvec-expanded Hamiltonian; System_w90.fromEPCReference builds it.
    """
    def __init__(self, electronic_model, source, *, acoustic_policy='mask',
                 workspaceBytes=256*1024**2, cacheBytes=64*1024**2):
        if type(workspaceBytes) is not int or workspaceBytes <= 0:
            raise ValueError('workspaceBytes must be a positive integer')
        self.source = source
        self.workspaceBytes = workspaceBytes
        if type(cacheBytes) is not int or cacheBytes < 0:
            raise ValueError('cacheBytes must be a nonnegative integer')
        self.cacheBytes = cacheBytes
        self.preferredKBatchSize = 1
        self._cache = OrderedDict(); self._cache_bytes = 0
        self._cache_lock = threading.RLock()
        super().__init__(electronic_model, ForceConstantPhonon(source.phonons,
                                                             acoustic_policy=acoustic_policy))
        self.capabilities = dict(source.capabilities)
        self.provenance = dict(source.provenance)
        self.validate()

    @classmethod
    def from_directory(cls, electronic_model, directory, *, acoustic_policy='mask',
                       workspaceBytes=256*1024**2, cacheBytes=64*1024**2, **source_options):
        from epc import WannierEPCModel
        return cls(electronic_model, WannierEPCModel(directory, **source_options),
                   acoustic_policy=acoustic_policy, workspaceBytes=workspaceBytes, cacheBytes=cacheBytes)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_cache_lock')
        state['_cache'] = OrderedDict(); state['_cache_bytes'] = 0
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._cache_lock = threading.RLock()

    def _cartesian(self, k, q, budget):
        # Cache only exact fixed-k Fourier contractions. The lock bounds
        # workspace for thread-backed BZ workers; process workers own caches.
        with self._cache_lock:
            limit = min(self.cacheBytes, budget//2)
            while self._cache and self._cache_bytes > limit:
                _, old = self._cache.popitem(last=False)
                self._cache_bytes -= old.resident_bytes
            key = tuple(k)
            prepared = self._cache.get(key)
            if prepared is None:
                prepared = self.source.epc.prepare_k(k, budget=budget-self._cache_bytes)
                size = getattr(prepared, 'resident_bytes', 0)
                if size and size <= limit:
                    while self._cache and self._cache_bytes+size > limit:
                        _, old = self._cache.popitem(last=False)
                        self._cache_bytes -= old.resident_bytes
                    self._cache[key] = prepared; self._cache_bytes += size
            elif key in self._cache:
                self._cache.move_to_end(key)
            other = self._cache_bytes-(getattr(prepared, 'resident_bytes', 0) if key in self._cache else 0)
            return prepared.cartesian(k, q, budget=budget-other)

    def validate(self):
        system, source = self.electronic_model, self.source
        validate_wannier_frame(system, source)
        # Exact real-space representation detects modifications off the coarse
        # mesh too. Equal eigenvalues alone cannot establish the EPC gauge.
        terms = {}
        for r, h, weight in zip(source.electrons.re_vectors, source.electrons.h_r_ev,
                                source.electrons.re_weights):
            key = tuple(r)
            terms[key] = terms.get(key, 0)+h*weight
        for r, h in zip(system.iRvec, system.ham_R):
            key = tuple(r)
            terms[key] = terms.get(key, 0)-h
        error = max((float(np.max(np.abs(h))) for h in terms.values()), default=0.)
        if not np.isfinite(error) or error > 1e-12:
            raise ValueError('Electronic Hamiltonian differs from the EPC wsvec reference; use System_w90.fromEPCReference')

    @property
    def has_second_order(self):
        # This linear provider supplies no H''. Unknown is recorded separately
        # in capabilities; it must never be inferred to be a physical zero.
        return False

    def secondDisplacementVertices(self, kpoints, q, displacements):
        raise NotImplementedError('VASP general displacement Hessian is unknown; RIA DW is a separate thermal self-energy')

    def displacementVertices(self, kpoints, q, displacements, *, workspaceBytes=None):
        kpoints = np.asarray(kpoints, float)
        q = np.asarray(q, float)
        displacements = np.asarray(displacements, complex)
        nat, nw = self.source.epc.shape[2], self.electronic_model.norbs
        if (kpoints.ndim != 2 or kpoints.shape[1] != 3 or q.shape != (3,)
                or displacements.ndim != 3 or displacements.shape[1:] != (nat, 3)
                or not all(np.isfinite(x).all() for x in (kpoints, q, displacements))):
            raise ValueError('Expected finite kpoints[k,3], q[3] and displacements[mode,atom,xyz]')
        size = len(kpoints)*len(displacements)*nw*nw*16
        budget = self.workspaceBytes if workspaceBytes is None else min(workspaceBytes, self.workspaceBytes)
        if 2*size >= budget:
            raise MemoryError('EPC output exceeds workspaceBytes; use smaller electronic batches')
        result = np.empty((len(kpoints), len(displacements), nw, nw), complex)
        for ik, k in enumerate(kpoints):
            d = self._cartesian(k, q, budget-2*size)
            result[ik] = np.einsum('axmn,vax->vmn', d, displacements, optimize=True)
        return result

    def matrixElementsBatchBounded(self, kpoints, q, vectors_k, vectors_kq, *, workspaceBytes):
        nw = self.electronic_model.norbs
        kpoints = np.asarray(kpoints, float)
        expected = (len(kpoints), nw, nw)
        if np.shape(vectors_k) != expected or np.shape(vectors_kq) != expected:
            raise ValueError('EPC eigenvectors must span the full shared Wannier basis')
        vertices = self.displacementVertices(kpoints, q, self.phonon_model.modeDisplacements(q),
                                             workspaceBytes=workspaceBytes)
        return vectors_kq.conj().swapaxes(-2, -1)[:, None]@vertices@vectors_k[:, None]
