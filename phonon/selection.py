"""Phonon wavevector and branch quadrature."""

import numpy as np

class PhononModeSelection:
    """Phonon quadrature: fractional q points, branches and weights per q.

    This selector does not define frequencies, polarizations or a dispersion
    relation. The same branch list is applied at every selected q. Physical
    displacement construction and sampling belong to the phonon model.

    Conjugate partners are inserted with equal weight; an explicitly supplied
    partner is counted once. Weights are never normalized. ``branches=None``
    means all branches, with the same q weight for each branch. Branch labels
    at -q must correspond to conjugate polarizations at q; the selector
    does not infer this correspondence from eigenvectors.
    """

    def __init__(self, qpoints, branches=None, weights=None):
        points = np.atleast_2d(np.asarray(qpoints, float))
        if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
            raise ValueError('qpoints must be a nonempty finite (nq,3) array')
        values = np.ones(len(points)) if weights is None else np.asarray(weights, float)
        if values.shape != (len(points),) or not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError('one positive finite weight is required per input q point')
        if branches is not None:
            b = np.asarray(branches)
            if b.ndim != 1 or not len(b) or not np.isfinite(b).all() or np.any(b < 0) or np.any(b != b.astype(int)):
                raise ValueError('branches must be nonnegative integer indices')
            if len(np.unique(b)) != len(b):
                raise ValueError('branches must be distinct')
            branches = tuple(sorted(b.astype(int).tolist()))
        self.branches = branches
        entries = {}
        for q, weight in zip(points, values):
            for sign in (1, -1):
                key = tuple(np.round(np.mod(sign*q, 1), 14) % 1)
                if key in entries and not np.isclose(entries[key], weight, rtol=1e-12, atol=0):
                    raise ValueError('duplicate/conjugate q points must have equal weights')
                entries[key] = float(weight)
        keys = sorted(entries)
        self.qpoints = np.array(keys)
        self.weights = np.array([entries[key] for key in keys])
        self.qpoints.setflags(write=False)
        self.weights.setflags(write=False)

    @property
    def signature(self):
        return (self.qpoints.tobytes(), self.weights.tobytes(), self.branches)

    def branch_indices(self, nmodes):
        indices = np.arange(nmodes) if self.branches is None else np.array(self.branches, int)
        if np.any(indices >= nmodes):
            raise ValueError(f'phonon branch index exceeds available range 0..{nmodes-1}')
        return indices

    def mesh_shifts(self, grid):
        shifts = self.qpoints*np.asarray(grid)
        if not np.allclose(shifts, np.rint(shifts), rtol=0, atol=1e-9):
            raise ValueError('selected q points must map the electronic k grid onto itself')
        return np.rint(shifts).astype(int)

    def check_supercell(self, matrix):
        phases = self.qpoints@np.asarray(matrix).T
        if not np.allclose(phases, np.rint(phases), rtol=0, atol=1e-9):
            raise ValueError('selected q points are not commensurate with the supercell')

    def as_dict(self):
        return dict(qpoints=self.qpoints.tolist(), weights=self.weights.tolist(),
                    branches=None if self.branches is None else list(self.branches),
                    total_q_weight=float(self.weights.sum()), weight_normalization='none')


