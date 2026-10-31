"""Force-constant modes in the main Phonon interface, preserving VASP policies."""
import numpy as np
from .base import Phonon, _wavevector


class ForceConstantPhonon(Phonon):
    """Wrap the validated FC provider; branches include explicitly masked modes.

    ``acoustic_policy='mask'`` omits only the provider's Gamma translations.
    It is an explicit quadrature prescription, not a finite-q acoustic limit.
    Unstable modes are rejected. Energies are eV; zero-point displacements are
    Angstrom. The underlying provider owns force constants and their provenance.
    """
    def __init__(self, provider, *, acoustic_policy='mask'):
        if acoustic_policy not in ('mask', 'reject'):
            raise ValueError('acoustic_policy must be mask or reject')
        super().__init__(provider.masses_amu)
        self.provider = provider
        self.acoustic_policy = acoustic_policy
        self.provenance = dict(provider.provenance)

    @classmethod
    def from_file(cls, path, *, acoustic_policy='mask', **options):
        from epc.phonons import import_force_constants, PhononModel
        policies = {key: options.pop(key) for key in ('asr_policy', 'tolerance') if key in options}
        imported = import_force_constants(path, **options)
        provider = PhononModel(imported.r_vectors, imported.force_constants_ev_per_ang2,
            imported.masses_amu, imported.r_weights, constants=imported.constants,
            provenance=imported.provenance, **policies,
            **{key: options[key] for key in ('hermitian_policy', 'acoustic_tolerance_angular_thz') if key in options})
        return cls(provider, acoustic_policy=acoustic_policy)

    def modeRecord(self, q=(0, 0, 0)):
        q = _wavevector(q, 'q')
        record = self.provider.evaluate(q, acoustic_policy=self.acoustic_policy,
                                        unstable_policy='reject')
        for value in record.values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
        return record

    def modes(self, q=(0, 0, 0)):
        record = self.modeRecord(q)
        return record['energies_ev'], record['eigenvectors'].transpose(2, 0, 1)

    def activeModeMask(self, q=(0, 0, 0)):
        return self.modeRecord(q)['active_modes']

    def modeDisplacements(self, q=(0, 0, 0)):
        return self.modeRecord(q)['zero_point_displacements_ang'].transpose(2, 0, 1)
