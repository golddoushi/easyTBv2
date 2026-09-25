"""Electronic tensors and numerical settings consumed by response kernels."""

import numpy as np
from .electronic import ElectronicVertices, ElectronicTensorView
from .electronic_algebra import ElectronicAlgebra


class ResponseInputs(ElectronicTensorView, ElectronicAlgebra):
    """One response's inputs and derived electronic algebra, without outputs.

    Electronic arrays retain the prepared band basis. Fermi scans, photon
    frequencies and electronic temperature are independent response settings.
    """
    def __init__(self, electronic: ElectronicVertices, *, kBT=.026,
                 fermiList=(0.,), omegaList=(0.,), degen=1e-6,
                 eta=0., smr_type='Lorentzian'):
        if not isinstance(electronic, ElectronicVertices):
            raise TypeError('electronic must be ElectronicVertices')
        self.electronic = electronic
        self.states = electronic.states
        self.model = self.states.model
        self.kgrid = self.states.kgrid
        self.kOffset = self.states.kOffset
        self.kBT = float(kBT)
        self.fermiList = np.asarray(fermiList)
        self.omegaList = np.asarray(omegaList)
        self.degen, self.eta, self.smr_type = degen, eta, smr_type

    def calcDeltaEInverse(self):
        """Prepare band-pair algebra once when a DM kernel needs it.

        Static/trajectory solvers do not need these full-mesh arrays. The
        numerical settings of one ResponseInputs instance remain fixed.
        """
        if not hasattr(self, 'deltaE'):
            super().calcDeltaEInverse()
