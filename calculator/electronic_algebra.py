"""Electronic algebra on explicit arrays; no model preparation or response outputs."""

import numpy as np
from auxiliary import fermiDiracOccupations


class ElectronicAlgebra:
    def calcDeltaEInverse(self):
        '''
        Calc. the inversed energy difference matrix, the value at degenerated point is set to zero.

        Indexing: 
        
        deltaE[ik1, ik2, ik3, ieig, jeig] = E_{ki} - E_{kj}

        deltaEInverse[ik1, ik2, ik3, ieig, jeig] = 1/(E_{ki} - E_{kj} - 1j*eta)
        '''
        self.deltaE = self.energies[...,:,None] - self.energies[...,None,:]
        self.deltaEInverse =np.zeros_like(self.deltaE, dtype=complex)
        self.deltaEInverse_pure =np.zeros_like(self.deltaE, dtype=float)
        self.deltaEInverse2=np.zeros_like(self.deltaE, dtype=complex)
        nondegeneratedCondition=abs(self.deltaE)>self.degen
        self.deltaEInverse[nondegeneratedCondition] =1./(self.deltaE[nondegeneratedCondition] - 1j*self.eta)
        self.deltaEInverse_pure[nondegeneratedCondition] =1./self.deltaE[nondegeneratedCondition]
        self.deltaEInverse2[nondegeneratedCondition]=1./self.deltaE[nondegeneratedCondition]/(self.deltaE[nondegeneratedCondition] - 1j*self.eta)
        if self.smr_type == 'Lorentzian':
            return
        raise NotImplementedError()

    @property
    def fd_occupations(self):
        """
        Return Fermi-Dirac occupations with axes ``(fermi, *energies.shape)``.
        
        The piecewise evaluation avoids overflow far from the Fermi level.
        ``kBT=0`` gives the strict zero-temperature step convention used by
        the response calculators.
        """
        return fermiDiracOccupations(
            self.energies,
            self.fermiList,
            self.kBT,
        )

    def fdOccupation(self, fermi):
        """Return the Fermi-Dirac occupations for one chemical potential."""
        return fermiDiracOccupations(
            self.energies,
            np.asarray([fermi], dtype=float),
            self.kBT,
        )[0]

    @property
    def fd_energy_derivative(self):
        """Return ``df/dE`` for each Fermi level and band energy."""
        return self.fdEnergyDerivatives()[0]

    @property
    def fd_energy_second_derivative(self):
        """Return ``d2f/dE2`` for each Fermi level and band energy."""
        return self.fdEnergyDerivatives()[1]

    def fdEnergyDerivatives(self, occupations=None):
        """Return ``df/dE`` and ``d2f/dE2`` from supplied or computed occupations."""
        assert self.kBT > 0, "Fermi-Dirac energy derivatives require kBT > 0"
        f = self.fd_occupations if occupations is None else np.asarray(occupations)
        fp = -f * (1.0 - f) / self.kBT
        fpp = f * (1.0 - f) * (1.0 - 2.0 * f) / self.kBT**2
        return fp, fpp

    def projectorInEigenBasis(self,projectorInLocalBasis:np.ndarray):
        '''
        Transform the project operator in local basis into the eigen-basis
        '''
        if self.model.overlap:
            return self.vectors.conj().swapaxes(-2,-1) @ projectorInLocalBasis[None,None,None,:,:] @ self.O_K_AO @ self.vectors
        else:
            return self.vectors.conj().swapaxes(-2,-1) @ projectorInLocalBasis[None,None,None,:,:] @ self.vectors

    def buildBandEnergyDerivatives(self):
        """Build first and second covariant k derivatives of band energies."""
        velocity_diag = np.diagonal(self.velocity_K, axis1=-2, axis2=-1).real
        self.energy_deriv1_K = velocity_diag

        nbands = self.energies.shape[-1]
        self.energy_deriv2_K = np.zeros(self.kgrid + (3, 3, nbands), dtype=float)
        for idirection in range(3):
            velocity_direction = self.velocity_K[..., idirection, :, :]
            for ideriv in range(3):
                r_derivative_direction = self.r_K[..., ideriv, :, :]
                commutator = r_derivative_direction @ velocity_direction - velocity_direction @ r_derivative_direction
                hessian = self.w_K[..., ideriv, idirection, :, :] + 1j * commutator
                self.energy_deriv2_K[..., ideriv, idirection, :] = np.diagonal(hessian, axis1=-2, axis2=-1).real
        self.energy_deriv2_K = (self.energy_deriv2_K + self.energy_deriv2_K.swapaxes(-3, -2)) / 2

    def fermiKDerivatives(self, ifermi):
        """Return ``f``, ``partial_k f`` and ``partial_k partial_k f``."""
        if self.kBT <= 0:
            raise ValueError("Fermi-Dirac k derivatives require kBT > 0")
        if not hasattr(self, "energy_deriv2_K"):
            self.buildBandEnergyDerivatives()

        f = self.fdOccupation(self.fermiList[ifermi])
        fp, fpp = self.fdEnergyDerivatives(f)
        dfdk = fp[..., None, :] * self.energy_deriv1_K
        d2fdk2 = (
            fpp[..., None, None, :]
            * self.energy_deriv1_K[..., :, None, :]
            * self.energy_deriv1_K[..., None, :, :]
            + fp[..., None, None, :] * self.energy_deriv2_K
        )
        return f, dfdk, d2fdk2

    def fermiCovariantFirstDerivativeChannels(self, f, dfdk, direction):
        """Return the intraband and interband sectors of ``C_a f``, with ``C_a=iD_a``."""
        nbands = self.energies.shape[-1]
        f_nm = f[..., None, :] - f[..., :, None]
        intraband = np.zeros(f.shape + (nbands,), dtype=complex)
        indices = np.arange(nbands)
        intraband[..., indices, indices] = 1j * dfdk[..., direction, :]
        interband = self.r_K[..., direction, :, :] * f_nm * (1.0 - np.eye(nbands))
        return intraband, interband

    def fermiCovariantSecondDerivativeChannels(self, f, dfdk, d2fdk2, a, b):
        """Return II, EI, IE and EE sectors of ``C_a C_b f``, with ``C_a=iD_a``."""
        nbands = self.energies.shape[-1]
        indices = np.arange(nbands)
        diag = np.eye(nbands)
        off_diag = 1.0 - diag
        f_nm = f[..., None, :] - f[..., :, None]
        r_a = self.r_K[..., a, :, :]
        r_b = self.r_K[..., b, :, :]

        population_b = 1j * dfdk[..., b, :]
        channels = {
            "II": np.zeros(self.energies.shape + (nbands,), dtype=complex),
            "EI": r_a * (population_b[..., None, :] - population_b[..., :, None]) * off_diag,
        }
        channels["II"][..., indices, indices] = -d2fdk2[..., a, b, :]

        coherence_b = r_b * f_nm * off_diag
        delta_f_a = dfdk[..., a, None, :] - dfdk[..., a, :, None]
        semicolon = (
            self.r_deriv_K[..., a, b, :, :] * f_nm + r_b * delta_f_a
        ) * off_diag
        derivative = 1j * semicolon + r_a @ coherence_b - coherence_b @ r_a
        channels["IE"] = derivative * diag
        channels["EE"] = derivative * off_diag
        return channels
