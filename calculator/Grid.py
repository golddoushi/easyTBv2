#
# The module is for calculations on energy, berry curv., and several types of conductances
#
from system import System
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from auxiliary import showElapsedTime, print_matrix, projectionOperator
from auxiliary import PauliBlock, semiCirle, fermiDiracOccupations
from scipy.special import p_roots
from scipy.constants import elementary_charge, electron_mass, hbar, angstrom
from scipy.constants import k as kB
from .density_matrix import (CHANNEL_DESCRIPTIONS as DM_CHANNEL_DESCRIPTIONS,
                             CHANNEL_NAMES as DM_CHANNEL_NAMES,
                             LengthGaugeSecondOrderDMKernel,
                             VelocityGaugeSecondOrderDMKernel)

class Calculator():
    Iodd = False  # inversion gives minus
    Todd = False  # time reversal gives minus

    def __init__(self,**kwarg) -> None:
        '''
        Calculations on gamma-centered kmesh on BZ

        Parameters
        ----------
        - model : System
            The basis system, involving the hopping, basis coordinates, lattice matrix and perharps the overlap
        - kgrid : tuple(3, dtype=int)
            Size of kmesh
        - eta   : float
            The relaxation energy, in the same unit with energy, usually eV
        - kBT  : float
            Kinetic energy of electron, in the same unit with energy usually eV
        - smr_type : str, "Lorentzian"|"Gaussian"
        - degen : float
            The threthold two states are treated to be degenerated
        - skip_low : bool
            To skip lowest and seperated eigen-states IN ALL CASES
        - phaseshift : np.ndarray(self.model.nRvec, dtype=complex)
            Extra phase for the Bloch states, used for hybrid FFT algorithm. Default: 1, no extra phase shift
        - fermiList : np.ndarray(dtype=float)
            Fermi levels to scan
        - omegaList : np.ndarray(dtype=float)
            photon's frequencies to scan
        '''
        defaultSettings={
            "model":None,
            "kgrid":(1,1,1),
            "eta":0.0,
            "kBT":0.026,
            "smr_type":"Lorentzian",
            "degen":1e-6,
            "skip_low":False,

            "phaseshift":1,

            "fermiList":np.array([0,]),

            "omegaList":np.array([0,]),

            "greenFuncOmegaBatch":1,
            "greenFuncNWorkers":1,

            "verbose":0,
        }
        defaultSettings.update(kwarg)
        self.model:System=defaultSettings["model"]
        self.kgrid:tuple=defaultSettings["kgrid"]
        assert type(self.kgrid)==tuple, f"expect tuple type of kgrid but {type(self.kgrid)=}"
        self.eta=defaultSettings["eta"]
        self.kBT=defaultSettings["kBT"]
        self.smr_type=defaultSettings["smr_type"]
        self.degen=defaultSettings["degen"]
        self.skip_low=defaultSettings["skip_low"]

        self.phaseshift=defaultSettings["phaseshift"]
        self.phase=False
        if type(self.phaseshift)==np.ndarray: self.phase=True
        self.fermiList=defaultSettings["fermiList"]
        self.omegaList=defaultSettings["omegaList"]
        self.greenFuncOmegaBatch=max(1, int(defaultSettings["greenFuncOmegaBatch"]))
        self.greenFuncNWorkers=max(1, int(defaultSettings["greenFuncNWorkers"]))

        self.verbose=defaultSettings["verbose"]
        if self.verbose==2: print("initializing calculator module finished")

#
# Implemented calculaors
# Energy, velocity (ao and eigen gauges), spin velocity (ao and eigen gauges), 
# Berry curvature, spin Berry curvature
#
class Energy(Calculator):
    def __init__(self, **kwarg) -> None:
        '''
        Solve the eigen-eqs. to obtain eigen-vals. and vecs.

        Usually the second father of all calculators

        Indexing:
        self.energies[ik1, ik2, ik3, ieig] gives eigen-vals.

        self.vectors[ik1, ik2, ik3, ieig, jeig] gives eigen-vecs. in ao basis.
        '''
        super().__init__(**kwarg)
        self.ibmin=0             # ignore low-lying bands
        if self.verbose==2: showElapsedTime(offset=True)
        
        if self.model.overlap:
            self.solveGeneralEigenEquation()
        else:
            self.solveEigenEquation()

    def solveGeneralEigenEquation(self):
        '''
        Solve the general eig.-eqs. 

        H \dot U = O \dot U \dot \Gamma
        '''
        def solveOverlapWithSingleSpinTunnel(O_K:np.ndarray):
            return np.linalg.eigh(O_K)
        def solveOverlapWithDoubleSpinTunnels(O_K:np.ndarray):
            nR1,nR2,nR3,norbs,norbs=O_K.shape
            O_K_half=O_K[...,:norbs//2,:norbs//2]
            eigHalf, WHalf = np.linalg.eigh(O_K_half)
            eigFull=np.zeros((nR1,nR2,nR3,norbs))
            WFull=np.zeros(O_K.shape,dtype=complex)
            eigFull[...,:norbs//2]=eigHalf
            eigFull[...,norbs//2:]=eigHalf
            WFull[...,:norbs//2,:norbs//2]=WHalf
            WFull[...,norbs//2:,norbs//2:]=WHalf
            return eigFull, WFull
        def solveOverlap(O_K:np.ndarray):
            if self.model.ispin==1:
                return solveOverlapWithSingleSpinTunnel(O_K)
            elif self.model.ispin==2:
                return solveOverlapWithDoubleSpinTunnels(O_K)
            else:
                print(f"Only support ispin=1|2 {self.model.ispin=}")
                raise ValueError()

        if not self.phase:
            E_R_AO=self.model.energy_R
            O_R_AO=self.model.overlap_R
        else:
            assert len(self.phaseshift)==self.model.nRvec, f'{len(self.phaseshift)=} {self.model.nRvec=}'
            
            E_R_AO=self.model.energy_R*self.phaseshift[:,None,None]
            O_R_AO=self.model.overlap_R*self.phaseshift[:,None,None]
        
        self.E_R_intepolated=np.zeros(self.kgrid+(self.model.norbs,self.model.norbs),dtype=complex)
        self.O_R_intepolated=np.zeros(self.kgrid+(self.model.norbs,self.model.norbs),dtype=complex)
        for iR, Rvec in enumerate(self.model.iRvec):
            self.E_R_intepolated[tuple(Rvec)]=E_R_AO[iR]
            self.O_R_intepolated[tuple(Rvec)]=O_R_AO[iR]
        self.E_K_AO=np.fft.ifftn(self.E_R_intepolated,axes=(0,1,2,))*np.prod(self.kgrid)
        self.O_K_AO=np.fft.ifftn(self.O_R_intepolated,axes=(0,1,2,))*np.prod(self.kgrid)
        if self.verbose==2: showElapsedTime(extraInfo=f"FFT energy_R to energy_K and O_R to O_K on {self.kgrid=} finished")

        overlapEigValues, Wk =solveOverlap(self.O_K_AO)
        assert overlapEigValues.min() > 1e-6, f"eigen-value of overlap matrix is too small or negative {overlapEigValues.min()=}"
        if self.verbose==2: print(f"{overlapEigValues.min()=}")

        WkLam=Wk/np.sqrt(overlapEigValues[...,None,:])
        if self.verbose==2: showElapsedTime(extraInfo=f"WkLam")
        EnergyMatrixInK_tilde=WkLam.conj().transpose((0,1,2,4,3)) @ self.E_K_AO @ WkLam
        if self.verbose==2: showElapsedTime(extraInfo=f"Overlapped energy")
        self.energies, vecsBeforeOverlap = np.linalg.eigh(EnergyMatrixInK_tilde)
        if self.verbose==2: showElapsedTime(extraInfo=f"eigen-eqs.")
        self.vectors = WkLam @ vecsBeforeOverlap
        if self.verbose==2: showElapsedTime(extraInfo=f"overlapping on eig.-vecs.")

        if self.skip_low: self.getEnergyWindow()

    def solveEigenEquation(self):
        '''
        Solve the eig.-eqs. 

        H \dot U = U \dot \Gamma
        '''
        if not self.phase:
            HH_R_W=self.model.ham_R
        else:
            assert len(self.phaseshift)==self.model.nRvec, f'{len(self.phaseshift)=} {self.model.nRvec=}'
            
            HH_R_W=self.model.ham_R*self.phaseshift[:,None,None]
        self.ham_R_intepolated=np.zeros(self.kgrid+(self.model.norbs,self.model.norbs),dtype=complex)
        for iR, Rvec in enumerate(self.model.iRvec):
            self.ham_R_intepolated[tuple(Rvec)]=HH_R_W[iR]
        self.ham_K=np.fft.ifftn(self.ham_R_intepolated,axes=(0,1,2,))*np.prod(self.kgrid)
        if self.verbose==2: showElapsedTime(extraInfo=f"FFT ham_R to ham_K on {self.kgrid=} finished")
        self.energies, self.vectors = np.linalg.eigh(self.ham_K)
        if self.verbose==2: showElapsedTime(extraInfo=f"eig.-eqs. solved")

        if self.skip_low: self.getEnergyWindow()
    
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
    
    def getEnergyWindow(self,gap=5.0):
        energy=np.sort(self.energies.flatten()) - self.model.fermi
        valence_bands=energy[energy<=0]
        nvb=len(valence_bands)
        for ilevel in range(nvb-1):
            dE=valence_bands[nvb-1-ilevel] - valence_bands[nvb-2-ilevel]
            if dE > gap: break
        else:
            return
        mid_valence=(valence_bands[nvb-1-ilevel] + valence_bands[nvb-2-ilevel])/2
        self.ibmin=len(self.energies[(0,0,0)][self.energies[(0,0,0)]<mid_valence])
        self.energies=self.energies[:,:,:,self.ibmin:]
        self.vectors=self.vectors[:,:,:,:,self.ibmin:]
        if self.verbose: print(f"Low-lying bands skipped, {self.ibmin=}")
    
    def projectorInEigenBasis(self,projectorInLocalBasis:np.ndarray):
        '''
        Transform the project operator in local basis into the eigen-basis
        '''
        if self.model.overlap:
            return self.vectors.conj().swapaxes(-2,-1) @ projectorInLocalBasis[None,None,None,:,:] @ self.O_K_AO @ self.vectors
        else:
            return self.vectors.conj().swapaxes(-2,-1) @ projectorInLocalBasis[None,None,None,:,:] @ self.vectors

class FermiScan(Energy):
    def __init__(self, **kwarg):
        '''
        Compute the fermi function and search the band in range, based on the band energies, fermi energie(s), and beta.
        |E_{km} - E_f| < 20 kBT are thought to be involved
        self.ibmin, self.ibmax are integers, giving the involved min and max index of band (on whole BZ)

        Indexing
        --------
        self.fermiWeight[ifermi,k1,k2,k3,iband] gives the value of fermi weight the \partial_E f(E_{k,n})
        NOTE: iband only indexes the band in window, that is, iband=0 corresponding to the band self.ibmin.
        '''
        super().__init__(**kwarg)
        Emax, Emin=np.max(self.fermiList)+20*self.kBT, np.min(self.fermiList)-20*self.kBT
        condition_energyInRange=np.sum((self.energies>Emin) * (self.energies<Emax),axis=(0,1,2,))
        self.ibmin, self.ibmax= 0, self.energies.shape[-1]-1
        while(condition_energyInRange[self.ibmin]<0.5):
            self.ibmin+=1
        while(condition_energyInRange[self.ibmax]<0.5):
            self.ibmax-=1
        # Fermi weight for each band (in energy window)
        self.fermiWeight = self.fd_energy_derivative[..., self.ibmin:self.ibmax+1]
        if self.verbose==2: showElapsedTime(extraInfo=f"Band index window: {self.ibmin=} {self.ibmax=}")

class PDOS(Energy):
    def __init__(self, energyAxis:np.ndarray, eta=0.05, **kwarg) -> None:
        '''
        Compute the PDOS

        Parameters:
        -----------
        - energy Axis: energy axis for pdos intepolation, in absolute frame, NOT relative to Fermi
        - sigma: gaussian width
        '''
        super().__init__(**kwarg)

        NEDOS,=energyAxis.shape
        self.pdos=np.zeros((NEDOS,self.model.norbs),dtype=float)
        for ik1 in range(self.kgrid[0]):
            for ik2 in range(self.kgrid[1]):
                for ik3 in range(self.kgrid[2]):
                    for iband in range(self.model.norbs-self.ibmin):
                        E_band = self.energies[ik1,ik2,ik3,iband]
                        vecs   = self.vectors[ik1,ik2,ik3,:,iband]
                        rho    = abs(vecs)**2
                        if E_band < energyAxis[0] - 10*eta or E_band > energyAxis[-1] + 10*eta: continue
                        tdos=np.exp(-((E_band-energyAxis)/eta)**2/2)
                        self.pdos += tdos[:,None] * rho
        self.pdos /= np.sqrt(2*np.pi)*eta*np.prod(self.kgrid)

class Spin(Energy):
    def __init__(self, spinProjectionList=[], gS:float=2.0, _computeSpin=True, **kwarg,):
        '''
        Compute the spin magnetic-moment matrix in eigen-basis, in mu_B.

        Indexing:
        self.spin_K[ik1, ik2, ik3, idirection, ieig, jeig] gives the
        spin magnetic-moment matrix in eigen-basis.
        idirection can be 0 | 1 | 2, corresponding to x | y | z.
        '''
        super().__init__(**kwarg)
        if not _computeSpin:
            return
        vectors=self.vectors[:,:,:,None,:,:]
        spin_operator=-gS*self.model.spin/2
        if self.model.overlap:
            self.spin_K=vectors.conj().swapaxes(-2,-1) @ spin_operator[None,None,None,:,:,:] @ self.O_K_AO[...,None,:,:] @ vectors
        else:
            self.spin_K=vectors.conj().swapaxes(-2,-1) @ spin_operator[None,None,None,:,:,:] @ vectors
        if self.verbose==2: showElapsedTime(extraInfo=f"spin magnetic moment in eig.-basis calculated")

        self.projected_spin_K_list=[]
        for spinProjection in spinProjectionList:
            proj1=projectionOperator(spinProjection['source'],self.model.atomicOrbInfoList,self.model.norbs)
            proj2=projectionOperator(spinProjection['target'],self.model.atomicOrbInfoList,self.model.norbs)
            proj1_K=self.projectorInEigenBasis(proj1)[...,None,:,:]
            proj2_K=self.projectorInEigenBasis(proj2)[...,None,:,:]
            self.projected_spin_K_list.append((proj1_K@self.spin_K + self.spin_K@proj2_K)/2)

class Orbital(Energy):
    def __init__(self, orbitalProjectionList=[], gL:float=1.0, **kwarg):
        '''
        Compute the local orbital magnetic-moment matrix in eigen-basis,
        in mu_B, using the individual-atom approximation.

        Indexing:
        self.orbital_K[ik1, ik2, ik3, idirection, ieig, jeig] gives the
        orbital magnetic-moment matrix in eigen-basis.
        idirection can be 0 | 1 | 2, corresponding to x | y | z.
        '''
        super().__init__(**kwarg)
        vectors=self.vectors[:,:,:,None,:,:]
        orbital_operator=-gL*self.model.orbital
        if self.model.overlap:
            self.orbital_K=vectors.conj().swapaxes(-2,-1) @ orbital_operator[None,None,None,:,:,:] @ self.O_K_AO[...,None,:,:] @ vectors
        else:
            self.orbital_K=vectors.conj().swapaxes(-2,-1) @ orbital_operator[None,None,None,:,:,:] @ vectors
        if self.verbose==2: showElapsedTime(extraInfo=f"local orbital magnetic moment in eig.-basis calculated")

        self.projected_orbital_K_list=[]
        for orbitalProjection in orbitalProjectionList:
            proj1=projectionOperator(orbitalProjection['source'],self.model.atomicOrbInfoList,self.model.norbs)
            proj2=projectionOperator(orbitalProjection['target'],self.model.atomicOrbInfoList,self.model.norbs)
            proj1_K=self.projectorInEigenBasis(proj1)[...,None,:,:]
            proj2_K=self.projectorInEigenBasis(proj2)[...,None,:,:]
            self.projected_orbital_K_list.append((proj1_K@self.orbital_K + self.orbital_K@proj2_K)/2)

class SpinTorque(Spin):
    def __init__(self, spinTorqueProjectionList=[], **kwarg):
        '''
        Compute full spin torque matrix in eigen-basis

        T = -i [mu_S, H]

        In eigen-basis, H is diagonalized eigen-energy and mu_S is computed
        in the parent Spin class in units of mu_B.

        NOTE: the full spin torque includes exchange torque Tau_XC, spin-orbit torque Tau_SO,
        and the spin-flux divergence. Only Tau_XC corresponds to the transfer of electron spin
        to the macroscopic spin magnetization order. 

        Indexing
        --------
        self.spinTorque_K[ik1, ik2, ik3, idirection, ieig, jeig] gives spin torque matrix in eigen-basis.
        idirection can be 0 | 1 | 2, corresponding to x | y | z.
        '''
        super().__init__(**kwarg)

        self.spinTorque_K=-1j*(self.spin_K*self.energies[...,None,None,:] - self.spin_K*self.energies[...,None,:,None])
        if self.verbose==2: showElapsedTime(extraInfo=f"spin torque in eig.-basis finished")

        self.spinTorqueProjectionList=spinTorqueProjectionList
        self.projected_spinTorque_K_list=[]
        for spinTorqueProjection in self.spinTorqueProjectionList:
            proj1=projectionOperator(spinTorqueProjection['source'],self.model.atomicOrbInfoList,self.model.norbs)
            proj2=projectionOperator(spinTorqueProjection['target'],self.model.atomicOrbInfoList,self.model.norbs)
            proj1_K=self.projectorInEigenBasis(proj1)[...,None,:,:]
            proj2_K=self.projectorInEigenBasis(proj2)[...,None,:,:]
            self.projected_spinTorque_K_list.append((proj1_K@self.spinTorque_K + self.spinTorque_K@proj2_K)/2)
        

    def buildSpinTorqueVelocityCommutator(self):
        '''
        Build vector-potential torque vertices for velocity-gauge calculations.

        With H(A) = H0 - A_a v_a + 1/2 A_a A_b w_ab,
        T(A) = -i[mu_S,H(A)] = T0 + A_a T_a + 1/2 A_a A_b T_ab.
        '''
        nbasis = self.energies.shape[-1]
        self.spinTorqueA_K = np.zeros(self.kgrid + (3, 3, nbasis, nbasis), dtype=complex)
        self.spinTorqueAA_K = np.zeros(self.kgrid + (3, 3, 3, nbasis, nbasis), dtype=complex)

        for c in range(3):
            S_c = self.spin_K[..., c, :, :]
            for a in range(3):
                v_a = self.velocity_K[..., a, :, :]
                self.spinTorqueA_K[..., c, a, :, :] = 1j * (S_c @ v_a - v_a @ S_c)
                for b in range(3):
                    w_ab = self.w_K[..., a, b, :, :]
                    self.spinTorqueAA_K[..., c, a, b, :, :] = -1j * (S_c @ w_ab - w_ab @ S_c)

        self.projected_spinTorqueA_K_list = []
        self.projected_spinTorqueAA_K_list = []
        for spinTorqueProjection in self.spinTorqueProjectionList:
            proj1 = projectionOperator(spinTorqueProjection['source'], self.model.atomicOrbInfoList, self.model.norbs)
            proj2 = projectionOperator(spinTorqueProjection['target'], self.model.atomicOrbInfoList, self.model.norbs)
            proj1_K = self.projectorInEigenBasis(proj1)
            proj2_K = self.projectorInEigenBasis(proj2)
            self.projected_spinTorqueA_K_list.append(
                (proj1_K[..., None, None, :, :] @ self.spinTorqueA_K + self.spinTorqueA_K @ proj2_K[..., None, None, :, :]) / 2
            )
            self.projected_spinTorqueAA_K_list.append(
                (proj1_K[..., None, None, None, :, :] @ self.spinTorqueAA_K + self.spinTorqueAA_K @ proj2_K[..., None, None, None, :, :]) / 2
            )
        if self.verbose==2: showElapsedTime(extraInfo=f"velocity corrected spin torque in eig.-basis finished")

    def buildSpinTorqueDipoleCommutator(self):
        '''Build the electric-field torque vertex i[mu_S,r_a] for length-gauge calculations.'''
        nbasis = self.energies.shape[-1]
        self.spinTorqueE_K = np.zeros(self.kgrid + (3, 3, nbasis, nbasis), dtype=complex)
        for c in range(3):
            spin_c = self.spin_K[..., c, :, :]
            for a in range(3):
                r_a = self.r_K[..., a, :, :]
                self.spinTorqueE_K[..., c, a, :, :] = 1j * (spin_c @ r_a - r_a @ spin_c)

        self.projected_spinTorqueE_K_list = []
        for spinTorqueProjection in self.spinTorqueProjectionList:
            proj1 = projectionOperator(spinTorqueProjection['source'], self.model.atomicOrbInfoList, self.model.norbs)
            proj2 = projectionOperator(spinTorqueProjection['target'], self.model.atomicOrbInfoList, self.model.norbs)
            proj1_K = self.projectorInEigenBasis(proj1)[..., None, None, :, :]
            proj2_K = self.projectorInEigenBasis(proj2)[..., None, None, :, :]
            self.projected_spinTorqueE_K_list.append((proj1_K @ self.spinTorqueE_K + self.spinTorqueE_K @ proj2_K) / 2)
        if self.verbose==2: showElapsedTime(extraInfo="Dipole corrected spin torque in eig.-basis finished")

class SpinExchangeTorque(Spin):
    def __init__(self, spinTorqueProjectionList=[], **kwarg):
        '''
        Compute the spin exchange-torque vertex in the eigen-basis.
        PHYSICAL REVIEW RESEARCH 4, L042022 (2022), eq. S11
        Describes transfer of spin angular momentum from electrons to the
        magnetic order parameter.

        For mu_S = -g_S mu_B S/hbar, the paper's
        T_M = -[S,H_XC]/(i hbar) is represented in magnetic-moment units by
        -i[mu_S,H_XC] = g_S mu_B T_M. It is also hbar times the exchange
        contribution to the electronic magnetic-moment rate. H_XC is
        model.exchange_R. Unlike the total Hamiltonian, it is generally not
        diagonal in the eigen-basis, so the commutator is evaluated by matrix
        multiplication.

        This is an energy-like torque in mu_B eV. The LLG variable follows the
        ordered magnetic moment, which is opposite to its angular momentum.
        Converting this vertex to that magnetic-moment rate therefore requires
        the reaction minus sign; IFTDevice and outputIFTcoeffs perform it.

        The exposed spinTorque_K and projected_spinTorque_K_list attributes
        intentionally match SpinTorque, allowing response classes to switch
        torque definitions without changing their tensor contractions.

        Indexing
        --------
        self.spinTorque_K[ik1, ik2, ik3, idirection, ieig, jeig] gives spin torque matrix in eigen-basis.
        idirection can be 0 | 1 | 2, corresponding to x | y | z.
        '''
        super().__init__(**kwarg)
        assert self.model.exchange_R is not None, "SpinExchangeTorque requires model.exchange_R; call model.getExchangeR first"
        assert self.model.exchange_R.shape==self.model.ham_R.shape, f"exchange_R shape {self.model.exchange_R.shape} differs from ham_R shape {self.model.ham_R.shape}"

        exchange_R=self.model.exchange_R
        if self.phase:
            assert len(self.phaseshift)==self.model.nRvec, f"{len(self.phaseshift)=} {self.model.nRvec=}"
            exchange_R=exchange_R*self.phaseshift[:,None,None]
        self.exchange_R_intepolated=np.zeros(self.kgrid+(self.model.norbs,self.model.norbs),dtype=complex)
        for iR,Rvec in enumerate(self.model.iRvec):
            self.exchange_R_intepolated[tuple(Rvec)]=exchange_R[iR]
        self.exchange_K_AO=np.fft.ifftn(self.exchange_R_intepolated,axes=(0,1,2))*np.prod(self.kgrid)
        self.exchange_K=self.vectors.conj().swapaxes(-2,-1) @ self.exchange_K_AO @ self.vectors

        spin=self.spin_K
        exchange=self.exchange_K[...,None,:,:]
        self.spinExchangeTorque_K=-1j*(spin@exchange-exchange@spin)
        self.spinTorque_K=self.spinExchangeTorque_K

        self.spinTorqueProjectionList = spinTorqueProjectionList
        self.projected_spinExchangeTorque_K_list=[]
        for spinTorqueProjection in spinTorqueProjectionList:
            proj1=projectionOperator(spinTorqueProjection['source'],self.model.atomicOrbInfoList,self.model.norbs)
            proj2=projectionOperator(spinTorqueProjection['target'],self.model.atomicOrbInfoList,self.model.norbs)
            proj1_K=self.projectorInEigenBasis(proj1)[...,None,:,:]
            proj2_K=self.projectorInEigenBasis(proj2)[...,None,:,:]
            projected=(proj1_K@self.spinExchangeTorque_K+self.spinExchangeTorque_K@proj2_K)/2
            self.projected_spinExchangeTorque_K_list.append(projected)
        self.projected_spinTorque_K_list=self.projected_spinExchangeTorque_K_list
        if self.verbose==2: showElapsedTime(extraInfo="spin exchange torque in eig.-basis finished")

    def buildSpinTorqueVelocityCommutator(self):
        '''
        Build vector-potential vertices for the exchange torque operator.

        For H_XC(A)=H_XC-A_a v_XC,a+1/2 A_a A_b w_XC,ab, the vertices are
        i[mu_S,v_XC,a] and -i[mu_S,w_XC,ab]. If exchangeVelocity was disabled,
        retain the local field-independent approximation with zero vertices.
        '''
        nbasis=self.energies.shape[-1]
        self.spinExchangeTorqueA_K=np.zeros(self.kgrid+(3,3,nbasis,nbasis),dtype=complex)
        self.spinExchangeTorqueAA_K=np.zeros(self.kgrid+(3,3,3,nbasis,nbasis),dtype=complex)
        if getattr(self,"exchangeVelocity",False):
            assert hasattr(self,"velocity_xc_K") and hasattr(self,"w_xc_K"), \
                "exchange velocity matrices were requested but not constructed"
            for c in range(3):
                spin_c=self.spin_K[...,c,:,:]
                for a in range(3):
                    velocity_xc_a=self.velocity_xc_K[...,a,:,:]
                    self.spinExchangeTorqueA_K[...,c,a,:,:]=1j*(spin_c@velocity_xc_a-velocity_xc_a@spin_c)
                    for b in range(3):
                        w_xc_ab=self.w_xc_K[...,a,b,:,:]
                        self.spinExchangeTorqueAA_K[...,c,a,b,:,:]=-1j*(spin_c@w_xc_ab-w_xc_ab@spin_c)
        self.spinTorqueA_K=self.spinExchangeTorqueA_K
        self.spinTorqueAA_K=self.spinExchangeTorqueAA_K

        self.projected_spinExchangeTorqueA_K_list=[]
        self.projected_spinExchangeTorqueAA_K_list=[]
        for spinTorqueProjection in self.spinTorqueProjectionList:
            proj1=projectionOperator(spinTorqueProjection['source'],self.model.atomicOrbInfoList,self.model.norbs)
            proj2=projectionOperator(spinTorqueProjection['target'],self.model.atomicOrbInfoList,self.model.norbs)
            proj1_K=self.projectorInEigenBasis(proj1)
            proj2_K=self.projectorInEigenBasis(proj2)
            self.projected_spinExchangeTorqueA_K_list.append(
                (proj1_K[...,None,None,:,:]@self.spinExchangeTorqueA_K
                 +self.spinExchangeTorqueA_K@proj2_K[...,None,None,:,:])/2)
            self.projected_spinExchangeTorqueAA_K_list.append(
                (proj1_K[...,None,None,None,:,:]@self.spinExchangeTorqueAA_K
                 +self.spinExchangeTorqueAA_K@proj2_K[...,None,None,None,:,:])/2)
        self.projected_spinTorqueA_K_list=self.projected_spinExchangeTorqueA_K_list
        self.projected_spinTorqueAA_K_list=self.projected_spinExchangeTorqueAA_K_list
        if self.verbose==2: showElapsedTime(extraInfo="velocity corrected spin exchange torque finished")

    def buildSpinTorqueDipoleCommutator(self):
        '''
        Build zero dipole-potential vertices for the exchange torque operator.
    
        H_XC is a field-independent local effective potential. The electric
        field changes its expectation through the driven density matrix, not
        through explicit dipole-dependent exchange-torque vertices.
        '''
        nbasis=self.energies.shape[-1]
        self.spinExchangeTorqueE_K=np.zeros(self.kgrid+(3,3,nbasis,nbasis),dtype=complex)
        self.spinTorqueE_K=self.spinExchangeTorqueE_K
        self.projected_spinExchangeTorqueE_K_list=[np.zeros_like(self.spinExchangeTorqueE_K) for _ in self.spinTorqueProjectionList]
        self.projected_spinTorqueE_K_list=self.projected_spinExchangeTorqueE_K_list
        if self.verbose==2: showElapsedTime(extraInfo="zero dipole corrected spin exchange torque finished")

class OrbitalTorque(Orbital):
    def __init__(self, orbitalTorqueProjectionList=[], **kwarg):
        '''
        Compute orbital torque matrix in eigen-basis

        T = -i [mu_L, H]

        In eigen-basis, H is diagonalized eigen-energy and mu_L is computed
        in the parent Orbital class in units of mu_B.

        Indexing
        --------
        self.orbitalTorque_K[ik1, ik2, ik3, idirection, ieig, jeig] gives orbital torque matrix in eigen-basis.
        idirection can be 0 | 1 | 2, corresponding to x | y | z.
        '''
        super().__init__(**kwarg)

        self.orbitalTorque_K=-1j*(self.orbital_K*self.energies[...,None,None,:] - self.orbital_K*self.energies[...,None,:,None])
        if self.verbose==2: showElapsedTime(extraInfo=f"orbital torque in eig.-basis finished")

        self.orbitalTorqueProjectionList=orbitalTorqueProjectionList
        self.projected_orbitalTorque_K_list=[]
        for orbitalTorqueProjection in self.orbitalTorqueProjectionList:
            proj1=projectionOperator(orbitalTorqueProjection['source'],self.model.atomicOrbInfoList,self.model.norbs)
            proj2=projectionOperator(orbitalTorqueProjection['target'],self.model.atomicOrbInfoList,self.model.norbs)
            proj1_K=self.projectorInEigenBasis(proj1)[...,None,:,:]
            proj2_K=self.projectorInEigenBasis(proj2)[...,None,:,:]
            self.projected_orbitalTorque_K_list.append((proj1_K@self.orbitalTorque_K + self.orbitalTorque_K@proj2_K)/2)

    def buildOrbitalTorqueVelocityCommutator(self):
        '''
        Build vector-potential torque vertices for velocity-gauge calculations.

        With H(A) = H0 - A_a v_a + 1/2 A_a A_b w_ab,
        T(A) = -i[mu_L,H(A)] = T0 + A_a T_a + 1/2 A_a A_b T_ab.
        '''
        nbasis = self.energies.shape[-1]
        self.orbitalTorqueA_K = np.zeros(self.kgrid + (3, 3, nbasis, nbasis), dtype=complex)
        self.orbitalTorqueAA_K = np.zeros(self.kgrid + (3, 3, 3, nbasis, nbasis), dtype=complex)

        for c in range(3):
            L_c = self.orbital_K[..., c, :, :]
            for a in range(3):
                v_a = self.velocity_K[..., a, :, :]
                self.orbitalTorqueA_K[..., c, a, :, :] = 1j * (L_c @ v_a - v_a @ L_c)
                for b in range(3):
                    w_ab = self.w_K[..., a, b, :, :]
                    self.orbitalTorqueAA_K[..., c, a, b, :, :] = -1j * (L_c @ w_ab - w_ab @ L_c)

        self.projected_orbitalTorqueA_K_list = []
        self.projected_orbitalTorqueAA_K_list = []
        for orbitalTorqueProjection in self.orbitalTorqueProjectionList:
            proj1 = projectionOperator(orbitalTorqueProjection['source'], self.model.atomicOrbInfoList, self.model.norbs)
            proj2 = projectionOperator(orbitalTorqueProjection['target'], self.model.atomicOrbInfoList, self.model.norbs)
            proj1_K = self.projectorInEigenBasis(proj1)
            proj2_K = self.projectorInEigenBasis(proj2)
            self.projected_orbitalTorqueA_K_list.append(
                (proj1_K[..., None, None, :, :] @ self.orbitalTorqueA_K + self.orbitalTorqueA_K @ proj2_K[..., None, None, :, :]) / 2
            )
            self.projected_orbitalTorqueAA_K_list.append(
                (proj1_K[..., None, None, None, :, :] @ self.orbitalTorqueAA_K + self.orbitalTorqueAA_K @ proj2_K[..., None, None, None, :, :]) / 2
            )
        if self.verbose==2: showElapsedTime(extraInfo=f"velocity correctd orbital torque in eig.-basis finished")

    def buildOrbitalTorqueDipoleCommutator(self):
        '''Build the electric-field torque vertex i[mu_L,r_a] for length-gauge calculations.'''
        nbasis = self.energies.shape[-1]
        self.orbitalTorqueE_K = np.zeros(self.kgrid + (3, 3, nbasis, nbasis), dtype=complex)
        for c in range(3):
            orbital_c = self.orbital_K[..., c, :, :]
            for a in range(3):
                r_a = self.r_K[..., a, :, :]
                self.orbitalTorqueE_K[..., c, a, :, :] = 1j * (orbital_c @ r_a - r_a @ orbital_c)

        self.projected_orbitalTorqueE_K_list = []
        for orbitalTorqueProjection in self.orbitalTorqueProjectionList:
            proj1 = projectionOperator(orbitalTorqueProjection['source'], self.model.atomicOrbInfoList, self.model.norbs)
            proj2 = projectionOperator(orbitalTorqueProjection['target'], self.model.atomicOrbInfoList, self.model.norbs)
            proj1_K = self.projectorInEigenBasis(proj1)[..., None, None, :, :]
            proj2_K = self.projectorInEigenBasis(proj2)[..., None, None, :, :]
            self.projected_orbitalTorqueE_K_list.append((proj1_K @ self.orbitalTorqueE_K + self.orbitalTorqueE_K @ proj2_K) / 2)
        if self.verbose==2: showElapsedTime(extraInfo="dipole corrected orbital torque in eig.-basis finished")

class Spin_diag(FermiScan):
    def __init__(self, spinProjectionList=[], gS:float=2.0, **kwarg):
        '''
        Compute spin magnetic-moment expectations in eigen-basis, in mu_B.

        Indexing
        --------
        self.spin_diag_K[ik1, ik2, ik3, idirection, ieig] gives the spin
        magnetic-moment expectation of an eigenstate.
        idirection can be 0 | 1 | 2, corresponding to x | y | z.
        '''
        super().__init__(**kwarg)

        vectors=self.vectors[:,:,:,None,:,self.ibmin:self.ibmax+1]
        spin_operator=-gS*self.model.spin/2
        if self.model.overlap:
            _sv=spin_operator[None,None,None,:,:,:] @ self.O_K_AO[...,None,:,:] @ vectors
            self.spin_diag_K=np.sum(vectors.conj() * _sv, axis=(4,))
        else:
            _sv=spin_operator[None,None,None,:,:,:] @ vectors
            self.spin_diag_K=np.sum(vectors.conj() * _sv, axis=(4,))
        if self.verbose==2: showElapsedTime(extraInfo=f"spin magnetic moment of eig.-basis calculated")

        self.projected_spin_diag_K_list=[]
        for spinProjection in spinProjectionList:
            proj1=projectionOperator(spinProjection['source'],self.model.atomicOrbInfoList,self.model.norbs)
            proj2=projectionOperator(spinProjection['target'],self.model.atomicOrbInfoList,self.model.norbs)
            spin_proj=(proj1@spin_operator@proj2 + proj2@spin_operator@proj1)/2
            if self.model.overlap:
                _sv=spin_proj[None,None,None,:,:,:] @ self.O_K_AO[...,None,:,:] @ vectors
                projected_spin_diag_K=np.sum(vectors.conj() * _sv, axis=(4,))
            else:
                _sv=spin_proj[None,None,None,:,:,:] @ vectors
                projected_spin_diag_K=np.sum(vectors.conj() * _sv, axis=(4,))
            self.projected_spin_diag_K_list.append(projected_spin_diag_K)
            #proj1_K=proj1[None,None,None,:,:]@self.vectors[...,self.ibmin:self.ibmax+1]
            #proj2_K=proj2[None,None,None,:,:]@self.vectors[...,self.ibmin:self.ibmax+1]

class Orbital_diag(FermiScan):
    def __init__(self, orbitalProjectionList=[], gL:float=1.0, **kwarg):
        '''
        Compute local orbital magnetic-moment expectations in eigen-basis,
        in mu_B, using the individual-atom approximation.

        Indexing
        --------
        self.orbital_diag_K[ik1, ik2, ik3, idirection, ieig] gives the
        orbital magnetic-moment expectation of an eigenstate.
        idirection can be 0 | 1 | 2, corresponding to x | y | z.
        '''
        super().__init__(**kwarg)

        vectors=self.vectors[:,:,:,None,:,self.ibmin:self.ibmax+1]
        orbital_operator=-gL*self.model.orbital
        if self.model.overlap:
            _ov=orbital_operator[None,None,None,:,:,:] @ self.O_K_AO[...,None,:,:] @ vectors
            self.orbital_diag_K=np.sum(vectors.conj() * _ov, axis=(4,))
        else:
            _ov=orbital_operator[None,None,None,:,:,:] @ vectors
            self.orbital_diag_K=np.sum(vectors.conj() * _ov, axis=(4,))
        if self.verbose==2: showElapsedTime(extraInfo=f"local orbital magnetic moment of eig.-basis calculated")

        self.projected_orbital_diag_K_list=[]
        for orbitalProjection in orbitalProjectionList:
            proj1=projectionOperator(orbitalProjection['source'],self.model.atomicOrbInfoList,self.model.norbs)
            proj2=projectionOperator(orbitalProjection['target'],self.model.atomicOrbInfoList,self.model.norbs)
            orbital_proj=(proj1@orbital_operator@proj2 + proj2@orbital_operator@proj1)/2
            if self.model.overlap:
                _ov=orbital_proj[None,None,None,:,:,:] @ self.O_K_AO[...,None,:,:] @ vectors
                projected_orbital_diag_K=np.sum(vectors.conj() * _ov, axis=(4,))
            else:
                _ov=orbital_proj[None,None,None,:,:,:] @ vectors
                projected_orbital_diag_K=np.sum(vectors.conj() * _ov, axis=(4,))
            self.projected_orbital_diag_K_list.append(projected_orbital_diag_K)

class Velocity(Energy):
    def __init__(self, velocityProjectionList=[], highOrderVelocity=False, exchangeVelocity=False, **kwarg) -> None:
        '''
        Compute velocity matrix in original basis and eigen-basis
        And the velocity projected in specific basis

        Indexing:
        self.RHam_K[ik1, ik2, ik3, idirection, iorb, jorb] gives velocity matrix in original basis,
        idirection can be 0 | 1 | 2 corresponding to x | y | z direction.

        self.velocity_K[ik1, ik2, ik3, idirection, ieig, jeig] gives velocity matrix in eigen-basis.
        idirection can be 0 | 1 | 2 corresponding to x | y | z direction.

        self.projected_velocity_K_list[iproj] gives each projected velocity matrix
        '''
        self.exchangeVelocity=exchangeVelocity
        highOrderVelocity=bool(highOrderVelocity or exchangeVelocity)
        super().__init__(**kwarg)
        
        RH_R_W=self.model.RHam_R
        self.RHam_R_intepolated=np.zeros(self.kgrid+(3,self.model.norbs,self.model.norbs),dtype=complex)
        AA_R_W=self.model.AA_R_W # Berry connection in local gauge
        AA_R_W_intepolated=np.zeros(self.kgrid+(3,self.model.norbs,self.model.norbs),dtype=complex)
        if self.phase:
            RH_R_W=RH_R_W*self.phaseshift[:,None,None,None]
            AA_R_W=AA_R_W*self.phaseshift[:,None,None,None]
        for iR, Rvec in enumerate(self.model.iRvec):
            self.RHam_R_intepolated[tuple(Rvec)]=RH_R_W[iR]
            AA_R_W_intepolated[tuple(Rvec)]=AA_R_W[iR]
        self.RHam_K= 1j*np.fft.ifftn(self.RHam_R_intepolated,axes=(0,1,2,))*np.prod(self.kgrid)
        if self.verbose==2: showElapsedTime(extraInfo=f"FFT RHam_R to RHam_K to calc. velocity in ao-basis finished")
        vectors=self.vectors[:,:,:,None,:,:]
        velocity_bare_K=vectors.conj().swapaxes(-2,-1) @ self.RHam_K @ vectors

        # ==========================================================
        # Gauge-Covariant Geometric Corrections for velocity
        # ==========================================================
        # v_a = H^(1)_a - i[r_a, H^(0)]
        AA_K_W        =  np.fft.ifftn(AA_R_W_intepolated,axes=(0,1,2,))*np.prod(self.kgrid) # Berry connection in Bloch gauge
        AA_K_H = vectors.conj().swapaxes(-2,-1) @ AA_K_W @ vectors # Berry connection in Hamiltonian gauge
        AA_K_H_dot_H = AA_K_H * self.energies[...,None,None,:]            # gauge correction term, the commutator [AA_K, Ham.]
        self.velocity_K = velocity_bare_K - 1j*(AA_K_H_dot_H - AA_K_H_dot_H.conj().swapaxes(-2,-1)) # gauge-covariant velocity

        if exchangeVelocity:
            assert self.model.exchange_R is not None, "exchangeVelocity requires model.exchange_R; call model.getExchangeR first"
            assert self.model.RExchange_R is not None, "RExchange_R is required; call model.getRExchange_R first"
            assert self.model.RRExchange_R is not None, "RRExchange_R is required; call model.getRRExchange_R first"

            exchange_R_W=self.model.exchange_R
            RExchange_R_W=self.model.RExchange_R
            RRExchange_R_W=self.model.RRExchange_R
            if self.phase:
                exchange_R_W=exchange_R_W*self.phaseshift[:,None,None]
                RExchange_R_W=RExchange_R_W*self.phaseshift[:,None,None,None]
                RRExchange_R_W=RRExchange_R_W*self.phaseshift[:,None,None,None,None]

            exchange_R_interpolated=np.zeros(self.kgrid+(self.model.norbs,self.model.norbs),dtype=complex)
            RExchange_R_interpolated=np.zeros(self.kgrid+(3,self.model.norbs,self.model.norbs),dtype=complex)
            RRExchange_R_interpolated=np.zeros(self.kgrid+(3,3,self.model.norbs,self.model.norbs),dtype=complex)
            for iR,Rvec in enumerate(self.model.iRvec):
                exchange_R_interpolated[tuple(Rvec)]=exchange_R_W[iR]
                RExchange_R_interpolated[tuple(Rvec)]=RExchange_R_W[iR]
                RRExchange_R_interpolated[tuple(Rvec)]=RRExchange_R_W[iR]

            fft_scale=np.prod(self.kgrid)
            self.exchange_K_AO=np.fft.ifftn(exchange_R_interpolated,axes=(0,1,2))*fft_scale
            self.RExchange_K=1j*np.fft.ifftn(RExchange_R_interpolated,axes=(0,1,2))*fft_scale
            self.RRExchange_K=-np.fft.ifftn(RRExchange_R_interpolated,axes=(0,1,2))*fft_scale
            del exchange_R_interpolated,RExchange_R_interpolated,RRExchange_R_interpolated

            vectors0=self.vectors
            self.exchange_K=vectors0.conj().swapaxes(-2,-1) @ self.exchange_K_AO @ vectors0
            vectors1=self.vectors[...,None,:,:]
            self.velocity_xc_bare_K=vectors1.conj().swapaxes(-2,-1) @ self.RExchange_K @ vectors1
            self.velocity_xc_K=self.velocity_xc_bare_K-1j*(AA_K_H@self.exchange_K[...,None,:,:]
                -self.exchange_K[...,None,:,:]@AA_K_H)
            vectors2=self.vectors[...,None,None,:,:]
            self.w_xc_bare_K=vectors2.conj().swapaxes(-2,-1) @ self.RRExchange_K @ vectors2
            del vectors0,vectors1,vectors2
            if self.verbose==2: showElapsedTime(extraInfo=f"bare and first-order exchange velocity finished")

        if self.verbose==2: showElapsedTime(extraInfo=f"velocity modification with position operators finished")
        # check hermity
        #dconj=np.sum(abs(self.velocity_K.transpose((0,1,2,3,5,4)).conj()-self.velocity_K))
        #assert dconj<1e-4, f"Hermity of velocity is not satisfied {dconj=}"

        '''
        # extra modifications to velocity due to orbital overlaps
        # according to Gemini, this part is not necessary since the effect is automatically included in the dipole correction
        if self.model.overlap:
            RO_R_W=self.model.RO_R
            self.RO_R_intepolated=np.zeros(self.kgrid+(3,self.model.norbs,self.model.norbs),dtype=complex)
            if self.phase:
                RO_R_W=RO_R_W*self.phaseshift[:,None,None,None]
            for iR, Rvec in enumerate(self.model.iRvec):
                self.RO_R_intepolated[tuple(Rvec)]=RO_R_W[iR]
            self.RO_K= 1j*np.fft.ifftn(self.RO_R_intepolated,axes=(0,1,2,))*np.prod(self.kgrid)
            __UdOkU = (vectors.conj().swapaxes(-2,-1) @ self.RO_K @ vectors)*self.energies[...,None,:,None]
            self.velocity_K -= (__UdOkU + __UdOkU.conj().swapaxes(-2,-1))/2
        #'''

        self.projected_velocity_K_list=[]
        for velocityProjection in velocityProjectionList:
            proj1=projectionOperator(velocityProjection['source'],self.model.atomicOrbInfoList,self.model.norbs)
            proj2=projectionOperator(velocityProjection['target'],self.model.atomicOrbInfoList,self.model.norbs)

            proj1_K=self.projectorInEigenBasis(proj1)[...,None,:,:]
            proj2_K=self.projectorInEigenBasis(proj2)[...,None,:,:]
            self.projected_velocity_K_list.append((proj1_K@self.velocity_K@proj2_K + proj2_K@self.velocity_K@proj1_K)/2)

        if highOrderVelocity:
            RRH_R_W  = getattr(self.model, "RRHam_R", None)
            RRRH_R_W = getattr(self.model, "RRRHam_R", None)
            assert RRH_R_W is not None,  "RRHam_R is required for high order velocity,  execute getRRHam_R first"
            assert RRRH_R_W is not None, "RRRHam_R is required for high order velocity, execute getRRRHam_R first"
        
            RRH_R_intepolated  = np.zeros(self.kgrid + (3, 3, self.model.norbs, self.model.norbs), dtype=complex)
            RRRH_R_intepolated = np.zeros(self.kgrid + (3, 3, 3, self.model.norbs, self.model.norbs), dtype=complex)
        
            if self.phase:
                RRH_R_W  = RRH_R_W  * self.phaseshift[:, None, None, None, None]
                RRRH_R_W = RRRH_R_W * self.phaseshift[:, None, None, None, None, None]
                
            for iR, Rvec in enumerate(self.model.iRvec):
                RRH_R_intepolated[ tuple(Rvec)] = RRH_R_W[iR]
                RRRH_R_intepolated[tuple(Rvec)] = RRRH_R_W[iR]
                
            # w_K = d^2H/dk^2
            RRHam_K  = -1 * np.fft.ifftn(RRH_R_intepolated, axes=(0,1,2,)) * np.prod(self.kgrid)
            vectors=self.vectors[:,:,:,None,None,:,:]
            self.w_K = vectors.conj().swapaxes(-2,-1) @ RRHam_K @ vectors
        
            # u_K = d^3H/dk^3
            RRRHam_K = -1j * np.fft.ifftn(RRRH_R_intepolated, axes=(0,1,2,)) * np.prod(self.kgrid)
            vectors=self.vectors[:,:,:,None,None,None,:,:]
            self.u_K = vectors.conj().swapaxes(-2,-1) @ RRRHam_K @ vectors

            # ==============================================================
            # Gauge-Covariant Geometric Corrections for high order velocity
            # ==============================================================
            def comm(A, B):
                # Helper to compute commutator [A, B] broadcasting over k-grid
                return A @ B - B @ A

            # Ordinary k derivatives of the Wannier-gauge connection.
            # These terms vanish in the diagonal tight-binding limit, but are
            # required when the off-diagonal position matrix depends on k.
            r1_R_intepolated = np.zeros(self.kgrid + (3, 3, self.model.norbs, self.model.norbs), dtype=complex)
            for iR, Rvec in enumerate(self.model.iRvec):
                R_cart = Rvec @ self.model.lattice
                r1_R_intepolated[tuple(Rvec)] = 1j * R_cart[:, None, None, None] * AA_R_W[iR][None, :, :, :]

            r1_K_AO = np.fft.ifftn(r1_R_intepolated, axes=(0, 1, 2)) * np.prod(self.kgrid)
            vectors1 = self.vectors[..., None, None, :, :]
            connectionDerivative1 = vectors1.conj().swapaxes(-2, -1) @ r1_K_AO @ vectors1
            del r1_R_intepolated, r1_K_AO, vectors1

            # Construct the larger second-derivative tensor only after the
            # first-derivative FFT buffers have been released.
            r2_R_intepolated = np.zeros(self.kgrid + (3, 3, 3, self.model.norbs, self.model.norbs), dtype=complex)
            for iR, Rvec in enumerate(self.model.iRvec):
                R_cart = Rvec @ self.model.lattice
                r2_R_intepolated[tuple(Rvec)] = (
                    -R_cart[:, None, None, None, None] * R_cart[None, :, None, None, None] * AA_R_W[iR][None, None, :, :, :]
                )
            r2_K_AO = np.fft.ifftn(r2_R_intepolated, axes=(0, 1, 2)) * np.prod(self.kgrid)
            vectors2 = self.vectors[..., None, None, None, :, :]
            connectionDerivative2 = vectors2.conj().swapaxes(-2, -1) @ r2_K_AO @ vectors2
            del r2_R_intepolated, r2_K_AO, vectors2

            # Store the bare spatial moments and H in the eigenbasis.
            H0 = self.energies[..., :, None] * np.eye(self.energies.shape[-1], dtype=complex)
            H1 = velocity_bare_K
            H2 = self.w_K.copy()

            # partialV_ab = partial_b v_a
            partialV = np.zeros_like(self.w_K)
            for a in range(3):
                for b in range(3):
                    partialV[..., a, b, :, :] = (
                        H2[..., a, b, :, :]
                        - 1j * comm(connectionDerivative1[..., b, a, :, :], H0)
                        - 1j * comm(AA_K_H[..., a, :, :], H1[..., b, :, :])
                    )
                    # w_ab = D_b v_a
                    self.w_K[..., a, b, :, :] = (partialV[..., a, b, :, :] 
                        - 1j * comm(AA_K_H[..., b, :, :], self.velocity_K[..., a, :, :])
                    )

            if exchangeVelocity:
                exchange0=self.exchange_K
                exchange1=self.velocity_xc_bare_K
                exchange2=self.w_xc_bare_K
                self.w_xc_K=np.zeros_like(exchange2)
                for a in range(3):
                    for b in range(3):
                        partialVxc=(exchange2[...,a,b,:,:]
                            -1j*comm(connectionDerivative1[...,b,a,:,:],exchange0)
                            -1j*comm(AA_K_H[...,a,:,:],exchange1[...,b,:,:]))
                        self.w_xc_K[...,a,b,:,:]=(partialVxc
                            -1j*comm(AA_K_H[...,b,:,:],self.velocity_xc_K[...,a,:,:]))
                if self.verbose==2: showElapsedTime(extraInfo=f"second-order exchange velocity finished")

            # u_abc = D_c w_ab.  connectionDerivative1[..., b, a]
            # is partial_b A_a, and connectionDerivative2[..., c, b, a]
            # is partial_c partial_b A_a.
            H3 = self.u_K.copy()
            for a in range(3):
                for b in range(3):
                    for c in range(3):
                        partial2V = (H3[..., a, b, c, :, :]  
                            - 1j * comm(connectionDerivative2[..., c, b, a, :, :], H0)
                            - 1j * comm(connectionDerivative1[..., b, a, :, :], H1[..., c, :, :])
                            - 1j * comm(connectionDerivative1[..., c, a, :, :], H1[..., b, :, :])
                            - 1j * comm(AA_K_H[..., a, :, :], H2[..., b, c, :, :])
                        )
                        partialW = (partial2V
                            - 1j * comm(connectionDerivative1[..., c, b, :, :], self.velocity_K[..., a, :, :])
                            - 1j * comm(AA_K_H[..., b, :, :], partialV[..., a, c, :, :])
                        )
                        self.u_K[..., a, b, c, :, :] = (partialW
                            - 1j * comm(AA_K_H[..., c, :, :], self.w_K[..., a, b, :, :])
                        )
            del connectionDerivative1, connectionDerivative2, partialV
            if self.verbose==2: showElapsedTime(extraInfo=f"higer order velocity modification with position operators finished")

class Dipole(Velocity):
    def __init__(self, highOrderDipole=False, **kwarg) -> None:
        '''
        Compute the off-diagonal dipole matrix in the Hamiltonian gauge.

        Indexing
        --------
        self.r_K[ik1, ik2, ik3, idirection, m, n] gives
        r_mn = v_mn / (i * (E_m - E_n)). Diagonal and numerically
        degenerate matrix elements are set to zero.

        If highOrderDipole is True,
        self.r_deriv_K[ik1, ik2, ik3, ideriv, idirection, m, n]
        gives the generalized derivative D_ideriv r^idirection_mn.
        '''
        highOrderVelocity = bool(kwarg.pop("highOrderVelocity", False) or highOrderDipole)
        super().__init__(highOrderVelocity=highOrderVelocity, **kwarg)

        norbs = self.energies.shape[-1]
        self.calcDeltaEInverse()

        off_diag = 1.0 - np.eye(norbs)
        self.r_K = self.velocity_K * self.deltaEInverse_pure[..., None, :, :]  / 1j * off_diag
        if self.verbose==2: showElapsedTime(extraInfo=f"dipole operators (off diagonal) finished")

        if highOrderDipole:
            velocity_diag = np.diagonal(self.velocity_K, axis1=-2, axis2=-1)
            deltaVelocity = velocity_diag[..., :, :, None] - velocity_diag[..., :, None, :]
            self.r_deriv_K = np.zeros(self.kgrid + (3, 3, norbs, norbs), dtype=complex)
            for ideriv in range(3):
                r_derivative_direction = self.r_K[..., ideriv, :, :]
                deltaVelocity_deriv = deltaVelocity[..., ideriv, :, :]
                for idirection in range(3):
                    velocity_direction = self.velocity_K[..., idirection, :, :]
                    commutator = r_derivative_direction @ velocity_direction \
                        - velocity_direction @ r_derivative_direction
                    self.r_deriv_K[..., ideriv, idirection, :, :] = (
                        self.w_K[..., ideriv, idirection, :, :]
                        - velocity_direction * deltaVelocity_deriv * self.deltaEInverse_pure
                        + 1j * commutator
                    ) * self.deltaEInverse_pure / 1j * off_diag
            if self.verbose==2: showElapsedTime(extraInfo=f"generalized k-derivatives of dipole operators finished")

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

        f = self.fd_occupations[ifermi]
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

class OrbitalModern_LC(Velocity):
    def __init__(self, orbitalProjectionList=[], _computeOrbitalModernLC=True, **kwarg,):
        '''
        Compute the local-circulation (LC), or Bloch-state self-rotation,
        orbital magnetic-moment matrix from Eq. S9 of PRL 136, 236302
        (2026).

        The velocity matrices used by this code are dH/dk = hbar*v_phys in
        eV*angstrom. ``orbital_K`` is returned in units of mu_B:

        orbital_K[ik1, ik2, ik3, idirection, iband, jband].
        '''
        super().__init__(**kwarg)
        if not _computeOrbitalModernLC:
            return

        nbands = self.energies.shape[-1]
        moment_sum = np.zeros(self.kgrid + (3, nbands, nbands), dtype=complex)
        cross_pairs = ((1, 2), (2, 0), (0, 1))

        for p in range(nbands):
            gap_p_band = self.energies[..., p, None] - self.energies
            valid_gap = np.abs(gap_p_band) > self.degen
            inverse_gap = np.zeros_like(gap_p_band)
            inverse_gap[valid_gap] = 1.0 / gap_p_band[valid_gap]

            weight = inverse_gap[..., :, None] + inverse_gap[..., None, :]
            weight *= valid_gap[..., :, None] & valid_gap[..., None, :]

            for component, (a, b) in enumerate(cross_pairs):
                v_a_lp = self.velocity_K[..., a, :, p]
                v_b_lp = self.velocity_K[..., b, :, p]
                v_a_pn = self.velocity_K[..., a, p, :]
                v_b_pn = self.velocity_K[..., b, p, :]
                velocity_cross = (
                    v_a_lp[..., :, None] * v_b_pn[..., None, :]
                    - v_b_lp[..., :, None] * v_a_pn[..., None, :]
                )
                moment_sum[..., component, :, :] += weight * velocity_cross

        # hbar^2/(2*m_e), expressed in eV*angstrom^2.  With velocity_K=dH/dk,
        # Eq. S9 divided by mu_B has the prefactor 1/(4*i*constant).
        hbar2_over_2me = hbar**2 / (2 * electron_mass * elementary_charge * angstrom**2)
        self.orbital_K = moment_sum / (4j * hbar2_over_2me)
        self.orbital_K = (
            self.orbital_K +
            self.orbital_K.conj().swapaxes(-2, -1)
        ) / 2

        self.projected_orbital_K_list = []
        for orbitalProjection in orbitalProjectionList:
            proj1 = projectionOperator(
                orbitalProjection['source'],
                self.model.atomicOrbInfoList,
                self.model.norbs,
            )
            proj2 = projectionOperator(
                orbitalProjection['target'],
                self.model.atomicOrbInfoList,
                self.model.norbs,
            )
            proj1_K = self.projectorInEigenBasis(proj1)[..., None, :, :]
            proj2_K = self.projectorInEigenBasis(proj2)[..., None, :, :]
            projected_orbital_K = (
                proj1_K @ self.orbital_K
                + self.orbital_K @ proj2_K
            ) / 2
            self.projected_orbital_K_list.append(projected_orbital_K)

        if self.verbose == 2:
            showElapsedTime(extraInfo="modern LC orbital magnetic moment in eig.-basis finished")

class Velocity_diag(FermiScan):
    def __init__(self, velocityProjectionList=[], **kwarg) -> None:
        '''
        Compute group velocity in original basis and eigen-basis

        Indexing
        --------
        self.RHam_K[ik1, ik2, ik3, idirection, iorb, jorb] gives velocity matrix in original basis,
        idirection can be 0 | 1 | 2 corresponding to x | y | z direction.

        self.velocity_diag_K[ik1, ik2, ik3, idirection, ieig] gives group velocity of each eigen-basis.
        idirection can be 0 | 1 | 2 corresponding to x | y | z direction.
        '''
        super().__init__(**kwarg)
        
        RH_R_W=self.model.RHam_R
        self.RHam_R_intepolated=np.zeros(self.kgrid+(3,self.model.norbs,self.model.norbs),dtype=complex)
        if self.phase:
            RH_R_W=RH_R_W*self.phaseshift[:,None,None,None]
        for iR, Rvec in enumerate(self.model.iRvec):
            self.RHam_R_intepolated[tuple(Rvec)]=RH_R_W[iR]
        self.RHam_K= 1j*np.fft.ifftn(self.RHam_R_intepolated,axes=(0,1,2,))*np.prod(self.kgrid)

        if self.verbose==2: showElapsedTime(extraInfo=f"FFT RHam_R to RHam_K to calc. velocity in ao-basis finished")
        vectors=self.vectors[:,:,:,None,:,self.ibmin:self.ibmax+1]
        _RHam_K_vector=self.RHam_K @ vectors
        self.velocity_diag_K=np.sum(vectors.conj() * _RHam_K_vector, axis=(4,))
        if self.verbose==2: showElapsedTime(extraInfo=f"group velocity of eig.-basis finished")

        # extra modifications to velocity due to orbital overlaps
        if self.model.overlap:
            RO_R_W=self.model.RO_R
            self.RO_R_intepolated=np.zeros(self.kgrid+(3,self.model.norbs,self.model.norbs),dtype=complex)
            if self.phase:
                RO_R_W=RO_R_W*self.phaseshift[:,None,None,None]
            for iR, Rvec in enumerate(self.model.iRvec):
                self.RO_R_intepolated[tuple(Rvec)]=RO_R_W[iR]
            self.RO_K= 1j*np.fft.ifftn(self.RO_R_intepolated,axes=(0,1,2,))*np.prod(self.kgrid)
            _RO_K_vector=self.RO_K @ (vectors *self.energies[...,None,None,self.ibmin:self.ibmax+1])
            self.velocity_diag_K -= np.sum(vectors.conj() *_RO_K_vector, axis=(4,))

        self.projected_velocity_diag_K_list=[]
        for velocityProjection in velocityProjectionList:
            proj1=projectionOperator(velocityProjection['source'],self.model.atomicOrbInfoList,self.model.norbs)
            proj2=projectionOperator(velocityProjection['target'],self.model.atomicOrbInfoList,self.model.norbs)
            proj1=proj1[None,None,None,None,:,:]
            proj2=proj2[None,None,None,None,:,:]
            RHam_K_proj=(proj1@self.RHam_K@proj2 + proj2@self.RHam_K@proj1)/2
            _RHam_K_proj_vector=RHam_K_proj @ vectors
            projected_velocity_diag_K=np.sum(vectors.conj() * _RHam_K_proj_vector, axis=(4,))

            if self.model.overlap:
                RO_K_proj=(proj1@self.RO_K@proj2 + proj2@self.RO_K@proj1)/2
                _RO_K_proj_vector=RO_K_proj @ (vectors *self.energies[...,None,None,:])
                projected_velocity_diag_K -= np.sum(vectors.conj() *_RO_K_proj_vector, axis=(4,))
            self.projected_velocity_diag_K_list.append(projected_velocity_diag_K)

class SpinVelocity(Velocity, Spin):
    def __init__(self, velocityProjectionList=[], spinvelocityProjectionList=[], **kwarg) -> None:
        '''
        Compute spin velocity matrix in original basis and eigen-basis
        and projected spin velocity matrix.
        Note, the projection is on the velocity component, corresponding to the spin-currents running in specific basis.

        Indexing:
        self.SRHam_K[ik1, ik2, ik3, idirection, igamma, iorb, jorb] gives spin velocity matrix in original basis,
        idirection can be 0 | 1 | 2, corresponding to x | y | z direction. 
        igamma can be 0 | 1 | 2, corresponding to x | y | z spin magnetic moment.

        self.spinVelocity_K[ik1, ik2, ik3, idirection, igamma, ieig, jeig] gives spin velocity matrix in eigen-basis.
        idirection can be 0 | 1 | 2, corresponding to x | y | z direction. 
        igamma can be 0 | 1 | 2, corresponding to x | y | z spin magnetic moment.

        if spinvelocityProjctionList is not empty
        self.spinVelocity_K_proj[iproj] gives each projected spin velocity matrix in eigen-basis.
        '''
        super().__init__(velocityProjectionList=velocityProjectionList, **kwarg)
        
        __sV=self.spin_K[...,None,:,:,:] @ self.velocity_K[...,None,:,:]
        self.spinVelocity_K = (__sV + __sV.swapaxes(-2,-1).conj())/2

        self.projected_spinVelocity_K_list=[]
        #for iproj, proj in enumerate(spinvelocityProjectionList):
        #    projected_velocity_K=self.projected_velocity_K_list.pop(iproj+len(velocityProjectionList))
        #    __sV=self.spin_K[...,None,:,:,:] @ projected_velocity_K[...,None,:,:]
        #    self.projected_spinVelocity_K_list.append((__sV + __sV.swapaxes(-2,-1).conj())/2)
        for spinvelocityProj in enumerate(spinvelocityProjectionList):
            proj1=projectionOperator(spinvelocityProj['source'],self.model.atomicOrbInfoList,self.model.norbs)
            proj2=projectionOperator(spinvelocityProj['target'],self.model.atomicOrbInfoList,self.model.norbs)
            
            proj1_K=self.projectorInEigenBasis(proj1)[...,None,None,:,:]
            proj2_K=self.projectorInEigenBasis(proj2)[...,None,None,:,:]
            self.projected_spinVelocity_K_list.append((proj1_K@self.spinVelocity_K@proj2_K + proj2_K@self.spinVelocity_K@proj1_K)/2)
        
        if self.verbose==2: showElapsedTime(extraInfo=f"spin velocity in eig.-basis finished")

class OrbitalVelocity(Velocity, Orbital):
    def __init__(self, velocityProjectionList=[], orbitalvelocityProjectionList=[], **kwarg) -> None:
        '''
        Compute orbital velocity matrix in original basis and eigen-basis
        and projected orbital velocity matrix.
        Note, the projection is on the velocity component, corresponding to the orbital-currents running in specific basis.

        Indexing:
        self.ORHam_K[ik1, ik2, ik3, idirection, igamma, iorb, jorb] gives orbital velocity matrix in original basis,
        idirection can be 0 | 1 | 2, corresponding to x | y | z direction. 
        igamma can be 0 | 1 | 2, corresponding to x | y | z orbital magnetic moment.

        self.orbitalVelocity_K[ik1, ik2, ik3, idirection, igamma, ieig, jeig] gives orbital velocity matrix in eigen-basis.
        idirection can be 0 | 1 | 2, corresponding to x | y | z direction. 
        igamma can be 0 | 1 | 2, corresponding to x | y | z orbital magnetic moment.

        if orbitalvelocityProjctionList is not empty
        self.orbitalVelocity_K_proj[iproj] gives each projected orbital velocity matrix in eigen-basis.
        '''
        super().__init__(velocityProjectionList=velocityProjectionList, **kwarg)
        
        __oV=self.orbital_K[...,None,:,:,:] @ self.velocity_K[...,None,:,:]
        self.orbitalVelocity_K = (__oV + __oV.swapaxes(-2,-1).conj())/2

        self.projected_orbitalVelocity_K_list=[]
        for iproj, orbitalvelocityProj in enumerate(orbitalvelocityProjectionList):
            #projected_velocity_K=self.projected_velocity_K_list.pop(iproj+len(velocityProjectionList))
            #__oV=self.orbital_K[...,None,:,:,:] @ projected_velocity_K[...,None,:,:]
            #self.projected_orbitalVelocity_K_list.append((__oV + __oV.swapaxes(-2,-1).conj())/2)
            proj1=projectionOperator(orbitalvelocityProj['source'],self.model.atomicOrbInfoList,self.model.norbs)
            proj2=projectionOperator(orbitalvelocityProj['target'],self.model.atomicOrbInfoList,self.model.norbs)
            
            proj1_K=self.projectorInEigenBasis(proj1)[...,None,None,:,:]
            proj2_K=self.projectorInEigenBasis(proj2)[...,None,None,:,:]
            self.projected_orbitalVelocity_K_list.append((proj1_K@self.orbitalVelocity_K@proj2_K + proj2_K@self.orbitalVelocity_K@proj1_K)/2)
        
        if self.verbose==2: showElapsedTime(extraInfo=f"orbital velocity in eig.-basis finished")

class SpinVelocity_diag(Velocity, Spin):
    def __init__(self, velocityProjectionList=[], spinvelocityProjectionList=[], **kwarg) -> None:
        '''
        Compute the diagonal elements of spin velocity matrix in original basis and eigen-basis
        and projected spin velocity matrix.
        Note, the projection is on the velocity component, corresponding to the spin-currents running in specific basis.

        Indexing:
        self.SRHam_K[ik1, ik2, ik3, idirection, igamma, iorb, jorb] gives spin velocity matrix in original basis,
        idirection can be 0 | 1 | 2, corresponding to x | y | z direction. 
        igamma can be 0 | 1 | 2, corresponding to x | y | z spin magnetic moment.

        self.spinVelocity_K[ik1, ik2, ik3, idirection, igamma, ieig, jeig] gives spin velocity matrix in eigen-basis.
        idirection can be 0 | 1 | 2, corresponding to x | y | z direction. 
        igamma can be 0 | 1 | 2, corresponding to x | y | z spin magnetic moment.

        if spinvelocityProjctionList is not empty
        self.spinVelocity_K_proj[iproj] gives each projected spin velocity matrix in eigen-basis.
        '''
        super().__init__(velocityProjectionList=velocityProjectionList+spinvelocityProjectionList, **kwarg)
        
        self.spinVelocity_diag_K = np.sum(self.spin_K[...,None,:,:,:] * self.velocity_K[...,None,:,:].swapaxes(-2,-1),axis=-1).real

        self.projected_spinVelocity_diag_K_list=[]
        for iproj, proj in enumerate(spinvelocityProjectionList):
            projected_velocity_K=self.projected_velocity_K_list.pop(iproj+len(velocityProjectionList))
            self.projected_spinVelocity_diag_K_list.append(np.sum(self.spin_K[...,None,:,:,:] * projected_velocity_K[...,None,:,:].swapaxes(-2,-1),axis=-1).real)
        
        if self.verbose==2: showElapsedTime(extraInfo=f"diagonal elements of spin velocity in eig.-basis finished")

class OrbitalVelocity_diag(Velocity, Orbital):
    def __init__(self, velocityProjectionList=[], orbitalvelocityProjectionList=[], **kwarg) -> None:
        '''
        Compute the diagonal elements of orbital velocity matrix in original basis and eigen-basis
        and projected orbital velocity matrix.
        Note, the projection is on the velocity component, corresponding to the orbital-currents running in specific basis.

        Indexing:
        self.ORHam_K[ik1, ik2, ik3, idirection, igamma, iorb, jorb] gives orbital velocity matrix in original basis,
        idirection can be 0 | 1 | 2, corresponding to x | y | z direction. 
        igamma can be 0 | 1 | 2, corresponding to x | y | z orbital magnetic moment.

        self.orbitalVelocity_K[ik1, ik2, ik3, idirection, igamma, ieig, jeig] gives orbital velocity matrix in eigen-basis.
        idirection can be 0 | 1 | 2, corresponding to x | y | z direction. 
        igamma can be 0 | 1 | 2, corresponding to x | y | z orbital magnetic moment.

        if orbitalvelocityProjctionList is not empty
        self.orbitalVelocity_K_proj[iproj] gives each projected orbital velocity matrix in eigen-basis.
        '''
        super().__init__(velocityProjectionList=velocityProjectionList+orbitalvelocityProjectionList, **kwarg)
        
        self.orbitalVelocity_K = np.sum(self.orbital_K[...,None,:,:,:] * self.velocity_K[...,None,:,:].swapaxes(-2,-1),axis=-1).real

        self.projected_orbitalVelocity_K_list=[]
        for iproj, proj in enumerate(orbitalvelocityProjectionList):
            projected_velocity_K=self.projected_velocity_K_list.pop(iproj+len(velocityProjectionList))
            self.projected_orbitalVelocity_K_list.append(np.sum(self.orbital_K[...,None,:,:,:] * projected_velocity_K[...,None,:,:].swapaxes(-2,-1),axis=-1).real)
        
        if self.verbose==2: showElapsedTime(extraInfo=f"diagonal elements of orbital velocity in eig.-basis finished")
        
class BerryCurv(Velocity):
    
    def __init__(self, velocityProjectionList=[],
                 _simultaneousVelocityProjection=False, **kwarg) -> None:
        '''
        Calc. Berry curvature \Omega_{alpha,beta}_{k,mn}.

        alpha and beta can be x, y, z corresponds the (cart.) space direction, k denote the crystal moment, m, n index the bands.

        Indexing: BerryCurv[ik1, ik2, ik3, ialpha, ibeta, iband, jband]
        '''
        super().__init__(velocityProjectionList=velocityProjectionList, **kwarg)
        self.calcDeltaEInverse()
        dEInverse2 = self.deltaEInverse2 # The real response kernel, \frac{1}{(Em - En)(Em - En - j \eta)}

        velocity_K_list = [self.velocity_K]+self.projected_velocity_K_list
        berry_shape = self.kgrid+(3,3,self.energies.shape[-1])
        if _simultaneousVelocityProjection:
            self.projected_BerryCurv=np.zeros((len(velocity_K_list),)+berry_shape)
            velocity_pairs = (
                (iprojection, iprojection, velocity_K, velocity_K)
                for iprojection, velocity_K in enumerate(velocity_K_list)
            )
        else:
            self.projected_BerryCurv=np.zeros(
                (len(velocity_K_list),len(velocity_K_list))+berry_shape
            )
            velocity_pairs = (
                (ivelocity_alpha, ivelocity_beta,
                 proj_velocity_alpha_K, proj_velocity_beta_K)
                for ivelocity_alpha, proj_velocity_alpha_K in enumerate(velocity_K_list)
                for ivelocity_beta, proj_velocity_beta_K in enumerate(velocity_K_list)
            )

        for ivelocity_alpha, ivelocity_beta, proj_velocity_alpha_K, proj_velocity_beta_K in velocity_pairs:
            if _simultaneousVelocityProjection:
                berry_curv = self.projected_BerryCurv[ivelocity_alpha]
            else:
                berry_curv = self.projected_BerryCurv[ivelocity_alpha,ivelocity_beta]
            for ialpha in range(3):
                for ibeta in range(3):
                    berry_curv[...,ialpha,ibeta,:]=-2*np.diagonal(
                        proj_velocity_beta_K[...,ibeta,:,:] @
                        (dEInverse2*proj_velocity_alpha_K[...,ialpha,:,:]),
                        0,-2,-1
                    ).imag

        if _simultaneousVelocityProjection:
            self.BerryCurv=self.projected_BerryCurv[0]
        else:
            self.BerryCurv=self.projected_BerryCurv[0,0]
        if self.verbose==2: showElapsedTime(extraInfo=f"Berry curvature finished")

class OrbitalModern_IC(BerryCurv):
    def __init__(self, orbitalProjectionList=[], **kwarg) -> None:
        '''
        Compute the itinerant-circulation (IC) contribution to the
        equilibrium orbital magnetization, in mu_B.

        Unlike the LC magnetic-moment operator, this is a band-diagonal
        ground-state contribution. A projected IC term applies the same
        projection to both velocity vertices of the Berry curvature.

        Indexing:
        orbital_IC_K[ik1, ik2, ik3, idirection, iband].
        '''
        super().__init__(
            velocityProjectionList=orbitalProjectionList,
            _simultaneousVelocityProjection=True,
            orbitalProjectionList=orbitalProjectionList,
            **kwarg,
        )

        hbar2_over_2me = hbar**2 / (
            2 * electron_mass * elementary_charge * angstrom**2
        )

        def berryCurvatureToItinerantMoment(berryCurvature):
            orbital_tensor = (
                (self.energies[...,None,None,:] - self.model.fermi)
                * berryCurvature / hbar2_over_2me
            )
            return np.stack((
                orbital_tensor[...,1,2,:],
                orbital_tensor[...,2,0,:],
                orbital_tensor[...,0,1,:],
            ),axis=-2)

        self.orbital_IC_K = berryCurvatureToItinerantMoment(self.BerryCurv)
        self.projected_orbital_IC_K_list = [
            berryCurvatureToItinerantMoment(projected_BerryCurv)
            for projected_BerryCurv in self.projected_BerryCurv[1:]
        ]
        if self.verbose==2:
            showElapsedTime(extraInfo="modern IC orbital magnetization finished")

class OrbitalModern_Full(OrbitalModern_IC, OrbitalModern_LC):
    def __init__(self, orbitalProjectionList=[], **kwarg) -> None:
        '''
        Compute LC and IC orbital magnetism using one eigensystem and one
        velocity calculation.

        ``orbital_K`` is the LC magnetic-moment operator. The band-diagonal
        equilibrium integrand including IC is ``orbital_full_diag_K``.
        '''
        super().__init__(orbitalProjectionList=orbitalProjectionList, **kwarg)
        orbital_LC_diag_K = np.diagonal(
            self.orbital_K,axis1=-2,axis2=-1
        )
        self.orbital_full_diag_K = orbital_LC_diag_K+self.orbital_IC_K
        self.projected_orbital_full_diag_K_list = [
            np.diagonal(projected_orbital_K,axis1=-2,axis2=-1)
            + projected_orbital_IC_K
            for projected_orbital_K, projected_orbital_IC_K in zip(
                self.projected_orbital_K_list,
                self.projected_orbital_IC_K_list,
            )
        ]

class SpinOrbital(Spin, OrbitalModern_Full):
    '''
    Compute spin, LC orbital, and IC orbital magnetism while sharing the
    eigensystem and velocity calculation.
    '''
    def __init__(self, magneticProjectionList=[], gS = 2, **kwarg):
        super().__init__(spinProjectionList=magneticProjectionList, orbitalProjections=magneticProjectionList, gS=gS, **kwarg)

class AnormalousHallConductance(BerryCurv):

    def __init__(self, currentProjectionList=[], finiteTemperatureOccupations=False, **kwarg) -> None:
        super().__init__(velocityProjectionList=currentProjectionList, **kwarg)
        self.Iodd = False  # inversion gives minus
        self.Todd = True   # timereversal gives minus

        nProjections=self.projected_BerryCurv.shape[:2]
        projected_sigma=np.zeros(nProjections+(len(self.fermiList),3,3))

        occupations = self.fd_occupations if finiteTemperatureOccupations else None
        for ifermi, fermi in enumerate(self.fermiList):
            if occupations is None:
                f = np.int8(self.energies < fermi)
            else:
                f = occupations[ifermi]
            for ivelocity_alpha in range(nProjections[0]):
                for ivelocity_beta in range(nProjections[1]):
                    for ialpha in range(3):
                        for ibeta in range(3):
                            projected_sigma[ivelocity_alpha,ivelocity_beta,ifermi,ialpha,ibeta]=np.sum(self.projected_BerryCurv[ivelocity_alpha,ivelocity_beta,...,ialpha,ibeta,:]*f)
        
        #self.projected_sigma=projected_sigma/np.prod(self.kgrid)*(elementary_charge**2/hbar/angstrom/2)/self.model.Vcell
        self.projected_sigma=projected_sigma*(2*np.pi/(np.linalg.det(self.model.lattice[:2,:2])*np.prod(self.kgrid[:2]))) # for checking the TKNN SHC of 2D topo./trivial sys.
        if self.verbose==2: showElapsedTime(extraInfo=f"Anormalous Hall conductance finished")

class VelocityVelocityCorrelation_diag(Velocity_diag):
    def __init__(self, velocityProjectionList=[], **kwarg) -> None:
        '''
        Calc. intra-band contributions on the correlation func of velocity velocity \Omega^{alpha,beta}_{k,m}.

        alpha and beta can be x, y, z, denoting the (cart.) space direction of applied currents and the propagation of Hall currents
        k denotes the crystal moment, m indexes the bands.

        Indexing
        --------
        BerryCurv_diag[ik1, ik2, ik3, ialpha, ibeta, ieig]
        '''
        super().__init__(velocityProjectionList=velocityProjectionList,**kwarg)

        self.projected_BerryCurv_diag=np.zeros((len(velocityProjectionList)+1,len(velocityProjectionList)+1)+self.kgrid+(3,3,self.ibmax+1-self.ibmin))
        for ivelocity_alpha, proj_velocity_diag_K_alpha in enumerate([self.velocity_diag_K]+self.projected_velocity_diag_K_list):
            for ivelocity_beta, proj_velocity_diag_K_beta in enumerate([self.velocity_diag_K]+self.projected_velocity_diag_K_list):
                for ialpha in range(3):
                    for ibeta in range(3):
                        self.projected_BerryCurv_diag[ivelocity_alpha,ivelocity_beta,...,ialpha,ibeta,:]=-(proj_velocity_diag_K_beta[...,ibeta,:]*proj_velocity_diag_K_alpha[...,ialpha,:]).real/self.eta
        
        self.BerryCurv_diag=self.projected_BerryCurv_diag[0,0]
        if self.verbose==2: showElapsedTime(extraInfo=f"evaluations on intra band contributions on correlation functions <velocity><velocity> finished")

class AnormalousHallConductance_diag(VelocityVelocityCorrelation_diag):
    def __init__(self, currentProjectionList=[], **kwarg) -> None:
        '''
        Intra band contributions on the spin Hall conductance in (S/m)
        '''
        super().__init__(velocityProjectionList=currentProjectionList, **kwarg)

        nProjections=self.projected_BerryCurv_diag.shape[:2]
        projected_sigma=np.zeros(nProjections+(len(self.fermiList),3,3))
        
        for ifermi, weight in enumerate(self.fermiWeight):
            for ivelocity_alpha in range(nProjections[0]):
                for ivelocity_beta in range(nProjections[1]):
                    for ialpha in range(3):
                        for ibeta in range(3):
                            projected_sigma[ivelocity_alpha,ivelocity_beta,ifermi,ialpha,ibeta]=np.sum(self.projected_BerryCurv_diag[ivelocity_alpha,ivelocity_beta,...,ialpha,ibeta,:]*weight)
        
        #self.projected_sigma=projected_sigma/np.prod(self.kgrid)*(elementary_charge**2/hbar/angstrom/2)/self.model.Vcell
        self.projected_sigma=projected_sigma*(2*np.pi/(np.linalg.det(self.model.lattice[:2,:2])*np.prod(self.kgrid[:2]))) # for checking the TKNN SHC of 2D topo./trivial sys.
        if self.verbose==2: showElapsedTime(extraInfo=f"Intra-band spin Galvanic conductance finished")

class AnormalousHallConductance_LG_DM(Velocity):

    def __init__(self, currentProjectionList=[], match_resonator=False, **kwarg) -> None:
        '''
        Calc. AHE from the dc length-gauge Liouville density matrix.

        The default denominator uses the physical interband dipole convention,
        1 / (DeltaE * (DeltaE - i eta)). With match_resonator=True it uses
        1 / (DeltaE - i eta)^2, matching BerryCurv exactly at finite eta.
        '''
        super().__init__(velocityProjectionList=currentProjectionList, **kwarg)
        self.Iodd = False  # inversion gives minus
        self.Todd = True   # timereversal gives minus

        nProjections = (len(currentProjectionList)+1, len(currentProjectionList)+1)
        projected_sigma = np.zeros(nProjections + (len(self.fermiList), 3, 3))

        deltaE_mn = self.energies[..., :, None] - self.energies[..., None, :]
        nondegenerate = abs(deltaE_mn) > self.degen

        response_factor_base = np.zeros_like(deltaE_mn, dtype=complex)
        if match_resonator:
            response_factor_base[nondegenerate] = 1.0 / (deltaE_mn[nondegenerate] - 1j * self.eta)**2
        else:
            response_factor_base[nondegenerate] = 1.0 / deltaE_mn[nondegenerate] / (deltaE_mn[nondegenerate] - 1j * self.eta)

        for ifermi, fermi in enumerate(self.fermiList):
            f = np.int8(self.energies < fermi)
            f_nm_source = f[..., None, :] - f[..., :, None]
            response_factor = 1j * f_nm_source * response_factor_base

            for ivelocity_alpha, proj_velocity_alpha_K in enumerate([self.velocity_K] + self.projected_velocity_K_list):
                for ivelocity_beta, proj_velocity_beta_K in enumerate([self.velocity_K] + self.projected_velocity_K_list):
                    for ialpha in range(3):
                        rho1_alpha = response_factor * proj_velocity_alpha_K[..., ialpha, :, :]
                        for ibeta in range(3):
                            current_k = np.sum(
                                proj_velocity_beta_K[..., ibeta, :, :].swapaxes(-1, -2) * rho1_alpha,
                                axis=(-2, -1)
                            ).real
                            projected_sigma[ivelocity_alpha, ivelocity_beta, ifermi, ialpha, ibeta] = np.sum(current_k)

        self.projected_sigma = projected_sigma * (2*np.pi/(np.linalg.det(self.model.lattice[:2,:2])*np.prod(self.kgrid[:2])))
        self.sigma = self.projected_sigma[0, 0]
        if self.verbose==2: showElapsedTime(extraInfo=f"Length-gauge density-matrix AHE finished")

class BerryConnectionPolarizability(Velocity):
    def __init__(self, velocityProjectionList=[], **kwarg) -> None:
        '''
        Compute the Berry-connection polarizability (BCP) tensor
        c.f. eq. S2, PHYSICAL REVIEW LETTERS 130, 126303 (2023)

                            v^a_nm v^b_mn
        G^n_ab = 2 Re sum ------------------- .
                            (E_n-E_m)^3
                      m != n

        ``velocity_K`` is dH/dk in eV*angstrom, so ``G_K`` is in
        angstrom^2/eV. Degenerate terms selected by ``degen`` are omitted.

        Indexing:
        G_K[ik1, ik2, ik3, ia, ib, n]

        projected_G_K[iproj_a, iproj_b, ik1, ik2, ik3, ia, ib, n]
        applies the two projections independently to the two velocity
        vertices.
        '''
        super().__init__(velocityProjectionList=velocityProjectionList, **kwarg,)

        self.calcDeltaEInverse()
        deltaEInverse3 = self.deltaEInverse_pure**3

        velocity_K_list = [self.velocity_K]+self.projected_velocity_K_list
        bcp_shape = self.kgrid+(3,3,self.energies.shape[-1])
        self.projected_G_K = np.zeros((len(velocity_K_list),len(velocity_K_list))+bcp_shape,dtype=float,)

        for iproj_a, velocity_a_K in enumerate(velocity_K_list):
            for iproj_b, velocity_b_K in enumerate(velocity_K_list):
                self.projected_G_K[iproj_a, iproj_b] = 2*np.einsum(
                    "...anm,...bmn,...nm->...abn",
                    velocity_a_K,
                    velocity_b_K,
                    deltaEInverse3,
                    optimize="greedy",
                ).real

        self.G_K = self.projected_G_K[0,0]
        if self.verbose==2: showElapsedTime(extraInfo="Berry-connection polarizability finished")

class BerryConnectionPolarizabilityDipole(BerryConnectionPolarizability):
    Iodd = True
    Todd = False

    def __init__(self, **kwarg) -> None:
        '''
        Compute the analytical k-space dipole of the BCP,

        G_Dipole_K[..., a, b, c, n] = partial_{k_a} G^n_{bc}.

        The derivative is evaluated locally at each k point using velocity
        matrices, their gauge-covariant derivatives, and band-energy
        derivatives. Its unit is angstrom^3/eV. The isolated-band expression
        omits intermediate states separated by less than ``degen``.
        '''
        super().__init__(highOrderVelocity=True, **kwarg)
        nbands = self.energies.shape[-1]

        eigenstateDeltaEInverse   = self.deltaEInverse_pure
        eigenstateDeltaEInverse_T = self.deltaEInverse_pure.swapaxes(-1, -2)
        deltaEInverse3 = self.deltaEInverse_pure**3
        deltaEInverse4 = self.deltaEInverse_pure**4
        bandVelocity = np.diagonal(self.velocity_K, axis1=-2, axis2=-1)

        self.G_Dipole_K = np.zeros(
            self.kgrid + (3, 3, 3, nbands),
            dtype=float,
        )
        for a in range(3):
            velocity_a = self.velocity_K[..., a, :, :]
            braDerivative = velocity_a * eigenstateDeltaEInverse
            ketDerivative = velocity_a * eigenstateDeltaEInverse_T
            # D_a V_b includes changes of both eigenstates. Keeping only one
            # derivative direction in memory matters for large Wannier models.
            velocityDerivative_K = np.zeros(self.kgrid + (3, nbands, nbands), dtype=complex)
            for b in range(3):
                velocity_b = self.velocity_K[..., b, :, :]
                velocityDerivative_K[..., b, :, :] = (
                    self.w_K[..., a, b, :, :]
                    + braDerivative @ velocity_b
                    + velocity_b @ ketDerivative
                )

            numeratorDerivative = np.einsum(
                "...bnm,...cmn,...nm->...bcn",
                velocityDerivative_K,
                self.velocity_K,
                deltaEInverse3,
                optimize="greedy",
            ) + np.einsum(
                "...bnm,...cmn,...nm->...bcn",
                self.velocity_K,
                velocityDerivative_K,
                deltaEInverse3,
                optimize="greedy",
            )
            
            deltaVelocity =  bandVelocity[..., a, :, None] - bandVelocity[..., a, None, :]
            
            denominatorDerivative = -3*np.einsum(
                "...bnm,...cmn,...nm->...bcn",
                self.velocity_K,
                self.velocity_K,
                deltaVelocity*deltaEInverse4,
                optimize="greedy",
            )
            self.G_Dipole_K[..., a, :, :, :] =  2*(numeratorDerivative + denominatorDerivative).real

        if self.verbose==2:
            showElapsedTime(extraInfo="Berry-connection polarizability dipole finished")

class SecondOrderAHC_BCP_FermiSea(BerryConnectionPolarizabilityDipole):
    Iodd = True
    Todd = True

    def __init__(self, **kwarg) -> None:
        '''
        Second-order nonlinear DC conductivity in the Fermi-sea form,
        c.f. eq. S6, PHYSICAL REVIEW LETTERS 130, 126303 (2023)

        chi_abc = sum_n integral[dk] f_n
                  (partial_b G^n_ac - partial_a G^n_bc).

        Indexing
        --------
        - sigma[ifermi, a, b, c]
            3rd-ranked conductivity. a: orientation of current; b, c: orientations of applied efield
        - response[ik1, ik2, ik3, ifermi, a, b, c]
            response function on BZ
        '''
        super().__init__(**kwarg)

        self.response=np.zeros(self.kgrid+(len(self.fermiList),3,3,3),dtype=float)
        for ifermi, occupation in enumerate(self.fd_occupations):
            for a in range(3):
                for b in range(3):
                    for c in range(3):
                        self.response[...,ifermi,a,b,c]=np.sum(
                            (self.G_Dipole_K[...,b,a,c,:]-self.G_Dipole_K[...,a,b,c,:])*occupation,axis=-1)
        
        self.sigma=np.sum(self.response,axis=(0,1,2,))/np.prod(self.kgrid)
        if self.verbose==2: showElapsedTime(extraInfo="2nd-order anomalous Hall conductance (Fermi sea) finished")

class SecondOrderAHC_BCP_FermiSurf(BerryConnectionPolarizability, FermiScan):
    Iodd = True
    Todd = True

    def __init__(self, **kwarg) -> None:
        '''
        Second-order nonlinear DC conductivity in the Fermi-surface form,
        obtained from Eq. S6 of PHYSICAL REVIEW LETTERS 130, 126303 (2023)
        by integration by parts:

        chi_abc = sum_n integral[dk] f'_n
                  (v^n_a G^n_bc - v^n_b G^n_ac).

        ``fermiWeight`` is f'_n and is evaluated only for the band window
        selected by ``FermiScan``.

        Indexing
        --------
        - sigma[ifermi, a, b, c]
            3rd-rank conductivity. a: current direction; b, c: electric-field
            directions.
        - response[ik1, ik2, ik3, ifermi, a, b, c]
            Fermi-surface response on the BZ after summing over bands.
        '''
        super().__init__(**kwarg)

        bandWindow = slice(self.ibmin, self.ibmax+1)
        bandVelocity = np.diagonal(self.velocity_K, axis1=-2, axis2=-1).real[..., bandWindow]

        responseFermiFirst = np.einsum(
            "f...n,...an,...bcn->f...abc",
            self.fermiWeight,
            bandVelocity,
            self.G_K[..., bandWindow],
            optimize="greedy",
        )
        responseFermiFirst = responseFermiFirst - responseFermiFirst.swapaxes(-3, -2)
        self.response = np.moveaxis(responseFermiFirst, 0, 3)
        self.sigma = np.sum(self.response, axis=(0, 1, 2)) / np.prod(self.kgrid)

        if self.verbose==2: showElapsedTime(extraInfo="2nd-order anomalous Hall conductance (Fermi surface) finished")

class BerryConnectionPolarizabilityMagSusceptibility(BerryConnectionPolarizability,Spin,OrbitalModern_LC,):
    Iodd = False
    Todd = True

    def __init__(self, mag_type="total", gS=2.0, **kwarg) -> None:
        '''
        Compute the magnetic susceptibility of the BCP from Eq. S7,

        Lambda_K[..., a, b, d, n] = partial_{B_d} G^n_{ab}.

        ``mag_type`` selects ``"spin"``, modern local-circulation
        ``"orbital"``, or their ``"total"``. Magnetic moments are represented
        in mu_B, so Lambda_K is in mu_B*angstrom^2/eV^2. Degenerate
        intermediate states selected by ``degen`` are omitted.
        '''
        validMagTypes = ("total", "spin", "orbital")
        if mag_type not in validMagTypes:
            raise ValueError(
                f"mag_type must be one of {validMagTypes}, got {mag_type!r}"
            )

        super().__init__(gS=gS,_computeSpin=mag_type in ("total", "spin"),_computeOrbitalModernLC=mag_type in ("total", "orbital"),**kwarg,)
        nbands = self.energies.shape[-1]

        self.magneticMoment_K = np.zeros(self.kgrid + (3, nbands, nbands), dtype=complex)
        if mag_type in ("total", "spin"):
            self.magneticMoment_K += self.spin_K
        if mag_type in ("total", "orbital"):
            self.magneticMoment_K += self.orbital_K
        self.mag_type = mag_type

        nondegenerate = np.abs(self.deltaE) > self.degen
        eigenstateDeltaEInverse   = self.deltaEInverse_pure
        eigenstateDeltaEInverse_T = self.deltaEInverse_pure.swapaxes(-1, -2)
        deltaEInverse3 = self.deltaEInverse**3
        deltaEInverse4 = self.deltaEInverse**4
        momentDiagonal = np.diagonal(self.magneticMoment_K, axis1=-2, axis2=-1)

        self.Lambda_K = np.zeros(self.kgrid + (3, 3, 3, nbands), dtype=float)
        for d in range(3):
            # H_B=-M_d B_d. These two terms are the changes of the bra and
            # ket eigenstates in each velocity matrix element.
            magneticPerturbation = -self.magneticMoment_K[..., d, :, :]
            braDerivative = magneticPerturbation * eigenstateDeltaEInverse
            ketDerivative = magneticPerturbation * eigenstateDeltaEInverse_T
            velocityMagDerivative_K = np.zeros(self.kgrid + (3, nbands, nbands),dtype=complex,)
            for a in range(3):
                velocity_a = self.velocity_K[..., a, :, :]
                velocityMagDerivative_K[..., a, :, :] =  braDerivative @ velocity_a + velocity_a @ ketDerivative

            numeratorDerivative = np.einsum(
                "...anm,...bmn,...nm->...abn",
                velocityMagDerivative_K, self.velocity_K, deltaEInverse3,
                optimize="greedy",
            ) + np.einsum(
                "...anm,...bmn,...nm->...abn",
                self.velocity_K, velocityMagDerivative_K, deltaEInverse3,
                optimize="greedy",
            )
            deltaMoment =  momentDiagonal[..., d, :, None] - momentDiagonal[..., d, None, :]
            denominatorDerivative = 3*np.einsum(
                "...anm,...bmn,...nm->...abn",
                self.velocity_K, self.velocity_K, deltaMoment*deltaEInverse4,
                optimize="greedy",
            )
            self.Lambda_K[..., :, :, d, :] = 2*(numeratorDerivative + denominatorDerivative).real

        if self.verbose==2: showElapsedTime(extraInfo="Berry-connection polarizability magnetic susceptibility finished")

class IntrinsicNPHE_BCP(BerryConnectionPolarizabilityDipole,BerryConnectionPolarizabilityMagSusceptibility,FermiScan):
    Iodd = True
    Todd = False

    def __init__(self, mag_type="total", gS=2.0, **kwarg) -> None:
        '''
        Compute the intrinsic nonlinear planar Hall response from Eq. S6.

        Indexing:
        ---------
        - BCP_susc_K[..., a, b, c, d, n]
            Band-resolved geometric bracket in Eq. S6, before f'_0.
        - BCP_susc_integrand_K[ifermi, ..., a, b, c, d]
            Full k-resolved integrand after the Fermi-surface band sum.
        - Upsilon[ifermi, a, b, c, d]
            Uniform-k average of the integrand. a: current direction, b and c: electric field direction, d: mag. field direction

        Magnetic moments are in mu_B, so Upsilon is in
        mu_B*angstrom^3/eV^2 before the later SI conversion.
        '''
        super().__init__(mag_type=mag_type, gS=gS, **kwarg)

        bandVelocity = np.diagonal(self.velocity_K,axis1=-2,axis2=-1).real
        momentDiagonal = np.diagonal(self.magneticMoment_K,axis1=-2,axis2=-1,).real

        self.BCP_susc_K = np.einsum(
            "...an,...bcdn->...abcdn",
            bandVelocity,self.Lambda_K,
            optimize="greedy",
        )
        self.BCP_susc_K -= np.einsum(
            "...bn,...acdn->...abcdn",
            bandVelocity,self.Lambda_K,
            optimize="greedy",
        )
        self.BCP_susc_K += np.einsum(
            "...abcn,...dn->...abcdn",
            self.G_Dipole_K,momentDiagonal,
            optimize="greedy",
        )
        self.BCP_susc_K -= np.einsum(
            "...abcn,...dn->...abcdn",
            self.G_Dipole_K.swapaxes(-4, -3),momentDiagonal,
            optimize="greedy",
        )

        bandWindow = slice(self.ibmin, self.ibmax+1)
        BCP_susc_integrand_K = np.einsum(
            "f...n,...abcdn->f...abcd",
            self.fermiWeight,self.BCP_susc_K[..., bandWindow],
            optimize="greedy",
        )
        if self.model.ispin == 1:
            BCP_susc_integrand_K *= 2

        self.response = np.moveaxis(BCP_susc_integrand_K, 0, 3)
        self.Upsilon = np.sum(BCP_susc_integrand_K,axis=(1, 2, 3)) / np.prod(self.kgrid)
        self.projected_Upsilon=self.Upsilon[None,None,...]

        if self.verbose==2: showElapsedTime(extraInfo="intrinsic NPHE from BCP finished")

class BulkPhotovoltaicEffect(Velocity):
    def __init__(self, velocityProjectionList=[], responseOmegaIndexList=[], **kwarg) -> None:
        '''
        Only the I3 contribution, the Tr {va vb vc} term
        The projections are implemented.

        Parameters
        ----------
        - responseOmegaIndexList : list
            the BZ distribution of response is only recorded for the frequencies pointed here to save memory
        '''
        super().__init__(velocityProjectionList=velocityProjectionList, **kwarg)
        self.Iodd = True  # inversion gives minus
        self.Todd = True  # timereversal gives minus

        # energy difference, Ekm - Ekn, index: [ik1, ik2, ik3, ibm, ibn]
        dEmn = self.energies[...,:,None] - self.energies[...,None,:]
        # energy difference with three band axes, and fermi energy axis, 3-spatial axes 
        # index: [ik1, ik2, ik3, 1(ifermi), 1(ialpha1), 1(ialpha2), 1(ibeta), 1(ibl), ibm, ibn]
        dEnm    = -dEmn[..., None, None,None,None, None,:,:] - 1j* self.eta # (En-Em-i\eta)
        dEnl    = -dEmn[..., None, None,None,None, :,None,:] - 1j* self.eta # (En-El-i\eta)

        # occupation number [ik1, ik2, ik3, ifermi, ib]
        f = np.int8(self.energies[..., None, :] < self.fermiList[None, None, None, :, None])
        # occupation number difference fl - fn, index: [ik1, ik2, ik3, ifermi, 1(ialpha1), 1(ialpha2), 1(ibeta), ibl, 1(ibm), ibn]
        fnl = f[..., None,None,None, None,None,:] - f[..., None,None,None, :,None,None]
        
        self.projected_sigma = np.zeros((len(velocityProjectionList)+1, len(velocityProjectionList)+1, len(velocityProjectionList)+1, # iproj1, iproj2, iproj3
                                         len(self.fermiList), len(self.omegaList),   # ifermi, iomega 
                                         3, 3, 3), dtype=float)                      # ialpha1, ialpha2, ibeta
        self.projected_response = np.zeros((len(velocityProjectionList)+1, len(velocityProjectionList)+1, len(velocityProjectionList)+1, # iproj1, iproj2, iproj3
                                         self.kgrid[0], self.kgrid[1], self.kgrid[2],
                                         len(self.fermiList), len(responseOmegaIndexList),   # ifermi, iomega 
                                         3, 3, 3), dtype=float)                      # ialpha1, ialpha2, ibeta
        irecord=0
        for iomega, Omega in enumerate(self.omegaList):
            # compute the energy terms
            den    = 1/dEnm * (1/(dEnl - Omega) + 1/(dEnl + Omega))
            # compute the velocity products
            for ivelocity_alpha1, proj_velocity_alpha1_K in enumerate([self.velocity_K] + self.projected_velocity_K_list):
                # index: [ik1, ik2, ik3, 1(ifermi), ialpha1, 1(ialpha2), 1(ialpha3), 1(iorbA), ibn, 1(ibm), ibl]
                v_alpha1 = proj_velocity_alpha1_K.swapaxes(-1, -2)[..., None, :, None, None, :, None, :]
                for ivelocity_alpha2, proj_velocity_alpha2_K in enumerate([self.velocity_K] + self.projected_velocity_K_list):
                    # index: [ik1, ik2, ik3, 1(ifermi), 1(ialpha1), ialpha2, 1(ialpha3), ibl, ibm, 1(ibn)]
                    v_alpha2 = proj_velocity_alpha2_K[..., None, None, :, None, :, :, None]
                    for ivelocity_beta, proj_velocity_beta_K in enumerate([self.velocity_K] + self.projected_velocity_K_list):
                        # index: [ik1, ik2, ik3, 1(ifermi), 1(ialpha1), 1(ialpha2), ialpha3, 1(ibl), ibm, ibn]
                        v_beta   = proj_velocity_beta_K[..., None, None, None, :, None, :, :]
                        # Sum over the 3 band indices to compute trace
                        bubble_sum = np.sum(
                            v_alpha1 * v_alpha2 * v_beta * fnl * den, 
                            axis=(0, 1, 2, -3, -2, -1)
                        ).real
                        
                        # E(t) = -dA(t)/dt = -A0 (i Omega) exp(i Omega t). So, E(Omega) = -i Omega A(Omega), E(Omega)E(-Omega) = Omega^2 A(Omega) A(-Omega)
                        self.projected_sigma[ivelocity_alpha1, ivelocity_alpha2, ivelocity_beta, :, iomega, :, :, :] = -2 * bubble_sum / (-Omega**2)

                        if iomega not in responseOmegaIndexList: continue
                        bubble_sum_k = np.sum(
                            v_alpha1 * v_alpha2 * v_beta * fnl * den, 
                            axis=(-3, -2, -1)
                        ).real                        
                        self.projected_response[ivelocity_alpha1, ivelocity_alpha2, ivelocity_beta, :, :, :, :, irecord, :, :, :] = -2 * bubble_sum_k / (-Omega**2)
                        irecord+=1
            
        self.projected_sigma /= np.prod(self.kgrid)
        # unit conversion
        factor_SI = 4*elementary_charge**2/hbar/self.model.Vcell 
        if self.model.ispin==1: factor_SI*=2
        self.projected_sigma *= factor_SI # in unit A / V^2

        # to responsibity (uA/W)
        #factor_responsivity = (753.46 / self.model.refractive_index) * 1e6 # 1e6 for micro-Amps
        #self.projected_responsibility = self.projected_sigma * factor_responsivity
        if self.verbose == 2: showElapsedTime(extraInfo=f"Bulk photovoltaic conductance finished")

class BulkPhotovoltaicEffect_VG(Velocity):
    def __init__(self, velocityProjectionList=[], responseOmegaIndexList=[], **kwarg) -> None:
        '''
        Calculates the COMPLETE Velocity Gauge BPVE by summing all four Feynman diagrams, using resonator denominator approach
        '''
        super().__init__(velocityProjectionList=velocityProjectionList, highOrderVelocity=True, **kwarg)
        
        self.projected_response = np.zeros((len(velocityProjectionList)+1, len(velocityProjectionList)+1, len(velocityProjectionList)+1, # iproj1, iproj2, iproj3
                                         self.kgrid[0], self.kgrid[1], self.kgrid[2],
                                         len(self.fermiList), len(responseOmegaIndexList),   # ifermi, iomega 
                                         3, 3, 3), dtype=complex)                      # ialpha1, ialpha2, ibeta
        self.sigma_VG_Full = np.zeros((len(self.fermiList), len(self.omegaList), 3, 3, 3), dtype=complex)
        
        # 3D energy array, index (ik1, ik2, ik3, m, n, l)
        E_m = self.energies[..., :, None, None]
        E_n = self.energies[..., None, :, None]
        E_l = self.energies[..., None, None, :]
        D_mn = 1.0 / (E_m - E_n - 1j * self.eta)

        # 2D energy arrays, index [ik1, ik2, ik3, m, n]
        E_m_2 = self.energies[..., :, None]
        E_n_2 = self.energies[..., None, :]
        D_mn_2 = 1.0 / (E_m_2 - E_n_2 - 1j * self.eta)

        for ifermi, fermi in enumerate(self.fermiList):
            # occupation number: index [ik1, ik2, ik3, ib]
            f = np.int8(self.energies < fermi)
            # 3D occupation arrays
            f_m =             f[..., :, None, None]
            f_n =             f[..., None, :, None]
            f_l =             f[..., None, None, :]
            f_nl= f_n - f_l
            f_lm= f_l - f_m

            # 2D occupation arrays
            f_m_2 =             f[..., :, None]
            f_n_2 =             f[..., None, :]
            f_mn_2= f_m_2 - f_n_2    

            for iomega, Omega in enumerate(self.omegaList):
                # Resonant Denominators
                S_ln_pos = 1.0 / (E_l - E_n + Omega - 1j * self.eta)
                S_ln_neg = 1.0 / (E_l - E_n - Omega - 1j * self.eta)
                S_ml_pos = 1.0 / (E_m - E_l + Omega - 1j * self.eta)
                S_ml_neg = 1.0 / (E_m - E_l - Omega - 1j * self.eta)

                term_ba = D_mn * (f_nl * S_ln_pos - f_lm * S_ml_neg)
                term_ab = D_mn * (f_nl * S_ln_neg - f_lm * S_ml_pos)
            
                S_mn_pos = 1.0 / (E_m_2 - E_n_2 - Omega - 1j * self.eta)
                S_mn_neg = 1.0 / (E_m_2 - E_n_2 + Omega - 1j * self.eta)
                G_mn_2= f_mn_2 * D_mn_2
            
                for a in range(3):
                    for b in range(3):
                        for c in range(3):
                        
                            # (ik1, ik2, ik3, norbs, norbs)
                            v_a = self.velocity_K[..., a, :, :]
                            v_b = self.velocity_K[..., b, :, :]
                            v_c = self.velocity_K[..., c, :, :]
                            v_c_nm = v_c.swapaxes(-1, -2)
                        
                            w_ab = 0.5 * (self.w_K[..., a, b, :, :] + self.w_K[..., b, a, :, :])
                            w_ca = self.w_K[..., c, a, :, :]
                            w_cb = self.w_K[..., c, b, :, :]
                            w_ca_nm = w_ca.swapaxes(-1, -2)
                            w_cb_nm = w_cb.swapaxes(-1, -2)
                        
                            u_cab = 0.5 * (self.u_K[..., c, a, b, :, :] + self.u_K[..., c, b, a, :, :])
                            
                            # =======================================================
                            # 1. Three-Point Bubble: Tr[ v^c v^a v^b ]
                            # =======================================================
                            vvv_ba = np.einsum('abcmn, abcml, abcln -> abcmnl', v_c_nm, v_b, v_a)
                            vvv_ab = np.einsum('abcmn, abcml, abcln -> abcmnl', v_c_nm, v_a, v_b)
                            I3 = np.sum(vvv_ba * term_ba + vvv_ab * term_ab, axis=(-3, -2, -1))
                            
                            # =======================================================
                            # 2. Two-Point Bubble A (Diamagnetic Perturbation): Tr[ v^c w^{ab} ]
                            # =======================================================
                            rho_w = w_ab * G_mn_2
                            I2A = np.einsum('abcmn, abcmn -> abc', v_c_nm, rho_w)
                            
                            # =======================================================
                            # 3. Two-Point Bubble B (Diamagnetic Current): Tr[ w^{ca} v^b ]
                            # =======================================================
                            rho_v_a = v_a * f_mn_2 * S_mn_neg
                            rho_v_b = v_b * f_mn_2 * S_mn_pos
                            I2B = np.einsum('abcmn, abcmn -> abc', w_cb_nm, rho_v_a) + \
                                  np.einsum('abcmn, abcmn -> abc', w_ca_nm, rho_v_b)
                            
                            # =======================================================
                            # 4. One-Point Bubble (Ground State): Tr[ u^{cab} ]
                            # =======================================================
                            u_nn = np.einsum('abcnn -> abcn', u_cab)
                            I1 = np.einsum('abcn, abcn -> abc', u_nn, f)
                            
                            # =======================================================
                            # Total Summation
                            # =======================================================
                            sigma_total_k = (I3 + I2A + I2B + I1) / Omega**2
                            # only the paramagentic contribution
                            #sigma_total_k = I1 / Omega**2

                            # BZ integration
                            self.sigma_VG_Full[ifermi, iomega, a, b, c] = np.sum(sigma_total_k) / np.prod(self.kgrid)
                            
                            # saving BZ response 
                            for irecord, iomega_record in enumerate(responseOmegaIndexList):
                                if iomega_record == iomega: break
                            else:
                                continue
                            self.projected_response[0, 0, 0, :, :, :, ifermi, irecord, a, b, c] = sigma_total_k
                            irecord+=1

        self.projected_sigma = self.sigma_VG_Full[None,None,None,...]
        # unit conversion
        #factor_SI = 4*elementary_charge**2/hbar/self.model.Vcell 
        #if self.model.ispin==1: factor_SI*=2
        #self.projected_sigma *= factor_SI # in unit A / V^2

        # to responsibity (uA/W)
        #factor_responsivity = (753.46 / self.model.refractive_index) * 1e6 # 1e6 for micro-Amps
        #self.projected_responsibility = self.projected_sigma * factor_responsivity
        if self.verbose == 2: showElapsedTime(extraInfo=f"Full Velocity Gauge BPVE finished")

class BulkPhotovoltaicEffect_VG_DM(Dipole):
    def __init__(self, velocityProjectionList=[], responseOmegaIndexList=[], responseFermiIndexList=None,
                 finiteTemperatureOccupations=False, fullCovariantDerivative=False, **kwarg) -> None:
        '''
        Calculates the Velocity Gauge BPVE using the frequency-domain
        Liouville density-matrix approach. ``fullCovariantDerivative=True``
        adds the gauge-covariant relaxation target and retains II, EI, IE and
        EE channels. The default reproduces the historical interband result.
        '''
        currentProjectionList = kwarg.pop("currentProjectionList", None)
        if currentProjectionList is not None and len(velocityProjectionList) == 0:
            velocityProjectionList = currentProjectionList
        super().__init__(velocityProjectionList=velocityProjectionList, highOrderVelocity=True,
                         highOrderDipole=fullCovariantDerivative, **kwarg)
        self.fullCovariantDerivative = bool(fullCovariantDerivative)

        self.channel_names = DM_CHANNEL_NAMES
        self.channel_descriptions = dict(DM_CHANNEL_DESCRIPTIONS)
        self.channel_descriptions["II"] += ", including current contact vertices"
        response_shape = (len(self.fermiList), len(self.omegaList), 3, 3, 3)
        self.sigma_VG_DM_channels = {
            name: np.zeros(response_shape, dtype=complex) for name in self.channel_names
        }

        responseOmegaIndexList = _normalize_response_index_list(responseOmegaIndexList, len(self.omegaList))
        if responseFermiIndexList is None and len(responseOmegaIndexList) > 0:
            responseFermiIndexList = range(len(self.fermiList))
        responseFermiIndexList = _normalize_response_index_list(responseFermiIndexList, len(self.fermiList))
        responseOmegaRecord = {iomega: irecord for irecord, iomega in enumerate(responseOmegaIndexList)}
        responseFermiRecord = {ifermi: irecord for irecord, ifermi in enumerate(responseFermiIndexList)}
        nproj = len(velocityProjectionList) + 1
        projected_shape = ((nproj, nproj, nproj) + self.kgrid
                           + (len(responseFermiIndexList), len(responseOmegaIndexList), 3, 3, 3))
        self.projected_response_channels = {
            name: np.zeros(projected_shape, dtype=complex) for name in self.channel_names
        }

        kernel = VelocityGaugeSecondOrderDMKernel(
            self, self.fullCovariantDerivative, finiteTemperatureOccupations
        )
        self.density_matrix_kernel = kernel
        for ifermi, fermi in enumerate(self.fermiList):
            fermi_state = kernel.fermi_state(ifermi, fermi)
            relaxation_response_k = {
                name: np.zeros(self.kgrid + (3, 3, 3), dtype=complex) for name in self.channel_names
            }
            if self.fullCovariantDerivative:
                for a in range(3):
                    for b in range(3):
                        rho2_relaxation = fermi_state.relaxation(a, b)
                        for name in self.channel_names:
                            for c in range(3):
                                relaxation_response_k[name][..., a, b, c] = np.sum(
                                    rho2_relaxation[name]
                                    * self.velocity_K[..., c, :, :].swapaxes(-1, -2), axis=(-2, -1)
                                )
                        del rho2_relaxation

            for iomega, Omega in enumerate(self.omegaList):
                frequency_state = fermi_state.at_frequency(Omega)
                irecord_fermi = responseFermiRecord.get(ifermi)
                irecord_omega = responseOmegaRecord.get(iomega)
                for a in range(3):
                    for b in range(3):
                        block = frequency_state.block(a, b)
                        for c in range(3):
                            v_c_nm = self.velocity_K[..., c, :, :].swapaxes(-1, -2)
                            w_ca_nm = self.w_K[..., c, a, :, :].swapaxes(-1, -2)
                            w_cb_nm = self.w_K[..., c, b, :, :].swapaxes(-1, -2)
                            u_cab = 0.5 * (self.u_K[..., c, a, b, :, :]
                                           + self.u_K[..., c, b, a, :, :])
                            response_k = {
                                name: np.sum(rho2 * v_c_nm, axis=(-2, -1))
                                + relaxation_response_k[name][..., a, b, c]
                                for name, rho2 in block.rho2.items()
                            }
                            response_k["EE"] -= np.sum(
                                block.rho1_minus["E"] * w_cb_nm
                                + block.rho1_plus["E"] * w_ca_nm, axis=(-2, -1)
                            )
                            response_k["II"] += np.sum(
                                block.rho0 * u_cab.swapaxes(-1, -2), axis=(-2, -1)
                            )
                            if self.fullCovariantDerivative:
                                response_k["II"] -= np.sum(
                                    block.rho1_minus["I"] * w_cb_nm
                                    + block.rho1_plus["I"] * w_ca_nm, axis=(-2, -1)
                                )
                            for name in self.channel_names:
                                sigma_k = response_k[name] / Omega**2
                                self.sigma_VG_DM_channels[name][ifermi, iomega, a, b, c] = (
                                    np.sum(sigma_k) / np.prod(self.kgrid)
                                )
                                if irecord_fermi is not None and irecord_omega is not None:
                                    self.projected_response_channels[name][
                                        0, 0, 0, :, :, :, irecord_fermi, irecord_omega, a, b, c
                                    ] = sigma_k

        self.sigma_VG_DM = sum(self.sigma_VG_DM_channels.values())
        self.sigma_channels = self.sigma_VG_DM_channels
        self.projected_response = sum(self.projected_response_channels.values())
        self.projected_sigma = self.sigma_VG_DM[None, None, None, ...]
        self.projected_sigma_channels = {
            name: sigma[None, None, None, ...] for name, sigma in self.sigma_VG_DM_channels.items()
        }
        for name in self.channel_names:
            setattr(self, f"sigma_VG_DM_{name}", self.sigma_VG_DM_channels[name])
            setattr(self, f"projected_response_{name}", self.projected_response_channels[name])
            setattr(self, f"projected_sigma_{name}", self.projected_sigma_channels[name])
        if self.verbose == 2: showElapsedTime(extraInfo=f"Velocity Gauge BPVE density matrix finished")

class BulkPhotovoltaicEffect_LG_Shift(Dipole):
    def __init__(self, velocityProjectionList=[], **kwarg) -> None:
        '''
        Calculates the Universal Shift Current (BPVE) using the Sipe-Shkrebtii Length Gauge.
        Natively evaluates BOTH the Linear Shift Current and Circular Shift Current.
        '''
        super().__init__(velocityProjectionList=velocityProjectionList, highOrderDipole=True, **kwarg)
        omega_mn = self.deltaE
        
        # Tensor Shape: [ifermi, iomega, ialpha(drive), ibeta(drive), igamma(current)]
        self.sigma_shift = np.zeros((len(self.fermiList), len(self.omegaList), 3, 3, 3), dtype=complex)

        for iomega, Omega in enumerate(self.omegaList):
            L_mn_pos = (1/np.pi) * self.eta / ((omega_mn - Omega)**2 + self.eta**2)
            L_mn_neg = (1/np.pi) * self.eta / ((omega_mn + Omega)**2 + self.eta**2)
            L_mn = L_mn_pos + L_mn_neg

            for ialpha in range(3): # Drive E-field a
                r_a = self.r_K[..., ialpha, :, :]
                for ibeta in range(3): # Drive E-field b
                    r_b = self.r_K[..., ibeta, :, :]
                    for igamma in range(3): # Measured Current c
                        r_a_deriv_c = self.r_deriv_K[..., igamma, ialpha, :, :]
                        r_b_deriv_c = self.r_deriv_K[..., igamma, ibeta, :, :]

                        # Shift current is symmetric in the two optical fields.
                        I_mn = -0.5j * (r_b.conj() * r_a_deriv_c + r_a.conj() * r_b_deriv_c)
                        
                        for ifermi, fermi in enumerate(self.fermiList):
                            f = np.int8(self.energies < fermi)
                            f_nm = f[..., :, None] - f[..., None, :] 
                            
                            shift_integrand = f_nm * I_mn * L_mn
                            shift_sum = np.sum(shift_integrand) * np.pi
                            
                            self.sigma_shift[ifermi, iomega, ialpha, ibeta, igamma] = shift_sum / np.prod(self.kgrid)

        self.projected_sigma = self.sigma_shift[None,None,None,...]
        
        # Unit conversion (Ensure it supports complex broadcasting)
        #factor_SI = 4 * elementary_charge**2 / hbar / self.model.Vcell 
        #if self.model.ispin == 1: factor_SI *= 2
        #self.projected_sigma *= factor_SI # in unit A / V^2

        if self.verbose == 2: showElapsedTime(extraInfo=f"Length Gauge Shift Current finished")

class BulkPhotovoltaicEffect_LG_Injection(Dipole):
    def __init__(self, velocityProjectionList=[], **kwarg) -> None:
        '''
        Calculates the Circular Injection Current (Circular BPVE) using the 
        Sipe-Shkrebtii Length Gauge formalism.
        '''
        # No high order velocity needed! This makes LG much faster than VG.
        super().__init__(velocityProjectionList=velocityProjectionList, highOrderVelocity=False, **kwarg)
        
        omega_mn = self.deltaE
        
        # Tensor Shape: [ifermi, iomega, ialpha(drive), ibeta(drive), igamma(current)]
        # Must be COMPLEX because the Circular BPVE susceptibility is purely imaginary!
        self.sigma_inj = np.zeros((len(self.fermiList), len(self.omegaList), 3, 3, 3), dtype=complex)

        # 1. Compute Group Velocity Differences (Delta^c_nm)
        v_diag = np.diagonal(self.velocity_K, axis1=-2, axis2=-1) 
        Delta_v = v_diag[..., :, :, None] - v_diag[..., :, None, :] 

        # 3. Relaxation Time (tau = 1 / 2\eta in natural units)
        # Injection current scales linearly with the scattering time!
        tau = 1.0 / (2.0 * self.eta)

        for iomega, Omega in enumerate(self.omegaList):
            # Energy conservation delta functions (Lorentzians)
            L_mn_pos = (1/np.pi) * self.eta / ((omega_mn - Omega)**2 + self.eta**2)
            # Use the positive-frequency transition once, with the explicit
            # factor two from the two optical time orderings. Adding the
            # reverse transition here would cancel the ordered circular phase.
            L_mn = 2.0 * L_mn_pos

            for ialpha in range(3): # Drive E-field a
                r_a = self.r_K[..., ialpha, :, :]
                
                for ibeta in range(3): # Drive E-field b
                    r_b = self.r_K[..., ibeta, :, :]
                    
                    # Full Complex Optical Transition Tensor
                    # Real part drives Linear Injection (if magnetic)
                    # Imaginary part drives Circular Injection
                    rho_ab = r_a.conj() * r_b
                    
                    for igamma in range(3): # Measured Current c
                        Delta_c = Delta_v[..., igamma, :, :]
                        
                        # Universal Sipe Injection Integrand
                        I_mn = tau * Delta_c * rho_ab
                        
                        for ifermi, fermi in enumerate(self.fermiList):
                            f = np.int8(self.energies < fermi)
                            f_mn = f[..., :, None] - f[..., None, :] 
                            
                            inj_integrand = f_mn * I_mn * L_mn
                            inj_sum = np.sum(inj_integrand) * (-2 * np.pi)
                            
                            self.sigma_inj[ifermi, iomega, ialpha, ibeta, igamma] = inj_sum / np.prod(self.kgrid)

        self.projected_sigma = self.sigma_inj[None,None,None,...]
        
        # Unit conversion
        #factor_SI = 4 * elementary_charge**2 / hbar / self.model.Vcell 
        #if self.model.ispin == 1: factor_SI *= 2
        #self.projected_sigma *= factor_SI # in unit A / V^2

        if self.verbose == 2: showElapsedTime(extraInfo=f"Length Gauge Injection Current finished")

class BulkPhotovoltaicEffect_LG_DM(Dipole):
    def __init__(self, velocityProjectionList=[], responseOmegaIndexList=[], responseFermiIndexList=None,
                 finiteTemperatureOccupations=False, fullCovariantDerivative=False, **kwarg) -> None:
        '''
        Calculates the length-gauge BPVE using the frequency-domain Liouville
        density-matrix equation with an analytic covariant derivative.

        This is the finite-eta density-matrix counterpart of the resonant
        shift/injection formulas. It keeps the full complex denominators, so it
        is not expected to be identical to the Lorentzian-only LG formulas away
        from the clean resonant limit. ``fullCovariantDerivative=True`` uses
        the complete common LG kernel; the default preserves the historical
        analytic interband derivative for backward compatibility.
        '''
        currentProjectionList = kwarg.pop("currentProjectionList", None)
        if currentProjectionList is not None and len(velocityProjectionList) == 0:
            velocityProjectionList = currentProjectionList
        super().__init__(velocityProjectionList=velocityProjectionList, highOrderDipole=True, **kwarg)
        self.fullCovariantDerivative = bool(fullCovariantDerivative)

        self.channel_names = DM_CHANNEL_NAMES
        self.channel_descriptions = dict(DM_CHANNEL_DESCRIPTIONS)
        response_shape = (len(self.fermiList), len(self.omegaList), 3, 3, 3)
        self.sigma_LG_DM_channels = {
            name: np.zeros(response_shape, dtype=complex) for name in self.channel_names
        }

        responseOmegaIndexList = _normalize_response_index_list(responseOmegaIndexList, len(self.omegaList))
        if responseFermiIndexList is None and len(responseOmegaIndexList) > 0:
            responseFermiIndexList = range(len(self.fermiList))
        responseFermiIndexList = _normalize_response_index_list(responseFermiIndexList, len(self.fermiList))
        responseOmegaRecord = {iomega: irecord for irecord, iomega in enumerate(responseOmegaIndexList)}
        responseFermiRecord = {ifermi: irecord for irecord, ifermi in enumerate(responseFermiIndexList)}
        nproj = len(velocityProjectionList) + 1
        projected_shape = ((nproj, nproj, nproj) + self.kgrid
                           + (len(responseFermiIndexList), len(responseOmegaIndexList), 3, 3, 3))
        self.projected_response_channels = {
            name: np.zeros(projected_shape, dtype=complex) for name in self.channel_names
        }

        kernel = LengthGaugeSecondOrderDMKernel(
            self, self.fullCovariantDerivative, finiteTemperatureOccupations
        )
        self.density_matrix_kernel = kernel
        for ifermi, fermi in enumerate(self.fermiList):
            fermi_state = kernel.fermi_state(ifermi, fermi)
            for iomega, Omega in enumerate(self.omegaList):
                frequency_state = fermi_state.at_frequency(Omega)
                irecord_fermi = responseFermiRecord.get(ifermi)
                irecord_omega = responseOmegaRecord.get(iomega)
                for a in range(3):
                    for b in range(3):
                        if self.fullCovariantDerivative:
                            block = frequency_state.block(a, b)
                        else:
                            block = frequency_state.legacy_bpve_block(a, b)
                        for c in range(3):
                            v_c_nm = self.velocity_K[..., c, :, :].swapaxes(-1, -2)
                            for name in self.channel_names:
                                sigma_k = np.sum(block.rho2[name] * v_c_nm, axis=(-2, -1))
                                self.sigma_LG_DM_channels[name][ifermi, iomega, a, b, c] = (
                                    np.sum(sigma_k) / np.prod(self.kgrid)
                                )
                                if irecord_fermi is not None and irecord_omega is not None:
                                    self.projected_response_channels[name][
                                        0, 0, 0, :, :, :, irecord_fermi, irecord_omega, a, b, c
                                    ] = sigma_k

        self.sigma_LG_DM = sum(self.sigma_LG_DM_channels.values())
        self.sigma_channels = self.sigma_LG_DM_channels
        self.projected_response = sum(self.projected_response_channels.values())
        self.projected_sigma = self.sigma_LG_DM[None, None, None, ...]
        self.projected_sigma_channels = {
            name: sigma[None, None, None, ...] for name, sigma in self.sigma_LG_DM_channels.items()
        }
        for name in self.channel_names:
            setattr(self, f"sigma_LG_DM_{name}", self.sigma_LG_DM_channels[name])
            setattr(self, f"projected_response_{name}", self.projected_response_channels[name])
            setattr(self, f"projected_sigma_{name}", self.projected_sigma_channels[name])
        if self.verbose == 2: showElapsedTime(extraInfo=f"Length Gauge BPVE density matrix finished")

class BulkPhotovoltaicEffect_TDSE(Velocity):
    def __init__(self, velocityProjectionList=[], E0=0.001, n_steps_per_cycle=100,
                 highOrderVelocity=True, include_offdiag=True, **kwarg) -> None:
        '''
        Calculates the 2nd order DC photovoltaic conductance using Real-Time TDSE.
        Evolves the density matrix using a 4th-Order Runge-Kutta (RK4) integrator.
        
        Outputs
        -------
        self.sigma_TDSE[ifermi, iomega, ialpha_drive_1, ialpha_drive_2, ibeta_measure]
        '''
        super().__init__(velocityProjectionList=velocityProjectionList, highOrderVelocity=highOrderVelocity, **kwarg)
        
        self.sigma_TDSE = np.zeros((len(self.fermiList), len(self.omegaList), 3, 3, 3), dtype=float)
        self.sigma_TDSE_symmetric = np.zeros_like(self.sigma_TDSE, dtype=float)
        self.sigma_TDSE_quadrature = np.zeros_like(self.sigma_TDSE, dtype=float)
        
        # 1. Create the unperturbed Hamiltonian diagonal matrix H0
        H_diag = np.zeros(self.energies.shape + (self.model.norbs,), dtype=complex)
        for i in range(self.model.norbs):
            H_diag[..., i, i] = self.energies[..., i]

        for ifermi, fermi in enumerate(self.fermiList):
            # 2. Initial equilibrium density matrix (Fermi-Dirac)
            f = np.int8(self.energies < fermi)
            rho0 = np.zeros_like(H_diag)
            for i in range(self.model.norbs):
                rho0[..., i, i] = f[..., i]

            J0 = np.zeros(3)
            for ibeta in range(3):
                j0_val = np.sum(rho0 * self.velocity_K[..., ibeta, :, :].swapaxes(-1, -2), axis=(-1, -2)).real
                J0[ibeta] = np.sum(j0_val)

            for iomega, Omega in enumerate(self.omegaList):
                T = 2 * np.pi / Omega
                dt = T / n_steps_per_cycle
                
                # Compute cycles needed for the transient to decay (e.g., 10 lifetimes)
                t_transient = 10.0 / self.eta
                n_cycles_transient = int(np.ceil(t_transient / T))
                n_cycles_measure = 1
                
                if self.verbose:
                    print(f"TDSE [Fermi={fermi:.2f}, Omega={Omega:.2f}]: Simulating {n_cycles_transient} transient cycles + {n_cycles_measure} measurement cycle.")

                def run_drive(directions, phases):
                    rho = rho0.copy()
                    t = 0.0
                    
                    # Phase 1: Evolve through transients to reach Non-Equilibrium Steady State
                    for _ in range(n_cycles_transient * n_steps_per_cycle):
                        rho = self._rk4_step(rho, t, dt, H_diag, directions, phases, Omega, rho0, E0)
                        t += dt
                        
                    # Phase 2: Time-average the current over the final optical cycle
                    J_dc = np.zeros(3)
                    for _ in range(n_cycles_measure * n_steps_per_cycle):
                        rho = self._rk4_step(rho, t, dt, H_diag, directions, phases, Omega, rho0, E0)
                        t += dt

                        A_t = [-2 * E0 / Omega * np.sin(Omega * t + phase) for phase in phases]
                        
                        # Measure current in all 3 directions
                        for ibeta in range(3):
                            J_op = self.velocity_K[..., ibeta, :, :]
                            for i, A_i in enumerate(A_t):
                                J_op = J_op - A_i * self.w_K[..., ibeta, directions[i], :, :]
                            for i, A_i in enumerate(A_t):
                                for j, A_j in enumerate(A_t):
                                    J_op = J_op + 0.5 * A_i * A_j * self.u_K[..., ibeta, directions[i], directions[j], :, :]

                            # Tr(rho * v) summed over all k-points and bands
                            j_val = np.sum(rho * J_op.swapaxes(-1, -2), axis=(-1,-2)).real
                            J_dc[ibeta] += np.sum(j_val) / (n_cycles_measure * n_steps_per_cycle)

                    return (J_dc - J0) / E0**2 / np.prod(self.kgrid)

                diagonal_response = np.zeros((3, 3))
                for ialpha in range(3): # Direction of the driving field
                    response = run_drive((ialpha,), (0.0,))
                    # Conductance = J / E^2
                    # Divide by volume/N_k
                    self.sigma_TDSE[ifermi, iomega, ialpha, ialpha, :] = response
                    self.sigma_TDSE_symmetric[ifermi, iomega, ialpha, ialpha, :] = response
                    diagonal_response[ialpha] = response

                if include_offdiag:
                    for ialpha in range(3):
                        for ibeta in range(ialpha + 1, 3):
                            same_phase = run_drive((ialpha, ibeta), (0.0, 0.0))
                            cross_symmetric = same_phase - diagonal_response[ialpha] - diagonal_response[ibeta]

                            quadrature_phase = run_drive((ialpha, ibeta), (0.0, -np.pi / 2))
                            cross_quadrature = quadrature_phase - diagonal_response[ialpha] - diagonal_response[ibeta]

                            self.sigma_TDSE_symmetric[ifermi, iomega, ialpha, ibeta, :] = cross_symmetric
                            self.sigma_TDSE_symmetric[ifermi, iomega, ibeta, ialpha, :] = cross_symmetric
                            self.sigma_TDSE_quadrature[ifermi, iomega, ialpha, ibeta, :] = cross_quadrature
                            self.sigma_TDSE_quadrature[ifermi, iomega, ibeta, ialpha, :] = -cross_quadrature

                            # Same-phase driving gives Re[chi_ab + chi_ba]. The
                            # quadrature drive gives the phase-sensitive cross term.
                            self.sigma_TDSE[ifermi, iomega, ialpha, ibeta, :] = 0.5 * (cross_symmetric - cross_quadrature)
                            self.sigma_TDSE[ifermi, iomega, ibeta, ialpha, :] = 0.5 * (cross_symmetric + cross_quadrature)
        self.projected_sigma=self.sigma_TDSE[None,None,None,...]
        # unit conversion
        #factor_SI = 4*elementary_charge**2/hbar/self.model.Vcell 
        #if self.model.ispin==1: factor_SI*=2
        #self.projected_sigma *= factor_SI # in unit A / V^2

        # to responsibity (uA/W)
        #factor_responsivity = (753.46 / self.model.refractive_index) * 1e6 # 1e6 for micro-Amps
        #self.projected_responsibility = self.projected_sigma * factor_responsivity
        if self.verbose == 2: showElapsedTime(extraInfo=f"TDSE BPVE finished")

    def _rk4_step(self, rho, t, dt, H_diag, directions, phases, Omega, rho0, E0):
        # A(t) = A0 exp(-iwt)
        # E(t) = -dA/dt = (iw) A0 exp(-iw)
        # E(w) = iw A(w), A(w) = (-i/w) E(w)
        # Assuming real electric field: E(W) = E(-W) = E0, E(t) = E(W) exp(-iWt) + E(-W) exp(iWt) = 2 E0 cos(Wt)
        # A(t) = A(W) exp(-iWt) + A(-W) exp(iWt) = (-i/W) E(W) exp(-iWt) + (i/W) E(-W) exp(iWt) = -2 E0/W sin(Wt)
        def dRho_dt(rho_state, time_val):
            A_t = [-2 * E0 / Omega * np.sin(Omega * time_val + phase) for phase in phases]
            # Full Minimal Coupling: H = H0 - A*v + 1/2 A^2 w - 1/6 A^3 u
            H_t = H_diag
            for i, A_i in enumerate(A_t):
                H_t = H_t - A_i * self.velocity_K[..., directions[i], :, :]
            for i, A_i in enumerate(A_t):
                for j, A_j in enumerate(A_t):
                    H_t = H_t + 0.5 * A_i * A_j * self.w_K[..., directions[i], directions[j], :, :]
            comm = H_t @ rho_state - rho_state @ H_t
            return -1j * comm - self.eta * (rho_state - rho0)

        k1 = dRho_dt(rho, t)
        k2 = dRho_dt(rho + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = dRho_dt(rho + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = dRho_dt(rho + dt * k3, t + dt)
        return rho + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
class BulkPhotovoltaicEffect_TDSE_LG_Naive(Dipole):
    def __init__(self, velocityProjectionList=[], E0=0.001, n_steps_per_cycle=100, **kwarg) -> None:
        '''
        Calculates the LG DC response using isolated k-point TDSE.
        WARNING: Because it lacks the k-derivative (Nabla_k rho), this will ONLY capture 
        the Injection-like optical transitions. It will FAIL to reproduce the Shift current.
        '''
        super().__init__(velocityProjectionList=velocityProjectionList, highOrderVelocity=False, **kwarg)
        
        self.sigma_TDSE = np.zeros((len(self.fermiList), len(self.omegaList), 3, 3, 3), dtype=float)
        
        # 1. Diagonal Hamiltonian
        H_diag = np.zeros(self.energies.shape + (self.model.norbs,), dtype=complex)
        for i in range(self.model.norbs):
            H_diag[..., i, i] = self.energies[..., i]

        for ifermi, fermi in enumerate(self.fermiList):
            f = np.int8(self.energies < fermi)
            rho0 = np.zeros_like(H_diag)
            for i in range(self.model.norbs):
                rho0[..., i, i] = f[..., i]

            for iomega, Omega in enumerate(self.omegaList):
                T = 2 * np.pi / Omega
                dt = T / n_steps_per_cycle
                
                n_cycles_transient = int(np.ceil((10.0 / self.eta) / T))
                n_cycles_measure = 1

                for ialpha in range(3): # Direction of the driving E-field
                    r_alpha = self.r_K[..., ialpha, :, :]
                    rho = rho0.copy()
                    t = 0.0
                    
                    for _ in range(n_cycles_transient * n_steps_per_cycle):
                        rho = self._rk4_step_LG(rho, t, dt, H_diag, r_alpha, Omega, rho0, E0)
                        t += dt
                        
                    J_dc = np.zeros(3)
                    for _ in range(n_cycles_measure * n_steps_per_cycle):
                        rho = self._rk4_step_LG(rho, t, dt, H_diag, r_alpha, Omega, rho0, E0)
                        t += dt

                        for ibeta in range(3):
                            v_beta = self.velocity_K[..., ibeta, :, :]
                            # Current Operator in LG is strictly just 'v'
                            j_val = np.sum(rho * v_beta.swapaxes(-1, -2), axis=(-1,-2)).real
                            J_dc[ibeta] += np.sum(j_val) / (n_cycles_measure * n_steps_per_cycle)
                            
                    self.sigma_TDSE[ifermi, iomega, ialpha, ialpha, :] = J_dc / E0**2 / np.prod(self.kgrid)
        
        self.projected_sigma = self.sigma_TDSE[None,None,None,...]
        if self.verbose == 2: showElapsedTime(extraInfo=f"TDSE Naive LG finished")

    def _rk4_step_LG(self, rho, t, dt, H_diag, r_alpha, Omega, rho0, E0):
        def dRho_dt(rho_state, time_val):
            # A(t) = -2E0/Omega * sin(Omega t) -> E(t) = -dA/dt = 2*E0*cos(Omega t)
            E_t = 2.0 * E0 * np.cos(Omega * time_val)
            H_t = H_diag - E_t * r_alpha
            
            comm = H_t @ rho_state - rho_state @ H_t
            return -1j * comm - self.eta * (rho_state - rho0)

        k1 = dRho_dt(rho, t)
        k2 = dRho_dt(rho + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = dRho_dt(rho + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = dRho_dt(rho + dt * k3, t + dt)
        return rho + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def _normalize_response_index_list(index_list, axis_length):
    if index_list is None:
        return []
    return [int(index) for index in index_list if 0 <= int(index) < axis_length]

class InverseFaradayEffect_LG(Dipole, Spin, Orbital):
    def __init__(self, ife_type="total", velocityProjectionList=[], responseOmegaIndexList=[],
                 responseFermiIndexList=None, **kwarg) -> None:
        '''
        Computing IFE with length gauge, using resonator denominator approach.
        unit: \muB / (V/\angst.)^2

        Indexing
        --------
        - chi_IFE[ifermi, iomega, ialpha_drive1, ialpha_drive2, ibeta_measure] : np.ndarray(dtype=complex)
        '''
        super().__init__(velocityProjectionList=velocityProjectionList, highOrderVelocity=False, **kwarg)
        
        nbasis=self.energies.shape[-1]
        M_K = np.zeros(self.kgrid + (3, nbasis, nbasis), dtype=complex)
        if ife_type in ["total", "spin"]: M_K += self.spin_K
        if ife_type in ["total", "orbital"]: M_K += self.orbital_K

        self.chi_IFE = np.zeros((len(self.fermiList), len(self.omegaList), 3, 3, 3), dtype=complex)

        responseOmegaIndexList = _normalize_response_index_list(responseOmegaIndexList, len(self.omegaList))
        if responseFermiIndexList is None and len(responseOmegaIndexList) > 0:
            responseFermiIndexList = range(len(self.fermiList))
        responseFermiIndexList = _normalize_response_index_list(responseFermiIndexList, len(self.fermiList))
        responseOmegaRecord = {iomega: irecord for irecord, iomega in enumerate(responseOmegaIndexList)}
        responseFermiRecord = {ifermi: irecord for irecord, ifermi in enumerate(responseFermiIndexList)}
        self.projected_response = np.zeros((1, 1, 1) + self.kgrid +
                                           (len(responseFermiIndexList), len(responseOmegaIndexList), 3, 3, 3),
                                           dtype=complex)

        # Band-axis convention below is (m, n, l), matching
        # Tr[M rho^(2)] = sum_mn M_mn rho^(2)_nm.
        E_m_3D = self.energies[..., :, None, None]
        E_n_3D = self.energies[..., None, :, None]
        E_l_3D = self.energies[..., None, None, :]
        D_nm_3D = 1.0 / (E_n_3D - E_m_3D - 1j * self.eta)

        for ifermi, fermi in enumerate(self.fermiList):
            f = np.int8(self.energies < fermi)
            f_m_3D = f[..., :, None, None]
            f_n_3D = f[..., None, :, None]
            f_l_3D = f[..., None, None, :]
            
            f_lm = f_l_3D - f_m_3D
            f_nl = f_n_3D - f_l_3D

            for iomega, Omega in enumerate(self.omegaList):
                S_lm_pos = 1.0 / (E_l_3D - E_m_3D + Omega - 1j * self.eta)
                S_nl_pos = 1.0 / (E_n_3D - E_l_3D + Omega - 1j * self.eta)
                S_lm_neg = 1.0 / (E_l_3D - E_m_3D - Omega - 1j * self.eta)
                S_nl_neg = 1.0 / (E_n_3D - E_l_3D - Omega - 1j * self.eta)
                
                term_ba = D_nm_3D * (-f_lm * S_lm_pos + f_nl * S_nl_neg)
                term_ab = D_nm_3D * ( f_nl * S_nl_pos - f_lm * S_lm_neg)
                irecord_omega = responseOmegaRecord.get(iomega)

                for c in range(3):
                    M_c = M_K[..., c, :, :]
                    for a in range(3):
                        r_a = self.r_K[..., a, :, :]
                        for b in range(3):
                            r_b = self.r_K[..., b, :, :]
                            
                            Mrr_ba = np.einsum('...mn, ...nl, ...lm -> ...mnl', M_c, r_b, r_a)
                            Mrr_ab = np.einsum('...mn, ...nl, ...lm -> ...mnl', M_c, r_a, r_b)
                            
                            integrand = Mrr_ba * term_ba + Mrr_ab * term_ab
                            chi_k = np.sum(integrand, axis=(-3, -2, -1))
                            self.chi_IFE[ifermi, iomega, a, b, c] = np.sum(chi_k) / np.prod(self.kgrid)
                            irecord_fermi = responseFermiRecord.get(ifermi)
                            if irecord_fermi is not None and irecord_omega is not None:
                                self.projected_response[0, 0, 0, :, :, :, irecord_fermi, irecord_omega, a, b, c] = chi_k

        self.projected_sigma = self.chi_IFE[None,None,None,...]
        if self.verbose == 2: showElapsedTime(extraInfo=f"Length Gauge IFE finished")

class InverseFaradayEffect_LG_DM(Dipole, Spin, Orbital):
    def __init__(self, ife_type="total", velocityProjectionList=[], responseOmegaIndexList=[],
                 responseFermiIndexList=None, fullCovariantDerivative=False, **kwarg) -> None:
        '''
        Compute IFE in the length gauge with a frequency-domain density matrix.
        unit: \muB / (V/\angst.)^2

        ``fullCovariantDerivative=True`` replaces every dipole commutator by
        ``i D_k rho`` and preserves the II, EI, IE and EE channels separately.
        The first letter labels the second-order density-matrix sector and the
        second letter labels the first-order sector (I: intraband, E: interband).

        Indexing
        --------
        - chi_IFE[ifermi, iomega, ialpha_drive1, ialpha_drive2, ibeta_measure]
        '''
        super().__init__(velocityProjectionList=velocityProjectionList,
                         highOrderDipole=fullCovariantDerivative, **kwarg)
        self.fullCovariantDerivative = bool(fullCovariantDerivative)
        if self.fullCovariantDerivative and self.kBT <= 0:
            raise ValueError("fullCovariantDerivative requires kBT > 0")
        if self.fullCovariantDerivative and abs(self.eta) < 1e-15:
            raise ValueError("fullCovariantDerivative requires finite eta for the dc density matrix")

        nbasis = self.energies.shape[-1]
        M_K = np.zeros(self.kgrid + (3, nbasis, nbasis), dtype=complex)
        if ife_type in ["total", "spin"]: M_K += self.spin_K
        if ife_type in ["total", "orbital"]: M_K += self.orbital_K

        response_shape = (len(self.fermiList), len(self.omegaList), 3, 3, 3)
        self.channel_names = DM_CHANNEL_NAMES
        self.channel_descriptions = dict(DM_CHANNEL_DESCRIPTIONS)
        self.chi_IFE_channels = {name: np.zeros(response_shape, dtype=complex) for name in self.channel_names}

        responseOmegaIndexList = _normalize_response_index_list(responseOmegaIndexList, len(self.omegaList))
        if responseFermiIndexList is None and len(responseOmegaIndexList) > 0:
            responseFermiIndexList = range(len(self.fermiList))
        responseFermiIndexList = _normalize_response_index_list(responseFermiIndexList, len(self.fermiList))
        responseOmegaRecord = {iomega: irecord for irecord, iomega in enumerate(responseOmegaIndexList)}
        responseFermiRecord = {ifermi: irecord for irecord, ifermi in enumerate(responseFermiIndexList)}
        projected_shape = ((1, 1, 1) + self.kgrid +
                           (len(responseFermiIndexList), len(responseOmegaIndexList), 3, 3, 3))
        self.projected_response_channels = {
            name: np.zeros(projected_shape, dtype=complex) for name in self.channel_names
        }

        kernel = LengthGaugeSecondOrderDMKernel(self, self.fullCovariantDerivative)
        self.density_matrix_kernel = kernel
        for ifermi, fermi in enumerate(self.fermiList):
            fermi_state = kernel.fermi_state(ifermi, fermi)
            for iomega, Omega in enumerate(self.omegaList):
                irecord_fermi = responseFermiRecord.get(ifermi)
                irecord_omega = responseOmegaRecord.get(iomega)
                frequency_state = fermi_state.at_frequency(Omega)
                for a in range(3):
                    for b in range(3):
                        block = frequency_state.block(a, b)
                        for c in range(3):
                            M_c_nm = M_K[..., c, :, :].swapaxes(-1, -2)
                            for name in self.channel_names:
                                chi_k = np.sum(block.rho2[name] * M_c_nm, axis=(-2, -1))
                                self.chi_IFE_channels[name][ifermi, iomega, a, b, c] = (
                                    np.sum(chi_k) / np.prod(self.kgrid)
                                )
                                if irecord_fermi is not None and irecord_omega is not None:
                                    self.projected_response_channels[name][
                                        0, 0, 0, :, :, :, irecord_fermi, irecord_omega, a, b, c
                                    ] = chi_k

        self.chi_IFE = sum(self.chi_IFE_channels.values())
        self.projected_response = sum(self.projected_response_channels.values())
        self.projected_sigma = self.chi_IFE[None, None, None, ...]
        self.projected_sigma_channels = {
            name: response[None, None, None, ...] for name, response in self.chi_IFE_channels.items()
        }
        for name in self.channel_names:
            setattr(self, f"chi_IFE_{name}", self.chi_IFE_channels[name])
            setattr(self, f"projected_response_{name}", self.projected_response_channels[name])
            setattr(self, f"projected_sigma_{name}", self.projected_sigma_channels[name])
        if self.verbose == 2: showElapsedTime(extraInfo=f"Length Gauge IFE density matrix finished")

class InverseFaradayEffect_VG_DM(Dipole, Spin, Orbital):
    def __init__(self, ife_type="total", velocityProjectionList=[], responseOmegaIndexList=[],
                 responseFermiIndexList=None, fullCovariantDerivative=False, **kwarg) -> None:
        '''
        Compute IFE in the velocity gauge with a frequency-domain density matrix.
        unit: \muB / (V/\angst.)^2

        ``fullCovariantDerivative=True`` adds the gauge-covariant relaxation
        vertices ``eta C_a f`` and ``i eta {C_a,C_b}f/2``, where ``C_a=iD_a``.
        These vertices supply the intraband population response missing from
        the clean commutator ``[v_a,f]``. II, EI, IE and EE density-matrix
        channels are retained separately. The magnetic-moment operator is
        assumed to have no explicit vector-potential dependence.

        Indexing
        --------
        - chi_IFE[ifermi, iomega, ialpha_drive1, ialpha_drive2, ibeta_measure] : np.ndarray(dtype=complex)
        '''
        super().__init__(velocityProjectionList=velocityProjectionList, highOrderVelocity=True,
                         highOrderDipole=fullCovariantDerivative, **kwarg)
        self.fullCovariantDerivative = bool(fullCovariantDerivative)
        if self.fullCovariantDerivative and self.kBT <= 0:
            raise ValueError("fullCovariantDerivative requires kBT > 0")
        if self.fullCovariantDerivative and abs(self.eta) < 1e-15:
            raise ValueError("fullCovariantDerivative requires finite eta for gauge-covariant relaxation")

        nbasis = self.energies.shape[-1]
        M_K = np.zeros(self.kgrid + (3, nbasis, nbasis), dtype=complex)
        if ife_type in ["total", "spin"]: M_K += self.spin_K
        if ife_type in ["total", "orbital"]: M_K += self.orbital_K

        response_shape = (len(self.fermiList), len(self.omegaList), 3, 3, 3)
        self.channel_names = DM_CHANNEL_NAMES
        self.channel_descriptions = dict(DM_CHANNEL_DESCRIPTIONS)
        self.channel_descriptions["EE"] += " and the contact vertex"
        self.chi_IFE_channels = {name: np.zeros(response_shape, dtype=complex) for name in self.channel_names}

        responseOmegaIndexList = _normalize_response_index_list(responseOmegaIndexList, len(self.omegaList))
        if responseFermiIndexList is None and len(responseOmegaIndexList) > 0:
            responseFermiIndexList = range(len(self.fermiList))
        responseFermiIndexList = _normalize_response_index_list(responseFermiIndexList, len(self.fermiList))
        responseOmegaRecord = {iomega: irecord for irecord, iomega in enumerate(responseOmegaIndexList)}
        responseFermiRecord = {ifermi: irecord for irecord, ifermi in enumerate(responseFermiIndexList)}
        projected_shape = ((1, 1, 1) + self.kgrid +
                           (len(responseFermiIndexList), len(responseOmegaIndexList), 3, 3, 3))
        self.projected_response_channels = {
            name: np.zeros(projected_shape, dtype=complex) for name in self.channel_names
        }

        kernel = VelocityGaugeSecondOrderDMKernel(self, self.fullCovariantDerivative)
        self.density_matrix_kernel = kernel
        for ifermi, fermi in enumerate(self.fermiList):
            fermi_state = kernel.fermi_state(ifermi, fermi)

            relaxation_response_k = {
                name: np.zeros(self.kgrid + (3, 3, 3), dtype=complex) for name in self.channel_names
            }
            if self.fullCovariantDerivative:
                for a in range(3):
                    for b in range(3):
                        rho2_relaxation = fermi_state.relaxation(a, b)
                        for name in self.channel_names:
                            for c in range(3):
                                relaxation_response_k[name][..., a, b, c] = np.sum(
                                    rho2_relaxation[name] * M_K[..., c, :, :].swapaxes(-1, -2),
                                    axis=(-2, -1)
                                )
                        del rho2_relaxation

            for iomega, Omega in enumerate(self.omegaList):
                irecord_fermi = responseFermiRecord.get(ifermi)
                irecord_omega = responseOmegaRecord.get(iomega)
                frequency_state = fermi_state.at_frequency(Omega)
                for a in range(3):
                    for b in range(3):
                        block = frequency_state.block(a, b)
                        for c in range(3):
                            M_c_nm = M_K[..., c, :, :].swapaxes(-1, -2)
                            for name in self.channel_names:
                                sigma_k = (
                                    np.sum(block.rho2[name] * M_c_nm, axis=(-2, -1))
                                    + relaxation_response_k[name][..., a, b, c]
                                ) / Omega**2
                                self.chi_IFE_channels[name][ifermi, iomega, a, b, c] = (
                                    np.sum(sigma_k) / np.prod(self.kgrid)
                                )
                                if irecord_fermi is not None and irecord_omega is not None:
                                    self.projected_response_channels[name][
                                        0, 0, 0, :, :, :, irecord_fermi, irecord_omega, a, b, c
                                    ] = sigma_k

        self.chi_IFE = sum(self.chi_IFE_channels.values())
        self.projected_response = sum(self.projected_response_channels.values())
        self.projected_sigma = self.chi_IFE[None, None, None, ...]
        self.projected_sigma_channels = {
            name: response[None, None, None, ...] for name, response in self.chi_IFE_channels.items()
        }
        for name in self.channel_names:
            setattr(self, f"chi_IFE_{name}", self.chi_IFE_channels[name])
            setattr(self, f"projected_response_{name}", self.projected_response_channels[name])
            setattr(self, f"projected_sigma_{name}", self.projected_sigma_channels[name])
        if self.verbose == 2: showElapsedTime(extraInfo=f"Velocity Gauge IFE density matrix finished")

class InverseFaradayEffect_VG(Velocity, Spin, Orbital):
    def __init__(self, ife_type="total", velocityProjectionList=[], responseOmegaIndexList=[],
                 responseFermiIndexList=None, **kwarg) -> None:
        '''
        Computing IFE with velocity-gauge (vector potential/current coupling), using resonator denominator approach
        unit: \muB / (V/\angst.)^2

        Indexing
        --------
        - chi_IFE[ifermi, iomega, ialpha_drive1, ialpha_drive2, ibeta_measure] : np.ndarray(dtype=complex)
        '''
        super().__init__(velocityProjectionList=velocityProjectionList, highOrderVelocity=True, **kwarg)
        
        nbasis=self.energies.shape[-1]
        M_K = np.zeros(self.kgrid + (3, nbasis, nbasis), dtype=complex)
        if ife_type in ["total", "spin"]: M_K += self.spin_K
        if ife_type in ["total", "orbital"]: M_K += self.orbital_K

        self.chi_IFE = np.zeros((len(self.fermiList), len(self.omegaList), 3, 3, 3), dtype=complex)

        responseOmegaIndexList = _normalize_response_index_list(responseOmegaIndexList, len(self.omegaList))
        if responseFermiIndexList is None and len(responseOmegaIndexList) > 0:
            responseFermiIndexList = range(len(self.fermiList))
        responseFermiIndexList = _normalize_response_index_list(responseFermiIndexList, len(self.fermiList))
        responseOmegaRecord = {iomega: irecord for irecord, iomega in enumerate(responseOmegaIndexList)}
        responseFermiRecord = {ifermi: irecord for irecord, ifermi in enumerate(responseFermiIndexList)}
        self.projected_response = np.zeros((1, 1, 1) + self.kgrid +
                                           (len(responseFermiIndexList), len(responseOmegaIndexList), 3, 3, 3),
                                           dtype=complex)
        # Matrix axes are ordered as (m, n), same convention as omega_mn in LG.
        E_m = self.energies[..., :, None, None]
        E_n = self.energies[..., None, :, None]
        E_l = self.energies[..., None, None, :]
        D_mn = 1.0 / (E_m - E_n - 1j * self.eta)

        E_m_2D = self.energies[..., :, None]
        E_n_2D = self.energies[..., None, :]
        D_mn_2D = 1.0 / (E_m_2D - E_n_2D - 1j * self.eta)

        for ifermi, fermi in enumerate(self.fermiList):
            f = np.int8(self.energies < fermi)
            f_m = f[..., :, None, None]
            f_n = f[..., None, :, None]
            f_l = f[..., None, None, :]
            f_nl = f_n - f_l
            f_lm = f_l - f_m
            f_mn_2D = f[..., :, None] - f[..., None, :]

            for iomega, Omega in enumerate(self.omegaList):
                irecord_omega = responseOmegaRecord.get(iomega)
                S_ln_pos = 1.0 / (E_l - E_n + Omega - 1j * self.eta)
                S_ln_neg = 1.0 / (E_l - E_n - Omega - 1j * self.eta)
                S_ml_pos = 1.0 / (E_m - E_l + Omega - 1j * self.eta)
                S_ml_neg = 1.0 / (E_m - E_l - Omega - 1j * self.eta)

                term_ba = D_mn * (f_nl * S_ln_pos - f_lm * S_ml_neg)
                term_ab = D_mn * (f_nl * S_ln_neg - f_lm * S_ml_pos)

                
                for a in range(3):
                    v_a = self.velocity_K[..., a, :, :]
                    for b in range(3):
                        v_b = self.velocity_K[..., b, :, :]
                        w_ab = 0.5 * (self.w_K[..., a, b, :, :] + self.w_K[..., b, a, :, :])
                        for c in range(3):
                            M_c_mn = M_K[..., c, :, :]
                            M_c_nm = M_c_mn.swapaxes(-1, -2)

                            Mvv_ba = np.einsum('...mn, ...ml, ...ln -> ...mnl', M_c_nm, v_b, v_a)
                            Mvv_ab = np.einsum('...mn, ...ml, ...ln -> ...mnl', M_c_nm, v_a, v_b)
                            para_k = np.sum(Mvv_ba * term_ba + Mvv_ab * term_ab, axis=(-3, -2, -1))

                            dia_k = np.sum(M_c_nm * w_ab * f_mn_2D * D_mn_2D, axis=(-2, -1))
                            sigma_k = (para_k + dia_k) / Omega**2
                            self.chi_IFE[ifermi, iomega, a, b, c] = np.sum(sigma_k) / np.prod(self.kgrid)
                            irecord_fermi = responseFermiRecord.get(ifermi)
                            if irecord_fermi is not None and irecord_omega is not None:
                                self.projected_response[0, 0, 0, :, :, :, irecord_fermi, irecord_omega, a, b, c] = sigma_k

        self.projected_sigma = self.chi_IFE[None,None,None,...]
        if self.verbose == 2: showElapsedTime(extraInfo=f"Velocity Gauge IFE finished")

class InverseFaradayEffect_TDSE(Dipole, Spin, Orbital):
    def __init__(self, ife_type="total", velocityProjectionList=[], E0=0.001, n_steps_per_cycle=100,
                 include_offdiag=True, fullCovariantDerivative=False, **kwarg) -> None:
        '''
        Compute the velocity-gauge IFE by time-domain density-matrix propagation.
        unit: \muB / (V/\angst.)^2

        The propagation is resolved to first and second order in the vector
        potential. ``fullCovariantDerivative=True`` relaxes toward the
        field-shifted equilibrium density matrix through second order,

            rho_eq(A) = f + i A_a C_a f - A_a A_b C_a C_b f / 2,

        where C_a=iD_a. This is the time-domain counterpart of the covariant
        relaxation vertices in ``InverseFaradayEffect_VG_DM``. The II, EI, IE
        and EE channels are retained separately. No k-resolved response is
        stored by this validation calculator.

        Indexing
        --------
        - chi_TDSE[ifermi, iomega, drive1, drive2, measure]
        - chi_IFE_channels[name][ifermi, iomega, drive1, drive2, measure]

        For off-diagonal driving, ``chi_TDSE_symmetric`` and
        ``chi_TDSE_quadrature`` contain the directly measured same-phase and
        quadrature cross responses, respectively.
        '''
        super().__init__(velocityProjectionList=velocityProjectionList, highOrderVelocity=True,
                         highOrderDipole=fullCovariantDerivative, **kwarg)
        self.fullCovariantDerivative = bool(fullCovariantDerivative)
        if self.fullCovariantDerivative and self.kBT <= 0:
            raise ValueError("fullCovariantDerivative requires kBT > 0")
        if self.fullCovariantDerivative and abs(self.eta) < 1e-15:
            raise ValueError("fullCovariantDerivative requires finite eta for gauge-covariant relaxation")
        if np.any(np.abs(self.omegaList) < 1e-15):
            raise ValueError("velocity-gauge TDSE IFE is singular at exactly Omega=0")

        nbasis = self.energies.shape[-1]
        M_K = np.zeros(self.kgrid + (3, nbasis, nbasis), dtype=complex)
        if ife_type in ["total", "spin"]: M_K += self.spin_K
        if ife_type in ["total", "orbital"]: M_K += self.orbital_K

        response_shape = (len(self.fermiList), len(self.omegaList), 3, 3, 3)
        self.channel_names = DM_CHANNEL_NAMES
        self.channel_descriptions = {
            "II": "second-order intraband from first-order intraband",
            "EI": "second-order interband from first-order intraband",
            "IE": "second-order intraband from first-order interband",
            "EE": "second-order interband from first-order interband and the contact vertex",
        }
        self.chi_IFE_channels = {name: np.zeros(response_shape, dtype=complex) for name in self.channel_names}
        self.chi_TDSE_channels = self.chi_IFE_channels
        self.chi_TDSE_symmetric_channels = {
            name: np.zeros(response_shape, dtype=float) for name in self.channel_names
        }
        self.chi_TDSE_quadrature_channels = {
            name: np.zeros(response_shape, dtype=float) for name in self.channel_names
        }
        diag = np.eye(nbasis)
        off_diag = 1.0 - diag
        indices = np.arange(nbasis)

        for ifermi, fermi in enumerate(self.fermiList):
            if self.fullCovariantDerivative:
                f, dfdk, d2fdk2 = self.fermiKDerivatives(ifermi)
            else:
                f = np.int8(self.energies < fermi)
                dfdk = d2fdk2 = None
            rho0 = np.zeros(self.energies.shape + (nbasis,), dtype=complex)
            rho0[..., indices, indices] = f

            for iomega, Omega in enumerate(self.omegaList):
                period = 2 * np.pi / Omega
                dt = period / n_steps_per_cycle
                n_cycles_transient = int(np.ceil((10.0 / self.eta) / period))
                n_cycles_measure = 1

                def run_drive(directions, phases):
                    drive_data = self._prepare_covariant_tdse_drive(
                        directions, rho0, f, dfdk, d2fdk2, diag, off_diag
                    )
                    zero = np.zeros_like(rho0)
                    state = tuple(zero.copy() for _ in range(6))
                    t = 0.0
                    for _ in range(n_cycles_transient * n_steps_per_cycle):
                        state = self._rk4_step_channels(state, t, dt, phases, Omega, E0,
                                                        drive_data, diag, off_diag)
                        t += dt

                    M_dc = {name: np.zeros(3) for name in self.channel_names}
                    for _ in range(n_cycles_measure * n_steps_per_cycle):
                        state = self._rk4_step_channels(state, t, dt, phases, Omega, E0,
                                                        drive_data, diag, off_diag)
                        t += dt
                        for ichannel, name in enumerate(self.channel_names, start=2):
                            rho2 = state[ichannel]
                            for c in range(3):
                                value_k = np.sum(rho2 * M_K[..., c, :, :].swapaxes(-1, -2), axis=(-2, -1)).real
                                M_dc[name][c] += np.sum(value_k) / (n_cycles_measure * n_steps_per_cycle)
                    normalization = E0**2 * np.prod(self.kgrid)
                    return {name: value / normalization for name, value in M_dc.items()}

                diagonal_response = {name: np.zeros((3, 3)) for name in self.channel_names}
                for a in range(3):
                    response = run_drive((a,), (0.0,))
                    for name in self.channel_names:
                        diagonal_response[name][a] = response[name]
                        self.chi_IFE_channels[name][ifermi, iomega, a, a, :] = response[name]
                        self.chi_TDSE_symmetric_channels[name][ifermi, iomega, a, a, :] = response[name]

                if include_offdiag:
                    for a in range(3):
                        for b in range(a + 1, 3):
                            same_phase = run_drive((a, b), (0.0, 0.0))
                            quadrature_phase = run_drive((a, b), (0.0, -np.pi / 2))
                            for name in self.channel_names:
                                cross_symmetric = same_phase[name] - diagonal_response[name][a] - diagonal_response[name][b]
                                cross_quadrature = quadrature_phase[name] - diagonal_response[name][a] - diagonal_response[name][b]
                                self.chi_TDSE_symmetric_channels[name][ifermi, iomega, a, b, :] = cross_symmetric
                                self.chi_TDSE_symmetric_channels[name][ifermi, iomega, b, a, :] = cross_symmetric
                                self.chi_TDSE_quadrature_channels[name][ifermi, iomega, a, b, :] = cross_quadrature
                                self.chi_TDSE_quadrature_channels[name][ifermi, iomega, b, a, :] = -cross_quadrature
                                self.chi_IFE_channels[name][ifermi, iomega, a, b, :] = 0.5 * (
                                    cross_symmetric - 1j * cross_quadrature
                                )
                                self.chi_IFE_channels[name][ifermi, iomega, b, a, :] = 0.5 * (
                                    cross_symmetric + 1j * cross_quadrature
                                )

        self.chi_IFE = sum(self.chi_IFE_channels.values())
        self.chi_TDSE = self.chi_IFE
        self.chi_TDSE_symmetric = sum(self.chi_TDSE_symmetric_channels.values())
        self.chi_TDSE_quadrature = sum(self.chi_TDSE_quadrature_channels.values())
        self.projected_sigma = self.chi_IFE[None, None, None, ...]
        self.projected_sigma_channels = {
            name: response[None, None, None, ...] for name, response in self.chi_IFE_channels.items()
        }
        self.projected_sigma_complex = self.projected_sigma
        self.projected_sigma_complex_channels = self.projected_sigma_channels
        for name in self.channel_names:
            setattr(self, f"chi_IFE_{name}", self.chi_IFE_channels[name])
            setattr(self, f"chi_TDSE_{name}", self.chi_TDSE_channels[name])
            setattr(self, f"projected_sigma_{name}", self.projected_sigma_channels[name])
        if self.verbose == 2: showElapsedTime(extraInfo="TDSE IFE finished")

    def _prepare_covariant_tdse_drive(self, directions, rho0, f, dfdk, d2fdk2, diag, off_diag):
        def commutator(left, right):
            return left @ right - right @ left

        data = {
            "velocity": [self.velocity_K[..., direction, :, :] for direction in directions],
            "comm_v_rho0": [], "comm_w_rho0": [], "q1_I": [], "q1_E": [], "c2": [],
        }
        data["comm_v_rho0"] = [commutator(v, rho0) for v in data["velocity"]]
        for direction_i in directions:
            row = []
            for direction_j in directions:
                w_ij = self.w_K[..., direction_i, direction_j, :, :]
                row.append(commutator(w_ij, rho0))
            data["comm_w_rho0"].append(row)

        if self.fullCovariantDerivative:
            for direction in directions:
                c1_i, c1_e = self.fermiCovariantFirstDerivativeChannels(f, dfdk, direction)
                data["q1_I"].append(1j * c1_i * diag)
                data["q1_E"].append(1j * c1_e * off_diag)
            for direction_i in directions:
                row = []
                for direction_j in directions:
                    row.append(self.fermiCovariantSecondDerivativeChannels(
                        f, dfdk, d2fdk2, direction_i, direction_j
                    ))
                data["c2"].append(row)
        return data

    def _rk4_step_channels(self, state, t, dt, phases, Omega, E0, drive_data, diag, off_diag):
        def commutator(left, right):
            return left @ right - right @ left

        def free_evolution(rho):
            return -1j * self.deltaE * rho - self.eta * rho

        def derivative(current_state, time):
            rho1_i, rho1_e, rho2_ii, rho2_ei, rho2_ie, rho2_ee = current_state
            A_t = [-2 * E0 / Omega * np.sin(Omega * time + phase) for phase in phases]
            result = [free_evolution(rho) for rho in current_state]

            for i, A_i in enumerate(A_t):
                result[1] += 1j * A_i * drive_data["comm_v_rho0"][i]
                if self.fullCovariantDerivative:
                    result[0] += self.eta * A_i * drive_data["q1_I"][i]
                    result[1] += self.eta * A_i * drive_data["q1_E"][i]

                source_i = 1j * A_i * commutator(drive_data["velocity"][i], rho1_i)
                source_e = 1j * A_i * commutator(drive_data["velocity"][i], rho1_e)
                result[3] += source_i * off_diag
                result[4] += source_e * diag
                result[5] += source_e * off_diag

                for j, A_j in enumerate(A_t):
                    result[5] -= 0.5j * A_i * A_j * drive_data["comm_w_rho0"][i][j] * off_diag
                    if self.fullCovariantDerivative:
                        coefficient = -0.5 * self.eta * A_i * A_j
                        for ichannel, name in enumerate(self.channel_names, start=2):
                            result[ichannel] += coefficient * drive_data["c2"][i][j][name]
            return tuple(result)

        k1 = derivative(state, t)
        k2 = derivative(tuple(rho + 0.5 * dt * k for rho, k in zip(state, k1)), t + 0.5 * dt)
        k3 = derivative(tuple(rho + 0.5 * dt * k for rho, k in zip(state, k2)), t + 0.5 * dt)
        k4 = derivative(tuple(rho + dt * k for rho, k in zip(state, k3)), t + dt)
        return tuple(rho + (dt / 6.0) * (a + 2*b + 2*c + d)
                     for rho, a, b, c, d in zip(state, k1, k2, k3, k4))

class InverseFaradayEffect_BlochAcceleration_TDSE(Spin, Orbital):
    def __init__(self, ife_type="total", velocityProjectionList=[], E0=0.001,
                 n_steps_per_cycle=100, include_offdiag=True, **kwarg) -> None:
        '''
        Computing IFE by time propagation with the finite Bloch-acceleration
        Hamiltonian H(k-A(t)), evaluated directly from real-space hoppings.

        This is the nonperturbative counterpart of InverseFaradayEffect_TDSE.
        It currently supports orthogonal local bases only.

        Indexing
        --------
        - chi_IFE[ifermi, iomega, ialpha_drive1, ialpha_drive2, ibeta_measure] : np.ndarray(dtype=float)
        '''
        assert len(velocityProjectionList) == 0, "Velocity projections are not used in Bloch-acceleration IFE"
        super().__init__(**kwarg)
        assert not self.model.overlap, "Bloch-acceleration TDSE currently supports only orthogonal bases"

        nbasis = self.energies.shape[-1]
        M_K = np.zeros(self.kgrid + (3, nbasis, nbasis), dtype=complex)
        if ife_type in ["total", "spin"]: M_K += self.spin_K
        if ife_type in ["total", "orbital"]: M_K += self.orbital_K

        self._R_cart = self.model.iRvec @ self.model.lattice
        self._r_cart = self.model.rList
        if not self.phase:
            self._ham_R_phase = self.model.ham_R
        else:
            assert len(self.phaseshift) == self.model.nRvec, f'{len(self.phaseshift)=} {self.model.nRvec=}'
            self._ham_R_phase = self.model.ham_R * self.phaseshift[:, None, None]

        self.chi_TDSE = np.zeros((len(self.fermiList), len(self.omegaList), 3, 3, 3), dtype=float)
        self.chi_TDSE_symmetric = np.zeros_like(self.chi_TDSE, dtype=float)
        self.chi_TDSE_quadrature = np.zeros_like(self.chi_TDSE, dtype=float)

        H_diag = np.zeros(self.energies.shape + (nbasis,), dtype=complex)
        for iband in range(nbasis):
            H_diag[..., iband, iband] = self.energies[..., iband]

        for ifermi, fermi in enumerate(self.fermiList):
            f = np.int8(self.energies < fermi)
            rho0 = np.zeros_like(H_diag)
            for iband in range(nbasis):
                rho0[..., iband, iband] = f[..., iband]

            M0 = np.zeros(3)
            for ic in range(3):
                M_val0 = np.sum(rho0 * M_K[..., ic, :, :].swapaxes(-1, -2), axis=(-1, -2)).real
                M0[ic] = np.sum(M_val0)

            for iomega, Omega in enumerate(self.omegaList):
                period = 2 * np.pi / Omega
                dt = period / n_steps_per_cycle
                n_cycles_transient = int(np.ceil((10.0 / self.eta) / period))
                n_cycles_measure = 1

                def run_drive(directions, phases):
                    rho = rho0.copy()
                    t = 0.0
                    for _ in range(n_cycles_transient * n_steps_per_cycle):
                        rho = self._rk4_step_bloch_acceleration(rho, t, dt, directions, phases, Omega, rho0, E0)
                        t += dt

                    M_dc = np.zeros(3)
                    for _ in range(n_cycles_measure * n_steps_per_cycle):
                        rho = self._rk4_step_bloch_acceleration(rho, t, dt, directions, phases, Omega, rho0, E0)
                        t += dt

                        for ic in range(3):
                            M_val = np.sum(rho * M_K[..., ic, :, :].swapaxes(-1, -2), axis=(-1, -2)).real
                            M_dc[ic] += np.sum(M_val) / (n_cycles_measure * n_steps_per_cycle)

                    return (M_dc - M0) / E0**2 / np.prod(self.kgrid)

                diagonal_response = np.zeros((3, 3))
                for ialpha in range(3):
                    response = run_drive((ialpha,), (0.0,))
                    diagonal_response[ialpha] = response
                    self.chi_TDSE[ifermi, iomega, ialpha, ialpha, :] = response
                    self.chi_TDSE_symmetric[ifermi, iomega, ialpha, ialpha, :] = response

                if include_offdiag:
                    for ialpha in range(3):
                        for ibeta in range(ialpha + 1, 3):
                            same_phase = run_drive((ialpha, ibeta), (0.0, 0.0))
                            cross_symmetric = same_phase - diagonal_response[ialpha] - diagonal_response[ibeta]

                            quadrature_phase = run_drive((ialpha, ibeta), (0.0, -np.pi / 2))
                            cross_quadrature = quadrature_phase - diagonal_response[ialpha] - diagonal_response[ibeta]

                            self.chi_TDSE_symmetric[ifermi, iomega, ialpha, ibeta, :] = cross_symmetric
                            self.chi_TDSE_symmetric[ifermi, iomega, ibeta, ialpha, :] = cross_symmetric
                            self.chi_TDSE_quadrature[ifermi, iomega, ialpha, ibeta, :] = cross_quadrature
                            self.chi_TDSE_quadrature[ifermi, iomega, ibeta, ialpha, :] = -cross_quadrature

                            self.chi_TDSE[ifermi, iomega, ialpha, ibeta, :] = 0.5 * (cross_symmetric - cross_quadrature)
                            self.chi_TDSE[ifermi, iomega, ibeta, ialpha, :] = 0.5 * (cross_symmetric + cross_quadrature)

        self.chi_IFE = self.chi_TDSE
        self.projected_sigma = self.chi_TDSE[None, None, None, ...]
        if self.verbose == 2: showElapsedTime(extraInfo=f"Bloch-acceleration TDSE IFE finished")

    def _hamiltonian_eigenbasis_at_vector_potential(self, A_cart):
        phase_A = np.exp(-1j * (self._R_cart @ A_cart))
        orbital_phase = np.exp(1j * (A_cart @ self._r_cart))
        ham_R_A = (self._ham_R_phase * phase_A[:, None, None]
                   * orbital_phase[None, :, None] * orbital_phase.conj()[None, None, :])
        ham_R_grid = np.zeros(self.kgrid + (self.model.norbs, self.model.norbs), dtype=complex)
        for iR, Rvec in enumerate(self.model.iRvec):
            ham_R_grid[tuple(Rvec)] = ham_R_A[iR]
        ham_K_AO = np.fft.ifftn(ham_R_grid, axes=(0, 1, 2)) * np.prod(self.kgrid)
        ham_K_AO = 0.5 * (ham_K_AO + ham_K_AO.conj().swapaxes(-1, -2))
        return self.vectors.conj().swapaxes(-2, -1) @ ham_K_AO @ self.vectors

    def _rk4_step_bloch_acceleration(self, rho, t, dt, directions, phases, Omega, rho0, E0):
        def dRho_dt(rho_state, time_val):
            A_t = np.zeros(3)
            for direction, phase in zip(directions, phases):
                A_t[direction] += -2 * E0 / Omega * np.sin(Omega * time_val + phase)
            H_t = self._hamiltonian_eigenbasis_at_vector_potential(A_t)
            comm = H_t @ rho_state - rho_state @ H_t
            return -1j * comm - self.eta * (rho_state - rho0)

        k1 = dRho_dt(rho, t)
        k2 = dRho_dt(rho + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = dRho_dt(rho + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = dRho_dt(rho + dt * k3, t + dt)
        return rho + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

class InverseFaradayTorque_TDSE(Dipole, SpinExchangeTorque, OrbitalTorque):
    def __init__(self, ife_type="total", velocityProjectionList=[], torqueProjectionList=[], E0=0.001,
                 n_steps_per_cycle=100, include_offdiag=True, exchangeVelocity=True,
                 fullCovariantDerivative=False, **kwarg) -> None:
        '''
        Compute light-induced d.c. torque by velocity-gauge time-domain density-matrix propagation.
        unit: \muB eV / (V/\angst.)^2; divide by \hbar in eV fs for \muB/fs / (V/\angst.)^2.

        ``fullCovariantDerivative=True`` uses the field-shifted relaxation
        target through second order and preserves II, EI, IE and EE. The
        field-dependent torque vertices are included consistently: T1*rho1_I
        and T2*rho0 are assigned to II, while T1*rho1_E is assigned to EE.
        ``chi_TDSE`` is the directly comparable complex response tensor.
        '''
        super().__init__(velocityProjectionList=velocityProjectionList, highOrderVelocity=True,
                         highOrderDipole=fullCovariantDerivative, exchangeVelocity=exchangeVelocity,
                         spinTorqueProjectionList=torqueProjectionList,
                         orbitalTorqueProjectionList=torqueProjectionList, **kwarg)
        self.fullCovariantDerivative = bool(fullCovariantDerivative)
        if self.fullCovariantDerivative and self.kBT <= 0:
            raise ValueError("fullCovariantDerivative requires kBT > 0")
        if self.fullCovariantDerivative and abs(self.eta) < 1e-15:
            raise ValueError("fullCovariantDerivative requires finite eta for gauge-covariant relaxation")
        if np.any(np.abs(self.omegaList) < 1e-15):
            raise ValueError("velocity-gauge TDSE IFT is singular at exactly Omega=0")
        if ife_type in ["total", "spin"]: self.buildSpinTorqueVelocityCommutator()
        if ife_type in ["total", "orbital"]: self.buildOrbitalTorqueVelocityCommutator()

        nbasis = self.energies.shape[-1]
        nproj = len(torqueProjectionList) + 1
        proj_T_K = np.zeros((nproj,) + self.kgrid + (3, nbasis, nbasis), dtype=complex)
        proj_TA_K = np.zeros((nproj,) + self.kgrid + (3, 3, nbasis, nbasis), dtype=complex)
        proj_TAA_K = np.zeros((nproj,) + self.kgrid + (3, 3, 3, nbasis, nbasis), dtype=complex)
        if ife_type in ["total", "spin"]:
            proj_T_K[0] += self.spinTorque_K
            proj_TA_K[0] += self.spinTorqueA_K
            proj_TAA_K[0] += self.spinTorqueAA_K
        if ife_type in ["total", "orbital"]:
            proj_T_K[0] += self.orbitalTorque_K
            proj_TA_K[0] += self.orbitalTorqueA_K
            proj_TAA_K[0] += self.orbitalTorqueAA_K
        for iproj in range(len(torqueProjectionList)):
            if ife_type in ["total", "spin"]:
                proj_T_K[iproj+1] += self.projected_spinTorque_K_list[iproj]
                proj_TA_K[iproj+1] += self.projected_spinTorqueA_K_list[iproj]
                proj_TAA_K[iproj+1] += self.projected_spinTorqueAA_K_list[iproj]
            if ife_type in ["total", "orbital"]:
                proj_T_K[iproj+1] += self.projected_orbitalTorque_K_list[iproj]
                proj_TA_K[iproj+1] += self.projected_orbitalTorqueA_K_list[iproj]
                proj_TAA_K[iproj+1] += self.projected_orbitalTorqueAA_K_list[iproj]

        response_shape = (1, 1, nproj, len(self.fermiList), len(self.omegaList), 3, 3, 3)
        self.channel_names = ("II", "EI", "IE", "EE")
        self.channel_descriptions = {
            "II": "intraband channel, including T1*rho1_I and T2*rho0",
            "EI": "second-order interband from first-order intraband",
            "IE": "second-order intraband from first-order interband",
            "EE": "interband channel, including T1*rho1_E and the contact Hamiltonian vertex",
        }
        self.projected_chi_IFT_channels = {
            name: np.zeros(response_shape, dtype=complex) for name in self.channel_names
        }
        self.projected_chi_TDSE_symmetric_channels = {
            name: np.zeros(response_shape, dtype=float) for name in self.channel_names
        }
        self.projected_chi_TDSE_quadrature_channels = {
            name: np.zeros(response_shape, dtype=float) for name in self.channel_names
        }
        diag = np.eye(nbasis)
        off_diag = 1.0 - diag
        indices = np.arange(nbasis)

        for ifermi, fermi in enumerate(self.fermiList):
            if self.fullCovariantDerivative:
                f, dfdk, d2fdk2 = self.fermiKDerivatives(ifermi)
            else:
                f = np.int8(self.energies < fermi)
                dfdk = d2fdk2 = None
            rho0 = np.zeros(self.energies.shape + (nbasis,), dtype=complex)
            rho0[..., indices, indices] = f

            for iomega, Omega in enumerate(self.omegaList):
                period = 2 * np.pi / Omega
                dt = period / n_steps_per_cycle
                n_cycles_transient = int(np.ceil((10.0 / self.eta) / period))
                n_cycles_measure = 1

                def run_drive(directions, phases):
                    drive_data = self._prepare_covariant_tdse_drive(
                        directions, rho0, f, dfdk, d2fdk2, diag, off_diag
                    )
                    zero = np.zeros_like(rho0)
                    state = tuple(zero.copy() for _ in range(6))
                    t = 0.0
                    for _ in range(n_cycles_transient * n_steps_per_cycle):
                        state = self._rk4_step_channels(state, t, dt, phases, Omega, E0,
                                                        drive_data, diag, off_diag)
                        t += dt

                    T_dc = {name: np.zeros((nproj, 3)) for name in self.channel_names}
                    for _ in range(n_cycles_measure * n_steps_per_cycle):
                        state = self._rk4_step_channels(state, t, dt, phases, Omega, E0,
                                                        drive_data, diag, off_diag)
                        t += dt
                        A_t = [-2 * E0 / Omega * np.sin(Omega * t + phase) for phase in phases]
                        rho1_i, rho1_e = state[:2]
                        rho2 = dict(zip(self.channel_names, state[2:]))
                        for iproj_T in range(nproj):
                            for c in range(3):
                                T0_nm = proj_T_K[iproj_T, ..., c, :, :].swapaxes(-1, -2)
                                value_k = {
                                    name: np.sum(rho2[name] * T0_nm, axis=(-2, -1))
                                    for name in self.channel_names
                                }
                                T1 = np.zeros_like(proj_T_K[iproj_T, ..., c, :, :])
                                T2 = np.zeros_like(T1)
                                for i, A_i in enumerate(A_t):
                                    T1 += A_i * proj_TA_K[iproj_T, ..., c, directions[i], :, :]
                                    for j, A_j in enumerate(A_t):
                                        T2 += 0.5 * A_i * A_j * proj_TAA_K[
                                            iproj_T, ..., c, directions[i], directions[j], :, :
                                        ]
                                T1_nm = T1.swapaxes(-1, -2)
                                value_k["II"] += np.sum(rho1_i * T1_nm + rho0 * T2.swapaxes(-1, -2),
                                                        axis=(-2, -1))
                                value_k["EE"] += np.sum(rho1_e * T1_nm, axis=(-2, -1))
                                for name in self.channel_names:
                                    T_dc[name][iproj_T, c] += np.sum(value_k[name].real) / (
                                        n_cycles_measure * n_steps_per_cycle
                                    )
                    normalization = E0**2 * np.prod(self.kgrid)
                    return {name: value / normalization for name, value in T_dc.items()}

                diagonal_response = {name: np.zeros((3, nproj, 3)) for name in self.channel_names}
                for a in range(3):
                    response = run_drive((a,), (0.0,))
                    for name in self.channel_names:
                        diagonal_response[name][a] = response[name]
                        self.projected_chi_IFT_channels[name][0, 0, :, ifermi, iomega, a, a, :] = response[name]
                        self.projected_chi_TDSE_symmetric_channels[name][
                            0, 0, :, ifermi, iomega, a, a, :
                        ] = response[name]

                if include_offdiag:
                    for a in range(3):
                        for b in range(a + 1, 3):
                            same_phase = run_drive((a, b), (0.0, 0.0))
                            quadrature_phase = run_drive((a, b), (0.0, -np.pi / 2))
                            for name in self.channel_names:
                                cross_symmetric = same_phase[name] - diagonal_response[name][a] - diagonal_response[name][b]
                                cross_quadrature = quadrature_phase[name] - diagonal_response[name][a] - diagonal_response[name][b]
                                self.projected_chi_TDSE_symmetric_channels[name][
                                    0, 0, :, ifermi, iomega, a, b, :
                                ] = cross_symmetric
                                self.projected_chi_TDSE_symmetric_channels[name][
                                    0, 0, :, ifermi, iomega, b, a, :
                                ] = cross_symmetric
                                self.projected_chi_TDSE_quadrature_channels[name][
                                    0, 0, :, ifermi, iomega, a, b, :
                                ] = cross_quadrature
                                self.projected_chi_TDSE_quadrature_channels[name][
                                    0, 0, :, ifermi, iomega, b, a, :
                                ] = -cross_quadrature
                                self.projected_chi_IFT_channels[name][
                                    0, 0, :, ifermi, iomega, a, b, :
                                ] = 0.5 * (cross_symmetric - 1j * cross_quadrature)
                                self.projected_chi_IFT_channels[name][
                                    0, 0, :, ifermi, iomega, b, a, :
                                ] = 0.5 * (cross_symmetric + 1j * cross_quadrature)

        self.projected_chi_IFT = sum(self.projected_chi_IFT_channels.values())
        self.chi_IFT_channels = {
            name: response[0, 0, 0] for name, response in self.projected_chi_IFT_channels.items()
        }
        self.chi_IFT = sum(self.chi_IFT_channels.values())
        self.chi_TDSE = self.chi_IFT
        self.chi_TDSE_channels = self.chi_IFT_channels
        self.projected_sigma = self.projected_chi_IFT
        self.projected_sigma_channels = self.projected_chi_IFT_channels
        self.projected_chi_TDSE_symmetric = sum(self.projected_chi_TDSE_symmetric_channels.values())
        self.projected_chi_TDSE_quadrature = sum(self.projected_chi_TDSE_quadrature_channels.values())
        self.chi_TDSE_symmetric_channels = {
            name: response[0, 0, 0] for name, response in self.projected_chi_TDSE_symmetric_channels.items()
        }
        self.chi_TDSE_quadrature_channels = {
            name: response[0, 0, 0] for name, response in self.projected_chi_TDSE_quadrature_channels.items()
        }
        self.chi_TDSE_symmetric = self.projected_chi_TDSE_symmetric[0, 0, 0]
        self.chi_TDSE_quadrature = self.projected_chi_TDSE_quadrature[0, 0, 0]
        for name in self.channel_names:
            setattr(self, f"chi_IFT_{name}", self.chi_IFT_channels[name])
            setattr(self, f"chi_TDSE_{name}", self.chi_TDSE_channels[name])
            setattr(self, f"projected_chi_IFT_{name}", self.projected_chi_IFT_channels[name])
        if self.verbose == 2: showElapsedTime(extraInfo="TDSE IFT finished")

    _prepare_covariant_tdse_drive = InverseFaradayEffect_TDSE._prepare_covariant_tdse_drive
    _rk4_step_channels = InverseFaradayEffect_TDSE._rk4_step_channels

class InverseFaradayTorque_LG_DM(Dipole, SpinExchangeTorque, OrbitalTorque):
    def __init__(self, ife_type="total", velocityProjectionList=[], torqueProjectionList=[],
                 responseOmegaIndexList=[], responseFermiIndexList=None,
                 finiteTemperatureOccupations=False, fullCovariantDerivative=False, **kwarg) -> None:
        '''
        Compute light-induced d.c. torque in the length gauge with a frequency-domain density matrix.
        unit: \muB eV / (V/\angst.)^2; divide by \hbar in eV fs for \muB/fs / (V/\angst.)^2.
        The spin part is the exchange electronic torque -i[mu_S,H_XC(t)].

        ``fullCovariantDerivative=True`` replaces the interband-only dipole
        commutators by iD_k and retains II, EI, IE and EE separately. Explicit
        first-order torque vertices contracted with rho1_I and rho1_E are
        assigned to II and EE, respectively.

        Indexing
        --------
        - chi_IFT[ifermi, iomega, ialpha_drive1, ialpha_drive2, ibeta_measure] : np.ndarray(dtype=complex)
        '''
        super().__init__(velocityProjectionList=velocityProjectionList,
                         highOrderDipole=fullCovariantDerivative,
                         spinTorqueProjectionList=torqueProjectionList,
                         orbitalTorqueProjectionList=torqueProjectionList, **kwarg)
        self.fullCovariantDerivative = bool(fullCovariantDerivative)
        if self.fullCovariantDerivative and self.kBT <= 0:
            raise ValueError("fullCovariantDerivative requires kBT > 0")
        if self.fullCovariantDerivative and abs(self.eta) < 1e-15:
            raise ValueError("fullCovariantDerivative requires finite eta for the dc density matrix")

        nbasis = self.energies.shape[-1]
        # Slot 0 is the full torque; following slots are projected torque operators.
        proj_T_K = np.zeros((len(torqueProjectionList)+1,) + self.kgrid + (3, nbasis, nbasis), dtype=complex)
        if ife_type in ["total", "spin"]: proj_T_K[0] += self.spinTorque_K
        if ife_type in ["total", "orbital"]: proj_T_K[0] += self.orbitalTorque_K
        for iproj in range(len(torqueProjectionList)):
            if ife_type in ["total", "spin"]: proj_T_K[iproj+1] += self.projected_spinTorque_K_list[iproj]
            if ife_type in ["total", "orbital"]: proj_T_K[iproj+1] += self.projected_orbitalTorque_K_list[iproj]

        # H_LG(t) = H0 - E_a(t) r_a, hence the full torque contains
        # T_c(E) = -i[mu_c,H_LG] = T_c(0) + E_a i[mu_c,r_a].
        proj_torqueE_K = np.zeros((len(torqueProjectionList)+1,) + self.kgrid + (3, 3, nbasis, nbasis), dtype=complex)
        if ife_type in ["total", "spin"]:
            self.buildSpinTorqueDipoleCommutator()
            proj_torqueE_K[0] += self.spinTorqueE_K
        if ife_type in ["total", "orbital"]:
            self.buildOrbitalTorqueDipoleCommutator()
            proj_torqueE_K[0] += self.orbitalTorqueE_K
        for iproj in range(len(torqueProjectionList)):
            if ife_type in ["total", "spin"]: proj_torqueE_K[iproj+1] += self.projected_spinTorqueE_K_list[iproj]
            if ife_type in ["total", "orbital"]: proj_torqueE_K[iproj+1] += self.projected_orbitalTorqueE_K_list[iproj]

        response_shape = (1, 1, len(torqueProjectionList)+1,
                          len(self.fermiList), len(self.omegaList), 3, 3, 3)
        self.channel_names = DM_CHANNEL_NAMES
        self.channel_descriptions = {
            "II": "intraband channel, including torqueE*rho1_I",
            "EI": "second-order interband from first-order intraband",
            "IE": "second-order intraband from first-order interband",
            "EE": "interband channel, including torqueE*rho1_E",
        }
        self.projected_chi_IFT_channels = {
            name: np.zeros(response_shape, dtype=complex) for name in self.channel_names
        }
        responseOmegaIndexList = _normalize_response_index_list(responseOmegaIndexList, len(self.omegaList))
        if responseFermiIndexList is None and len(responseOmegaIndexList) > 0:
            responseFermiIndexList = range(len(self.fermiList))
        responseFermiIndexList = _normalize_response_index_list(responseFermiIndexList, len(self.fermiList))
        responseOmegaRecord = {iomega: irecord for irecord, iomega in enumerate(responseOmegaIndexList)}
        responseFermiRecord = {ifermi: irecord for irecord, ifermi in enumerate(responseFermiIndexList)}
        projected_response_shape = ((1, 1, len(torqueProjectionList)+1) + self.kgrid +
                                    (len(responseFermiIndexList), len(responseOmegaIndexList), 3, 3, 3))
        self.projected_response_channels = {
            name: np.zeros(projected_response_shape, dtype=complex) for name in self.channel_names
        }

        kernel = LengthGaugeSecondOrderDMKernel(
            self, self.fullCovariantDerivative, finiteTemperatureOccupations
        )
        self.density_matrix_kernel = kernel
        for ifermi, fermi in enumerate(self.fermiList):
            fermi_state = kernel.fermi_state(ifermi, fermi)
            for iomega, Omega in enumerate(self.omegaList):
                irecord_omega = responseOmegaRecord.get(iomega)
                frequency_state = fermi_state.at_frequency(Omega)
                for a in range(3):
                    for b in range(3):
                        block = frequency_state.block(a, b)
                        for iproj_T, T_K in enumerate(proj_T_K):
                            for c in range(3):
                                T_c_nm = T_K[..., c, :, :].swapaxes(-1, -2)
                                TE_c_b_nm = proj_torqueE_K[iproj_T, ..., c, b, :, :].swapaxes(-1, -2)
                                TE_c_a_nm = proj_torqueE_K[iproj_T, ..., c, a, :, :].swapaxes(-1, -2)
                                response_k = {
                                    name: np.sum(rho2 * T_c_nm, axis=(-2, -1))
                                    for name, rho2 in block.rho2.items()
                                }
                                response_k["EE"] += np.sum(
                                    block.rho1_minus["E"] * TE_c_b_nm
                                    + block.rho1_plus["E"] * TE_c_a_nm, axis=(-2, -1)
                                )
                                if self.fullCovariantDerivative:
                                    response_k["II"] += np.sum(
                                        block.rho1_minus["I"] * TE_c_b_nm
                                        + block.rho1_plus["I"] * TE_c_a_nm, axis=(-2, -1)
                                    )
                                irecord_fermi = responseFermiRecord.get(ifermi)
                                for name in self.channel_names:
                                    self.projected_chi_IFT_channels[name][
                                        0, 0, iproj_T, ifermi, iomega, a, b, c
                                    ] = np.sum(response_k[name]) / np.prod(self.kgrid)
                                    if irecord_fermi is not None and irecord_omega is not None:
                                        self.projected_response_channels[name][
                                            0, 0, iproj_T, :, :, :, irecord_fermi, irecord_omega, a, b, c
                                        ] = response_k[name]

        self.projected_chi_IFT = sum(self.projected_chi_IFT_channels.values())
        self.projected_response = sum(self.projected_response_channels.values())
        self.chi_IFT_channels = {
            name: response[0, 0, 0] for name, response in self.projected_chi_IFT_channels.items()
        }
        self.chi_IFT = sum(self.chi_IFT_channels.values())
        self.projected_sigma = self.projected_chi_IFT
        self.projected_sigma_channels = self.projected_chi_IFT_channels
        for name in self.channel_names:
            setattr(self, f"chi_IFT_{name}", self.chi_IFT_channels[name])
            setattr(self, f"projected_chi_IFT_{name}", self.projected_chi_IFT_channels[name])
            setattr(self, f"projected_response_{name}", self.projected_response_channels[name])
        if self.verbose == 2: showElapsedTime(extraInfo=f"Length Gauge IFT density matrix finished")

class InverseFaradayTorque_VG_DM(Dipole, SpinExchangeTorque, OrbitalTorque):
    def __init__(self, ife_type="total", velocityProjectionList=[], torqueProjectionList=[],
                 responseOmegaIndexList=[], responseFermiIndexList=None,
                 finiteTemperatureOccupations=False, exchangeVelocity=True,
                 fullCovariantDerivative=False, **kwarg) -> None:
        '''
        Compute light-induced d.c. torque in the velocity gauge with a frequency-domain density matrix.
        unit: \muB eV / (V/\angst.)^2; divide by \hbar in eV fs for \muB/fs / (V/\angst.)^2.

        ``fullCovariantDerivative=True`` adds the gauge-covariant relaxation
        vertices and preserves II, EI, IE and EE. T1*rho1_I and T2*rho0 are
        assigned to II; T1*rho1_E is assigned to EE.
        '''
        super().__init__(velocityProjectionList=velocityProjectionList, highOrderVelocity=True,
                         highOrderDipole=fullCovariantDerivative, exchangeVelocity=exchangeVelocity,
                         spinTorqueProjectionList=torqueProjectionList,
                         orbitalTorqueProjectionList=torqueProjectionList, **kwarg)
        self.fullCovariantDerivative = bool(fullCovariantDerivative)
        if self.fullCovariantDerivative and self.kBT <= 0:
            raise ValueError("fullCovariantDerivative requires kBT > 0")
        if self.fullCovariantDerivative and abs(self.eta) < 1e-15:
            raise ValueError("fullCovariantDerivative requires finite eta for gauge-covariant relaxation")

        nbasis = self.energies.shape[-1]
        nproj = len(torqueProjectionList) + 1
        proj_T_K = np.zeros((nproj,) + self.kgrid + (3, nbasis, nbasis), dtype=complex)
        proj_TA_K = np.zeros((nproj,) + self.kgrid + (3, 3, nbasis, nbasis), dtype=complex)
        proj_TAA_K = np.zeros((nproj,) + self.kgrid + (3, 3, 3, nbasis, nbasis), dtype=complex)
        if ife_type in ["total", "spin"]:
            self.buildSpinTorqueVelocityCommutator()
            proj_T_K[0] += self.spinTorque_K
            proj_TA_K[0] += self.spinTorqueA_K
            proj_TAA_K[0] += self.spinTorqueAA_K
        if ife_type in ["total", "orbital"]:
            self.buildOrbitalTorqueVelocityCommutator()
            proj_T_K[0] += self.orbitalTorque_K
            proj_TA_K[0] += self.orbitalTorqueA_K
            proj_TAA_K[0] += self.orbitalTorqueAA_K
        for iproj in range(len(torqueProjectionList)):
            if ife_type in ["total", "spin"]:
                proj_T_K[iproj+1] += self.projected_spinTorque_K_list[iproj]
                proj_TA_K[iproj+1] += self.projected_spinTorqueA_K_list[iproj]
                proj_TAA_K[iproj+1] += self.projected_spinTorqueAA_K_list[iproj]
            if ife_type in ["total", "orbital"]:
                proj_T_K[iproj+1] += self.projected_orbitalTorque_K_list[iproj]
                proj_TA_K[iproj+1] += self.projected_orbitalTorqueA_K_list[iproj]
                proj_TAA_K[iproj+1] += self.projected_orbitalTorqueAA_K_list[iproj]

        response_shape = (1, 1, nproj, len(self.fermiList), len(self.omegaList), 3, 3, 3)
        self.channel_names = DM_CHANNEL_NAMES
        self.channel_descriptions = {
            "II": "intraband channel, including T1*rho1_I and T2*rho0",
            "EI": "second-order interband from first-order intraband",
            "IE": "second-order intraband from first-order interband",
            "EE": "interband channel, including T1*rho1_E and the contact Hamiltonian vertex",
        }
        self.projected_chi_IFT_channels = {
            name: np.zeros(response_shape, dtype=complex) for name in self.channel_names
        }
        responseOmegaIndexList = _normalize_response_index_list(responseOmegaIndexList, len(self.omegaList))
        if responseFermiIndexList is None and len(responseOmegaIndexList) > 0:
            responseFermiIndexList = range(len(self.fermiList))
        responseFermiIndexList = _normalize_response_index_list(responseFermiIndexList, len(self.fermiList))
        responseOmegaRecord = {iomega: irecord for irecord, iomega in enumerate(responseOmegaIndexList)}
        responseFermiRecord = {ifermi: irecord for irecord, ifermi in enumerate(responseFermiIndexList)}
        projected_response_shape = ((1, 1, nproj) + self.kgrid +
                                    (len(responseFermiIndexList), len(responseOmegaIndexList), 3, 3, 3))
        self.projected_response_channels = {
            name: np.zeros(projected_response_shape, dtype=complex) for name in self.channel_names
        }

        kernel = VelocityGaugeSecondOrderDMKernel(
            self, self.fullCovariantDerivative, finiteTemperatureOccupations
        )
        self.density_matrix_kernel = kernel
        for ifermi, fermi in enumerate(self.fermiList):
            fermi_state = kernel.fermi_state(ifermi, fermi)

            relaxation_response_k = {
                name: np.zeros((nproj,) + self.kgrid + (3, 3, 3), dtype=complex)
                for name in self.channel_names
            }
            if self.fullCovariantDerivative:
                for a in range(3):
                    for b in range(3):
                        rho2_relaxation = fermi_state.relaxation(a, b)
                        for name in self.channel_names:
                            for iproj_T in range(nproj):
                                for c in range(3):
                                    relaxation_response_k[name][iproj_T, ..., a, b, c] = np.sum(
                                        rho2_relaxation[name]
                                        * proj_T_K[iproj_T, ..., c, :, :].swapaxes(-1, -2),
                                        axis=(-2, -1)
                                    )
                        del rho2_relaxation

            for iomega, Omega in enumerate(self.omegaList):
                irecord_fermi = responseFermiRecord.get(ifermi)
                irecord_omega = responseOmegaRecord.get(iomega)
                frequency_state = fermi_state.at_frequency(Omega)
                for a in range(3):
                    for b in range(3):
                        block = frequency_state.block(a, b)
                        for iproj_T in range(nproj):
                            for c in range(3):
                                T0_nm = proj_T_K[iproj_T, ..., c, :, :].swapaxes(-1, -2)
                                T1_b_nm = proj_TA_K[iproj_T, ..., c, b, :, :].swapaxes(-1, -2)
                                T1_a_nm = proj_TA_K[iproj_T, ..., c, a, :, :].swapaxes(-1, -2)
                                T2_ab = 0.5 * (
                                    proj_TAA_K[iproj_T, ..., c, a, b, :, :]
                                    + proj_TAA_K[iproj_T, ..., c, b, a, :, :]
                                )
                                response_k = {
                                    name: np.sum(rho2 * T0_nm, axis=(-2, -1))
                                    + relaxation_response_k[name][iproj_T, ..., a, b, c]
                                    for name, rho2 in block.rho2.items()
                                }
                                response_k["EE"] += np.sum(
                                    block.rho1_minus["E"] * T1_b_nm
                                    + block.rho1_plus["E"] * T1_a_nm, axis=(-2, -1)
                                )
                                response_k["II"] += np.sum(
                                    block.rho0 * T2_ab.swapaxes(-1, -2), axis=(-2, -1)
                                )
                                if self.fullCovariantDerivative:
                                    response_k["II"] += np.sum(
                                        block.rho1_minus["I"] * T1_b_nm
                                        + block.rho1_plus["I"] * T1_a_nm, axis=(-2, -1)
                                    )
                                for name in self.channel_names:
                                    sigma_k = response_k[name] / Omega**2
                                    self.projected_chi_IFT_channels[name][
                                        0, 0, iproj_T, ifermi, iomega, a, b, c
                                    ] = np.sum(sigma_k) / np.prod(self.kgrid)
                                    if irecord_fermi is not None and irecord_omega is not None:
                                        self.projected_response_channels[name][
                                            0, 0, iproj_T, :, :, :, irecord_fermi, irecord_omega, a, b, c
                                        ] = sigma_k

        self.projected_chi_IFT = sum(self.projected_chi_IFT_channels.values())
        self.projected_response = sum(self.projected_response_channels.values())
        self.chi_IFT_channels = {
            name: response[0, 0, 0] for name, response in self.projected_chi_IFT_channels.items()
        }
        self.chi_IFT = sum(self.chi_IFT_channels.values())
        self.projected_sigma = self.projected_chi_IFT
        self.projected_sigma_channels = self.projected_chi_IFT_channels
        for name in self.channel_names:
            setattr(self, f"chi_IFT_{name}", self.chi_IFT_channels[name])
            setattr(self, f"projected_chi_IFT_{name}", self.projected_chi_IFT_channels[name])
            setattr(self, f"projected_response_{name}", self.projected_response_channels[name])
        if self.verbose == 2: showElapsedTime(extraInfo="Velocity Gauge IFT density matrix finished")

class SpinBerryCurv(SpinVelocity):
    def __init__(self, velocityProjectionList=[], spinvelocityProjectionList=[], **kwarg) -> None:
        '''
        Calc. spin Berry curvature \Omega^{alpha,gamma}_{k,mn}.

        alpha can be x, y, z, denoting the (cart.) space direction. 
        gamma can be x, y, or z, corresponding to the spin magnetic-moment orientation.
        k denotes the crystal moment, m, n index the bands.

        Indexing: spinBerryCurv[ik1, ik2, ik3, ialpha, igamma, ieig]
        '''
        super().__init__(velocityProjectionList=velocityProjectionList, spinvelocityProjectionList=spinvelocityProjectionList, **kwarg)
        self.calcDeltaEInverse()
        dEInverse2 = self.deltaEInverse**2

        self.projected_spinBerryCurv=np.zeros((len(velocityProjectionList)+1,len(spinvelocityProjectionList)+1)+self.kgrid+(3,3,3,self.model.norbs-self.ibmin))
        for ivelocity, proj_velocity_K in enumerate([self.velocity_K]+self.projected_velocity_K_list):
            for ispinvelocty, proj_spinVelocity_K in enumerate([self.spinVelocity_K]+self.projected_spinVelocity_K_list):
                for ialpha in range(3):
                    for ibeta in range(3):
                        for igamma in range(3):
                            self.projected_spinBerryCurv[ivelocity,ispinvelocty,...,ialpha,ibeta,igamma,:]=-2*np.diagonal(proj_spinVelocity_K[...,ibeta,igamma,:,:] @ (dEInverse2*proj_velocity_K[...,ialpha,:,:]),0,-2,-1).imag
        
        self.spinBerryCurv=self.projected_spinBerryCurv[0,0]
        if self.verbose==2: showElapsedTime(extraInfo=f"Spin Berry curvature finished")

class SpinHallConductance(SpinBerryCurv):
    def __init__(self, currentProjectionList=[], spincurrentProjectionList=[], **kwarg) -> None:
        '''
        Spin hall conductance in (S/m hbar/2e)
        '''
        super().__init__(velocityProjectionList=currentProjectionList, spinvelocityProjectionList=spincurrentProjectionList, **kwarg)

        nProjections=self.projected_spinBerryCurv.shape[:2]
        projected_sigma=np.zeros(nProjections+(len(self.fermiList),3,3,3))
        
        for ifermi, fermi in enumerate(self.fermiList):
            occupy=self.energies<fermi
            for ivelocity in range(nProjections[0]):
                for ispinvelocity in range(nProjections[1]):
                    for ialpha in range(3):
                        for ibeta in range(3):
                            for igamma in range(3):
                                projected_sigma[ivelocity,ispinvelocity,ifermi,ialpha,ibeta,igamma]=np.sum(self.projected_spinBerryCurv[ivelocity,ispinvelocity,...,ialpha,ibeta,igamma,:][occupy])
            #exit()
        
        #self.projected_sigma=projected_sigma/np.prod(self.kgrid)*(elementary_charge**2/hbar/angstrom/2)/self.model.Vcell
        self.projected_sigma=projected_sigma*(2*np.pi/(np.linalg.det(self.model.lattice[:2,:2])*np.prod(self.kgrid[:2]))) # for checking the TKNN SHC of 2D topo./trivial sys.
        if self.verbose==2: showElapsedTime(extraInfo=f"Spin Hall conductance finished")

class OrbitalBerryCurv(OrbitalVelocity):
    def __init__(self, velocityProjectionList=[], orbitalvelocityProjectionList=[], **kwarg) -> None:
        '''
        Calc. orbital Berry curvature \Omega^{alpha,gamma}_{k,mn}.

        alpha can be x, y, z, denoting the (cart.) space direction. 
        gammma can be Lx, Ly, Lz, corresponding to the orbital-polarization. 
        k denotes the crystal moment, m, n index the bands.

        Indexing: orbitalBerryCurv[ik1, ik2, ik3, ialpha, igamma, ieig]
        '''
        super().__init__(velocityProjectionList=velocityProjectionList, spinvelocityProjectionList=orbitalvelocityProjectionList, **kwarg)
        self.calcDeltaEInverse()
        dEInverse2 = self.deltaEInverse**2

        self.projected_orbitalBerryCurv=np.zeros((len(velocityProjectionList)+1,len(orbitalvelocityProjectionList)+1)+self.kgrid+(3,3,3,self.model.norbs-self.ibmin))
        for ivelocity, proj_velocity_K in enumerate([self.velocity_K]+self.projected_velocity_K_list):
            for iorbitalvelocty, proj_orbitalVelocity_K in enumerate([self.orbitalVelocity_K]+self.projected_orbitalVelocity_K_list):
                for ialpha in range(3):
                    for ibeta in range(3):
                        for igamma in range(3):
                            self.projected_orbitalBerryCurv[ivelocity,iorbitalvelocty,...,ialpha,ibeta,igamma,:]=-2*np.diagonal(proj_orbitalVelocity_K[...,ibeta,igamma,:,:] @ (dEInverse2*proj_velocity_K[...,ialpha,:,:]),0,-2,-1).imag
        
        self.orbitalBerryCurv=self.projected_orbitalBerryCurv[0,0]
        if self.verbose==2: showElapsedTime(extraInfo=f"Orbital Berry curvature finished")

class OrbitalHallConductance(OrbitalBerryCurv):
    def __init__(self, fermiList=[], currentProjectionList=[], orbitalcurrentProjectionList=[], **kwarg) -> None:
        '''
        Orbital hall conductance in (S/m hbar/e)
        '''
        super().__init__(velocityProjectionList=currentProjectionList, orbitalvelocityProjectionList=orbitalcurrentProjectionList, **kwarg)

        sigma=np.zeros((len(fermiList),3,3,3))

        for ifermi, fermi in enumerate(fermiList):
            occupy=self.energies<fermi
            for ialpha in range(3):
                for ibeta in range(3):
                    for igamma in range(3):
                        sigma[ifermi,ialpha,ibeta,igamma]=np.sum(self.orbitalBerryCurv[...,ialpha,ibeta,igamma,:][occupy])
            #exit()
        
        self.sigma=sigma/np.prod(self.kgrid)*(elementary_charge**2/hbar/angstrom/2)/self.model.Vcell
        #self.sigma=sigma*(2*np.pi/(np.linalg.det(self.model.lattice[:2,:2])*np.prod(self.kgrid[:2]))) # for checking the TKNN SHC of 2D topo./trivial sys.
        if self.verbose==2: showElapsedTime(extraInfo=f"Orbital Hall conductance finished")

class SpinvelocityVelocityCorrelation_diag(SpinVelocity_diag, Velocity_diag):
    def __init__(self, velocityProjectionList=[], spinvelocityProjectionList=[], **kwarg) -> None:
        '''
        Calc. intra-band contributions on the correlation func of spin-velocity and velocity \Omega^{alpha,beta,gamma}_{k,m}.

        alpha and beta can be x, y, z, denoting the (cart.) space direction of applied currents and the propagation of spin currents
        gammma can be Sx, Sy, Sz, corresponding to the orientation of spin-polarization. 
        k denotes the crystal moment, m indexes the bands.

        Indexing
        --------
        spinVelocityCorrelation_diag[ik1, ik2, ik3, ialpha, ibeta, igamma, ieig]
        '''
        super().__init__(velocityProjectionList=velocityProjectionList,spinvelocityProjectionList=spinvelocityProjectionList, **kwarg)

        self.projected_spinvelocityVelocityCorrelation_diag=np.zeros((len(velocityProjectionList)+1,len(spinvelocityProjectionList)+1)+self.kgrid+(3,3,3,self.ibmax+1-self.ibmin))
        for ivelocity, proj_velocity_diag_K in enumerate([self.velocity_diag_K]+self.projected_velocity_diag_K_list):
            for ispinvelocity, proj_spinvelocity_diag_K in enumerate([self.spinVelocity_diag_K]+self.projected_spinVelocity_diag_K_list):
                for ialpha in range(3):
                    for ibeta in range(3):
                        for igamma in range(3):
                            self.projected_spinvelocityVelocityCorrelation_diag[ivelocity,ispinvelocity,...,ialpha,ibeta,igamma,:]=-(proj_spinvelocity_diag_K[...,ibeta,igamma,:]*proj_velocity_diag_K[...,ialpha,:]).real/self.eta
        
        self.spinvelocityVelocityCorrelation_diag=self.projected_spinvelocityVelocityCorrelation_diag[0,0]
        if self.verbose==2: showElapsedTime(extraInfo=f"evaluations on intra band contributions on correlation functions <Spinvelocity><velocity> finished")

class SpinHallConductance_diag(SpinvelocityVelocityCorrelation_diag):
    def __init__(self, currentProjectionList=[], spincurrentProjectionList=[], **kwarg) -> None:
        '''
        Intra band contributions on the spin Hall conductance in (S/m hbar/2e)
        '''
        super().__init__(velocityProjectionList=currentProjectionList, spinvelocityProjectionList=spincurrentProjectionList, **kwarg)

        nProjections=self.projected_spinvelocityVelocityCorrelation_diag.shape[:2]
        projected_sigma=np.zeros(nProjections+(len(self.fermiList),3,3,3))
        
        for ifermi, weight in enumerate(self.fermiWeight):
            for ivelocity in range(nProjections[0]):
                for ispinvelocity in range(nProjections[1]):
                    for ialpha in range(3):
                        for ibeta in range(3):
                            for igamma in range(3):
                                projected_sigma[ivelocity,ispinvelocity,ifermi,ialpha,ibeta,igamma]=np.sum(self.projected_spinvelocityVelocityCorrelation_diag[ivelocity,ispinvelocity,...,ialpha,ibeta,igamma,:]*weight)
        
        self.projected_sigma=projected_sigma/np.prod(self.kgrid)*(elementary_charge**2/hbar/angstrom/2)/self.model.Vcell
        #self.projected_sigma=projected_sigma*(2*np.pi/(np.linalg.det(self.model.lattice[:2,:2])*np.prod(self.kgrid[:2]))) # for checking the TKNN SHC of 2D topo./trivial sys.
        if self.verbose==2: showElapsedTime(extraInfo=f"Intra-band spin Galvanic conductance finished")

class SpinVelocityCorrelation(Spin,Velocity):
    def __init__(self, velocityProjectionList=[], spinProjectionList=[], **kwarg) -> None:
        '''
        Calc. inter-band contributions on the correlation func of spin and velocity \Omega^{alpha,gamma}_{k,m}.

        alpha can be x, y, z, denoting the (cart.) space direction. 
        gammma can be Sx, Sy, Sz, corresponding to the orientation of spin-polarization. 
        k denotes the crystal moment, m indexes the bands.

        Indexing
        --------
        spinVelocityCorrelation[ik1, ik2, ik3, ialpha, igamma, ieig]
        '''
        super().__init__(velocityProjectionList=velocityProjectionList,spinProjectionList=spinProjectionList, **kwarg)
        self.calcDeltaEInverse()
        dEInverse2 = self.deltaEInverse2

        self.projected_spinVelocityCorrelation=np.zeros((len(velocityProjectionList)+1,len(spinProjectionList)+1)+self.kgrid+(3,3,self.model.norbs-self.ibmin))
        for ivelocity, proj_velocity_K in enumerate([self.velocity_K]+self.projected_velocity_K_list):
            for ispin, proj_spin_K in enumerate([self.spin_K]+self.projected_spin_K_list):
                for ialpha in range(3):
                    for igamma in range(3):
                        self.projected_spinVelocityCorrelation[ivelocity,ispin,...,ialpha,igamma,:]=-2*np.diagonal(proj_spin_K[...,igamma,:,:] @ (dEInverse2*proj_velocity_K[...,ialpha,:,:]),0,-2,-1).imag
        
        self.spinVelocityCorrelation=self.projected_spinVelocityCorrelation[0,0]
        if self.verbose==2: showElapsedTime(extraInfo=f"evaluations on inter band contributions on correlation functions <Spin \dot velocity> finished")

class SpinGalvanicConductance(SpinVelocityCorrelation):
    def __init__(self, fermiList=[], currentProjectionList=[], spinProjectionList=[], **kwarg) -> None:
        '''
        Inter band contributions on the spin Galvanic conductance in (1/cell 1/(V/angst.) hbar/2)
        '''
        super().__init__(velocityProjectionList=currentProjectionList, spinProjectionList=spinProjectionList, **kwarg)

        nProjections=self.projected_spinVelocityCorrelation.shape[:2]
        projected_sigma=np.zeros(nProjections+(len(fermiList),3,3))

        for ifermi, fermi in enumerate(fermiList):
            occupy=self.energies<fermi
            for ivelocity in range(nProjections[0]):
                for ispin in range(nProjections[1]):
                    for ialpha in range(3):
                        for igamma in range(3):
                            projected_sigma[ivelocity,ispin,ifermi,ialpha,igamma]=np.sum(self.projected_spinVelocityCorrelation[ivelocity,ispin,...,ialpha,igamma,:][occupy])
        
        self.projected_sigma=projected_sigma/np.prod(self.kgrid)
        if self.verbose==2: showElapsedTime(extraInfo=f"Inter-band spin Galvanic conductance finished")

class OrbitalVelocityCorrelation(Orbital,Velocity):
    def __init__(self, velocityProjectionList=[], orbitalProjectionList=[], **kwarg) -> None:
        '''
        Calc. inter-band contributions to the correlation function of
        orbital magnetic moment and velocity.

        alpha can be x, y, z, denoting the (cart.) space direction. 
        gammma can be Lx, Ly, Lz, corresponding to the orientation of orbital-polarization. 
        k denotes the crystal moment, m indexes the bands.

        Indexing
        --------
        orbitalVelocityCorrelation[ik1, ik2, ik3, ialpha, igamma, ieig]
        '''
        super().__init__(velocityProjectionList=velocityProjectionList,orbitalProjectionList=orbitalProjectionList, **kwarg)
        self.calcDeltaEInverse()
        dEInverse2 = self.deltaEInverse2

        self.projected_orbitalVelocityCorrelation=np.zeros((len(velocityProjectionList)+1,len(orbitalProjectionList)+1)+self.kgrid+(3,3,self.model.norbs-self.ibmin))
        for ivelocity, proj_velocity_K in enumerate([self.velocity_K]+self.projected_velocity_K_list):
            for iorbital, proj_orbital_K in enumerate([self.orbital_K]+self.projected_orbital_K_list):
                for ialpha in range(3):
                    for igamma in range(3):
                        self.projected_orbitalVelocityCorrelation[ivelocity,iorbital,...,ialpha,igamma,:]=-2*np.diagonal(proj_orbital_K[...,igamma,:,:] @ (dEInverse2*proj_velocity_K[...,ialpha,:,:]),0,-2,-1).imag
        
        self.orbitalVelocityCorrelation=self.projected_orbitalVelocityCorrelation[0,0]
        if self.verbose==2: showElapsedTime(extraInfo=f"evaluations on inter band contributions on correlation functions <L \dot velocity> finished")

class OrbitalGalvanicConductance(OrbitalVelocityCorrelation):
    def __init__(self, fermiList=[], currentProjectionList=[], orbitalProjectionList=[], **kwarg) -> None:
        '''
        Inter band contributions on the orbital Galvanic conductance in (1/cell 1/(V/angst.) hbar)
        '''
        super().__init__(velocityProjectionList=currentProjectionList, orbitalProjectionList=orbitalProjectionList, **kwarg)

        nProjections=self.projected_orbitalVelocityCorrelation.shape[:2]
        projected_sigma=np.zeros(nProjections+(len(fermiList),3,3))

        for ifermi, fermi in enumerate(fermiList):
            occupy=self.energies<fermi
            for ivelocity in range(nProjections[0]):
                for iorbital in range(nProjections[1]):
                    for ialpha in range(3):
                        for igamma in range(3):
                            projected_sigma[ivelocity,iorbital,ifermi,ialpha,igamma]=np.sum(self.projected_orbitalVelocityCorrelation[ivelocity,iorbital,...,ialpha,igamma,:][occupy])
        
        self.projected_sigma=projected_sigma/np.prod(self.kgrid)
        if self.verbose==2: showElapsedTime(extraInfo=f"Inter-band orbital Galvanic conductance finished")

class SpinVelocityCorrelation_diag(Spin_diag, Velocity_diag):
    def __init__(self, velocityProjectionList=[], spinProjectionList=[], **kwarg) -> None:
        '''
        Calc. intra-band contributions on the correlation func of spin and velocity \Omega^{alpha,gamma}_{k,m}.

        alpha can be x, y, z, denoting the (cart.) space direction. 
        gammma can be Sx, Sy, Sz, corresponding to the orientation of spin-polarization. 
        k denotes the crystal moment, m indexes the bands.

        Indexing
        --------
        spinVelocityCorrelation_diag[ik1, ik2, ik3, ialpha, igamma, ieig]
        '''
        super().__init__(velocityProjectionList=velocityProjectionList,spinProjectionList=spinProjectionList, **kwarg)

        self.projected_spinVelocityCorrelation_diag=np.zeros((len(velocityProjectionList)+1,len(spinProjectionList)+1)+self.kgrid+(3,3,self.ibmax+1-self.ibmin))
        for ivelocity, proj_velocity_K in enumerate([self.velocity_diag_K]+self.projected_velocity_diag_K_list):
            for ispin, proj_spin_K in enumerate([self.spin_diag_K]+self.projected_spin_diag_K_list):
                for ialpha in range(3):
                    for igamma in range(3):
                        self.projected_spinVelocityCorrelation_diag[ivelocity,ispin,...,ialpha,igamma,:]=-(proj_spin_K[...,igamma,:]*proj_velocity_K[...,ialpha,:]).real/self.eta
        
        self.spinVelocityCorrelation_diag=self.projected_spinVelocityCorrelation_diag[0,0]
        if self.verbose==2: showElapsedTime(extraInfo=f"evaluations on intra band contributions on correlation functions <Spin><velocity> finished")

class SpinGalvanicConductance_diag(SpinVelocityCorrelation_diag):
    def __init__(self, currentProjectionList=[], spinProjectionList=[], **kwarg) -> None:
        '''
        Intra band contributions on the spin Galvanic conductance in (1/cell 1/(V/angst.) hbar/2)
        '''
        super().__init__(velocityProjectionList=currentProjectionList, spinProjectionList=spinProjectionList, **kwarg)

        nProjections=self.projected_spinVelocityCorrelation_diag.shape[:2]
        projected_sigma=np.zeros(nProjections+(len(self.fermiList),3,3))
        
        for ifermi, weight in enumerate(self.fermiWeight):
            for ivelocity in range(nProjections[0]):
                for ispin in range(nProjections[1]):
                    for ialpha in range(3):
                        for igamma in range(3):
                            projected_sigma[ivelocity,ispin,ifermi,ialpha,igamma]=np.sum(self.projected_spinVelocityCorrelation_diag[ivelocity,ispin,...,ialpha,igamma,:]*weight)
        
        self.projected_sigma=projected_sigma/np.prod(self.kgrid)
        if self.verbose==2: showElapsedTime(extraInfo=f"Intra-band spin Galvanic conductance finished")

class OrbitalVelocityCorrelation_diag(Orbital_diag, Velocity_diag):
    def __init__(self, velocityProjectionList=[], orbitalProjectionList=[], **kwarg) -> None:
        '''
        Calc. intra-band contributions to the correlation function of
        orbital magnetic moment and velocity.

        alpha can be x, y, z, denoting the (cart.) space direction. 
        gammma can be Lx, Ly, Lz, corresponding to the orientation of orbital-polarization. 
        k denotes the crystal moment, m indexes the bands.

        Indexing
        --------
        orbitalVelocityCorrelation_diag[ik1, ik2, ik3, ialpha, igamma, ieig]
        '''
        super().__init__(velocityProjectionList=velocityProjectionList,orbitalProjectionList=orbitalProjectionList, **kwarg)

        self.projected_orbitalVelocityCorrelation_diag=np.zeros((len(velocityProjectionList)+1,len(orbitalProjectionList)+1)+self.kgrid+(3,3,self.ibmax+1-self.ibmin))
        for ivelocity, proj_velocity_K in enumerate([self.velocity_diag_K]+self.projected_velocity_diag_K_list):
            for iorbital, proj_orbital_K in enumerate([self.orbital_diag_K]+self.projected_orbital_diag_K_list):
                for ialpha in range(3):
                    for igamma in range(3):
                        self.projected_orbitalVelocityCorrelation_diag[ivelocity,iorbital,...,ialpha,igamma,:]=-(proj_orbital_K[...,igamma,:]*proj_velocity_K[...,ialpha,:]).real/self.eta
        
        self.orbitalVelocityCorrelation_diag=self.projected_orbitalVelocityCorrelation_diag[0,0]
        if self.verbose==2: showElapsedTime(extraInfo=f"evaluations on intra band contributions on correlation functions <L><velocity> finished")

class OrbitalGalvanicConductance_diag(OrbitalVelocityCorrelation_diag):
    def __init__(self, currentProjectionList=[], orbitalProjectionList=[], **kwarg) -> None:
        '''
        Intra band contributions on the orbital Galvanic conductance in (1/cell 1/(V/angst.) hbar)
        '''
        super().__init__(velocityProjectionList=currentProjectionList, orbitalProjectionList=orbitalProjectionList, **kwarg)

        nProjections=self.projected_orbitalVelocityCorrelation_diag.shape[:2]
        projected_sigma=np.zeros(nProjections+(len(self.fermiList),3,3))
        
        for ifermi, weight in enumerate(self.fermiWeight):
            for ivelocity in range(nProjections[0]):
                for iorbital in range(nProjections[1]):
                    for ialpha in range(3):
                        for igamma in range(3):
                            projected_sigma[ivelocity,iorbital,ifermi,ialpha,igamma]=np.sum(self.projected_orbitalVelocityCorrelation_diag[ivelocity,iorbital,...,ialpha,igamma,:]*weight)
        
        self.projected_sigma=projected_sigma/np.prod(self.kgrid)
        if self.verbose==2: showElapsedTime(extraInfo=f"Intra-band orbital Galvanic conductance finished")

class SpintorqueVelocityCorrelation(SpinExchangeTorque,Velocity):
    def __init__(self, velocityProjectionList=[], spinTorqueProjectionList=[], **kwarg) -> None:
        '''
        Calc. correlation func of spin torque and velocity \Omega^{alpha,gamma}_{k,mn}.
        Only the exchange torque, not the full torque, suitable to fed to LLG dynamics 

        alpha can be x, y, z, denoting the (cart.) space direction. 
        gamma can be x, y, or z, corresponding to the spin magnetic-moment torque orientation.
        k denotes the crystal moment, m, n index the bands.

        Indexing
        --------
        spinBerryCurv[ik1, ik2, ik3, ialpha, igamma, ieig]
        '''
        super().__init__(velocityProjectionList=velocityProjectionList,
                         spinTorqueProjectionList=spinTorqueProjectionList, **kwarg)
        self.calcDeltaEInverse()
        dEInverse2 = self.deltaEInverse**2

        self.projected_spintorqueVelocityCorrelation=np.zeros((len(velocityProjectionList)+1,len(spinTorqueProjectionList)+1)+self.kgrid+(3,3,self.model.norbs-self.ibmin))
        for ivelocity, proj_velocity_K in enumerate([self.velocity_K]+self.projected_velocity_K_list):
            for ispinTorque, proj_spinTorque_K in enumerate([self.spinTorque_K]+self.projected_spinTorque_K_list):
                for ialpha in range(3):
                    for igamma in range(3):
                        self.projected_spintorqueVelocityCorrelation[ivelocity,ispinTorque,...,ialpha,igamma,:]=-2*np.diagonal(proj_spinTorque_K[...,igamma,:,:] @ (dEInverse2*proj_velocity_K[...,ialpha,:,:]),0,-2,-1).imag
        
        if self.verbose==2: showElapsedTime(extraInfo=f"evaluations on inter band contributions on correlation function <Torque_Spin velocity> finished")

class SpintorqueGalvanicConductance(SpintorqueVelocityCorrelation):
    def __init__(self, currentProjectionList=[], spinTorqueProjectionList=[], **kwarg) -> None:
        '''
        Inter band contributions on the spin torque Galvanic conductance in (1/cell eV /(V/angst.) hbar/2)
        Only the exchange torque, not the full torque, suitable to fed to LLG dynamics 
        '''
        super().__init__(velocityProjectionList=currentProjectionList, spinTorqueProjectionList=spinTorqueProjectionList, **kwarg)

        nProjections=self.projected_spintorqueVelocityCorrelation.shape[:2]
        projected_sigma=np.zeros(nProjections+(len(self.fermiList),3,3))

        for ifermi, fermi in enumerate(self.fermiList):
            occupy=self.energies<fermi
            for ivelocity in range(nProjections[0]):
                for ispin in range(nProjections[1]):
                    for ialpha in range(3):
                        for igamma in range(3):
                            projected_sigma[ivelocity,ispin,ifermi,ialpha,igamma]=np.sum(self.projected_spintorqueVelocityCorrelation[ivelocity,ispin,...,ialpha,igamma,:][occupy])
        
        self.projected_sigma=projected_sigma/np.prod(self.kgrid)
        if self.verbose==2: showElapsedTime(extraInfo=f"Inter-band spin Galvanic conductance finished")

class Chern_FHS(Energy):
    def __init__(self, **kwarg):
        '''
        Computes the exact Chern number OF 2D MATERIAL (z-surf) using the Fukui-Hatsugai-Suzuki (FHS) 
        U(N) link variable method.
        
        This algorithm guarantees an exact integer output and converges 
        orders of magnitude faster than the Kubo velocity integration.

        Parameters
        ----------
        - model : System
            the model for computation
        - kgrid : tuple(3)
            kgrid for sampling the BZ
        '''
        # Hijack the Energy class to build the Hamiltonian and diagonalize it over the grid
        super().__init__(**kwarg)
        
        # 1. Identify the occupied manifold
        # Assuming a global gap, we find the number of occupied bands at the Gamma point
        n_occ_Gamma = np.sum(self.energies[0, 0, 0] <= self.model.fermi)
        n_occ = np.sum(self.energies <= self.model.fermi)//np.prod(self.kgrid)
        assert n_occ==n_occ_Gamma, f"The occ. at Gamma {n_occ_Gamma} is inconsistent to occ. on BZ {n_occ}, recheck {self.model.fermi=}"
        self.n_occ=n_occ
        if self.verbose: 
            print(f"\n--- FHS Chern Number Calculation ---")
            print(f"Occupied bands identified: {self.n_occ} out of {self.model.norbs}")
            
        # Extract the occupied eigenvectors over the 2D grid
        # Shape: (Nx, Ny, norbs, n_occ)
        V = self.vectors[:, :, 0, :, :self.n_occ]
        
        # Determine the conjugate transpose, factoring in overlap if non-orthogonal
        if self.model.overlap:
            O_K = self.O_K_AO[:, :, 0, :, :]
            V_dag = np.matmul(V.conj().transpose(0, 1, 3, 2), O_K)
        else:
            V_dag = V.conj().transpose(0, 1, 3, 2)
            
        # 2. Compute U(N) Link Variables
        # Shift vectors to get adjacent k-points. np.roll natively handles the periodic boundary!
        V_shift_x = np.roll(V, shift=-1, axis=0)
        V_shift_y = np.roll(V, shift=-1, axis=1)
        
        # Overlap matrices: M_x = V^\dagger(k) V(k+dx)
        M_x = np.matmul(V_dag, V_shift_x)
        M_y = np.matmul(V_dag, V_shift_y)
        
        # Determinants of the overlap matrices
        det_x = np.linalg.det(M_x)
        det_y = np.linalg.det(M_y)
        
        # Link variables (normalized U(1) phases of the U(N) determinant)
        U_x = det_x / np.abs(det_x)
        U_y = det_y / np.abs(det_y)
        
        # 3. Compute Plaquette Flux (Lattice Berry Curvature)
        U_x_shift_y = np.roll(U_x, shift=-1, axis=1)
        U_y_shift_x = np.roll(U_y, shift=-1, axis=0)
        
        # F_12 = ln( U_1(k) * U_2(k+1) * U_1(k+2)^-1 * U_2(k)^-1 )
        # np.log automatically takes the principal branch, wrapping the phase to (-pi, pi]
        F_12 = np.log(U_x * U_y_shift_x * np.conj(U_x_shift_y) * np.conj(U_y))
        
        # The Chern number is the sum of the fluxes divided by 2*pi*i
        self.chern_number = np.sum(F_12.imag) / (2 * np.pi)
        
        if self.verbose:
            print(f"Computed Chern Number: {self.chern_number:.6f}")
            print(f"Nearest Integer: {int(np.round(self.chern_number))}")
            showElapsedTime(extraInfo="FHS Chern calculation finished")

class Z2_FHS(Energy):
    def __init__(self, **kwarg):
        '''
        Computes the Z2 topological invariant OF 2D MATERIAL (z-surf) using the Spin-Chern / FHS method.
        
        Projects the S_z operator onto the occupied bands, dynamically splits them 
        into '+' and '-' spin sectors, and applies the FHS algorithm to the '+' sector.
        Z2 = C_+ mod 2.

        Parameters
        ----------
        - model : System
            the model for computation
        - kgrid : tuple(3)
            kgrid for sampling the BZ
        '''
        super().__init__(**kwarg)
        
        n_occ_Gamma = np.sum(self.energies[0, 0, 0] <= self.model.fermi)
        n_occ = np.sum(self.energies <= self.model.fermi)//np.prod(self.kgrid)
        assert n_occ==n_occ_Gamma, f"The occ. at Gamma {n_occ_Gamma} is inconsistent to occ. on BZ {n_occ}, recheck {self.model.fermi=}"
        self.n_occ=n_occ
        if self.verbose: 
            print(f"\n--- FHS Z2 (Spin-Chern) Calculation ---")
            print(f"Occupied bands identified: {self.n_occ} out of {self.model.norbs}")
            
        V = self.vectors[:, :, 0, :, :self.n_occ]
        
        if self.model.overlap:
            O_K = self.O_K_AO[:, :, 0, :, :]
            V_dag = np.matmul(V.conj().transpose(0, 1, 3, 2), O_K)
        else:
            V_dag = V.conj().transpose(0, 1, 3, 2)
            
        # 1. Project S_z onto the occupied subspace
        Sz_op = self.model.spin[2] # sigma_z from your Spin class matrices
        
        # P S_z P = V^\dagger S_z V
        Sz_matrix = np.matmul(Sz_op, V)
        if self.model.overlap:
            Sz_matrix = np.matmul(O_K, Sz_matrix)
            
        P_Sz_P = np.matmul(V_dag, Sz_matrix)
        
        # 2. Diagonalize the projected S_z
        # eigh automatically sorts eigenvalues from lowest to highest
        spin_evals, spin_evecs = np.linalg.eigh(P_Sz_P)
        
        # The upper half of the sorted eigenvectors belongs to the "+" spin sector
        half_occ = self.n_occ // 2
        V_plus_coeff = spin_evecs[:, :, :, half_occ:]
        
        # Rotate the occupied tight-binding vectors into the purified "+" spin sector
        V_plus = np.matmul(V, V_plus_coeff)
        
        # 3. Apply FHS logic strictly to the "+" sector
        if self.model.overlap:
            V_plus_dag = np.matmul(V_plus.conj().transpose(0, 1, 3, 2), O_K)
        else:
            V_plus_dag = V_plus.conj().transpose(0, 1, 3, 2)
            
        V_plus_shift_x = np.roll(V_plus, shift=-1, axis=0)
        V_plus_shift_y = np.roll(V_plus, shift=-1, axis=1)
        
        M_x = np.matmul(V_plus_dag, V_plus_shift_x)
        M_y = np.matmul(V_plus_dag, V_plus_shift_y)
        
        det_x = np.linalg.det(M_x)
        det_y = np.linalg.det(M_y)
        
        U_x = det_x / np.abs(det_x)
        U_y = det_y / np.abs(det_y)
        
        U_x_shift_y = np.roll(U_x, shift=-1, axis=1)
        U_y_shift_x = np.roll(U_y, shift=-1, axis=0)
        
        F_12 = np.log(U_x * U_y_shift_x * np.conj(U_x_shift_y) * np.conj(U_y))
        
        self.chern_plus = np.sum(F_12.imag) / (2 * np.pi)
        self.z2_invariant = int(np.round(np.abs(self.chern_plus))) % 2
        
        if self.verbose:
            print(f"Spin-Chern C_+ : {self.chern_plus:.6f}")
            print(f"Z2 Invariant   : {self.z2_invariant}")
            showElapsedTime(extraInfo="FHS Z2 calculation finished")

class RetardedLatticeGreenFunc(Energy):
    def __init__(self, latVecList=None, relevantOrbitalIndexList=None, calcGreen=False, **kwarg) -> None:
        '''
        Compute retarded lattice Green's functions on selected real-space
        lattice vectors and selected local orbitals.

        Indexing:
        self.rG_w_lat[iomega, ilat, iorb, jorb]

        The orbital indexes of the last two axes follow
        self.relevantOrbitalIndexList.
        '''
        super().__init__(**kwarg)
        self.rG_w_lat = None
        self.latVecList = None
        self.relevantOrbitalIndexList = None
        self.orbitalIndexMap = None
        if calcGreen:
            assert latVecList is not None, "latVecList is required when calcGreen=True"
            assert relevantOrbitalIndexList is not None, "relevantOrbitalIndexList is required when calcGreen=True"
            self.calcRetardedLatticeGreenFunc(
                omegaList=self.omegaList,
                latVecList=latVecList,
                relevantOrbitalIndexList=relevantOrbitalIndexList,
            )

    def calcRetardedLatticeGreenFunc(self, omegaList, latVecList, relevantOrbitalIndexList):
        '''
        Calculate G_R(z) = 1/Nk sum_k exp(-i k.R) G_k(z).

        Parameters
        ----------
        omegaList : array-like(complex)
            Complex energy points.
        latVecList : array-like(int), shape (nLat, 3)
            Bravais lattice vectors where the Green's function is needed.
        relevantOrbitalIndexList : array-like(int)
            Local-basis orbital indexes retained in the Green's function.
        '''
        if self.verbose: showElapsedTime("start computations on retarded lattice Green's function")
        omegaList = np.atleast_1d(np.asarray(omegaList, dtype=complex))
        latVecList = np.asarray(latVecList, dtype=int).reshape((-1,3))
        relevantOrbitalIndexList = np.asarray(relevantOrbitalIndexList, dtype=int)

        self.omegaList = omegaList
        self.latVecList = latVecList
        self.relevantOrbitalIndexList = relevantOrbitalIndexList
        self.orbitalIndexMap = np.zeros(self.model.norbs, dtype=int) - 1
        for iorb_reduced, iorb in enumerate(relevantOrbitalIndexList):
            self.orbitalIndexMap[iorb] = iorb_reduced

        nkpt = int(np.prod(self.kgrid))
        nlat = len(latVecList)
        nbands = self.energies.shape[-1]
        norb_reduced = len(relevantOrbitalIndexList)

        energies = self.energies.reshape((nkpt, nbands))
        vectors = self.vectors.reshape((nkpt, self.model.norbs, nbands))
        selected_vectors = vectors[:, relevantOrbitalIndexList, :]

        k_index = np.indices(self.kgrid).reshape((3, nkpt)).T
        k_frac = k_index / np.array(self.kgrid)[None,:]
        lattice_factors = np.exp(-2j*np.pi*(k_frac @ latVecList.T)).T

        selected_vectors_conj = selected_vectors.conj()

        def calc_omega_batch(iomega_start, omega_batch):
            denominator = 1.0/(omega_batch[:, None, None] - energies[None, :, :] + 1j*self.eta)
            G_k = np.einsum(
                'kai,wki,kbi->wkab',
                selected_vectors, denominator, selected_vectors_conj,
                optimize=True,
            )
            rG_batch = np.einsum('lk,wkab->wlab', lattice_factors, G_k, optimize=True) / nkpt
            return iomega_start, rG_batch

        self.rG_w_lat = np.zeros((len(omegaList), nlat, norb_reduced, norb_reduced), dtype=complex)
        omega_batches = [
            (iomega_start, omegaList[iomega_start:iomega_start+self.greenFuncOmegaBatch])
            for iomega_start in range(0, len(omegaList), self.greenFuncOmegaBatch)
        ]
        if self.greenFuncNWorkers == 1 or len(omega_batches) == 1:
            for iomega_start, omega_batch in omega_batches:
                _, rG_batch = calc_omega_batch(iomega_start, omega_batch)
                self.rG_w_lat[iomega_start:iomega_start+len(omega_batch)] = rG_batch
        else:
            with ThreadPoolExecutor(max_workers=self.greenFuncNWorkers) as executor:
                futures = [
                    executor.submit(calc_omega_batch, iomega_start, omega_batch)
                    for iomega_start, omega_batch in omega_batches
                ]
                for future in futures:
                    iomega_start, rG_batch = future.result()
                    self.rG_w_lat[iomega_start:iomega_start+rG_batch.shape[0]] = rG_batch
        
        if self.verbose:
            showElapsedTime(extraInfo=f"retarded lattice Green function on {len(omegaList)} energies, {nlat} lattice vectors finished")
        return self.rG_w_lat

class ExchangeCoupling(RetardedLatticeGreenFunc):
    def __init__(self, exchangeRoutes=None, erange=(-500,500), extraEnergyEmpty=5.0,
                 lowOrder=5, highOrder=20, Jcvg=1e-3,
                 symmetricExchangeSign=-1.0, dmiSign=1.0, **kwarg) -> None:
        '''
        Compute exchange couplings from retarded lattice Green's functions.

        Parameters
        ----------
        exchangeRoutes : list[dict] | list[list[dict]]
            Routes returned by ``System.classifyExchangeRoutes``. A flat route
            list is also accepted.
        erange : tuple(float, float)
            Hard energy integration window. The upper bound is also clipped by
            each Fermi level.
        extraEnergyEmpty : float
            Extra energy margin below the band bottom.
        lowOrder, highOrder : int
            Gaussian-quadrature orders tested sequentially. The interval is
            ``range(lowOrder, highOrder)``.
        Jcvg : float
            Convergence threshold for all computed couplings. Negative values
            disable convergence checking and use the first quadrature order.
        symmetricExchangeSign, dmiSign : float
            Sign-convention factors. The default symmetric-exchange sign matches
            the historical ``couplingCalc.py`` output for the current route and
            Green-function convention.
        '''
        self.exchangeRoutes = self._flattenExchangeRoutes(exchangeRoutes)
        self.erange = erange
        self.extraEnergyEmpty = extraEnergyEmpty
        self.lowOrder = lowOrder
        self.highOrder = highOrder
        self.Jcvg = Jcvg
        self.symmetricExchangeSign = symmetricExchangeSign
        self.dmiSign = dmiSign
        super().__init__(**kwarg)
        assert self.model.ispin == 2, "ExchangeCoupling requires a spinful model with ispin=2"
        assert len(self.exchangeRoutes) > 0, "no exchange route is provided"
        self.calcExchangeCoupling()

    @staticmethod
    def _flattenExchangeRoutes(exchangeRoutes):
        if exchangeRoutes is None:
            return []
        if len(exchangeRoutes) == 0:
            return []
        if isinstance(exchangeRoutes[0], dict):
            return list(exchangeRoutes)
        routes = []
        for group in exchangeRoutes:
            routes.extend(group)
        return routes

    def _prepareRouteMaps(self):
        overlat_set = {(0,0,0)}
        relevant_orbitals = set()
        half_norbs = self.model.norbs//2
        for route in self.exchangeRoutes:
            overlat = tuple(int(x) for x in route["overlat"])
            overlat_set.add(overlat)
            overlat_set.add(tuple(-np.array(overlat, dtype=int)))
            for iorb in route["source_orbitals"]:
                relevant_orbitals.add(int(iorb))
                relevant_orbitals.add(int(iorb) + half_norbs)
            for iorb in route["target_orbitals"]:
                relevant_orbitals.add(int(iorb))
                relevant_orbitals.add(int(iorb) + half_norbs)

        self.overlatList = np.array(sorted(overlat_set), dtype=int)
        self.overlatIndex = {tuple(overlat): ilat for ilat, overlat in enumerate(self.overlatList)}
        self.relevantOrbitalIndexList = np.array(sorted(relevant_orbitals), dtype=int)

    def _routeOrbitalIndexes(self, route):
        half_norbs = self.model.norbs//2
        source_up = [int(iorb) for iorb in route["source_orbitals"]]
        target_up = [int(iorb) for iorb in route["target_orbitals"]]
        source_full = source_up + [iorb + half_norbs for iorb in source_up]
        target_full = target_up + [iorb + half_norbs for iorb in target_up]
        source_reduced = list(self.orbitalIndexMap[source_full])
        target_reduced = list(self.orbitalIndexMap[target_full])
        return source_reduced, target_reduced

    @staticmethod
    def _contractExchange(p_source, p_target, G_forward, G_backward):
        G_forward_blocks = PauliBlock(G_forward)
        G_backward_blocks = PauliBlock(G_backward)
        AMat = np.zeros((4,4), dtype=complex)
        for i in range(4):
            route1 = p_source @ G_forward_blocks[i]
            for j in range(4):
                route2 = p_target @ G_backward_blocks[j]
                AMat[i,j] = np.sum(route1 * route2.T)
        return AMat

    @staticmethod
    def _spinVertex(p_mat, iaxis):
        norb = p_mat.shape[0]
        vertex = np.zeros((2*norb, 2*norb), dtype=complex)
        if iaxis == 0:
            vertex[:norb, norb:] = p_mat
            vertex[norb:, :norb] = p_mat
        elif iaxis == 1:
            vertex[:norb, norb:] = -1j * p_mat
            vertex[norb:, :norb] = 1j * p_mat
        elif iaxis == 2:
            vertex[:norb, :norb] = p_mat
            vertex[norb:, norb:] = -p_mat
        else:
            raise ValueError(f"spin vertex axis should be 0, 1, or 2, got {iaxis}")
        return vertex

    @classmethod
    def _contractSpinRotation(cls, p_source, iaxis_source, p_target, iaxis_target,
                              G_forward, G_backward):
        vertex_source = cls._spinVertex(p_source, iaxis_source)
        vertex_target = cls._spinVertex(p_target, iaxis_target)
        return np.trace(vertex_source @ G_forward @ vertex_target @ G_backward)

    def calcExchangeCoupling(self):
        self._prepareRouteMaps()
        nfermi = len(self.fermiList)
        nroutes = len(self.exchangeRoutes)
        self.exchangeEnergy = np.zeros((nfermi, nroutes))
        self.Jani = np.zeros((nfermi, nroutes, 2, 2))
        self.DMI = np.zeros((nfermi, nroutes, 3))
        self.rotationCurvature = np.zeros((nfermi, nroutes, 3, 3))
        self.exchangeInfo = []

        bandEnergyMin = float(self.energies.min())
        emin = max(self.erange[0], bandEnergyMin) - self.extraEnergyEmpty

        for ifermi, fermi in enumerate(self.fermiList):
            emax = min(self.erange[1], float(fermi))
            J_old = np.zeros((nroutes, 3, 3))
            for iteration_num, gaussianOrder in enumerate(range(self.lowOrder, self.highOrder)):
                points, weights = p_roots(gaussianOrder)
                omega_path, domega_path = semiCirle(emin, emax, points)
                self.calcRetardedLatticeGreenFunc(
                    omegaList=omega_path,
                    latVecList=self.overlatList,
                    relevantOrbitalIndexList=self.relevantOrbitalIndexList,
                )

                J_new = np.zeros((nroutes, 3, 3))
                for iroute, route in enumerate(self.exchangeRoutes):
                    overlat = np.array(route["overlat"], dtype=int)
                    i_overlat_forward = self.overlatIndex[tuple(overlat)]
                    i_overlat_backward = self.overlatIndex[tuple(-overlat)]
                    source_reduced, target_reduced = self._routeOrbitalIndexes(route)
                    p_source = self.model.getAtomicSelfEnergy(route["source_orbitals"])
                    p_target = self.model.getAtomicSelfEnergy(route["target_orbitals"])

                    J = 0j
                    J_a_xx = 0j
                    J_a_yy = 0j
                    J_a_xy = 0j
                    C_xy = 0j
                    C_yx = 0j
                    DMI = np.zeros(3, dtype=complex)
                    for iomega in range(len(omega_path)):
                        G_forward = self.rG_w_lat[iomega, i_overlat_forward][source_reduced,:][:,target_reduced]
                        G_backward = self.rG_w_lat[iomega, i_overlat_backward][target_reduced,:][:,source_reduced]
                        AMat = self._contractExchange(p_source, p_target, G_forward, G_backward)

                        weight = domega_path[iomega] * weights[iomega]
                        J += 2*(AMat[0,0]-AMat[1,1]-AMat[2,2]-AMat[3,3]) * weight
                        J_a_xx += 2*(AMat[1,1]+AMat[1,1]) * weight
                        J_a_yy += 2*(AMat[2,2]+AMat[2,2]) * weight
                        J_a_xy += 2*(AMat[1,2]+AMat[2,1]) * weight
                        DMI[0] += 2j*(AMat[0,1]-AMat[1,0]) * weight
                        DMI[1] += 2j*(AMat[0,2]-AMat[2,0]) * weight
                        C_xy += self._contractSpinRotation(p_source, 0, p_target, 1, G_forward, G_backward) * weight
                        C_yx += self._contractSpinRotation(p_source, 1, p_target, 0, G_forward, G_backward) * weight

                    self.exchangeEnergy[ifermi, iroute] = self.symmetricExchangeSign * J.imag/2
                    self.Jani[ifermi, iroute] = self.symmetricExchangeSign * np.array([[J_a_xx, J_a_xy], [J_a_xy, J_a_yy]]).imag/2
                    self.rotationCurvature[ifermi, iroute, 0, 1] = self.symmetricExchangeSign * C_xy.imag/2
                    self.rotationCurvature[ifermi, iroute, 1, 0] = self.symmetricExchangeSign * C_yx.imag/2
                    DMI[2] = 0.5 * (C_yx - C_xy)
                    self.DMI[ifermi, iroute] = self.dmiSign * DMI.imag/2
                    J_new[iroute] = np.diag([self.exchangeEnergy[ifermi, iroute]]*3)

                dJ_max = float(np.max(np.abs(J_new - J_old)))
                if self.verbose:
                    print(f"ExchangeCoupling fermi {fermi:.6f}, Gaussian order {gaussianOrder}, max dJ {dJ_max:.6e}")
                if self.Jcvg < 0:
                    break
                if iteration_num > 0 and dJ_max < self.Jcvg:
                    break
                J_old = J_new.copy()

            self.exchangeInfo.append({
                "fermi": float(fermi),
                "emin": float(emin),
                "emax": float(emax),
                "gaussianOrder": int(gaussianOrder),
                "max_dJ": dJ_max,
            })

        for iroute, route in enumerate(self.exchangeRoutes):
            route["exchangeEnergy"] = self.exchangeEnergy[:, iroute]
            route["Jani"] = self.Jani[:, iroute]
            route["DMI"] = self.DMI[:, iroute]
        return self.exchangeEnergy
