#
# The module is for calculations on energy, berry curv., and several types of conductances along a given k-line path
#
from system import System
import numpy as np
from auxiliary import showElapsedTime, print_matrix
from scipy.constants import elementary_charge, hbar, angstrom

class CalculatorOnPath():

    def __init__(self,KPOINTS="./KPOINTS",verbose=0) -> None:
        '''
        Calculations on gamma-centered kmesh on BZ

        Parameters:
        -----------
        - model : System
        The basis system, involving the hopping, basis coordinates, lattice matrix and perharps the overlap
        - KPOINTS : str
        The position of file KPOINTS
        - eta   : float
        The relaxation energy, in the same unit with energy, usually eV
        - smr_type : str, "Lorentzian"|"Gaussian"
        - degen : float
        The threthold two states are degenerated
        '''
        if verbose==2: print("initializing calculator module")
        self.model=None
        self.eta=0.
        self.smr_type="Lorentzian"
        self.degen=1e-6
        self.verbose=verbose

        self.kpath=np.loadtxt(KPOINTS,skiprows=3)[:,:3]
        self.nkpt=len(self.kpath)

class EnergyOnPath(CalculatorOnPath):
    def __init__(self, model:System, KPOINTS="./KPOINTS", verbose=0) -> None:
        '''
        Solve the eigen-eqs. to obtain eigen-vals. and vecs.

        Usually the second father of all calculators

        Indexing:
        self.energies[ik, ieig] gives eigen-vals.

        self.vectors[ik1, ieig, jeig] gives eigen-vecs. in ao basis.
        '''
        super().__init__(KPOINTS, verbose)
        self.model=model
        self.ibmin=0             # ignore low-lying bands
        self.ibmax=model.norbs   # ignore high bands
        if verbose==2: showElapsedTime(offset=True)
        
        if model.overlap:
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
            nR,norbs,norbs=O_K.shape
            O_K_half=O_K[...,:norbs//2,:norbs//2]
            eigHalf, WHalf = np.linalg.eigh(O_K_half)
            eigFull=np.zeros((nR,norbs))
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
        
        self.E_K_AO=np.zeros((self.nkpt,self.model.norbs,self.model.norbs),dtype=complex)
        self.O_K_AO=np.zeros((self.nkpt,self.model.norbs,self.model.norbs),dtype=complex)
        for ikpt, kpt in enumerate(self.kpath):
            for Rvec, ER, OR in zip(self.model.iRvec,self.model.energy_R,self.model.overlap_R):
                phase=np.exp(2j*np.pi*kpt@Rvec)
                self.E_K_AO[ikpt]+=phase*ER
                self.O_K_AO[ikpt]+=phase*OR
        if self.verbose==2: showElapsedTime(extraInfo=f"FFT energy_R to energy_K and O_R to O_K on path of {self.nkpt=} finished")

        overlapEigValues, Wk =solveOverlap(self.O_K_AO)
        assert overlapEigValues.min() > 1e-6, f"eigen-value of overlap matrix is too small or negative {overlapEigValues.min()=}"
        if self.verbose==2: print(f"{overlapEigValues.min()=}")

        WkLam=Wk/np.sqrt(overlapEigValues[...,None,:])
        if self.verbose==2: showElapsedTime(extraInfo=f"WkLam")
        EnergyMatrixInK_tilde=WkLam.conj().transpose((0,2,1,)) @ self.E_K_AO @ WkLam
        if self.verbose==2: showElapsedTime(extraInfo=f"Overlapped energy")
        self.energies, vecsBeforeOverlap = np.linalg.eigh(EnergyMatrixInK_tilde)
        if self.verbose==2: showElapsedTime(extraInfo=f"eigen-eqs.")
        self.vectors = WkLam @ vecsBeforeOverlap
        if self.verbose==2: showElapsedTime(extraInfo=f"overlapping on eig.-vecs.")

        #self.getEnergyWindow()

    def solveEigenEquation(self):
        '''
        Solve the eig.-eqs. 

        H \dot U = U \dot \Gamma
        '''
        self.ham_K_W=np.zeros((self.nkpt,self.model.norbs,self.model.norbs),dtype=complex)
        for ikpt, kpt in enumerate(self.kpath):
            for Rvec, ER in zip(self.model.iRvec,self.model.ham_R):
                phase=np.exp(2j*np.pi*kpt@Rvec)
                self.ham_K_W[ikpt]+=phase*ER
        if self.verbose==2: showElapsedTime(extraInfo=f"Ham_R_W to Ham_K_W on path of {self.nkpt=} finished")
        self.energies, self.vectors = np.linalg.eigh(self.ham_K_W)
        if self.verbose==2: showElapsedTime(extraInfo=f"eig.-eqs. solved")
    
    def calcDeltaEInverse(self):
        '''
        Calc. the inversed energy difference matrix, the value at degenerated point is set to zero.

        Indexing: 
        
        deltaE[ik, ieig, jeig] = E_{ki} - E_{kj}

        deltaEInverse[ik, ieig, jeig] = 1/(E_{ki} - E_{kj} - 1j*eta)
        '''
        self.deltaE = self.energies[:,self.ibmin:,None] - self.energies[:,None,self.ibmin:]
        self.deltaEInverse=1./(self.deltaE - 1j*self.eta)
        self.deltaEInverse[abs(self.deltaE)<self.degen]=0
        if self.smr_type == 'Lorentzian':
            return
        raise NotImplementedError()
    
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
        self.ibmin=len(self.energies[0][self.energies[0]<mid_valence])

class VelocityOnPath(EnergyOnPath):
    def __init__(self, model: System, KPOINTS="./KPOINTS", verbose=0) -> None:
        '''
        Compute velocity matrix in original basis and eigen-basis

        Indexing:
        self.RHam_K[ik, idirection, iorb, jorb] gives velocity matrix in original basis,
        idirection can be 0 | 1 | 2 corresponding to x | y | z direction.

        self.velocity_K[ik, idirection, ieig, jeig] gives velocity matrix in eigen-basis.
        idirection can be 0 | 1 | 2 corresponding to x | y | z direction.
        '''
        super().__init__(model, KPOINTS, verbose)
        self.Todd = True
        self.Iodd = True
        
        self.RHam_K=np.zeros((self.nkpt,3,model.norbs,model.norbs),dtype=complex)
        for ikpt, kpt in enumerate(self.kpath):
            for Rvec, RHamR in zip(self.model.iRvec,model.RHam_R):
                phase=np.exp(2j*np.pi*kpt@Rvec)
                self.RHam_K[ikpt]+=1j*phase*RHamR

        if verbose==2: showElapsedTime(extraInfo=f"Exp. sum. RHam_R to RHam_K to calc. velocity in ao-basis finished")
        vectors=self.vectors[:,None,:,self.ibmin:]
        self.velocity_K=vectors.conj().transpose((0,1,3,2)) @ self.RHam_K @ vectors
        if verbose==2: showElapsedTime(extraInfo=f"velocity in eig.-basis finished")

        if model.overlap:
            self.RO_K=np.zeros((self.nkpt,3,model.norbs,model.norbs),dtype=complex)
            for ikpt, kpt in enumerate(self.kpath):
                for Rvec, RO_R in zip(self.model.iRvec,model.RO_R):
                    phase=np.exp(2j*np.pi*kpt@Rvec)
                    self.RO_K[ikpt]+=1j*phase*RO_R

            self.velocity_K -= vectors.conj().transpose((0,1,3,2)) @ self.RO_K @ (vectors *self.energies[...,None,None,self.ibmin:])

class BerryCurvOnPath(VelocityOnPath):
    
    def __init__(self, model: System, KPOINTS="./KPOINTS", eta=1e-4, smr_type="Lorentzian", degen=1e-4, verbose=0) -> None:
        '''
        Calc. Berry curvature \Omega^{alpha}_{k,mn}.

        alpha can be x, y, z corresponds the (cart.) space direction, k denote the crystal moment, m, n index the bands.

        Indexing: BerryCurv[ik1, ik2, ik3, idrection, iband, jband]
        '''
        super().__init__(model, KPOINTS, verbose)
        self.Todd = True

        self.eta=eta
        self.smr_type=smr_type
        self.degen=degen # update the degeneracy threthold
        self.calcDeltaEInverse()
        dEInverse2 = self.deltaEInverse**2
        
        self.BerryCurv=np.zeros((self.nkpt,3,model.norbs-self.ibmin))
        eps=np.arange(3)
        for ialpha in range(3):
            self.BerryCurv[...,ialpha,:]=-2*np.diagonal(self.velocity_K[...,eps[(ialpha+1)%3],:,-4:] @ (dEInverse2[...,-4:,:]*self.velocity_K[...,eps[(ialpha+2)%3],-4:,:]),0,-2,-1).imag
        if verbose==2: showElapsedTime(extraInfo=f"Berry curvature finished")

class SpinVelocityOnPath(VelocityOnPath):
    def __init__(self, model: System, KPOINTS="./KPOINTS", verbose=0) -> None:
        '''
        Compute spin velocity matrix in original basis and eigen-basis

        Indexing:
        self.SRHam_K[ik, idirection, igamma, iorb, jorb] gives spin velocity matrix in original basis,
        idirection can be 0 | 1 | 2, corresponding to x | y | z direction. 
        igamma can be 0 | 1 | 2 | 3, corresponding to sigma^x | sigma^y | sigma^z | sigma^0 spin-polarization. 

        self.spinVelocity_K[ik, idirection, igamma, ieig, jeig] gives spin velocity matrix in eigen-basis.
        idirection can be 0 | 1 | 2, corresponding to x | y | z direction. 
        igamma can be 0 | 1 | 2 | 3, corresponding to sigma^x | sigma^y | sigma^z | sigma^0 spin-polarization. 
        '''
        super().__init__(model, KPOINTS,verbose)
        self.Iodd = True
        #model.getSpinMatrice()
        #showElapsedTime(extraInfo=f"spin op. in ao.-basis constructed")
        vectors=self.vectors[:,None,:,self.ibmin:]
        if model.overlap:
            spin_eig=vectors.conj().transpose((0,1,3,2)) @ model.spin[None,:,:,:] @ self.O_K_AO[...,None,:,:] @ vectors
        else:
            spin_eig=vectors.conj().transpose((0,1,3,2)) @ model.spin[None,:,:,:] @ vectors
        if verbose==2: showElapsedTime(extraInfo=f"spin op. in eig.-basis calculated")
        spin=spin_eig[...,None,:,:,:]
        velocity=self.velocity_K[...,None,:,:]
        sV=spin @ velocity
        self.spinVelocity_K = (sV + sV.transpose((0,1,2,4,3)).conj())/2#(spin @ velocity + velocity @ spin)/2
        if verbose==2: showElapsedTime(extraInfo=f"spin velocity in eig.-basis finished")

class SpinBerryCurvOnPath(SpinVelocityOnPath):
    def __init__(self, model: System, KPOINTS="./KPOINTS", eta=0.005, smr_type="Lorentzian", degen=1e-6, verbose=0) -> None:
        '''
        Calc. spin Berry curvature \Omega^{alpha,gamma}_{k,mn}.

        alpha can be x, y, z, denoting the (cart.) space direction. 
        gammma can be sigma^x, sigma^y, sigma^z, sigma^0, corresponding to the spin-polarization. 
        k denotes the crystal moment, m, n index the bands.

        Indexing: spinBerryCurv[ik, ialpha, igamma, ieig]
        '''
        super().__init__(model, KPOINTS, verbose)
        self.degen=degen # update the degeneracy threthold
        self.eta=eta
        self.smr_type=smr_type
        self.calcDeltaEInverse()
        dEInverse2 = self.deltaEInverse**2

        self.spinBerryCurv=np.zeros((self.nkpt,3,3,3,model.norbs-self.ibmin))
        #eps=np.arange(3)
        for ikpt in range(self.nkpt):
            nvalence=sum(self.energies[ikpt]<=model.fermi)
            for ialpha in range(3):
                for ibeta in range(3):
                    for igamma in range(3):
                        self.spinBerryCurv[ikpt,ialpha,ibeta,igamma,:nvalence]=-2*np.diagonal(self.spinVelocity_K[ikpt,ibeta,igamma,:nvalence,nvalence:] @ (dEInverse2[ikpt,nvalence:,:nvalence]*self.velocity_K[ikpt,ialpha,nvalence:,:nvalence]),0,-2,-1).imag
        if verbose==2: showElapsedTime(extraInfo=f"Spin Berry curvature finished")