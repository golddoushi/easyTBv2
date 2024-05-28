#
# The module is for calculations on energy, berry curv., and several types of conductances
#
from system import System
import numpy as np
from auxiliary import showElapsedTime, print_matrix
from scipy.constants import elementary_charge, hbar, angstrom

class Calculator():
    Iodd = False  # inversion gives minus
    Todd = False  # time reversal gives minus

    def __init__(self,verbose=0) -> None:
        '''
        Calculations on gamma-centered kmesh on BZ

        Parameters:
        -----------
        - model : System
        The basis system, involving the hopping, basis coordinates, lattice matrix and perharps the overlap
        - kgrid : size-3 int. tuple
        The size of kmesh
        - eta   : float
        The relaxation energy, in the same unit with energy, usually eV
        - smr_type : str, "Lorentzian"|"Gaussian"
        - degen : float
        The threthold two states are degenerated
        '''
        if verbose==2: print("initializing calculator module")
        self.model=None
        self.kgrid=(1,1,1)
        self.eta=0.
        self.smr_type="Lorentzian"
        self.degen=1e-6
        self.verbose=verbose

#
# Implemented calculaors
# Energy, velocity (ao and eigen gauges), spin velocity (ao and eigen gauges), 
# Berry curvature, spin Berry curvature
#
class Energy(Calculator):
    def __init__(self, model:System, kgrid=(1,1,1), phaseshift=1, verbose=0) -> None:
        '''
        Solve the eigen-eqs. to obtain eigen-vals. and vecs.

        Usually the second father of all calculators

        Indexing:
        self.energies[ik1, ik2, ik3, ieig] gives eigen-vals.

        self.vectors[ik1, ik2, ik3, ieig, jeig] gives eigen-vecs. in ao basis.
        '''
        super().__init__()
        self.kgrid=tuple(kgrid)
        self.model=model
        self.phase=False
        self.ibmin=0             # ignore low-lying bands
        self.ibmax=model.norbs   # ignore high bands
        if type(phaseshift)==np.ndarray: self.phase=True
        if verbose==2: showElapsedTime(offset=True)
        
        if model.overlap:
            self.solveGeneralEigenEquation(phaseshift)
        else:
            self.solveEigenEquation(phaseshift)

    def solveGeneralEigenEquation(self,phaseshift):
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
            assert len(phaseshift)==self.model.nRvec, f'{len(phaseshift)=} {self.model.nRvec=}'
            
            E_R_AO=self.model.energy_R*phaseshift[:,None,None]
            O_R_AO=self.model.overlap_R*phaseshift[:,None,None]
        '''
        # imposing heimit
        energy_K=np.fft.fftn(self.model.energy_R,axes=(0,1,2,))
        energy_K = (energy_K + energy_K.transpose((0,1,2,4,3)).conj())/2
        O_K=np.fft.fftn(self.model.overlap_R,axes=(0,1,2,))
        O_K = (O_K + O_K.transpose((0,1,2,4,3)).conj())/2
        
        energy_R = np.fft.ifftn(energy_K,axes=(0,1,2,))
        O_R = np.fft.ifftn(O_K,axes=(0,1,2,))
        self.model.setEnergyR(energy_R)
        self.model.setOR(O_R)
        showElapsedTime(extraInfo=f"The hemit condition is imposed")
        #'''
        
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

        self.getEnergyWindow()

    def solveEigenEquation(self,phaseshift):
        '''
        Solve the eig.-eqs. 

        H \dot U = U \dot \Gamma
        '''
        if not self.phase:
            HH_R_W=self.model.ham_R
        else:
            assert len(phaseshift)==self.model.nRvec, f'{len(phaseshift)=} {self.model.nRvec=}'
            
            HH_R_W=self.model.ham_R*phaseshift[:,None,None]
        self.ham_R_intepolated=np.zeros(self.kgrid+(self.model.norbs,self.model.norbs),dtype=complex)
        for iR, Rvec in enumerate(self.model.iRvec):
            self.ham_R_intepolated[tuple(Rvec)]=HH_R_W[iR]
        self.ham_K=np.fft.ifftn(self.ham_R_intepolated,axes=(0,1,2,))*np.prod(self.kgrid)
        if self.verbose==2: showElapsedTime(extraInfo=f"FFT ham_R to ham_K on {self.kgrid=} finished")
        self.energies, self.vectors = np.linalg.eigh(self.ham_K)
        if self.verbose==2: showElapsedTime(extraInfo=f"eig.-eqs. solved")
    
    def calcDeltaEInverse(self):
        '''
        Calc. the inversed energy difference matrix, the value at degenerated point is set to zero.

        Indexing: 
        
        deltaE[ik1, ik2, ik3, ieig, jeig] = E_{ki} - E_{kj}

        deltaEInverse[ik1, ik2, ik3, ieig, jeig] = 1/(E_{ki} - E_{kj} - 1j*eta)
        '''
        self.deltaE = self.energies[:,:,:,self.ibmin:,None] - self.energies[:,:,:,None,self.ibmin:]
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
        mid_valence=(valence_bands[nvb-1-ilevel] + valence_bands[nvb-2-ilevel])/2
        self.ibmin=len(self.energies[(0,0,0)][self.energies[(0,0,0)]<mid_valence])

        #print(self.ibmin,mid_valence,energy[0])


class Velocity(Energy):
    def __init__(self, model: System, kgrid=(1,1,1), phaseshift=1, verbose=0) -> None:
        '''
        Compute velocity matrix in original basis and eigen-basis

        Indexing:
        self.RHam_K[ik1, ik2, ik3, idirection, iorb, jorb] gives velocity matrix in original basis,
        idirection can be 0 | 1 | 2 corresponding to x | y | z direction.

        self.velocity_K[ik1, ik2, ik3, idirection, ieig, jeig] gives velocity matrix in eigen-basis.
        idirection can be 0 | 1 | 2 corresponding to x | y | z direction.
        '''
        super().__init__(model, kgrid, phaseshift, verbose)
        self.Todd = True
        self.Iodd = True
        
        RH_R_W=model.RHam_R
        self.RHam_R_intepolated=np.zeros(self.kgrid+(3,model.norbs,model.norbs),dtype=complex)
        if self.phase:
            RH_R_W=RH_R_W*phaseshift[:,None,None,None]
        for iR, Rvec in enumerate(model.iRvec):
            self.RHam_R_intepolated[tuple(Rvec)]=RH_R_W[iR]
        self.RHam_K= 1j*np.fft.ifftn(self.RHam_R_intepolated,axes=(0,1,2,))*np.prod(self.kgrid)

        if verbose==2: showElapsedTime(extraInfo=f"FFT RHam_R to RHam_K to calc. velocity in ao-basis finished")
        vectors=self.vectors[:,:,:,None,:,self.ibmin:]
        self.velocity_K=vectors.conj().transpose((0,1,2,3,5,4)) @ self.RHam_K @ vectors
        if verbose==2: showElapsedTime(extraInfo=f"velocity in eig.-basis finished")

        if model.overlap:
            RO_R_W=model.RO_R
            self.RO_R_intepolated=np.zeros(self.kgrid+(3,model.norbs,model.norbs),dtype=complex)
            if self.phase:
                RO_R_W=RO_R_W*phaseshift[:,None,None,None]
            for iR, Rvec in enumerate(model.iRvec):
                self.RO_R_intepolated[tuple(Rvec)]=RO_R_W[iR]
            self.RO_K= 1j*np.fft.ifftn(self.RO_R_intepolated,axes=(0,1,2,))*np.prod(self.kgrid)

            self.velocity_K -= vectors.conj().transpose((0,1,2,3,5,4)) @ self.RO_K @ (vectors *self.energies[...,None,None,self.ibmin:])

class BerryCurv(Velocity):
    
    def __init__(self, model: System, kgrid=(1, 1, 1), phaseshift=1, eta=1e-4, smr_type="Lorentzian", degen=1e-6, verbose=0) -> None:
        '''
        Calc. Berry curvature \Omega^{alpha}_{k,mn}.

        alpha can be x, y, z corresponds the (cart.) space direction, k denote the crystal moment, m, n index the bands.

        Indexing: BerryCurv[ik1, ik2, ik3, idrection, iband, jband]
        '''
        super().__init__(model, kgrid, phaseshift, verbose)
        self.Todd = True

        self.eta=eta
        self.smr_type=smr_type
        self.degen=degen # update the degeneracy threthold
        self.calcDeltaEInverse()
        dEInverse2 = self.deltaEInverse**2
        
        self.BerryCurv=np.zeros(self.kgrid+(3,model.norbs-self.ibmin))
        eps=np.arange(3)
        for ialpha in range(3):
            self.BerryCurv[...,ialpha,:]=-2*np.diagonal(self.velocity_K[...,eps[(ialpha+1)%3],:,:] @ (dEInverse2*self.velocity_K[...,eps[(ialpha+2)%3],:,:]),0,-2,-1).imag
        if verbose==2: showElapsedTime(extraInfo=f"Berry curvature finished")

class AnormalousHallConductance(BerryCurv):

    def __init__(self, model: System, kgrid=(1, 1, 1), phaseshift=1, eta=0.0001, smr_type="Lorentzian", degen=0.000001, fermiList=[], verbose=0) -> None:
        super().__init__(model, kgrid, phaseshift, eta, smr_type, degen, verbose)
        self.Todd = True

        sigma=np.zeros(len(fermiList))

        berryCurv=self.BerryCurv[...,2,:]
        for ifermi, fermi in enumerate(fermiList):
            occupy=self.energies[:,:,:,self.ibmin:]<fermi
            print(f"{ifermi=:3d}  {len(self.energies[occupy])/np.prod(self.kgrid):8.4f}")
            sigma[ifermi]=np.sum(berryCurv[occupy])
        
        self.sigma=sigma/np.prod(self.kgrid)*(elementary_charge**2/hbar/angstrom/2)/model.Vcell
        if verbose==2: showElapsedTime(extraInfo=f"Spin Hall conductance finished")

class SpinVelocity(Velocity):
    def __init__(self, model: System, kgrid=(1,1,1), phaseshift=1, verbose=0) -> None:
        '''
        Compute spin velocity matrix in original basis and eigen-basis

        Indexing:
        self.SRHam_K[ik1, ik2, ik3, idirection, igamma, iorb, jorb] gives spin velocity matrix in original basis,
        idirection can be 0 | 1 | 2, corresponding to x | y | z direction. 
        igamma can be 0 | 1 | 2 | 3, corresponding to sigma^x | sigma^y | sigma^z | sigma^0 spin-polarization. 

        self.spinVelocity_K[ik1, ik2, ik3, idirection, igamma, ieig, jeig] gives spin velocity matrix in eigen-basis.
        idirection can be 0 | 1 | 2, corresponding to x | y | z direction. 
        igamma can be 0 | 1 | 2 | 3, corresponding to sigma^x | sigma^y | sigma^z | sigma^0 spin-polarization. 
        '''
        super().__init__(model, kgrid, phaseshift,verbose)
        self.Iodd = True
        #model.getSpinMatrice()
        #showElapsedTime(extraInfo=f"spin op. in ao.-basis constructed")
        vectors=self.vectors[:,:,:,None,:,self.ibmin:]
        if model.overlap:
            spin_eig=vectors.conj().transpose((0,1,2,3,5,4)) @ model.spin[None,None,None,:,:,:] @ self.O_K_AO[...,None,:,:] @ vectors
        else:
            spin_eig=vectors.conj().transpose((0,1,2,3,5,4)) @ model.spin[None,None,None,:,:,:] @ vectors
        if verbose==2: showElapsedTime(extraInfo=f"spin op. in eig.-basis calculated")
        spin=spin_eig[...,None,:,:,:]
        velocity=self.velocity_K[...,None,:,:]
        sV=spin @ velocity
        self.spinVelocity_K = (sV + sV.transpose((0,1,2,3,4,6,5)).conj())/2#(spin @ velocity + velocity @ spin)/2
        if verbose==2: showElapsedTime(extraInfo=f"spin velocity in eig.-basis finished")

class SpinBerryCurv(SpinVelocity):
    def __init__(self, model: System, kgrid=(1, 1, 1), phaseshift=1, eta=0.005, smr_type="Lorentzian", degen=1e-6, verbose=0) -> None:
        '''
        Calc. spin Berry curvature \Omega^{alpha,gamma}_{k,mn}.

        alpha can be x, y, z, denoting the (cart.) space direction. 
        gammma can be sigma^x, sigma^y, sigma^z, sigma^0, corresponding to the spin-polarization. 
        k denotes the crystal moment, m, n index the bands.

        Indexing: spinBerryCurv[ik1, ik2, ik3, ialpha, igamma, ieig]
        '''
        super().__init__(model, kgrid, phaseshift, verbose)
        self.degen=degen # update the degeneracy threthold
        self.eta=eta
        self.smr_type=smr_type
        self.calcDeltaEInverse()
        dEInverse2 = self.deltaEInverse**2

        self.spinBerryCurv=np.zeros(self.kgrid+(3,3,3,model.norbs-self.ibmin))
        #eps=np.arange(3)
        for ialpha in range(3):
            for ibeta in range(3):
                for igamma in range(3):
                    self.spinBerryCurv[...,ialpha,ibeta,igamma,:]=-2*np.diagonal(self.spinVelocity_K[...,ibeta,igamma,:,:] @ (dEInverse2*self.velocity_K[...,ialpha,:,:]),0,-2,-1).imag
        if verbose==2: showElapsedTime(extraInfo=f"Spin Berry curvature finished")

class SpinHallConductance(SpinBerryCurv):
    def __init__(self, model: System, kgrid=(1, 1, 1), phaseshift=1, eta=0.005, smr_type="Lorentzian", degen=0.000001, fermiList=[0.,], verbose=0) -> None:
        '''
        Spin hall conductance in (S/m hbar/2e)
        '''
        super().__init__(model, kgrid, phaseshift, eta, smr_type, degen, verbose)

        sigma=np.zeros((len(fermiList),3,3,3))

        for ifermi, fermi in enumerate(fermiList):
            occupy=self.energies[:,:,:,self.ibmin:]<fermi
            for ialpha in range(3):
                for ibeta in range(3):
                    for igamma in range(3):
                        sigma[ifermi,ialpha,ibeta,igamma]=np.sum(self.spinBerryCurv[...,ialpha,ibeta,igamma,:][occupy])
            #exit()
        
        self.sigma=sigma/np.prod(self.kgrid)*(elementary_charge**2/hbar/angstrom/2)/model.Vcell
        if verbose==2: showElapsedTime(extraInfo=f"Spin Hall conductance finished")
