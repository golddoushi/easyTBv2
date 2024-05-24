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
        if type(phaseshift)==np.ndarray: self.phase=True
        if verbose==2: showElapsedTime(offset=True)
        
        if model.overlap:
            self.solveGeneralEigenEquation()
        else:
            self.solveEigenEquation(phaseshift)

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

        '''
        # 第一种方案，在能量矩阵的（小）格子上设置哈密顿系数矩阵
        O_K=np.fft.fftn(self.model.overlap_R,axes=(0,1,2,))
        energy_K=np.fft.fftn(self.model.energy_R,axes=(0,1,2,))
        O_K_inv=np.linalg.inv(O_K)
        ham_K = O_K_inv @ energy_K @ O_K_inv
        ham_R = np.fft.ifftn(ham_K,axes=(0,1,2,))
        self.model.setHamR(ham_R)
        R1Axis=np.arange(self.model.energy_R.shape[0])-self.model.energy_R.shape[0]//2
        R2Axis=np.arange(self.model.energy_R.shape[1])-self.model.energy_R.shape[1]//2
        R3Axis=np.arange(self.model.energy_R.shape[2])-self.model.energy_R.shape[2]//2
        R1Grid,R2Grid,R3Grid=np.meshgrid(R1Axis,R2Axis,R3Axis)
        dRList=np.array([R1Grid.reshape(-1),R2Grid.reshape(-1),R3Grid.reshape(-1)]).T
        #self.model.setRList(dRList)
        showElapsedTime(extraInfo=f"Ab inito lattice Ham. calculated")
        #'''

        energy_R_intepolated=np.zeros(self.kgrid+(self.model.norbs,self.model.norbs),dtype=complex)
        O_R_intepolated=np.zeros(self.kgrid+(self.model.norbs,self.model.norbs),dtype=complex)
        for dR in self.model.dRList:
            energy_R_intepolated[dR[0],dR[1],dR[2]]=self.model.energy_R[dR[0],dR[1],dR[2]]
            O_R_intepolated[dR[0],dR[1],dR[2]]=self.model.overlap_R[dR[0],dR[1],dR[2]]
        energy_K=np.fft.fftn(energy_R_intepolated,axes=(0,1,2,))
        self.O_K=np.fft.fftn(O_R_intepolated,axes=(0,1,2,))
        showElapsedTime(extraInfo=f"FFT energy_R to energy_K and O_R to O_K on {self.kgrid=} finished")

        overlapEigValues, Wk =solveOverlap(self.O_K)
        assert overlapEigValues.min() > 1e-6, f"eigen-value of overlap matrix is too small or negative {overlapEigValues.min()=}"
        print(f"{overlapEigValues.min()=}")

        WkLam=Wk/np.sqrt(overlapEigValues[...,None,:])
        showElapsedTime(extraInfo=f"WkLam")
        EnergyMatrixInK_tilde=WkLam.conj().transpose((0,1,2,4,3)) @ energy_K @ WkLam
        showElapsedTime(extraInfo=f"Overlapped energy")
        self.energies, vecsBeforeOverlap = np.linalg.eigh(EnergyMatrixInK_tilde)
        showElapsedTime(extraInfo=f"eigen-eqs.")
        self.vectors = WkLam @ vecsBeforeOverlap
        showElapsedTime(extraInfo=f"overlapping on eig.-vecs.")
        self.vectors_inv = np.linalg.inv(self.vectors)
        showElapsedTime(extraInfo=f"general (Energy gauge) eig.-eqs. solved with {self.energies.min()=} {self.energies.max()=}")
        #print("Eigen-energy at Gamma:")
        #for ieig, eig in enumerate(self.energies[0,0,0,:]): print(f"{ieig:5d} {eig/27.2113845:15.6f}")
        #print()
        #exit()

        '''
        # 第二种方案，在插值后的（大得多的）格子上设置哈密顿系数矩阵
        O_K_inv=np.linalg.inv(self.O_K)
        ham_K = O_K_inv @ energy_K @ O_K_inv
        ham_R = np.fft.ifftn(ham_K,axes=(0,1,2,))
        self.model.setHamR(ham_R)
        R1Axis=np.arange(self.kgrid[0])-self.kgrid[0]//2
        R2Axis=np.arange(self.kgrid[1])-self.kgrid[1]//2
        R3Axis=np.arange(self.kgrid[2])-self.kgrid[2]//2
        R1Grid,R2Grid,R3Grid=np.meshgrid(R1Axis,R2Axis,R3Axis)
        dRList=np.array([R1Grid.reshape(-1),R2Grid.reshape(-1),R3Grid.reshape(-1)]).T
        self.model.setRList(np.array(dRList))
        showElapsedTime(extraInfo=f"Large lattice Ham. calculated")
        #'''

        '''
        # 第三种方案，直接用能量矩阵
        self.model.setHamR(self.model.energy_R)
        #self.vectors_inv=self.vectors.conj().transpose((0,1,2,4,3))
        #'''

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
        self.deltaE = self.energies[:,:,:,:,None] - self.energies[:,:,:,None,:]
        self.deltaEInverse=1./(self.deltaE - 1j*self.eta)
        self.deltaEInverse[abs(self.deltaE)<self.degen]=0
        if self.smr_type == 'Lorentzian':
            return
        raise NotImplementedError()

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
        self.RHam_R_intepolated=np.zeros(self.kgrid+(3,model.norbs,model.norbs),dtype=complex)
        
        if model.overlap:
            for dR in model.dRList:
                self.RHam_R_intepolated[dR[0],dR[1],dR[2]]=model.momentum_R[dR[0],dR[1],dR[2]]
            self.RHam_K=-1j*np.fft.fftn(self.RHam_R_intepolated,axes=(0,1,2,))
            if verbose==2: showElapsedTime(extraInfo=f"FFT RHam_R to RHam_K to calc. velocity in ao-basis finished")
            self.O_K_inv=np.linalg.inv(self.O_K)[...,None,:,:]
            vectors_inv=self.vectors_inv[...,None,:,:]
            self.velocity_K=vectors_inv @ (self.O_K_inv@self.RHam_K@self.O_K_inv) @ vectors_inv.conj().transpose((0,1,2,3,5,4))
        else:
            if not self.phase:
                RH_R_W=model.RHam_R
            else:
                RH_R_W=model.RHam_R*phaseshift[:,None,None,None]
            for iR, Rvec in enumerate(model.iRvec):
                self.RHam_R_intepolated[tuple(Rvec)]=RH_R_W[iR]
            self.RHam_K= -1j*np.fft.ifftn(self.RHam_R_intepolated,axes=(0,1,2,))*np.prod(self.kgrid)
            if verbose==2: showElapsedTime(extraInfo=f"FFT RHam_R to RHam_K to calc. velocity in ao-basis finished")
            vectors=self.vectors[:,:,:,None,:,:]
            self.velocity_K=vectors.conj().transpose((0,1,2,3,5,4)) @ self.RHam_K @ vectors
        
        if verbose==2: showElapsedTime(extraInfo=f"velocity in eig.-basis finished")

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
        
        self.BerryCurv=np.zeros(self.kgrid+(3,model.norbs))
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
            occupy=self.energies<fermi
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
        if model.overlap:
            vectors_inv=self.vectors_inv[...,None,:,:]
            spin_eig=vectors_inv @ self.O_K_inv @model.spin[None,None,None,:,:,:] @ vectors_inv.conj().transpose((0,1,2,3,5,4))
            #vectors=self.vectors[:,:,:,None,:,:]
            #spin_eig=vectors.conj().transpose((0,1,2,3,5,4)) @ model.spin[None,None,None,:,:,:] @ vectors
        else:
            vectors=self.vectors[:,:,:,None,:,:]
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

        self.spinBerryCurv=np.zeros(self.kgrid+(3,3,3,model.norbs))
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

        #spinBerry=self.spinBerryCurv[...,0,1,2,:]
        for ifermi, fermi in enumerate(fermiList):
            occupy=self.energies<fermi
            #print(f"{ifermi=:3d}  {len(self.energies[occupy])/np.prod(self.kgrid):8.4f}")
            #sigma[ifermi]=np.sum(spinBerry[occupy])
            for ialpha in range(3):
                for ibeta in range(3):
                    for igamma in range(3):
                        sigma[ifermi,ialpha,ibeta,igamma]=np.sum(self.spinBerryCurv[...,ialpha,ibeta,igamma,:][occupy])
            #exit()
        
        self.sigma=sigma/np.prod(self.kgrid)*(elementary_charge**2/hbar/angstrom/2)/model.Vcell
        if verbose==2: showElapsedTime(extraInfo=f"Spin Hall conductance finished")
