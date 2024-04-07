#
# The module is for calculations on energy, berry curv., and several types of conductances
#
from system import System
import numpy as np
from auxiliary import showElapsedTime, print_matrix
from scipy.constants import elementary_charge, hbar, angstrom

class Calculator():

    def __init__(self,) -> None:
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
        print("initializing calculator module")
        self.model=None
        self.kgrid=(1,1,1)
        self.eta=0.
        self.smr_type="Lorentzian"
        self.degen=1e-6

#
# Implemented calculaors
# Energy, velocity (ao and eigen gauges), spin velocity (ao and eigen gauges), 
# Berry curvature, spin Berry curvature
#
class Energy(Calculator):
    def __init__(self, model:System, kgrid=(1,1,1)) -> None:
        '''
        Solve the eigen-eqs. to obtain eigen-vals. and vecs.

        Usually the second father of all calculators

        Indexing:
        self.energies[ik1, ik2, ik3, ieig] gives eigen-vals.

        self.vectors[ik1, ik2, ik3, ieig, jeig] gives eigen-vecs. in ao basis.
        '''
        super().__init__()
        self.kgrid=kgrid
        self.model=model

        showElapsedTime(offset=True)
        
        if model.overlap:
            self.solveGeneralEigenEquation()
        else:
            self.solveEigenEquation()

    def solveGeneralEigenEquation(self):
        '''
        Solve the general eig.-eqs. 

        H \dot U = O \dot U \dot \Gamma
        '''
        energy_K=np.fft.fftn(self.model.energy_R,axes=(0,1,2,))
        energy_K = (energy_K + energy_K.transpose((0,1,2,4,3)).conj())/2
        O_K=np.fft.fftn(self.model.overlap_R,axes=(0,1,2,))
        O_K = (O_K + O_K.transpose((0,1,2,4,3)).conj())/2
        
        energy_R = np.fft.ifftn(energy_K,axes=(0,1,2,))
        O_R = np.fft.ifftn(O_K,axes=(0,1,2,))
        self.model.setEnergyR(energy_R)
        self.model.setOR(O_R)
        showElapsedTime(extraInfo=f"The hemit condition is imposed")

        '''
        # 第一种方案，在能量矩阵的（小）格子上设置哈密顿系数矩阵
        O_K_inv=np.linalg.inv(O_K)
        ham_K = O_K_inv @ energy_K @ O_K_inv
        ham_R = np.fft.ifftn(ham_K,axes=(0,1,2,))
        self.model.setHamR(ham_R)
        '''

        self.energy_R_intepolated=np.zeros(self.kgrid+(self.model.norbs,self.model.norbs),dtype=complex)
        self.O_R_intepolated=np.zeros(self.kgrid+(self.model.norbs,self.model.norbs),dtype=complex)
        for dR in self.model.dRList:
            self.energy_R_intepolated[dR[0],dR[1],dR[2]]=self.model.energy_R[dR[0],dR[1],dR[2]]
            self.O_R_intepolated[dR[0],dR[1],dR[2]]=self.model.overlap_R[dR[0],dR[1],dR[2]]
        self.energy_K=np.fft.fftn(self.energy_R_intepolated,axes=(0,1,2,))
        self.O_K=np.fft.fftn(self.O_R_intepolated,axes=(0,1,2,))
        showElapsedTime(extraInfo=f"FFT energy_R to energy_K and O_R to O_K on {self.kgrid=} finished")

        overlapEigValues, Wk =np.linalg.eigh(self.O_K)
        assert overlapEigValues.min() > 1e-6, f"eigen-value of overlap matrix is too small or negative {overlapEigValues.min()=}"
        print(f"{overlapEigValues.min()=}")

        WkLam=Wk/np.sqrt(overlapEigValues[...,None,:])
        EnergyMatrixInK_tilde=WkLam.conj().transpose((0,1,2,4,3)) @ self.energy_K @ WkLam
        self.energies, vecsBeforeOverlap = np.linalg.eigh(EnergyMatrixInK_tilde)
        self.vectors = WkLam @ vecsBeforeOverlap
        self.vectors_inv = np.linalg.inv(self.vectors)
        showElapsedTime(extraInfo=f"general (Energy gauge) eig.-eqs. solved with {self.energies.min()=} {self.energies.max()=}")

        self.O_K_inv=np.linalg.inv(self.O_K)
        showElapsedTime(extraInfo=f"O_k^-1 finished")

        # 第二种方案，在插值后的（大得多的）格子上设置哈密顿系数矩阵
        ham_K = self.O_K_inv @ self.energy_K @ self.O_K_inv
        ham_R = np.fft.ifftn(ham_K,axes=(0,1,2,))
        self.model.setHamR(ham_R)
        R1Axis=np.arange(self.kgrid[0])-self.kgrid[0]//2
        R2Axis=np.arange(self.kgrid[1])-self.kgrid[1]//2
        R3Axis=np.arange(self.kgrid[2])-self.kgrid[2]//2
        R1Grid,R2Grid,R3Grid=np.meshgrid(R1Axis,R2Axis,R3Axis)
        dRList=np.array([R1Grid.reshape(-1),R2Grid.reshape(-1),R3Grid.reshape(-1)]).T
        self.model.setRList(np.array(dRList))
        showElapsedTime(extraInfo=f"Lattice Ham. finished")

    def solveEigenEquation(self):
        '''
        Solve the eig.-eqs. 

        H \dot U = U \dot \Gamma
        '''
        self.ham_R_intepolated=np.zeros(self.kgrid+(self.model.norbs,self.model.norbs),dtype=complex)
        for dR in self.model.dRList:
            self.ham_R_intepolated[dR[0],dR[1],dR[2]]=self.model.ham_R[dR[0],dR[1],dR[2]]
        self.ham_K=np.fft.fftn(self.ham_R_intepolated,axes=(0,1,2,))
        showElapsedTime(extraInfo=f"FFT ham_R to ham_K on {self.kgrid=} finished")
        self.energies, self.vectors = np.linalg.eigh(self.ham_K)
        showElapsedTime(extraInfo=f"eig.-eqs. solved")
    
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
    def __init__(self, model: System, kgrid=(1,1,1)) -> None:
        '''
        Compute velocity matrix in original basis and eigen-basis

        Indexing:
        self.RHam_K[ik1, ik2, ik3, idirection, iorb, jorb] gives velocity matrix in original basis,
        idirection can be 0 | 1 | 2 corresponding to x | y | z direction.

        self.velocity_K[ik1, ik2, ik3, idirection, ieig, jeig] gives velocity matrix in eigen-basis.
        idirection can be 0 | 1 | 2 corresponding to x | y | z direction.
        '''
        super().__init__(model, kgrid=kgrid)
        model.getRHam_R()
        self.RHam_R_intepolated=np.zeros(self.kgrid+(3,model.norbs,model.norbs),dtype=complex)
        for dR in model.dRList:
            self.RHam_R_intepolated[dR[0],dR[1],dR[2]]=model.RHam_R[dR[0],dR[1],dR[2]]
        self.RHam_K=-1j*np.fft.fftn(self.RHam_R_intepolated,axes=(0,1,2,))
        showElapsedTime(extraInfo=f"FFT RHam_R to RHam_K finished")
        if model.overlap:
            vectors_inv=self.vectors_inv[:,:,:,None,:,:]
            self.velocity_K=vectors_inv @ self.RHam_K @ vectors_inv.conj().transpose((0,1,2,3,5,4))
        else:
            vectors=self.vectors[:,:,:,None,:,:]
            self.velocity_K=vectors.conj().transpose((0,1,2,3,5,4)) @ self.RHam_K @ vectors
        showElapsedTime(extraInfo=f"velocity matrix finished")

        #print("Hk in AO gauge")
        #print_matrix(self.ham_K[10,0,0],(0,1,2,3,),(0,1,2,3,))
        #print("Uk in AO gauge")
        #print_matrix(self.vectors[10,0,0],(0,1,2,3,),(0,1,2,3,))
        #print("Uk^+ @ Hk @ Uk")
        #print_matrix(self.vectors[10,0,0].conj().T@self.ham_K[10,0,0]@self.vectors[10,0,0],(0,1,2,3,),(0,1,2,3,))
        #print("Vk in AO gauge")
        #print_matrix(self.RHam_K[10,0,0,0],(0,1,2,3,),(0,1,2,3,))
        #print("Vk in eig. gauge")
        #print_matrix(self.velocity_K[10,0,0,0],(0,1,2,3,),(0,1,2,3,))
        #exit()

class BerryCurv(Velocity):
    
    def __init__(self, model: System, kgrid=(1, 1, 1), eta=1e-4, smr_type="Lorentzian", degen=1e-6) -> None:
        '''
        Calc. Berry curvature \Omega^{alpha}_{k,mn}.

        alpha can be x, y, z corresponds the (cart.) space direction, k denote the crystal moment, m, n index the bands.

        Indexing: BerryCurv[ik1, ik2, ik3, idrection, iband, jband]
        '''
        super().__init__(model, kgrid=kgrid)
        self.eta=eta
        self.smr_type=smr_type
        self.degen=degen # update the degeneracy threthold
        self.calcDeltaEInverse()
        dEInverse2 = self.deltaEInverse**2
        
        self.BerryCurv=np.zeros(self.kgrid+(3,model.norbs))
        eps=np.arange(3)
        for ialpha in range(3):
            self.BerryCurv[...,ialpha,:]=-2*(self.velocity_K[...,eps[(ialpha+1)%3],:,:] @ (dEInverse2[:,:,:,:,:]*self.velocity_K[...,eps[(ialpha+2)%3],:,:])).imag
        showElapsedTime(extraInfo=f"Berry curvature finished")

class SpinVelocity(Velocity):
    def __init__(self, model: System, kgrid=(1,1,1)) -> None:
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
        super().__init__(model, kgrid=kgrid)
        model.getSpinMatrice()
        showElapsedTime(extraInfo=f"spin matrice constructed")
        if model.overlap:
            vectors_inv=self.vectors_inv[:,:,:,None,:,:]
            spin_eig=vectors_inv @ model.spin[None,None,None,:,:,:] @ vectors_inv.conj().transpose((0,1,2,3,5,4))
        else:
            vectors=self.vectors[:,:,:,None,:,:]
            spin_eig=vectors.conj().transpose((0,1,2,3,5,4)) @ model.spin[None,None,None,:,:,:] @ vectors
        showElapsedTime(extraInfo=f"spin op. in eig.-basis calculated")
        spin=spin_eig[...,None,:,:,:]
        velocity=self.velocity_K[...,None,:,:]
        self.spinVelocity_K = (spin @ velocity + velocity @ spin)/2
        #self.SRHam_K=np.zeros(self.kgrid+(3,4,model.norbs,model.norbs),dtype=complex)
        #for igamma in range(4):
        #    s_op=model.spin[igamma][None,None,None,None,:,:]
        #    if model.overlap:
        #        pass
        #        #self.SRHam_K[...,igamma,:,:] = (s_op @ self.O_K[...,None,:,:] @ self.RHam_K + self.RHam_K @ self.O_K[...,None,:,:] @ s_op)/2
        #    else:
        #        vectors=self.vectors[:,:,:,None,None,:,:]
        #        s_op_eig=vectors.conj().transpose((0,1,2,3,4,6,5)) @ s_op @ vectors
        #        self.SRHam_K[...,igamma,:,:] = (s_op @ self.RHam_K + self.RHam_K @ s_op)/2
        #    showElapsedTime(extraInfo=f"spin velociy {igamma=} in AO basis constructed")
        
        #if model.overlap:
        #    vectors_inv=self.vectors_inv[:,:,:,None,None,:,:]
        #    self.spinVelocity_K=vectors_inv @ self.SRHam_K @ vectors_inv.conj().transpose((0,1,2,3,4,6,5))
        #else:
            #vectors=self.vectors[:,:,:,None,None,:,:]
            #self.spinVelocity_K=vectors.conj().transpose((0,1,2,3,4,6,5)) @ self.SRHam_K @ vectors
        #    pass
        showElapsedTime(extraInfo=f"spin velocity finished")

class SpinBerryCurv(SpinVelocity):
    def __init__(self, model: System, kgrid=(1, 1, 1), eta=0.005, smr_type="Lorentzian", degen=1e-6) -> None:
        '''
        Calc. spin Berry curvature \Omega^{alpha,gamma}_{k,mn}.

        alpha can be x, y, z, denoting the (cart.) space direction. 
        gammma can be sigma^x, sigma^y, sigma^z, sigma^0, corresponding to the spin-polarization. 
        k denotes the crystal moment, m, n index the bands.

        Indexing: spinBerryCurv[ik1, ik2, ik3, ialpha, igamma, ieig]
        '''
        super().__init__(model, kgrid)
        self.degen=degen # update the degeneracy threthold
        self.eta=eta
        self.smr_type=smr_type
        self.calcDeltaEInverse()
        dEInverse2 = self.deltaEInverse**2

        self.spinBerryCurv=np.zeros(self.kgrid+(3,4,model.norbs))
        eps=np.arange(3)
        for ialpha in range(3):
            for igamma in range(4):
                self.spinBerryCurv[...,ialpha,igamma,:]=-2*np.diagonal(self.spinVelocity_K[...,eps[(ialpha+1)%3],igamma,:,:] @ (dEInverse2*self.velocity_K[...,eps[(ialpha+2)%3],:,:]),0,-2,-1).imag
        showElapsedTime(extraInfo=f"Spin Berry curvature finished")

class SpinHallConductance(SpinBerryCurv):
    def __init__(self, model: System, kgrid=(1, 1, 1), eta=0.005, smr_type="Lorentzian", degen=0.000001, fermiList=[0.,]) -> None:
        '''
        Spin hall conductance in (S/m hbar/2e)
        '''
        super().__init__(model, kgrid, eta, smr_type, degen)

        sigma=np.zeros(len(fermiList))

        spinBerry=self.spinBerryCurv[...,2,2,:]
        for ifermi, fermi in enumerate(fermiList):
            occupy=self.energies<fermi
            sigma[ifermi]=np.sum(spinBerry[occupy])
        
        self.sigma=sigma*(elementary_charge**2/hbar/angstrom/2)/np.prod(self.kgrid)/model.Vcell
        showElapsedTime(extraInfo=f"Spin Hall conductance finished")
