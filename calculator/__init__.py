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

        Indexing:
        self.energies[ik1, ik2, ik3, ieig] gives eigen-vals.

        self.vectors[ik1, ik2, ik3, ieig, jeig] gives eigen-vecs. in ao basis.
        '''
        super().__init__()
        self.kgrid=kgrid

        showElapsedTime(offset=True)
        self.ham_R_intepolated=np.zeros(self.kgrid+(model.norbs,model.norbs),dtype=complex)
        for dR in model.dRList:
            self.ham_R_intepolated[dR[0],dR[1],dR[2]]=model.ham_R[dR[0],dR[1],dR[2]]
        self.ham_K=np.fft.fftn(self.ham_R_intepolated,axes=(0,1,2,))
        showElapsedTime(extraInfo=f"FFT ham_R to ham_K on {self.kgrid=} finished")
        self.energies, self.vectors = np.linalg.eigh(self.ham_K)
        showElapsedTime(extraInfo=f"eig.-eqs. solved")
    
    def calcDeltaEInverse(self):
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
        vectors=self.vectors[:,:,:,None,:,:]
        self.velocity_K=vectors.conj().transpose((0,1,2,3,5,4)) @ self.RHam_K @ vectors
        showElapsedTime(extraInfo=f"velocity matrix finished")

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

        self.SRHam_K=np.zeros(self.kgrid+(3,4,model.norbs,model.norbs),dtype=complex)
        for igamma in range(4):
            s_op=model.spin[igamma][None,None,None,None,:,:]
            self.SRHam_K[...,igamma,:,:] = (s_op @ self.RHam_K + self.RHam_K @ s_op)/2

        vectors=self.vectors[:,:,:,None,None,:,:]
        self.spinVelocity_K=vectors.conj().transpose((0,1,2,3,4,6,5)) @ self.SRHam_K @ vectors
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
