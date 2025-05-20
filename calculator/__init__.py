#
# The module is for calculations on energy, berry curv., and several types of conductances
#
from system import System
import numpy as np
from auxiliary import showElapsedTime, print_matrix, projectionOperator
from scipy.constants import elementary_charge, hbar, angstrom
from scipy.constants import k as kB

class Calculator():
    Iodd = False  # inversion gives minus
    Todd = False  # time reversal gives minus

    def __init__(self,**kwarg) -> None:
        '''
        Calculations on gamma-centered kmesh on BZ

        Parameters:
        -----------
        - model : System
        The basis system, involving the hopping, basis coordinates, lattice matrix and perharps the overlap
        - kgrid : tuple(3, dtype=int)
        Size of kmesh
        - eta   : float
        The relaxation energy, in the same unit with energy, usually eV
        - smr_type : str, "Lorentzian"|"Gaussian"
        - degen : float
        The threthold two states are treated to be degenerated
        - skip_low : bool
        To skip lowest and seperated eigen-states IN ALL CASES

        - phaseshift : np.ndarray(self.model.nRvec, dtype=complex)
        Extra phase for the Bloch states, used for hybrid FFT algorithm. Default: 1, no extra phase shift
        '''
        defaultSettings={
            "model":None,
            "kgrid":(1,1,1),
            "eta":0.0,
            "smr_type":"Lorentzian",
            "degen":1e-6,
            "skip_low":False,

            "phaseshift":1,

            "verbose":0,
        }
        defaultSettings.update(kwarg)
        self.model:System=defaultSettings["model"]
        self.kgrid:tuple=defaultSettings["kgrid"]
        assert type(self.kgrid)==tuple, f"expect tuple type of kgrid but {type(self.kgrid)=}"
        self.eta=defaultSettings["eta"]
        self.smr_type=defaultSettings["smr_type"]
        self.degen=defaultSettings["degen"]
        self.skip_low=defaultSettings["skip_low"]

        self.phaseshift=defaultSettings["phaseshift"]
        self.phase=False
        if type(self.phaseshift)==np.ndarray: self.phase=True

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
        #self.deltaE = self.energies[:,:,:,self.ibmin:,None] - self.energies[:,:,:,None,self.ibmin:]
        self.deltaE = self.energies[:,:,:,:,None] - self.energies[:,:,:,None,:]
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
        self.ibmin=len(self.energies[(0,0,0)][self.energies[(0,0,0)]<mid_valence])
        self.energies=self.energies[:,:,:,self.ibmin:]
        self.vectors=self.vectors[:,:,:,:,self.ibmin:]

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
    def __init__(self, spinProjectionList=[], **kwarg):
        '''
        Compute spin matrix in eigen-basis

        Indexing:
        self.spin_K[ik1, ik2, ik3, idirection, ieig, jeig] gives spin matrix in eigen-basis.
        idirection can be 0 | 1 | 2, corresponding to sigma_x | sigma_y | sigma_z.
        '''
        super().__init__(**kwarg)
        self.Todd = True

        vectors=self.vectors[:,:,:,None,:,:]
        if self.model.overlap:
            self.spin_K=vectors.conj().transpose((0,1,2,3,5,4)) @ self.model.spin[None,None,None,:,:,:] @ self.O_K_AO[...,None,:,:] @ vectors
        else:
            self.spin_K=vectors.conj().transpose((0,1,2,3,5,4)) @ self.model.spin[None,None,None,:,:,:] @ vectors
        if self.verbose==2: showElapsedTime(extraInfo=f"spin op. in eig.-basis calculated")

        self.projected_spin_K_list=[]
        for spinProjection in spinProjectionList:
            proj1=projectionOperator(spinProjection['source'],self.model.atomicOrbInfoList,self.model.norbs)
            proj2=projectionOperator(spinProjection['target'],self.model.atomicOrbInfoList,self.model.norbs)
            spin_proj=(proj1@self.model.spin@proj2 + proj2@self.model.spin@proj1)/2
            if self.model.overlap:
                projected_spin_K=vectors.conj().transpose((0,1,2,3,5,4)) @ spin_proj[None,None,None,:,:,:] @ self.O_K_AO[...,None,:,:] @ vectors
            else:
                projected_spin_K=vectors.conj().transpose((0,1,2,3,5,4)) @ spin_proj[None,None,None,:,:,:] @ vectors
            self.projected_spin_K_list.append(projected_spin_K)

class Spin_diag(Energy):
    def __init__(self, spinProjectionList=[], **kwarg):
        '''
        Compute spin expectation of eigen-basis

        Indexing:
        self.spin_diag_K[ik1, ik2, ik3, idirection, ieig] gives spin expectation of eigen-basis.
        idirection can be 0 | 1 | 2, corresponding to sigma_x | sigma_y | sigma_z.
        '''
        super().__init__(**kwarg)
        self.Todd = True

        vectors=self.vectors[:,:,:,None,:,:]
        if self.model.overlap:
            _sv=self.model.spin[None,None,None,:,:,:] @ self.O_K_AO[...,None,:,:] @ vectors
            self.spin_diag_K=np.sum(vectors.conj() * _sv, axis=(4,))
        else:
            _sv=self.model.spin[None,None,None,:,:,:] @ vectors
            self.spin_diag_K=np.sum(vectors.conj() * _sv, axis=(4,))
        if self.verbose==2: showElapsedTime(extraInfo=f"<spin> of eig.-basis calculated")

        self.projected_spin_diag_K_list=[]
        for spinProjection in spinProjectionList:
            proj1=projectionOperator(spinProjection['source'],self.model.atomicOrbInfoList,self.model.norbs)
            proj2=projectionOperator(spinProjection['target'],self.model.atomicOrbInfoList,self.model.norbs)
            spin_proj=(proj1@self.model.spin@proj2 + proj2@self.model.spin@proj1)/2
            if self.model.overlap:
                _sv=spin_proj[None,None,None,:,:,:] @ self.O_K_AO[...,None,:,:] @ vectors
                projected_spin_diag_K=np.sum(vectors.conj() * _sv, axis=(4,))
            else:
                _sv=spin_proj[None,None,None,:,:,:] @ vectors
                projected_spin_diag_K=np.sum(vectors.conj() * _sv, axis=(4,))
            self.projected_spin_diag_K_list.append(projected_spin_diag_K)

class Velocity(Energy):
    def __init__(self, velocityProjectionList=[], **kwarg) -> None:
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
        super().__init__(**kwarg)
        self.Todd = True
        self.Iodd = True
        
        RH_R_W=self.model.RHam_R
        self.RHam_R_intepolated=np.zeros(self.kgrid+(3,self.model.norbs,self.model.norbs),dtype=complex)
        r_R=self.model.AA_R_W
        r_R_intepolated=np.zeros(self.kgrid+(3,self.model.norbs,self.model.norbs),dtype=complex)
        if self.phase:
            RH_R_W=RH_R_W*self.phaseshift[:,None,None,None]
            r_R=r_R*self.phaseshift[:,None,None,None]
        for iR, Rvec in enumerate(self.model.iRvec):
            self.RHam_R_intepolated[tuple(Rvec)]=RH_R_W[iR]
            r_R_intepolated[tuple(Rvec)]=r_R[iR]
        self.RHam_K= 1j*np.fft.ifftn(self.RHam_R_intepolated,axes=(0,1,2,))*np.prod(self.kgrid)
        if self.verbose==2: showElapsedTime(extraInfo=f"FFT RHam_R to RHam_K to calc. velocity in ao-basis finished")
        vectors=self.vectors[:,:,:,None,:,:]
        self.velocity_K=vectors.conj().transpose((0,1,2,3,5,4)) @ self.RHam_K @ vectors

        r_K        =  1j*np.fft.ifftn(r_R_intepolated,axes=(0,1,2,))*np.prod(self.kgrid)
        r_K_H = (vectors.conj().transpose((0,1,2,3,5,4)) @ r_K @ vectors) * self.energies[...,None,None,:]
        self.velocity_K += -1j*(r_K_H - r_K_H.conj().transpose((0,1,2,3,5,4)))
        if self.verbose==2: showElapsedTime(extraInfo=f"velocity modification with position operators finished")
        if self.verbose==2: showElapsedTime(extraInfo=f"velocity in eig.-basis finished")
        # check hermity
        #dconj=np.sum(abs(self.velocity_K.transpose((0,1,2,3,5,4)).conj()-self.velocity_K))
        #assert dconj<1e-4, f"Hermity of velocity is not satisfied {dconj=}"

        # extra modifications to velocity due to orbital overlaps
        if self.model.overlap:
            RO_R_W=self.model.RO_R
            self.RO_R_intepolated=np.zeros(self.kgrid+(3,self.model.norbs,self.model.norbs),dtype=complex)
            if self.phase:
                RO_R_W=RO_R_W*self.phaseshift[:,None,None,None]
            for iR, Rvec in enumerate(self.model.iRvec):
                self.RO_R_intepolated[tuple(Rvec)]=RO_R_W[iR]
            self.RO_K= 1j*np.fft.ifftn(self.RO_R_intepolated,axes=(0,1,2,))*np.prod(self.kgrid)
            #self.velocity_K -= vectors.conj().transpose((0,1,2,3,5,4)) @ self.RO_K @ (vectors *self.energies[...,None,None,self.ibmin:])
            __UdOkU = (vectors.conj().transpose((0,1,2,3,5,4)) @ self.RO_K @ vectors)*self.energies[...,None,:,None]
            self.velocity_K -= (__UdOkU + __UdOkU.conj().transpose((0,1,2,3,5,4)))/2

        self.projected_velocity_K_list=[]
        for velocityProjection in velocityProjectionList:
            proj1=projectionOperator(velocityProjection['source'],self.model.atomicOrbInfoList,self.model.norbs)
            proj2=projectionOperator(velocityProjection['target'],self.model.atomicOrbInfoList,self.model.norbs)
            proj1=proj1[None,None,None,None,:,:]
            proj2=proj2[None,None,None,None,:,:]
            RHam_K_proj=(proj1@self.RHam_K@proj2 + proj2@self.RHam_K@proj1)/2
            velocity_K_proj=vectors.conj().transpose((0,1,2,3,5,4)) @ RHam_K_proj @ vectors

            r_K_proj=(proj1@r_K@proj2 + proj2@r_K@proj1)/2
            r_K_H_proj = (vectors.conj().transpose((0,1,2,3,5,4)) @ r_K_proj @ vectors) * self.energies[...,None,None,:]
            velocity_K_proj += -1j*(r_K_H_proj - r_K_H_proj.conj().transpose((0,1,2,3,5,4)))

            if self.model.overlap:
                RO_K_proj=(proj1@self.RO_K@proj2 + proj2@self.RO_K@proj1)/2
                __UdOkU = (vectors.conj().transpose((0,1,2,3,5,4)) @ RO_K_proj @ vectors)*self.energies[...,None,:,None]
                velocity_K_proj -= (__UdOkU + __UdOkU.conj().transpose((0,1,2,3,5,4)))/2
            self.projected_velocity_K_list.append(velocity_K_proj)

class Velocity_diag(Energy):
    def __init__(self, velocityProjectionList=[], **kwarg) -> None:
        '''
        Compute group velocity in original basis and eigen-basis

        Indexing:
        self.RHam_K[ik1, ik2, ik3, idirection, iorb, jorb] gives velocity matrix in original basis,
        idirection can be 0 | 1 | 2 corresponding to x | y | z direction.

        self.velocity_diag_K[ik1, ik2, ik3, idirection, ieig] gives group velocity of each eigen-basis.
        idirection can be 0 | 1 | 2 corresponding to x | y | z direction.
        '''
        super().__init__(**kwarg)
        self.Todd = True
        self.Iodd = True
        
        RH_R_W=self.model.RHam_R
        self.RHam_R_intepolated=np.zeros(self.kgrid+(3,self.model.norbs,self.model.norbs),dtype=complex)
        if self.phase:
            RH_R_W=RH_R_W*self.phaseshift[:,None,None,None]
        for iR, Rvec in enumerate(self.model.iRvec):
            self.RHam_R_intepolated[tuple(Rvec)]=RH_R_W[iR]
        self.RHam_K= 1j*np.fft.ifftn(self.RHam_R_intepolated,axes=(0,1,2,))*np.prod(self.kgrid)

        if self.verbose==2: showElapsedTime(extraInfo=f"FFT RHam_R to RHam_K to calc. velocity in ao-basis finished")
        vectors=self.vectors[:,:,:,None,:,:]
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
            _RO_K_vector=self.RO_K @ (vectors *self.energies[...,None,None,:])
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

class BerryCurv(Velocity):
    
    def __init__(self, projection=None, **kwarg) -> None:
        '''
        Calc. Berry curvature \Omega^{alpha}_{k,mn}.

        alpha can be x, y, z corresponds the (cart.) space direction, k denote the crystal moment, m, n index the bands.

        Indexing: BerryCurv[ik1, ik2, ik3, idrection, iband, jband]
        '''
        super().__init__(projection=projection, **kwarg)
        self.Todd = True

        self.calcDeltaEInverse()
        dEInverse2 = self.deltaEInverse**2

        velocity_alpha_K=self.velocity_K
        velocity_beta_K=self.velocity_K
        if projection!=None:
            velocity_alpha_K=self.velocity_K_proj
        
        self.BerryCurv=np.zeros(self.kgrid+(3,3,self.model.norbs-self.ibmin))
        for ialpha in range(3):
            for ibeta in range(3):
                self.BerryCurv[...,ialpha,ibeta,:]=-2*np.diagonal(velocity_beta_K[...,ibeta,:,:] @ (dEInverse2*velocity_alpha_K[...,ialpha,:,:]),0,-2,-1).imag
        if self.verbose==2: showElapsedTime(extraInfo=f"Spin Berry curvature finished")

class AnormalousHallConductance(BerryCurv):

    def __init__(self, fermiList=[], projection=None, **kwarg) -> None:
        super().__init__(projection=projection, **kwarg)
        self.Todd = True

        sigma=np.zeros((len(fermiList),3,3))

        for ifermi, fermi in enumerate(fermiList):
            #occupy=self.energies[:,:,:,self.ibmin:]<fermi
            occupy=self.energies<fermi
            for ialpha in range(3):
                for ibeta in range(3):
                    sigma[ifermi,ialpha,ibeta]=np.sum(self.BerryCurv[...,ialpha,ibeta,:][occupy])
        
        self.sigma=sigma/np.prod(self.kgrid)*(elementary_charge**2/hbar/angstrom/2)/self.model.Vcell
        if self.verbose==2: showElapsedTime(extraInfo=f"Anormalous Hall conductance finished")

class SpinVelocity(Velocity, Spin):
    def __init__(self, velocityProjectionList=[], spinvelocityProjectionList=[], **kwarg) -> None:
        '''
        Compute spin velocity matrix in original basis and eigen-basis
        and projected spin velocity matrix.
        Note, the projection is on the velocity component, corresponding to the spin-currents running in specific basis.

        Indexing:
        self.SRHam_K[ik1, ik2, ik3, idirection, igamma, iorb, jorb] gives spin velocity matrix in original basis,
        idirection can be 0 | 1 | 2, corresponding to x | y | z direction. 
        igamma can be 0 | 1 | 2 | 3, corresponding to sigma^x | sigma^y | sigma^z | sigma^0 spin-polarization. 

        self.spinVelocity_K[ik1, ik2, ik3, idirection, igamma, ieig, jeig] gives spin velocity matrix in eigen-basis.
        idirection can be 0 | 1 | 2, corresponding to x | y | z direction. 
        igamma can be 0 | 1 | 2 | 3, corresponding to sigma^x | sigma^y | sigma^z | sigma^0 spin-polarization. 

        if spinvelocityProjctionList is not empty
        self.spinVelocity_K_proj[iproj] gives each projected spin velocity matrix in eigen-basis.
        '''
        super().__init__(velocityProjectionList=velocityProjectionList+spinvelocityProjectionList, **kwarg)
        self.Iodd = True
        
        __sV=self.spin_K[...,None,:,:,:] @ self.velocity_K[...,None,:,:]
        self.spinVelocity_K = (__sV + __sV.transpose((0,1,2,3,4,6,5)).conj())/2

        self.projected_spinVelocity_K_list=[]
        for iproj, proj in enumerate(spinvelocityProjectionList):
            projected_velocity_K=self.projected_velocity_K_list.pop(iproj+len(velocityProjectionList))
            __sV=self.spin_K[...,None,:,:,:] @ projected_velocity_K[...,None,:,:]
            self.projected_spinVelocity_K_list.append((__sV + __sV.transpose((0,1,2,3,4,6,5)).conj())/2)
        
        if self.verbose==2: showElapsedTime(extraInfo=f"spin velocity in eig.-basis finished")

class SpinVelocity_diag(Velocity_diag, Spin_diag):
    def __init__(self, velocityProjectionList=[], spinProjectionList=[], **kwarg) -> None:
        '''
        Compute spin velocity matrix in original basis and eigen-basis
        and projected spin velocity matrix.
        Note, the projection is on the velocity component, corresponding to the spin-currents running in specific basis.

        Indexing:
        self.spinVelocity_diag_K[ik1, ik2, ik3, idirection, igamma, ieig] gives spin velocity expectation of eigen-basis.
        idirection can be 0 | 1 | 2, corresponding to x | y | z direction. 
        igamma can be 0 | 1 | 2 | 3, corresponding to sigma^x | sigma^y | sigma^z | sigma^0 spin-polarization. 

        if spinvelocityProjctionList is not empty
        self.spinVelocity_K_proj[iproj] gives each projected spin velocity matrix in eigen-basis.
        '''
        super().__init__(velocityProjectionList=velocityProjectionList, spinProjectionList=spinProjectionList, **kwarg)
        self.Iodd = True
        
        self.spinVelocity_diag_K=self.spin_diag_K[...,None,:,:] * self.velocity_diag_K[...,None,:]

        self.projected_spinVelocity_K_list=[]
        if self.verbose==2: showElapsedTime(extraInfo=f"spin velocity in eig.-basis finished")

class SpinBerryCurv(SpinVelocity):
    def __init__(self, velocityProjectionList=[], spinvelocityProjectionList=[], **kwarg) -> None:
        '''
        Calc. spin Berry curvature \Omega^{alpha,gamma}_{k,mn}.

        alpha can be x, y, z, denoting the (cart.) space direction. 
        gammma can be sigma^x, sigma^y, sigma^z, sigma^0, corresponding to the spin-polarization. 
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
                            self.projected_spinBerryCurv[...,ialpha,ibeta,igamma,:]=-2*np.diagonal(proj_spinVelocity_K[...,ibeta,igamma,:,:] @ (dEInverse2*proj_velocity_K[...,ialpha,:,:]),0,-2,-1).imag
        
        self.spinBerryCurv=self.projected_spinBerryCurv[0,0]
        if self.verbose==2: showElapsedTime(extraInfo=f"Spin Berry curvature finished")

class SpinVeloVelo(SpinVelocity):
    def __init__(self, model: System, kgrid=(1, 1, 1), phaseshift=1, verbose=0, **kwarg) -> None:
        '''
        Calc. correlation of spin velocity and velocity at each k-point \Omega^{alpha,beta,gamma}_{k,m}.

        alpha, beta can be x, y, z, denoting the (cart.) space direction, corresponding to the direction of velocity and spin velocity
        gammma can be sigma^x, sigma^y, sigma^z, sigma^0, corresponding to the spin-polarization. 
        k denotes the crystal moment, m indexes the bands.

        Indexing: spinVeloVelo[ik1, ik2, ik3, ialpha, ibeta, igamma, ieig]
        '''
        super().__init__(model, kgrid, phaseshift, verbose, **kwarg)

        self.spinVeloVelo=np.zeros(self.kgrid+(3,3,3,model.norbs-self.ibmin))
        for ialpha in range(3):
            for ibeta in range(3):
                for igamma in range(3):
                    self.spinVeloVelo[...,ialpha,ibeta,igamma,:]=(np.diagonal(self.spinVelocity_K[...,ibeta,igamma,:,:],0,-2,-1) * np.diagonal(self.velocity_K[...,ialpha,:,:],0,-2,-1)).real
                    #self.spinVeloVelo[...,ialpha,ibeta,igamma,:]=(np.diagonal(self.velocity_K[...,ibeta,:,:],0,-2,-1) * np.diagonal(self.velocity_K[...,ialpha,:,:],0,-2,-1)).real
        if verbose==2: showElapsedTime(extraInfo=f"Spin Berry curvature finished")

class Spin_x_Velocity(Velocity):
    def __init__(self, model: System, kgrid=(1, 1, 1), phaseshift=1, eta=0.005, smr_type="Lorentzian", degen=1e-6, verbose=0, **kwarg) -> None:
        '''
        Calc. spin x velocity correlation at each k-point: S_alpha^gamma_k = \Sigma_n s^{\gamma}_mn v_{\alpha,nm} / (E_n - E_m +i\eta)**2 

        alpha can be x, y, z, denoting the (cart.) space direction. 
        gammma can be sigma^x, sigma^y, sigma^z, sigma^0, corresponding to the spin-polarization. 
        k denotes the crystal moment, m, n index the bands.

        Indexing: spin_x_velocity[ik1, ik2, ik3, ialpha, igamma, ieig]
        '''
        super().__init__(model, kgrid, phaseshift, verbose, **kwarg)
        self.Todd = True
        self.Iodd = True

        self.degen=degen # update the degeneracy threthold
        self.eta=eta
        self.smr_type=smr_type
        self.calcDeltaEInverse()
        dEInverse2 = self.deltaEInverse**2
        
        #vectors=self.vectors[:,:,:,None,:,self.ibmin:]
        vectors=self.vectors[:,:,:,None,:,:]
        if model.overlap:
            spin_eig=vectors.conj().transpose((0,1,2,3,5,4)) @ model.spin[None,None,None,:,:,:] @ self.O_K_AO[...,None,:,:] @ vectors
        else:
            spin_eig=vectors.conj().transpose((0,1,2,3,5,4)) @ model.spin[None,None,None,:,:,:] @ vectors
        if verbose==2: showElapsedTime(extraInfo=f"spin op. in eig.-basis calculated")

        self.spin_x_velocity=np.zeros(self.kgrid+(3,3,model.norbs-self.ibmin))
        for ialpha in range(3):
            for igamma in range(3):
                self.spin_x_velocity[...,ialpha,igamma,:]=-2*np.diagonal(spin_eig[...,igamma,:,:] @ (dEInverse2*self.velocity_K[...,ialpha,:,:]),0,-2,-1).imag

        if verbose==2: showElapsedTime(extraInfo=f"Spin_x_Velocity in eig.-basis finished")

class SpinHallConductance(SpinBerryCurv):
    def __init__(self, fermiList=[], currentProjectionList=[], spincurrentProjectionList=[], **kwarg) -> None:
        '''
        Spin hall conductance in (S/m hbar/2e)
        '''
        super().__init__(velocityProjectionList=currentProjectionList, spinvelocityProjectionList=spincurrentProjectionList, **kwarg)

        sigma=np.zeros((len(fermiList),3,3,3))

        for ifermi, fermi in enumerate(fermiList):
            occupy=self.energies<fermi
            for ialpha in range(3):
                for ibeta in range(3):
                    for igamma in range(3):
                        sigma[ifermi,ialpha,ibeta,igamma]=np.sum(self.spinBerryCurv[...,ialpha,ibeta,igamma,:][occupy])
            #exit()
        
        self.sigma=sigma/np.prod(self.kgrid)*(elementary_charge**2/hbar/angstrom/2)/self.model.Vcell
        #self.sigma=sigma*(2*np.pi/(np.linalg.det(self.model.lattice[:2,:2])*np.prod(self.kgrid[:2]))) # for checking the TKNN SHC of 2D topo./trivial sys.
        if self.verbose==2: showElapsedTime(extraInfo=f"Spin Hall conductance finished")

class SpinGalvanic_diag(Velocity_diag, Spin_diag):
    def __init__(self, fermiList=[], currentProjectionList=[], spinProjectionList=[], **kwarg) -> None:
        '''
        Spin galvanic (\hbar/2 per cell per V/m)
        '''
        super().__init__(velocityProjectionList=currentProjectionList, spinProjectionList=spinProjectionList, **kwarg)
        # search energy window
        Emax, Emin=np.max(fermiList)+20*self.eta, np.min(fermiList)-20*self.eta
        condition_energyInRange=np.sum((self.energies>Emin) * (self.energies<Emax),axis=(0,1,2,))
        ibmin, ibmax= 0, self.energies.shape[-1]-1
        while(condition_energyInRange[ibmin]<0.5):
            ibmin+=1
        while(condition_energyInRange[ibmax]<0.5):
            ibmax-=1
        if self.verbose==2: showElapsedTime(extraInfo=f"Band index window: {ibmin=} {ibmax=}")
        #
        energies=self.energies[:,:,:,ibmin:ibmax+1]
        
        # <n|spin|n> * <n|velocity|n>, only intra-band contribution
        spin_diag_K_list=[self.spin_diag_K]+self.projected_spin_diag_K_list
        velocity_diag_K_list=[self.velocity_diag_K]+self.projected_velocity_diag_K_list
        spinVelocity_diag_K  = spin_diag_K_list[-1][...,None,:,ibmin:ibmax+1] * velocity_diag_K_list[-1][...,None,ibmin:ibmax+1]

        sigma=np.zeros((len(fermiList),3,3))
        self.spinGalvanicK=np.zeros((len(fermiList),)+self.kgrid+(3,3),dtype=float)
        for ifermi, fermi in enumerate(fermiList):
            exp=np.exp((energies-fermi)/self.eta)
            weight=(-exp/(exp+1)**2/self.eta)
            self.spinGalvanicK[ifermi] += np.sum(spinVelocity_diag_K.real * weight[...,None,None,:],axis=(5,))
            sigma[ifermi]=np.sum(self.spinGalvanicK[ifermi],axis=(0,1,2))
        
        self.sigma=sigma/np.prod(self.kgrid)*(elementary_charge/hbar*angstrom)
        #self.sigma=sigma*(2*np.pi/(np.linalg.det(self.model.lattice[:2,:2])*np.prod(self.kgrid[:2]))) # for checking the TKNN SHC of 2D topo./trivial sys.
        if self.verbose==2: showElapsedTime(extraInfo=f"Spin Hall conductance finished")

class SpinHallConductanceRashiba(SpinVeloVelo):
    def __init__(self, model: System, kgrid=(1, 1, 1), phaseshift=1, T=300, fermiList=[0.,], verbose=0, **kwarg) -> None:
        '''
        Rashiba-Elderson effect induced Spin hall conductance in (S/m hbar/2e)
        '''
        super().__init__(model, kgrid, phaseshift, verbose, **kwarg)

        sigma=np.zeros((len(fermiList),3,3,3))

        for ifermi, fermi in enumerate(fermiList):
            #condition=abs(self.energies[:,:,:,self.ibmin:]-fermi)<10*(kB*T/elementary_charge)
            #exp=np.exp((self.energies[:,:,:,self.ibmin:][condition]-fermi)/(kB*T/elementary_charge))
            condition=abs(self.energies-fermi)<10*(kB*T/elementary_charge)
            exp=np.exp((self.energies[condition]-fermi)/(kB*T/elementary_charge))
            f_E=-1/(kB/elementary_charge*T)*exp/(1+exp)**2
            for ialpha in range(3):
                for ibeta in range(3):
                    for igamma in range(3):
                        sigma[ifermi,ialpha,ibeta,igamma]=np.sum(self.spinVeloVelo[...,ialpha,ibeta,igamma,:][condition]*f_E)
        
        self.sigma=sigma/np.prod(self.kgrid)*(elementary_charge**3/hbar**2/angstrom/2)/model.Vcell
        if verbose==2: showElapsedTime(extraInfo=f"Rashiba spin Hall conductance finished")

from .Path import EnergyOnPath, VelocityOnPath, BerryCurvOnPath, SpinBerryCurvOnPath, SpinExpectationOnPath
