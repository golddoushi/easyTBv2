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
        self.deltaEInverse =np.zeros_like(self.deltaE, dtype=complex)
        self.deltaEInverse2=np.zeros_like(self.deltaE, dtype=complex)
        nondegeneratedCondition=abs(self.deltaE)>self.degen
        self.deltaEInverse[nondegeneratedCondition] =1./(self.deltaE[nondegeneratedCondition] - 1j*self.eta)
        self.deltaEInverse2[nondegeneratedCondition]=1./self.deltaE[nondegeneratedCondition]/(self.deltaE[nondegeneratedCondition] - 1j*self.eta)
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
        exp=np.exp(-1/self.kBT*abs(self.energies[None,...,self.ibmin:self.ibmax+1]-self.fermiList[:,None,None,None,None]))
        self.fermiWeight=-1/self.kBT/(exp+1/exp+2)
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
    def __init__(self, spinProjectionList=[], **kwarg):
        '''
        Compute spin matrix in eigen-basis

        Indexing:
        self.spin_K[ik1, ik2, ik3, idirection, ieig, jeig] gives spin matrix in eigen-basis.
        idirection can be 0 | 1 | 2, corresponding to sigma_x | sigma_y | sigma_z.
        '''
        super().__init__(**kwarg)
        vectors=self.vectors[:,:,:,None,:,:]
        if self.model.overlap:
            self.spin_K=vectors.conj().swapaxes(-2,-1) @ self.model.spin[None,None,None,:,:,:] @ self.O_K_AO[...,None,:,:] @ vectors
        else:
            self.spin_K=vectors.conj().swapaxes(-2,-1) @ self.model.spin[None,None,None,:,:,:] @ vectors
        if self.verbose==2: showElapsedTime(extraInfo=f"spin op. in eig.-basis calculated")

        self.projected_spin_K_list=[]
        for spinProjection in spinProjectionList:
            proj1=projectionOperator(spinProjection['source'],self.model.atomicOrbInfoList,self.model.norbs)
            proj2=projectionOperator(spinProjection['target'],self.model.atomicOrbInfoList,self.model.norbs)
            #spin_proj=(proj1@self.model.spin@proj2 + proj2@self.model.spin@proj1)/2
            #if self.model.overlap:
            #    projected_spin_K=vectors.conj().swapaxes(-2,-1) @ spin_proj[None,None,None,:,:,:] @ self.O_K_AO[...,None,:,:] @ vectors
            #else:
            #    projected_spin_K=vectors.conj().swapaxes(-2,-1) @ spin_proj[None,None,None,:,:,:] @ vectors
            #self.projected_spin_K_list.append(projected_spin_K)
            proj1_K=self.projectorInEigenBasis(proj1)[...,None,:,:]
            proj2_K=self.projectorInEigenBasis(proj2)[...,None,:,:]
            self.projected_spin_K_list.append((proj1_K@self.spin_K + self.spin_K@proj2_K)/2)

class Orbital(Energy):
    def __init__(self, orbitalProjectionList=[], **kwarg):
        '''
        Compute orbital matrix in eigen-basis

        Indexing:
        self.orbital_K[ik1, ik2, ik3, idirection, ieig, jeig] gives orbital matrix in eigen-basis.
        idirection can be 0 | 1 | 2, corresponding to L_x | L_y | L_z.
        '''
        super().__init__(**kwarg)
        vectors=self.vectors[:,:,:,None,:,:]
        if self.model.overlap:
            self.orbital_K=vectors.conj().swapaxes(-2,-1) @ self.model.orbital[None,None,None,:,:,:] @ self.O_K_AO[...,None,:,:] @ vectors
        else:
            self.orbital_K=vectors.conj().swapaxes(-2,-1) @ self.model.orbital[None,None,None,:,:,:] @ vectors
        if self.verbose==2: showElapsedTime(extraInfo=f"orbital op. in eig.-basis calculated")

        self.projected_orbital_K_list=[]
        for orbitalProjection in orbitalProjectionList:
            proj1=projectionOperator(orbitalProjection['source'],self.model.atomicOrbInfoList,self.model.norbs)
            proj2=projectionOperator(orbitalProjection['target'],self.model.atomicOrbInfoList,self.model.norbs)
            orbital_proj=(proj1@self.model.orbital@proj2 + proj2@self.model.orbital@proj1)/2
            if self.model.overlap:
                projected_orbital_K=vectors.conj().swapaxes(4,5) @ orbital_proj[None,None,None,:,:,:] @ self.O_K_AO[...,None,:,:] @ vectors
            else:
                projected_orbital_K=vectors.conj().swapaxes(4,5) @ orbital_proj[None,None,None,:,:,:] @ vectors
            self.projected_orbital_K_list.append(projected_orbital_K)

class SpinTorque(Spin):
    def __init__(self, spinTorqueProjectionList=[], **kwarg):
        '''
        Compute spin torque matrix in eigen-basis

        T = (-i/\hbar) [S, H]

        In eigen-basis, H is diagonalized eigen-energy, S is computed in parent Spin class. So, the only work is matrix product

        Indexing
        --------
        self.spinTorque_K[ik1, ik2, ik3, idirection, ieig, jeig] gives spin torque matrix in eigen-basis.
        idirection can be 0 | 1 | 2, corresponding to sigma_x | sigma_y | sigma_z.
        '''
        super().__init__(**kwarg)

        self.spinTorque_K=-1j*(self.spin_K*self.energies[...,None,None,:] - self.spin_K*self.energies[...,None,:,None])
        if self.verbose==2: showElapsedTime(extraInfo=f"spin torque in eig.-basis finished")
        self.projected_spinTorque_K_list=[]
        
        for spinTorqueProjection in spinTorqueProjectionList:
            proj1=projectionOperator(spinTorqueProjection['source'],self.model.atomicOrbInfoList,self.model.norbs)
            proj2=projectionOperator(spinTorqueProjection['target'],self.model.atomicOrbInfoList,self.model.norbs)
            proj1_K=self.projectorInEigenBasis(proj1)[...,None,:,:]
            proj2_K=self.projectorInEigenBasis(proj2)[...,None,:,:]
            self.projected_spinTorque_K_list.append((proj1_K@self.spinTorque_K + self.spinTorque_K@proj2_K)/2)

class Spin_diag(FermiScan):
    def __init__(self, spinProjectionList=[], **kwarg):
        '''
        Compute spin expectation of eigen-basis

        Indexing
        --------
        self.spin_diag_K[ik1, ik2, ik3, idirection, ieig] gives spin expectation of eigen-basis.
        idirection can be 0 | 1 | 2, corresponding to sigma_x | sigma_y | sigma_z.
        '''
        super().__init__(**kwarg)

        vectors=self.vectors[:,:,:,None,:,self.ibmin:self.ibmax+1]
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
            #proj1_K=proj1[None,None,None,:,:]@self.vectors[...,self.ibmin:self.ibmax+1]
            #proj2_K=proj2[None,None,None,:,:]@self.vectors[...,self.ibmin:self.ibmax+1]

class Orbital_diag(FermiScan):
    def __init__(self, orbitalProjectionList=[], **kwarg):
        '''
        Compute orbital expectation of eigen-basis

        Indexing
        --------
        self.orbital_diag_K[ik1, ik2, ik3, idirection, ieig] gives orbital angular moment expectation of eigen-basis.
        idirection can be 0 | 1 | 2, corresponding to L_x | L_y | L_z.
        '''
        super().__init__(**kwarg)

        vectors=self.vectors[:,:,:,None,:,self.ibmin:self.ibmax+1]
        if self.model.overlap:
            _ov=self.model.orbital[None,None,None,:,:,:] @ self.O_K_AO[...,None,:,:] @ vectors
            self.orbital_diag_K=np.sum(vectors.conj() * _ov, axis=(4,))
        else:
            _ov=self.model.orbital[None,None,None,:,:,:] @ vectors
            self.orbital_diag_K=np.sum(vectors.conj() * _ov, axis=(4,))
        if self.verbose==2: showElapsedTime(extraInfo=f"<L> of eig.-basis calculated")

        self.projected_orbital_diag_K_list=[]
        for orbitalProjection in orbitalProjectionList:
            proj1=projectionOperator(orbitalProjection['source'],self.model.atomicOrbInfoList,self.model.norbs)
            proj2=projectionOperator(orbitalProjection['target'],self.model.atomicOrbInfoList,self.model.norbs)
            orbital_proj=(proj1@self.model.orbital@proj2 + proj2@self.model.orbital@proj1)/2
            if self.model.overlap:
                _ov=orbital_proj[None,None,None,:,:,:] @ self.O_K_AO[...,None,:,:] @ vectors
                projected_orbital_diag_K=np.sum(vectors.conj() * _ov, axis=(4,))
            else:
                _ov=orbital_proj[None,None,None,:,:,:] @ vectors
                projected_orbital_diag_K=np.sum(vectors.conj() * _ov, axis=(4,))
            self.projected_orbital_diag_K_list.append(projected_orbital_diag_K)

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
        self.velocity_K=vectors.conj().swapaxes(-2,-1) @ self.RHam_K @ vectors

        r_K        =  np.fft.ifftn(r_R_intepolated,axes=(0,1,2,))*np.prod(self.kgrid)
        r_K_H = (vectors.conj().swapaxes(-2,-1) @ r_K @ vectors) * self.energies[...,None,None,:]
        self.velocity_K += -1j*(r_K_H - r_K_H.conj().swapaxes(-2,-1))
        if self.verbose==2: showElapsedTime(extraInfo=f"velocity modification with position operators finished")
        if self.verbose==2: showElapsedTime(extraInfo=f"velocity in eig.-basis finished")
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
        '''

        self.projected_velocity_K_list=[]
        for velocityProjection in velocityProjectionList:
            proj1=projectionOperator(velocityProjection['source'],self.model.atomicOrbInfoList,self.model.norbs)
            proj2=projectionOperator(velocityProjection['target'],self.model.atomicOrbInfoList,self.model.norbs)

            proj1_K=self.projectorInEigenBasis(proj1)[...,None,:,:]
            proj2_K=self.projectorInEigenBasis(proj2)[...,None,:,:]
            self.projected_velocity_K_list.append((proj1_K@self.velocity_K@proj2_K + proj2_K@self.velocity_K@proj1_K)/2)

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
        igamma can be 0 | 1 | 2 | 3, corresponding to sigma^x | sigma^y | sigma^z | sigma^0 spin-polarization. 

        self.spinVelocity_K[ik1, ik2, ik3, idirection, igamma, ieig, jeig] gives spin velocity matrix in eigen-basis.
        idirection can be 0 | 1 | 2, corresponding to x | y | z direction. 
        igamma can be 0 | 1 | 2 | 3, corresponding to sigma^x | sigma^y | sigma^z | sigma^0 spin-polarization. 

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
        igamma can be 0 | 1 | 2, corresponding to Lx | Ly | Lz orbital-polarization. 

        self.orbitalVelocity_K[ik1, ik2, ik3, idirection, igamma, ieig, jeig] gives orbital velocity matrix in eigen-basis.
        idirection can be 0 | 1 | 2, corresponding to x | y | z direction. 
        igamma can be 0 | 1 | 2, corresponding to Lx | Ly | Lz orbital-polarization. 

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
        igamma can be 0 | 1 | 2 | 3, corresponding to sigma^x | sigma^y | sigma^z | sigma^0 spin-polarization. 

        self.spinVelocity_K[ik1, ik2, ik3, idirection, igamma, ieig, jeig] gives spin velocity matrix in eigen-basis.
        idirection can be 0 | 1 | 2, corresponding to x | y | z direction. 
        igamma can be 0 | 1 | 2 | 3, corresponding to sigma^x | sigma^y | sigma^z | sigma^0 spin-polarization. 

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
        igamma can be 0 | 1 | 2, corresponding to Lx | Ly | Lz orbital-polarization. 

        self.orbitalVelocity_K[ik1, ik2, ik3, idirection, igamma, ieig, jeig] gives orbital velocity matrix in eigen-basis.
        idirection can be 0 | 1 | 2, corresponding to x | y | z direction. 
        igamma can be 0 | 1 | 2, corresponding to Lx | Ly | Lz orbital-polarization. 

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
    
    def __init__(self, velocityProjectionList=[], **kwarg) -> None:
        '''
        Calc. Berry curvature \Omega_{alpha,beta}_{k,mn}.

        alpha and beta can be x, y, z corresponds the (cart.) space direction, k denote the crystal moment, m, n index the bands.

        Indexing: BerryCurv[ik1, ik2, ik3, ialpha, ibeta, iband, jband]
        '''
        super().__init__(velocityProjectionList=velocityProjectionList, **kwarg)
        self.calcDeltaEInverse()
        dEInverse2 = self.deltaEInverse**2

        self.projected_BerryCurv=np.zeros((len(velocityProjectionList)+1,len(velocityProjectionList)+1)+self.kgrid+(3,3,self.model.norbs-self.ibmin))
        for ivelocity_alpha, proj_velocity_alpha_K in enumerate([self.velocity_K]+self.projected_velocity_K_list):
            for ivelocity_beta, proj_velocity_beta_K in enumerate([self.velocity_K]+self.projected_velocity_K_list):
                for ialpha in range(3):
                    for ibeta in range(3):
                        self.projected_BerryCurv[ivelocity_alpha,ivelocity_beta,...,ialpha,ibeta,:]=-2*np.diagonal(proj_velocity_beta_K[...,ibeta,:,:] @ (dEInverse2*proj_velocity_alpha_K[...,ialpha,:,:]),0,-2,-1).imag
        
        self.BerryCurv=self.projected_BerryCurv[0,0]
        if self.verbose==2: showElapsedTime(extraInfo=f"Berry curvature finished")

class AnormalousHallConductance(BerryCurv):

    def __init__(self, currentProjectionList=[], **kwarg) -> None:
        super().__init__(velocityProjectionList=currentProjectionList, **kwarg)
        self.Iodd = False  # inversion gives minus
        self.Todd = True   # timereversal gives minus

        nProjections=self.projected_BerryCurv.shape[:2]
        projected_sigma=np.zeros(nProjections+(len(self.fermiList),3,3))
        
        for ifermi, fermi in enumerate(self.fermiList):
            occupy=self.energies<fermi
            for ivelocity_alpha in range(nProjections[0]):
                for ivelocity_beta in range(nProjections[1]):
                    for ialpha in range(3):
                        for ibeta in range(3):
                            projected_sigma[ivelocity_alpha,ivelocity_beta,ifermi,ialpha,ibeta]=np.sum(self.projected_BerryCurv[ivelocity_alpha,ivelocity_beta,...,ialpha,ibeta,:][occupy])
        
        #self.projected_sigma=projected_sigma/np.prod(self.kgrid)*(elementary_charge**2/hbar/angstrom/2)/self.model.Vcell
        self.projected_sigma=projected_sigma*(2*np.pi/(np.linalg.det(self.model.lattice[:2,:2])*np.prod(self.kgrid[:2]))) # for checking the TKNN SHC of 2D topo./trivial sys.
        if self.verbose==2: showElapsedTime(extraInfo=f"Anormalous Hall conductance finished")

class VelocityVelocityVelocityCorrelation_AC(Velocity):
    def __init__(self, velocityProjectionList=[], **kwarg) -> None:
        '''
        Calc. the 3rd order velocity correlation function \Omega_{alpha1, alpha2,beta}_{k,mn} and the BPVE conductance
        '''
        super().__init__(velocityProjectionList=velocityProjectionList, **kwarg)
        self.Iodd = True  # inversion gives minus
        self.Todd = True  # timereversal gives minus

        # inversed energy difference [nk1, nk2, nk3, 1 (for fermi), ib1, ib2]
        den1 = self.energies[..., None, None, None, :] - self.energies[..., None, None, :, None] - 1j* self.eta # (En-Em-i\eta)
        #dE_inv = 1 / (dE + 1j * self.eta)

        # occupation number [nk1, nk2, nk3, ifermi, ib]
        f = np.int8(self.energies[..., None, :] < self.fermiList[None, None, None, :, None])
        # occupation number difference [nk1, nk2, nk3, ifermi, ibl, ibm, ibn]
        fln = f[..., :, None, None] - f[..., None, None, :]
        
        self.projected_sigma = np.zeros((len(velocityProjectionList)+1, len(velocityProjectionList)+1, len(velocityProjectionList)+1) + (len(self.fermiList), len(self.omegaList),) + (3, 3, 3))
        
        for iomega, Omega in enumerate(self.omegaList):
            projected_velocityVelocityVelocityCorrelation = np.zeros((len(velocityProjectionList)+1, len(velocityProjectionList)+1, len(velocityProjectionList)+1) + self.kgrid + (len(self.fermiList), 3, 3, 3))
            for ivelocity_alpha1, proj_velocity_alpha1_K in enumerate([self.velocity_K] + self.projected_velocity_K_list):
                for ivelocity_alpha2, proj_velocity_alpha2_K in enumerate([self.velocity_K] + self.projected_velocity_K_list):
                    for ivelocity_beta, proj_velocity_beta_K in enumerate([self.velocity_K] + self.projected_velocity_K_list):
                        for ialpha1 in range(3):
                            v_alpha1 = proj_velocity_alpha1_K[...,ialpha1,:,:]
                            for ialpha2 in range(3):
                                v_alpha2 = proj_velocity_alpha2_K[...,ialpha2,:,:]
                                for ibeta in range(3):
                                    v_beta = proj_velocity_beta_K[...,ibeta,:,:]
                                    num = np.einsum('abcnl,abclm,abcmn->abclmn', v_alpha1, v_alpha2, v_beta)

                                    for W in [Omega, -Omega]:
                                        den2 = self.energies[..., None, None, None, :] - self.energies[..., None, :, None, None] - W - 1j* self.eta # (En - El - \Omega - i\eta)
                                        projected_velocityVelocityVelocityCorrelation[ivelocity_alpha1, ivelocity_alpha2, ivelocity_beta, ..., ialpha1, ialpha2, ibeta] += -2*np.sum(fln * num[...,None,:,:,:] / (den1 * den2), axis=(-3,-2,-1)).real
                                    
            # Sum over the BZ grid axes (3,4,5) and divide by Omega^2
            self.projected_sigma[..., iomega, :, :, :] = np.sum(projected_velocityVelocityVelocityCorrelation, axis=(3, 4, 5)) / (Omega**2)
            
        self.projected_sigma /= np.prod(self.kgrid)
        if self.verbose == 2: showElapsedTime(extraInfo=f"Bulk photovoltaic conductance finished")

class __VelocityVelocityVelocityCorrelation_AC(Velocity):
    def __init__(self, velocityProjectionList=[], **kwarg) -> None:
        '''
        Calc. the 3rd order velocity correlation function \Omega_{alpha1, alpha2,beta}_{k,mn} and the BPVE conductance
        '''
        super().__init__(velocityProjectionList=velocityProjectionList, **kwarg)
        self.Iodd = True  # inversion gives minus
        self.Todd = True  # timereversal gives minus

        # inversed energy difference [nk1, nk2, nk3, 1 (for fermi), ib1, ib2]
        dE = self.energies[..., None, :, None] - self.energies[..., None, None, :]
        dE_inv = 1 / (dE + 1j * self.eta)

        # occupation number [nk1, nk2, nk3, ifermi, ib]
        f = np.int8(self.energies[..., None, :] < self.fermiList[None, None, None, :, None])
        # occupation number difference [nk1, nk2, nk3, ifermi, ib1, ib2]
        df = f[..., :, None] - f[..., None, :]
        
        self.projected_sigma = np.zeros((len(velocityProjectionList)+1, len(velocityProjectionList)+1, len(velocityProjectionList)+1) + (len(self.fermiList), len(self.omegaList),) + (3, 3, 3))
        
        for iomega, Omega in enumerate(self.omegaList):
            # transition rate [nk1, nk2, nk3, ifermi, ib1, ib2]
            t = df * (1 / (Omega + dE + 1j * self.eta) + 1 / (-Omega + dE + 1j * self.eta))
            # dt shape: [nk1, nk2, nk3, ifermi, orbA, orbB, orbC]
            dt = t[..., :, None, :] - t.swapaxes(-1, -2)[..., None, :, :]
            
            for ivelocity_alpha1, proj_velocity_alpha1_K in enumerate([self.velocity_K] + self.projected_velocity_K_list):
                for ivelocity_alpha2, proj_velocity_alpha2_K in enumerate([self.velocity_K] + self.projected_velocity_K_list):
                    for ivelocity_beta, proj_velocity_beta_K in enumerate([self.velocity_K] + self.projected_velocity_K_list):
                        for ialpha1 in range(3):
                            for ialpha2 in range(3):
                                for ibeta in range(3):
                                    
                                    # Extract velocity components and inject 'None' for the nfermi dimension 
                                    # so shapes jump from 6D to 7D, perfectly aligning with dt!
                                    v_alpha1 = proj_velocity_alpha1_K[..., None, ialpha1, None, :, :]
                                    v_alpha2 = proj_velocity_alpha2_K.swapaxes(-1, -2)[..., None, ialpha2, :, None, :]
                                    v_beta   = proj_velocity_beta_K[..., None, ibeta, :, :, None]
                                    
                                    # Sum ONLY over the 3 orbital axes (-3, -2, -1) to preserve (nk1, nk2, nk3, nfermi)
                                    bubble_sum = np.sum(
                                        v_beta * v_alpha1 * v_alpha2 * dE_inv[..., :, :, None] * dt, 
                                        axis=(0, 1, 2, -3, -2, -1)
                                    ).real
                                    
                                    self.projected_sigma[ivelocity_alpha1, ivelocity_alpha2, ivelocity_beta, :, iomega, ialpha1, ialpha2, ibeta] = bubble_sum
                                    
            # Sum over the BZ grid axes (3,4,5) and divide by Omega^2
            #self.projected_sigma[..., iomega, :, :, :] = np.sum(self.projected_velocityVelocityVelocityCorrelation, axis=(3, 4, 5)) / (Omega**2)
            
        self.projected_sigma /= np.prod(self.kgrid)
        if self.verbose == 2: showElapsedTime(extraInfo=f"Bulk photovoltaic conductance finished")

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
        Calc. inter-band contributions on the correlation func of orbital angular moment and velocity \Omega^{alpha,gamma}_{k,m}.

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
        Calc. intra-band contributions on the correlation func of orbital angular moment and velocity \Omega^{alpha,gamma}_{k,m}.

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

class SpintorqueVelocityCorrelation(SpinTorque,Velocity):
    def __init__(self, velocityProjectionList=[], spinTorqueProjectionList=[], **kwarg) -> None:
        '''
        Calc. correlation func of spin torque and velocity \Omega^{alpha,gamma}_{k,mn}.

        alpha can be x, y, z, denoting the (cart.) space direction. 
        gammma can be sigma^x, sigma^y, sigma^z, sigma^0, corresponding to the torque orientation. 
        k denotes the crystal moment, m, n index the bands.

        Indexing
        --------
        spinBerryCurv[ik1, ik2, ik3, ialpha, igamma, ieig]
        '''
        super().__init__(velocityProjectionList=velocityProjectionList,spinTorqueProjectionList=spinTorqueProjectionList, **kwarg)
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

from .Path import EnergyOnPath, VelocityOnPath, BerryCurvOnPath, SpinBerryCurvOnPath, SpinExpectationOnPath, SpinOrbitalExpectationOnPath
from .Path import SurfaceBulkSpectraOnPath
