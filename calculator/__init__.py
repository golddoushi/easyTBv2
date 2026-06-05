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
    def __init__(self, velocityProjectionList=[], highOrderVelocity=False, **kwarg) -> None:
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
        velocity_bare_K=vectors.conj().swapaxes(-2,-1) @ self.RHam_K @ vectors

        # ==========================================================
        # Gauge-Covariant Geometric Corrections for velocity
        # ==========================================================
        # v_a = H^(1)_a - i[r_a, H^(0)]
        r_K        =  np.fft.ifftn(r_R_intepolated,axes=(0,1,2,))*np.prod(self.kgrid)
        self.position_K = vectors.conj().swapaxes(-2,-1) @ r_K @ vectors # position matrix in eigen representation
        r_K_H = self.position_K * self.energies[...,None,None,:]
        self.velocity_K = velocity_bare_K -1j*(r_K_H - r_K_H.conj().swapaxes(-2,-1))

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

            # ==========================================================
            # Gauge-Covariant Geometric Corrections
            # ==========================================================
            def comm(A, B):
                # Helper to compute commutator [A, B] broadcasting over k-grid
                return A @ B - B @ A

            # Store the bare spatial moments
            H1 = velocity_bare_K #self.RHam_K
            H2 = self.w_K.copy()

            # 1. Correct w_K
            # w_ab = H^(2)_ab - i[r_a, H^(1)_b] - i[r_b, v_a]
            for a in range(3):
                for b in range(3):
                    c1 = comm(self.position_K[..., a, :, :], H1[..., b, :, :])
                    c2 = comm(self.position_K[..., b, :, :], self.velocity_K[..., a, :, :])
                    self.w_K[..., a, b, :, :] = H2[..., a, b, :, :] - 1j * c1 - 1j * c2

            # 2. Correct u_K
            # u_abc = H^(3)_abc - i[r_a, H^(2)_bc] - i[r_b, H^(2)_ac] - [r_b, [r_a, H^(1)_c]] - i[r_c, w_ab]
            H3 = self.u_K.copy()
            for a in range(3):
                for b in range(3):
                    for c in range(3):
                        c1 = comm(self.position_K[..., a, :, :], H2[..., b, c, :, :])
                        c2 = comm(self.position_K[..., b, :, :], H2[..., a, c, :, :])
                        
                        c3_inner = comm(self.position_K[..., a, :, :], H1[..., c, :, :])
                        c3 = comm(self.position_K[..., b, :, :], c3_inner)
                        
                        # Note: This explicitly uses the CORRECTED w_K calculated above
                        c4 = comm(self.position_K[..., c, :, :], self.w_K[..., a, b, :, :])

                        self.u_K[..., a, b, c, :, :] = H3[..., a, b, c, :, :] - 1j*c1 - 1j*c2 - c3 - 1j*c4

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

class BulkPhotovoltaicConductance(Velocity):
    def __init__(self, velocityProjectionList=[], responseOmegaIndexList=[], **kwarg) -> None:
        '''
        Calc. the 3rd order velocity correlation function \Omega_{alpha1, alpha2, beta}_{k,mn} and the BPVE conductance

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

class BulkPhotovoltaicConductance_VG_Full(Velocity):
    def __init__(self, velocityProjectionList=[], responseOmegaIndexList=[], **kwarg) -> None:
        '''
        Calculates the COMPLETE Velocity Gauge BPVE by summing all four Feynman diagrams.
        Rewritten using np.einsum to guarantee exact quantum mechanical index tracking.
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
        D_nm = 1.0 / (E_n - E_m - 1j * self.eta)

        # 2D energy arrays, index [ik1, ik2, ik3, m, n]
        E_m_2 = self.energies[..., :, None]
        E_n_2 = self.energies[..., None, :]
        D_mn_2 = 1.0 / (E_m_2 - E_n_2 - 1j * self.eta)

        for ifermi, fermi in enumerate(self.fermiList):
            # occupation number: index [ik1, ik2, ik3, ib]
            f = np.int8(self.energies < fermi)
            # 3D occupation arrays
            f_n =             f[..., None, :, None]
            f_l =             f[..., None, None, :]
            f_nl= f_n - f_l

            # 2D occupation arrays
            f_m_2 =             f[..., :, None]
            f_n_2 =             f[..., None, :]
            f_mn_2= f_m_2 - f_n_2    

            for iomega, Omega in enumerate(self.omegaList):
                # Resonant Denominators
                S2_pos = 1.0 / (E_n - E_l - Omega - 1j * self.eta)
                S2_neg = 1.0 / (E_n - E_l + Omega - 1j * self.eta)
                Gw_mnl = f_nl * D_nm * (S2_pos + S2_neg)
            
                S_mn_pos = 1.0 / (E_m_2 - E_n_2 - Omega - 1j * self.eta)
                S_mn_neg = 1.0 / (E_m_2 - E_n_2 + Omega - 1j * self.eta)
                Gw_mn_2  = f_mn_2 * (S_mn_pos + S_mn_neg)
                G_mn_2= f_mn_2 * D_mn_2
            
                for a in range(3):
                    for b in range(3):
                        for c in range(3):
                        
                            # (ik1, ik2, ik3, norbs, norbs)
                            v_a = self.velocity_K[..., a, :, :]
                            v_b = self.velocity_K[..., b, :, :]
                            v_c = self.velocity_K[..., c, :, :]
                        
                            w_ab = self.w_K[..., a, b, :, :]
                            w_ca = self.w_K[..., c, a, :, :]
                            w_cb = self.w_K[..., c, b, :, :]
                        
                            u_cab = self.u_K[..., c, a, b, :, :]
                            
                            # =======================================================
                            # 1. Three-Point Bubble: Tr[ v^c v^a v^b ]
                            # =======================================================
                            # Einsum maps: v^c_{mn} * v^a_{nl} * v^b_{lm}
                            vvv = np.einsum('abcmn, abcnl, abclm -> abcmnl', v_c, v_a, v_b)
                            I3 = 2* np.sum(vvv * Gw_mnl, axis=(-3, -2, -1))
                            
                            # =======================================================
                            # 2. Two-Point Bubble A (Diamagnetic Perturbation): Tr[ v^c w^{ab} ]
                            # =======================================================
                            rho_w = w_ab * G_mn_2
                            I2A = np.einsum('abcmn, abcnm -> abc', v_c, rho_w)
                            
                            # =======================================================
                            # 3. Two-Point Bubble B (Diamagnetic Current): Tr[ w^{ca} v^b ]
                            # =======================================================
                            rho_v_b = v_b * Gw_mn_2
                            rho_v_a = v_a * Gw_mn_2
                        
                            # Symmetrize driving fields
                            I2B = 0.5 * np.einsum('abcmn, abcnm -> abc', w_ca, rho_v_b) + \
                                  0.5 * np.einsum('abcmn, abcnm -> abc', w_cb, rho_v_a)
                            
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

class BulkPhotovoltaicConductance_LG_Shift(Velocity):
    def __init__(self, velocityProjectionList=[], **kwarg) -> None:
        '''
        Calculates the Universal Shift Current (BPVE) using the Sipe-Shkrebtii Length Gauge.
        Natively evaluates BOTH the Linear Shift Current and Circular Shift Current.
        '''
        super().__init__(velocityProjectionList=velocityProjectionList, highOrderVelocity=True, **kwarg)
        #self.calcDeltaEInverse()
        #omega_nm = self.deltaE 
        #inv_omega = self.deltaEInverse

        # 2D energy arrays, index [ik1, ik2, ik3, m, n]
        E_m_2 = self.energies[..., :, None]
        E_n_2 = self.energies[..., None, :]
        omega_mn = E_m_2 - E_n_2
        nondegenerate=abs(omega_mn)>self.degen
        inv_omega = np.zeros_like(omega_mn,dtype=float)
        inv_omega[nondegenerate]=1/(omega_mn[nondegenerate])

        norbs = self.model.norbs - self.ibmin
        off_diag = 1.0 - np.eye(norbs)
        
        # Tensor Shape: [ifermi, iomega, ialpha(drive), ibeta(drive), igamma(current)]
        self.sigma_shift = np.zeros((len(self.fermiList), len(self.omegaList), 3, 3, 3), dtype=complex)

        # 1. Compute Dipole Matrices (r^a_nm)
        r_K = self.velocity_K * inv_omega[..., None, :, :] / 1j * off_diag

        # 2. Compute Group Velocity Differences (Delta^a_nm)
        v_diag = np.diagonal(self.velocity_K, axis1=-2, axis2=-1) 
        Delta_v = v_diag[..., :, :, None] - v_diag[..., :, None, :] 

        for iomega, Omega in enumerate(self.omegaList):
            L_mn_pos = (1/np.pi) * self.eta / ((omega_mn - Omega)**2 + self.eta**2)
            L_mn_neg = (1/np.pi) * self.eta / ((omega_mn + Omega)**2 + self.eta**2)
            L_mn = L_mn_pos + L_mn_neg

            for ialpha in range(3): # Drive E-field a
                for ibeta in range(3): # Drive E-field b
                    for igamma in range(3): # Measured Current c
                        
                        v_a = self.velocity_K[..., ialpha, :, :]
                        v_c = self.velocity_K[..., igamma, :, :]  
                        
                        r_b = r_K[..., ibeta, :, :]
                        r_c = r_K[..., igamma, :, :]
                        
                        w_ca = self.w_K[..., igamma, ialpha, :, :]
                        Delta_a = Delta_v[..., ialpha, :, :]
                        
                        # Generalized Derivative w.r.t 'a' (Only 'a' is needed now!)
                        comm_rv_a = r_c @ v_a - v_a @ r_c
                        r_c_deriv_a = (w_ca - v_c * Delta_a * inv_omega - comm_rv_a) * inv_omega / 1j * off_diag
                        
                        # Universal Complex Shift Integrand: I^{cab}_mn
                        # The factor of -1j rotates the complex plane so that:
                        # 1. Re[I_mn] matches Sipe's original Im[...] formula for Linear Shift
                        # 2. Im[I_mn] cleanly isolates the Circular Shift Current
                        I_mn = -1j * r_b.conj() * r_c_deriv_a
                        
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

class BulkPhotovoltaicConductance_LG_Injection(Velocity):
    def __init__(self, velocityProjectionList=[], **kwarg) -> None:
        '''
        Calculates the Circular Injection Current (Circular BPVE) using the 
        Sipe-Shkrebtii Length Gauge formalism.
        '''
        # No high order velocity needed! This makes LG much faster than VG.
        super().__init__(velocityProjectionList=velocityProjectionList, highOrderVelocity=False, **kwarg)
        
        # 2D energy arrays, index [ik1, ik2, ik3, m, n]
        E_m_2 = self.energies[..., :, None]
        E_n_2 = self.energies[..., None, :]
        omega_mn = E_m_2 - E_n_2
        nondegenerate=abs(omega_mn)>self.degen
        inv_omega = np.zeros_like(omega_mn,dtype=float)
        inv_omega[nondegenerate]=1/(omega_mn[nondegenerate])
        
        norbs = self.model.norbs - self.ibmin
        off_diag = 1.0 - np.eye(norbs)
        
        # Tensor Shape: [ifermi, iomega, ialpha(drive), ibeta(drive), igamma(current)]
        # Must be COMPLEX because the Circular BPVE susceptibility is purely imaginary!
        self.sigma_inj = np.zeros((len(self.fermiList), len(self.omegaList), 3, 3, 3), dtype=complex)

        # 1. Compute Interband Dipole Matrices (r^a_nm)
        r_K = self.velocity_K * inv_omega[..., None, :, :] / 1j * off_diag

        # 2. Compute Group Velocity Differences (Delta^c_nm)
        v_diag = np.diagonal(self.velocity_K, axis1=-2, axis2=-1) 
        Delta_v = v_diag[..., :, :, None] - v_diag[..., :, None, :] 

        # 3. Relaxation Time (tau = 1 / 2\eta in natural units)
        # Injection current scales linearly with the scattering time!
        tau = 1.0 / (2.0 * self.eta)

        for iomega, Omega in enumerate(self.omegaList):
            # Energy conservation delta functions (Lorentzians)
            L_mn_pos = (1/np.pi) * self.eta / ((omega_mn - Omega)**2 + self.eta**2)
            L_mn_neg = (1/np.pi) * self.eta / ((omega_mn + Omega)**2 + self.eta**2)
            L_mn = L_mn_pos + L_mn_neg

            for ialpha in range(3): # Drive E-field a
                r_a = r_K[..., ialpha, :, :]
                
                for ibeta in range(3): # Drive E-field b
                    r_b = r_K[..., ibeta, :, :]
                    
                    # Full Complex Optical Transition Tensor
                    # Real part drives Linear Injection (if magnetic)
                    # Imaginary part drives Circular Injection
                    rho_ab = r_a * r_b.conj()
                    
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

class BulkPhotovoltaicConductance_TDSE(Velocity):
    def __init__(self, velocityProjectionList=[], E0=0.001, n_steps_per_cycle=100, highOrderVelocity=True, **kwarg) -> None:
        '''
        Calculates the 2nd order DC photovoltaic conductance using Real-Time TDSE.
        Evolves the density matrix using a 4th-Order Runge-Kutta (RK4) integrator.
        
        Outputs
        -------
        self.sigma_TDSE[ifermi, iomega, ialpha_drive, ibeta_measure]
        This matches the [ialpha, ialpha, ibeta] diagonal of the Kubo AC tensor.
        '''
        super().__init__(velocityProjectionList=velocityProjectionList, highOrderVelocity=highOrderVelocity, **kwarg)
        
        # Shape: [ifermi, iomega, ialpha(driving field), ibeta(measured current)]
        self.sigma_TDSE = np.zeros((len(self.fermiList), len(self.omegaList), 3, 3, 3), dtype=float)
        
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

            for iomega, Omega in enumerate(self.omegaList):
                T = 2 * np.pi / Omega
                dt = T / n_steps_per_cycle
                
                # Compute cycles needed for the transient to decay (e.g., 10 lifetimes)
                t_transient = 10.0 / self.eta
                n_cycles_transient = int(np.ceil(t_transient / T))
                n_cycles_measure = 1
                
                if self.verbose:
                    print(f"TDSE [Fermi={fermi:.2f}, Omega={Omega:.2f}]: Simulating {n_cycles_transient} transient cycles + {n_cycles_measure} measurement cycle.")

                for ialpha in range(3): # Direction of the driving field
                    v_alpha = self.velocity_K[..., ialpha, :, :]
                    w_alpha_alpha = self.w_K[..., ialpha, ialpha, :, :]
                    u_alpha_alpha_alpha = self.u_K[..., ialpha, ialpha, ialpha, :, :]

                    rho = rho0.copy()
                    t = 0.0
                    
                    # Phase 1: Evolve through transients to reach Non-Equilibrium Steady State
                    for _ in range(n_cycles_transient * n_steps_per_cycle):
                        rho = self._rk4_step(rho, t, dt, H_diag, v_alpha, w_alpha_alpha, u_alpha_alpha_alpha, Omega, rho0, E0)
                        t += dt
                        
                    # Phase 2: Time-average the current over the final optical cycle
                    J_dc = np.zeros(3)
                    for _ in range(n_cycles_measure * n_steps_per_cycle):
                        rho = self._rk4_step(rho, t, dt, H_diag, v_alpha, w_alpha_alpha, u_alpha_alpha_alpha, Omega, rho0, E0)
                        t += dt

                        A_t = -2 * E0/Omega * np.sin(Omega * t)
                        
                        # Measure current in all 3 directions
                        for ibeta in range(3):
                            v_beta = self.velocity_K[..., ibeta, :, :]
                            w_alpha_beta = self.w_K[..., ialpha, ibeta, :, :]
                            u_alpha_alpha_beta = self.u_K[..., ialpha, ialpha, ibeta, :, :]

                            # Full Current Operator: J = v - A*w + 1/2 A^2 u
                            J_op = v_beta - A_t * w_alpha_beta + 0.5 * (A_t**2) * u_alpha_alpha_beta
                            # only the paramagnetic current
                            #J_op = 0.5 * (A_t**2) * u_alpha_alpha_beta

                            # Tr(rho * v) summed over all k-points and bands
                            j_val = np.sum(rho * J_op.swapaxes(-1, -2), axis=(-1,-2)).real
                            J_dc[ibeta] += np.sum(j_val) / (n_cycles_measure * n_steps_per_cycle)
                            
                    # Conductance = J / E^2
                    # Divide by volume/N_k
                    self.sigma_TDSE[ifermi, iomega, ialpha, ialpha, :] = J_dc / E0**2 / np.prod(self.kgrid)
        self.projected_sigma=self.sigma_TDSE[None,None,None,...]
        # unit conversion
        #factor_SI = 4*elementary_charge**2/hbar/self.model.Vcell 
        #if self.model.ispin==1: factor_SI*=2
        #self.projected_sigma *= factor_SI # in unit A / V^2

        # to responsibity (uA/W)
        #factor_responsivity = (753.46 / self.model.refractive_index) * 1e6 # 1e6 for micro-Amps
        #self.projected_responsibility = self.projected_sigma * factor_responsivity
        if self.verbose == 2: showElapsedTime(extraInfo=f"TDSE BPVE finished")

    def _rk4_step(self, rho, t, dt, H_diag, v_alpha, w_alpha_alpha, u_alpha_alpha_alpha, Omega, rho0, E0):
        # A(t) = A0 exp(-iwt)
        # E(t) = -dA/dt = (iw) A0 exp(-iw)
        # E(w) = iw A(w), A(w) = (-i/w) E(w)
        # Assuming real electric field: E(W) = E(-W) = E0, E(t) = E(W) exp(-iWt) + E(-W) exp(iWt) = 2 E0 cos(Wt)
        # A(t) = A(W) exp(-iWt) + A(-W) exp(iWt) = (-i/W) E(W) exp(-iWt) + (i/W) E(-W) exp(iWt) = -2 E0/W sin(Wt)
        def dRho_dt(rho_state, time_val):
            A_t = -2 * E0/Omega * np.sin(Omega * time_val)
            #H_t = H_diag - A_t * v_alpha
            # Full Minimal Coupling: H = H0 - A*v + 1/2 A^2 w - 1/6 A^3 u
            H_t = (H_diag 
                   - A_t * v_alpha                           # paramagentic drive
                   + 1./2 * (A_t**2) * w_alpha_alpha         # diamagentic drive 
                   #- 1./6 * (A_t**3) * u_alpha_alpha_alpha  # Fermi sea drive
                   )
            comm = H_t @ rho_state - rho_state @ H_t
            return -1j * comm - self.eta * (rho_state - rho0)

        k1 = dRho_dt(rho, t)
        k2 = dRho_dt(rho + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = dRho_dt(rho + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = dRho_dt(rho + dt * k3, t + dt)
        return rho + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
class BulkPhotovoltaicConductance_TDSE_LG_Naive(Velocity):
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

        # 2. Interband Dipole Matrix r_{nm} = v_{nm} / (i * omega_{mn})
        omega_mn = self.energies[..., :, None] - self.energies[..., None, :]
        inv_omega = np.zeros_like(omega_mn, dtype=float)
        nondegenerate = abs(omega_mn) > self.degen
        inv_omega[nondegenerate] = 1.0 / omega_mn[nondegenerate]
        
        off_diag = 1.0 - np.eye(self.model.norbs)
        r_K = self.velocity_K * inv_omega[..., None, :, :] / 1j * off_diag

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
                    r_alpha = r_K[..., ialpha, :, :]
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

class __InverseFaradayEffect_LG(Velocity, Spin, Orbital):
    def __init__(self, ife_type="total", velocityProjectionList=[], **kwarg) -> None:
        '''
        Calculates the Inverse Faraday Effect (IFE) Susceptibility Tensor.
        Utilizes the Length Gauge to ensure absolute stability at low frequencies.
        
        Parameters:
        - ife_type: "total", "spin", or "orbital" (Selects the magnetization operator)
        '''
        # highOrderVelocity=False because Length Gauge does NOT need the w_K or u_K diamagnetic tensors!
        super().__init__(velocityProjectionList=velocityProjectionList, highOrderVelocity=False, **kwarg)
        
        norbs = self.model.norbs
        
        # =======================================================
        # 1. Fetch Magnetic Moment Operator (M = L + g_s S)
        # =======================================================
        M_K = np.zeros(self.kgrid+(3, norbs, norbs), dtype=complex)
        
        if ife_type in ["total", "spin"]:
            # the spin matrice is constructed with sigma matrix which is the double of real spin matrice
            # so there is no need for the extra factor 2
            M_K += self.spin_K
            if self.verbose: print("IFE: Spin Magnetization Operator Loaded.")
            
        if ife_type in ["total", "orbital"]:
            M_K += self.orbital_K
            if self.verbose: print("IFE: Orbital Magnetization Operator Loaded.")
        
        # =======================================================
        # 2. Setup Energy Denominators
        # =======================================================
        E_m = self.energies[..., :, None, None]
        E_n = self.energies[..., None, :, None]
        E_l = self.energies[..., None, None, :]
        
        D_nm = 1.0 / (E_n - E_m - 1j * self.eta)
        
        # Tensor Shape: [ifermi, iomega, c(Magnetization), a(E-field 1), b(E-field 2)]
        self.chi_IFE = np.zeros((len(self.fermiList), len(self.omegaList), 3, 3, 3), dtype=complex)
        
        for iomega, Omega in enumerate(self.omegaList):
            
            # Sipe Interband Perturbation Denominators
            S_lm_pos = 1.0 / (E_l - E_m + Omega - 1j * self.eta)
            S_nl_pos = 1.0 / (E_n - E_l + Omega - 1j * self.eta)
            S_lm_neg = 1.0 / (E_l - E_m - Omega - 1j * self.eta)
            S_nl_neg = 1.0 / (E_n - E_l - Omega - 1j * self.eta)
            
            for c in range(3): # Measurement Direction (M_c)
                M_c = M_K[..., c, :, :]
                for a in range(3): # Drive Field 1 (E_a)
                    r_a = self.position_K[..., a, :, :]
                    for b in range(3): # Drive Field 2 (E_b)
                        r_b = self.position_K[..., b, :, :]
                        
                        # Execute the exact Tr[ M r_b r_a ] and Tr[ M r_a r_b ] operators
                        Mrr_ba = np.einsum('ijkmn, ijknl, ijklm -> ijkmnl', M_c, r_b, r_a)
                        Mrr_ab = np.einsum('ijkmn, ijknl, ijklm -> ijkmnl', M_c, r_a, r_b)
                        
                        for ifermi, fermi in enumerate(self.fermiList):
                            f   = np.int8(self.energies < fermi)
                            f_m = f[..., :, None, None]
                            f_n = f[..., None, :, None]
                            f_l = f[..., None, None, :]
                            
                            f_lm = f_l - f_m
                            f_nl = f_n - f_l
                            
                            # Liouville Density Matrix Commutator terms
                            term_ba = D_nm * (-f_lm * S_lm_pos + f_nl * S_nl_neg)
                            term_ab = D_nm * ( f_nl * S_nl_pos - f_lm * S_lm_neg)
                            
                            integrand = Mrr_ba * term_ba + Mrr_ab * term_ab
                            
                            self.chi_IFE[ifermi, iomega, c, a, b] = np.sum(integrand) / np.prod(self.kgrid)

        self.projected_sigma = self.chi_IFE[None,None,None,...]
        if self.verbose == 2: showElapsedTime(extraInfo=f"Length Gauge IFE finished")

class InverseFaradayEffect_LG(Velocity, Spin, Orbital):
    def __init__(self, ife_type="total", velocityProjectionList=[], **kwarg) -> None:
        super().__init__(velocityProjectionList=velocityProjectionList, highOrderVelocity=False, **kwarg)
        
        M_K = np.zeros(self.kgrid + (3, self.model.norbs, self.model.norbs), dtype=complex)
        if ife_type in ["total", "spin"]: M_K += self.spin_K
        if ife_type in ["total", "orbital"]: M_K += self.orbital_K

        omega_mn = self.energies[..., :, None] - self.energies[..., None, :]
        inv_omega = np.zeros_like(omega_mn, dtype=float)
        nondegenerate = abs(omega_mn) > self.degen
        inv_omega[nondegenerate] = 1.0 / omega_mn[nondegenerate]
        
        off_diag = 1.0 - np.eye(self.model.norbs - self.ibmin)
        r_K = self.velocity_K * inv_omega[..., None, :, :] / 1j * off_diag

        self.chi_IFE = np.zeros((len(self.fermiList), len(self.omegaList), 3, 3, 3), dtype=complex)

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

                for c in range(3):
                    M_c = M_K[..., c, :, :]
                    for a in range(3):
                        r_a = r_K[..., a, :, :]
                        for b in range(3):
                            r_b = r_K[..., b, :, :]
                            
                            Mrr_ba = np.einsum('...mn, ...nl, ...lm -> ...mnl', M_c, r_b, r_a)
                            Mrr_ab = np.einsum('...mn, ...nl, ...lm -> ...mnl', M_c, r_a, r_b)
                            
                            integrand = Mrr_ba * term_ba + Mrr_ab * term_ab
                            
                            # CORRECTED: np.sum() now integrates over all bands AND the k-grid
                            self.chi_IFE[ifermi, iomega, c, a, b] = np.sum(integrand) / np.prod(self.kgrid)

        self.projected_sigma = self.chi_IFE[None,None,None,...]
        if self.verbose == 2: showElapsedTime(extraInfo=f"Length Gauge IFE finished")

class InverseFaradayEffect_VG(Velocity, Spin, Orbital):
    def __init__(self, ife_type="total", velocityProjectionList=[], **kwarg) -> None:
        super().__init__(velocityProjectionList=velocityProjectionList, highOrderVelocity=True, **kwarg)
        
        M_K = np.zeros(self.kgrid + (3, self.model.norbs, self.model.norbs), dtype=complex)
        if ife_type in ["total", "spin"]: M_K += self.spin_K
        if ife_type in ["total", "orbital"]: M_K += self.orbital_K

        self.chi_IFE = np.zeros((len(self.fermiList), len(self.omegaList), 3, 3, 3), dtype=complex)
        
        # The response tensor below is built as M_{mn} v_{nl} v_{lm}, so
        # the three band axes are ordered as (m, n, l).
        E_m = self.energies[..., :, None, None]
        E_n = self.energies[..., None, :, None]
        E_l = self.energies[..., None, None, :]
        D_nm = 1.0 / (E_n - E_m - 1j * self.eta)

        E_m_2 = self.energies[..., :, None]
        E_n_2 = self.energies[..., None, :]
        D_mn_2 = 1.0 / (E_m_2 - E_n_2 - 1j * self.eta)

        for ifermi, fermi in enumerate(self.fermiList):
            f = np.int8(self.energies < fermi)
            # 3D occupation arrays
            f_n =             f[..., None, :, None]
            f_l =             f[..., None, None, :]
            f_nl= f_n - f_l

            # 2D occupation arrays
            f_m_2 =             f[..., :, None]
            f_n_2 =             f[..., None, :]
            f_mn_2= f_m_2 - f_n_2  

            for iomega, Omega in enumerate(self.omegaList):
                # Resonant Denominators
                S2_pos = 1.0 / (E_n - E_l - Omega - 1j * self.eta)
                S2_neg = 1.0 / (E_n - E_l + Omega - 1j * self.eta)
                Gw_mnl = f_nl * D_nm * (S2_pos + S2_neg)
            
                S_mn_pos = 1.0 / (E_m_2 - E_n_2 - Omega - 1j * self.eta)
                S_mn_neg = 1.0 / (E_m_2 - E_n_2 + Omega - 1j * self.eta)
                Gw_mn_2  = f_mn_2 * (S_mn_pos + S_mn_neg)
                G_mn_2= f_mn_2 * D_mn_2

                for c in range(3):
                    M_c_mn = M_K[..., c, :, :]
                    for a in range(3):
                        v_a = self.velocity_K[..., a, :, :]
                        for b in range(3):
                            v_b = self.velocity_K[..., b, :, :]
                            w_ab = self.w_K[..., a, b, :, :]
                            
                            # =======================================================
                            # 1. Three-Point Bubble: Tr[ M^c v^a v^b ]
                            # =======================================================
                            # Einsum maps: v^c_{mn} * v^a_{nl} * v^b_{lm}
                            Mvv = np.einsum('ijkmn, ijknl, ijklm -> ijkmnl', M_c_mn, v_a, v_b)
                            I3 = 2* np.sum(Mvv * Gw_mnl, axis=(-3, -2, -1))
                            
                            # =======================================================
                            # 2. Two-Point Bubble A (Diamagnetic Perturbation): Tr[ M^c w^{ab} ]
                            # =======================================================
                            rho_w = w_ab * G_mn_2
                            I2A = np.einsum('ijkmn, ijknm -> ijk', M_c_mn, rho_w)

                            # =======================================================
                            # Total Summation
                            # =======================================================
                            sigma_total_k = (I3 + I2A) / Omega**2
                            
                            
                            self.chi_IFE[ifermi, iomega, c, a, b] = np.sum(sigma_total_k) / np.prod(self.kgrid)

        self.projected_sigma = self.chi_IFE[None,None,None,...]
        if self.verbose == 2: showElapsedTime(extraInfo=f"Velocity Gauge IFE finished")

class InverseFaradayEffect_TDSE(Velocity, Spin, Orbital):
    def __init__(self, ife_type="total", velocityProjectionList=[], E0=0.001, n_steps_per_cycle=100, include_offdiag=True, **kwarg) -> None:
        super().__init__(velocityProjectionList=velocityProjectionList, highOrderVelocity=True, **kwarg)
        
        M_K = np.zeros(self.kgrid + (3, self.model.norbs, self.model.norbs), dtype=complex)
        if ife_type in ["total", "spin"]: M_K += self.spin_K
        if ife_type in ["total", "orbital"]: M_K += self.orbital_K

        self.chi_TDSE = np.zeros((len(self.fermiList), len(self.omegaList), 3, 3, 3), dtype=float)
        self.chi_TDSE_symmetric = np.zeros_like(self.chi_TDSE, dtype=float)
        self.chi_TDSE_quadrature = np.zeros_like(self.chi_TDSE, dtype=float)
        
        H_diag = np.zeros(self.energies.shape + (self.model.norbs,), dtype=complex)
        for i in range(self.model.norbs):
            H_diag[..., i, i] = self.energies[..., i]

        for ifermi, fermi in enumerate(self.fermiList):
            f = np.int8(self.energies < fermi)
            rho0 = np.zeros_like(H_diag)
            for i in range(self.model.norbs):
                rho0[..., i, i] = f[..., i]

            # Calculate Unperturbed Ground State Magnetization M0
            M0 = np.zeros(3)
            for ic in range(3):
                M_val0 = np.sum(rho0 * M_K[..., ic, :, :].swapaxes(-1, -2), axis=(-1,-2)).real
                M0[ic] = np.sum(M_val0)

            for iomega, Omega in enumerate(self.omegaList):
                T = 2 * np.pi / Omega
                dt = T / n_steps_per_cycle
                
                n_cycles_transient = int(np.ceil((10.0 / self.eta) / T))
                n_cycles_measure = 1

                def run_drive(directions, phases):
                    rho = rho0.copy()
                    t = 0.0
                    for _ in range(n_cycles_transient * n_steps_per_cycle):
                        rho = self._rk4_step(rho, t, dt, H_diag, directions, phases, Omega, rho0, E0)
                        t += dt
                        
                    M_dc = np.zeros(3)
                    for _ in range(n_cycles_measure * n_steps_per_cycle):
                        rho = self._rk4_step(rho, t, dt, H_diag, directions, phases, Omega, rho0, E0)
                        t += dt

                        for ic in range(3):
                            M_val = np.sum(rho * M_K[..., ic, :, :].swapaxes(-1, -2), axis=(-1,-2)).real
                            M_dc[ic] += np.sum(M_val) / (n_cycles_measure * n_steps_per_cycle)

                    return (M_dc - M0) / E0**2 / np.prod(self.kgrid)

                diagonal_response = np.zeros((3, 3))
                for ialpha in range(3): 
                    response = run_drive((ialpha,), (0.0,))
                    diagonal_response[ialpha] = response
                    self.chi_TDSE[ifermi, iomega, :, ialpha, ialpha] = response
                    self.chi_TDSE_symmetric[ifermi, iomega, :, ialpha, ialpha] = response

                if include_offdiag:
                    for ialpha in range(3):
                        for ibeta in range(ialpha + 1, 3):
                            same_phase = run_drive((ialpha, ibeta), (0.0, 0.0))
                            cross_symmetric = same_phase - diagonal_response[ialpha] - diagonal_response[ibeta]
                            
                            quadrature_phase = run_drive((ialpha, ibeta), (0.0, -np.pi / 2))
                            cross_quadrature = quadrature_phase - diagonal_response[ialpha] - diagonal_response[ibeta]
                            
                            self.chi_TDSE_symmetric[ifermi, iomega, :, ialpha, ibeta] = cross_symmetric
                            self.chi_TDSE_symmetric[ifermi, iomega, :, ibeta, ialpha] = cross_symmetric
                            self.chi_TDSE_quadrature[ifermi, iomega, :, ialpha, ibeta] = cross_quadrature
                            self.chi_TDSE_quadrature[ifermi, iomega, :, ibeta, ialpha] = -cross_quadrature
                            
                            # Same-phase driving gives Re[chi_ab + chi_ba]. With the
                            # beta field shifted by -pi/2, the real DC signal gives
                            # the phase-sensitive circular cross response.
                            self.chi_TDSE[ifermi, iomega, :, ialpha, ibeta] = 0.5 * (cross_symmetric - cross_quadrature)
                            self.chi_TDSE[ifermi, iomega, :, ibeta, ialpha] = 0.5 * (cross_symmetric + cross_quadrature)
        
        self.projected_sigma = self.chi_TDSE[None,None,None,...]
        if self.verbose == 2: showElapsedTime(extraInfo=f"TDSE IFE finished")

    def _rk4_step(self, rho, t, dt, H_diag, directions, phases, Omega, rho0, E0):
        v_drives = [self.velocity_K[..., direction, :, :] for direction in directions]
        w_drives = [[self.w_K[..., direction_i, direction_j, :, :]
                     for direction_j in directions] for direction_i in directions]
        
        def dRho_dt(rho_state, time_val):
            A_t = [-2 * E0 / Omega * np.sin(Omega * time_val + phase) for phase in phases]
            H_t = H_diag
            for i, A_i in enumerate(A_t):
                H_t = H_t - A_i * v_drives[i]
            for i, A_i in enumerate(A_t):
                for j, A_j in enumerate(A_t):
                    H_t = H_t + 0.5 * A_i * A_j * w_drives[i][j]
            comm = H_t @ rho_state - rho_state @ H_t
            return -1j * comm - self.eta * (rho_state - rho0)

        k1 = dRho_dt(rho, t)
        k2 = dRho_dt(rho + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = dRho_dt(rho + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = dRho_dt(rho + dt * k3, t + dt)
        return rho + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


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
