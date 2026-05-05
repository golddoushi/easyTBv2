#
# The module is for calculations on energy, berry curv., and several types of conductances along a given k-line path
#
from multiprocessing import Pool
from system import System
import numpy as np
from auxiliary import showElapsedTime, print_matrix
from scipy.constants import elementary_charge, hbar, angstrom

class CalculatorOnPath():

    def __init__(self,**kwarg) -> None:
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
        defaultSettings={
            "model":None,
            "KPOINTS":"./KPOINTS",
            "eta":0.026,
            "kBT":0.026,
            "smr_type":"Lorentzian",
            "degen":1e-6,
            "verbose":0,
        }
        defaultSettings.update(kwarg)
        
        self.model=defaultSettings["model"]
        self.eta=defaultSettings["eta"]
        self.smr_type=defaultSettings["smr_type"]
        self.degen=defaultSettings["degen"]
        self.verbose=defaultSettings["verbose"]
        self.KPOINTS=defaultSettings["KPOINTS"]

        if self.verbose==2: print("initializing calculator module")
        with open(self.KPOINTS,"r") as fin:
            data=fin.read().split("\n")
        self.nkpt=int(data[1].split()[0])
        self.kpath=np.loadtxt(data[3:3+self.nkpt])[:,:3]

class EnergyOnPath(CalculatorOnPath):
    def __init__(self, **kwarg) -> None:
        '''
        Solve the eigen-eqs. to obtain eigen-vals. and vecs.

        Usually the second father of all calculators

        Indexing:
        self.energies[ik, ieig] gives eigen-vals.

        self.vectors[ik1, ieig, jeig] gives eigen-vecs. in ao basis.
        '''
        super().__init__(**kwarg)
        self.ibmin=0             # ignore low-lying bands
        self.ibmax=self.model.norbs   # ignore high bands
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
        if self.verbose==2: showElapsedTime(extraInfo=f"Normal FT energy_R to energy_K and O_R to O_K on path of {self.nkpt=} finished")

        overlapEigValues, Wk =solveOverlap(self.O_K_AO)
        assert overlapEigValues.min() > 1e-6, f"eigen-value of overlap matrix is too small or negative {overlapEigValues.min()=}"
        if self.verbose==2: print(f"{overlapEigValues.min()=}")

        WkLam=Wk/np.sqrt(overlapEigValues[...,None,:])
        if self.verbose==2: showElapsedTime(extraInfo=f"WkLam")
        EnergyMatrixInK_tilde=WkLam.conj().transpose((0,2,1,)) @ self.E_K_AO @ WkLam
        if self.verbose==2: showElapsedTime(extraInfo=f"Overlapped energy")
        self.energies, vecsBeforeOverlap = np.linalg.eigh(EnergyMatrixInK_tilde)
        if self.verbose==2: showElapsedTime(extraInfo=f"general eigen-eqs.")
        self.vectors = WkLam @ vecsBeforeOverlap
        if self.verbose==2: showElapsedTime(extraInfo=f"overlapping on eig.-vecs.")

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
    def __init__(self, **kwarg) -> None:
        '''
        Compute velocity matrix in original basis and eigen-basis

        Indexing:
        self.RHam_K[ik, idirection, iorb, jorb] gives velocity matrix in original basis,
        idirection can be 0 | 1 | 2 corresponding to x | y | z direction.

        self.velocity_K[ik, idirection, ieig, jeig] gives velocity matrix in eigen-basis.
        idirection can be 0 | 1 | 2 corresponding to x | y | z direction.
        '''
        super().__init__(**kwarg)
        
        self.RHam_K=np.zeros((self.nkpt,3,self.model.norbs,self.model.norbs),dtype=complex)
        for ikpt, kpt in enumerate(self.kpath):
            for Rvec, RHamR in zip(self.model.iRvec,self.model.RHam_R):
                phase=np.exp(2j*np.pi*kpt@Rvec)
                self.RHam_K[ikpt]+=1j*phase*RHamR

        if self.verbose==2: showElapsedTime(extraInfo=f"Exp. sum. RHam_R to RHam_K to calc. velocity in ao-basis finished")
        vectors=self.vectors[:,None,:,self.ibmin:]
        self.velocity_K=vectors.conj().transpose((0,1,3,2)) @ self.RHam_K @ vectors
        if self.verbose==2: showElapsedTime(extraInfo=f"velocity in eig.-basis finished")

        if self.model.overlap:
            self.RO_K=np.zeros((self.nkpt,3,self.model.norbs,self.model.norbs),dtype=complex)
            for ikpt, kpt in enumerate(self.kpath):
                for Rvec, RO_R in zip(self.model.iRvec,self.model.RO_R):
                    phase=np.exp(2j*np.pi*kpt@Rvec)
                    self.RO_K[ikpt]+=1j*phase*RO_R

            self.velocity_K -= vectors.conj().transpose((0,1,3,2)) @ self.RO_K @ (vectors *self.energies[...,None,None,self.ibmin:])

class BerryCurvOnPath(VelocityOnPath):
    
    def __init__(self, **kwarg) -> None:
        '''
        Calc. Berry curvature \Omega^{alpha}_{k,mn}.

        alpha can be x, y, z corresponds the (cart.) space direction, k denote the crystal moment, m, n index the bands.

        Indexing: BerryCurv[ik1, ik2, ik3, idrection, iband, jband]
        '''
        super().__init__(**kwarg)
        self.calcDeltaEInverse()
        dEInverse2 = self.deltaEInverse**2
        
        self.BerryCurv=np.zeros((self.nkpt,3,self.model.norbs-self.ibmin))
        eps=np.arange(3)
        for ialpha in range(3):
            self.BerryCurv[...,ialpha,:]=-2*np.diagonal(self.velocity_K[...,eps[(ialpha+1)%3],:,-4:] @ (dEInverse2[...,-4:,:]*self.velocity_K[...,eps[(ialpha+2)%3],-4:,:]),0,-2,-1).imag
        if self.verbose==2: showElapsedTime(extraInfo=f"Berry curvature finished")

class SpinExpectationOnPath(EnergyOnPath):
    def __init__(self, **kwarg):
        '''
        Compute spin matrix in eigen-basis

        Indexing
        ---------
        - self.spin_eig[ik, igamma, ieig, jeig] gives spin matrix in eigen-basis. 
          igamma can be 0 | 1 | 2 | 3, corresponding to sigma^x | sigma^y | sigma^z | sigma^0 spin-polarization. 
        '''
        super().__init__(**kwarg)
        vectors=self.vectors[:,None,:,self.ibmin:]
        if self.model.overlap:
            self.spin_eig=vectors.conj().swapaxes(-2,-1) @ self.model.spin[None,:,:,:] @ self.O_K_AO[...,None,:,:] @ vectors
        else:
            self.spin_eig=vectors.conj().swapaxes(-2,-1) @ self.model.spin[None,:,:,:] @ vectors
        if self.verbose==2: showElapsedTime(extraInfo=f"spin op. in eig.-basis calculated")

class OrbitalExpectationOnPath(EnergyOnPath):
    def __init__(self, **kwarg):
        '''
        Compute orbital matrix in eigen-basis

        Indexing
        --------
        - self.orbital_eig[ik, igamma, ieig, jeig] gives orbital matrix in eigen-basis. 
          igamma can be 0 | 1 | 2 | 3, corresponding to L^x | L^y | L^z | L^0 orbital-polarization. 
        '''
        super().__init__(**kwarg)
        vectors=self.vectors[:,None,:,self.ibmin:]
        if self.model.overlap:
            self.orbital_eig=vectors.conj().swapaxes(-2,-1) @ self.model.orbital[None,:,:,:] @ self.O_K_AO[...,None,:,:] @ vectors
        else:
            self.orbital_eig=vectors.conj().swapaxes(-2,-1) @ self.model.orbital[None,:,:,:] @ vectors
        if self.verbose==2: showElapsedTime(extraInfo=f"orbital op. in eig.-basis calculated")

class SpinOrbitalExpectationOnPath(SpinExpectationOnPath, OrbitalExpectationOnPath):
    def __init__(self, **kwarg):
        '''
        Compute spin and orbital matrix in eigen-basis

        Indexing
        --------
        - self.spin_eig[ik, igamma, ieig, jeig] gives spin matrix in eigen-basis.
          igamma can be 0 | 1 | 2 | 3, corresponding to S^x | S^y | S^z | S^0 orbital-polarization. 

        - self.orbital_eig[ik, igamma, ieig, jeig] gives orbital matrix in eigen-basis. 
          igamma can be 0 | 1 | 2 | 3, corresponding to L^x | L^y | L^z | L^0 orbital-polarization. 
        '''
        super().__init__(**kwarg)

class SpinVelocityOnPath(VelocityOnPath):
    def __init__(self, **kwarg) -> None:
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
        super().__init__(**kwarg)
        self.Iodd = True
        #model.getSpinMatrice()
        #showElapsedTime(extraInfo=f"spin op. in ao.-basis constructed")
        vectors=self.vectors[:,None,:,self.ibmin:]
        if self.model.overlap:
            spin_eig=vectors.conj().transpose((0,1,3,2)) @ self.model.spin[None,:,:,:] @ self.O_K_AO[...,None,:,:] @ vectors
        else:
            spin_eig=vectors.conj().transpose((0,1,3,2)) @ self.model.spin[None,:,:,:] @ vectors
        if self.verbose==2: showElapsedTime(extraInfo=f"spin op. in eig.-basis calculated")
        spin=spin_eig[...,None,:,:,:]
        velocity=self.velocity_K[...,None,:,:]
        sV=spin @ velocity
        self.spinVelocity_K = (sV + sV.transpose((0,1,2,4,3)).conj())/2#(spin @ velocity + velocity @ spin)/2
        if self.verbose==2: showElapsedTime(extraInfo=f"spin velocity in eig.-basis finished")

class SpinBerryCurvOnPath(SpinVelocityOnPath):
    def __init__(self, **kwarg) -> None:
        '''
        Calc. spin Berry curvature \Omega^{alpha,gamma}_{k,mn}.

        alpha can be x, y, z, denoting the (cart.) space direction. 
        gammma can be sigma^x, sigma^y, sigma^z, sigma^0, corresponding to the spin-polarization. 
        k denotes the crystal moment, m, n index the bands.

        Indexing: spinBerryCurv[ik, ialpha, igamma, ieig]
        '''
        super().__init__(kwarg)
        self.calcDeltaEInverse()
        dEInverse2 = self.deltaEInverse**2

        self.spinBerryCurv=np.zeros((self.nkpt,3,3,3,self.model.norbs-self.ibmin))
        #eps=np.arange(3)
        for ikpt in range(self.nkpt):
            nvalence=sum(self.energies[ikpt]<=self.model.fermi)
            for ialpha in range(3):
                for ibeta in range(3):
                    for igamma in range(3):
                        self.spinBerryCurv[ikpt,ialpha,ibeta,igamma,:nvalence]=-2*np.diagonal(self.spinVelocity_K[ikpt,ibeta,igamma,:nvalence,nvalence:] @ (dEInverse2[ikpt,nvalence:,:nvalence]*self.velocity_K[ikpt,ialpha,nvalence:,:nvalence]),0,-2,-1).imag
        if self.verbose==2: showElapsedTime(extraInfo=f"Spin Berry curvature finished")

# --- TOP LEVEL WORKER INFRASTRUCTURE ---
# This must be outside the class so Python can use shared memory for the global parameters
_shared_context = {}

def init_worker(shared_data):
    """
    Initializer function for multiprocessing.Pool.
    Ensures that Windows (spawn) workers get the shared context.
    """
    global _shared_context
    _shared_context = shared_data

def _parallel_worker_for_surface_green_compute(param):
    ikpt, kpt = param
    
    # Instantly access massive matrices via Linux Copy-on-Write memory sharing (Zero pickling overhead)
    ctx = _shared_context
    norbs = ctx['norbs']
    omega = ctx['omega']
    n_iter = ctx['n_iter']
    I = np.eye(norbs, dtype=complex)
    
    G_bulk_k = np.zeros((len(omega), norbs, norbs), dtype=complex)
    G_surf_L_k = np.zeros((len(omega), norbs, norbs), dtype=complex)
    G_surf_R_k = np.zeros((len(omega), norbs, norbs), dtype=complex)

    # 1. Fourier Transform the Principal Layer Blocks
    H_00_k = np.zeros((norbs, norbs), dtype=complex)
    for R, HR in zip(ctx['R_00'], ctx['H_00_R']):
        H_00_k += HR * np.exp(2j * np.pi * np.dot(kpt, R))
            
    H_01_k = np.zeros((norbs, norbs), dtype=complex)
    for R, HR in zip(ctx['R_01'], ctx['H_01_R']):
        H_01_k += HR * np.exp(2j * np.pi * np.dot(kpt, R))
            
    H_m10_k = np.zeros((norbs, norbs), dtype=complex)
    for R, HR in zip(ctx['R_m10'], ctx['H_m10_R']):
        H_m10_k += HR * np.exp(2j * np.pi * np.dot(kpt, R))

    # 2. Iterative Decimation for each complex frequency
    for iomega, w in enumerate(omega):
        alpha = H_01_k.copy()
        beta = H_m10_k.copy()
        eps = H_00_k.copy()
        eps_s_L = H_00_k.copy()
        eps_s_R = H_00_k.copy()
            
        for i in range(n_iter):
            # Compute the propagator: g = (w*I - eps)^-1
            g = np.linalg.inv(w * I - eps)
            # Precompute heavy matrix multiplications
            a_g = alpha @ g
            b_g = beta @ g
            
            # Compute off-diagonal effective couplings
            a_next = a_g @ alpha
            b_next = b_g @ beta
            
            # Compute effective diagonal shifts
            a_g_b = a_g @ beta
            b_g_a = b_g @ alpha
            
            # Update local density potentials
            eps_s_L = eps_s_L + a_g_b
            eps_s_R = eps_s_R + b_g_a
            eps = eps + a_g_b + b_g_a
            
            alpha = a_next
            beta = b_next
                
        # 3. Final Matrix Inversions
        G_bulk_k[iomega]   = np.linalg.inv(w * I - eps)
        G_surf_L_k[iomega] = np.linalg.inv(w * I - eps_s_L)
        G_surf_R_k[iomega] = np.linalg.inv(w * I - eps_s_R)
        
    return ikpt, G_bulk_k, G_surf_L_k, G_surf_R_k

class SurfaceBulkGreenOnPath(CalculatorOnPath):
    def __init__(self, surfaceDir: int, omega:np.ndarray|list, n_iter:int=10, ncores:int=1, **kwarg):
        """
        Computes the Left Surface, Right Surface, and Bulk Green's Functions via the
        iterative decimation algorithm (Lopez Sancho et al. 1985).
        
        Parameters
        ----------
        model : System
            The tight-binding model (must have called constructPrincipleLayer first).
        surfaceDir : int
            The normal direction of the surface (0, 1, or 2).
        omega : list or np.ndarray
            Array of complex frequencies to evaluate.
        n_iter : int
            Number of decimation iterations. n_iter = n simulates a slab of 2^n layers.
        ncores : int
            Number of cores used for the parallel computations
        """
        super().__init__(**kwarg)
        self.surfaceDir = surfaceDir
        self.omega = np.array(omega, dtype=complex)
        self.n_iter = n_iter
        self.ncores=ncores
        
        assert hasattr(self.model, 'H_00_R'), "You must call model.constructPrincipleLayer(surfaceDir) before GreenOnPath!"
        
        # 1. Enforce open boundary conditions in k-space
        for ikpt, kpt in enumerate(self.kpath):
            if abs(kpt[surfaceDir]) > 1e-6:
                if self.verbose == 2:
                    print(f"Warning: k-point {kpt} has non-zero component in surface normal direction {surfaceDir}. Forcing to 0.")
                self.kpath[ikpt, surfaceDir] = 0.0
                
        self.nomega = len(self.omega)
        norbs = self.model.norbs
        
        self.G_bulk = np.zeros((self.nkpt, self.nomega, norbs, norbs), dtype=complex)
        self.G_surf_L = np.zeros((self.nkpt, self.nomega, norbs, norbs), dtype=complex)
        self.G_surf_R = np.zeros((self.nkpt, self.nomega, norbs, norbs), dtype=complex)
        
        if self.verbose: print(f"\n--- Computing Green's Functions ({self.nkpt} k-points, {self.nomega} frequencies, {self.n_iter} iterations) ---")
        if self.verbose == 2: showElapsedTime(offset=True)

    def setOmega(self, omega:np.ndarray|list, verbose=True):
        self.omega=omega
        self.nomega = len(self.omega)
        if verbose: print(f"Frequencies reset. Now {self.nomega} points are involved")

    def solveGreensFunction(self, verbose=True):
        global _shared_context
        
        # Pack the required data into the global dictionary before forking.
        # This takes 0 seconds and guarantees all child processes can read it natively.
        _shared_context = {
            'norbs': self.model.norbs,
            'omega': self.omega,
            'n_iter': self.n_iter,
            'R_00': self.model.R_00,
            'H_00_R': self.model.H_00_R,
            'R_01': self.model.R_01,
            'H_01_R': self.model.H_01_R,
            'R_m10': self.model.R_m10,
            'H_m10_R': self.model.H_m10_R,
        }

        paramList = [(ikpt, kpt) for ikpt, kpt in enumerate(self.kpath)]
        ncomplete=0

        if self.ncores == 1:
            for param in paramList: 
                ikpt, Gb, GsL, GsR = _parallel_worker_for_surface_green_compute(param)
                self.G_bulk[ikpt] = Gb
                self.G_surf_L[ikpt] = GsL
                self.G_surf_R[ikpt] = GsR
                ncomplete+=1
                if verbose: showElapsedTime(offset=False,extraInfo=f"{ncomplete}/{len(paramList):3d} finished")
        else:
            with Pool(processes=self.ncores, initializer=init_worker, initargs=(_shared_context,)) as pool:
                for ikpt, Gb, GsL, GsR in pool.imap_unordered(_parallel_worker_for_surface_green_compute, paramList): 
                    self.G_bulk[ikpt] = Gb
                    self.G_surf_L[ikpt] = GsL
                    self.G_surf_R[ikpt] = GsR
                    ncomplete+=1
                    if ncomplete % self.ncores == 1 and verbose: showElapsedTime(offset=False,extraInfo=f"{ncomplete}/{len(paramList):3d} finished")

        # Clear the memory so it doesn't persist unnecessarily
        _shared_context.clear()

        if verbose: showElapsedTime(extraInfo="Surface Green's function computation finished")

class SurfaceBulkSpectraOnPath(SurfaceBulkGreenOnPath):
    def __init__(self, surfaceDir: int, fermi: np.ndarray, n_iter:int=10, ncores:int=1, **kwarg):
        """
        Computes the Left Surface, Right Surface, and Bulk Green's Functions via the
        highly convergent iterative decimation algorithm (Lopez Sancho et al. 1985).
        
        Parameters
        ----------
        model : System
            The tight-binding model (must have called constructPrincipleLayer first).
        surfaceDir : int
            The normal direction of the surface (0, 1, or 2).
        fermi : np.ndarray
            Array of real energies to evaluate.
        n_iter : int
            Number of decimation iterations. n_iter = n simulates a slab of 2^n layers.
        """
        super().__init__(surfaceDir=surfaceDir, omega=fermi, n_iter=n_iter, ncores=ncores, **kwarg)
        self.setOmega(fermi-1j*self.eta)

        self.solveGreensFunction()
        self.A_bulk   = np.sum(np.diagonal(self.G_bulk,  axis1=-2,axis2=-1),axis=-1).imag/np.pi
        self.A_surf_L = np.sum(np.diagonal(self.G_surf_L,axis1=-2,axis2=-1),axis=-1).imag/np.pi
        self.A_surf_R = np.sum(np.diagonal(self.G_surf_R,axis1=-2,axis2=-1),axis=-1).imag/np.pi