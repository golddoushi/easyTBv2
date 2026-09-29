"""Electronic computation stages used by explicit preparation and Grid entry points.

These builders contain no response solvers, collision settings or phonon models.
"""

import numpy as np
from system import System
from auxiliary import showElapsedTime, projectionOperator, PauliBlock, fermiDiracOccupations
from scipy.constants import elementary_charge, electron_mass, hbar, angstrom
from .electronic_algebra import ElectronicAlgebra
from .electronic import resolve_states, states_from_stage, vertices_from_stage, restore_states, restore_vertices

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
            Broadening for calculators using eta directly, usually eV. 
            NOTE: It does not select or configure a collision operator for DM 
            responses.
        - kBT  : float
            Kinetic energy of electron, in the same unit with energy usually eV
        - smr_type : str, "Lorentzian"|"Gaussian"
        - degen : float
            The threthold two states are treated to be degenerated
        - skip_low : bool
            To skip lowest and seperated eigen-states IN ALL CASES
        - kOffset : array-like, shape (3,)
            Fractional reciprocal-space offset of the uniform FFT mesh. The
            Energy stage derives its real-space phase factors from this value.
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

            "kOffset":(0.,0.,0.),

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

        offset=defaultSettings["kOffset"]
        self.kOffset=np.zeros(3) if offset is None else np.asarray(offset,dtype=float)
        if self.kOffset.shape != (3,) or not np.all(np.isfinite(self.kOffset)):
            raise ValueError("kOffset must contain three finite fractional coordinates")
        self.fermiList=defaultSettings["fermiList"]
        self.omegaList=defaultSettings["omegaList"]
        self.greenFuncOmegaBatch=max(1, int(defaultSettings["greenFuncOmegaBatch"]))
        self.greenFuncNWorkers=max(1, int(defaultSettings["greenFuncNWorkers"]))

        self.verbose=defaultSettings["verbose"]
        if self.verbose==2: print("initializing calculator module finished")

class Energy(Calculator, ElectronicAlgebra):
    def __init__(self, *, states=None, **kwarg) -> None:
        '''
        Solve the eigen-eqs. to obtain eigen-vals. and vecs.

        Usually the second father of all calculators

        Indexing:
        self.energies[ik1, ik2, ik3, ieig] gives eigen-vals.

        self.vectors[ik1, ik2, ik3, ieig, jeig] gives eigen-vecs. in ao basis.
        '''
        super().__init__(**resolve_states(states, kwarg))
        if states is not None:
            self.states = states
            restore_states(self, states)
            return
        self.phaseFactorR=np.exp(2j*np.pi*(self.model.iRvec @ self.kOffset))
        self.ibmin=0             # ignore low-lying bands
        if self.verbose==2: showElapsedTime(offset=True)
        
        if self.model.overlap:
            self.solveGeneralEigenEquation()
        else:
            self.solveEigenEquation()
        self.states = states_from_stage(self)

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

        E_R_AO=self.model.energy_R*self.phaseFactorR[:,None,None]
        O_R_AO=self.model.overlap_R*self.phaseFactorR[:,None,None]
        
        self.E_R_intepolated=np.zeros(self.kgrid+(self.model.norbs,self.model.norbs),dtype=complex)
        self.O_R_intepolated=np.zeros(self.kgrid+(self.model.norbs,self.model.norbs),dtype=complex)
        for iR, Rvec in enumerate(self.model.iRvec):
            index = tuple(np.asarray(Rvec) % self.kgrid)
            self.E_R_intepolated[index] += E_R_AO[iR]
            self.O_R_intepolated[index] += O_R_AO[iR]
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

        ham_R stores signed Hamiltonian matrix elements. The inverse FFT evaluates 
        H(k)=sum_R ham_R(R)*exp(+2*pi*i*k.R); no extra hopping minus sign is applied. 
        A model writing H_ij=-t_ij must store -t_ij in the hamR file.
        '''
        HH_R_W=self.model.ham_R*self.phaseFactorR[:,None,None]
        self.ham_R_intepolated=np.zeros(self.kgrid+(self.model.norbs,self.model.norbs),dtype=complex)
        for iR, Rvec in enumerate(self.model.iRvec):
            # Distinct R blocks can alias on a small/shifted mesh. Their
            # (already phase-shifted) contributions must all enter the sum.
            self.ham_R_intepolated[tuple(np.asarray(Rvec) % self.kgrid)] += HH_R_W[iR]
        self.ham_K=np.fft.ifftn(self.ham_R_intepolated,axes=(0,1,2,))*np.prod(self.kgrid)
        if self.verbose==2: showElapsedTime(extraInfo=f"FFT ham_R to ham_K on {self.kgrid=} finished")
        self.energies, self.vectors = np.linalg.eigh(self.ham_K)
        if self.verbose==2: showElapsedTime(extraInfo=f"eig.-eqs. solved")

        if self.skip_low: self.getEnergyWindow()
    
    




    
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

class Velocity(Energy):
    def __init__(self, velocityProjectionList=(), highOrderVelocity=False, exchangeVelocity=False,
                 *, electronic=None, orderedVelocityDerivative=False, **kwarg) -> None:
        '''
        Compute velocity matrix in original basis and eigen-basis
        And the velocity projected in specific basis

        orderedVelocityDerivative also builds ordered_velocity_derivative_K
        with components D_b V_a, using Hamiltonian derivatives through order
        two. highOrderVelocity additionally builds the third-order vertex.

        Indexing:
        self.RHam_K[ik1, ik2, ik3, idirection, iorb, jorb] gives velocity matrix in original basis,
        idirection can be 0 | 1 | 2 corresponding to x | y | z direction.

        self.velocity_K[ik1, ik2, ik3, idirection, ieig, jeig] gives velocity matrix in eigen-basis.
        idirection can be 0 | 1 | 2 corresponding to x | y | z direction.

        self.projected_velocity_K_list[iproj] gives each projected velocity matrix
        '''
        self.exchangeVelocity=exchangeVelocity
        highOrderVelocity=bool(highOrderVelocity or exchangeVelocity)
        orderedVelocityDerivative=bool(orderedVelocityDerivative or highOrderVelocity)
        if electronic is not None:
            kwarg['states'] = electronic.states
        super().__init__(**kwarg)
        if (electronic is not None
                and (not highOrderVelocity or (electronic.w_K is not None and electronic.u_K is not None))
                and (not orderedVelocityDerivative or electronic.ordered_velocity_derivative_K is not None)
                and (not exchangeVelocity or electronic.exchangeVelocity)
                and tuple(velocityProjectionList) == electronic.velocityProjectionList):
            self.electronic = electronic
            restore_vertices(self, electronic)
            return

        RH_R_W=self.model.RHam_R*self.phaseFactorR[:,None,None,None]
        self.RHam_R_intepolated=np.zeros(self.kgrid+(3,self.model.norbs,self.model.norbs),dtype=complex)
        AA_R_W=self.model.AA_R_W*self.phaseFactorR[:,None,None,None] # Berry connection in local gauge
        AA_R_W_intepolated=np.zeros(self.kgrid+(3,self.model.norbs,self.model.norbs),dtype=complex)
        for iR, Rvec in enumerate(self.model.iRvec):
            index = tuple(np.asarray(Rvec) % self.kgrid)
            self.RHam_R_intepolated[index]+=RH_R_W[iR]
            AA_R_W_intepolated[index]+=AA_R_W[iR]
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
        # The position/Berry-connection operator is Hermitian. Small violations
        # from the finite Wannier interpolation are strongly amplified by its
        # k derivatives and would otherwise make the contact vertices acausal.
        AA_K_H = (AA_K_H + AA_K_H.conj().swapaxes(-2,-1)) / 2
        self.connection_K = AA_K_H
        AA_K_H_dot_H = AA_K_H * self.energies[...,None,None,:]            # gauge correction term, the commutator [AA_K, Ham.]
        self.velocity_K = velocity_bare_K - 1j*(AA_K_H_dot_H - AA_K_H_dot_H.conj().swapaxes(-2,-1)) # gauge-covariant velocity

        if exchangeVelocity:
            assert self.model.exchange_R is not None, "exchangeVelocity requires model.exchange_R; call model.getExchangeR first"
            assert self.model.RExchange_R is not None, "RExchange_R is required; call model.getRExchange_R first"
            assert self.model.RRExchange_R is not None, "RRExchange_R is required; call model.getRRExchange_R first"

            exchange_R_W=self.model.exchange_R*self.phaseFactorR[:,None,None]
            RExchange_R_W=self.model.RExchange_R*self.phaseFactorR[:,None,None,None]
            RRExchange_R_W=self.model.RRExchange_R*self.phaseFactorR[:,None,None,None,None]

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
            RO_R_W=RO_R_W*self.phaseFactorR[:,None,None,None]
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

        if orderedVelocityDerivative:
            RRH_R_W  = getattr(self.model, "RRHam_R", None)
            if highOrderVelocity:
                RRRH_R_W = getattr(self.model, "RRRHam_R", None)
            assert RRH_R_W is not None,  "RRHam_R is required for high order velocity,  execute getRRHam_R first"
            if highOrderVelocity:
                assert RRRH_R_W is not None, "RRRHam_R is required for high order velocity, execute getRRRHam_R first"

            RRH_R_intepolated  = np.zeros(self.kgrid + (3, 3, self.model.norbs, self.model.norbs), dtype=complex)
            if highOrderVelocity:
                RRRH_R_intepolated = np.zeros(self.kgrid + (3, 3, 3, self.model.norbs, self.model.norbs), dtype=complex)

            RRH_R_W  = RRH_R_W  * self.phaseFactorR[:, None, None, None, None]
            if highOrderVelocity:
                RRRH_R_W = RRRH_R_W * self.phaseFactorR[:, None, None, None, None, None]

            for iR, Rvec in enumerate(self.model.iRvec):
                index = tuple(np.asarray(Rvec) % self.kgrid)
                RRH_R_intepolated[index] += RRH_R_W[iR]
                if highOrderVelocity:
                    RRRH_R_intepolated[index] += RRRH_R_W[iR]

            # w_K = d^2H/dk^2
            RRHam_K  = -1 * np.fft.ifftn(RRH_R_intepolated, axes=(0,1,2,)) * np.prod(self.kgrid)
            vectors=self.vectors[:,:,:,None,None,:,:]
            self.w_K = vectors.conj().swapaxes(-2,-1) @ RRHam_K @ vectors

            if highOrderVelocity:
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
                r1_R_intepolated[tuple(np.asarray(Rvec) % self.kgrid)] += 1j * R_cart[:, None, None, None] * AA_R_W[iR][None, :, :, :]

            r1_K_AO = np.fft.ifftn(r1_R_intepolated, axes=(0, 1, 2)) * np.prod(self.kgrid)
            vectors1 = self.vectors[..., None, None, :, :]
            connectionDerivative1 = vectors1.conj().swapaxes(-2, -1) @ r1_K_AO @ vectors1
            connectionDerivative1 = (
                connectionDerivative1 + connectionDerivative1.conj().swapaxes(-2,-1)
            ) / 2
            del r1_R_intepolated, r1_K_AO, vectors1

            if highOrderVelocity:
                # Construct the larger second-derivative tensor only after the
                # first-derivative FFT buffers have been released.
                r2_R_intepolated = np.zeros(self.kgrid + (3, 3, 3, self.model.norbs, self.model.norbs), dtype=complex)
                for iR, Rvec in enumerate(self.model.iRvec):
                    R_cart = Rvec @ self.model.lattice
                    r2_R_intepolated[tuple(np.asarray(Rvec) % self.kgrid)] += (
                        -R_cart[:, None, None, None, None] * R_cart[None, :, None, None, None] * AA_R_W[iR][None, None, :, :, :]
                    )
                r2_K_AO = np.fft.ifftn(r2_R_intepolated, axes=(0, 1, 2)) * np.prod(self.kgrid)
                vectors2 = self.vectors[..., None, None, None, :, :]
                connectionDerivative2 = vectors2.conj().swapaxes(-2, -1) @ r2_K_AO @ vectors2
                connectionDerivative2 = (
                    connectionDerivative2 + connectionDerivative2.conj().swapaxes(-2,-1)
                ) / 2
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

            # Preserve ordered D_b V_a for differentiating observables.
            # Only the contact-current vertex w_ab is Cartesian symmetrized,
            # since it multiplies the commuting product A_a A_b.
            self.w_K = (self.w_K + self.w_K.conj().swapaxes(-2,-1)) / 2
            self.ordered_velocity_derivative_K = self.w_K.copy()
            self.w_K = (self.w_K + self.w_K.swapaxes(-4,-3)) / 2

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

            if highOrderVelocity:
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
                del connectionDerivative2
            del connectionDerivative1, partialV
            if self.verbose==2: showElapsedTime(extraInfo=f"higer order velocity modification with position operators finished")

        self.electronic = vertices_from_stage(self, velocityProjectionList)

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

    def spinCovariantDerivative(self, connection):
        """D_a mu_c for the local Wannier spin representation used by Spin.

        A nonlocal Spin implementation must also supply the ordinary derivative
        of its Wannier spin matrix here. Eigenvector derivatives are not taken.
        Axes: kgrid + (spin component, derivative direction, band, band).
        """
        spin = self.spin_K[..., :, None, :, :]
        connection = connection[..., None, :, :, :]
        return -1j*(connection @ spin-spin @ connection)

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

class Orbital_BlochSelfRotate(Velocity):
    def __init__(self, orbitalProjectionList=[], _computeOrbitalBlochSelfRotate=True,
                 orbitalContact=False, **kwarg):
        '''
        Compute the Bloch-state self-rotation
        orbital magnetic-moment matrix from Eq. S9 of PRL 136, 236302
        (2026).

        The velocity matrices used by this code are dH/dk = hbar*v_phys in
        eV*angstrom. ``orbital_K`` is returned in units of mu_B:

        orbital_K[ik1, ik2, ik3, idirection, iband, jband].
        '''
        ordered_derivative = bool(kwarg.pop("orderedVelocityDerivative", False) or orbitalContact)
        super().__init__(orderedVelocityDerivative=ordered_derivative, **kwarg)
        if not _computeOrbitalBlochSelfRotate:
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

        if orbitalContact:
            self.orbitalA_K = self.bloch_self_rotation_contact()
            self.projected_orbitalA_K_list = []
            connection = self.connection_K[..., None, :, :, :]
            for projection in orbitalProjectionList:
                projectors = [self.projectorInEigenBasis(projectionOperator(
                    projection[key], self.model.atomicOrbInfoList, self.model.norbs))
                    [..., None, None, :, :] for key in ('source', 'target')]
                ps, pt = projectors
                ds = -1j*(connection @ ps-ps @ connection)
                dt = -1j*(connection @ pt-pt @ connection)
                moment = self.orbital_K[..., :, None, :, :]
                self.projected_orbitalA_K_list.append((ds @ moment + ps @ self.orbitalA_K
                    + self.orbitalA_K @ pt + moment @ dt)/2)

        if self.verbose == 2:
            showElapsedTime(extraInfo="Bloch self-rotation magnetic moment in eig.-basis finished")

    def bloch_self_rotation_contact(self):
        """D_d m_c in mu_B*angstrom for normalized a=e*A/hbar.

        Differentiate the BSR sum in the instantaneous band frame, then remove
        [m,K], with K_nm=V_nm/(E_m-E_n). W_ad must be the ordered D_d V_a.
        The gap cutoff is held fixed: this retains the excluded-gap prescription,
        not a degenerate-subspace extension of the isolated-band formula.
        """
        if self.model.overlap or self.skip_low:
            raise NotImplementedError('BSR contacts require a complete orthogonal basis')
        e, v = self.energies, self.velocity_K
        gap = e[..., None, :] - e[..., :, None]
        inv = np.zeros_like(gap)
        np.divide(1., gap, out=inv, where=np.abs(gap) > self.degen)
        contact = np.zeros(self.kgrid+(3, 3, e.shape[-1], e.shape[-1]), dtype=complex)
        constant = hbar**2/(2*electron_mass*elementary_charge*angstrom**2)
        for d in range(3):
            rotation = v[..., d, :, :]*inv
            dv = self.ordered_velocity_derivative_K[..., :, d, :, :] + (
                v @ rotation[..., None, :, :] - rotation[..., None, :, :] @ v)
            de = np.diagonal(v[..., d, :, :], axis1=-2, axis2=-1).real
            dm = np.zeros_like(self.orbital_K)
            for p in range(e.shape[-1]):
                inverse = inv[..., :, p]  # 1/(E_p-E_n)
                mask = np.abs(e[..., p, None]-e) > self.degen
                valid = mask[..., :, None] & mask[..., None, :]
                weight = (inverse[..., :, None]+inverse[..., None, :])*valid
                dinverse = -(de[..., p, None]-de)*inverse**2
                dweight = (dinverse[..., :, None]+dinverse[..., None, :])*valid
                for c, (a, b) in enumerate(((1, 2), (2, 0), (0, 1))):
                    va = v[..., a, :, p][..., :, None]
                    vb = v[..., b, :, p][..., :, None]
                    av = v[..., a, p, :][..., None, :]
                    bv = v[..., b, p, :][..., None, :]
                    cross = va*bv-vb*av
                    dcross = (dv[..., a, :, p][..., :, None]*bv
                        + va*dv[..., b, p, :][..., None, :]
                        - dv[..., b, :, p][..., :, None]*av
                        - vb*dv[..., a, p, :][..., None, :])
                    dm[..., c, :, :] += dweight*cross+weight*dcross
            dm /= 4j*constant
            dm -= (self.orbital_K @ rotation[..., None, :, :]
                   - rotation[..., None, :, :] @ self.orbital_K)
            contact[..., d, :, :] = (dm+dm.conj().swapaxes(-1, -2))/2
        return contact

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

        With H(A) = H0 + A_a v_a + 1/2 A_a A_b w_ab,
        T(A) = -i[mu_S,H(A)] = T0 + A_a T_a + 1/2 A_a A_b T_ab.
        '''
        nbasis = self.energies.shape[-1]
        self.spinTorqueA_K = np.zeros(self.kgrid + (3, 3, nbasis, nbasis), dtype=complex)
        self.spinTorqueAA_K = np.zeros(self.kgrid + (3, 3, 3, nbasis, nbasis), dtype=complex)

        for c in range(3):
            S_c = self.spin_K[..., c, :, :]
            for a in range(3):
                v_a = self.velocity_K[..., a, :, :]
                self.spinTorqueA_K[..., c, a, :, :] = -1j * (S_c @ v_a - v_a @ S_c)
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
        '''Build -i[mu_S,r_a], the torque derivative for H_L=H0+E_a*r_a.'''
        nbasis = self.energies.shape[-1]
        self.spinTorqueE_K = np.zeros(self.kgrid + (3, 3, nbasis, nbasis), dtype=complex)
        for c in range(3):
            spin_c = self.spin_K[..., c, :, :]
            for a in range(3):
                r_a = self.r_K[..., a, :, :]
                self.spinTorqueE_K[..., c, a, :, :] = -1j * (spin_c @ r_a - r_a @ spin_c)

        self.projected_spinTorqueE_K_list = []
        for spinTorqueProjection in self.spinTorqueProjectionList:
            proj1 = projectionOperator(spinTorqueProjection['source'], self.model.atomicOrbInfoList, self.model.norbs)
            proj2 = projectionOperator(spinTorqueProjection['target'], self.model.atomicOrbInfoList, self.model.norbs)
            proj1_K = self.projectorInEigenBasis(proj1)[..., None, None, :, :]
            proj2_K = self.projectorInEigenBasis(proj2)[..., None, None, :, :]
            self.projected_spinTorqueE_K_list.append((proj1_K @ self.spinTorqueE_K + self.spinTorqueE_K @ proj2_K) / 2)
        if self.verbose==2: showElapsedTime(extraInfo="Dipole corrected spin torque in eig.-basis finished")

class SpinSOCTorque(Spin):
    """Electronic spin-moment torque -i[mu_S,H_SOC], in mu_B*eV.

    Consumes model.soc_R, registered during model construction (e.g. getSOCR
    or addRashbaSOC). No SOC decomposition or LLG reaction sign is applied
    here. Divide by hbar to obtain the electronic magnetic-moment rate.
    """
    def __init__(self, spinTorqueProjectionList=(), spinSOCContact=False, gS=2.0, **kwarg):
        super().__init__(gS=gS, **kwarg)
        if self.model.overlap or self.skip_low:
            raise NotImplementedError('Spin SOC torque requires a complete orthogonal basis')
        if self.model.soc_R is None:
            raise ValueError('Spin SOC torque requires model.soc_R; call getSOCR or add SOC first')
        soc = self.model.soc_R
        mesh = np.zeros(self.kgrid+soc.shape[-2:], dtype=complex)
        derivative = np.zeros(self.kgrid+(3, self.model.norbs, self.model.norbs),
                              dtype=complex) if spinSOCContact else None
        for i, r in enumerate(self.model.iRvec):
            index = tuple(np.asarray(r) % self.kgrid)
            value = soc[i]*self.phaseFactorR[i]
            mesh[index] += value
            if spinSOCContact:
                derivative[index] += 1j*(r @ self.model.lattice)[:, None, None]*value
        self.soc_K_AO = np.fft.ifftn(mesh, axes=(0, 1, 2))*np.prod(self.kgrid)
        u = self.vectors
        self.soc_K = u.conj().swapaxes(-1, -2) @ self.soc_K_AO @ u
        soc_band = self.soc_K[..., None, :, :]
        self.spinTorque_K = -1j*(self.spin_K @ soc_band-soc_band @ self.spin_K)
        if spinSOCContact:
            velocity = Velocity(states=self.states, electronic=kwarg.get('electronic'))
            aa = velocity.connection_K
            u1 = u[..., None, :, :]
            dsoc = u1.conj().swapaxes(-1, -2) @ (np.fft.ifftn(derivative, axes=(0, 1, 2))*np.prod(self.kgrid)) @ u1
            dsoc -= 1j*(aa @ soc_band-soc_band @ aa)
            # Product rule: D_a T_c = -i([D_a mu_c,H_SOC]+[mu_c,D_a H_SOC]).
            # Spin owns D_a mu_c; the torque stage makes no local-spin reduction.
            dspin = self.spinCovariantDerivative(aa)
            spin = self.spin_K[..., :, None, :, :]
            soc_band = self.soc_K[..., None, None, :, :]
            dsoc = dsoc[..., None, :, :, :]
            self.spinTorqueA_K = -1j*(dspin @ soc_band-soc_band @ dspin
                                     +spin @ dsoc-dsoc @ spin)
            connection = aa[..., None, :, :, :]
            moment = self.spinTorque_K[..., :, None, :, :]
        self.spinTorqueProjectionList = spinTorqueProjectionList
        self.projected_spinTorque_K_list = []
        self.projected_spinTorqueA_K_list = []
        for projection in spinTorqueProjectionList:
            ps, pt = [self.projectorInEigenBasis(projectionOperator(
                projection[key], self.model.atomicOrbInfoList, self.model.norbs))
                [..., None, :, :] for key in ('source', 'target')]
            self.projected_spinTorque_K_list.append(
                (ps @ self.spinTorque_K+self.spinTorque_K @ pt)/2)
            if spinSOCContact:
                ps, pt = ps[..., None, :, :], pt[..., None, :, :]
                ds = -1j*(connection @ ps-ps @ connection)
                dt = -1j*(connection @ pt-pt @ connection)
                self.projected_spinTorqueA_K_list.append((ds @ moment+ps @ self.spinTorqueA_K
                    +self.spinTorqueA_K @ pt+moment @ dt)/2)

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

        exchange_R=self.model.exchange_R*self.phaseFactorR[:,None,None]
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

        For H_XC(A)=H_XC+A_a v_XC,a+1/2 A_a A_b w_XC,ab, the vertices are
        -i[mu_S,v_XC,a] and -i[mu_S,w_XC,ab]. If exchangeVelocity was disabled,
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
                    self.spinExchangeTorqueA_K[...,c,a,:,:]=-1j*(spin_c@velocity_xc_a-velocity_xc_a@spin_c)
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

        With H(A) = H0 + A_a v_a + 1/2 A_a A_b w_ab,
        T(A) = -i[mu_L,H(A)] = T0 + A_a T_a + 1/2 A_a A_b T_ab.
        '''
        nbasis = self.energies.shape[-1]
        self.orbitalTorqueA_K = np.zeros(self.kgrid + (3, 3, nbasis, nbasis), dtype=complex)
        self.orbitalTorqueAA_K = np.zeros(self.kgrid + (3, 3, 3, nbasis, nbasis), dtype=complex)

        for c in range(3):
            L_c = self.orbital_K[..., c, :, :]
            for a in range(3):
                v_a = self.velocity_K[..., a, :, :]
                self.orbitalTorqueA_K[..., c, a, :, :] = -1j * (L_c @ v_a - v_a @ L_c)
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
        '''Build -i[mu_L,r_a], the torque derivative for H_L=H0+E_a*r_a.'''
        nbasis = self.energies.shape[-1]
        self.orbitalTorqueE_K = np.zeros(self.kgrid + (3, 3, nbasis, nbasis), dtype=complex)
        for c in range(3):
            orbital_c = self.orbital_K[..., c, :, :]
            for a in range(3):
                r_a = self.r_K[..., a, :, :]
                self.orbitalTorqueE_K[..., c, a, :, :] = -1j * (orbital_c @ r_a - r_a @ orbital_c)

        self.projected_orbitalTorqueE_K_list = []
        for orbitalTorqueProjection in self.orbitalTorqueProjectionList:
            proj1 = projectionOperator(orbitalTorqueProjection['source'], self.model.atomicOrbInfoList, self.model.norbs)
            proj2 = projectionOperator(orbitalTorqueProjection['target'], self.model.atomicOrbInfoList, self.model.norbs)
            proj1_K = self.projectorInEigenBasis(proj1)[..., None, None, :, :]
            proj2_K = self.projectorInEigenBasis(proj2)[..., None, None, :, :]
            self.projected_orbitalTorqueE_K_list.append((proj1_K @ self.orbitalTorqueE_K + self.orbitalTorqueE_K @ proj2_K) / 2)
        if self.verbose==2: showElapsedTime(extraInfo="dipole corrected orbital torque in eig.-basis finished")

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
                        self.w_K[..., idirection, ideriv, :, :]
                        - velocity_direction * deltaVelocity_deriv * self.deltaEInverse_pure
                        + 1j * commutator
                    ) * self.deltaEInverse_pure / 1j * off_diag
            if self.verbose==2: showElapsedTime(extraInfo=f"generalized k-derivatives of dipole operators finished")

class SpinVelocity(Velocity, Spin):
    def __init__(self, velocityProjectionList=[], spinvelocityProjectionList=[], **kwarg) -> None:
        '''
        Compute spin magnetic-moment current {mu_S,v}/2 and its projections.
        spin_K is mu_S/mu_B = -gS*S/hbar, so this is a magnetic-moment
        current, not the spin-angular-momentum current {S,v}/2. No additional
        electron charge multiplies this observable.
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
        for spinvelocityProj in spinvelocityProjectionList:
            proj1=projectionOperator(spinvelocityProj['source'],self.model.atomicOrbInfoList,self.model.norbs)
            proj2=projectionOperator(spinvelocityProj['target'],self.model.atomicOrbInfoList,self.model.norbs)
            
            proj1_K=self.projectorInEigenBasis(proj1)[...,None,None,:,:]
            proj2_K=self.projectorInEigenBasis(proj2)[...,None,None,:,:]
            self.projected_spinVelocity_K_list.append((proj1_K@self.spinVelocity_K@proj2_K + proj2_K@self.spinVelocity_K@proj1_K)/2)
        
        if self.verbose==2: showElapsedTime(extraInfo=f"spin velocity in eig.-basis finished")

class _OrbitalCurrentVertex:
    """Build a current from the moment supplied by the next electronic stage."""
    def __init__(self, velocityProjectionList=(), orbitalvelocityProjectionList=(), **kwarg):
        """Prepare {m, hbar*v}/2 and projections of the complete current vertex.

        orbitalVelocity_K has axes kgrid + (flow, polarization, band, band).
        Projections select source and target shells of the current matrix.
        """
        super().__init__(velocityProjectionList=velocityProjectionList, **kwarg)
        
        __oV=self.orbital_K[...,None,:,:,:] @ self.velocity_K[...,None,:,:]
        self.orbitalVelocity_K = (__oV + __oV.swapaxes(-2,-1).conj())/2

        self.projected_orbitalVelocity_K_list=[]
        for orbitalvelocityProj in orbitalvelocityProjectionList:
            proj1=projectionOperator(orbitalvelocityProj['source'],self.model.atomicOrbInfoList,self.model.norbs)
            proj2=projectionOperator(orbitalvelocityProj['target'],self.model.atomicOrbInfoList,self.model.norbs)
            
            proj1_K=self.projectorInEigenBasis(proj1)[...,None,None,:,:]
            proj2_K=self.projectorInEigenBasis(proj2)[...,None,None,:,:]
            self.projected_orbitalVelocity_K_list.append((proj1_K@self.orbitalVelocity_K@proj2_K + proj2_K@self.orbitalVelocity_K@proj1_K)/2)
        
        if self.verbose==2: showElapsedTime(extraInfo=f"orbital velocity in eig.-basis finished")

class OrbitalVelocity(_OrbitalCurrentVertex, Orbital, Velocity):
    """Atom-centered orbital magnetic-moment current {m_ACA, hbar*v}/2."""

class OrbitalVelocity_BlochSelfRotate(_OrbitalCurrentVertex, Orbital_BlochSelfRotate):
    """Transport vertex {m_self, hbar*v}/2 for Bloch self-rotation.

    orbitalVelocity_K has axes kgrid + (flow, polarization, band, band)
    and units mu_B * eV * angstrom, matching OrbitalVelocity.
    """
