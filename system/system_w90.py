import numpy as np
import functools

from numpy.core.multiarray import array as array, zeros as zeros
from system import System
from auxiliary import obtainProjections, obtainProjectionsPlain, showElapsedTime, iterate3dpm, fourier_q_to_R
from fileio import readHrDat, readPOSCAR, readRDat
from fileio import W90CHK, W90MMN, W90EIG

class System_w90(System):

    def __init__(self,
                 mainDir:str,hamR:str="wannier90",ignoreImagPart=False,cutoff=-1.,format="VASP",
                 poscar:str="POSCAR",projections:dict=None,ispin=1,
                 getAA:bool=True,trans_inv=True,
                 ) -> None:
        '''
        The system in basis of wannier orbitals, with convention as below:
        - Basis are normalized and orthorgonal to each other.
        - For case with double spin tunnels, the basis of spin-up are collected in 1st block while spin-down are in the other block.
        - The basis in L=1 shell are in order p_z, p_x, p_y; 
          L=2 are d_{z^2}, d_{xz}, d_{yz}, d_{x^2-y^2}, d_{xy}; 
          L=3 are f_{z^3}, f_{xz^2}, f_{yz^2}, f_{z(x^2-y^2)}, f_{xyz}, f_{x(x^2-3y^2)}, f_{y(3x^2-y^2)}.
        
        Parameters:
        -----------
        - mainDir : str  
        the path to working directory
        - hamR : str
        seedname of real-space hamiltonian, in w90 format
        - ignoreImagPart : logic
        to skip the imaginary part of hopping and overlap
        - cutoff : float
        to skip the bond with strength below the criteria
        - format : "VASP"|"QE"
        the spin tunnel order can be in VASP format, which the spin-down are collected, or in QE format
        - poscarPath : str
        seedname of POSCAR
        - projections : dictionary
        The atomic orbitals of each element. Key name corresponds to the element, item corresponds to the
          sequence of atomic orbitals, e.g.,
        - ispin : int
        The number of spin tunnels. NOTE: set ispin=2 for spinor case
          projections={
              "Fe":("s","s","p","p","d","d"),
              "Cl":(0,0,1,1,2,3),
          }

          in which 0 is "s", 1 is "p", 2 is "d", 3 is "d" and 4 is "f". By now, only these 4 kinds are supported.
        - trans_inv : bool
        switch the periodic boundary condition
        '''
        super().__init__()
        # wannier basis are normalized and othorgonal
        self.overlap=False
        # TB approximation
        self.getAA=getAA
        showElapsedTime()
        # 1. read in .chk file to obtain the wannier projection trans. Vk
        chk=W90CHK(seedname="%s/%s"%(mainDir,hamR))
        showElapsedTime(extraInfo="chk read")
        # 3. read in eigs computed by DFT, which is also the Hamiltonian in DFT gauge
        ham_k_DFT=W90EIG(seedname="%s/%s"%(mainDir,hamR))
        showElapsedTime(extraInfo=f"H_k_DFT: {ham_k_DFT.data.min():.6f} ~ {ham_k_DFT.data.max():.6f} eigs. read")
        Ham_k_W = np.array([self.toWannierGauge(H_k, ik, ik, chk) for ik, H_k in enumerate(ham_k_DFT.data)])
        showElapsedTime(extraInfo="Wannier gauge trans. done")

        # 5. store lattice info.
        self.norbs=chk.num_wann#ham_R.shape[-1]
        self.mp_grid=chk.mp_grid
        self.lattice=chk.real_lattice
        self.reciLattice=np.linalg.inv(self.lattice)*np.pi*2
        self.iRvec, self.iRvecIndex, self.Ndegen = self.wigner_seitz(chk.mp_grid)
        self.nRvec=len(self.iRvec)

        kpt_mp_grid = [
            tuple(k) for k in np.array(np.round(chk.kpt_list * np.array(chk.mp_grid)[None,:]), dtype=int) % chk.mp_grid
        ]

        fourier_q_to_R_loc = functools.partial(
            fourier_q_to_R,
            mp_grid=chk.mp_grid,
            kpt_mp_grid=kpt_mp_grid,
            iRvec=self.iRvec,
            ndegen=self.Ndegen)
        
        # 6. FFT to obtain H_W in R-space
        self.ham_R=fourier_q_to_R_loc(Ham_k_W)
        
        '''
        # 7. check the eigs
        ham_R_W=np.zeros(tuple(chk.mp_grid)+(self.norbs,self.norbs),dtype=complex)
        for iR, Rvec in enumerate(self.iRvec):
            ham_R_W[tuple(Rvec)]=self.ham_R[iR]
            
        ham_k_W=np.fft.ifftn(ham_R_W,axes=(0,1,2,))*np.prod(chk.mp_grid)
        ham_k_H, _=np.linalg.eigh(ham_k_W)
        for ik, kpt in enumerate(kpt_mp_grid):
            print(kpt)
            for eig in ham_k_DFT.data[ik,:self.norbs]: print(f"{eig:8.4f}",end='')
            print()
            for eig in ham_k_H[kpt]: print(f"{eig:8.4f}",end='')
            print()
        #exit()'''
        '''
        print(self.ham_R.shape)
        with open("%s/auto_hr.dat"%mainDir,"w") as fout:
            for iR, hR in zip(self.iRvec,self.ham_R):
                for isource in range(self.norbs):
                    for itarget in range(self.norbs):
                        fout.write("%5d%5d%5d%5d%5d%10.6f%10.6f\n"%(*iR,isource+1,itarget+1,hR[isource,itarget].real,hR[isource,itarget].imag))
        exit()'''
        
        # 7. compute Berry connection in wannier gauge
        if getAA:
            M_k_DFT=W90MMN(seedname="%s/%s"%(mainDir,hamR),kpt_list=chk.kpt_list,reci_lat=chk.recip_lattice)
            showElapsedTime(extraInfo="mmn read")
            Mmn_k_W = np.array([[self.toWannierGauge(Mmnk1k2, ik1, M_k_DFT.innkpt[ik1,ib], chk) for ib, Mmnk1k2 in enumerate(Mmnk1)] for ik1, Mmnk1 in enumerate(M_k_DFT.Mmn)])
            AA_k_W=np.zeros((chk.num_kpts,3,self.norbs,self.norbs),dtype=complex)
            delta=np.eye(self.norbs)
            for ikpt, MMNk in enumerate(Mmn_k_W):
                for bvec, wb, overlap in zip(M_k_DFT.bvecs[ikpt],M_k_DFT.wbvecs,MMNk):
                    # Eq.44 [PRB 74, 195118 (2006)]
                    _AA_k_W=1j*wb*bvec[:,None,None]*(overlap-delta)[None,:,:]
                    if trans_inv:
                        # Diagnonal term corrected via Eq.13 [PRB 56, 12847 (1997)]
                        _AA_k_W[:,np.arange(self.norbs,dtype=int),np.arange(self.norbs,dtype=int)]=-(
                            wb*bvec[:,None]*(np.log(np.diag(overlap)).imag)[None,:])
                    AA_k_W[ikpt,...]+=_AA_k_W
            # enforce the Hermite condition for AA_k_W
            #AA_k_W=(AA_k_W+AA_k_W.transpose((0,1,3,2)).conj())/2
            #np.set_printoptions(precision=3)
            #for kpt, AA_k in zip(chk.kpt_list,AA_k_W):
            #    print(kpt)
            #    print(AA_k[0,:4,:4])
            #exit()
            # 8. FFT to obtain AA_W in R-space
            self.AA_R_W=fourier_q_to_R_loc(AA_k_W)
            for Rvec, AA in zip(self.iRvec,self.AA_R_W):
            #    print(Rvec)
            #    print(AA[0,:3,:3])
                if Rvec@Rvec==0: self.rList=np.diagonal(AA,axis1=-2,axis2=-1).real
            #exit()
            print("diagonal element of AA_R_W and the wcc produced in .chk")
            for AA_r, wcc in zip(self.rList.T,chk.wannier_centers): 
                print(f"{AA_r[0].real:10.6f}{AA_r[1].real:10.6f}{AA_r[2].real:10.6f} - {wcc[0]:10.6f}{wcc[1]:10.6f}{wcc[2]:10.6f}")
            showElapsedTime(extraInfo="AA_R_W calc. done")

        # 12. read in the centers of each basis
        print("set wcc to align the chk")
        self.rList=chk.wannier_centers.T#np.zeros((3,self.norbs))
        self.ispin=ispin
        
        self.posinfo=readPOSCAR('%s/%s'%(mainDir,poscar))
        orb_list=obtainProjectionsPlain('%s/%s'%(mainDir,poscar),projections,ispin=ispin)
        assert self.norbs == sum([len(atom) for atom in orb_list])
        print("pls. check the wcc and atom coors.")
        print("At present, the atom coor will be used (for furthur symmerization)")
        for iatom, (atom_coor, atom_shell) in enumerate(zip(self.posinfo['pos ions (cart.)'],orb_list)):
            for iorb in atom_shell:
                wcc=self.rList[:,iorb]
                print(f"orb{iorb:3d} {wcc[0]:6.3f} {wcc[1]:6.3f} {wcc[2]:6.3f} atom{iatom:3d} {atom_coor[0]:6.3f} {atom_coor[1]:6.3f} {atom_coor[2]:6.3f}")
                self.rList[:,iorb]=atom_coor
        
        self.atomicOrbInfoList=obtainProjections('%s/%s'%(mainDir,poscar),projections,ispin=ispin)
        '''
        # tight-binding approach
        hrDat=readHrDat('%s/%s_hr.dat'%(mainDir,hamR),ignoreImagPart=ignoreImagPart,overlap=False,format=format)
        self.iRvec=hrDat['iRvec']
        self.iRvecIndex=hrDat['iRvecIndex']
        self.nRvec=len(self.iRvec)
        self.mp_grid=hrDat['mp_grid']
        self.ham_R=hrDat['ham_R']
        self.norbs=self.ham_R.shape[-1]
        # r_mn of tight-binding model is diagonal
        self.AA_R_W=np.zeros((self.nRvec,3,self.norbs,self.norbs),dtype=complex)
        for iRvec, Rvec in enumerate(self.iRvec):
            if Rvec@Rvec==0: 
                for iorb in range(self.norbs): self.AA_R_W[iRvec,:,iorb,iorb]=self.rList[:,iorb]
        #'''
        # 12. compute the [r, Ham]
        self.getRHam_R()
        self.getSpinMatrice()
    
    def toWannierGauge(self, mat:np.ndarray, ik1:int, ik2:int, chk:W90CHK):
        '''
        Transform mat on DFT gauge into Wannier gauge\n
        mat should be an array in form of ..., NB, NB\n
        The last two axis is to peform gauge trans.
        '''
        # 
        if len(mat.shape) == 1:
            mat = np.diag(mat)
        assert mat.shape[-2:] == (chk.num_bands, ) * 2
        mat = mat[...,chk.win_min[ik1]:chk.win_max[ik1], chk.win_min[ik2]:chk.win_max[ik2]]
        v1 = chk.v_matrix[ik1].conj()
        v2 = chk.v_matrix[ik2].T
        
        return v1@mat@v2

    def getRHam_R(self):
        super().getRHam_R() # for now, it is unecessary to compute the real velocity but the tight-binding velocity
        return 
    
        nR1,nR2,nR3,norbs,norbs=self.ham_R.shape
        self.RHam_R=np.zeros((nR1,nR2,nR3,3,norbs,norbs),dtype=complex)
        print(f"calculating R \dot H_R on lattice {nR1}x{nR2}x{nR3} with lattice points {len(self.dRList)=}")
        
        for d in self.dRList:
            d_xyz=np.array(d)@self.lattice
            self.RHam_R[d[0],d[1],d[2]]=self.ham_R[d[0],d[1],d[2],None,:,:]*d_xyz[None,None,None,:,None,None]
            for v in self.dRList_r:
                d_minus_v=(d[0]-v[0],d[1]-v[1],d[2]-v[2])
                if d_minus_v in self.dRList:
                    t_d_minus_v = self.ham_R[d_minus_v[0],d_minus_v[1],d_minus_v[2],None,:,:]
                    r_v=self.r_R[v[0],v[1],v[2]]
                    self.RHam_R[d[0],d[1],d[2]] += t_d_minus_v@r_v - r_v@t_d_minus_v
    
    def getSpinMatrice(self):
        super().getSpinMatrice()
        return
    
    @property
    def get_mp_grid(self): return self.mp_grid

    def wigner_seitz(self, mp_grid):
        ws_search_size = np.ones(3,dtype=int)
        dist_dim = np.prod((ws_search_size + 1) * 2 + 1)
        origin = divmod((dist_dim + 1), 2)[0] - 1
        real_metric = self.lattice @ self.lattice.T
        mp_grid = np.array(mp_grid)
        iRvec = []
        iRvecIndex={}
        ndegen = []
        for n in iterate3dpm(mp_grid * ws_search_size):
            dist = []
            for i in iterate3dpm((1, 1, 1) + ws_search_size):
                ndiff = n - i * mp_grid
                dist.append(ndiff.dot(real_metric.dot(ndiff)))
            dist_min = np.min(dist)
            if abs(dist[origin] - dist_min) < 1.e-7:
                iRvecIndex[tuple(n)]=len(iRvec)
                iRvec.append(np.array(n))
                ndegen.append(np.sum(abs(dist - dist_min) < 1.e-7))
                
        return iRvec, iRvecIndex, np.array(ndegen)
    
    def symmerizeHamAndOverlap(self, spinor=False, saxis: np.ndarray = np.array([0,0,-1]), spin_basis: np.ndarray = np.zeros((2,2)), symOpSelection: list = [], atomicMagMomList: list = [], print_row=(0,1,), print_col=(0,1,)):
        super().symmerizeHamAndOverlap(spinor, saxis, spin_basis, symOpSelection, atomicMagMomList, 'w90', print_row, print_col)
        self.getRHam_R()