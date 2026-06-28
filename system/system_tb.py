import numpy as np
from system import System
from auxiliary import obtainProjectionsPlain, obtainProjections
from fileio import readFullHrDat, readPOSCAR

class System_tb(System):

    def __init__(self,
                 mainDir:str,hamR:str="wannier90",ignoreImagPart=False,cutoff=-1.,format="VASP",
                 poscar:str="POSCAR",projections:dict=None,
                 ispin=1,spinSeparated=None,
                 ) -> None:
        '''
        The system in basis of tight-binding orbitals, with convention as below:
        - Basis are normalized AND orthorgonal to each other.
        - For case with double spin tunnels, the basis of spin-up are collected in 1st block while spin-down are in the other block.
        - The basis are collected in w90 order:
          L=1 are p_z, p_x, p_y; 
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
        - spinSeparated : None | bool
        If True, read ``{hamR}.up_hr.dat`` and ``{hamR}.dn_hr.dat`` as separate
        collinear spin channels and build a block-diagonal spinful Hamiltonian.
        If None, this mode is selected automatically when ``ispin=2`` and the
        single ``{hamR}_hr.dat`` file is absent but both spin-channel files are
        present.

          projections={
              "Fe":("s","s","p","p","d","d"),
              "Cl":(0,0,1,1,2,3),
          }

          in which 0 is "s", 1 is "p", 2 is "d", 3 is "d" and 4 is "f". By now, only these 4 kinds are supported.
        
        '''
        super().__init__()
        # atomic orbital basis are normalized but nonothorgonal
        self.overlap=False
        # read in real-space ham
        hrDat=readFullHrDat(mainDir, hamR, ignoreImagPart, format, ispin, spinSeparated, cutoff=cutoff)
        self.iRvec=hrDat['iRvec']
        self.iRvecIndex=hrDat['iRvecIndex']
        self.nRvec=len(self.iRvec)
        self.mp_grid=hrDat['mp_grid']
        self.ham_R=hrDat['ham_R']
        self.norbs=self.ham_R.shape[-1]
        # read in the centers of each basis
        self.rList=np.zeros((3,self.norbs))
        self.ispin=ispin
        self.posinfo=readPOSCAR('%s/%s'%(mainDir,poscar))
        orb_list=obtainProjectionsPlain('%s/%s'%(mainDir,poscar),projections,ispin=ispin)
        assert self.norbs == sum([len(atom) for atom in orb_list]), f'{self.norbs=}, {sum([len(atom) for atom in orb_list])=}'
        for atom_coor, atom_shell in zip(self.posinfo['pos ions (cart.)'],orb_list):
            for iorb in atom_shell: self.rList[:,iorb]=atom_coor
        # r_mn of tight-binding model is diagonal
        self.AA_R_W=np.zeros((self.nRvec,3,self.norbs,self.norbs),dtype=complex)
        for iRvec, Rvec in enumerate(self.iRvec):
            if Rvec@Rvec==0: 
                for iorb in range(self.norbs): self.AA_R_W[iRvec,:,iorb,iorb]=self.rList[:,iorb]

        self.atomicOrbInfoList=obtainProjections('%s/%s'%(mainDir,poscar),projections,ispin=ispin)
        self.lattice=self.posinfo['lattice']
        self.reciLattice=np.linalg.inv(self.lattice)*np.pi*2

        self.getRHam_R()
        self.getRRHam_R()
        self.getRRRHam_R()
        self.getSpinMatrice()

        # compute orbital angular moment operator
        self.atomOrbAM=self.orbitalAngularMomentCubic()
        self.getOrbitalMatrice()
    
    def orbitalAngularMomentCubic(self):
        '''
        Orbital angular moment operator represented in the cubic hamonics (s, px, py, pz, ...)
        in unit of \hbar
        For now, only l=s, p, d, f was implemented

        Return
        ------
        - Lop : list
          Lop[l][idirection][m1, m2]. 
          l=s, p, d, e, denoting the orbital quantum number
          idirection=0, 1, 2, denoting the spatial orientation of x, y, z
          m1, m2 is the magnetic quantum number
        '''
        orbitalAngularMomentSpherical=self.orbitalAngularMomentSpherical()
        w=1/np.sqrt(2)
        def orbitalAngularP():
            '''
            Representation of orbital angular moment in space of cubic harmonics pz, px, py
            '''
            L=np.array([
                [    0, 1,    0],   # pz
                [   -w, 0,    w],   # px 
                [ 1j*w, 0, 1j*w],   # py
                ])
            return L.conj() @ orbitalAngularMomentSpherical[1] @ L.T

        def orbitalAngularD():
            '''
            Representation of orbital angular moment in space of cubic harmonics dz2, dxz, dyz, dx2-y2, dxy
            '''
            L=np.array([
                [    0,    0, 1,    0,    0],   # dz2
                [    0,   -w, 0,    w,    0],   # dxz
                [    0, 1j*w, 0, 1j*w,    0],   # dyz
                [    w,    0, 0,    0,    w],   # dx2-y2
                [-1j*w,    0, 0,    0, 1j*w],   # dxy
                ])
            return L.conj() @ orbitalAngularMomentSpherical[2] @ L.T

        def orbitalAngularF():
            '''
            Representation of orbital angular moment in space of cubic harmonics fz3, fxz2, fyz2, fz(x2-y2), fxyz, fx(x2-3y2), fy(3x2-y2)
            '''
            L=np.array([
                [    0,    0,    0,  1,    0,    0,    0],   # fz3
                [    0,    0,   -w,  0,    w,    0,    0],   # fxz2
                [    0,    0, 1j*w,  0, 1j*w,    0,    0],   # fyz2
                [    0,    w,    0,  0,    0,    w,    0],   # fz(x2-y2)
                [    0,-1j*w,    0,  0,    0, 1j*w,    0],   # fxyz
                [   -w,    0,    0,  0,    0,    0,    w],   # fx(x2-3y2)
                [ 1j*w,    0,    0,  0,    0,    0, 1j*w],   # fy(3x2-y2)
                ])
            
            return L.conj() @ orbitalAngularMomentSpherical[3] @ L.T
        return (0, orbitalAngularP(), orbitalAngularD(), orbitalAngularF())

    def symmerizeHamAndOverlap(self, spinor=False, saxis: np.ndarray = np.array([0,0,-1]), spin_basis: np.ndarray = np.zeros((2,2)), symOpSelection: list = [], atomicMagMomList: list = [], print_row=(0,1,2,3,), print_col=(0,1,2,3,)):
        super().symmerizeHamAndOverlap(spinor, saxis, spin_basis, symOpSelection, atomicMagMomList, "w90", print_row, print_col)
        self.getRHam_R()
        self.getRRHam_R()
        self.getRRRHam_R()
