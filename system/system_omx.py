import numpy as np
from numpy.core.multiarray import array as array, zeros as zeros
from system import System
from auxiliary import obtainProjectionsPlain, obtainProjections
from fileio import readHrDat, readPOSCAR

class System_omx(System):

    def __init__(self,
                 mainDir:str,hamR:str="wannier90",ignoreImagPart=False,cutoff=-1.,format="VASP",
                 poscar:str="POSCAR",projections:dict=None,ispin=1,
                 ) -> None:
        '''
        The system in basis of atomic orbitals, with convention as below:
        - Basis are normalized but non-orthorgonal to each other.
        - For case with double spin tunnels, the basis of spin-up are collected in 1st block while spin-down are in the other block.
        - The basis in L=1 shell are in order p_x, p_y, p_z; 
          L=2 are d_{z^2}, d_{x^2-y^2}, d_{xy}, d_{xz}, d_{yz}; 
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
        
        '''
        super().__init__()
        # wannier basis are normalized and othorgonal
        self.overlap=True
        # read in real-space ham
        hrDat=readHrDat('%s/%s_hr.dat'%(mainDir,hamR),ignoreImagPart=ignoreImagPart,overlap=True,format=format)
        self.iRvec=hrDat['iRvec']
        self.iRvecIndex=hrDat['iRvecIndex']
        self.nRvec=len(self.iRvec)
        self.mp_grid=hrDat['mp_grid']
        self.energy_R=hrDat['ham_R']  # in LCAO formalism, they actually use energy matrix rather
        self.overlap_R=hrDat['overlap_R']
        self.momentum_R=hrDat['momentum_R']
        self.norbs=self.energy_R.shape[-1]
        # read in the centers of each basis
        self.rList=np.zeros((3,self.norbs))
        self.ispin=ispin
        self.posinfo=readPOSCAR('%s/%s'%(mainDir,poscar))
        orb_list=obtainProjectionsPlain('%s/%s'%(mainDir,poscar),projections,ispin=ispin)
        assert self.norbs == sum([len(atom) for atom in orb_list]), f'{self.norbs=}, {sum([len(atom) for atom in orb_list])=}'
        for atom_coor, atom_shell in zip(self.posinfo['pos ions (cart.)'],orb_list):
            for iorb in atom_shell: self.rList[:,iorb]=atom_coor
        
        self.atomicOrbInfoList=obtainProjections('%s/%s'%(mainDir,poscar),projections,ispin=ispin)

        self.lattice=self.posinfo['lattice']
        self.reciLattice=np.linalg.inv(self.lattice)*np.pi*2

        self.getRHam_R()
        self.getRO_R()
        self.getSpinMatrice()
    
    def getRHam_R(self):
        #return super().getRHam_R()
        #'''
        # compute with tight-binding approximation
        self.RHam_R=np.zeros((self.nRvec,3,self.norbs,self.norbs),dtype=complex)
        print(f"calculating R \dot H_R on lattice with {self.nRvec} lattice points")
        dr=self.rList[:,:,None]-self.rList[:,None,:] # distance in cell, r_source - r_target
        for iR, Rvec in enumerate(self.iRvec):
            self.RHam_R[iR]=self.energy_R[iR,None,:,:]*((Rvec@self.lattice)[:,None,None] - dr)
        return 
        #'''
        # using momentum
        self.RHam_R=self.momentum_R.transpose((1,0,2,3)) * 27.2113845/2
    
    def getRO_R(self):
        #return super().getRHam_R()
        #'''
        # compute with tight-binding approximation
        self.RO_R=np.zeros((self.nRvec,3,self.norbs,self.norbs),dtype=complex)
        print(f"calculating R \dot O_R on lattice with {self.nRvec} lattice points")
        dr=self.rList[:,:,None]-self.rList[:,None,:] # distance in cell, r_source - r_target
        for iR, Rvec in enumerate(self.iRvec):
            self.RO_R[iR]=self.overlap_R[iR,None,:,:]*((Rvec@self.lattice)[:,None,None] - dr)
    
    def symmerizeHamAndOverlap(self, spinor=False, saxis: np.ndarray = np.array([0,0,1]), spin_basis: np.ndarray = np.zeros((2,2)), symOpSelection: list = [], atomicMagMomList: list = [], format="omx", print_row=(0,1,2,3,), print_col=(0,1,2,3,)):
        super().symmerizeHamAndOverlap(spinor, saxis, spin_basis, symOpSelection, atomicMagMomList, format, print_row, print_col)
        self.getRHam_R()
        self.getRO_R()