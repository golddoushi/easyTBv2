import numpy as np
from hrReader import Hr
import posReader as pR
from system import System
from auxiliary import obtainProjectionsPlain
from fileio import readHrDat, readPOSCAR

class System_w90(System):

    def __init__(self,
                 mainDir:str,hamR:str="wannier90",ignoreImagPart=False,cutoff=-1.,format="VASP",
                 poscar:str="POSCAR",projections:dict=None,ispin=1,
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
        
        '''
        super().__init__()

        # read in real-space ham
        hrDat=readHrDat('%s/%s_hr.dat'%(mainDir,hamR),ignoreImagPart=ignoreImagPart,cutoff=cutoff,format=format)
        self.dRList=hrDat['dRList']
        self.ham_R=hrDat['ham_R']
        self.overlap_R=hrDat['overlap_R']
        self.overlap=hrDat['overlap']
        self.norbs=self.ham_R.shape[-1]
        # read in the centers of each basis
        self.rList=np.zeros((3,self.norbs))
        posinfo=readPOSCAR('%s/%s'%(mainDir,poscar))
        orb_list=obtainProjectionsPlain('%s/%s'%(mainDir,poscar),projections,ispin=ispin)
        assert self.norbs == sum([len(atom) for atom in orb_list])
        for atom_coor, atom_shell in zip(posinfo['pos ions (cart.)'],orb_list):
            for iorb in atom_shell: self.rList[:,iorb]=atom_coor
        self.lattice=posinfo['lattice']
        #print(self.rList)
    
    def getRHam_R(self):
        '''
        Calc. the matrix (R+d) * H_R, in unit length*energy, usually angstrom*eV. 
        
        Indexing: RH_R[iR1,iR2,iR3,idirection,iorb,jorb]

        The matrix element of H_R is t_{d+R}, where d is the dist. between orbs. in home cell, R is Bravais vector (both in cart. coors.).
        So, the matrix element of (R+d) * H_R is R t_{d+R}. Since R+d is rank-1 and 3-d, R t_{d+R} is also 3-d.
        '''
        nR1,nR2,nR3,norbs,norbs=self.ham_R.shape
        self.RHam_R=np.zeros((nR1,nR2,nR3,3,norbs,norbs),dtype=complex)

        dr=self.rList[:,:,None]-self.rList[:,None,:] # distance in cell, r_source - r_target
        for dR in self.dRList:
            # NOTE: dr is r_source - r_target, dR@lat is R_target - R_source, thus a negative sign must be imposed before dr
            self.RHam_R[dR[0],dR[1],dR[2]]=self.ham_R[dR[0],dR[1],dR[2],None,:,:]*((dR@self.lattice)[:,None,None] - dr)
    
    def getSpinMatrice(self):
        '''
        Calc. the spin matrices.

        Indexing: spin[icomponent=0, 1, 2, 3][ispin][jspin].

        spin[0], spin[1], spin[2] and spin[3] correspond to sigma^x,  sigma^y,  sigma^z and  sigma^0.
        '''
        assert 2*(self.norbs//2) == self.norbs, f"Spin matrices only make sense for system with double spin tunnels and even basis, but {self.norbs=}"
        self.spin=np.zeros((4,self.norbs,self.norbs),dtype=complex)
        II=np.eye(self.norbs//2)
        # sigma^x
        self.spin[0,:self.norbs//2,self.norbs//2:] = II
        self.spin[0,self.norbs//2:,:self.norbs//2] = II
        # sigma^y
        self.spin[1,:self.norbs//2,self.norbs//2:] = -1j*II
        self.spin[1,self.norbs//2:,:self.norbs//2] =  1j*II
        # sigma^z
        self.spin[2,:self.norbs//2,:self.norbs//2] =  II
        self.spin[2,self.norbs//2:,self.norbs//2:] = -II
        # sigma^0
        self.spin[3,:self.norbs//2,:self.norbs//2] =  II
        self.spin[3,self.norbs//2:,self.norbs//2:] =  II
    
    @property
    def Vcell(self):
        return np.linalg.det(self.lattice)



        

