import numpy as np

class System:
    '''
    The top level of a system, which is extended by its child classes e.g., System_w90, System_lcao
    '''
    default_parameters = {
        'mainDir'      : "./mcalc/Pt/",
        'seedname'     : "wannier90",
        'POSCAR'       : "POSCAR",
        'KPOINTS'      : "KPOINTS",
        'ham_R'        : None,
        'overlap_R'    : None,
        'mp_grid'      : (1,1,1),    # the BZ mesh
        'ham_K'        : None,
        'overlap_K'    : None,
        'overlap'      : False,      # if True then the basis is non-orthorgonal
        'energy'       : True,       # compute  the eigen-energies
        'velocity'     : False,      # contains the velocity matrix
        'spin velocity': False,      # contains the spin velocity matrix
    }
    def __init__(self) -> None:
        # number of basis in unit cell
        self.norbs:int=-1  
        # real-space hamiltonian produced by ab inito
        self.ham_R:np.ndarray=None
        # real-space energy matrix produced by openmx
        self.energy_R:np.ndarray=None
        # the Bravais vectors of real-space hamiltonian
        self.dRList:np.ndarray=None
        # imposing overlap (non-othorgonal basis)
        self.overlap=False
        # real-space overlap matrix
        self.overlap_R:np.ndarray=None
        # real-space overlap matrix
        self.RHam_R:np.ndarray=None
        # spin convention, "VASP" collect the same spin at one block, while "QE" collect the same orbital at one small block
        self.format="VASP"
        # spin matrix, in basis of home cell
        self.spin:np.ndarray
    
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
        if self.format=="VASP":
            # sigma^x
            self.spin[0,:self.norbs//2,self.norbs//2:] = II
            self.spin[0,self.norbs//2:,:self.norbs//2] = II
            # sigma^y
            self.spin[1,:self.norbs//2,self.norbs//2:] = -1j*II
            self.spin[1,self.norbs//2:,:self.norbs//2] =  1j*II
            # sigma^z
            self.spin[2,:self.norbs//2,:self.norbs//2] =  II
            self.spin[2,self.norbs//2:,self.norbs//2:] = -II
        elif self.format=="QE":
            raise NotImplementedError()
        # sigma^0
        self.spin[3,:self.norbs//2,:self.norbs//2] =  II
        self.spin[3,self.norbs//2:,self.norbs//2:] =  II

    def setEnergyR(self,ene_R) -> None:
        '''set lattice Energy matrix'''
        self.energy_R=ene_R

    def setHamR(self,ham_R) -> None:
        '''set lattice Hamiltonian'''
        self.ham_R=ham_R
    
    def setOR(self,O_R) -> None:
        '''set lattice Overlap'''
        self.overlap_R=O_R
    
    def setRList(self,dRList) -> None:
        '''set dR list'''
        self.dRList=dRList
    
    @property
    def Vcell(self):
        return np.linalg.det(self.lattice)

from .system_w90 import System_w90
from .system_omx import System_omx