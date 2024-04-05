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
        # the Bravais vectors of real-space hamiltonian
        self.dRList:np.ndarray=None
        # real-space overlap matrix
        self.overlap_R:np.ndarray=None
        # real-space overlap matrix
        self.RHam_R:np.ndarray=None
        # spin matrix, in basis of home cell
        self.spin:np.ndarray
    
    def getRHam_R(self) -> None:pass

    def getSpinMatrice(self) -> None:pass

    @property
    def Vcell(self) -> float:return 1.

from .system_w90 import System_w90