import numpy as np
import copy
from scipy import linalg
from scipy.special import j0
from auxiliary import showElapsedTime, pseudoRotation, print_matrix, getWholeSymmetryOperator
from auxiliary import PauliBlockSigmaNorm
from auxiliary import SYMPREC
from math import acos, sin, sqrt, cos

from symmetry import Symmetry, Group

class System:
    '''
    The top level of a system, which is extended by its child classes e.g., System_w90, System_lcao
    '''
    def __init__(self) -> None:
        # number of basis in unit cell
        self.norbs:int=-1  
        # real-space hamiltonian produced by ab inito
        self.ham_R:np.ndarray=None
        # real-space energy matrix produced by openmx
        self.energy_R:np.ndarray=None
        # position x overlap
        self.RO_R:np.ndarray=None
        # the Bravais lattice points
        self.iRvec:np.ndarray
        # the index of each Bravais lattice points
        self.iRvecIndex:dict
        # total lattice points
        self.nRvec:int
        # imposing overlap (non-othorgonal basis)
        self.overlap=False
        # real-space overlap matrix
        self.overlap_R:np.ndarray=None
        # position x Hamitonian
        self.RHam_R:np.ndarray=None
        self.RRHam_R:np.ndarray=None
        self.RRRHam_R:np.ndarray=None
        self.AA_R_W:np.ndarray=None # Berry connection, the expectation of position r
        # spin convention, "VASP" collect the same spin at one block, while "QE" collect the same orbital at one small block
        self.format="VASP"
        # num. of spin tunnels
        self.ispin=-1
        # spin matrix, in basis of home cell
        self.spin:np.ndarray=None
        # torque matrix, rely on the spin matrix defined
        self.spinTorque_R:np.ndarray=None
        # atomic orbital angular moment operator represented in cubic harmonics
        self.atomOrbAM:tuple=None
        # orbital matrix, in basis of home cell
        self.orbital:np.ndarray=None
        # lattice matrix
        self.lattice:np.ndarray=None
        # reciprocal lattice
        self.reciLattice:np.ndarray=None
        # structure infomation
        self.posinfo:dict
        # mapping from atom to orbitals
        self.atomicOrbInfoList:list
        # name of each orbital, indexing: atomicOrbNameList[iL][im] = label
        self.atomicOrbNameList:list=[
            ["s",]*2,
            ["p_z","p_x","p_y",]*2,
            ["d_{z^2}","d_{xz}","d_{yz}","d_{x^2-y^2}","d_{xy}",]*2,
            ["f_{z^3}","f_{xz^2}","f_{yz^2}","f_{z(x^2-y^2)}","f_{xyz}","f_{x(x^2-3y^2)}","f_{y(3x^2-y^2)}"]*2,]
        # Fermi level
        self.fermi:float
        # symmetry group
        self.pointGroup=Group([Symmetry(np.eye(3))],real_lattice=None,recip_lattice=None)
        # refractive index
        self.refractive_index:float = 1.0

        # reporting tag
        self.verbose:bool=True
    
    def getRHam_R(self):
        '''
        Calc. the matrix (R+d) * H_R, in unit length*energy, usually angstrom*eV. 
        
        Indexing: RH_R[iR1,iR2,iR3,idirection,iorb,jorb]

        The matrix element of H_R is t_{d+R}, where d is the dist. between orbs. in home cell, R is Bravais vector (both in cart. coors.).
        So, the matrix element of (R+d) * H_R is R t_{d+R}. Since R+d is rank-1 and 3-d, R t_{d+R} is also 3-d.
        '''
        self.RHam_R=np.zeros((self.nRvec,3,self.norbs,self.norbs),dtype=complex)
        if self.verbose: print(f"calculating R \dot H_R on lattice with {self.nRvec} lattice points")
        dr=self.rList[:,:,None]-self.rList[:,None,:] # distance in cell, r_source - r_target
        for iR, Rvec in enumerate(self.iRvec):
            #self.RHam_R[iR]=self.ham_R[iR,None,:,:]*((Rvec@self.lattice)[:,None,None] - dr)
            self.RHam_R[iR]=self.ham_R[iR,None,:,:]*(Rvec@self.lattice)[:,None,None]

    def getRRHam_R(self):
        # 2nd derivative spatial moment (w)
        self.RRHam_R = np.zeros((self.nRvec, 3, 3, self.norbs, self.norbs), dtype=complex)
        if self.verbose: print(f"calculating R \dot R \dot H_R on lattice with {self.nRvec} lattice points")
        for iR, Rvec in enumerate(self.iRvec):
            R_cart = Rvec @ self.lattice
            R_tensor2 = R_cart[:, None] * R_cart[None, :]
            self.RRHam_R[iR] = self.ham_R[iR, None, None, :, :] * R_tensor2[:, :, None, None]

    def getRRRHam_R(self):
        # 3rd derivative spatial moment (u)
        if self.verbose: print(f"calculating R \dot R \dot R \dot H_R on lattice with {self.nRvec} lattice points")
        self.RRRHam_R = np.zeros((self.nRvec, 3, 3, 3, self.norbs, self.norbs), dtype=complex)
        for iR, Rvec in enumerate(self.iRvec):
            R_cart = Rvec @ self.lattice
            R_tensor3 = R_cart[:, None, None] * R_cart[None, :, None] * R_cart[None, None, :]
            self.RRRHam_R[iR] = self.ham_R[iR, None, None, None, :, :] * R_tensor3[:, :, :, None, None]

    def getSpinMatrice(self,projectedOrbIndex=[]):
        '''
        Construct and compute spin angular moment operator represented in the eigen. space of Sz
        in unit of (1/2)\hbar

        Indexing: spin[icomponent=0, 1, 2][ispin][jspin].

        spin[0], spin[1], spin[2] correspond to sigma^x,  sigma^y,  sigma^z.
        '''
        assert 2*(self.norbs//2) == self.norbs, f"Spin matrices only make sense for system with double spin tunnels and even basis, but {self.norbs=}"
        
        self.spin=np.zeros((3,self.norbs,self.norbs),dtype=complex)
        if len(projectedOrbIndex)==0:
            II=np.eye(self.norbs//2)
        else:
            ii=np.zeros(self.norbs//2)
            for iorb in projectedOrbIndex:
                if iorb>=self.norbs//2: break
                ii[iorb]=1
            II=np.diag(ii)
        if self.format=="VASP" or self.format=="openmx": # ideal spinor basis, with blocked tunnels
            # sigma^x
            self.spin[0,:self.norbs//2,self.norbs//2:] = II
            self.spin[0,self.norbs//2:,:self.norbs//2] = II
            # sigma^y
            self.spin[1,:self.norbs//2,self.norbs//2:] = -1j*II
            self.spin[1,self.norbs//2:,:self.norbs//2] =  1j*II
            # sigma^z
            self.spin[2,:self.norbs//2,:self.norbs//2] =  II
            self.spin[2,self.norbs//2:,self.norbs//2:] = -II
        elif self.format=="VASP6":
            # TODO: read spin operator from wannier90.spn
            # < u_nk| simga_i | u_mk > , i=x,y,z
            # i(fastest) , n, m, k
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
        #self.spin[3,:self.norbs//2,:self.norbs//2] =  II
        #self.spin[3,self.norbs//2:,self.norbs//2:] =  II

    def orbitalAngularMomentSpherical(self):
        '''
        Orbital angular moment operator represented in the spherical hamonics (Y_{l, l}, Y_{l, l-1}, ...,  Y_{l, -l})
        in unit of \hbar
        For now, only l=0, 1, 2, 3 was implemented

        Return
        ------
        - Lop : list
          Lop[l][idirection][m1, m2]. 
          l=0, 1, 2, 3, denoting the orbital quantum number
          idirection=0, 1, 2, denoting the spatial orientation of x, y, z
          m1, m2 is the magnetic quantum number
        '''

        def orbitalAngularL1():
            '''
            Orbital moment operator represented in the spherical hamonics (Y_{1, +1}, Y_{1, 0}, Y_{1, -1})
            in unit of \hbar
            '''
            Lx=(1/np.sqrt(2))*np.array([
                [ 0, 1, 0],
                [ 1, 0, 1],
                [ 0, 1, 0],
            ])
            Ly=(1/np.sqrt(2))*np.array([
                [  0, -1j,   0],
                [ 1j,   0, -1j],
                [  0,  1j,   0],
            ])
            Lz=np.array([
                [ 1, 0, 0],
                [ 0, 0, 0],
                [ 0, 0,-1],
            ])
            return np.array([Lx,Ly,Lz])
        
        def orbitalAngularL2():
            '''
            Orbital moment operator represented in the spherical hamonics (Y_{2, +2}, Y_{2, +1}, Y_{2, 0}, Y_{2, -1}, Y_{2, -2})
            in unit of \hbar
            '''
            w=np.sqrt(3./2)
            Lx=np.array([
                [ 0, 1, 0, 0, 0],
                [ 1, 0, w, 0, 0],
                [ 0, w, 0, w, 0],
                [ 0, 0, w, 0, 1],
                [ 0, 0, 0, 1, 0],
            ])
            Ly=np.array([
                [  0,   -1j,     0,     0,   0],
                [ 1j,     0, -1j*w,     0,   0],
                [  0,  1j*w,     0, -1j*w,   0],
                [  0,     0,  1j*w,     0, -1j],
                [  0,     0,     0,    1j,   0],
            ])
            Lz=np.array([
                [ 2, 0, 0, 0, 0],
                [ 0, 1, 0, 0, 0],
                [ 0, 0, 0, 0, 0],
                [ 0, 0, 0,-1, 0],
                [ 0, 0, 0, 0,-2],
            ])
            return np.array([Lx,Ly,Lz])
    
        def orbitalAngularL3():
            '''
            Orbital moment operator represented in the spherical harmonics 
            (Y_{3, +3}, Y_{3, +2}, Y_{3, +1}, Y_{3, 0}, Y_{3, -1}, Y_{3, -2}, Y_{3, -3})
            in units of \hbar.
            '''
            w1 = np.sqrt(3./2)
            w2 = np.sqrt(5./2)
            w3 = np.sqrt(3.)
        
            Lx = np.array([
                [ 0, w1,  0,  0,  0,  0,  0],
                [w1,  0, w2,  0,  0,  0,  0],
                [ 0, w2,  0, w3,  0,  0,  0],
                [ 0,  0, w3,  0, w3,  0,  0],
                [ 0,  0,  0, w3,  0, w2,  0],
                [ 0,  0,  0,  0, w2,  0, w1],
                [ 0,  0,  0,  0,  0, w1,  0],
            ])
        
            Ly = np.array([
                [    0, -1j*w1,      0,      0,      0,      0,      0],
                [1j*w1,      0, -1j*w2,      0,      0,      0,      0],
                [    0,  1j*w2,      0, -1j*w3,      0,      0,      0],
                [    0,      0,  1j*w3,      0, -1j*w3,      0,      0],
                [    0,      0,      0,  1j*w3,      0, -1j*w2,      0],
                [    0,      0,      0,      0,  1j*w2,      0, -1j*w1],
                [    0,      0,      0,      0,      0,  1j*w1,      0],
            ])
        
            Lz = np.array([
                [ 3,  0,  0,  0,  0,  0,  0],
                [ 0,  2,  0,  0,  0,  0,  0],
                [ 0,  0,  1,  0,  0,  0,  0],
                [ 0,  0,  0,  0,  0,  0,  0],
                [ 0,  0,  0,  0, -1,  0,  0],
                [ 0,  0,  0,  0,  0, -2,  0],
                [ 0,  0,  0,  0,  0,  0, -3],
            ])
            return np.array([Lx, Ly, Lz])
    
        return (0, orbitalAngularL1(), orbitalAngularL2(), orbitalAngularL3())
    
    def getOrbitalMatrice(self):
        '''
        Calc. the orbital matrices.

        Indexing: orbital[icomponent=0, 1, 2][iorb, jorb].

        spin[0], spin[1], spin[2] correspond to Lx,  Ly,  Lz.
        '''
        self.orbital=np.zeros((3,self.norbs,self.norbs),dtype=complex)

        iorb=0
        for atom in self.atomicOrbInfoList:
            for l in atom["L"]:
                if l==0:
                    iorb+=1
                else:
                    self.orbital[:,iorb:(iorb+2*l+1),iorb:(iorb+2*l+1)]=self.atomOrbAM[l]
                    iorb+=(2*l+1)

        if iorb<self.norbs:
            assert iorb*2==self.norbs, f"we have set all magnetic quantum tunnel, but the total orbital tunnel is not reached {iorb=}, {self.norbs=}"
            self.orbital[:,iorb:self.norbs,iorb:self.norbs]=self.orbital[:,:iorb,:iorb]
    
    def setEnergyR(self,ene_R) -> None:
        '''set lattice Energy matrix'''
        self.energy_R=ene_R

    def setHamR(self,ham_R) -> None:
        '''set lattice Hamiltonian'''
        self.ham_R=ham_R
    
    def setOR(self,O_R) -> None:
        '''set lattice Overlap'''
        self.overlap_R=O_R
    
    def setFermi(self,fermi:float):
        self.fermi=fermi

    def setRList(self,dRList) -> None:
        '''set dR list'''
        self.dRList=dRList

    def getSpatialOrbitalIndexList(self, atom_id:int, shells=None) -> list:
        '''
        Return the spatial-orbital indexes of one atom.

        For spinful models in the VASP/openmx convention, only the first spin
        block is returned. The second spin block is obtained by adding
        ``self.norbs//2`` when needed.
        '''
        atom_info = self.atomicOrbInfoList[atom_id]
        if shells is None:
            shells = range(len(atom_info["orbital index"]))
        half_norbs = self.norbs//2 if self.ispin == 2 else self.norbs
        orb_list = []
        for ishell in shells:
            for iorb in atom_info["orbital index"][ishell]:
                if iorb < half_norbs:
                    orb_list.append(int(iorb))
        return orb_list

    def getAtomicSelfEnergy(self, spatial_orbital_index_list:list) -> np.ndarray:
        '''
        Extract the local magnetic self-energy of selected spatial orbitals.

        The result follows the old exchange-coupling convention: the home-cell
        onsite block is decomposed into Pauli blocks, and the spin-direction
        normalized magnetic part is returned in the spatial-orbital basis.
        '''
        assert self.ispin == 2, "exchange self-energy requires a spinful model with ispin=2"
        assert self.norbs % 2 == 0, f"spinful model should have even basis size, but {self.norbs=}"
        iR0 = self.iRvecIndex.get((0,0,0), -1)
        assert iR0 >= 0, "cannot find home-cell R=(0,0,0) matrix"

        half_norbs = self.norbs//2
        spatial_orbital_index_list = [int(iorb) for iorb in spatial_orbital_index_list]
        full_orbital_index_list = spatial_orbital_index_list + [iorb + half_norbs for iorb in spatial_orbital_index_list]

        if self.overlap:
            local_matrix = self.energy_R[iR0][full_orbital_index_list,:][:,full_orbital_index_list]
        else:
            local_matrix = self.ham_R[iR0][full_orbital_index_list,:][:,full_orbital_index_list]
        return PauliBlockSigmaNorm(local_matrix)

    def classifyExchangeRoutes(self, sourceAtoms=None, projection=None, pbc=(0,1,2),
                               maxRange=3, calRange=1, tolerance=SYMPREC,
                               includeAllRoutes=False, verbose=True):
        '''
        Search and classify exchange routes between selected magnetic atoms.

        Parameters
        ----------
        sourceAtoms : list[int] | None
            Magnetic atom indexes. If None, all atoms are used.
        projection : list[tuple[int, tuple[int]]] | None
            Optional atom-shell selection in the same style as other System
            projection APIs, e.g. ``[(0, (2,)), (1, (2,))]``. If None, all
            shells of each selected atom are used.
        pbc : tuple[int]
            Lattice-vector axes to search. Axes absent from this tuple are fixed
            to zero translation.
        maxRange : int
            Search translations from ``-maxRange`` to ``maxRange`` on PBC axes.
        calRange : int
            Number of distance shells to return.
        tolerance : float
            Distances differing by less than this value are grouped together.
        includeAllRoutes : bool
            If False, inverse duplicates such as i->j[R] and j->i[-R] are
            removed. If True, all directed routes are kept.

        Returns
        -------
        list[list[dict]]
            Exchange routes grouped by increasing bond length.
        '''
        if sourceAtoms is None:
            sourceAtoms = list(range(self.posinfo['tot ions']))
        sourceAtoms = [int(iatom) for iatom in sourceAtoms]
        pbc = tuple(int(axis) for axis in pbc)
        assert all(axis in (0,1,2) for axis in pbc), f"pbc axes should be chosen from 0, 1, 2, but {pbc=}"
        assert maxRange >= 0, f"{maxRange=} should be non-negative"
        assert calRange > 0, f"{calRange=} should be positive"

        projection_by_atom = {}
        if projection is not None:
            for atom_id, shells in projection:
                projection_by_atom[int(atom_id)] = tuple(shells)

        atom_orbitals = {}
        for atom_id in sourceAtoms:
            shells = projection_by_atom.get(atom_id, None)
            orb_list = self.getSpatialOrbitalIndexList(atom_id, shells=shells)
            if len(orb_list) > 0:
                atom_orbitals[atom_id] = orb_list
        sourceAtoms = [iatom for iatom in sourceAtoms if iatom in atom_orbitals]
        assert len(sourceAtoms) > 0, "no source atom with selected orbitals was found"

        axis_ranges = []
        for axis in range(3):
            if axis in pbc:
                axis_ranges.append(range(-maxRange, maxRange+1))
            else:
                axis_ranges.append((0,))

        routes = []
        inverse_seen = set()
        for isource, source_atom in enumerate(sourceAtoms):
            target_start = 0 if includeAllRoutes else isource
            source_frac = self.posinfo['pos ions (frac.)'][source_atom]
            for target_atom in sourceAtoms[target_start:]:
                target_frac = self.posinfo['pos ions (frac.)'][target_atom]
                for R0 in axis_ranges[0]:
                    for R1 in axis_ranges[1]:
                        for R2 in axis_ranges[2]:
                            overlat = (int(R0), int(R1), int(R2))
                            if source_atom == target_atom and overlat == (0,0,0):
                                continue
                            route_key = (source_atom, target_atom) + overlat
                            inverse_key = (target_atom, source_atom, -overlat[0], -overlat[1], -overlat[2])
                            if not includeAllRoutes and inverse_key in inverse_seen:
                                continue
                            inverse_seen.add(route_key)

                            bond_vec_frac = target_frac + np.array(overlat) - source_frac
                            bond_vec_cart = bond_vec_frac @ self.lattice
                            bond_length = float(np.linalg.norm(bond_vec_cart))
                            routes.append({
                                "source_atom": source_atom,
                                "target_atom": target_atom,
                                "overlat": np.array(overlat, dtype=int),
                                "bond_vector_frac": bond_vec_frac,
                                "bond_vector_cart": bond_vec_cart,
                                "bond_length": bond_length,
                                "source_orbitals": atom_orbitals[source_atom],
                                "target_orbitals": atom_orbitals[target_atom],
                            })

        routes.sort(key=lambda route: route["bond_length"])
        route_groups = []
        for route in routes:
            if len(route_groups) == 0 or route["bond_length"] > route_groups[-1][0]["bond_length"] + tolerance:
                if len(route_groups) >= calRange:
                    break
                route_groups.append([])
            route["route_type"] = len(route_groups)-1
            route["route_id"] = sum(len(group) for group in route_groups)
            route_groups[-1].append(route)

        self.exchangeRoutes = route_groups
        if verbose:
            print(f"Classified {sum(len(group) for group in route_groups)} exchange routes in {len(route_groups)} distance shells")
            for igroup, group in enumerate(route_groups):
                route0 = group[0]
                print(f"route shell {igroup}: length {route0['bond_length']:.6f}, routes {len(group)}")
        return route_groups
    
    def symmerizeTensorInAtomicBasis(self,tensor:np.ndarray,axisRep=("iRvec","atomic orb","atomic orb"),
                        spinor=False,saxis:np.ndarray=np.array([0,0,-1]),spin_basis:np.ndarray=np.zeros((2,2),dtype=complex),
                        symOpSelection:list=[],atomicMagMomList:list=[],format="w90",
                        print_row=(0,),print_col=(0,), # To print specialized block of local Hams.
                        reportSymOps=False, # Report the available symmetry operations
                        ):
        '''
        To enforce the symmerization constrains on the Hamiltonian or energy matrix.
        
        Parameters:
        -----------
        atomicOrbIndexList - dictionary 
            Contains the orbital types and index associated to each atom. Generated by auxiliary::obtainProjections.
        cutoff - float
            If the abs. sum. of bond matrix and overlap matrix are below cutoff, the atom-pair is skipped.
        spinor - logic
            Switches on the spinor. Also works for collinear spin-polarized case.
        saxis - ndarray
            spin-direction
        spin_basis - ndarray
            2x2 complex ndarray to indicate the spin basis. Usually, only the saxis is enough
        symOpSelection - list containing integers
            select some of the sym. ops. to apply in this function. NOTE: the selected ops. should form group with close 
            condition. Otherwise, the symmerization cannot be perfectly imposed.
        atomicMagMomList - list
            define the magnetic moment by 3-number array for each atom.
            use to calc. the sym. group elements with time reversal symmery
        format - str "w90"|"openmx"
            decide the orbital-order conventions
        maxPrint - int
            output lines of symmerized bonds
        print_row - tuple with ints
            the printed row index of bond matrix
        print_col - tuple with ints
            the printed col index of bond matrix 
        symmerizationMode - str "E"|"H"
            same for the symmerizations on Hr read from w90.
            For file from openmx, "E" mode denote the symmerization is directly composed on energy matrix, while "H" mode composes symmerization on the Hams. i.e. H = O_inv @ E @ O_inv, H, E are the Ham. factor matrix and energy matrix. O_inv is the inversed overlap matrix.
            NOTE: it seems "H" mode fails in most cases. I d n t know why.
        
            
        '''
        def checkRedundancy(checkBond,allSymmerizedBondAndOverlap):
            for symmerizedBondSet in allSymmerizedBondAndOverlap:
                if abs(symmerizedBondSet['bond length'] - checkBond['bond length']) > 0.1*SYMPREC: continue
                for bond in symmerizedBondSet['bond set']:
                    if bond['bond info'] == checkBond['bond info']:
                        return True
            return False
        def vecAround(vec): 
            newVec=[]
            for x in vec:
                x_int=int(np.around(x))
                if abs(x-x_int)<SYMPREC:
                    newVec.append(x_int)
                else:
                    newVec.append(x)
            return np.array(newVec)     
        def getSymOpsAndSymSiteMap(atomicMagMomList=[]):
            # constructe all sym. ops and fetch ion mapping under sym. ops
            symRotateList=[]
            symTranslateList=[]
            symIonMap=[]
            sym= getWholeSymmetryOperator(self.posinfo)
            rotation, translation= sym['rotations'], sym['translations']
            translation=[vecAround(translate)%1 for translate in translation]
            if reportSymOps:
                print('Lattice symmetry group order: %d'%rotation.shape[0])
                print('Warning: at present, pls. ensure that all the translation vectors are zero')
                print('         which can be achieve by chosing correct origin')
            for rotate, translate in zip(rotation,translation):
                #print(rotate)
                #print(translate)
                symRotateList.append(rotate)
                symTranslateList.append(translate)
                subMap=[]
                for iion, ion_coor in enumerate(self.posinfo['pos ions (frac.)']):
                    ion_coor_trans=rotate@ion_coor+translate
                    #print(f'atom{iion} ({ion_coor[0]:6.3f}{ion_coor[1]:6.3f}{ion_coor[2]:6.3f}) transformed to ({ion_coor_trans[0]:6.3f}{ion_coor_trans[1]:6.3f}{ion_coor_trans[2]:6.3f})')
                    for iion_origin, ion_coor_origin in enumerate(self.posinfo['pos ions (frac.)']):
                        d_coor=vecAround(ion_coor_trans-ion_coor_origin)%1#(ion_coor_trans-ion_coor_origin)%1
                        d_coor_len=sum(abs(d_coor))
                        #print(f'atom{iion_origin} at ({ion_coor_origin[0]:6.3f}{ion_coor_origin[1]:6.3f}{ion_coor_origin[2]:6.3f}) diff=({d_coor[0]:6.3f}{d_coor[1]:6.3f}{d_coor[2]:6.3f})')
                        if d_coor_len<SYMPREC:
                            #print('%d -> %d'%(iion,iion_origin))
                            #print(d_coor)
                            subMap.append(iion_origin)
                if len(subMap)!=sum(self.posinfo['nions']):
                    print("Error, cannot find symmetric ion for symmetry op:")
                    print(rotate)
                    print(translate)
                    exit()
                symIonMap.append(subMap)
            
            magmom_list=atomicMagMomList
            if spinor: # find proper sub-group consist with magnetism and time reversal symmetry
                new_symRotateList, new_symTranslateList, new_symIonMap, new_timeReversalList= [], [], [], []
                for isym in range(len(symIonMap)):
                    rotate_r, translate, ionMap= symRotateList[isym], symTranslateList[isym], symIonMap[isym]
                    magmom_rotated_list=[]
                    for m in atomicMagMomList:
                        magmom_rotated_list.append(pseudoRotation(rotate_r,self.lattice,m))
                    diff_magmom=[magmom_rotated_list[i_ion]-magmom_list[ionMap[i_ion]] for i_ion in range(sum(self.posinfo['nions']))]
                    if sum([np.sqrt(dmag@dmag) for dmag in diff_magmom]) < SYMPREC:
                        if reportSymOps:
                            print('crystal sym. op. preserved:')
                            print(rotate_r)
                            print(translate)
                            print('dmag:')
                            for idmag, dmag in enumerate(diff_magmom):
                                if magmom_list[idmag]@magmom_list[idmag]>SYMPREC: 
                                    print("%3d-%3d %6.3f%6.3f%6.3f->%6.3f%6.3f%6.3f?=%6.3f%6.3f%6.3f dmag:%6.3f%6.3f%6.3f"%(
                                        idmag,ionMap[idmag],
                                        *atomicMagMomList[idmag],*magmom_rotated_list[idmag],*atomicMagMomList[ionMap[idmag]],
                                        *dmag))
                        new_symRotateList.append(rotate_r)
                        new_symTranslateList.append(translate)
                        new_symIonMap.append(ionMap)
                        new_timeReversalList.append(False)
                    else:
                        if reportSymOps:
                            print(f'crystal sym. op. broken due to magnetism. res:{sum([np.sqrt(dmag@dmag) for dmag in diff_magmom]):15.6E}')
                            print(rotate_r)
                            print(translate)
                            print('dmag:')
                            for idmag, dmag in enumerate(diff_magmom):
                                if magmom_list[idmag]@magmom_list[idmag]>SYMPREC: 
                                    print("%3d-%3d %6.3f%6.3f%6.3f->%6.3f%6.3f%6.3f?=%6.3f%6.3f%6.3f dmag:%6.3f%6.3f%6.3f"%(
                                        idmag,ionMap[idmag],
                                        *atomicMagMomList[idmag],*magmom_rotated_list[idmag],*atomicMagMomList[ionMap[idmag]],
                                        *dmag))
                    #continue
                    # check time reversal symmetry
                    magmom_rotated_list=[]
                    for m in atomicMagMomList:
                        magmom_rotated_list.append(-pseudoRotation(rotate_r,self.lattice,m))
                    diff_magmom=[magmom_rotated_list[i_ion]-magmom_list[ionMap[i_ion]] for i_ion in range(sum(self.posinfo['nions']))]
                    if sum([np.sqrt(dmag@dmag) for dmag in diff_magmom]) < SYMPREC:
                        if reportSymOps:
                            print('crystal sym. op. + time reversal preserved:')
                            print(rotate_r)
                            print(translate)
                            print('dmag:')
                            for idmag, dmag in enumerate(diff_magmom):
                                if magmom_list[idmag]@magmom_list[idmag]>SYMPREC: 
                                    print("%3d-%3d %6.3f%6.3f%6.3f->%6.3f%6.3f%6.3f?=%6.3f%6.3f%6.3f dmag:%6.3f%6.3f%6.3f"%(
                                        idmag,ionMap[idmag],
                                        *atomicMagMomList[idmag],*magmom_rotated_list[idmag],*atomicMagMomList[ionMap[idmag]],
                                        *dmag))
                        new_symRotateList.append(rotate_r)
                        new_symTranslateList.append(translate)
                        new_symIonMap.append(ionMap)
                        new_timeReversalList.append(True)
                    else:
                        if reportSymOps:
                            print(f'crystal sym. op. + time reversal broken due to magnetism. res:{sum([np.sqrt(dmag@dmag) for dmag in diff_magmom]):15.6E}')
                            print(rotate_r)
                            print(translate)
                            print('dmag:')
                            for idmag, dmag in enumerate(diff_magmom):
                                if magmom_list[idmag]@magmom_list[idmag]>SYMPREC: 
                                    print("%3d-%3d %6.3f%6.3f%6.3f->%6.3f%6.3f%6.3f?=%6.3f%6.3f%6.3f dmag:%6.3f%6.3f%6.3f"%(
                                        idmag,ionMap[idmag],
                                        *atomicMagMomList[idmag],*magmom_rotated_list[idmag],*atomicMagMomList[ionMap[idmag]],
                                        *dmag))
                symRotateList, symTranslateList, symIonMap=new_symRotateList, new_symTranslateList, new_symIonMap
                return {'rotations':symRotateList, 'translations':symTranslateList, 'atom map':symIonMap, 'time reversal':new_timeReversalList}
            else:
                return {'rotations':symRotateList, 'translations':symTranslateList, 'atom map':symIonMap, 'time reversal':[False]*len(symRotateList)}
        def generateAtomicWFCSymOp(symRotateList,format="w90"):
            '''generate transformation matrix of cubic harmonics for each sym. Op
            output:
            data[0]:1x1 array: s; 
            data[1]:3x3 array: pz,px,py (w90 format); px,py,pz (openmx format); py,pz,px (VASP6 format)
            data[2]:5x5 array: dz2, dxz, dyz, dx2-y2, dxy (w90 format); dz2, dx2-y2, dxy, dxz, dyz (openmx format); dxy,dyz,dz2,dxz,dx2-y2 (VASP6 format)
            data[3]:7x7 array: fz3, fxz2, fyz2, fz(x2-y2), fxyz, fx(x2-3y2), fy(3x2-y2) (w90 and openmx); fy(3x2-y2), fxyz, fyz2, fz3, fxz2, fz(x2-y2), fx(x2-3y2) (VASP6 format)
            data[4]:2x2 array: up, dn
            '''
            def defineVecByAngles(theta,phi):
                '''angular rep. to cartesian rep.'''
                return np.array([sin(theta)*cos(phi),sin(theta)*sin(phi),cos(theta)])
            def pOrb(vec,format="w90"):
                '''p-orbitals defined in line with wannier90 v2.1
                   order: pz, px, py
                '''
                x,y,z=vec
                if format=="openmx": return np.array([x,y,z])
                if format=="VASP6": return np.array([y,z,x])
                return np.array([z,x,y]) # in order: pz, px, py, 
            def dOrb(vec,format="w90"):
                '''d-orbitals defined in line with wannier90 v2.1
                   order: dz2, dxz, dyz, dx2-y2, dxy
                '''
                x,y,z=vec
                if format=="openmx":
                    return np.array([
                        sqrt(15/4)*x*y,
                        sqrt(15/4)*y*z,
                        sqrt(5/16)*(3*z*z-1),
                        sqrt(15/4)*x*z,
                        sqrt(15/16)*(x*x-y*y),
                    ])
                if format=="VASP6":
                    return np.array([
                        sqrt(15/4)*x*y,
                        sqrt(15/4)*y*z,
                        sqrt(5/16)*(3*z*z-1),
                        sqrt(15/4)*x*z,
                        sqrt(15/16)*(x*x-y*y),
                    ])
                return np.array([
                    sqrt(5/16)*(3*z*z-1),
                    sqrt(15/4)*x*z,
                    sqrt(15/4)*y*z,
                    sqrt(15/16)*(x*x-y*y),
                    sqrt(15/16)*x*y*2
                ])
            def fOrb(vec,format="w90"):
                '''f-orbitals defined in line with wannier90 v2.1
                   order: fz3, fxz2, fyz2, fz(x2-y2), fxyz, fx(x2-3y2), fy(3x2-y2)
                '''
                x,y,z=vec
                if format=="VASP6":
                    return np.array([
                    sqrt(7)   *(5*z**3-3*z),
                    sqrt(21/2)*(5*z**2-1)*x,
                    sqrt(21/2)*(5*z**2-1)*y,
                    sqrt(105) *(x*x-y*y)*z,
                    sqrt(105) *(x*y*2)*z,
                    sqrt(35/2)*(x*x-3*y*y)*x,
                    sqrt(35/2)*(3*x*x-y*y)*y
                    ])/4
                return np.array([
                    sqrt(35/2)*(3*x*x-y*y)*y,
                    sqrt(105) *(x*y*2)*z,
                    sqrt(21/2)*(5*z**2-1)*y,
                    sqrt(7)   *(5*z**3-3*z),
                    sqrt(21/2)*(5*z**2-1)*x,
                    sqrt(105) *(x*x-y*y)*z,
                    sqrt(35/2)*(x*x-3*y*y)*x,
                ])/4
            
            pn1,pn2,pn3=np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1])
            pNBasis=[pn1,pn2,pn3]
            pList=np.array([pOrb(n_vec,format=format) for n_vec in pNBasis])
            ao_p_basis_std=np.linalg.inv(pList)
        
            theta1=acos(np.sqrt(1/3))
            theta2=np.pi-theta1
            dn1=defineVecByAngles(0,0)
            dn2=defineVecByAngles(theta1,0)
            dn3=defineVecByAngles(theta1,np.pi/2)
            dn4=defineVecByAngles(theta2,np.pi*0/6)
            dn5=defineVecByAngles(theta2,np.pi*(0/6+1/4))
            dNBasis=[dn1,dn2,dn3,dn4,dn5]
            dList=np.array([dOrb(n_vec,format=format) for n_vec in dNBasis])
            #print('determinals of d-orb basis:',np.linalg.det(dList))
            ao_d_basis_std=np.linalg.inv(dList)
            #exit()
        
            theta1=acos(np.sqrt(3/5))
            theta2=np.pi-theta1
            fn1=defineVecByAngles(0,0)
            fn2=defineVecByAngles(theta1,0)
            fn3=defineVecByAngles(theta1,np.pi/2)
            fn4=defineVecByAngles(theta2,0)
            fn5=defineVecByAngles(theta2,np.pi/4)
            fn6=defineVecByAngles(np.pi/2,0)
            fn7=defineVecByAngles(np.pi/2,np.pi/6)
            fNBasis=[fn1,fn2,fn3,fn4,fn5,fn6,fn7]
            fList=np.array([fOrb(n_vec,format=format) for n_vec in fNBasis])
            #print('determinals of f-orb basis',np.linalg.det(fList))
            ao_f_basis_std=np.linalg.inv(fList)

            atomicWaveSymOpList=[]
            invLat=np.linalg.inv(self.lattice)
            for rotate in symRotateList:
                rotate_xyz=invLat@(rotate.T)@self.lattice

                pTransList=np.array([pOrb(n_vec@rotate_xyz,format=format) for n_vec in pNBasis])
                ao_p_basis_trans=pTransList.T

                dTransList=np.array([dOrb(n_vec@rotate_xyz,format=format) for n_vec in dNBasis])
                ao_d_basis_trans=dTransList.T

                fTransList=np.array([fOrb(n_vec@rotate_xyz,format=format) for n_vec in fNBasis])
                ao_f_basis_trans=fTransList.T

                atomicWaveSymOpList.append([
                    np.array([1]),
                    ao_p_basis_trans@ao_p_basis_std.T,
                    ao_d_basis_trans@ao_d_basis_std.T,
                    ao_f_basis_trans@ao_f_basis_std.T,
                    ])
            return atomicWaveSymOpList
        def generateSpinWFCSymOp(rotate,timeReversal=False):
            '''generate represent matrix of rotations in spin orbital space
            output 2x2 array
            '''
            L=self.lattice
            L_inv=np.linalg.inv(L)
            
            if not spinor:
                return np.eye(2)
            rotate_xyz=L_inv.dot(rotate.T).dot(L)
            x,y,z=[vecAround(vec) for vec in rotate_xyz]
            _z=np.array([x[1]*y[2]-x[2]*y[1],x[2]*y[0]-x[0]*y[2],x[0]*y[1]-x[1]*y[0]])
            if np.dot(z,_z)<0: # improper rotation
                x,y,z=-x,-y,-z # implement symmetric spin orbs

            beta=acos(z[2])
            sin_beta=sin(beta)
            if sin_beta>SYMPREC:
                cos_alpha=z[0]/sin_beta
                sin_alpha=z[1]/sin_beta
                alpha=acos(cos_alpha) if sin_alpha>=0 else acos(-cos_alpha)+np.pi

                sin_gamma=y[2]/sin_beta
                cos_gamma=-x[2]/sin_beta
                gamma=acos(cos_gamma) if sin_gamma>=0 else acos(-cos_gamma)+np.pi
            else:
                gamma=0
                beta=0 if z[2]>0 else np.pi

                cos_alpha=y[1]
                sin_alpha=-y[0]
                alpha=acos(cos_alpha) if sin_alpha>=0 else acos(-cos_alpha)+np.pi

            spinRotation=np.array([
                [np.exp(-1j/2*(alpha+gamma))*cos(beta/2),-np.exp(-1j/2*(alpha-gamma))*sin(beta/2)],
                [np.exp( 1j/2*(alpha-gamma))*sin(beta/2), np.exp( 1j/2*(alpha+gamma))*cos(beta/2)]
            ])
            if timeReversal:
                return spinRotation@np.array([[0,-1],[1,0]])
            return spinRotation
        def generateO3SymOp(rotate):
            '''
            generate O3 reps. of rotations, which is just the rotation matrix in xyz coors.
            
            :param symRotateList: rotation matrix in fractional coor.
            '''
            return self.lattice.T@rotate@np.linalg.inv(self.lattice.T)
        showElapsedTime()
        # check the projections
        orbName=['s','p','d','f']
        lmax=0
        for iatom, atom in enumerate(self.atomicOrbInfoList):
            atomL,atomOrb=atom["L"],atom["orbital index"]
            print('atom %d, lmax=%d'%(iatom,max(atomL)))
            lmax=max([lmax,max(atomL)])
            for ishell, (orbL, orbIndex) in enumerate(zip(atomL,atomOrb)):
                print('%s orb:'%orbName[orbL],orbIndex)
        # checking the shape
        assert len(tensor.shape)==len(axisRep), f"there are {len(tensor.shape)=} axis in tensor, but axis rep is {axisRep}"
        assert tensor.shape[0]==self.nRvec, f"first axis should be the iRvec, {self.nRvec=}, but {tensor.shape[0]=}"
        assert tensor.shape[-1]==self.norbs, f"the last two axes should be the iorb and jorb, {self.norbs=}, but {tensor.shape[-1]=}"
        assert tensor.shape[-2]==self.norbs, f"the last two axes should be the iorb and jorb, {self.norbs=}, but {tensor.shape[-2]=}"
        
        # enforce the hermit
        print(f"Enforce the hermit condition for real-space matrix {self.nRvec=} {self.norbs=}")
        tensor_herm    =np.zeros(tensor.shape,dtype=complex)
        for iR, R in enumerate(self.iRvec):
            iR_dual=self.iRvecIndex[(-R[0],-R[1],-R[2])]
            tensor_herm[iR,...]    = (tensor[iR,...]    + tensor[iR_dual,...].swapaxes(-2,-1).conj())/2
        ham_R   =tensor_herm
        showElapsedTime(extraInfo="Hermit processing finished")

        # obtain all sym. Ops. consistent with magnetism
        assert self.posinfo['tot ions'] == len(atomicMagMomList), f'inconsistent magmom with num. of atoms in structure. {self.posinfo["tot ions"]=} {len(atomicMagMomList)=}'
        sym= getSymOpsAndSymSiteMap(atomicMagMomList=atomicMagMomList)
        symRotateList, symTranslateList, symSiteMapList, symTimeReversalList=sym['rotations'], sym['translations'], sym['atom map'], sym['time reversal']
        if len(symOpSelection)>0:
            symRotateList=[symRotateList[isym] for isym in symOpSelection]
            symTranslateList=[symTranslateList[isym] for isym in symOpSelection]
            symSiteMapList=[symSiteMapList[isym] for isym in symOpSelection]
            symTimeReversalList=[symTimeReversalList[isym] for isym in symOpSelection]
        
        # set up symmetry group (for terms on BZ)
        # in now set up, the lattice is column-major and the reci-lat is row-major, which is opposite to VASP exactly
        # and the R is the rotation with cartesian basis
        lat,lat_inv=self.lattice.T,np.linalg.inv(self.lattice.T)
        self.symOpList=[]
        for rotate, timeReversal in zip (symRotateList,symTimeReversalList):
            self.symOpList.append(
                Symmetry(R=lat@rotate@lat_inv,TR=timeReversal)
            )
        self.pointGroup=Group(self.symOpList,real_lattice=self.lattice)
        
        # creating reps for each sym. op. 
        # firstly, the reps of spin basis
        symSpinRotateList=[]
        for rotate, timeReversal in zip(symRotateList,symTimeReversalList):
            symSpinRotateList.append(generateSpinWFCSymOp(rotate,timeReversal))

        # secondly, the reps of atom orbital basis
        atomicWaveSymOpList= generateAtomicWFCSymOp(symRotateList,format=format) # reps for rotation op. on s, p, d, f shell
        #print_matrix(atomicWaveSymOpList[1][2],rowIndex=(0,1,2,3,4,),colIndex=(0,1,2,3,4,))
        #exit()

        # for one specific atom, it might contains mutiple diffrent or same shell. So, here constructs reps for rotation op. on each atom
        # TODO: seperate the spin and atom orbital basis into different axis and operate seperately
        localSymOpList, localSymOpInvList=[], [] # localSymOpList[isymOp][iatom] give sym. op. in atomic subspace
        for isymOp in range(len(symRotateList)):
            localSymOp, localSymOpInv=[], []
            for iatom, atomInfo in enumerate(self.atomicOrbInfoList):
                atomL, atom = atomInfo["L"], atomInfo["orbital index"]
                atom_size=sum([len(shell) for shell in atom])
                atomSymOp=np.zeros((atom_size,atom_size),dtype=complex)
                cnt=0
                for L, shell in zip(atomL,atom):
                    nLorbitTunnels=2*L+1
                    assert len(shell)%nLorbitTunnels==0, 'Error: in atom%d size of orbit L=%d is %d'%(iatom,L,len(shell))
                    if spinor:
                        nLorbit=(len(shell)//2)//nLorbitTunnels
                        assert nLorbit==1,  'Error: in atom%d size of orbit L=%d is %d, not accommondate to (spinful) orbit size %d'%(iatom,L,len(shell),2*nLorbitTunnels)
                        
                        spinRotate= symSpinRotateList[isymOp]
                        for iLorbit in range(nLorbit):
                            atomSymOp[cnt:(cnt+nLorbitTunnels),                   cnt:(cnt+nLorbitTunnels)                   ]=atomicWaveSymOpList[isymOp][L]*spinRotate[0,0]
                            atomSymOp[cnt:(cnt+nLorbitTunnels),                   (cnt+nLorbitTunnels):(cnt+2*nLorbitTunnels)]=atomicWaveSymOpList[isymOp][L]*spinRotate[0,1]
                            atomSymOp[(cnt+nLorbitTunnels):(cnt+2*nLorbitTunnels),cnt:(cnt+nLorbitTunnels)                   ]=atomicWaveSymOpList[isymOp][L]*spinRotate[1,0]
                            atomSymOp[(cnt+nLorbitTunnels):(cnt+2*nLorbitTunnels),(cnt+nLorbitTunnels):(cnt+2*nLorbitTunnels)]=atomicWaveSymOpList[isymOp][L]*spinRotate[1,1]
                            cnt+=2*nLorbitTunnels
                    elif self.ispin==2:
                        nLorbit=(len(shell)//2)//nLorbitTunnels
                        #assert nLorbit>0, 'Error: in atom%d size of orbit L=%d is %d, not accommondate to (spinful) orbit size %d'%(iatom,L,len(shell),2*nLorbitTunnels)
                        #if nLorbit>1: print("Warning: found more than %d L=%d tunnels in atom%d, assuming that they are in good order"%(nLorbit,L,iatom))
                        assert nLorbit==1,  'Error: in atom%d size of orbit L=%d is %d, not accommondate to (spinful) orbit size %d'%(iatom,L,len(shell),2*nLorbitTunnels)

                        for iLorbit in range(nLorbit):
                            atomSymOp[cnt:(cnt+nLorbitTunnels),                   cnt:(cnt+nLorbitTunnels)                   ]=atomicWaveSymOpList[isymOp][L]
                            atomSymOp[(cnt+nLorbitTunnels):(cnt+2*nLorbitTunnels),(cnt+nLorbitTunnels):(cnt+2*nLorbitTunnels)]=atomicWaveSymOpList[isymOp][L]
                            cnt+=2*nLorbitTunnels
                    else:
                        nLorbit=len(shell)//nLorbitTunnels
                        #assert nLorbit>0, 'Error: in atom%d size of orbit L=%d is %d, not accommondate to orbit size %d'%(iatom,iL,len(shell),nLorbitTunnels)
                        #if nLorbit>1: print("Warning: found more than %d L=%d tunnels in atom%d, assuming that they are in good order"%(nLorbit,L,iatom))
                        assert nLorbit==1,  'Error: in atom%d size of orbit L=%d is %d, not accommondate to (spinful) orbit size %d'%(iatom,L,len(shell),2*nLorbitTunnels)

                        for iLorbit in range(nLorbit):
                            atomSymOp[cnt:(cnt+nLorbitTunnels),cnt:(cnt+nLorbitTunnels)]=atomicWaveSymOpList[isymOp][L]
                            cnt+=nLorbitTunnels
                        #print(atomicWaveSymOpList[isymOp][L])
                        
                localSymOp.append(atomSymOp)
                localSymOpInv.append(np.linalg.inv(atomSymOp))
            localSymOpList.append(localSymOp)
            localSymOpInvList.append(localSymOpInv)
        
        # thirdly, the reps of rotation for other possible axis
        repFuncMapping={
            "r":generateO3SymOp,
            "p":generateO3SymOp,
        }
        timeReversalSign={
            "r":1,
            "p":-1,
        }
        dimRep={
            "r":3,
            "p":3,
        }
        symMidRotateList,symMidRotateInvList=[], []  # symMidOpList[isymOp][imidAxis]
        for rotate, timeReversal in zip(symRotateList,symTimeReversalList):
            symMidRotateList.append([])
            symMidRotateInvList.append([])
            for icomponent, component in enumerate(axisRep[1:-2]):
                assert component in repFuncMapping.keys(), f"for now, only the reps. for following quantities are available: {repFuncMapping.keys()}"
                assert tensor.shape[1+icomponent]==dimRep[component], f"the dimension of axis{1+icomponent}: {tensor.shape[1+icomponent]} in consist with the reps of {component} {dimRep[component]}"
                repMat=repFuncMapping[component](rotate)
                if timeReversal: repMat*=timeReversalSign[component]
                symMidRotateList[-1].append(repMat)
                symMidRotateInvList[-1].append(np.linalg.inv(repMat))

        #np.set_printoptions(precision=4)
        # impose symmetry constrains
        #'''
        # In the following, we only need the mapping from iaotm to iorbs
        atomicOrbMap=[]
        for atom in self.atomicOrbInfoList:
            orbIndex=[]
            for shell in atom['orbital index']: orbIndex += shell
            atomicOrbMap.append(orbIndex)

        def constructSymmetryBondSet(R0,iatom_source,iatom_target,verbose=False):
            iR0=self.iRvecIndex[R0]

            source_coor=self.posinfo['pos ions (frac.)'][iatom_source]  # coor. of source atom
            # calc. the coor. of target atom, according to the bond info
            target_coor=self.posinfo['pos ions (frac.)'][iatom_target] + R0
            dR_xyz=(target_coor-source_coor)@self.lattice
            dR_xyz_len=np.sqrt(dR_xyz@dR_xyz)

            symmerizedBondAndOverlapSet={'preserve':True,'bond set':[],'bond length':dR_xyz_len} # recording the symmerization process
            if verbose: print(f'start construct symmetry recipe for bond {R0}({iR0}) {iatom_source} {iatom_target}')
            if verbose: print_matrix(ham_R[iR0][...,atomicOrbMap[iatom_source],:][...,atomicOrbMap[iatom_target]],rowIndex=print_row,colIndex=print_col)
            for rotate, translate, localSymOp, localSymOpInv, middleSymOps, middleSymOpsInv,  siteMap, timeReversal in zip(symRotateList,symTranslateList,localSymOpList,localSymOpInvList,symMidRotateList,symMidRotateInvList,symSiteMapList,symTimeReversalList):
                iatom_source_trans=siteMap[iatom_source]
                iatom_target_trans=siteMap[iatom_target]

                source_coor_trans=rotate@source_coor+translate
                target_coor_trans=rotate@target_coor+translate
                dR_xyz_rotated=(target_coor_trans-source_coor_trans)@self.lattice
                dR_xyz_rotated_len=np.sqrt(dR_xyz_rotated@dR_xyz_rotated)
                assert abs(dR_xyz_rotated_len-dR_xyz_len)<SYMPREC, 'sym. op. change bond length'

                R_tmp=(target_coor_trans-self.posinfo['pos ions (frac.)'][iatom_target_trans]) - (source_coor_trans - self.posinfo['pos ions (frac.)'][iatom_source_trans])#source_coor_trans-target_coor_trans+self.posinfo['pos ions (frac.)'][iatom_target_trans]-self.posinfo['pos ions (frac.)'][iatom_source_trans]
                for icomp, x in enumerate(R_tmp):  assert abs(x-np.around(x))<SYMPREC, 'dR[%d] is not interger %.6f'%(icomp,x)
                R=tuple([int(np.around(x)) for x in R_tmp])
                if R not in self.iRvecIndex:
                    return {'preserve':False}
                else:
                    iR=self.iRvecIndex[R]
                    bond   =ham_R[iR][...,atomicOrbMap[iatom_source_trans],:][...,atomicOrbMap[iatom_target_trans]].copy()
                    
                    bond_rotated=(localSymOpInv[iatom_source_trans])@bond@localSymOp[iatom_target_trans]
                    if timeReversal: 
                        bond_rotated=bond_rotated.conj()
                    
                    # then rotate the extra axes
                    for iExtraAxis, midSymOpInv in enumerate(middleSymOpsInv):
                        bond_rotated=np.tensordot(midSymOpInv,bond_rotated,axes=((-1,),(iExtraAxis,)))
                
                    if verbose: 
                        print(f'symmetry bond {R}({iR}) {iatom_source_trans} {iatom_target_trans} {timeReversal=}')
                        print("sym. op.:")
                        print(rotate)
                        print(translate)
                        print(f"{timeReversal=}")
                        print("original:")
                        print_matrix(bond,rowIndex=print_row,colIndex=print_col)
                        print("back to I hopping:")
                        print_matrix(bond_rotated,rowIndex=print_row,colIndex=print_col)
                    
                    symmerizedBondAndOverlapSet['bond set'].append({
                            'conjugation': False,
                            'on grid': True,
                            'bond info':(iatom_source_trans,iatom_target_trans,R[0],R[1],R[2]),
                            'original hopping':bond,
                            'sym. op. on source':localSymOp[iatom_source_trans],
                            'sym. op. on target':localSymOpInv[iatom_target_trans],
                            'sym. ops. on middle':middleSymOps,
                            'time reversal':timeReversal,
                            'back to I hopping':bond_rotated,
                        })
                    symmerizedBondAndOverlapSet['bond set'].append({
                            'conjugation': True,
                            'on grid': True,
                            'bond info':(iatom_target_trans,iatom_source_trans,-R[0],-R[1],-R[2]),
                            'original hopping':bond.conj().swapaxes(-2,-1),
                            'sym. op. on source':localSymOp[iatom_source_trans],
                            'sym. op. on target':localSymOpInv[iatom_target_trans],
                            'sym. ops. on middle':middleSymOps,
                            'time reversal':timeReversal,
                            'back to I hopping':bond_rotated,
                        })
            return symmerizedBondAndOverlapSet
        
        def calcSymmerizedBond(symSet,verbose=False):
            assert symSet['preserve'], 'the set is marked as out-of-range but u make me to process it'
            symOrder=len(symSet['bond set'])
            bond_matrix_avg = np.zeros(symSet['bond set'][0]['original hopping'].shape,dtype=complex)
            for bond in symSet['bond set']:
                bond_matrix_avg += bond['back to I hopping']
            bond_matrix_avg /= symOrder

            res_list=[]
            for bond in symSet['bond set']:
                opSource, opTarget, opsMiddle= bond['sym. op. on source'], bond['sym. op. on target'], bond['sym. ops. on middle']
                if bond['time reversal']:
                    bond['symmerized hopping'] = opSource @ (bond_matrix_avg.conj()) @ opTarget
                else:
                    bond['symmerized hopping'] = opSource @ bond_matrix_avg @ opTarget
                # processing middle part if any
                for iExtraAxis, midSymOp in enumerate(opsMiddle):
                    bond['symmerized hopping']=np.tensordot(midSymOp,bond['symmerized hopping'],axes=((-1,),(iExtraAxis,)))
                
                if bond['conjugation']:                
                    bond['symmerized hopping'] = bond['symmerized hopping'].conj().swapaxes(-2,-1)

                diff  = bond['symmerized hopping'] - bond['original hopping']
                res_list.append(np.max(abs(diff.real)) + 1j*np.max(abs(diff.imag)))
            res_max=np.max(np.array(res_list).real) + 1j*np.max(np.array(res_list).imag)
            if verbose: print(f"{res_max=:28.6E} ")
            return res_max
        
        def setSymmerizedBond(symSet,ham_sym:np.ndarray):
            for bond in symSet['bond set']:
                bond_t_matrix=bond['symmerized hopping']
                #bond_o_matrix=bond['symmerized overlap']
                # set to global bond structure
                iatom_source, iatom_target, R0, R1, R2 = bond['bond info']
                iR = self.iRvecIndex[(R0,R1,R2)]
                
                for iLocalOrbI, iGlobalOrbI in enumerate(atomicOrbMap[iatom_source]):
                    for iLocalOrbJ, iGlobalOrbJ in enumerate(atomicOrbMap[iatom_target]):
                        ham_sym[iR, ..., iGlobalOrbI, iGlobalOrbJ]=bond_t_matrix[..., iLocalOrbI, iLocalOrbJ]
        
        #symBondSet=constructSymmetryBondSet((0,0,0),0,1,verbose=True)
        #res=calcSymmerizedBond(symBondSet,verbose=True)
        #setSymmerizedBond(symBondSet,ham_R)
        #print("Check consistency of symmerization =====>>>>")
        #symBondSet=constructSymmetryBondSet((0,0,0),0,1,verbose=True)
        #res=calcSymmerizedBond(symBondSet,verbose=True)
        #exit()

        showElapsedTime()
        nDropedBond=0
        allSymmerizedBondAndOverlap=[]
        resList=[]
        for R0 in self.iRvec:
            for iatom_source in range(self.posinfo['tot ions']):
                source_coor=self.posinfo['pos ions (frac.)'][iatom_source] 
                for iatom_target in range(self.posinfo['tot ions']):
                    target_coor=self.posinfo['pos ions (frac.)'][iatom_target] + R0
                    dR_xyz=(target_coor-source_coor)@self.lattice
                    dR_xyz_len=np.sqrt(dR_xyz@dR_xyz)
                    if checkRedundancy({
                        'bond info':(iatom_source,iatom_target,R0[0],R0[1],R0[2]),
                        'bond length':dR_xyz_len
                    },allSymmerizedBondAndOverlap): continue
                    symBondSet=constructSymmetryBondSet(tuple(R0),iatom_source,iatom_target)
                    if not symBondSet['preserve']: 
                        nDropedBond+=1
                        continue
                    
                    ilen_order=0
                    for ilen_order, existedBondSet in enumerate(allSymmerizedBondAndOverlap):
                        if existedBondSet['bond length']>symBondSet['bond length']: break
                    else:
                        ilen_order+=1
                    allSymmerizedBondAndOverlap.insert(ilen_order,symBondSet)

                    resList.append({"R0":R0,"isource":iatom_source,"itarget":iatom_target,"res":calcSymmerizedBond(symBondSet)})
                    #print(f"sym set: [{R0[0]:2d}{R0[1]:2d}{R0[2]:2d}]{iatom_source:3d}{iatom_target:3d} {res=:20.3E}")
        ires=np.argsort([abs(res['res']) for res in resList])[-1]
        res=resList[ires]
        print(f"sym set with largest res: [{res['R0'][0]:2d}{res['R0'][1]:2d}{res['R0'][2]:2d}]{res['isource']:3d}{res['itarget']:3d} res={res['res']:20.3E}")
        showElapsedTime(extraInfo=f"Symmetry analysis finished")
        print(f"{nDropedBond} atom pairs skipped due to out of Wigner-Seiz supercell/ negligible strength")
        
        print("set symmery-constrained Ham. and overlap matrix")
        ham_R_sym   =np.zeros(ham_R.shape,dtype=complex)
        for iset, symBondSet in enumerate(allSymmerizedBondAndOverlap):
            setSymmerizedBond(symBondSet,ham_R_sym)
        showElapsedTime(extraInfo=f"symmery-constrained Ham. and overlap matrix constructed")
        return ham_R_sym

    def symmerizeHamAndOverlap(self,spinor=False,saxis:np.ndarray=np.array([0,0,-1]),spin_basis:np.ndarray=np.zeros((2,2),dtype=complex),
                        symOpSelection:list=[],atomicMagMomList:list=[],format="w90",
                        print_row=(0,),print_col=(0,), # To print specialized block of local Hams.
                        ):
        #'''
        if self.overlap:
            print("Symerizing energy matrix")
            self.energy_R=self.symmerizeTensorInAtomicBasis(self.energy_R,axisRep=("iRvec","atomic orb","atomic orb"),
                        spinor=spinor,saxis=saxis,spin_basis=spin_basis,
                        symOpSelection=symOpSelection,atomicMagMomList=atomicMagMomList,format=format,
                        print_row=print_row,print_col=print_col,reportSymOps=True,
                                                            )
        else:
            print("Symerizing Hamiltonian matrix")
            self.ham_R=self.symmerizeTensorInAtomicBasis(self.ham_R,axisRep=("iRvec","atomic orb","atomic orb"),
                        spinor=spinor,saxis=saxis,spin_basis=spin_basis,
                        symOpSelection=symOpSelection,atomicMagMomList=atomicMagMomList,format=format,
                        print_row=print_row,print_col=print_col,reportSymOps=True,
                                                         )
        #'''
        if self.AA_R_W is not None:
            print("Symerizing position matrix")
            # Zero out the diagonal elements which should not be transformed!
            iR0 = self.iRvecIndex[(0,0,0)]
            AA_0_W = np.zeros((3,self.norbs), dtype=complex)
            for iorb in range(self.norbs): 
                AA_0_W[:,iorb] = self.AA_R_W[iR0, :, iorb, iorb]
                self.AA_R_W[iR0, :, iorb, iorb] = 0

            # Only symertrize the off-diagonal transition dipoles (which strictly obey pure rotation)
            self.AA_R_W=self.symmerizeTensorInAtomicBasis(self.AA_R_W,axisRep=("iRvec","r","atomic orb","atomic orb"),
                        spinor=spinor,saxis=saxis,spin_basis=spin_basis,
                        symOpSelection=symOpSelection,atomicMagMomList=atomicMagMomList,format=format,
                        print_row=print_row,print_col=print_col,reportSymOps=False,
                                                            )
            # Restore the true, translationally-correct diagonal using the symmetric lattice coordinates
            for iorb in range(self.norbs): self.AA_R_W[iR0, :, iorb, iorb] = AA_0_W[:,iorb]
        #'''
        #exit()
    
    def make_supercell(self, Tmatrix):
        """
        Construct a supercell model directly from the primitive model.
        
        Parameters
        ----------
        Tmatrix : list or np.ndarray
            3x3 integer matrix defining the supercell lattice.
            The supercell lattice is defined as: lat_sc = T @ lat_pc
        """
        if self.verbose: print(f"\n--- Constructing Supercell ---")
        T = np.array(Tmatrix, dtype=int)
        invT = np.linalg.inv(T)
        Nsc = int(np.round(abs(np.linalg.det(T))))
        if self.verbose: print(f"Supercell volume ratio: {Nsc}")
        
        # 1. Find internal cells (primitive cell origins located inside the supercell)
        Rmax = np.max(np.abs(T)) * 2 + 2
        internal_cells = []
        for i in range(-Rmax, Rmax + 1):
            for j in range(-Rmax, Rmax + 1):
                for k in range(-Rmax, Rmax + 1):
                    c = np.array([i, j, k])
                    f = c @ invT
                    # Check if fractional coordinate is inside the [0, 1) region
                    if np.all(f >= -1e-6) and np.all(f < 1 - 1e-6):
                        internal_cells.append(c)
        
        assert len(internal_cells) == Nsc, f"Found {len(internal_cells)} internal cells, expected {Nsc}"
        
        def get_internal_cell_index(c_target):
            for idx, c in enumerate(internal_cells):
                if np.allclose(c, c_target):
                    return idx
            return -1
            
        # 2. Update structural information and orbital projections
        new_posinfo = {
            'lattice': T @ self.posinfo['lattice'],
            'pos ions (frac.)': [],
            'pos ions (cart.)': [],
            'nions': [],
            'label list': self.posinfo.get('label list', []),
            'tot ions': self.posinfo['tot ions'] * Nsc
        }
        
        new_atomicOrbInfoList = []
        norbs_pc = self.norbs
        norbs_sc = norbs_pc * Nsc
        new_rList = np.zeros((3, norbs_sc))
        
        # orb_map[iorb_pc, icell] gives the new global index in the supercell
        orb_map = np.zeros((norbs_pc, Nsc), dtype=int)
        
        # Maintain spin-blocking convention (e.g., all spin-up first, then all spin-down)
        if self.ispin == 2 and self.format in ["VASP", "openmx"]:
            half_pc = norbs_pc // 2
            half_sc = norbs_sc // 2
            cnt_up, cnt_dn = 0, half_sc
            for icell in range(Nsc):
                for iorb in range(half_pc):
                    orb_map[iorb, icell] = cnt_up
                    cnt_up += 1
                for iorb in range(half_pc, norbs_pc):
                    orb_map[iorb, icell] = cnt_dn
                    cnt_dn += 1
        elif self.ispin == 2:
            print(f"ERRRO: for now, make supercell for {self.format=} is still in development.")
            exit()
        else: # Standard sequential formatting
            cnt = 0
            for icell in range(Nsc):
                for iorb in range(norbs_pc):
                    orb_map[iorb, icell] = cnt
                    cnt += 1
                    
        # Update atoms and coordinate lists
        for i_elem, nions in enumerate(self.posinfo['nions']):
            new_posinfo['nions'].append(nions * Nsc)
            
        iatom_sc = 0
        for icell, c in enumerate(internal_cells):
            for iatom_pc, atom_info in enumerate(self.atomicOrbInfoList):
                frac_pc = self.posinfo['pos ions (frac.)'][iatom_pc]
                
                # New fractional coordinate in supercell
                frac_sc = (frac_pc + c) @ invT
                frac_sc = frac_sc % 1.0 # Fold strictly into home cell
                cart_sc = frac_sc @ new_posinfo['lattice']
                
                new_posinfo['pos ions (frac.)'].append(frac_sc)
                new_posinfo['pos ions (cart.)'].append(cart_sc)
                
                new_atom_info = {
                    "label": atom_info["label"],
                    "L": atom_info["L"].copy(),
                    "orbital index": []
                }
                for shell in atom_info["orbital index"]:
                    new_shell = [orb_map[iorb, icell] for iorb in shell]
                    new_atom_info["orbital index"].append(new_shell)
                    
                new_atomicOrbInfoList.append(new_atom_info)
                iatom_sc += 1
                
        # Update Wannier centers / atomic coordinates
        for icell, c in enumerate(internal_cells):
            for iorb_pc in range(norbs_pc):
                iorb_sc = orb_map[iorb_pc, icell]
                new_rList[:, iorb_sc] = self.rList[:, iorb_pc] + c @ self.lattice
                
        # 3. Reconstruct Hamiltonian, Overlap, and Position Matrices
        new_ham_dict = {}
        new_overlap_dict = {} 
        new_AA_dict = {}     
        
        has_overlap = getattr(self, "overlap_R", None) is not None
        has_AA = getattr(self, "AA_R_W", None) is not None
        has_energy = getattr(self, "energy_R", None) is not None
        
        if self.verbose: print(f"Folding {self.nRvec} primitive bonds into supercell grid...")
        for iR, R_pc in enumerate(self.iRvec):
            for icell1, c1 in enumerate(internal_cells):
                # The spatial shift target cell in primitive coordinates
                target_cell_pc = c1 + R_pc
                
                # Decompose into a supercell vector + internal cell origin
                frac_target = target_cell_pc @ invT
                R_sc = np.floor(frac_target + 1e-6).astype(int)
                c2 = np.round(target_cell_pc - R_sc @ T).astype(int)
                
                icell2 = get_internal_cell_index(c2)
                assert icell2 != -1, "Target internal cell lost during folding!"
                
                R_sc_tuple = tuple(R_sc)
                if R_sc_tuple not in new_ham_dict:
                    new_ham_dict[R_sc_tuple] = np.zeros((norbs_sc, norbs_sc), dtype=complex)
                    if has_overlap: new_overlap_dict[R_sc_tuple] = np.zeros((norbs_sc, norbs_sc), dtype=complex)
                    if has_AA: new_AA_dict[R_sc_tuple] = np.zeros((3, norbs_sc, norbs_sc), dtype=complex)
                        
                # Map matrices. We use += to correctly fold multiple primitive hoppings 
                # into the same supercell bond (e.g. self-interaction within the supercell)
                for i_pc in range(norbs_pc):
                    for j_pc in range(norbs_pc):
                        I_sc = orb_map[i_pc, icell1]
                        J_sc = orb_map[j_pc, icell2]
                        new_ham_dict[R_sc_tuple][I_sc, J_sc] += self.ham_R[iR, i_pc, j_pc]
                        if has_overlap: new_overlap_dict[R_sc_tuple][I_sc, J_sc] += self.overlap_R[iR, i_pc, j_pc]
                        if has_AA: new_AA_dict[R_sc_tuple][:, I_sc, J_sc] += self.AA_R_W[iR, :, i_pc, j_pc]
                            
        # 4. Pack into standard System arrays
        new_nRvec = len(new_ham_dict)
        self.iRvec = np.empty((new_nRvec, 3), dtype=int)
        self.iRvecIndex = {}
        self.ham_R = np.zeros((new_nRvec, norbs_sc, norbs_sc), dtype=complex)
        
        if has_overlap: self.overlap_R = np.zeros((new_nRvec, norbs_sc, norbs_sc), dtype=complex)
        if has_AA: self.AA_R_W = np.zeros((new_nRvec, 3, norbs_sc, norbs_sc), dtype=complex)
            
        for idx, R_sc_tuple in enumerate(new_ham_dict.keys()):
            self.iRvec[idx] = R_sc_tuple
            self.iRvecIndex[R_sc_tuple] = idx
            self.ham_R[idx] = new_ham_dict[R_sc_tuple]
            if has_overlap: self.overlap_R[idx] = new_overlap_dict[R_sc_tuple]
            if has_AA: self.AA_R_W[idx] = new_AA_dict[R_sc_tuple]
        Rx_max, Rx_min, Ry_max, Ry_min, Rz_max, Rz_min= self.iRvec[:,0].max(), self.iRvec[:,0].min(), self.iRvec[:,1].max(), self.iRvec[:,1].min(), self.iRvec[:,2].max(), self.iRvec[:,2].min()
        new_mp_grid=np.array([Rx_max-Rx_min+1,Ry_max-Ry_min+1,Rz_max-Rz_min+1,])

        if has_energy:
            # For OpenMX models, energy_R is structurally identical to ham_R
            self.energy_R = self.ham_R.copy()

        # 5. Overwrite System attributes
        self.norbs = norbs_sc
        self.lattice = new_posinfo['lattice']
        self.reciLattice = np.linalg.inv(self.lattice) * 2 * np.pi
        self.posinfo = new_posinfo
        self.atomicOrbInfoList = new_atomicOrbInfoList
        self.rList = new_rList
        self.nRvec = new_nRvec
        self.mp_grid=new_mp_grid
        
        # 6. Reconstruct dependent operators
        self.getRHam_R()
        if self.RRHam_R is not None: self.getRRHam_R()
        if self.RRRHam_R is not None: self.getRRRHam_R()
        if hasattr(self, 'getRO_R'): self.getRO_R()
        self.getSpinMatrice()
        self.atomOrbAM = self.orbitalAngularMomentCubic()
        self.getOrbitalMatrice()
        
        
        if self.verbose: print(f"--- Supercell {self.nRvec} lattice points constructed successfully ---\n")

    def shiftOrigin(self, surfaceDir, shift=None):
        """
        Shifts the origin of the unit cell along surfaceDir to minimize hoppings across the surface.
        Reassigns the R vectors of the hoppings accordingly without deleting any bonds.
        
        Parameters
        ----------
        surfaceDir : int
            The lattice direction to shift (0, 1, or 2).
        shift : float, optional
            The fractional shift of the cell boundary. If None, it is auto-optimized.
        """
        if self.verbose: print(f"\n--- Shifting Origin along Direction {surfaceDir} ---")
        invLattice = np.linalg.inv(self.lattice)
        
        # Get fractional coordinates of orbitals
        rList_frac = invLattice @ self.rList
        
        # 1. Auto-optimize the boundary shift to minimize boundary-crossing bonds
        if shift is None:
            if self.verbose: print(f"Optimizing cell origin to minimize boundary crossing...")
            shifts = np.linspace(0, 1, 100, endpoint=False)
            best_shift = 0
            min_cut_weight = float('inf')
            
            for s in shifts:
                cut_weight = 0
                c_I = np.floor(rList_frac[surfaceDir, :] + s).astype(int)
                
                for iR, R in enumerate(self.iRvec):
                    # R'_surfaceDir = R + c_target - c_source
                    R_prime_surfaceDir = R[surfaceDir] + c_I[None, :] - c_I[:, None]
                    mask_cut = (R_prime_surfaceDir != 0)
                    cut_weight += np.sum(np.abs(self.ham_R[iR])[mask_cut])
                    
                if cut_weight < min_cut_weight:
                    min_cut_weight = cut_weight
                    best_shift = s
                    
            shift = best_shift
            if self.verbose: print(f"Optimal fractional shift found: {shift:.4f} (Crossing weight: {min_cut_weight:.4f})")
        else:
            if self.verbose: print(f"Using provided shift: {shift:.4f}")
            
        # 2. Apply the chosen shift and remap the hoppings to new R vectors
        c_I = np.floor(rList_frac[surfaceDir, :] + shift).astype(int)
        
        new_ham_dict = {}
        new_overlap_dict = {}
        new_AA_dict = {}
        
        has_overlap = getattr(self, "overlap_R", None) is not None
        has_AA = getattr(self, "AA_R_W", None) is not None
        
        for iR, R in enumerate(self.iRvec):
            for I in range(self.norbs):
                for J in range(self.norbs):
                    # Minor optimization: skip exactly zero elements to save memory
                    if abs(self.ham_R[iR, I, J]) < 1e-12 and (not has_overlap or abs(self.overlap_R[iR, I, J]) < 1e-12):
                        continue
                        
                    # Reassign the R vector based on the new cell boundaries
                    R_new = R.copy()
                    R_new[surfaceDir] = R[surfaceDir] + c_I[J] - c_I[I]
                    R_new_tuple = tuple(R_new)
                    
                    if R_new_tuple not in new_ham_dict:
                        new_ham_dict[R_new_tuple] = np.zeros((self.norbs, self.norbs), dtype=complex)
                        if has_overlap: new_overlap_dict[R_new_tuple] = np.zeros((self.norbs, self.norbs), dtype=complex)
                        if has_AA: new_AA_dict[R_new_tuple] = np.zeros((3, self.norbs, self.norbs), dtype=complex)
                        
                    new_ham_dict[R_new_tuple][I, J] += self.ham_R[iR, I, J]
                    if has_overlap: new_overlap_dict[R_new_tuple][I, J] += self.overlap_R[iR, I, J]
                    if has_AA: new_AA_dict[R_new_tuple][:, I, J] += self.AA_R_W[iR, :, I, J]
                    
        if self.verbose: print(f"Origin Shift Complete: Re-mapped into {len(new_ham_dict)} lattice vectors.")
        
        # 3. Update physical coordinates (atoms and orbitals are shifted to the new home cell)
        for iatom, frac in enumerate(self.posinfo['pos ions (frac.)']):
            frac[surfaceDir] = (frac[surfaceDir] + shift) % 1.0
            self.posinfo['pos ions (frac.)'][iatom] = frac
            self.posinfo['pos ions (cart.)'][iatom] = frac @ self.lattice
            
        rList_frac[surfaceDir, :] = (rList_frac[surfaceDir, :] + shift) % 1.0
        self.rList = self.lattice @ rList_frac
        
        # 4. Repack the matrices into the System object
        new_nRvec = len(new_ham_dict)
        self.iRvec = np.empty((new_nRvec, 3), dtype=int)
        self.iRvecIndex = {}
        self.ham_R = np.zeros((new_nRvec, self.norbs, self.norbs), dtype=complex)
        if has_overlap: self.overlap_R = np.zeros((new_nRvec, self.norbs, self.norbs), dtype=complex)
        if has_AA: self.AA_R_W = np.zeros((new_nRvec, 3, self.norbs, self.norbs), dtype=complex)
        
        for idx, R_new_tuple in enumerate(new_ham_dict.keys()):
            self.iRvec[idx] = R_new_tuple
            self.iRvecIndex[R_new_tuple] = idx
            self.ham_R[idx] = new_ham_dict[R_new_tuple]
            if has_overlap: self.overlap_R[idx] = new_overlap_dict[R_new_tuple]
            if has_AA: self.AA_R_W[idx] = new_AA_dict[R_new_tuple]
        self.nRvec = new_nRvec
        
        # 5. Re-initialize dependent operators
        if hasattr(self, 'energy_R') and self.energy_R is not None:
            self.energy_R = self.ham_R.copy()
            
        self.getRHam_R()
        if self.RRHam_R is not None: self.getRRHam_R()
        if self.RRRHam_R is not None: self.getRRRHam_R()
        if hasattr(self, 'getRO_R') and has_overlap: self.getRO_R()
        if hasattr(self, 'getSpinMatrice'): self.getSpinMatrice()
        if hasattr(self, 'orbitalAngularMomentCubic'): self.atomOrbAM = self.orbitalAngularMomentCubic()
        self.getOrbitalMatrice()

    def cutEdge(self, cutDir):
        """
        Creates an open boundary condition (OBC) along the specified lattice direction
        by completely deleting any inter-cell hoppings that cross the boundary.
        
        NOTE: It is highly recommended to call shiftOrigin() before this to ensure 
        the boundary is placed optimally and atoms aren't severed mid-bond.
        """
        if self.verbose: print(f"\n--- Cutting Edge (OBC) along Direction {cutDir} ---")
        
        new_ham_dict = {}
        new_overlap_dict = {}
        new_AA_dict = {}
        
        has_overlap = getattr(self, "overlap_R", None) is not None
        has_AA = getattr(self, "AA_R_W", None) is not None
        
        dropped_blocks = 0
        kept_blocks = 0
        
        # Because shiftOrigin aligned the boundary, any hopping that crosses 
        # the boundary now strictly possesses a non-zero R component in cutDir.
        for iR, R in enumerate(self.iRvec):
            if R[cutDir] != 0:
                dropped_blocks += 1
                continue # Bond crosses the edge; it is completely dropped
                
            kept_blocks += 1
            R_tuple = tuple(R)
            
            # Keep the entire block matrix intact
            new_ham_dict[R_tuple] = self.ham_R[iR].copy()
            if has_overlap: new_overlap_dict[R_tuple] = self.overlap_R[iR].copy()
            if has_AA: new_AA_dict[R_tuple] = self.AA_R_W[iR].copy()
                
        if self.verbose: print(f"Edge Cut Complete: {dropped_blocks} inter-cell R-blocks dropped, {kept_blocks} kept.")
        
        # Repack the matrices into the System object
        new_nRvec = len(new_ham_dict)
        self.iRvec = np.empty((new_nRvec, 3), dtype=int)
        self.iRvecIndex = {}
        self.ham_R = np.zeros((new_nRvec, self.norbs, self.norbs), dtype=complex)
        if has_overlap: self.overlap_R = np.zeros((new_nRvec, self.norbs, self.norbs), dtype=complex)
        if has_AA: self.AA_R_W = np.zeros((new_nRvec, 3, self.norbs, self.norbs), dtype=complex)
        
        for idx, R_new_tuple in enumerate(new_ham_dict.keys()):
            self.iRvec[idx] = R_new_tuple
            self.iRvecIndex[R_new_tuple] = idx
            self.ham_R[idx] = new_ham_dict[R_new_tuple]
            if has_overlap: self.overlap_R[idx] = new_overlap_dict[R_new_tuple]
            if has_AA: self.AA_R_W[idx] = new_AA_dict[R_new_tuple]
            
        self.nRvec = new_nRvec
        self.mp_grid[cutDir] = 1 # Flatten the grid in the cut direction
        
        # Re-initialize dependent operators
        if hasattr(self, 'energy_R') and self.energy_R is not None:
            self.energy_R = self.ham_R.copy()
            
        self.getRHam_R()
        if self.RRHam_R is not None: self.getRRHam_R()
        if self.RRRHam_R is not None: self.getRRRHam_R()
        if hasattr(self, 'getRO_R') and has_overlap: self.getRO_R()
        if hasattr(self, 'getSpinMatrice'): self.getSpinMatrice()
        if hasattr(self, 'orbitalAngularMomentCubic'): self.atomOrbAM = self.orbitalAngularMomentCubic()
        self.getOrbitalMatrice()

    def constructPrincipleLayer(self, surfaceDir):
        """
        Constructs a principal layer supercell for surface Green's function calculations.
        The principal layer is defined such that hoppings in the surfaceDir direction 
        only reach the immediately adjacent supercells (i.e., dR_surfaceDir = 0, +1, -1).
        
        Parameters
        ----------
        surfaceDir : int
            The normal direction of the surface (0, 1, or 2).
        """
        if self.verbose: print(f"\n--- Constructing Principal Layer along Direction {surfaceDir} ---")
        
        # 1. Determine the maximum hopping range in the surface normal direction
        max_range = 0
        for iR, R in enumerate(self.iRvec):
            # Only consider non-negligible hoppings
            if np.max(np.abs(self.ham_R[iR])) > 1e-6:
                max_range = max(max_range, abs(R[surfaceDir]))
                
        if self.verbose: print(f"Maximum hopping range in direction {surfaceDir} is {max_range} primitive cells.")
        
        Nsc = max(1, max_range)
        if Nsc > 1:
            if self.verbose: print(f"Creating a {Nsc}x supercell along direction {surfaceDir} to form the principal layer.")
            Tmatrix = np.eye(3, dtype=int)
            Tmatrix[surfaceDir, surfaceDir] = Nsc
            self.make_supercell(Tmatrix)
        else:
            if self.verbose: print("Original cell is already a valid principal layer.")
            
        # 2. Partition the Hamiltonian into H_00, H_01, and H_m10
        self.surfaceDir = surfaceDir
        self.R_00, self.H_00_R = [], []
        self.R_01, self.H_01_R = [], []
        self.R_m10, self.H_m10_R = [], []
        
        for iR, R in enumerate(self.iRvec):
            if abs(R[surfaceDir]) > 1:
                # Should not happen if supercell construction was successful
                if np.max(np.abs(self.ham_R[iR])) > 1e-6:
                    print(f"WARNING: Found non-negligible hopping beyond principal layer at R={R}")
            
            if R[surfaceDir] == 0:
                self.R_00.append(R)
                self.H_00_R.append(self.ham_R[iR])
            elif R[surfaceDir] == 1:
                self.R_01.append(R)
                self.H_01_R.append(self.ham_R[iR])
            elif R[surfaceDir] == -1:
                self.R_m10.append(R)
                self.H_m10_R.append(self.ham_R[iR])
                
        self.R_00 = np.array(self.R_00)
        self.H_00_R = np.array(self.H_00_R)
        self.R_01 = np.array(self.R_01)
        self.H_01_R = np.array(self.H_01_R)
        self.R_m10 = np.array(self.R_m10)
        self.H_m10_R = np.array(self.H_m10_R)
        
        if self.verbose:
            print(f"Principal layer partitioning complete:")
            print(f"  H_00: {len(self.H_00_R)} hopping matrices")
            print(f"  H_01: {len(self.H_01_R)} hopping matrices")
            print(f"  H_10: {len(self.H_m10_R)} hopping matrices\n")

    def addExchangeSplitting(self, projection, exchange_vector):
        """
        Adds an exchange splitting (Zeeman-like term) to specified atoms and shells.
        
        Parameters
        ----------
        projection : list of tuples
            Format: [(atom_id, (shell_id_1, shell_id_2, ...)), ...]
            Specifies which atoms and shells to apply the splitting to.
        exchange_vector : np.ndarray or list
            A 3-element array [Mx, My, Mz] representing the strength of the 
            splitting in eV for the sigma_x, sigma_y, and sigma_z directions.
        """
        Mx, My, Mz = exchange_vector
        
        if self.verbose: 
            print(f"\n--- Adding Exchange Splitting ---")
            print(f"Magnetic Vector (Mx, My, Mz) = ({Mx}, {My}, {Mz}) eV")
            
        # 1. Locate the R=(0,0,0) home cell matrix
        home_R_tuple = (0, 0, 0)
        if hasattr(self, 'iRvecIndex') and home_R_tuple in self.iRvecIndex:
            iR0 = self.iRvecIndex[home_R_tuple]
        else:
            iR0 = -1
            for i, R in enumerate(self.iRvec):
                if tuple(R) == home_R_tuple:
                    iR0 = i
                    break
        assert iR0 != -1, "Could not find the R=(0,0,0) home cell in the Hamiltonian!"
        
        # 2. Map and apply the exchange splitting
        half_norbs = self.norbs // 2
        applied_orbitals = 0
        
        for iatom, shells in projection:
            atom_info = self.atomicOrbInfoList[iatom]
            for ishell in shells:
                shell_indices = atom_info["orbital index"][ishell]
                
                for iorb in shell_indices:
                    # Prevent double counting if the shell list already contains both up and down indices
                    if iorb >= half_norbs:
                        continue
                        
                    I_up = iorb
                    I_dn = I_up + half_norbs
                    
                    # Apply Mz * sigma_z (Diagonal)
                    if abs(Mz) > 1e-8:
                        self.ham_R[iR0, I_up, I_up] += Mz
                        self.ham_R[iR0, I_dn, I_dn] -= Mz
                    
                    # Apply Mx * sigma_x + My * sigma_y (Off-diagonal spin-flip terms)
                    if abs(Mx) > 1e-8 or abs(My) > 1e-8:
                        # H_up_dn = Mx - i My
                        # H_dn_up = Mx + i My
                        self.ham_R[iR0, I_up, I_dn] += (Mx - 1j * My)
                        self.ham_R[iR0, I_dn, I_up] += (Mx + 1j * My)
                    
                    applied_orbitals += 1
                    
        if self.verbose: 
            print(f"Successfully applied exchange splitting to {applied_orbitals} physical spatial orbitals.")
            
        # 3. Synchronize dependent system matrices
        if hasattr(self, 'energy_R') and self.energy_R is not None:
            # For OpenMX models where energy_R tracks ham_R
            self.energy_R[iR0, :, :] = self.ham_R[iR0, :, :].copy()
            
        # Rebuild dependent operators
        self.getRHam_R()
        if self.RRHam_R is not None: self.getRRHam_R()
        if self.RRRHam_R is not None: self.getRRRHam_R()
        if hasattr(self, 'getSpinMatrice'): self.getSpinMatrice()
        if hasattr(self, 'getOrbitalMatrice'): self.getOrbitalMatrice()

    def rotateExchangeField(self, projection, theta, phi):
        """
        Rotates the on-site exchange field for specified orbitals.
        Rotation is applied via Euler angles: around y-axis by theta, then z-axis by phi.
        """
        if self.verbose: 
            print(f"\n--- Rotating Exchange Field (theta={theta:.4f}, phi={phi:.4f}) ---")
            
        home_R_tuple = (0, 0, 0)
        iR0 = self.iRvecIndex.get(home_R_tuple, -1)
        if iR0 == -1:
            for i, R in enumerate(self.iRvec):
                if tuple(R) == home_R_tuple:
                    iR0 = i; break
        assert iR0 != -1, "Could not find home cell R=(0,0,0)"
        
        half_norbs = self.norbs // 2
        count = 0
        
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        cos_p, sin_p = np.cos(phi), np.sin(phi)
        
        for iatom, shells in projection:
            atom_info = self.atomicOrbInfoList[iatom]
            for ishell in shells:
                for iorb in atom_info["orbital index"][ishell]:
                    if iorb >= half_norbs: continue # Only loop over spatial orbitals once
                    I_up = iorb
                    I_dn = I_up + half_norbs
                    
                    # Extract 2x2 block
                    H_uu = self.ham_R[iR0, I_up, I_up]
                    H_dd = self.ham_R[iR0, I_dn, I_dn]
                    H_ud = self.ham_R[iR0, I_up, I_dn]
                    H_du = self.ham_R[iR0, I_dn, I_up]
                    
                    # Project to SU(2) Pauli basis
                    S0 = 0.5 * (H_uu + H_dd)
                    Sz = 0.5 * (H_uu - H_dd)
                    Sx = 0.5 * (H_ud + H_du)
                    Sy = 0.5j * (H_ud - H_du)
                    
                    # 1. Rotate around y-axis by theta
                    Sx_p = Sx * cos_t + Sz * sin_t
                    Sy_p = Sy
                    Sz_p = -Sx * sin_t + Sz * cos_t
                    
                    # 2. Rotate around z-axis by phi
                    Sx_pp = Sx_p * cos_p - Sy_p * sin_p
                    Sy_pp = Sx_p * sin_p + Sy_p * cos_p
                    Sz_pp = Sz_p
                    
                    # Reconstruct and replace
                    self.ham_R[iR0, I_up, I_up] = S0 + Sz_pp
                    self.ham_R[iR0, I_dn, I_dn] = S0 - Sz_pp
                    self.ham_R[iR0, I_up, I_dn] = Sx_pp - 1j * Sy_pp
                    self.ham_R[iR0, I_dn, I_up] = Sx_pp + 1j * Sy_pp
                    count += 1
                    
        if self.verbose: print(f"Rotated exchange field for {count} spatial orbitals.")
        
        # Synchronize dependent matrices
        if hasattr(self, 'energy_R') and self.energy_R is not None:
            self.energy_R[iR0] = self.ham_R[iR0].copy()
            
        self.getRHam_R()
        if self.RRHam_R is not None: self.getRRHam_R()
        if self.RRRHam_R is not None: self.getRRRHam_R()
        if hasattr(self, 'getSpinMatrice'): self.getSpinMatrice()
        if hasattr(self, 'getOrbitalMatrice'): self.getOrbitalMatrice()

    def scaleSOC(self, lam=1.0, projection=[]):
        """
        Scales the SOC-induced spin-flip hopping amplitudes (tx, ty, tz) by lambda,
        leaving the charge transport hopping (t0) intact.
        """
        if self.verbose: 
            print(f"\n--- Scaling SOC Hopping Terms by lambda={lam:.4f} ---")
            
        half_norbs = self.norbs // 2
        
        if len(projection)==0: # empty projection default to the whole system
            target_orbs = np.arange(half_norbs)    
        else: # Collect target spatial orbital indices from the projection list
            target_orbs = []
            for iatom, shells in projection:
                atom_info = self.atomicOrbInfoList[iatom]
                for ishell in shells:
                    for iorb in atom_info["orbital index"][ishell]:
                        if iorb < half_norbs:
                            target_orbs.append(iorb)
        
        count = 0
        # Iterate over all unit cells to capture inter-cell hopping
        for iR in range(self.nRvec):
            # Iterate over all pairs of orbitals in the target list
            for i_up in target_orbs:
                for j_up in target_orbs:
                    i_dn = i_up + half_norbs
                    j_dn = j_up + half_norbs
                    
                    # Extract 2x2 inter-site hopping block
                    H_uu = self.ham_R[iR, i_up, j_up]
                    H_dd = self.ham_R[iR, i_dn, j_dn]
                    H_ud = self.ham_R[iR, i_up, j_dn]
                    H_du = self.ham_R[iR, i_dn, j_up]
                    
                    # Project to SU(2) Pauli basis
                    t0 = 0.5 * (H_uu + H_dd)
                    tz = 0.5 * (H_uu - H_dd)
                    tx = 0.5 * (H_ud + H_du)
                    ty = 0.5j * (H_ud - H_du)
                    
                    # Scale only the SOC-relevant Pauli components
                    tz_p = tz * lam
                    tx_p = tx * lam
                    ty_p = ty * lam
                    
                    # Reconstruct and replace
                    self.ham_R[iR, i_up, j_up] = t0 + tz_p
                    self.ham_R[iR, i_dn, j_dn] = t0 - tz_p
                    self.ham_R[iR, i_up, j_dn] = tx_p - 1j * ty_p
                    self.ham_R[iR, i_dn, j_up] = tx_p + 1j * ty_p
                    
                    count += 1
                    
        if self.verbose: print(f"Successfully scaled {count} hopping matrix blocks.")
        
        # Synchronize dependent matrices
        if hasattr(self, 'energy_R') and self.energy_R is not None:
            self.energy_R = self.ham_R.copy()
            
        self.getRHam_R()
        if self.RRHam_R is not None: self.getRRHam_R()
        if self.RRRHam_R is not None: self.getRRRHam_R()
        if hasattr(self, 'getSpinMatrice'): self.getSpinMatrice()
        if hasattr(self, 'getOrbitalMatrice'): self.getOrbitalMatrice()

    def applyFloquetPeierlsDressing(self, vectorPotentialCos, vectorPotentialSin=None):
        """
        Apply the stage-1 off-resonant Floquet Peierls dressing in place.

        The time-dependent vector potential is represented as
        ``A(t) = A_cos cos(omega t) + A_sin sin(omega t)`` in Cartesian
        inverse-Angstrom units. The zeroth Fourier component renormalizes each
        hopping by

        ``J0(sqrt((A_cos dot d)^2 + (A_sin dot d)^2))``,

        where ``d = R + r_target - r_source`` is the hopping displacement in
        Angstrom. For linearly polarized light, pass only ``vectorPotentialCos``.
        For circular or elliptical light, pass both cosine and sine components.

        This is only the time-averaged Peierls term ``H_eff = H_0``. It does not
        include the van Vleck commutator correction.
        """
        vectorPotentialCos = np.asarray(vectorPotentialCos, dtype=float).reshape(3)
        if vectorPotentialSin is None:
            vectorPotentialSin = np.zeros(3)
        vectorPotentialSin = np.asarray(vectorPotentialSin, dtype=float).reshape(3)

        if self.verbose:
            print("\n--- Applying stage-1 Floquet Peierls dressing ---")
            print(f"A_cos = {vectorPotentialCos} 1/Angstrom")
            print(f"A_sin = {vectorPotentialSin} 1/Angstrom")

        r_source = self.rList[:, :, None]
        r_target = self.rList[:, None, :]
        intra_cell_displacement = r_target - r_source

        dressing_R = np.zeros((self.nRvec, self.norbs, self.norbs), dtype=float)
        for iR, Rvec in enumerate(self.iRvec):
            displacement = (Rvec @ self.lattice)[:, None, None] + intra_cell_displacement
            phase_cos = np.einsum('a,aij->ij', vectorPotentialCos, displacement, optimize=True)
            phase_sin = np.einsum('a,aij->ij', vectorPotentialSin, displacement, optimize=True)
            dressing_R[iR] = j0(np.sqrt(phase_cos**2 + phase_sin**2))

        if hasattr(self, 'ham_R') and self.ham_R is not None:
            self.ham_R = self.ham_R * dressing_R
        if hasattr(self, 'energy_R') and self.energy_R is not None:
            self.energy_R = self.energy_R * dressing_R

        self.getRHam_R()
        if self.RRHam_R is not None: self.getRRHam_R()
        if self.RRRHam_R is not None: self.getRRRHam_R()
        if hasattr(self, 'getSpinMatrice'): self.getSpinMatrice()
        if hasattr(self, 'getOrbitalMatrice'): self.getOrbitalMatrice()

        if self.verbose:
            print("--- Stage-1 Floquet Peierls dressing complete ---\n")

    def getFloquetPeierlsDressedModel(self, vectorPotentialCos, vectorPotentialSin=None):
        """
        Return a deep-copied model with stage-1 Floquet Peierls dressing applied.

        See ``applyFloquetPeierlsDressing`` for the physical convention and
        units. This non-mutating wrapper is usually safer for comparing dressed
        and undressed exchange couplings.
        """
        dressed_model = copy.deepcopy(self)
        dressed_model.applyFloquetPeierlsDressing(vectorPotentialCos, vectorPotentialSin)
        return dressed_model

    def rotateSpinBasisToGlobal(self, alpha, beta, gamma=0.0):
        """
        Rotates the tight-binding basis from the VASP local SAXIS quantization frame 
        back to the standard global Cartesian frame (sigma_z basis).
        
        Parameters:
        -----------
        alpha, beta, gamma : float
            The Euler angles outputted in the VASP OUTCAR (in radians).
        """
        
        if self.verbose: 
            print(f"\n--- Rotating Spin Basis to Global Frame ---")
            print(f"Euler Angles: ALPHA={alpha:.6f}, BETA={beta:.6f}, GAMMA={gamma:.6f}")
            
        # 1. Define standard Pauli matrices
        sz = np.array([[1, 0], [0, -1]], dtype=complex)
        sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
        I2 = np.eye(2, dtype=complex)
        
        # 2. VASP spinor transformation matrix W (Local -> Global)
        # W = exp(-i*alpha*sz/2) * exp(-i*beta*sy/2) * exp(-i*gamma*sz/2)
        def exp_ip(theta, sigma):
            return np.cos(theta / 2.0) * I2 - 1j * np.sin(theta / 2.0) * sigma
            
        W_alpha = exp_ip(alpha, sz)
        W_beta  = exp_ip(beta, sy)
        W_gamma = exp_ip(gamma, sz)
        
        W = W_alpha @ W_beta @ W_gamma
        
        # 3. Construct the full spatial-spin transformation block matrix (W_full = W \otimes I_spatial)
        half_norbs = self.norbs // 2
        W_full = np.zeros((self.norbs, self.norbs), dtype=complex)
        
        # Map the 2x2 SU(2) rotation onto the upper/lower spin blocks
        W_full[:half_norbs, :half_norbs] = W[0, 0] * np.eye(half_norbs)
        W_full[:half_norbs, half_norbs:] = W[0, 1] * np.eye(half_norbs)
        W_full[half_norbs:, :half_norbs] = W[1, 0] * np.eye(half_norbs)
        W_full[half_norbs:, half_norbs:] = W[1, 1] * np.eye(half_norbs)
        
        W_full_dag = W_full.conj().T
        
        # Helper function to apply the SU(2) rotation to multi-dimensional matrices
        def transform_matrix(M):
            if M is None: return None
            # Using np.matmul (@) perfectly broadcasts over the leading (..., nRvec) dimensions
            # Computing: H_global = W_full @ H_local @ W_full_dag
            return W_full @ M @ W_full_dag
            
        # 4. Rotate all relevant System matrices
        if hasattr(self, 'ham_R') and self.ham_R is not None:
            self.ham_R = transform_matrix(self.ham_R)
            if self.verbose: print("Transformed Hamiltonian (ham_R).")
            
        if hasattr(self, 'overlap_R') and self.overlap_R is not None:
            self.overlap_R = transform_matrix(self.overlap_R)
            if self.verbose: print("Transformed Overlap Matrix (overlap_R).")
            
        if hasattr(self, 'energy_R') and self.energy_R is not None:
            self.energy_R = transform_matrix(self.energy_R)
            if self.verbose: print("Transformed Energy Matrix (energy_R).")
            
        if hasattr(self, 'AA_R_W') and self.AA_R_W is not None:
            self.AA_R_W = transform_matrix(self.AA_R_W)
            if self.verbose: print("Transformed Position Operator (AA_R_W).")
            
        #if hasattr(self, 'RHam_R') and self.RHam_R is not None:
        #    self.RHam_R = transform_matrix(self.RHam_R)
        #    if verbose: print("Transformed Velocity Operator (RHam_R).")
            
        if hasattr(self, 'RO_R') and self.RO_R is not None:
            self.RO_R = transform_matrix(self.RO_R)
            if self.verbose: print("Transformed Overlap Velocity Operator (RO_R).")
            
        # 5. Trigger updates for internal operator dependencies
        self.getRHam_R()
        if self.RRHam_R is not None: self.getRRHam_R()
        if self.RRRHam_R is not None: self.getRRRHam_R()
        if hasattr(self, 'getSpinMatrice'): self.getSpinMatrice()
        if hasattr(self, 'getOrbitalMatrice'): self.getOrbitalMatrice()
        
        if self.verbose: print("--- Spin Basis Transformation Complete ---\n")

    @property
    def Vcell(self):
        return np.linalg.det(self.lattice)

from .system_w90 import System_w90
from .system_omx import System_omx
from .system_tb import System_tb
