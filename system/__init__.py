import numpy as np
from scipy import linalg
from auxiliary import showElapsedTime, pseudoRotation, print_matrix, getWholeSymmetryOperator
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
        # spin convention, "VASP" collect the same spin at one block, while "QE" collect the same orbital at one small block
        self.format="VASP"
        # num. of spin tunnels
        self.ispin=-1
        # spin matrix, in basis of home cell
        self.spin:np.ndarray
        # lattice matrix
        self.lattice:np.ndarray
        # reciprocal lattice
        self.reciLattice:np.ndarray
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
    
    def getRHam_R(self):
        '''
        Calc. the matrix (R+d) * H_R, in unit length*energy, usually angstrom*eV. 
        
        Indexing: RH_R[iR1,iR2,iR3,idirection,iorb,jorb]

        The matrix element of H_R is t_{d+R}, where d is the dist. between orbs. in home cell, R is Bravais vector (both in cart. coors.).
        So, the matrix element of (R+d) * H_R is R t_{d+R}. Since R+d is rank-1 and 3-d, R t_{d+R} is also 3-d.
        '''
        self.RHam_R=np.zeros((self.nRvec,3,self.norbs,self.norbs),dtype=complex)
        print(f"calculating R \dot H_R on lattice with {self.nRvec} lattice points")
        dr=self.rList[:,:,None]-self.rList[:,None,:] # distance in cell, r_source - r_target
        for iR, Rvec in enumerate(self.iRvec):
            self.RHam_R[iR]=self.ham_R[iR,None,:,:]*((Rvec@self.lattice)[:,None,None] - dr)

    def getSpinMatrice(self,projectedOrbIndex=[]):
        '''
        Calc. the spin matrices.

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
        #self.spin[3,:self.norbs//2,:self.norbs//2] =  II
        #self.spin[3,self.norbs//2:,self.norbs//2:] =  II

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
    
    def symmerizeHamAndOverlap(self,spinor=False,saxis:np.ndarray=np.array([0,0,-1]),spin_basis:np.ndarray=np.zeros((2,2),dtype=complex),
                        symOpSelection:list=[],atomicMagMomList:list=[],format="w90",
                        print_row=(0,),print_col=(0,), # To print specialized block of local Hams.
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
        def readSpinWFCSymOp(symRotateList):
            spinRotateList=[]
            L=self.lattice
            L_inv=np.linalg.inv(L)
            for rotate in symRotateList:
                if not spinor:
                    spinRotateList.append(np.eye(2))
                    continue
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
                spinRotateList.append(spinRotation)
            return spinRotateList        
        def getSymOpsAndSymSiteMap(atomicMagMomList=[]):
            # constructe all sym. ops and fetch ion mapping under sym. ops
            symRotateList=[]
            symTranslateList=[]
            symIonMap=[]
            sym= getWholeSymmetryOperator(self.posinfo)
            rotation, translation= sym['rotations'], sym['translations']
            translation=[vecAround(translate)%1 for translate in translation]
            print('Lattice symmetry group order: %d'%rotation.shape[0])
            print('Warning: at present, pls. ensure that all the translation vectors are zero')
            print('         which can be achieve by chosing correct origin')
            #print(sym)
            #exit()
            for rotate, translate in zip(rotation,translation):
                #print(rotate)
                #print(translate)
                symRotateList.append(rotate)
                symTranslateList.append(translate)
                subMap=[]
                for iion, ion_coor in enumerate(self.posinfo['pos ions (frac.)']):
                    ion_coor_trans=ion_coor.dot(rotate.T)+translate
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
            
            spinRotateList=readSpinWFCSymOp(symRotateList)
            # check if system involves nonzero local mag. mom.
            sx=np.array([[0,1],[1,0]],dtype=complex)
            sy=np.array([[0,-1j],[1j,0]],dtype=complex)
            sz=np.array([[1,0],[0,-1]],dtype=complex)
            magmom_list=atomicMagMomList
            if spinor: # find proper sub-group consist with magnetism and time reversal symmetry
                if abs(sum(sum(spin_basis.dot(spin_basis.conjugate().T)-np.eye(2)))-2)<SYMPREC:
                    Us=spin_basis
                else:
                    print("Warning: spin_basis is not set or not set correctly. So, I have to calc. this via saxis.")
                    print("         However, in some cases the saxis-inferred spin space is not satisfied.")
                    print("         Be caucious and re-check the result!")
                    assert np.sqrt(sum(saxis*saxis))==1, 'saxis should be normalized vector'
                    _,Us=np.linalg.eigh(saxis[0]*sx+saxis[1]*sy+saxis[2]*sz)
                    print("Calculated spin basis:")
                    print(Us)
                new_symRotateList, new_symTranslateList, new_symIonMap, new_spinRotateList, new_timeReversalList= [], [], [], [], []
                for isym in range(len(symIonMap)):
                    rotate_r, rotate_s, translate, ionMap= symRotateList[isym], spinRotateList[isym], symTranslateList[isym], symIonMap[isym]
                    magmom_rotated_list=[]
                    for m in atomicMagMomList:
                        magmom_rotated_list.append(pseudoRotation(rotate_r,self.lattice,m))
                    #transformed_magmom=[rotate_s.dot(m) for m in magmom_list]
                    diff_magmom=[magmom_rotated_list[i_ion]-magmom_list[ionMap[i_ion]] for i_ion in range(sum(self.posinfo['nions']))]
                    if sum([np.sqrt(np.dot(dmag,dmag)) for dmag in diff_magmom]) < SYMPREC:
                        print('crystal sym. op. preserved:')
                        print(rotate_r)
                        print(translate)
                        #print(rotate_s)
                        #print(Usy)
                        print_matrix(Us.conjugate().T @ rotate_s @ Us, rowIndex=(0,1,),colIndex=(0,1,))
                        print('dmag:')
                        for idmag, dmag in enumerate(diff_magmom):
                            if np.dot(magmom_list[idmag],magmom_list[idmag])>SYMPREC: 
                                print("%3d-%3d %6.3f%6.3f%6.3f->%6.3f%6.3f%6.3f?=%6.3f%6.3f%6.3f dmag:%6.3f%6.3f%6.3f"%(
                                    idmag,ionMap[idmag],
                                    *atomicMagMomList[idmag],*magmom_rotated_list[idmag],*atomicMagMomList[ionMap[idmag]],
                                    *dmag))
                        new_symRotateList.append(rotate_r)
                        new_symTranslateList.append(translate)
                        new_symIonMap.append(ionMap)
                        new_spinRotateList.append(Us.conjugate().T @ rotate_s @ Us)
                        new_timeReversalList.append(False)
                    else:
                        print('crystal sym. op. broken due to magnetism')
                        print(rotate_r)
                        print(translate)
                        print_matrix(Us.conjugate().T @ rotate_s @ Us, rowIndex=(0,1,),colIndex=(0,1,))
                        print('dmag:')
                        for idmag, dmag in enumerate(diff_magmom):
                            if np.dot(magmom_list[idmag],magmom_list[idmag])>SYMPREC: 
                                print("%3d-%3d %6.3f%6.3f%6.3f->%6.3f%6.3f%6.3f?=%6.3f%6.3f%6.3f dmag:%6.3f%6.3f%6.3f"%(
                                    idmag,ionMap[idmag],
                                    *atomicMagMomList[idmag],*magmom_rotated_list[idmag],*atomicMagMomList[ionMap[idmag]],
                                    *dmag))
                    #continue
                    # check time reversal symmetry
                    rotate_s_tr=rotate_s@np.array([[0,-1],[1,0]])
                    magmom_rotated_list=[]
                    for m in atomicMagMomList:
                        magmom_rotated_list.append(-pseudoRotation(rotate_r,self.lattice,m))
                    #transformed_magmom=[rotate_s.dot(m) for m in magmom_list]
                    diff_magmom=[magmom_rotated_list[i_ion]-magmom_list[ionMap[i_ion]] for i_ion in range(sum(self.posinfo['nions']))]
                    if sum([np.sqrt(np.dot(dmag,dmag)) for dmag in diff_magmom]) < SYMPREC:
                        print('crystal sym. op. + time reversal preserved:')
                        print(rotate_r)
                        print(translate)
                        #print(rotate_s_tr)
                        #print(Usy)
                        print_matrix(Us.T.conjugate() @ rotate_s_tr @ Us.conjugate(), rowIndex=(0,1,),colIndex=(0,1,))
                        print('dmag:')
                        for idmag, dmag in enumerate(diff_magmom):
                            if np.dot(magmom_list[idmag],magmom_list[idmag])>SYMPREC: 
                                print("%3d-%3d %6.3f%6.3f%6.3f->%6.3f%6.3f%6.3f?=%6.3f%6.3f%6.3f dmag:%6.3f%6.3f%6.3f"%(
                                    idmag,ionMap[idmag],
                                    *atomicMagMomList[idmag],*magmom_rotated_list[idmag],*atomicMagMomList[ionMap[idmag]],
                                    *dmag))
                        new_symRotateList.append(rotate_r)
                        new_symTranslateList.append(translate)
                        new_symIonMap.append(ionMap)
                        new_spinRotateList.append(Us.T.conjugate().dot(rotate_s_tr).dot(Us.conjugate()))
                        new_timeReversalList.append(True)
                    else:
                        print('crystal sym. op. + time reversal broken due to magnetism')
                        print(rotate_r)
                        print(translate)
                        print_matrix(Us.T.conjugate() @ rotate_s_tr @ Us.conjugate(), rowIndex=(0,1,),colIndex=(0,1,))
                        print('dmag:')
                        for idmag, dmag in enumerate(diff_magmom):
                            if np.dot(magmom_list[idmag],magmom_list[idmag])>SYMPREC: 
                                print("%3d-%3d %6.3f%6.3f%6.3f->%6.3f%6.3f%6.3f?=%6.3f%6.3f%6.3f dmag:%6.3f%6.3f%6.3f"%(
                                    idmag,ionMap[idmag],
                                    *atomicMagMomList[idmag],*magmom_rotated_list[idmag],*atomicMagMomList[ionMap[idmag]],
                                    *dmag))
                symRotateList, symTranslateList, symIonMap, spinRotateList=new_symRotateList, new_symTranslateList, new_symIonMap, new_spinRotateList
                return {'rotations':symRotateList, 'translations':symTranslateList, 'atom map':symIonMap, 'spin rotations':spinRotateList, 'time reversal':new_timeReversalList}
            else:
                return {'rotations':symRotateList, 'translations':symTranslateList, 'atom map':symIonMap, 'spin rotations':spinRotateList, 'time reversal':[False for sympos in symRotateList]}       
        def generateAtomicWFCSymOp(symRotateList,format="w90"):
            '''generate transformation matrix of cubic harmonics for each sym. Op
            output:
            data[0]:1x1 array: s; 
            data[1]:3x3 array: pz,px,py (w90 format); px,py,pz (openmx format)
            data[2]:5x5 array: dz2, dxz, dyz, dx2-y2, dxy (w90 format); dz2, dx2-y2, dxy, dxz, dyz (openmx format) 
            data[3]:7x7 array: fz3, fxz2, fyz2, fz(x2-y2), fxyz, fx(x2-3y2), fy(3x2-y2)
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
                if format=="openmx":
                    return np.array([x,y,z])
                return np.array([z,x,y]) # in order: pz, px, py, 
            def dOrb(vec,format="w90"):
                '''d-orbitals defined in line with wannier90 v2.1
                   order: dz2, dxz, dyz, dx2-y2, dxy
                '''
                x,y,z=vec
                if format=="openmx":
                    return np.array([
                        sqrt(5/16)*(3*z*z-1),
                        sqrt(15/16)*(x*x-y*y),
                        sqrt(15/4)*x*y,
                        sqrt(15/4)*x*z,
                        sqrt(15/4)*y*z,
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
                return np.array([
                    sqrt(7)   *(5*z**3-3*z),
                    sqrt(21/2)*(5*z**2-1)*x,
                    sqrt(21/2)*(5*z**2-1)*y,
                    sqrt(105) *(x*x-y*y)*z,
                    sqrt(105) *(x*y*2)*z,
                    sqrt(35/2)*(x*x-3*y*y)*x,
                    sqrt(35/2)*(3*x*x-y*y)*y
                ])/4#/sqrt(np.pi)
            
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
            print('determinals of d-orb basis:',np.linalg.det(dList))
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
            print('determinals of f-orb basis',np.linalg.det(fList))
            ao_f_basis_std=np.linalg.inv(fList)

            atomicWaveSymOpList=[]
            L=self.lattice
            L_inv=np.linalg.inv(L)
            for rotate in symRotateList:
                rotate_xyz=L_inv.dot(rotate.T).dot(L)

                pTransList=np.array([pOrb(n_vec.dot(rotate_xyz),format=format) for n_vec in pNBasis])
                ao_p_basis_trans=pTransList.T

                dTransList=np.array([dOrb(n_vec.dot(rotate_xyz),format=format) for n_vec in dNBasis])
                ao_d_basis_trans=dTransList.T

                fTransList=np.array([fOrb(n_vec.dot(rotate_xyz),format=format) for n_vec in fNBasis])
                ao_f_basis_trans=fTransList.T

                atomicWaveSymOpList.append([
                    np.array([1]),
                    np.dot(ao_p_basis_trans,ao_p_basis_std.T),
                    np.dot(ao_d_basis_trans,ao_d_basis_std.T),
                    np.dot(ao_f_basis_trans,ao_f_basis_std.T)
                    ])
            return atomicWaveSymOpList
        
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

        # obtain sym. Ops. of atomic wavefunctions
        sym= getSymOpsAndSymSiteMap(atomicMagMomList=atomicMagMomList)
        symRotateList, symTranslateList, symSiteMapList, symSpinRotateList, symTimeReversalList=sym['rotations'], sym['translations'], sym['atom map'], sym['spin rotations'], sym['time reversal']
        if len(symOpSelection)>0:
            symRotateList=[symRotateList[isym] for isym in symOpSelection]
            symTranslateList=[symTranslateList[isym] for isym in symOpSelection]
            symSiteMapList=[symSiteMapList[isym] for isym in symOpSelection]
            symSpinRotateList=[symSpinRotateList[isym] for isym in symOpSelection]
            symTimeReversalList=[symTimeReversalList[isym] for isym in symOpSelection]
        atomicWaveSymOpList= generateAtomicWFCSymOp(symRotateList,format=format)
        #print_matrix(atomicWaveSymOpList[1][2],rowIndex=(0,1,2,3,4,),colIndex=(0,1,2,3,4,))
        #exit()
        # in now set up, the lattice is column-major and the reci-lat is row-major, which is opposite to VASP exactly
        # and the R is the rotation with cartesian basis
        lat,lat_inv=self.lattice.T,np.linalg.inv(self.lattice.T)
        self.symOpList=[]
        for rotate, translate, timeReversal in zip (symRotateList,symTranslateList,symTimeReversalList):
            self.symOpList.append(
                Symmetry(R=lat@rotate@lat_inv,TR=timeReversal)
            )
        self.pointGroup=Group(self.symOpList,real_lattice=self.lattice,recip_lattice=np.linalg.inv(self.lattice.T)*2*np.pi)
        #exit()
        # construct local sym. Op. for each sym. op.
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
        
        #np.set_printoptions(precision=4)
        # impose symmetry constrains
        #'''
        # In the following, we only need the mapping from iaotm to iorbs
        atomicOrbMap=[]
        for atom in self.atomicOrbInfoList:
            orbIndex=[]
            for shell in atom['orbital index']: orbIndex += shell
            atomicOrbMap.append(orbIndex)
        
        print(f"Enforce the hermit condition for energy and overlap matrix {self.nRvec=} {self.norbs=}")
        hamR_herm    =np.zeros((self.nRvec,self.norbs,self.norbs),dtype=complex)
        #if self.overlap: overlapInVecForm_herm=np.zeros((nR,norb,norb),dtype=complex)
        for iR, R in enumerate(self.iRvec):
            iR_dual=self.iRvecIndex[(-R[0],-R[1],-R[2])]
            if self.overlap:
                hamR_herm[iR]    = (self.energy_R[iR]    + self.energy_R[iR_dual].T.conj())/2
            else:
                hamR_herm[iR]    = (self.ham_R[iR]    + self.ham_R[iR_dual].T.conj())/2
            #if self.overlap: overlapInVecForm_herm[iR] = (self.overlapInVecForm[iR] + self.overlapInVecForm[iR_dual].T.conj())/2
        ham_R   =hamR_herm
        #if self.overlap: self.overlapInVecForm=overlapInVecForm_herm
        showElapsedTime(extraInfo="Hermit processing finished")
        
        def constructSymmetryBondSet(R0,iatom_source,iatom_target,verbose=False):
            iR0=self.iRvecIndex[R0]

            source_coor=self.posinfo['pos ions (frac.)'][iatom_source]  # coor. of source atom
            # calc. the coor. of target atom, according to the bond info
            target_coor=self.posinfo['pos ions (frac.)'][iatom_target] + R0
            dR_xyz=(target_coor-source_coor)@self.lattice
            dR_xyz_len=np.sqrt(dR_xyz@dR_xyz)

            symmerizedBondAndOverlapSet={'preserve':True,'bond set':[],'bond length':dR_xyz_len} # recording the symmerization process
            if verbose: print(f'start construct symmetry recipe for bond {R0}({iR0}) {iatom_source} {iatom_target}')
            if verbose: print_matrix(ham_R[iR0,atomicOrbMap[iatom_source],:][:,atomicOrbMap[iatom_target]],rowIndex=print_row,colIndex=print_col)
            for rotate, translate, localSymOp, localSymOpInv, siteMap, timeReversal in zip(symRotateList,symTranslateList,localSymOpList,localSymOpInvList,symSiteMapList,symTimeReversalList):
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
                    bond   =ham_R[iR,atomicOrbMap[iatom_source_trans],:][:,atomicOrbMap[iatom_target_trans]].copy()
                        
                    if timeReversal: 
                        #bond_rotated=((localSymOp[iatom_source_trans].T.conj())@bond@(localSymOp[iatom_target_trans])).conj()
                        bond_rotated=((localSymOpInv[iatom_source_trans])@bond@(localSymOp[iatom_target_trans])).conj()
                    else:
                        #bond_rotated=(localSymOp[iatom_source_trans].T.conj())@bond@localSymOp[iatom_target_trans]
                        bond_rotated=(localSymOpInv[iatom_source_trans])@bond@localSymOp[iatom_target_trans]
                
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
                            'time reversal':timeReversal,
                            'back to I hopping':bond_rotated,
                        })
                    symmerizedBondAndOverlapSet['bond set'].append({
                            'conjugation': True,
                            'on grid': True,
                            'bond info':(iatom_target_trans,iatom_source_trans,-R[0],-R[1],-R[2]),
                            'original hopping':bond.conj().T,
                            'sym. op. on source':localSymOp[iatom_source_trans],
                            'sym. op. on target':localSymOpInv[iatom_target_trans],
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
                opSource, opTarget= bond['sym. op. on source'], bond['sym. op. on target']
                if bond['time reversal']:
                    bond['symmerized hopping'] = opSource @ (bond_matrix_avg.conj()) @ opTarget#.conj().T
                else:
                    bond['symmerized hopping'] = opSource @ bond_matrix_avg @ opTarget#.conj().T

                if bond['conjugation']:                
                    bond['symmerized hopping'] = bond['symmerized hopping'].conj().T

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
                        ham_sym[iR, iGlobalOrbI, iGlobalOrbJ]=bond_t_matrix[iLocalOrbI, iLocalOrbJ]
        
        #symBondSet=constructSymmetryBondSet((0,0,0),0,0,verbose=True)
        #res=calcSymmerizedBond(symBondSet,verbose=True)
        #ham_R_sym   =np.zeros((self.nRvec,self.norbs,self.norbs),dtype=complex)
        #setSymmerizedBond(symBondSet,ham_R)
        #print("Check consistency of symmerization =====>>>>")
        #symBondSet=constructSymmetryBondSet((0,0,0),0,0,verbose=True)
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
        ham_R_sym   =np.zeros((self.nRvec,self.norbs,self.norbs),dtype=complex)
        for iset, symBondSet in enumerate(allSymmerizedBondAndOverlap):
            setSymmerizedBond(symBondSet,ham_R_sym)
        showElapsedTime(extraInfo=f"symmery-constrained Ham. and overlap matrix constructed")
        #'''
        if self.overlap:
            self.energy_R=ham_R_sym
        else:
            self.ham_R=ham_R_sym

    def setFermi(self,fermi:float):
        self.fermi=fermi

    @property
    def Vcell(self):
        return np.linalg.det(self.lattice)

from .system_w90 import System_w90
from .system_omx import System_omx