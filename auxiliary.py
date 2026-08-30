'''
Log 2022/11/27
Appended lots of functions to achieve many miscs

Created on 2019 7 6
Main purpose: redefine the system-operations, such as cp, mkdir, etc.,
              so that I can get more D.O.F, i.e., error wrapping, reporting, debugging
@author: Andrew
'''
import numpy as np
import os
import shutil
import time
from math import acos
import spglib as spg
global time0
time0=0
SYMPREC=1e-4

BOHR_MAGNETON_EV_PER_TESLA = 5.7883818060e-5
HBAR_EV_FS = 0.6582119569
ELECTRON_GAMMA_RAD_PER_FS_T = 1.76085963023e-4

def getVBM(energies:np.ndarray,fermi=0.) -> np.ndarray:
    '''
    Caculate the highest occupied band energy at a grid of BZ.

    Parameters
    ----------
    - energies : array-like, float
       band energies on a grid or a path, the band index should be in the last axis
    - fermi : float
       fermi energy
    '''
    vb_condition=energies <= fermi
    low_bound = np.min(energies)
    return np.max((energies + low_bound) * vb_condition, axis=(-1,)) - low_bound

def getCBM(energies:np.ndarray,fermi=0.) -> np.ndarray:
    '''
    Caculate the lowest unoccupied band energy at a grid of BZ.

    Parameters
    ----------
    - energies : array-like, float
       band energies on a grid or a path, the band index should be in the last axis
    - fermi : float
       fermi energy
    '''
    cb_condition=energies > fermi
    up_bound = np.max(energies)
    return np.min((energies - up_bound) * cb_condition, axis=(-1,)) + up_bound

def getMSD(input,overavg=True,bestChoice=True,verbose=False):
    '''
    SHOULD BE ABANDONED, use np.std rather
    '''
    #check the shape
    shape=input.shape
    if(len(shape)==1):
        # the case of one line of data
        dim=int(shape[0])
        avg=1.*np.sum(input)/dim
        sd=0.
        diffList=[]
        for ele in input:
            diff=(ele-avg)*(ele-avg)
            diffList.append(diff)
            sd+=diff
        rmsd=np.sqrt(sd)/dim/abs(avg) if overavg else np.sqrt(sd)/dim
        if verbose:
            print('dimension is: %d'%dim)
            print('average is: %.6f'%avg)
            print('relative mean-squared-diff is: %.6f'%rmsd)
        if bestChoice:
            sortList=np.argsort(diffList)
            if verbose:
                print('the best chois is:',input[sortList[0]],' with derviation: ', diffList[sortList[0]])
            return rmsd,sortList[0]
        
    elif(len(shape)==2):
        # the case of 2x2 matrix
        ncol=int(shape[1])
        rmsdList=[]
        bestChoiceList=[]
        for icol in range(ncol):
            rmsd,bestChoiceID=getMSD(input[:,icol],bestChoice=True,verbose=verbose)
            rmsdList.append(rmsd)
            bestChoiceList.append(bestChoiceID)
        if bestChoice:
            return rmsdList,bestChoiceList
        else:
            return rmsdList
    else:
        print('cannot handle data dimension >2')
        exit()

def getCombination(inputList=[],ncomb=1):
    '''
    This function can generate combinations from extracted elements in
    inputList, e.g., if we want to select two elements from [1,2,3], 
    they maybe either [1,2], [1,3] or [2,3], thus we can use this facility as
    comb=getCombination(inputList=[1,2,3],ncomb=2)
    then we can get a list of: [ [1,2], [1,3], [2,3] ]
    
    params definition:
    inputList: the pool of elements
    ncomb    : num. of elements in combination
    '''
    #print 'conduct function'
    #print 'input:', inputList
    ntot=len(inputList)
    if(ncomb==1):
        result=[]
        for ele in inputList:
            result.append([ele])
        return result
    else:
        if(ncomb>ntot):
            print('Error: num. of combinations larger than input!')
            exit()
        else:
            result=[]
            for ihead in range(0,ntot-ncomb+1):
                headele=inputList[ihead]
                #print 'head element:',headele
                resList=list(inputList[ihead+1:ntot])
                #print 'reside list:',resList
                resCombList=getCombination(resList,ncomb=ncomb-1)
                for ele in resCombList:
                    ele.insert(0,headele)
                    #print 'ele',ele
                    result.append(list(ele))
                    #print 'ele',ele
                #result.append(result_head)
            return result

def doesTheTwoListHaveSameNumber(list1,list2):
    for n1 in list1:
        for n2 in list2:
            if n1==n2:
                return True
    return False

def findTheFirstSameNumberAmongTwoList(list1,list2): # if no same element presented, return -1
    for n1 in list1:
        for n2 in list2:
            if n1==n2:
                return n1
    return -1

def findTheSameNumberInTwoList(list1,list2):
    sameList=[]
    for n1 in list1:
        for n2 in list2:
            if n1==n2:
                sameList.append(n1)
    return sameList

class Node:
    def __init__(self,value=0,maxValue=9,leftNode=False):
        self.value=value
        self.maxValue=maxValue
        self.leftNode=leftNode
    def addone(self):
        self.value+=1
        if self.value>self.maxValue:
            # carry site
            if self.leftNode and self.leftNode.addone():
                self.value=self.leftNode.value+1
            else:
                return False
        if self.leftNode:
            self.leftNode.maxValue=self.value-1
        return True
    
def getCombinationLoop(input_list=[],ncomb=1,maxComb=-1):
    totInput=len(input_list)
    #index_list=[i for i in range(totInput)]
    # create linked list
    rootNode=Node(value=0,maxValue=totInput-1)
    for i in range(1,ncomb):
        node_i=Node(value=i,maxValue=totInput-1,leftNode=rootNode)
        rootNode=node_i
    # generate combination numbers in one loop
    def getTree(remoteNode):
        tree=[remoteNode.value]
        while(remoteNode.leftNode):
            tree.insert(0,remoteNode.leftNode.value)
            remoteNode=remoteNode.leftNode
        return tree
    comb_list=[[input_list[i] for i in getTree(rootNode)]]
    ncount=0
    while(rootNode.addone()):
        comb_list.append([input_list[i] for i in getTree(rootNode)])
        #print(getTree(rootNode))
        ncount+=1
        if maxComb>0 and ncount>=maxComb:
            break
    return comb_list

def quicksort(input_list=[]):
    '''
    THIS SHOULD BE ABANDONED, use np.sort rather
    Sort the input list and return a sort index list
    e.g. we want to sort input_list = [1, 3, 2, 5, 0]
    then the function will return the sorted INDEX:
        output = [4, 0, 2, 1, 3]
        its first element is 4, since the lowest num. 0
        is at 4th position in original input_list, and so on
    
    WARNING: the input_list will also be changed forever. 
    '''
    def __firstSort(input_list,sort_list,low,high):
        l=low;h=high
        split=input_list[low]
        
        while(l<h):
            while(input_list[h]>split):
                h-=1
                
            if(l<h):
                input_list[l]=input_list[h]
                input_list[h]=split
                
                sort_tmp=sort_list[l]
                sort_list[l]=sort_list[h]
                sort_list[h]=sort_tmp
                
                l+=1
            
            while(input_list[l]<split):
                l+=1
            
            if l<h :
                input_list[h]=input_list[l]
                input_list[l]=split
                
                sort_tmp=sort_list[l]
                sort_list[l]=sort_list[h]
                sort_list[h]=sort_tmp
                
                h-=1
        
        if l>low :
            __firstSort(input_list,sort_list,low,h-1)
            
        if h<high:
            __firstSort(input_list,sort_list,l+1,high)
    if len(input_list)==1:
        return [0]    
    if len(input_list)==0:
        print('ERROR: try to sort empty list')
        exit()
        return []
    sort_list=[]
    high=len(input_list)-1
    for i in range(0,high+1):
        sort_list.append(i)
    __firstSort(input_list,sort_list,0,high)            
    return sort_list

def quicksort_inverse(input_list=[]):
    '''
    THIS SHOULD BE ABANDONED, use np.sort rather
    Sort the input list and return a sort index list
    e.g. we want to sort input_list = [1, 3, 2, 5, 0]
    then the function will return the sorted INDEX:
        output = [3, 1, 2, 0, 4]
        its first element is 3, since the highest num. 5
        is at 3rd position in original input_list, and so on
    
    WARNING: the input_list will also be changed forever. 
    '''
    def __firstSort(input_list,sort_list,low,high):
        l=low;h=high
        split=input_list[low]
        
        while(l<h):
            while(input_list[h]<split):
                h-=1
                
            if(l<h):
                input_list[l]=input_list[h]
                input_list[h]=split
                
                sort_tmp=sort_list[l]
                sort_list[l]=sort_list[h]
                sort_list[h]=sort_tmp
                
                l+=1
            
            while(input_list[l]>split):
                l+=1
            
            if l<h :
                input_list[h]=input_list[l]
                input_list[l]=split
                
                sort_tmp=sort_list[l]
                sort_list[l]=sort_list[h]
                sort_list[h]=sort_tmp
                
                h-=1
        
        if l>low :
            __firstSort(input_list,sort_list,low,h-1)
            
        if h<high:
            __firstSort(input_list,sort_list,l+1,high)
    if len(input_list)==1:
        return [0]    
    if len(input_list)==0:
        print('ERROR: try to sort empty list')
        exit()
    sort_list=[]
    high=len(input_list)-1
    for i in range(0,high+1):
        sort_list.append(i)
    __firstSort(input_list,sort_list,0,high)            
    return sort_list

def createDir(path):
    try:
        os.mkdir(path)
    except OSError as err:
        print(err,'--> dir %s exist, skip'%path)
        #print('WARNING: %s exist'%path)

def ls(path):
    try:
        return os.listdir(path)
    except OSError as err:
        print(err, '--> cannot find %s, stop'%path)
        exit()

def cp(file_src, file_des):
    try:
        shutil.copy2(file_src,file_des)
    except IOError as err:
        print(err, '--> copy file %s to %s failed, stop'%(file_src,file_des))
        exit()

def rm(path,quiet=False):
    try:
        os.remove(path)
    except OSError as err:
        if not quiet: print(err, '--> remove file %s failed, skip'%path)

def showElapsedTime(offset=True,extraInfo=''):
    global time0
    if extraInfo!='':
        print('%s in %.3fms'%(extraInfo,(time.time()-time0)*1000),flush=True)
    if offset:
        time0=time.time()
        
def renormVec(vec:np.ndarray,cvg=SYMPREC) -> np.ndarray:
    '''shift the vector to wrap boders'''
    rVec=[]
    for x in vec:
        if abs(x)<cvg:
            rVec.append(0)
            continue
        if abs(x-1)<cvg:
            rVec.append(1)
            continue
        if abs(x+1)<cvg:
            rVec.append(-1)
            continue
        if abs(x-2)<cvg:
            rVec.append(2)
            continue
        if abs(x+2)<cvg:
            rVec.append(-2)
            continue
        rVec.append(x)
    return np.array(rVec)

def integeralizeVec(vec:np.ndarray, cvg=SYMPREC) -> np.ndarray:
    '''enforce the ele. of vec to be integer'''
    inteVec=[]
    for x in vec:
        inte_x=int(np.around(x))
        assert abs(x-inte_x)<cvg, 'make_supercell::Hr.Error: vec %.6f %.6f %.6f cannot be integeralized, symprec is %.6f'%(*vec,cvg)
        inteVec.append(inte_x)
    return np.array(inteVec)

def crossProduct(vec1:np.ndarray,vec2:np.ndarray) -> np.ndarray:
    '''
    This should be abandoned, and use np.cross
    '''
    return np.array([
        vec1[1]*vec2[2]-vec1[2]*vec2[1],
        vec1[2]*vec2[0]-vec1[0]*vec2[2],
        vec1[0]*vec2[1]-vec1[1]*vec2[0],
        ])

def SO3_SU2(rotate_xyz):
    '''
    Transform SO3 rotation matrix into SU2 rotation
    rotate_xyz is the column-wise SO3 rotation matrix, that is,
    each column is one of the basic vector of rotated frame
    And the rotation operation should be: rotate_xyz @ vec

    Return the SU2 rotation matrix rotate_su2
    the rotation should be: rotate_su2 @ spinor_vec
    '''
    x,y,z=rotate_xyz.T
    _z=np.array([x[1]*y[2]-x[2]*y[1],x[2]*y[0]-x[0]*y[2],x[0]*y[1]-x[1]*y[0]])
    if np.dot(z,_z)<0: # improper rotation
        x,y,z=-x,-y,-z # implement symmetric spin orbs

    beta=acos(z[2])
    sin_beta=np.sin(beta)
    if sin_beta>1e-4:
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

    return np.array([
            [np.exp(-1j/2*(alpha+gamma))*np.cos(beta/2),-np.exp(-1j/2*(alpha-gamma))*np.sin(beta/2)],
            [np.exp( 1j/2*(alpha-gamma))*np.sin(beta/2), np.exp( 1j/2*(alpha+gamma))*np.cos(beta/2)]
            ])#.T.conjugate()

def spinorToR(spinor):
    '''
    Convert a 2-component spinor into vector, via the expectations on Pauli matrices
    '''
    sigmax=np.array([[0,1],[1,0]])
    sigmay=np.array([[0,-1j],[1j,0]])
    sigmaz=np.array([[1,0],[0,-1]])
    return np.array([
        spinor.conjugate()@sigmax@spinor,
        spinor.conjugate()@sigmay@spinor,
        spinor.conjugate()@sigmaz@spinor
    ]).real

def RToSpinor(R):
    '''
    Convert an (supposed) axial vector into spinor. 
    Not satisfacted by now, since it's not one to one map
    '''
    if R@R<1e-4: return np.array([0,0],dtype=complex)
    Rn=R/np.sqrt(R@R)
    x,y,z=Rn
    if abs(z-1)<1e-2: return np.array([1,0])
    if abs(z+1)<1e-2: return np.array([0,1])
    theta=acos(z)
    phi=acos(x/np.sin(theta))
    if y>0: phi*=-1
    #print(theta/np.pi*180,phi/np.pi*180)
    so3=np.array([
        [ np.cos(phi)*np.cos(theta),-np.sin(phi),np.cos(phi)*np.sin(theta)],
        [ np.sin(phi)*np.cos(theta), np.cos(phi),np.sin(phi)*np.sin(theta)],
        [            -np.sin(theta),           0,            np.cos(theta)]
    ])
    #print(so3)
    su2=SO3_SU2(so3)
    return su2@np.array([1,0])

def pseudoRotation(rotate,lat,axial):
    '''
    Rotate an axial vector, which should be represented by cartesian coor.
    '''
    axial_len=np.sqrt(axial@axial)
    if axial_len<1e-6: return np.zeros(3,dtype=float)
    #print(axial)
    basis=np.identity(3)
    e_cross_list=[np.cross(basis[i],axial) for i in range(3)]
    e_cross_len=[np.sqrt(e_cross@e_cross) for e_cross in e_cross_list]
    sort_index=np.argsort(e_cross_len)
    n1=basis[sort_index[-1]]
    n2=np.cross(n1,axial)
    n3=np.cross(n2,axial)
    
    reconstructed_axial=np.cross(n3,n2)
    reconstructed_axial_len=np.sqrt(reconstructed_axial@reconstructed_axial)
    assert reconstructed_axial_len>1e-6, "Failed to convert psuedo vector into cross form"
    #reconstructed_axial*=axial_len/reconstructed_axial_len
    #print(reconstructed_axial)
    #print('before rotat:',n2,n3)
    rotate_xyz=(lat.T)@rotate@(np.linalg.inv(lat.T))
    n2_trans=rotate_xyz@n2
    n3_trans=rotate_xyz@n3
    #print('after rotat:',n2_trans,n3_trans)
    return np.cross(n3_trans,n2_trans)*axial_len/reconstructed_axial_len

def semiCirle(p1:float,p2:float,pathPoints:np.ndarray):
    '''
    Generate semi-cirle path on upper half complex plane
    p1, p2: the two points on real-axis
    pathPoints: the positions on path, should be in range -1~1
    '''
    p_right=max([p1,p2])
    p_left=min([p1,p2])
    r=(p_right-p_left)/2
    o=(p_right+p_left)/2
    thetaPath=(pathPoints+1)/2*np.pi
    return r*np.exp(1j*thetaPath)+o, r*1j*np.exp(1j*thetaPath)

def obtainProjections(filePathOfPOSCAR:str,projections=[],ispin=1):
    '''
    Produce the structured atomic orb. ids based on the projections given by Wannier90-style strings
    NOTE: only standard atomic orb s, p, d, f are supported by now.
    
    Parameters
    ----------
    - filePathOfPOSCAR : str
      The position of POSCAR
    - projections : dictionary
      The atomic orbitals of each element. Key name corresponds to the element, item corresponds to the
      sequence of atomic orbitals, e.g.,
      projections={
          "Fe":("s","s","p","p","d","d"),
          "Cl":(0,0,1,1,2,3),
      }
      in which 0 is "s", 1 is "p", 2 is "d", 3 is "d" and 4 is "f". By now, only these 4 kinds are supported.
    - ispin : int, 1|2
      The number of spin tunnels. For ispin=1, generate orbital indexes of only one spin tunnels. For ispin=2, 
      generate double spin tunnels. Set ispin=2 if one needs whole orbital indexes for both spin tunnels, e.g.,
      spinor Hams., spin-polarized Hams.
    
    Returns
    -------
    - atom_orb_info : list
       contains the information of atomic orbitals for each atom.
       atom_orb_info[iatom] gives the information of iatom, keys include:

       'label' : str
       The element label of the atom

       'L' : list
       The L-number of each shell in the atom

       'orbital index' : list
       the global orbital indexes of the atomic orbitals
    '''
    # 1. obtain information of composite
    with open(filePathOfPOSCAR,'r') as fin:
        data=fin.read().split("\n")
    ele_name_list=data[5].split()
    ele_num_list=[int(x) for x in data[6].split()]
    #print(ele_name_list)
    #print(ele_num_list)
    label_to_L_dic={'s':0,'p':1,'d':2,'f':3}
    L_to_label_dic={0:'s',1:'p',2:'d',3:'f'}
    atom_orb_info=[] # atomic information of each atom
    iorb=0           # global orbital index
    for name, num in zip(ele_name_list,ele_num_list):
        for iatom in range(num):
            atom_orb={"label":name,"L":[],"orbital index":[]}
            if name in projections:
                for orb in projections[name]:
                    if orb in label_to_L_dic: 
                        L=label_to_L_dic[orb]
                    elif orb in L_to_label_dic: 
                        L=orb
                    else:
                        print(f"Error: get {L=}. By now, only support s, p, d, f orbitals, or L = 0|1|2|3")
                        exit()
                    atom_orb["L"].append(L)
                    shell=[]
                    for m in range(2*L+1):
                        shell.append(iorb)
                        iorb+=1
                    atom_orb["orbital index"].append(shell)
            atom_orb_info.append(atom_orb)
    
    if ispin==2:
        for atom in atom_orb_info:
            for ishell, L in enumerate(atom["L"]):
                shell=[]
                for m in range(2*L+1):
                    shell.append(iorb)
                    iorb+=1
                atom["orbital index"][ishell]+=shell
    return atom_orb_info

def obtainProjectionsPlain(filePathOfPOSCAR:str,projections=[],ispin=1):
    '''
    A simple solution to produce the orbital index for each atom based on the projections
    NOTE: only standard atomic orb s, p, d, f are supported by now. And only one spin tunnel is generated
    
    Parameters
    ----------
    filePathOfPOSCAR : str
      The position of POSCAR
    projections : dictionary
      The atomic orbitals of each element. Key name corresponds to the element, item corresponds to the
      sequence of atomic orbitals, e.g.,
      projections={
          "Fe":("s","s","p","p","d","d"),
          "Cl":(0,0,1,1,2,3),
      }
      in which 0 is "s", 1 is "p", 2 is "d", 3 is "d" and 4 is "f". By now, only these 4 kinds are supported.
    
    Returns
    -------
    atom_orb_index_list : list
       atom_orb_index_list[iatom] gives the global orbital indexes relevant the the iatom
    '''
    atom_orb_info=obtainProjections(filePathOfPOSCAR,projections,ispin=ispin)
    atom_orb_index_list=[]
    for atom in atom_orb_info:
        atomOrb=[]
        for shell in atom['orbital index']: atomOrb += shell
        atom_orb_index_list.append(atomOrb)
    return atom_orb_index_list

def projectionOperator(projection,atom_orb_info,norbs):
    '''
    Build projection matrix based on the selected atom shells

    Parameters
    ----------
    projection : list
      contains all the atom shells, with form of (atom_id, (shell_id0, shell_id1,...)).
    atom_orb_info : list
      atom_orb_index_list[iatom] gives the global orbital indexes relevant the the iatom
    norbs : int
      total num. of orb-basis

    Returns
    -------
    proj_mat : np.ndarray
       diagonal matrix with element = 1 or 0. The projected orbitals have 1, others are 0.
    '''
    proj_orb_index_list=[]
    for atom_id, shell_ids in projection:
        for shell_id in shell_ids:
            proj_orb_index_list += atom_orb_info[atom_id]['orbital index'][shell_id]
    proj=np.zeros(norbs)
    proj[proj_orb_index_list]=1
    return np.diag(proj)
    
def PauliBlock(p_full:np.ndarray):
    shape=p_full.shape
    assert len(shape)==2, 'Only 2-rank matrix can be decomposed into Pauli Blocks'
    nrow, ncol = shape
    assert nrow//2*2==nrow and ncol//2*2==ncol, 'Only matrix with even order can be decomposed now %dx%d'%(nrow,ncol)
    uu=p_full[:nrow//2,:ncol//2]
    ud=p_full[:nrow//2,ncol//2:]
    du=p_full[nrow//2:,:ncol//2]
    dd=p_full[nrow//2:,ncol//2:]

    p0=(uu+dd)/2
    pz=(uu-dd)/2
    px=(ud+du)/2
    py=(du-ud)/2j
    return p0,px,py,pz

def PauliBlockSigmaNorm(p_full:np.ndarray):
    """
    M= MI * I + \vec{P} dot \vec{sigma}
    = MI*I + p * \vec{e} dot \vec{sigma}
    where p is the norm of P.
    """
    p0, px, py, pz = PauliBlock(p_full)
    ex, ey, ez = np.trace(px), np.trace(py), np.trace(pz)
    evec = np.array((ex, ey, ez))
    evec = evec / np.linalg.norm(evec)
    return px * evec[0] + py * evec[1] + pz * evec[2]

def print_matrix(mat,rowIndex,colIndex,thr=1e-6):
    '''
    Print a matrix in a more visualizable mode
    
    Parameters:
    -----------
    mat : array-like
      The matrix to be printed
    rowIndex : list|tuple|array
      The indexes of printed rows
    colIndex : list|tuple|array
      The indexes of printed cols
    thr: float
      The threthold to hide a number
    '''
    totalElements=np.prod(mat.shape,dtype=int)
    nstacks=totalElements//np.prod(mat.shape[-2:],dtype=int)
    print_mat=mat.reshape((nstacks,)+mat.shape[-2:])
    for istack in range(nstacks):
        if nstacks>1: print(f"stack {istack}")
        for irow in rowIndex:
            print("|",end='')
            for icol in colIndex: 
                for ele in (print_mat[istack,irow,icol].real, print_mat[istack,irow,icol].imag):
                    if abs(ele)>thr:
                        print(f"{ele:8.1E}",end='')
                    else:
                        print("        ",end='')
                print('I ',end='')
            print("|")

def iterate3dpm(size):
    return (
        np.array([i, j, k]) for i in range(-size[0], size[0] + 1) for j in range(-size[1], size[1] + 1)
        for k in range(-size[2], size[2] + 1))

def fourier_q_to_R(AA_q, mp_grid, kpt_mp_grid, iRvec, ndegen, ):
    mp_grid = tuple(mp_grid)
    shapeA = AA_q.shape[1:]  # remember the shapes after q
    AA_q_mp = np.zeros(tuple(mp_grid) + shapeA, dtype=complex)
    for i, k in enumerate(kpt_mp_grid):
        AA_q_mp[k] = AA_q[i]
    AA_q_mp = np.fft.fftn(AA_q_mp, axes=(0, 1, 2))
    AA_R = np.array([AA_q_mp[tuple(iR % mp_grid)] / nd for iR, nd in zip(iRvec, ndegen)]) / np.prod(mp_grid)
    return AA_R

def reindexOrbitalsQE2VASP(tensor:np.ndarray,axis:int):
    '''
    In convention (VASP<6, OPENMX), the orbitals are indexed like: s_up, p_up, d_up, s_dn, p_dn, d_dn 
    In VASP>=6 and QE, the orbitals are indexed like: s_up, s_dn, p_up, p_dn, d_up, d_dn
    This function is to change the QE convention into the OPENMX
    '''
    norbs=tensor.shape[axis]
    assert (norbs//2)*2 == norbs, f"the dimension should be even! {norbs=}"
    reindex=np.concatenate((np.arange(norbs//2)*2,np.arange(norbs//2)*2+1))
    return np.take(tensor,reindex,axis=axis)

def getWholeSymmetryOperator(posinfo:dict,DNA=[],symprec=SYMPREC):
    interception = int(np.max(abs(np.array(DNA))))*4+3 if len(DNA)>0 else 10
    typeNumbers=[]
    labels=[]
    for inion, nion in enumerate(posinfo['nions']):
        typeNumbers.extend([(inion*interception)]*nion)
        labels.extend([posinfo['label list'][inion]]*nion)
    if len(DNA)>0:
        for iorb in range(len(DNA)):
            typeNumbers[iorb]+=int(DNA[iorb]*2)
    
    sym=spg.get_symmetry_dataset((posinfo['lattice'],posinfo['pos ions (frac.)'],typeNumbers),symprec=symprec)
    return sym

def fermiDiracOccupations(energies, fermiList, kBT):
    """Return stable Fermi-Dirac occupations over one or more Fermi levels."""
    energies = np.asarray(energies, dtype=float)
    fermiList = np.atleast_1d(np.asarray(fermiList, dtype=float))
    chemical_potential = fermiList.reshape(
        (len(fermiList),) + (1,) * energies.ndim
    )
    if kBT < 0.0:
        raise ValueError("kBT must be non-negative.")
    if kBT == 0.0:
        return np.asarray(energies[None, ...] < chemical_potential, dtype=float)

    scaled_energy = (energies[None, ...] - chemical_potential) / kBT
    occupations = np.empty_like(scaled_energy, dtype=float)
    positive = scaled_energy >= 0.0
    exp_negative = np.exp(-scaled_energy[positive])
    occupations[positive] = exp_negative / (1.0 + exp_negative)
    exp_positive = np.exp(scaled_energy[~positive])
    occupations[~positive] = 1.0 / (1.0 + exp_positive)
    return occupations

def solve_chemical_potential(energies, target_electrons_per_cell, number_kpoints, kBT, tol=1.0e-12):
    """Find the chemical potential giving a fixed Fermi-Dirac electron count."""
    energies = np.asarray(energies, dtype=float)
    number_bands = energies.size / number_kpoints
    if not 0.0 < target_electrons_per_cell < number_bands:
        raise ValueError("target_electrons_per_cell must lie strictly between zero and the number of bands.")
    if kBT <= 0.0:
        raise ValueError("A positive kBT is required for Fermi-Dirac filling.")

    margin = max(40.0 * kBT, 1.0)
    lower = float(np.min(energies) - margin)
    upper = float(np.max(energies) + margin)
    for _ in range(200): # Bisector searching for the chemical potential that gives correct filling
        chemical_potential = 0.5 * (lower + upper)
        occupations = fermiDiracOccupations(energies, [chemical_potential], kBT)[0]
        electron_count = np.sum(occupations) / number_kpoints
        if abs(electron_count - target_electrons_per_cell) < tol:
            return chemical_potential
        if electron_count < target_electrons_per_cell:
            lower = chemical_potential
        else:
            upper = chemical_potential
    raise RuntimeError("The fixed-number chemical-potential search did not converge.")