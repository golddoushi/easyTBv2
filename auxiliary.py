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
import paramReader
from math import acos
global time0
time0=0

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
            sortList=quicksort(diffList)
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

def showElapsedTime(offset=False,extraInfo=''):
    global time0
    if offset:
        time0=time.time()
        return
    print('%s in %.3fms'%(extraInfo,(time.time()-time0)*1000))
    return
        

def renormVec(vec:np.ndarray,cvg=paramReader.SYMPREC) -> np.ndarray:
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

def integeralizeVec(vec:np.ndarray, cvg=paramReader.SYMPREC) -> np.ndarray:
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

def obtainProjections(filePathOfPOSCAR:str,projections=[],ispin=1,soc=False):
    '''
    Produce the structured atomic orb. ids based on the projections given by Wannier90-style strings
    Useful when i need to construct the atomicOrbList for many interfaces such as velocity calcs.
    This is better than read atomic orbs. from output files of Wannier90, since we may not have these files.
    NOTE: only standard atomic orb s, p, d, f are supported by now.
    '''
    # 1. obtain information of composite
    with open(filePathOfPOSCAR,'r') as fin:
        data=fin.read().split("\n")
    ele_name_list=data[5].split()
    ele_num_list=[int(x) for x in data[6].split()]
    #print(ele_name_list)
    #print(ele_num_list)
    atom_index_dic={}  # index of each atom
    cnt=0
    for name, num in zip(ele_name_list,ele_num_list):
        sub_list=[]
        for i in range(num):
            sub_list.append(cnt)
            cnt+=1
        atom_index_dic[name]={"atom id":sub_list,"s":False,"p":False,"d":False,"f":False}
    #print(atom_index_dic)
    #exit()
    # 2. analyz the strings in projections, add them into dictionary
    label_to_L_dic={
        's':0,
        'p':1,
        'd':2,
        'f':3}
    for projection in projections:
        ele, orbs = [x.strip() for x in projection.split(':')]
        for orb in [x.strip() for x in orbs.split(";")]:
            if orb in label_to_L_dic.keys():
                atom_index_dic[ele][orb]=True
            else:
                print("Error, not supported orb. type: %s, only s, p, d, f are supported by now"%orb)
                exit()    
    #print(atom_index_dic)
    # 3. now let's construct the atomicOrbList
    checking_label_order=['s','p','d','f']
    checking_label_inversed_order=['f','d','p','s']
    all_atoms_orb_list=[]
    orb_id=0
    for ele in ele_name_list:
        for iatom in atom_index_dic[ele]['atom id']:
            for ilabel, label in enumerate(checking_label_inversed_order):
               if atom_index_dic[ele][label]:
                    atom_orb_list=[[] for i in range(4-ilabel)]
                    break
            else:
                print("Error, neither s, p, d, or f orb is founded for %s"%ele)
                exit()
            for ilabel, L_orb_list in enumerate(atom_orb_list):
                if not atom_index_dic[ele][checking_label_order[ilabel]]:continue
                for l in range(2*label_to_L_dic[checking_label_order[ilabel]]+1):
                    L_orb_list.append(orb_id)
                    orb_id+=1
            all_atoms_orb_list.append(atom_orb_list)
    
    if ispin==2 or soc:
        for ele in ele_name_list:
            for iatom in atom_index_dic[ele]['atom id']:
                for ilabel, L_orb_list in enumerate(all_atoms_orb_list[iatom]):
                    if not atom_index_dic[ele][checking_label_order[ilabel]]:continue
                    for l in range(2*label_to_L_dic[checking_label_order[ilabel]]+1):
                        L_orb_list.append(orb_id)
                        orb_id+=1
    
    return all_atoms_orb_list