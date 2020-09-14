'''
Created on 2019 6 29 

@author: Andrew
'''
import re
import paramReader as param
import numpy as np
import auxiliary as aux
import time
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.interpolate as itp
import spglib as spg
import subprocess

global title   # title of POSCAR file
global scale   # scale constant in POSCAR
global lat     # lattice matrix
lat=[]
global nkpts   # number of kpoints
global ele_str # name of each element
global nions   # num. of each element
global tot_ions# total num. of ions in structure
global rwigs   # RWIGS of each elements
global type
type='D'
global pos_ions
global bondList
global orbList
global ncount

class Orb:
    def __init__(self,id,typeID,typeLabel,pos=[0,0,0],spin=0,magmom=0,magmom_on=[0,0,0],D=[0,0,0],wyckoffs='no'):
        self.id=id
        self.type=typeID
        self.pos=pos
        self.spin=spin
        self.magmom=magmom
        self.magmom_on=magmom_on
        self.D=D
        self.wyckoffs=wyckoffs

        self.label=typeLabel
        self.rwigs=1

        self.index_tmp=0 # one interface to store some information for external module

        self.wannIDList=[]  # the IDs of Wannier orbitals attached to this ions are listed here
                            # if spin-polarized model or SOC model was assumed, former and later half list represent spin-up and spin-dn Wannier orbitals. 
    
    def defineOrbTypeAndTrans(self,orbTypeList,rotations,translations):
        # construct equal orbitals corresponding to standard transformations
        def isZeroVector(vec):
            '''check if the vec is zero vec or just over 1.'''
            for x in vec:
                if abs(x%1)>param.SYMPREC and abs(1-x%1)>param.SYMPREC and abs(2-x%1)>param.SYMPREC:
                    return False
            return True
        
        self.sameTypeOrbList=[]
        self.sameTypeOrbOverLatList=[]
        self.sameTypeOrbTransList=[]
        for rotate, trans in zip(rotations, translations):
            self.sameTypeOrbTransList.append([rotate,trans])
            pos_ref=np.dot(rotate,self.pos)+trans
            for orb_ref in orbTypeList:
                if isZeroVector(pos_ref-orb_ref.pos):
                    self.sameTypeOrbList.append(orb_ref)
                    self.sameTypeOrbOverLatList.append(np.array([int(round(x,0)) for x in pos_ref-orb_ref.pos]))
        assert len(self.sameTypeOrbList)==len(self.sameTypeOrbTransList), 'the sym. op do not well correspond to sym. pos.'

    def haveSameTypeWithMe(self,orb_ref):
        for orb in self.sameTypeOrbList:
            if orb.id==orb_ref.id:
                return True
        return False
    
    def getThePositionInMyKindList(self,orb_ref):
        return [iorb for iorb, orb in enumerate(self.sameTypeOrbList) if orb.id==orb_ref.id]

def readPOS(path='./POSCAR',ini_mag=[],ini_S=[],wann_ID=[],rotations=[],translations=[]):
    global title,scale,lat,ele_str,nions,tot_ions,type,pos_ions,orbList,bondList
    global ncount
    lat=[]
    ele_str=[]
    nions=[]
    pos_ions=[]
    orbList=[]
    bondList=[]
    f=open(path,'r')
    data=f.read()
    f.close()
    data=[line for line in data.split('\n')]
    title=data[0]
    scale=float(data[1])
    
    for line in range(2,5):
        [x,y,z]=[float(l) for l in re.findall('[0-9.-]+',data[line])]
        lat.append(np.array([x,y,z]))
    lat=np.array(lat)
    ele_str=[l for l in re.findall('[A-Za-z]+',data[5])]
    nions=[int(l) for l in re.findall('[0-9]+',data[6])]
    tot_ions=np.sum(nions)
    type=re.findall('[A-Za-z]+',data[7])
    for line in range(8,8+tot_ions):
        [x,y,z]=[float(l) for l in re.findall('[0-9.-]+',data[line])]
        pos_ions.append(np.array([x,y,z]))
    ncount=0
    __toHome()
    if len(ini_mag)>0:
        getAllOrbs(ini_mag=ini_mag,ini_S=ini_S,wann_ID=wann_ID,rotations=rotations,translations=translations)

def getTypeNumberList(DNA=[],getLabels=False): # get numbers consist of type of each ions
    global nions,tot_ions,ele_str
    interception = int(np.max(abs(np.array(DNA))))*4+3 if len(DNA)>0 else 10
    numbers=[]
    labels=[]
    for inion, nion in enumerate(nions):
        numbers.extend([(inion*interception)]*nion)
        labels.extend([ele_str[inion]]*nion)
    if len(DNA)>0:
        for iorb in range(len(DNA)):
            numbers[iorb]+=int(DNA[iorb]*2)
    if getLabels:
        return numbers, labels
    return numbers

def getWholeSymmetryOperator(DNA=[],getSymObj=False):
    global pos_ions, lat
    typeNumbers=getTypeNumberList(DNA=DNA)
    sym=spg.get_symmetry_dataset((lat,pos_ions,typeNumbers),symprec=param.SYMPREC)
    rotations=sym['rotations']
    translations=sym['translations']
    translations_std=[np.array([0. if (abs(x)<param.SYMPREC or abs(1-x)<param.SYMPREC) else x for x in trans]) for trans in translations]
    if getSymObj:
        return sym
    return rotations, translations_std

def getAllOrbs(ini_mag,ini_S,rotations=[],translations=[],wann_ID=[]):
    global pos_ions, ele_str, rwigs, orbList
    typeNumber, labels=getTypeNumberList(DNA=ini_mag,getLabels=True)
    #try:
    assert (len(pos_ions)==len(ini_mag) and len(pos_ions)==len(ini_S)), 'Error, dimension of ini_mag is not equal to num. of ions'
    #except AssertionError as e:
    #    print(e,"Error, dimension of ini_mag is not equal to num. of ions")
    #    exit()
    sym=getWholeSymmetryOperator(DNA=ini_mag,getSymObj=True)
    orbList=[Orb(ipos,typeNumber[ipos],labels[ipos],pos_ions[ipos],ini_S[ipos],ini_mag[ipos],wyckoffs=sym['wyckoffs'][ipos]) for ipos in range(len(pos_ions))]
    # add ini_orb informations
    if len(wann_ID)>0:
        for orb, WannierIDList in zip(orbList,wann_ID):
            orb.wannIDList=WannierIDList
    # classification
    wykoffsList=[]
    for orb in orbList:
        haveFoundFriends=False
        for orbType in wykoffsList:
            for orb_classified in orbType:
                if orb_classified.wyckoffs==orb.wyckoffs and orb_classified.label==orb.label:
                    haveFoundFriends=True
                    orbType.append(orb)
                    break
            if haveFoundFriends: break
        if not haveFoundFriends:
            wykoffsList.append([orb])
    rotations, translations=getWholeSymmetryOperator(DNA=ini_mag) if len(rotations)==0 else rotations, translations
    for orbType in wykoffsList:
        for orb in orbType:
            orb.defineOrbTypeAndTrans(orbType,rotations,translations)
    print('orbitals are successfully classified according to symmetry')
    if len(wann_ID)==0:
        for iorbType, orbType in enumerate(wykoffsList):
            print('type%d: %s-%s num.: %d'%(iorbType,orbType[0].label,orbType[0].wyckoffs,len(orbType)))
        return
    
    for iorbType, orbType in enumerate(wykoffsList):
        for orb in orbType:
            print('type%d: %s-%s with Wannier'%(iorbType,orb.label,orb.wyckoffs)+(' orb%d'*len(orb.wannIDList))%(*orb.wannIDList,))
    
def findBestRWIGS(ini_rwigs):
    global pos_ions, ele_str, rwigs, orbList
    def findEleTypeID(orb):
        for iele, ele in enumerate(ele_str):
            if orb.label==ele:
                return iele
        print('Error: cannot find the element name to fit the orbital label <<< posReader::findBestRWIGS::findEleTypeID')
        exit()
        return -1
    if len(ini_rwigs)==0:
        rwigs=[]
        return orbList
    # try to find best RWIGS
    # initialize N x N matrix to store distance between elements
    distanceBetweenElements=[[100 for i in ele_str] for j in ele_str]
    bondList, bondLen=findBondList(source_list=[i for i in range(len(pos_ions))],maxRange=1,tolerance=param.SYMPREC,getBondLen=True)
    for inNNBond, nNNBond in enumerate(bondList):
        distance=bondLen[inNNBond]
        for bond in nNNBond:
            sourceType=findEleTypeID(orbList[bond[0][0]])
            targetType=findEleTypeID(orbList[bond[0][1]])
            distanceBetweenElements[sourceType][targetType] = distance if distanceBetweenElements[sourceType][targetType]>distance else distanceBetweenElements[sourceType][targetType]
            distanceBetweenElements[targetType][sourceType]=distanceBetweenElements[sourceType][targetType]
        # check distance matrix
        allDistanceFound=True
        for row in distanceBetweenElements:
            for ele in row:
                if ele==100:
                    allDistanceFound=False
                    break
        if allDistanceFound:
            break
    if not allDistanceFound:
        print("Error, cannot find proper distance between some elements")
        exit()
    finalRWIGSLIST=[]
    for irow, row in enumerate(distanceBetweenElements):
        ishortest=-1 
        shortestBond=100
        for idistance, distance in enumerate(row):
            if distance<shortestBond:
                shortestBond=distance
                ishortest=idistance
        #print(irow,idistance)
        print("%s (initial RWIGS:%.3f) nearest to %s (initial RWIGS:%.3f) with distance=%.3f"%(ele_str[ishortest],ini_rwigs[ishortest],ele_str[irow],ini_rwigs[irow],shortestBond))
        finalRWIGS=shortestBond*ini_rwigs[irow]/(ini_rwigs[irow]+ini_rwigs[ishortest])*(1+param.RWIGSOVERLAP)
        print("reset RWIGS of %s to %.3f (enlarged by %d percent overlapping)"%(ele_str[irow],finalRWIGS,param.RWIGSOVERLAP*100))
        finalRWIGSLIST.append(finalRWIGS)
    rwigs=finalRWIGSLIST
    # update orbitals
    for orb in orbList:
        orb.rwigs=finalRWIGSLIST[findEleTypeID(orb)]
    return orbList

def getUniqueName(debug=False):
    global lat,pos_ions,ele_str, nions
    numbers=getTypeNumberList()
    cell=(lat,pos_ions,numbers)
    sym=spg.get_symmetry_dataset(cell,symprec=param.SYMPREC)
    if debug:
        print(lat)
        print(pos_ions)
        print(numbers)
        print(sym)
        print(str(sym['number']))
        print(getFormula(ele_str,nions))
        exit()
        #name='spg'+str(sym['number'])+getFormula(ele_str,nions)
        #print name
    return 'spg'+str(sym['number'])+getFormula(ele_str,nions)

def getFormula(ele_str_list,nion_list):
    formula=''
    for (ele,nion) in zip(ele_str_list,nion_list):
        formula=formula+ele+str(nion) if nion>1 else formula+ele
    return formula

def getIrreducibleKmesh(kmesh,DNA=[]):
    global lat, pos_ions, ele_str, nions, tot_ions

    def isZeroVector(vec):
        '''check if the vec is zero vec or just over 1.'''
        for x in vec:
            if abs(x)>param.SYMPREC and abs(1-x)>param.SYMPREC and abs(2-x)>param.SYMPREC:
                return False
        return True

    def checkIfNew(pos,posList,getIndex=False):
        '''Check if the pos is already in the posList'''
        for ipos,exist_pos in enumerate(posList):
            if(isZeroVector(pos-exist_pos)):
                if getIndex:
                    return ipos
                return False
        if getIndex:
            return -1
        return True

    def isNewRotations(rotation0,checkList):
        '''check if the rotation is a totally new one'''
        for rotation in checkList:
            if(rotation0==rotation).all():
                return False
        return True

    def getEqualPos(pos0List,transform):
        '''give me any position and the total symmetrical transformations
           return you all the equal positions
        '''
        totalPos=[np.dot(trans[0],pos0)%1 for trans in transform for pos0 in pos0List]
        beautifulPos=[]
        #debug=False
        for pos in totalPos:
            tmp=np.array([0. if (x<param.SYMPREC or 1-x<param.SYMPREC or 2-x<param.SYMPREC) else x for x in pos])
            # gamma centered
            # tmp=np.array([x-1 if x > 0.5 else x for x in tmp0])
            #if isZeroVector(tmp-np.array([0.7,0.1,0.0])):
            #    debug=True
            if checkIfNew(tmp,beautifulPos):
                beautifulPos.append(tmp)
        #if debug:
        #    print 'here'
        #    print np.array(beautifulPos)
        return np.array(beautifulPos)

    def generateAllSYMTrans(all_trans):
        '''
        The bias only support phase enhance for wavefunc
        however, the k-point is still redundant
        '''
        transList=[]
        for trans in all_trans:
            if isNewRotations(trans,transList):
                transList.append(trans)
                #print trans
        
        return transList

    def getOptimalKrep(equal_kpts):
        #print 'equal pool'
        #for k in equal_kpts:
        #    print k
        def optAccordingToLength(input_pool):
            coor=[np.dot(k,k) for k in input_pool]
            sort=aux.quicksort(coor)
            min_coor=coor[0]
            pool=[]
            for i in range(len(coor)):
                if coor[i]-min_coor<param.SYMPREC:
                    pool.append(input_pool[sort[i]])
                else:
                    break
            return np.array(pool)
        def optAccordingToCoorIndex(input_pool,coorIndex):
            coor=list(input_pool[:,coorIndex])
            sort=aux.quicksort(coor)
            min_coor=coor[0]
            pool=[]
            for i in range(len(coor)):
                if coor[i]-min_coor<param.SYMPREC:
                    pool.append(input_pool[sort[i]])
                else:
                    break
            return np.array(pool)
        # sort for length
        lpool=optAccordingToLength(equal_kpts)
        # sort for z coor
        zpool=optAccordingToCoorIndex(lpool,2)
        # sort for y coor
        ypool=optAccordingToCoorIndex(zpool,1)
        # sort for x coor
        xpool=optAccordingToCoorIndex(ypool,0)
        return xpool[0]

    # get rotation list in real space (fractional representation)
    numbers=getTypeNumberList(DNA=DNA)
    cell=(lat,pos_ions,numbers)
    sym=spg.get_symmetry_dataset(cell,symprec=1e-4) # since the propose is to fit the IBZKPT thus set to the default value of VASP
    #print sym
    rotation_frac_list=generateAllSYMTrans(sym['rotations'])
    # get rotation list in reciprocal space
    S_inv=np.dot(lat,lat.T)
    S=np.linalg.inv(S_inv)
    rotation_frac_reciprocal=[]
    for rotate in rotation_frac_list:
        rotate_reci=np.dot(np.dot(S_inv.T,rotate),S.T)
        for irow, row in enumerate(rotate_reci):
            for icol, col in enumerate(row):
                if abs(col)<param.SYMPREC:
                    rotate_reci[irow][icol]=0
        rotation_frac_reciprocal.append([rotate_reci,np.zeros(3)])
        # time-reversal-symmetry involved operation
        rotation_frac_reciprocal.append([-rotate_reci,np.zeros(3)])
    #for rotat in rotation_frac_reciprocal:
    #    print rotat[0]
    # generate irreducible kmesh
    equalIndex=[]
    irkmesh=[]
    weight=[]
    for kpt in kmesh:
        equal_kpts=getEqualPos([kpt],rotation_frac_reciprocal)
        # get optimal representation
        optimal_rep=getOptimalKrep(equal_kpts)
        # check redundant
        newKpt=True
        for iexist, exist in enumerate(irkmesh):
            if isZeroVector(exist-optimal_rep):#checkIfNew(exist,equal_kpts,getIndex=True)>0:
                weight[iexist]+=1
                newKpt=False
                equalIndex.append(iexist)
                break
        if newKpt:
            irkmesh.append(optimal_rep)
            weight.append(1)
            equalIndex.append(-1)
    #print('irreducible k-points and weights (TRS included):')
    #for x,y in zip(irkmesh,weight):
    #    print(x,y)
    #exit()
    return irkmesh, weight, equalIndex

def readPrototypeAndCreateAllInitialPOSCAR(pathOfPrototype,pathOfMainDir):
    global title,scale,lat,ele_str,nions,tot_ions,type, pos_ions
    
    fPrototype=open(pathOfPrototype,'r')
    Prototype=[line for line in fPrototype.read().split('\n') if line]
    fPrototype.close()

    tot_ions=int(Prototype[0])
    #read lattice
    lat_line = re.findall('[A-Za-z0-9.\-]+',Prototype[1])
    lat=[[float(lat_line[l]) for l in range(1,4)],
         [float(lat_line[l]) for l in range(4,7)],
         [float(lat_line[l]) for l in range(7,10)]]
    lat=np.array(lat)
    lat_inv=np.linalg.inv(lat)
    #read ele_str and positions
    scale=1.0
    type='D'
    ele_str_first=[]
    #nions=[]
    pos_ions=[]
    #ele_old=''
    numbers=[]
    typeid=0

    def findParty(you,party):
        for iele, ele in enumerate(party):
            if ele==you:
                return iele
        return -1
        
    for iline in range(2,2+tot_ions):
        datapack=re.findall('[A-Za-z0-9.\-]+',Prototype[iline])
        ele=datapack[0]
        pos=np.dot(np.array([float(datapack[l]) for l in range(1,4)]),lat_inv)
        ipos=findParty(ele,ele_str_first)
        if ipos<0:
            pos_ions.append(pos)
            ele_str_first.append(ele)
            numbers.append(typeid)
            typeid+=1
        else:
            ele_str_first.insert(ipos,ele)
            pos_ions.insert(ipos,pos)
            numbers.insert(ipos,numbers[ipos])
    nions=[]
    old_type=-1
    ele_str_tmp=[]
    for itype, typeid in enumerate(numbers):
        if typeid!=old_type:
            nions.append(1)
            ele_str_tmp.append(ele_str_first[itype])
            old_type=typeid
        else:
            nions[len(nions)-1]+=1
    ele_str_first=ele_str_tmp
    # get space group
    sym=spg.get_symmetry_dataset((lat,np.array(pos_ions),numbers),symprec=param.SYMPREC)
    # create all ele_str
    ele_str_all_in_list=[ele_str_first]
    basicNLines=2+tot_ions
    if len(Prototype)>basicNLines:
        for iline in range(basicNLines,len(Prototype)):
            ele_str_all_in_list.append(re.findall('[A-Za-z]+',Prototype[iline]))

    subdir_list=[]
    for comb in ele_str_all_in_list:
        title=getFormula(comb,nions)
        subdir=pathOfMainDir+title+'/'
        aux.createDir(subdir)
        subdir_list.append(subdir)
        #decide tile and ele_str
        title=title+' '+sym['international']
        ele_str=comb
        createSC(filename=subdir+'POSCAR',size=[1,1,1])
    return subdir_list

def __toStandard():
    global lat, pos_ions, nions
    numbers=getTypeNumberList()
    cell=(lat, np.array(pos_ions), numbers)
    sym=spg.get_symmetry_dataset(cell,symprec=param.SYMPREC)
    print(sym['transformation_matrix'])
    print(sym['std_lattice'])
    print(sym['std_positions'])
    lat=sym['std_lattice']
    pos_ions=sym['std_positions']
    cell=(lat, np.array(pos_ions), numbers)
    sym=spg.get_symmetry_dataset(cell,symprec=param.SYMPREC)
    print(sym['transformation_matrix'])
    exit()
    
def getPG(source_list):
    global lat, pos_ions
    # get space group of mag. ions lat.
    mag_pos=np.array([pos_ions[isource] for isource in source_list])
    numbers=[1]*len(source_list)
    cell=(lat,mag_pos,numbers)
    return spg.get_symmetry_dataset(cell,symprec=param.SYMPREC)
    
def translatePOSCAR(path,vector):
    global title, scale, lat, ele_str, nions, type, pos_ions
    fpos=open(path+'POSCAR','w')
    fpos.write(title+' translated\n')
    fpos.write(str(scale)+'\n')
    for vec in lat:
        fpos.write(' {:.9f} {:.9f} {:.9f}\n'.format(vec[0],vec[1],vec[2]))
    fpos.write(' ')
    for ele in ele_str:
        fpos.write(ele+' ')
    fpos.write('\n')
    for nion in nions:
        fpos.write(' {:3d}'.format(nion))
    fpos.write('\n')
    fpos.write('D\n')
    for pos in pos_ions:
        pos_trans=(pos+vector)%1
        pos_bound=[]
        for x in pos_trans:
            if(abs(x)<param.SYMPREC or abs(1-x)<param.SYMPREC):
                pos_bound.append(0.)
            else:
                pos_bound.append(x)
        fpos.write(' {:.9f} {:.9f} {:.9f}\n'.format(pos_bound[0],pos_bound[1],pos_bound[2]))
    fpos.close()

def __standarizePos(sc_pos,vec):
    stdPos=[]
    for pos in sc_pos:
        pos_mod=(pos+vec)%1
        for ix,x in enumerate(pos_mod):
            if(x<0.001 or 1-x < 0.001):
                pos_mod[ix]=0
        stdPos.append(pos_mod)
    return stdPos

def extractDetail():
    global title, tot_ions
    datapack=re.findall('[a-zA-Z0-9\-\.\*]+',title)
    findDetail=False
    iele=0
    detailInformationDic={'haveDetailInfo':False}
    while(True):
        if iele>=len(datapack)-1:
            break
        ele=datapack[iele]
        if ele=='detail':
            findDetail=True
            detailInformationDic['haveDetailInfo']=True
            iele+=1
            continue
        # set magmom_list
        if findDetail and ele=='magmom':
            magmom_list=[float(datapack[imag]) for imag in range(iele+1,iele+1+tot_ions)]
            detailInformationDic['magmom_list']=magmom_list
            iele=iele+tot_ions+1
            continue
        # set total mag
        if findDetail and ele=='totmag':
            totmag=float(datapack[iele+1])
            detailInformationDic['tot_mag']=totmag
            iele+=2
            continue
        # set S
        if findDetail and ele=='Spin':
            S=np.array([float(datapack[imag]) for imag in range(iele+1,iele+1+tot_ions)])
            detailInformationDic['S']=S
            iele=iele+tot_ions+1
            continue
        # set Xi
        if findDetail and ele=='Xi':
            Xi=np.array([float(datapack[imag]) for imag in range(iele+1,iele+1+tot_ions)])
            detailInformationDic['Xi']=Xi
            iele=iele+tot_ions+1
            continue
        if findDetail and ele=='gap':
            gap=float(datapack[iele+1])
            detailInformationDic['gap']=gap
            iele+=2
            continue
        if findDetail and ele=='AMIX':
            AMIX=float(datapack[iele+1])
            detailInformationDic['AMIX']=AMIX
            iele+=2
            continue
        iele+=1
    return detailInformationDic

def createSC(filename='./POSCAR',size=[2,2],source_list=[],detailDic=None,addInfo='',debug=False):
    global title, scale, lat, ele_str, nions, type, pos_ions
    def checkSwapOp(trans):
        for row in trans:
            for ele in row:
                if ele%1>param.SYMPREC and 1-ele%1>param.SYMPREC:
                    print('WARNING: the symmetrilizing process change type of lattice, give up.')
                    return False
        return True

    # generalize to 3D case
    if len(size)==2:
        size.append(1)
    elif len(size)==1:
        size.extend([1,1])
    # standarize the original cell first if we have mag. source list
    def standarizePos(sc_pos,vec):
        stdPos=[]
        for pos in sc_pos:
            pos_mod=(pos+vec)%1
            for ix,x in enumerate(pos_mod):
                if(x<param.SYMPREC or 1-x <param.SYMPREC):
                    pos_mod[ix]=0
            stdPos.append(pos_mod)
        return stdPos

    # since the expanding will induce errors, we need standarize the position here
    numbers=getTypeNumberList()
    cell=(lat,pos_ions,numbers)
    primitive_sym=spg.get_symmetry_dataset(cell,symprec=param.SYMPREC)
    std_pos=primitive_sym['std_positions']
    std_lat=primitive_sym['std_lattice']
    eig, S_std_lat=np.linalg.eigh(std_lat)
    S_std_lat=abs(S_std_lat)
    eig, S_ori_lat=np.linalg.eigh(lat)
    S_ori_lat=abs(S_ori_lat)
    trans_S=np.dot(S_std_lat,np.linalg.inv(S_ori_lat))
    #pos_ions=np.dot(std_pos,trans_S)
    
    #-----------------------------------------------------------------------------
    # ATTENTION: following codes is valid only if current pos is primitive cell
    #            therefore, clean and re-read POSCAR (u.c.) before createSC
    #-----------------------------------------------------------------------------
    if len(std_pos)!=len(pos_ions):
        print('Error: createSC while present pos is not u.c. but some kinds of s.c.')
        print('       please try to clean first and re-read POSCAR of u.c.')
        #exit()
    if np.sum(abs(trans_S-np.eye(3))) > param.SYMPREC:
        print('WARNING: the symmetric operation in createSC func. of readPOS module')
        print('         has changed the order of basises, rotation:')
        print(trans_S)
        if checkSwapOp(trans_S):
            pos_ions=np.dot(std_pos,trans_S)
    else:
        pos_ions=np.dot(std_pos,trans_S)


    if(len(source_list)>0):
        # get positions for sources
        mag_pos=[pos_ions[index] for index in source_list]
        # decide numbers for sources
        numbers=[]
        for source in source_list:
            accumulant=0
            for itype, type_num in enumerate(nions):
                accumulant+=type_num
                if(source<accumulant): 
                    numbers.append(itype)
                    break
        # get spg
        origin_shift=primitive_sym['origin_shift']
        pos_std=standarizePos(pos_ions,origin_shift)
    else:
        pos_std=pos_ions        
    #----------------------------------------------------------#    

    fpos=open(filename,'w')
    if detailDic!=None:
        #if detailDic==None:
        #    print("Error: detai information required to write into POSCAR but no detaiDic is provided")
        #if debug:
        #    print name#primitive_sym
            
        title_sc=getUniqueName()
        # generate detail info.
        title_sc+=' detail:'
        # magmom ref. and S
        title_sc+=' magmom:'
        for mag in detailDic['magmom_list']:
            title_sc+=' %.3f'%mag
        title_sc+=' Spin:'
        for spin in detailDic['S']:
            title_sc+=' %.1f'%spin
        title_sc+=' gap:%.3f'%detailDic['gap']
        if detailDic['AMIX']>0:
            title_sc+=' AMIX:%.3f'%detailDic['AMIX']
        title_sc=title_sc+' '+addInfo
    else:
        title_sc=getUniqueName()+' automatically generated supercell '+str(size[0])+'x'+str(size[1])+'x'+str(size[2])

    fpos.write(title_sc+'\n')
    fpos.write(str(scale)+'\n')
    sc_lat=np.array([lat[0]*size[0],lat[1]*size[1],lat[2]*size[2]])
    for vec in sc_lat:
        fpos.write(' {:.9f} {:.9f} {:.9f}\n'.format(vec[0],vec[1],vec[2]))
    fpos.write(' ')
    for ele in ele_str:
        fpos.write(ele+' ')
    fpos.write('\n')
    numbers=[]
    for inion, nion in enumerate(nions):
        fpos.write(' {:3d}'.format(nion*size[0]*size[1]*size[2]))
        numbers.extend([inion]*nion*size[0]*size[1]*size[2])
    fpos.write('\n')
    fpos.write('D\n')
    #dr=np.array([1./size[0],1./size[1],0])
    da1,da2,da3=1./size[0],1./size[1],1./size[2]
    basicPos=[np.array([pos[0]/size[0],pos[1]/size[1],pos[2]/size[2]]) for pos in pos_std]
    pos_sub_list=np.array([(pos+np.array([da1*i,da2*j,da3*k]))%1 for pos in basicPos for i in range(size[0]) for j in range(size[1]) for k in range(size[2])])
    for pos in pos_sub_list:
        fpos.write(' {:.9f} {:.9f} {:.9f}\n'.format(pos[0],pos[1],pos[2]))

    fpos.close()

def createSymmetryOperatorsForSC(rotations_basic, translations_basic, size=[]):
    if len(size)==0:
        print("Error, size of supercell is not defined")
        exit()
    elif len(size)==1:
        size.extend([1,1])
    elif len(size)==2:
        size.append(1)
    elif len(size)>3:
        print("Error, size list of supercell is beyond 3-dimension")
        exit()
    expandS=np.diag(size)
    expandS_inv=np.linalg.inv(expandS)
    
    S_invD=[np.dot(expandS_inv,np.array([i,j,k])) for i in range(size[0]) for j in range(size[1]) for k in range(size[2])]
    
    rotations=[np.dot(np.dot(expandS_inv,rotate),expandS) for rotate in rotations_basic for SD in S_invD]
    translations=[np.dot(expandS_inv,translate)+SD for translate in translations_basic for SD in S_invD]
    #print(rotations)
    #print(translations)
    #exit()
    return rotations, translations

def clean():
    global title,scale,lat,nkpts,ele_str,rwigs,nions,tot_ions,type,pos_ions,bondList,ncount
    global orbList
    lat=[]
    nkpts=-1
    ele_str=[]
    rwigs=[]
    nions=[]
    tot_ions=-1
    type='D'
    pos_ions=[]
    bondList=[]
    ncount=0
    orbList=[]

def writeLattice(fout,reduced=True):
    global lat
    # reduced real lat.
    a=np.sqrt(np.dot(lat[0],lat[0])) if reduced else 1.
    a0=np.array([lat[0][0]/a, lat[0][1]/a])
    a1=np.array([lat[1][0]/a, lat[1][1]/a])

    # reciprocal lat.
    b0=np.array([a1[1],-a1[0]])
    b1=np.array([-a0[1],a0[0]])

    S=abs(np.dot(a0,b0))
    b0*=2*np.pi/S
    b1*=2*np.pi/S

    fout.write('real space lattice (a={:.6f}):\n'.format(a))
    fout.write(' {:.9f}  {:.9f}\n'.format(a0[0], a0[1]))
    fout.write(' {:.9f}  {:.9f}\n'.format(a1[0], a1[1]))
    fout.write('\n')
    fout.write('reciprocal lattice:\n')
    fout.write(' {:.9f}  {:.9f}\n'.format(b0[0], b0[1]))
    fout.write(' {:.9f}  {:.9f}\n'.format(b1[0], b1[1]))
    fout.write('\n')

def drawCell(rpath, magIonList, invMagList):
    global lat, pos_ions, BondList
    plt.figure(figsize=(8,8))
    def dragLine(p0,p1,color='black',style='-',label=None,curvPercentage=0.0):
        if curvPercentage<0.001:
            plt.plot([p0[0],p1[0]],[p0[1],p1[1]],style,color=color,label=label)
        else:
            # find mid point
            dp=p1-p0
            lineLen=np.sqrt(np.dot(dp,dp))
            midp=(p0+p1)/2.
            # orthorgonal direction
            normal_dir=np.array([-dp[1],dp[0]])*curvPercentage
            # interpolation 
            x=np.linspace(p0[0],p1[0],param.BONDINPT) 
            y=np.linspace(p0[1],p1[1],param.BONDINPT)
            for i in range(param.BONDINPT):
                bias=i*(param.BONDINPT-1.-i)/((param.BONDINPT-1.)/2.)**2*normal_dir
                x[i]+=bias[0]
                y[i]+=bias[1]
            #biased_pt=midp+normal_dir
            #x=[p0[0],biased_pt[0],p1[0]]
            #y=[p0[1],biased_pt[1],p1[1]]
            plt.plot(x,y,style,color=color,label=label,linewidth=3)

    printedOrb=[]
    def checkIfPrinted(xy):
        for orb in printedOrb:
            diff=orb-xy
            if np.dot(diff,diff)<1e-5:
                return True
        return False


    # firstly draw cell borders and its neighbors
    lat2d=np.array(lat)[:2,:2]
    dragLine([0,0],lat2d[0])
    dragLine([0,0],lat2d[1])
    dragLine(lat2d[0]+lat2d[1],lat2d[0])
    dragLine(lat2d[0]+lat2d[1],lat2d[1])
    # waiting for the neighbors

    # then draw the orbitals
    pos2d=np.dot(np.array(pos_ions)[:,:2],lat2d)
    for pos in pos2d:
        printedOrb.append(pos)
        dot=plt.scatter(pos[0],pos[1],s=72,color='grey')
        dot.set_zorder(100)
    for imag in magIonList:
        dot=plt.scatter(pos2d[imag][0],pos2d[imag][1],s=288,color='blue')
        dot.set_zorder(100)
    for inv in invMagList:
        dot=plt.scatter(pos2d[inv][0],pos2d[inv][1],s=288,color='red')
        dot.set_zorder(100)

    # draw bonds
    nColor=len(param.color_pool)
    bondList=[BondList[i] for i in range(param.CALRANGE)]
    edgePosList=[]
    for ibond, bondset in enumerate(bondList):
        if ibond>nColor-1: 
            bond_color='black'
        else:
            bond_color=param.color_pool[ibond]

        for bond in bondset:
            iedge0,iedge1=bond[0][0],bond[0][1]
            offset=np.dot(np.array(bond[1])[:2],lat2d)
            pos0,pos1=pos2d[iedge0],pos2d[iedge1]+offset
            edgePosList.append(pos0)
            edgePosList.append(pos1)
            # check afm bond
            fmbond=1
            edge1color=(0,0,1,0.3)
            for inv in invMagList:
                if(inv==iedge0):
                   fmbond*=-1
                if(inv==iedge1):
                   fmbond*=-1
                   edge1color=(1,0,0,0.3)
                   
            if fmbond>0:
                dragLine(pos0,pos1,color=bond_color,style='-',curvPercentage=param.BONDBIAS)
                plt.annotate(str(iedge0),xy=tuple(pos0),size=24)
                plt.annotate(str(iedge1),xy=tuple(pos1),size=24)
                if not checkIfPrinted(pos1):
                    printedOrb.append(pos1)
                    plt.scatter(pos1[0],pos1[1],s=288,color=edge1color)
            else:
                # using intepolation
                dragLine(pos0,pos1,color=bond_color,style='--',curvPercentage=param.BONDBIAS)
                plt.annotate(str(iedge0),xy=tuple(pos0),size=24)
                plt.annotate(str(iedge1),xy=tuple(pos1),size=24)
                if not checkIfPrinted(pos1):
                    printedOrb.append(pos1)
                    plt.scatter(pos1[0],pos1[1],s=288,color=edge1color)

    
    edgePosList=np.array(edgePosList)
    xmin,xmax,ymin,ymax=np.min(edgePosList[:,0]),np.max(edgePosList[:,0]),np.min(edgePosList[:,1]),np.max(edgePosList[:,1])
    xlen,ylen=xmax-xmin,ymax-ymin
    ox,oy=xmin+xlen/2,ymin+ylen/2
    
    pic_len=np.max([xlen,ylen])*1.2
    #save fig
    plt.xlim(ox-pic_len/2,ox+pic_len/2)
    plt.ylim(oy-pic_len/2,oy+pic_len/2)
    plt.axis('off')
    plt.savefig(rpath+'CouplingBonds.png',dpi=300)
    plt.close()

def __toHome():
    global tot_ions, pos_ions, lat, ncount
    return
    source_list=[i for i in range(tot_ions)]
    Bond,BondLen=findBondList(source_list,maxRange=1,
                              intraCell=False,getBondLen=True)
    if(BondLen[len(BondLen)-1]>=lat[2][2]*0.8):
        print('')
        print('Warning: the system may be locate on cell-boundary')
        print('         now i try to adjust by a translation along c axis')
        #print BondLen
        for pos in pos_ions:
            pos[2]=(pos[2]+0.1)%1
        ncount+=1
        if(ncount>=10):
            print('Error: cannot find the suitable center in c direction')
            print('       maybe 2D system is too thick comparing to vacuum')
            print('       i.e. thickness is more than 80% depth of cell')
            print('try to adjust the structure or modify codes in posReader def toHome')
            exit()
        __toHome()

def plotKSampling(kmesh):
    global lat
    plt.figure(figsize=(6,6))
    def drawBorder(endPoint,color='red'):
        px,py=endPoint[0],endPoint[1]
        plt.plot([px+py,px-py],[py-px,py+px],c=color)
    lat_inv=np.linalg.inv(lat)*2*np.pi
    #kmesh=np.array([[k1,k2,k3] for k1 in kpts1 for k2 in kpts2 for k3 in kpts3])
    kmesh_cart=np.dot(lat_inv,kmesh.T).transpose()
    xmax,xmin,ymax,ymin=np.max(kmesh_cart[:,0]),np.min(kmesh_cart[:,0]),np.max(kmesh_cart[:,1]),np.min(kmesh_cart[:,1])
    length=np.max([xmax-xmin,ymax-ymin])*1.2
    ox,oy=(xmax+xmin)/2, (ymax+ymin)/2
    for ikpt, kpt in enumerate(kmesh_cart):
        kx, ky, _ = kpt
        plt.scatter(kx,ky,c='black')
        plt.annotate('%d'%ikpt,(kx,ky))
    # irreducible BZ
    
    g1=np.array([lat_inv[0][0]/2, lat_inv[1][0]/2])
    g2=np.array([lat_inv[0][1]/2, lat_inv[1][1]/2])

    plt.plot([0,g1[0]],[0,g1[1]],c='blue') # g1
    plt.plot([0,g2[0]],[0,g2[1]],c='blue') # g2
    plt.plot([g1[0]-g2[0],g1[0]+g2[0]],[g1[1]-g2[1],g1[1]+g2[1]],c='blue') # g1
    plt.plot([g2[0]-g1[0],g2[0]+g1[0]],[g2[1]-g1[1],g2[1]+g1[1]],c='blue') # g2
    plt.plot([-g1[0]+g2[0],-g1[0]-g2[0]],[-g1[1]+g2[1],-g1[1]-g2[1]],c='blue') # g1
    plt.plot([-g2[0]+g1[0],-g2[0]-g1[0]],[-g2[1]+g1[1],-g2[1]-g1[1]],c='blue') # g2

    drawBorder(g1)
    drawBorder(g2)
    drawBorder(-g1)
    drawBorder(-g2)
    drawBorder(g1+g2)
    drawBorder(g1-g2)
    drawBorder(g2-g1)
    drawBorder(-g1-g2)
        
    plt.xlim(ox-length/2,ox+length/2)
    plt.ylim(oy-length/2,oy+length/2)
    plt.show()

def createKPOINTS(rpath=None,gam=False,getKMesh=False,forceBoder=False,provideSize=None): # if getKMesh, no file will be created
    global lat,nkpts
    a1, a2, a3=lat[0], lat[1], lat[2]
    a1_len=np.sqrt(np.dot(a1,a1))
    a2_len=np.sqrt(np.dot(a2,a2))
    a3_len=np.sqrt(np.dot(a3,a3))
    nkpt1=np.max([int(param.KDEN/a1_len),1])
    nkpt2=np.max([int(param.KDEN/a2_len),1])
    nkpt3=np.max([int(param.KDEN/a3_len),1])
    if forceBoder:
        nkpt1=int((nkpt1+1)/2)*2
        nkpt2=int((nkpt2+1)/2)*2
        nkpt3=int((nkpt3+1)/2)*2
    nkpt3=1 if (param.PBC==2 or param.PBC==1 or param.PBC==0) else nkpt3
    nkpt2=1 if (param.PBC==1 or param.PBC==0) else nkpt2
    nkpt1=1 if (param.PBC==0) else nkpt1
    
    if gam:
        nkpt1=nkpt2=nkpt3=1
    nkpts=nkpt1*nkpt2*nkpt3
    if provideSize!=None:
        nkpt1, nkpt2, nkpt3 = provideSize
    # Monkhorst-Pack sampling
    kpts1=np.array([(2*r-nkpt1+1)/2./nkpt1 for r in range(nkpt1)])
    kpts2=np.array([(2*r-nkpt2+1)/2./nkpt2 for r in range(nkpt2)])
    kpts3=np.array([(2*r-nkpt3+1)/2./nkpt3 for r in range(nkpt3)])
    if((np.dot(a1,a2)/a1_len/a2_len-0.5)<1e-5):  # a naive checking for hexagonal frame
        # half shift
        kpts1+=(nkpt1+1)%2/2./nkpt1
        kpts2+=(nkpt2+1)%2/2./nkpt2
        kpts3+=(nkpt3+1)%2/2./nkpt3

    if getKMesh: # if getKMesh, no file will be created
        return [np.array([k1,k2,k3]) for k1 in kpts1 for k2 in kpts2 for k3 in kpts3]
    
    assert rpath!=None
    # write KPOINTS file
    with open(rpath+'KPOINTS','w') as f_kpt:
        f_kpt.write('Automatic generate via XXZ\n')
        f_kpt.write('0\n')
        if((np.dot(a1,a2)/a1_len/a2_len-0.5)<1e-5):  # a naive checking for hexagonal frame
            f_kpt.write('G\n')
        else:
            f_kpt.write('M\n')

        f_kpt.write(str(nkpt1)+' '+str(nkpt2)+' '+str(nkpt3)+'\n')
        f_kpt.write('0 0 0')
    return

def autoGenerateKpath2D(rpath=None,onlyGetKPTList=False):
    '''
    generate k-path automatically,
    only for 2D case, temporally
    Parameter(s):
        nikpt: represents the number of interval KPs between two high-symmetric KP
    '''
    def genKPath(highSymK,nikpt=20):
        #generate kpath between high-symmetry kpoints
        kpath=[]
        npath=len(highSymK)-1
        for ipath in range(npath):
            dk=(highSymK[ipath+1]-highSymK[ipath])/nikpt[ipath]
            for ikpt in range(nikpt[ipath]):
                kpath.append(ikpt*dk+highSymK[ipath])
        kpath.append(highSymK[npath])
        return kpath

    global lat
    reci_lattice=np.linalg.inv(lat.T)
    b1=reci_lattice[0][:2]/2
    b2=reci_lattice[1][:2]/2
    b1b2=b1+b2
    # calc. coordinates of edge
    # in cartesian coor.
    factor=np.array([b1,b2])
    res=np.array([np.dot(b1,b1),np.dot(b2,b2)])
    K_cart=np.dot(np.linalg.inv(factor),res)
    # in reci-fractional coor.
    K0=np.dot(np.array([K_cart[0],K_cart[1],0.]),np.linalg.inv(reci_lattice))

    # another possible
    factor=np.array([b1,b1b2])
    res=np.array([np.dot(b1,b1),np.dot(b1b2,b1b2)])
    K_cart=np.dot(np.linalg.inv(factor),res)
    K1=np.dot(np.array([K_cart[0],K_cart[1],0.]),np.linalg.inv(reci_lattice))

    # chose the short one
    K=K0[:2] if np.dot(K0,K0)<np.dot(K1,K1) else K1[:2]

    # find the coordinates of bond
    # four possible positions
    M_list=[np.array([0.5,0]),np.array([-0.5,0]),np.array([0.,0.5]),np.array([0.,-0.5])]
    sort_list=aux.quicksort([np.dot(M-K,M-K) for M in M_list])
    # chose the nearest one
    M=M_list[sort_list[0]]
    G=np.array([0.,0.])
        
    # generate kpath
    print('')
    print('auto-generate kpath for bandstructure of 2D system')
    print('gamma:',G)
    print('edge :',K)
    print('bond :',M)
    kpath=genKPath([G,K,M,G],[param.KINTERVAL,param.KINTERVAL,param.KINTERVAL])
    if onlyGetKPTList:
        return [np.array([kpt[0],kpt[1],0]) for kpt in kpath]
    assert rpath!=None
    f_kpt=open(rpath+'KPOINTS','w')
    f_kpt.write('Automatic generated kpath for bandstructure calc. of 2D system via XXZ\n')
    f_kpt.write('%5d\n'%len(kpath))
    f_kpt.write('reciprocal\n')
    for kpt in kpath:
        f_kpt.write('{:.6f} {:.6f} {:.6f} {:.2f}\n'.format(kpt[0],kpt[1],0,1))
    f_kpt.close()

def findRotationList(source_list,highestOrder=1000):
    global lat, pos_ions
    mag_pos=np.array([pos_ions[isource] for isource in source_list])
    numbers=[1]*len(source_list)
    cell=(lat,mag_pos,numbers)
    sym=spg.get_symmetry_dataset(cell,symprec=param.SYMPREC)
    rotationList=[]
    #for rotation,translation in zip(sym['rotations'],sym['translations']):
    #    rotationList.append([rotation,translation])
    return zip(sym['rotations'],sym['translations'])

def findBondList(source_list=[],maxRange=3,tolerance=0.01,
                 intraCell=True,getBondLen=False,dimmerMode=False):
    # get basic sym operations
    #global lat, pos_ions,nions
    #numbers=[]
    #for inion, nion in enumerate(nions):
    #    numbers.extend([inion]*nion)
    #cell=(lat,pos_ions,numbers)
    #sym=spg.get_symmetry_dataset(cell,symprec=param.SYMPREC)
    #print(sym)
    #exit()

    # find out each nearest ions to the source
    totSource=len(source_list)
    if(totSource==0):
        print('Error: source_list is not pointed or empty')
        exit()
    distance_list=[]
    pos_list=[]
    for isource, source_index in enumerate(source_list):
        # source coordinates
        r0=pos_ions[source_index]
        # firstly, find intra-cell bonds
        if(intraCell):
            dR_set=[np.array([dR1,dR2,dR3]) for dR1 in range(-maxRange, maxRange+1) for dR2 in range(-maxRange, maxRange+1) for dR3 in range(-maxRange, maxRange+1)]
            # with different type
            # no warrying about the time-reversal vectors since orbitals are different
            for dR in dR_set:
                for itar in range(isource+1,totSource):
                    target_index=source_list[itar]
                    r1=pos_ions[target_index]+dR
                    d_vec=np.dot((r1-r0),lat)
                    distance_list.append(np.sqrt(np.dot(d_vec,d_vec)))
                    pos_list.append([[source_index,target_index],dR])
                    if dimmerMode:
                        break

            if not dimmerMode:
                # with the same type
                # firstly, generate the time-reversal independent vector-set
                # chose order wisely so that we can get rid of zero-point
                dR_independent_set=[]
                while(True):
                    res=len(dR_set)
                    if res>1:
                        dR_set.pop(res-1)
                    else:
                        break
                    dR_independent_set.append(dR_set.pop(0))
                # then get bonds
                for dR in dR_independent_set:
                    target_index=source_index
                    r1=pos_ions[target_index]+dR
                    d_vec=np.dot((r1-r0),lat)
                    distance_list.append(np.sqrt(np.dot(d_vec,d_vec)))
                    pos_list.append([[source_index,target_index],dR])
        else:
            for itar in range(isource+1,totSource):
                target_index=source_list[itar]
                r1=pos_ions[target_index]
                d_vec=np.dot((r1-r0),lat)
                distance_list.append(np.abs(d_vec[2]))
                pos_list.append([[source_index,target_index],np.array([0,0,0])])
                        
    sort_list=aux.quicksort(distance_list)
    dis_max=-1.
    nn_count=-1
    BondList=[]
    BondLength=[]
    for idis, dis in enumerate(distance_list):
        if(dis>dis_max+tolerance):
            # find a new nn
            new_nn=[]
            new_bl=[]
            BondList.append(new_nn)
            BondLength.append(dis)
            nn_count+=1
            dis_max=dis
            
        BondList[nn_count].append(pos_list[sort_list[idis]])
        
    if(getBondLen):
        return BondList, BondLength
    return BondList

class Bond:
    def __init__(self, id, sourceOrb, targetOrb, overLat, lat, lat_inv):
        self.id=id
        # order source and target
        #if sourceOrb.id<=targetOrb.id:
        #    self.sourceOrb=sourceOrb
        #    self.targetOrb=targetOrb
        #else:
        #    self.sourceOrb=targetOrb
        #    self.targetOrb=sourceOrb
        self.lat=lat
        self.lat_inv=lat_inv
        self.sourceOrb=sourceOrb
        self.targetOrb=targetOrb
        self.overLat=overLat
        self.calcCoordinates(lat)
        self.createLocalFrame()

        self.bondType=-1 # classified according to linked elements and bond length

        self.exchangeEnergy=0#np.array([[0,0,0],[0,0,0],[0,0,0]]) # exchange energy in colinear form = J*S1*S2 negative for FM type bond
        #self.exchangeEnergy_Heisenberg=np.array([0,0,0]) # diagonal form exchange energy
        #self.exchangeEnergy_DMI=np.array([[0,0,0],
        #                                  [0,0,0],
        #                                  [0,0,0]])      # fully anisotropic exchange involving non-diagonal elements
        self.E0=0
        self.calcStatue=[] # store some info. in DFT calc.
        self.DNALabels=[]

        self.couplingNumber=-1
        self.sameLinkList=[]

    def calcCoordinates(self,lat):
        self.sourceCoorFrac=self.sourceOrb.pos
        self.targetCoorFrac=self.targetOrb.pos+self.overLat
        self.bondCenter=(self.sourceCoorFrac+self.targetCoorFrac)/2

        self.sourceCoorCart=np.dot(self.sourceCoorFrac,lat)
        self.targetCoorCart=np.dot(self.targetCoorFrac,lat)
        self.bondCenterCart=np.dot(self.bondCenter,lat)
        self.bondVec=self.targetCoorCart-self.sourceCoorCart
        self.bondLen=np.sqrt(np.dot(self.bondVec,self.bondVec))
    
    def createLocalFrame(self):
        # keep the xaxis as the unit bond vector
        xaxis=self.bondVec/self.bondLen
        zaxis=np.array(param.ZAXIS)
        yaxis=np.cross(zaxis,xaxis)
        yaxis_len=np.sqrt(np.dot(yaxis,yaxis))
        if yaxis_len<param.SYMPREC:
            if np.dot(zaxis,zaxis)>param.SYMPREC: # throw warning for ZAXIS!=0
                print("Warning: The bond direction (x) is parallel to zaxis, therefore, chose standard local frame")
                print("         And note that the bond-symmetry analysis is nonsense under standard frame")
                print("         To solve this issue, maybe chose another ZAXIS in GlobalConst")
            xaxis=np.array([1,0,0])
            yaxis=np.array([0,1,0])
            zaxis=np.array([0,0,1])
        else:
            yaxis/=yaxis_len
            zaxis=np.cross(xaxis,yaxis) # true zaxis
        
        self.localFrameCart=np.array([xaxis,yaxis,zaxis])
        self.localFrameFrac=np.dot(self.localFrameCart,self.lat_inv)
    
    def broadCastExchange(self,type='Ising',verbose=False):
        for bond in self.sameTypeBondList:
            if verbose:
                print('Setting exchange energy %.6f to bond%d (ion%d-ion%d [%d %d %d] bond length:%.3f)'%(self.exchangeEnergy,bond.id,bond.sourceOrb.id,bond.targetOrb.id,*bond.overLat,bond.bondLen))
            bond.E0=self.E0
            if type=='Ising':
                bond.exchangeEnergy=self.exchangeEnergy

    def showLocalFrameInFrac(self):
        print("local frame of bond%d:"%self.id)
        print("source %d target %d overLat %d %d %d"%(
            self.sourceOrb.id,self.targetOrb.id,self.overLat[0],self.overLat[1],self.overLat[2]
        ))
        bondVec=self.targetCoorFrac-self.sourceCoorFrac#self.targetOrb.pos-self.sourceOrb.pos+self.overLat
        #bondVec_cart=np.dot(bondVec,self.lat)
        #bondVec/=np.sqrt(np.dot(bondVec_cart,bondVec_cart))
        print("centre: %.6f %.6f %.6f vbond: %.6f %.6f %.6f"%(self.bondCenter[0],self.bondCenter[1],self.bondCenter[2],bondVec[0],bondVec[1],bondVec[2]))
        #a1=self.localXAxisFrac+self.bondCenter
        #a2=self.localYAxisFrac+self.bondCenter
        #a3=self.localZAxisFrac+self.bondCenter
        print("local frame (fractional coordinates):")
        for i in range(3):
            print("%.6f %.6f %.6f"%(self.localFrameFrac[i][0],self.localFrameFrac[i][1],self.localFrameFrac[i][2]))
        
        print("local frame (cartesian coordinates):")
        for i in range(3):
            print("%.6f %.6f %.6f"%(self.localFrameCart[i][0],self.localFrameCart[i][1],self.localFrameCart[i][2]))
        print("end of local frame")
    
    def report(self,short=False):
        print('bond%d is reporting'%self.id)
        print('bond type %d'%self.bondType)
        print('link %s-%s and %s-%s'%(self.sourceOrb.label,self.sourceOrb.wyckoffs,self.targetOrb.label,self.targetOrb.wyckoffs))
        print('link orb%d - orb%d [%d %d %d]'%(self.sourceOrb.id,self.targetOrb.id,*self.overLat))
        print("length: %.3f"%self.bondLen)
        print('(%.6f %.6f %.6f) - (%.6f %.6f %.6f)'%(self.sourceCoorFrac[0],self.sourceCoorFrac[1],self.sourceCoorFrac[2],self.targetCoorFrac[0],self.targetCoorFrac[1],self.targetCoorFrac[2]))
        if short:
            print('there are %d bond with the same type (classified via length)'%(len(self.sameTypeBondList)))
            return
        
        sameTypeID=[]
        for bond in self.sameTypeBondList:
            # build the sametype list
            existed=False
            for existedID in sameTypeID:
                if existedID==bond.id:
                    existed=True
                    break
            if not existed: sameTypeID.append(bond.id)
        print('there are %d bond with the same type'%(len(sameTypeID)))
        
        if param.VERBOSE:
            for bond in self.sameTypeBondList:
                print('%d - %d [%d %d %d] centred at (%.3f %.3f %.3f)'%(
                    bond.sourceOrb.id,bond.targetOrb.id,bond.overLat[0],bond.overLat[1],bond.overLat[2],
                    bond.bondCenter[0],bond.bondCenter[1],bond.bondCenter[2]
                ))
        print('bond symmetries:')
        for symOp, symParity in zip(self.symmetryOperationList,self.symmetryParityList):
            print(symOp[0],symOp[1])
            transSource=np.dot(symOp[0],self.sourceCoorFrac)+symOp[1]
            transTarget=np.dot(symOp[0],self.targetCoorFrac)+symOp[1]
            print('trans to (%.6f %.6f %.6f) - (%.6f %.6f %.6f)'%(transSource[0],transSource[1],transSource[2],transTarget[0],transTarget[1],transTarget[2]))
            print('parity:')
            print(symParity)
        self.showLocalFrameInFrac()
        
        # show independent components
        jComponentsList=[['xx','xy','xz'],
                         ['yx','yy','yz'],
                         ['zx','zy','zz'],]
        for i in range(3):
            for j in range(3):
                if self.independentComponentPosition[i][j]:
                    print('independent component %s'%jComponentsList[i][j])
                    for equi in self.equivalentComponentPosition[i][j]:
                        print('component %s is forced to equal %s'%(jComponentsList[i][j],jComponentsList[equi[0]][equi[1]]))
                    for anti in self.antivalentComponentPosition[i][j]:
                        print('component %s is forced to oppose %s'%(jComponentsList[i][j],jComponentsList[anti[0]][anti[1]])) 
        
    def defineBondTypeAndTrans(self,bondCandidateList,rotations,translations):
        # only bonds with the same length with me would presented in bondCandidateList to reduce computations
        def isZeroVector(vec):
            '''check if the vec is zero vec or just over 1.'''
            for x in vec:
                if abs(x%1)>param.SYMPREC and abs(1-x%1)>param.SYMPREC and abs(2-x%1)>param.SYMPREC:
                    return False
            return True

        self.sameTypeBondList=[]
        self.sameTypeBondOverLatList=[]
        self.sameTypeBondTransList=[]
        self.sameTypeBondParityList=[]
        for rotate, trans in zip(rotations, translations):
            self.sameTypeBondTransList.append([rotate,trans])
            #getCorrBond=False
            #pos_ref=np.dot(rotate,self.pos)+trans
            source_trans=np.dot(rotate,self.sourceCoorFrac)+trans
            target_trans=np.dot(rotate,self.targetCoorFrac)+trans
            for bondCandidate in bondCandidateList:
                diff_source=bondCandidate.sourceCoorFrac-source_trans
                diff_target=bondCandidate.targetCoorFrac-target_trans
                diff_vec=diff_target-diff_source
                if isZeroVector(diff_source) and np.dot(diff_vec,diff_vec)<param.SYMPREC: # transformed bond is the periodic object of candidate bond
                    self.sameTypeBondList.append(bondCandidate)
                    self.sameTypeBondOverLatList.append(np.array([int(round(x,0)) for x in diff_source]))
                    #frame_trans=np.dot(np.dot(self.localFrameFrac,rotate.T),self.lat)
                    #parityMatrix=[[checkTheParallelOrAntiParallel(vec1, vec2) for vec2 in bondCandidate.localFrameCart] for vec1 in frame_trans]
                    parityMatrix=self.localFrameFrac.dot(rotate.T).dot(self.lat).dot(self.lat.T).dot(self.localFrameFrac.T)
                    self.sameTypeBondParityList.append(
                        parityMatrix
                        )
                diff_source=bondCandidate.sourceCoorFrac-target_trans
                diff_target=bondCandidate.targetCoorFrac-source_trans
                diff_vec=diff_target-diff_source
                if isZeroVector(diff_source) and np.dot(diff_vec,diff_vec)<param.SYMPREC: # transformed bond is the periodic object of candidate bond, but reversed
                    self.sameTypeBondList.append(bondCandidate)
                    self.sameTypeBondOverLatList.append(np.array([int(round(x,0)) for x in diff_source]))
                    #frame_trans=np.dot(np.dot(self.localFrameFrac,rotate.T),self.lat)
                    #parityMatrix=[[checkTheParallelOrAntiParallel(vec1, vec2) for vec2 in bondCandidate.localFrameCart] for vec1 in frame_trans]
                    parityMatrix=self.localFrameFrac.dot(rotate.T).dot(self.lat).dot(self.lat.T).dot(self.localFrameFrac.T)
                    self.sameTypeBondParityList.append(
                        parityMatrix
                        )
        assert len(self.sameTypeBondList)==len(self.sameTypeBondTransList), 'the num. of sym. op do not well correspond to sym. bonds, it maybe helpful to increase the searching range BONDSEARCHRANGE'

        # check the parity related to self, i.e. the symmetry of the bond
        self.symmetryOperationList=[]
        self.symmetryParityList=[]
        for bond, symOp, parity in zip(self.sameTypeBondList,self.sameTypeBondTransList,self.sameTypeBondParityList):
            if bond.id==self.id:
                self.symmetryOperationList.append(symOp)
                self.symmetryParityList.append(parity)
        self.findIndividualComponents()

    def haveSameTypeWithMe(self,bond_ref):
        for bond in self.sameTypeBondList:
            if bond.id==bond_ref.id:
                return True
        return False

    def findIndividualComponents(self):
        equivalentPosition=[[[],[],[]],
                            [[],[],[]],
                            [[],[],[]],]
        antivalentPosition=[[[],[],[]],
                            [[],[],[]],
                            [[],[],[]],]
        forbiddenPosition=[[False,False,False],
                           [False,False,False],
                           [False,False,False]]
        independentPosition=[[True,True,True],
                             [True,True,True],
                             [True,True,True]]
        #print(self.symmetryParityList)
        for i in range(3):
            for j in range(3):
                probeMatrix=np.zeros((3,3))
                probeMatrix[i][j]=1
                probeMatrix=np.array(probeMatrix)
                #print('probeMatrx:')
                #print(probeMatrix)
                for similarTrans in self.symmetryParityList:
                    transformedProbeMatrix=similarTrans.T.dot(probeMatrix).dot(similarTrans)
                    if similarTrans[0][0]<0:
                        transformedProbeMatrix=transformedProbeMatrix.T
                    #print('transformed matrix:')
                    #print(transformedProbeMatrix)
                    for it in range(3):
                        for jt in range(3):
                            Tij=transformedProbeMatrix[it][jt]
                            if abs(Tij)>param.SYMPREC:
                                if abs(Tij-1)<param.SYMPREC and (i!=it or j!=jt):
                                    equivalentPosition[i][j].append([it,jt])
                                elif abs(Tij+1)<param.SYMPREC:
                                    if i==it and j==jt: # aniti-valent to itself -- forbidden
                                        forbiddenPosition[i][j]=True
                                    else:
                                        antivalentPosition[i][j].append([it,jt])
                #print('%d, %d equivalent to'%(i,j),equivalentPosition[i][j])
                #print('%d, %d antivalent to'%(i,j),antivalentPosition[i][j])
        # check the forbidden
        for i in range(3):
            for j in range(3):
                for equi in equivalentPosition[i][j]:
                    for anti in antivalentPosition[i][j]:
                        if equi[0]==anti[0] and equi[1]==anti[1]:
                            forbiddenPosition[i][j]==True
                            break
                    if forbiddenPosition[i][j]: break
        
        #print('Restriction imposed by symmetry:')
        for i in range(3):
            for j in range(3):
                if forbiddenPosition[i][j]:
                    #print('component %s is forced to zero'%jComponentsList[i][j])
                    independentPosition[i][j]=False
                else:
                    if independentPosition[i][j]:
                        for equi in equivalentPosition[i][j]:
                            #print('component %s is forced to equal %s'%(jComponentsList[i][j],jComponentsList[equi[0]][equi[1]]))
                            independentPosition[equi[0]][equi[1]]=False
                        for anti in antivalentPosition[i][j]:
                            #print('component %s is forced to oppose %s'%(jComponentsList[i][j],jComponentsList[anti[0]][anti[1]]))
                            independentPosition[anti[0]][anti[1]]=False
        
        self.independentComponentPosition=independentPosition
        self.equivalentComponentPosition=equivalentPosition
        self.antivalentComponentPosition=antivalentPosition
        
def findBondListAndClassifyThem(rotations, translations, source_list=[],maxRange=3,tolerance=0.01):#,additionTranslation=[np.array([0,0,0])]):
    global pos_ions, lat, orbList, bondList
    lat_inv=np.linalg.inv(lat)
    print("--->>> start classifying bonds <<<---")
    aux.showElapsedTime(offset=True)
    bondList_all, bondLen=findBondList(source_list=source_list,maxRange=maxRange,tolerance=tolerance,getBondLen=True)
    aux.showElapsedTime(extraInfo='searching links');aux.showElapsedTime(offset=True)
    bondList_flat=[]
    id=0
    if len(bondLen)<param.MAXRANGE:
        print('cannot found enough bond types for the consideration of %d/%dNN'%(len(bondLen),param.MAXRANGE))
        return[]
    for iBondLength in range(param.MAXRANGE):
        bondListWithSameLength=[]
        for bond in bondList_all[iBondLength]:  # bond with same length
            sourceOrb=orbList[bond[0][0]]
            targetOrb=orbList[bond[0][1]]
            overLat=bond[1]
            newBond=Bond(id, sourceOrb, targetOrb, overLat, lat, lat_inv)
            bondList_flat.append(newBond)
            bondListWithSameLength.append(newBond)
            id+=1
        aux.showElapsedTime(extraInfo='construct bonds with %dth length'%iBondLength);aux.showElapsedTime(offset=True)
        for bond in bondListWithSameLength:
            bond.defineBondTypeAndTrans(bondListWithSameLength, rotations, translations)
        aux.showElapsedTime(extraInfo='classify bonds with %dth length'%iBondLength);aux.showElapsedTime(offset=True)
    
    bondListWithClassification=[]
    def getBondTypeInCurrentList(inquir_bond,startID=0):
        totType=len(bondListWithClassification)
        for ibondType in range(startID,totType):
            bondType=bondListWithClassification[ibondType]
            for bond in bondType:
                if bond.haveSameTypeWithMe(inquir_bond):
                    return ibondType
                #if bond.checkIfTheBondHasSameTypeWithMe(inquir_bond,rotations,translations):
                #    return ibondType
        #print(inquir_bond.bondLen,inquir_bond.sourceOrb.label,inquir_bond.targetOrb.label)
        return -1 # <0 means this is a new type of bond
    # generate classified list
    for bond in bondList_flat:
        bondTypeID=getBondTypeInCurrentList(bond)
        if bondTypeID>=0:
            bond.bondType=bondTypeID
            bondListWithClassification[bondTypeID].append(bond)
        else:
            bond.bondType=len(bondListWithClassification)
            bondListWithClassification.append([bond])
    # check the bond type
    #for bond in bondList_flat:
    #    print(bond.id,bond.bondType)
    aux.showElapsedTime(extraInfo='sort bonds into type-hierarchy');aux.showElapsedTime(offset=True)
    print("--->>> bonds are successfully classified <<<---")
    return bondListWithClassification

def findBondListAndClassifyThemViaLength(source_list=[],maxRange=3,tolerance=0.01):#,additionTranslation=[np.array([0,0,0])]):
    global pos_ions, lat, orbList, bondList
    lat_inv=np.linalg.inv(lat)
    print("--->>> start classifying bonds <<<---")
    aux.showElapsedTime(offset=True)
    bondList_all, bondLen=findBondList(source_list=source_list,maxRange=maxRange,tolerance=tolerance,getBondLen=True)
    if len(bondLen)<param.MAXRANGE:
        print('cannot found enough bond types for the consideration of %d/%dNN'%(len(bondLen),param.MAXRANGE))
        return[]
    aux.showElapsedTime(extraInfo='searching links');aux.showElapsedTime(offset=True)
    bondList_flat=[]
    id=0
    for iBondLength in range(param.MAXRANGE):
        bondListWithSameLength=[]
        for bond in bondList_all[iBondLength]:  # bond with same length
            sourceOrb=orbList[bond[0][0]]
            targetOrb=orbList[bond[0][1]]
            overLat=bond[1]
            newBond=Bond(id, sourceOrb, targetOrb, overLat, lat, lat_inv)
            bondList_flat.append(newBond)
            bondListWithSameLength.append(newBond)
            id+=1
        aux.showElapsedTime(extraInfo='construct bonds with %dth length'%iBondLength);aux.showElapsedTime(offset=True)
        for bond in bondListWithSameLength:
            bond.sameTypeBondList=bondListWithSameLength
        #    bond.defineBondTypeAndTrans(bondListWithSameLength, rotations, translations)
        aux.showElapsedTime(extraInfo='classify bonds with %dth length'%iBondLength);aux.showElapsedTime(offset=True)
    
    bondListWithClassification=[]
    def getBondTypeInCurrentList(inquir_bond,startID=0):
        totType=len(bondListWithClassification)
        for ibondType in range(startID,totType):
            bondType=bondListWithClassification[ibondType]
            for bond in bondType:
                if abs(bond.bondLen-inquir_bond.bondLen)<param.SYMPREC:
                    return ibondType
                #if bond.checkIfTheBondHasSameTypeWithMe(inquir_bond,rotations,translations):
                #    return ibondType
        #print(inquir_bond.bondLen,inquir_bond.sourceOrb.label,inquir_bond.targetOrb.label)
        return -1 # <0 means this is a new type of bond
    # generate classified list
    for bond in bondList_flat:
        bondTypeID=getBondTypeInCurrentList(bond)
        if bondTypeID>=0:
            bond.bondType=bondTypeID
            bondListWithClassification[bondTypeID].append(bond)
        else:
            bond.bondType=len(bondListWithClassification)
            bondListWithClassification.append([bond])
    # check the bond type
    #for bond in bondList_flat:
    #    print(bond.id,bond.bondType,bond.bondLen)
    #exit()
    aux.showElapsedTime(extraInfo='sort bonds into type-hierarchy');aux.showElapsedTime(offset=True)
    print("--->>> bonds are successfully classified according to bond length <<<---")
    return bondListWithClassification

def getDfFactor(source_list=[],basicRotation=[],cRange=2,tolerance=0.01,MAXINV=100,getBonds=False,getTrans=False):
    global BondList
    global lat, pos_ions
    '''
    give you the connections between inverted mag.mom. and num. of AFM bonds
    the result will be a list:
    BondList[number of inequal inverting strateges with spcified invert num]
            [number of strategies][detail of strategy, num of AFM bonds]
            i.e. BondList[0] means the strategy-list of only inverting one magnetic ion
                 BondList[1] means the strategy-list of inverting double magnetic ions
                 ...
                 BondList[0][0] the first strategy of inverting one magnetic ion and the info.
                                about the induced AFM bonds
    the num. of AFM bonds is represented as an np.array, the first element represents the 
    num. of AFM bonds with 1 NN, the second element represents the 2nd NN AFM bonds, and so on
    
    param. definition:
    source_list: the list involving the index of each magnetic ions
    basicRotation: the basic rotation symmetry that the AFM state should impose
    banList    : all the pointed inverted number will not included in the generation
    cRange     : the max range bond considered, for cRange=0, no Bonds will be considered,
                 cRange=1, we only consider the nearest coupling bonds
                 cRange=2, the second nearest coupling will be considered, too.
                 and so on
    tolerance  : param. for the bond list construction. Bonds with lenght difference smaller than
                 tolerance will be treated as the same order.

    results
    infoList    : [(combination of inv.),[num. of AFM bonds] ]
    pg_orderList: [the point group orders]
    num_bonds   : [the number of bonds (in order)]
    '''
    time0=time.time()
    # get space group of mag. ions lat.
    mag_pos=np.array([pos_ions[isource] for isource in source_list])
    numbers=[1]*len(source_list)
    cell=(lat,mag_pos,numbers)
    origin_sym=spg.get_symmetry_dataset(cell,symprec=param.SYMPREC)
    print('space group: %s ID: %d'%(origin_sym['international'],origin_sym['number']))
    nBasicRotation=len(basicRotation)
    # get bond info.
    BondList=findBondList(source_list,cRange+2,tolerance)
    # assume Z2 symmetry
    totSource=len(source_list)
    ##------- scheme 1-----------------------#
    #max_inv=np.min([totSource/2,MAXINV])
    #min_inv=1
    #ninv_list=[min_inv+i for i in range(max_inv-min_inv+1)]
    #----------------------------------------#
    ##-------- scheme2 --------------------#
    max_inv=totSource/2                 
    min_inv=np.max([max_inv-MAXINV,1])
    ninv_list=[max_inv-i for i in range(max_inv-min_inv+1)]
    #--------------------------------------#
    ##----------------------------------#
    #min_inv=1
    #max_inv=np.min([min_inv+MAXINV,totSource/2+1])
    infoList=[]
    pg_orderList=[]
    sym_violate=0
    for num_inv in ninv_list:
        if(len(infoList)==param.NINFO4STR):
            break
        # generate inv list for each combination
        combList=aux.getCombinationLoop(source_list, ncomb=num_inv, maxComb=param.MAXCOMB)
        indexList=aux.getCombinationLoop([i for i in range(len(source_list))], ncomb=num_inv, maxComb=param.MAXCOMB)
        print('generate {:6d} combinations with {:3d} inverted spins in {:.3f}s'.format(len(combList),num_inv,time.time()-time0))
        time0=time.time()
        for icomb, comb in enumerate(combList):

            infoCombList=[]
            infoCombList.append(comb)
            # check AFM bonds up to cRange
            nAFMBonds=np.zeros(cRange)
            for rbond in range(cRange):
                for bond in BondList[rbond]:
                    source_index=bond[0][0]
                    target_index=bond[0][1]
                    FMtype=1
                    for ele in comb:
                        if(ele==source_index):
                            FMtype*=-1
                        if(ele==target_index):
                            FMtype*=-1
                    if(FMtype<0):
                        nAFMBonds[rbond]+=1
            infoCombList.append(nAFMBonds)
            # check the redundancy
            nontrivial=True
            for ele in infoList:
                nAFMBonds_existed=ele[1]
                if(nAFMBonds==nAFMBonds_existed).all():
                    nontrivial=False
                    break
            if(nontrivial):
                # check symmetry
                numbers=[1]*len(source_list) 
                # decide the new group
                for i in indexList[icomb]:
                    numbers[i]=2

                #leastSym=nBasicRotation*param.PGHOLD
                cell=(lat,mag_pos,numbers)
                afm_sym=spg.get_symmetry_dataset(cell,symprec=param.SYMPREC)
                afm_pg_order=len(afm_sym['rotations'])
                infoCombList.append(afm_pg_order)
                if(getTrans):
                    infoCombList.append(__getTransBack(afm_sym))
                infoList.append(infoCombList)
                pg_orderList.append(afm_pg_order)
                #print afm_pg_order, numbers
                print(infoCombList[:3], 'magnetic space group: ', afm_sym['international'])
                if(len(infoList)==param.NINFO4STR):
                    print('configurations filled out!')
                    break
            
    print('reserve {:6d} AFM configuration(s) in {:.3f}s'.format(len(infoList),time.time()-time0))
    print('point group orders:',pg_orderList)
    if getBonds:
        numBonds=[]
        for bondrank in range(cRange):
            numBonds.append(len(BondList[bondrank]))
        return infoList, np.array(numBonds)
    return infoList

def getDfFactorViaSymOperator(source_list=[],basicRotation=[],cRange=2,tolerance=0.01,MAXINV=100,getBonds=False,getTrans=False):
    '''
    Generate infoList according to the rotation operation
    first, we need basicRotation list, it stores all rotation operators of primitive cell
    then we select some site stochastically, and rotate them via the operations, and find
    all sites which are equal to them and set them all to inversed spins.

    input parameters are the same as before, the result, however, is also the same
    infoList contains all the permited (constrained by symmetry) AFM states
    each element contains three part, the inverted site (in total index, as a list),
    indicated AFM bonds (in list form) and the order of symmetry (num. of rotations operators)

    the difference is that, i'm trying to return the translation vector which will 'fix' 
    the positions of atoms of AFM state, making it seems more 'symmetric'
    '''
    global BondList
    global lat, pos_ions
    
    def isZeroVector(vec):
        '''check if the vec is zero vec or just over 1.'''
        for x in vec:
            if abs(x)>param.SYMPREC and abs(1-x)>param.SYMPREC and abs(2-x)>param.SYMPREC:
                return False
        return True
        #dot=np.dot(vec,vec)%1
        #if dot<param.SYMPREC or 1-dot<param.SYMPREC:
        #    return True
        #return False

    def checkIfNew(pos,posList,getIndex=False):
        '''Check if the pos is already in the posList'''
        for ipos,exist_pos in enumerate(posList):
            if(isZeroVector(pos-exist_pos)):
                if getIndex:
                    return ipos
                return False
        if getIndex:
            return -1
        return True

    def isNewComb(comb0,combList):
        '''check if comb0 is a new combination for combList'''
        for exist_comb in combList:
            if(len(comb0)==len(exist_comb)):
                if(comb0==exist_comb).all():
                    return False
        return True

    def isNewRotations(rotation0,checkList):
        '''check if the rotation is a totally new one'''
        for rotation in checkList:
            if(rotation0[0]==rotation[0]).all():
                return False
        return True

    def checkOrder(rotation):
        '''
        give the rotation order, that is how many time should this 
        rotation impose to recover the unity
        '''
        unit=np.diag(np.array([1,1,1]))
        order=1
        rotation0=np.array(list(rotation))
        while(True):
            if(rotation0==unit).all():
                return order
            rotation0=np.dot(rotation0,rotation)
            order+=1

    def getEqualPos(pos0List,transform):
        '''give me any position and the total symmetrical transformations
           return you all the equal positions
        '''
        totalPos=[(np.dot(trans[0],pos0)+trans[1])%1 for trans in transform for pos0 in pos0List]
        beautifulPos=[]
        for pos in totalPos:
            tmp=np.array([0. if (x<param.SYMPREC or 1-x<param.SYMPREC) else x for x in pos])
            if checkIfNew(tmp,beautifulPos):
                beautifulPos.append(tmp)
        return beautifulPos
    
    def generateSYMTransWithSameBias(all_trans,pghold=-1):
        transList=[]
        exist_bias=[]
        for trans in all_trans:
            for ix, x in enumerate(trans[1]):
                if(abs(x)<param.SYMPREC or 1-abs(x)<param.SYMPREC):
                    trans[1][ix]=0.
            ibias=checkIfNew(trans[1],exist_bias,getIndex=True)
            if pghold>0:
                order=checkOrder(trans[0])
                if order==1 or order==pghold:
                    if ibias<0:
                        # construct new set
                        exist_bias.append(trans[1])
                        transList.append([trans])
                    elif isNewRotations(trans,transList[ibias]):
                        transList[ibias].append(trans)
            else:
                if ibias<0:
                    # construct new set
                    exist_bias.append(trans[1])
                    transList.append([trans])
                elif isNewRotations(trans,transList[ibias]):
                    transList[ibias].append(trans)
            
        return transList,exist_bias

    def getAntiSiteAndNumbers(origin_pos,inverted_pos,source_list):
        '''
        we have inverted the spin of inverted_pos
        thue give us the numbers list
        '''
        numbers=[1]*len(origin_pos)
        antiSite=[]
        for iinv, inv_pos in enumerate(inverted_pos):
            find_corr=False
            for iori, ori_pos in enumerate(origin_pos):
                if isZeroVector(ori_pos-inv_pos):
                    find_corr=True
                    numbers[iori]=0
                    antiSite.append(source_list[iori])
                    break
            if not find_corr:
                print('ERROR: site lost, maybe the supercell is not symmetric')
                print('inverted mag:')
                print(inverted_pos)
                print('origin mag:')
                print(origin_pos)
                exit()
        return np.array(antiSite), numbers
    
    def getDualAntiSite(antiSite,source_list):
        '''get the dual antiSite list if it exceeds half sources'''
        if(len(antiSite)>len(source_list)/2):
            dual_antiSite=[]
            for site in source_list:
                not_inverted=True
                for anti in antiSite:
                    if anti==site:
                        not_inverted=False
                        break
                if not_inverted:
                    dual_antiSite.append(site)
            return np.array(dual_antiSite)
        return np.array(antiSite)
    
    time0=time.time()
    # get space group of mag. ions lat.
    mag_pos=np.array([pos_ions[isource] for isource in source_list])
    numbers=[1]*len(source_list)
    cell=(lat,mag_pos,numbers)
    origin_sym=spg.get_symmetry_dataset(cell,symprec=param.SYMPREC)
    #for rotation in origin_sym['rotations']:
    #    order
    orderMax=np.max([checkOrder(rotation) for rotation in origin_sym['rotations']])
    order_reserve=int(orderMax*param.PGHOLD)
    print('origin space group: ',origin_sym['international'],' highest order of group element:',orderMax,' reserve ',order_reserve)

    rotationList,biasList=generateSYMTransWithSameBias(zip(origin_sym['rotations'],origin_sym['translations']),pghold=order_reserve)
    print('there are {:3d} types of bias:'.format(len(biasList)))
    for ibias, bias in enumerate(biasList):
        print('bias: {:.3f} {:.3f} {:.3f} ; rotations order: {:3d}'.format(bias[0],bias[1],bias[2],len(rotationList[ibias])))
    # get bond info.
    BondList=findBondList(source_list,cRange+2,tolerance)
    #print 'bonds:',BondList[:cRange]
    # assume Z2 symmetry
    totSource=len(source_list)
    #max_inv=np.min([(totSource+1)/2,MAXINV])
    ##-------- test --------------------#
    max_inv=totSource/2+1              #
    min_inv=np.max([max_inv-MAXINV,1]) #
    ##----------------------------------#
    #min_inv=1
    #max_inv=np.min([min_inv+MAXINV,totSource/2+1])
    #-----------------------------------#
    infoList=[]
    sym_violate=0
    triedAntiSiteList=[]
    for num_inv in range(min_inv,max_inv):
        if(len(infoList)>=param.NINFO4STR):
            break

        # generate inv list for each combination
        combList=aux.getCombinationLoop(source_list, ncomb=num_inv, maxComb=param.MAXCOMB)
        print('generate combinations with {:3d} inverted spins in {:.3f}s. including {:5d} combs'.format(num_inv,time.time()-time0,len(combList)))
        time0=time.time()
        #triedAFMList=[]
        num_nontrivial_strategies=0
        for icomb, comb in enumerate(combList):
            if(len(infoList)>=param.NINFO4STR):
                break
            # generate the whole AFM state which satisfies all rotational symmetry
            inv_mag_pos=[np.array(list(pos_ions[isource])) for isource in comb]
            for rotation_with_bias in rotationList:
                all_inv_mag_pos=getEqualPos(inv_mag_pos,rotation_with_bias)
                antiSite,numbers=getAntiSiteAndNumbers(mag_pos,all_inv_mag_pos,source_list)
                # give up all over half configurations
                if(len(antiSite)==totSource):
                    continue
                num_nontrivial_strategies+=1
                if(len(antiSite)>1):
                    sort=aux.quicksort(antiSite)
                antiSite=getDualAntiSite(antiSite,source_list)
                if isNewComb(antiSite,triedAntiSiteList):
                    triedAntiSiteList.append(antiSite)
                else:
                    continue
            
                infoCombList=[]
                infoCombList.append(antiSite)
                # check AFM bonds up to cRange
                nAFMBonds=np.zeros(cRange)
                for rbond in range(cRange):
                    for bond in BondList[rbond]:
                        source_index=bond[0][0]
                        target_index=bond[0][1]
                        FMtype=1
                        for ele in antiSite:
                            if(ele==source_index):
                                FMtype*=-1
                            if(ele==target_index):
                                FMtype*=-1
                        if(FMtype<0):
                            nAFMBonds[rbond]+=1
                infoCombList.append(nAFMBonds)
                # check the redundancy
                nontrivial=True
                for ele in infoList:
                    nAFMBonds_existed=ele[1]
                    if(nAFMBonds==nAFMBonds_existed).all():
                        nontrivial=False
                        break
                if(nontrivial):
                    cell=(lat,mag_pos,numbers)
                    afm_sym=spg.get_symmetry_dataset(cell,symprec=param.SYMPREC)
                    afm_pg_order=len(afm_sym['rotations'])
                    infoCombList.append(afm_pg_order)
                    if(getTrans):
                        infoCombList.append(__getTransBack(afm_sym))
                    infoList.append(infoCombList)
                    print('comb:',infoCombList[0],'AFM bonds:',infoCombList[1],'order:',infoCombList[2], 'magnetic space group: ', afm_sym['international'],' elapsd:%.3f s'%(time.time()-time0))
  
        if num_nontrivial_strategies==0:
            print('did not find any nontrivial configurations, searching thus stop.')
            break
    print('find {:6d} AFM configuration(s) in {:.3f}s'.format(len(infoList),time.time()-time0))

    if getBonds:
        numBonds=[]
        for bondrank in range(cRange):
            numBonds.append(len(BondList[bondrank]))
        return infoList, np.array(numBonds)

    return infoList

def __getTransBack(sym):
    transport_back=-np.dot(np.linalg.inv(sym['transformation_matrix']),sym['origin_shift'])
    trans=[]
    for ele in transport_back:
        if(abs(ele)<param.SYMPREC or abs(1.-ele)<param.SYMPREC):
            trans.append(0.)
        else:
            trans.append(ele)
    transport_back=np.array(trans)
    transport_back[2]=0.
    return transport_back
