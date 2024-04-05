import auxiliary as aux
import paramReader as param
import numpy as np
global version
version='0.1'

def readPOSCAR(filePath="./POSCAR"):
    '''
    Read in the POSCAR file and structualize the data in a dictionary\n
    '''
    resultDic={}
    with open(filePath,'r') as fin:
        data=fin.read().split("\n")

    resultDic['lattice']=np.array([[float(x) for x in data[iline].split()] for iline in range(2,5)])
    resultDic['label list']=data[5].split()
    resultDic['nions']=[int(x) for x in data[6].split()]
    resultDic['itype list']=[]
    for itype, num_atom in enumerate(resultDic['nions']):
        resultDic['itype list']+=[itype]*num_atom
    resultDic['tot ions']=sum(resultDic['nions'])
    resultDic['type']=data[7][0]
    iline_shift=0
    if resultDic['type']=='S':
        print(f"POSCAR with Selective dynamics {resultDic['type']}")
        iline_shift=1
        resultDic['type']=data[8][0]
        
    if resultDic['type']=='D':
        resultDic['pos ions (frac.)']=np.loadtxt(data[8+iline_shift:8+iline_shift+resultDic['tot ions']],usecols=(0,1,2,))
        resultDic['pos ions (cart.)']=resultDic['pos ions (frac.)']@resultDic['lattice']
    elif resultDic['type']=='C':
        resultDic['pos ions (cart.)']=np.loadtxt(data[8+iline_shift:8+iline_shift+resultDic['tot ions']],usecols=(0,1,2,))
        resultDic['pos ions (frac.)']=resultDic['pos ions (cart.)']@np.linalg.inv(resultDic['lattice'])
    else:
        print(f"Error: coordinates type should be either D or C, now is {resultDic['type']}")
        exit()
    return resultDic

def readHrDat(filepath,ignoreImagPart=False,cutoff=-1,format="VASP"):
    aux.showElapsedTime(True)
    with open(filepath,'r') as fin:
        headline=fin.readline()
        print('reading file %s, title line:%s'%(filepath,headline))
        norb=int(fin.readline())
        if format=="QE": assert norb%2==0, "format is set to QE, means we are doing soc case, but norb is odd"

        totWaveVectors=int(fin.readline())
        iline=3
        recordedWaveVectors=0
        while(True):
            waveVectorPack=fin.readline().split()
            recordedWaveVectors+=len(waveVectorPack)
            iline+=1
            if recordedWaveVectors>=totWaveVectors:break
            
    data=np.loadtxt(filepath,skiprows=iline)
    aux.showElapsedTime(extraInfo="%s loaded"%filepath)
    
    nlines, ncols= data.shape
    if ncols == 9:
        overlap=True
        print("Warning: overlap is read from hr.dat, the eigen-eq should be solved with considering overlap")
    elif ncols == 7:
        overlap=False
        print("Warning: overlap is not found, assuming the orthorgonal basis")
    else:
        print("Error: unkown format of hr.dat with %d cols"%ncols)

    dRList = np.int16(data[:,:3])
    sourceList, targetList = np.int16(data[:,3:5].T)-1
    tList = data[:,5] + 1j* data[:,6]
    if ignoreImagPart: tList=tList.real
    if overlap:
        oList = data[:,7] + 1j* data[:,8]
        if ignoreImagPart: oList=oList.real
    
    Rx_max, Rx_min, Ry_max, Ry_min, Rz_max, Rz_min= dRList[:,0].max(), dRList[:,0].min(), dRList[:,1].max(), dRList[:,1].min(), dRList[:,2].max(), dRList[:,2].min()
    #dRList=[tuple(dR) for dR in dRList]
    ham_R    =np.zeros((Rx_max-Rx_min+1,Ry_max-Ry_min+1,Rz_max-Rz_min+1,norb,norb),dtype=complex)
    overlap_R=np.zeros((Rx_max-Rx_min+1,Ry_max-Ry_min+1,Rz_max-Rz_min+1,norb,norb),dtype=complex)

    for ibond, (dR, source, target, t) in enumerate(zip(dRList,sourceList,targetList,tList)):
        if(iline+1)%((nlines//50+1))==0: print('>',end='') # process bar

        # unpack the bond info
        if overlap:
            o=oList[ibond]
        else:
            o=0
            if dR@dR==0 and source==target:  o=1
        # skip negligible bond
        if abs(t) < cutoff and abs(o) < cutoff: continue
        # add bond
        ham_R[dR[0],dR[1],dR[2],source,target]    =t
        overlap_R[dR[0],dR[1],dR[2],source,target]=o
        
    aux.showElapsedTime(extraInfo='real-space grid: %dx%dx%d, ham_R read'%(*ham_R.shape[:3],))
    return {
        "ham_R":ham_R,
        "overlap":overlap,
        "overlap_R":overlap_R,
        "dRList":dRList
    }

def readJ(mainDir, bondList):
    global version
    fileList=aux.ls(mainDir)
    if 'J.txt' in fileList:
        with open(mainDir+'J.txt','r') as fJ:
            data=[line for line in  fJ.read().split('\n') if line]
            if len(data)==0:
                print('File info: J.txt is empty')
                return False
            headline=data[0].split()
            if len(headline)==0:
                print('File info: head line is empty')
                return False
            versionInFile=headline[len(headline)-1]
            if versionInFile!=version:
                print('File info: version does not match')
                return False
            if (len(data)-3)!=sum([len(bond) for bond in bondList]):
                print('File info: exchange truncation dose not match %d/%d'%(len(data)-3,len(bondList)))
                return False
            if len(data[3].split())==8: # isotropic exchange, only one ele of J
                for iline in range(3,len(data)):
                    NN,  distance,  sourceID,  targetID,  Rx,  Ry,  Rz,  J = data[iline].split()
                    iBondType=iline-3
                    bond=bondList[iBondType][0]
                    if bond.sourceOrb.id==int(sourceID) and bond.targetOrb.id==int(targetID) and bond.overLat[0]==int(Rx) and bond.overLat[1]==int(Ry) and bond.overLat[2]==int(Rz):
                        bond.exchangeEnergy=float(J)
                    else:
                        print('File info: bond does not match ion%d-ion%d [%d %d %d] <-> ion%s-ion%s [%s %s %s]'%(
                            bond.sourceOrb.id,bond.targetOrb.id,*bond.overLat,sourceID,targetID,Rx,Ry,Rz))
                        return False
            elif len(data[3].split())==13: # anisotropic exchange
                tag=np.array([False for i in range(len(data)-3)])
                _ibond_abs=0
                for ibondType, bondType in enumerate(bondList):   # scan all bonds should be addressed
                    for ibond_inMem, bond in enumerate(bondType): # scan all bonds should be addressed
                        for ibond_fromTXT in range(len(data)-3):  # scan txt for the data
                            NN,  distance,  sourceID,  targetID,  Rx,  Ry,  Rz,  J, J_a_xx, J_a_yy, J_a_xy, Dx, Dy = data[3+ibond_fromTXT].split()
                            if bond.sourceOrb.id==int(sourceID) and bond.targetOrb.id==int(targetID) and bond.overLat[0]==int(Rx) and bond.overLat[1]==int(Ry) and bond.overLat[2]==int(Rz):
                                bond.exchangeEnergy=float(J)
                                bond.Jani=np.array([[float(J_a_xx),float(J_a_xy)],[float(J_a_xy),float(J_a_yy)]])
                                bond.DMI=np.array([float(Dx),float(Dy)])
                                tag[_ibond_abs]=True
                                data.pop(3+ibond_fromTXT)
                                _ibond_abs+=1
                                break
                        else:
                            print('File info: cannot find the info. of bond NN%d (bond%d) ion%d-ion%d [%d %d %d]'%(
                                ibondType,ibond_inMem,bond.sourceOrb.id,bond.targetOrb.id,*bond.overLat))
                            return False
                if not tag.all():
                    print("Failed to address all bonds",tag)
                    return False
        return True
    print('File info: J.txt does not exist')
    return False
            
def writeJ(mainDir, bondList, supercellSize):
    global version
    with open(mainDir+'J.txt','w') as fJ:
        fJ.write('Isotropic exchange energy J*S^2 up to %dNN calculated by Greens function. version: %s\n'%(param.CALRANGE,version))
        fJ.write('evaluated in %dx%dx%d supercell\n'%(*supercellSize,))
        fJ.write('#NN  distance  sourceID  targetID  Rx  Ry  Rz  nJ  J\n')

        for ibondType, bondType in enumerate(bondList):
            bond=bondType[0]
            fJ.write('%3d%12.6f%3d%3d%3d%3d%3d%3d%12.6f\n'%(
                (ibondType+1), bond.bondLen, bond.sourceOrb.id, bond.targetOrb.id, *bond.overLat, len(bondType), bond.exchangeEnergy))

def writeJmat(mainDir, bondList, supercellSize):
    global version
    with open(mainDir+'J.txt','w') as fJ:
        fJ.write('Anisotropic exchange energy J*S^2 up to %dNN calculated by Greens function. version: %s\n'%(param.CALRANGE,version))
        fJ.write('evaluated in %dx%dx%d supercell\n'%(*supercellSize,))
        fJ.write('#NN   dis.  is  it  Rx  Ry  Rz             J          Ja_x          Ja_y         Ja_yx            Dx            Dy\n')

        for ibondType, bondType in enumerate(bondList):
            for bond in bondType:
                fJ.write('%3d%7.3f%4d%4d%4d%4d%4d%14.6e%14.6e%14.6e%14.6e%14.6e%14.6e\n'%(
                    (ibondType+1), bond.bondLen, bond.sourceOrb.id, bond.targetOrb.id, *bond.overLat,  
                    bond.exchangeEnergy,bond.Jani[0,0],bond.Jani[1,1],bond.Jani[0,1],bond.DMI[0],bond.DMI[1]))