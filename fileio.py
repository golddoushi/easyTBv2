import auxiliary as aux
import paramReader as param

global version
version='0.1'

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
            if (len(data)-3)!=len(bondList):
                print('File info: exchange truncation dose not match %d/%d'%(len(data)-3,len(bondList)))
                return False
            
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