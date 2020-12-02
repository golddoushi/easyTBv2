'''
Created on 2019 7 6
check something
@author: Andrew
'''
#
# Use to calculate the magnetic coupling via AFM-FM method
#
import re
#import subprocess
import paramReader as param
import posReader as pR
import auxiliary as aux
import shutil
import numpy as np
import time
#import WannierKit
import hrReader as reader
import ham as hamilton
import fileio
import matplotlib.pyplot as plt
from multiprocessing import Pool, freeze_support

global bondList, supercellSize

def classifyBonds(mainDir, source_list, wann_list, ini_S, completelyConfine=True):
    global bondList, supercellSize
    pR.clean()
    ini_mag=[S*2 for S in ini_S]
    pR.readPOS(mainDir+'POSCAR',ini_mag=ini_mag,ini_S=ini_S)
    
    # loop to find the proper supercell size
    sizeL,sizeW,sizeH=param.MINCELL,param.MINCELL,param.MINCELL
    if param.PBC==2: sizeH=1
    if param.PBC==1: sizeW=1
    if param.PBC==0: sizeL=1
    phase=0
    while(True):
        print("check if the supercell %d x %d x %d is sufficiently large"%(sizeL,sizeW,sizeH))
        pR.clean()
        pR.readPOS(mainDir+'POSCAR')
        pR.createSC(filename=mainDir+'POSCAR.tmp',size=[sizeL,sizeW,sizeH])#,source_list=source_list)
        #rotations_sc, translations_sc=pR.createSymmetryOperatorsForSC(rotations, translations, size=[sizeL,sizeW,sizeH])
        source_list_lo,ini_mag_lo,ini_S_lo=__updateSource(source_list,ini_mag,ini_S, sizeL*sizeW*sizeH)
        #pR.readPOS(path=mainDir+'POSCAR.tmp',ini_mag=ini_mag_lo,ini_S=ini_S_lo,rotations=rotations_sc,translations=translations_sc)
        pR.readPOS(path=mainDir+'POSCAR.tmp',ini_mag=ini_mag_lo,ini_S=ini_S_lo)
        # analyse bonds
        bondListWithClassification=pR.findBondListAndClassifyThemViaLength(source_list_lo,maxRange=param.BONDSEARCHRANGE,tolerance=param.SYMPREC)
        if len(bondListWithClassification)==0:
            print(">>>>>> supercell is not large enough <<<<<<")
            if param.PBC==1:
                sizeL+=1
            if param.PBC==2:
                if param.SYMEXPAND:
                    sizeL+=1;sizeW+=1
                else:
                    if phase%2==0:
                        sizeL+=1
                    else:
                        sizeW+=1
                    phase+=1
            if param.PBC==3:
                if param.SYMEXPAND:
                    sizeL+=1;sizeW+=1;sizeH+=1
                else:
                    if phase%3==0:
                        sizeL+=1
                    elif phase%3==1:
                        sizeW+=1
                    else:
                        sizeH+=1
                    phase+=1
            continue

        bondLengthMax=np.max([bondType[0].bondLen for bondType in bondListWithClassification])
        bondConfinedList=[]
        print('>>>>>> information for bonds <<<<<<')
        print('under the consideration up to %d NN, we hanve found %d types of bond'%(param.MAXRANGE,len(bondListWithClassification)))
        print('the bond length is truncated at %.6f'%bondLengthMax)
        for ibondType, bondType in enumerate(bondListWithClassification):
            print('******Bond type%d stand out******'%ibondType)
            bondType[0].report(short=True)
            bondConfinedForEachType=[]
            for ibond, bond in enumerate(bondType):
                bondList, bondLength=pR.findBondList(source_list=[bond.sourceOrb.id,bond.targetOrb.id],dimmerMode=True,intraCell=True,
                                                     maxRange=1,tolerance=param.SYMPREC,getBondLen=True)
                #bond.couplingNumber=len(bondList[0])
                bond.sameLinkList=bondList[0]
                if len(bondLength)>1: # there is other possible link
                    secondNNisOutOfConsideration=(bondLength[1]>bondLengthMax+param.SYMPREC)
                    if ibond==0: 
                        print('next linking overlat [%d %d %d] length: %.6f/%.6f'%(*bondList[1][0][1],bondLength[1],bondLengthMax))
                        #print(bondList[1][0])
                    #print("Round bonds are longer than %.3f"%bondLength[1])
                    if secondNNisOutOfConsideration: # the other links are out of cutoff, this bond is well confined in SC
                        bondConfinedForEachType.append(bond)
                    if param.VERBOSE:
                        print("bond%d is affected by %d - %d [%d %d %d] %.3f %s"%(
                            bond.id,bondList[1][0][0][0],bondList[1][0][0][1],bondList[1][0][1][0],bondList[1][0][1][1],bondList[1][0][1][2],
                            bondLength[1],secondNNisOutOfConsideration))
                else:
                    bondConfinedForEachType.append(bond) # there is not any other possible link, this bond thus is well confined in SC
                    #print("No round bonds are possible")
            print('There are %d links with the same length as bond%d in SC'%(len(bond.sameLinkList),bond.id))
            bondConfinedList.append(bondConfinedForEachType)
        # verify strategies
        allBondTypesAreConfined=True
        for ibondType, confineType in enumerate(bondConfinedList):
            if len(confineType)==0:
                allBondTypesAreConfined=False
                print("All bonds in bondType%d are not confined in the supercell"%ibondType)
                break
        if completelyConfine:
            for ibondType, bondType in enumerate(bondListWithClassification):
                for ibond, bond in enumerate(bondType):
                    if len(bond.sameLinkList)>1:
                        allBondTypesAreConfined=False
                        print("bond%d in bondType%d have %d>1 links in the supercell"%(bond.id,ibondType,len(bond.sameLinkList)))
                        for sameLink in bond.sameLinkList:
                            print('orb%d-orb%d over [%d %d %d]'%(*sameLink[0],*sameLink[1]))
                        break
                if not allBondTypesAreConfined:
                    break
        if not allBondTypesAreConfined:
            print(">>>>>> supercell is not large enough <<<<<<")
            if param.PBC==1:
                sizeL+=1
            if param.PBC==2:
                if param.SYMEXPAND:
                    sizeL+=1;sizeW+=1
                else:
                    if phase%2==0:
                        sizeL+=1
                    else:
                        sizeW+=1
                    phase+=1
            if param.PBC==3:
                if param.SYMEXPAND:
                    sizeL+=1;sizeW+=1;sizeH+=1
                else:
                    if phase%3==0:
                        sizeL+=1
                    elif phase%3==1:
                        sizeW+=1
                    else:
                        sizeH+=1
                    phase+=1
            continue

        print('>>>>>> All bonds are well confined in supercell %d x %d x %d<<<<<<'%(sizeL,sizeW,sizeH))
        aux.rm(mainDir+'POSCAR.tmp')
        break
    
    # construct representative orbitals and bonds
    pR.clean()
    pR.readPOS(mainDir+'POSCAR',ini_mag=ini_mag,ini_S=ini_S,wann_ID=wann_list)
    bondListWithClassification=pR.findBondListAndClassifyThemViaLength(source_list,maxRange=np.max([sizeL,sizeW,sizeH]),tolerance=param.SYMPREC)
    print('following bonds are to be evaluated')
    for ibondType, bondType in enumerate(bondListWithClassification):
        if ibondType>=param.CALRANGE:
            break
        bond=bondType[0]
        print('bond%d, linking orb%d (%s-%s) and orb%d (%s-%s) over (%d %d %d)'%(
            bond.id,bond.sourceOrb.id,bond.sourceOrb.label,bond.sourceOrb.wyckoffs,
            bond.targetOrb.id,bond.targetOrb.label,bond.targetOrb.wyckoffs,
            bond.overLat[0],bond.overLat[1],bond.overLat[2]
        ))
    
    bondList=bondListWithClassification
    supercellSize=[sizeL,sizeW,sizeH]
    #if param.MAGTYPE=='ISING':
    #    calcIsotropicJ(mainDir,bondList=bondListWithClassification,supercellSize=[sizeL,sizeW,sizeH])

def calcIsotropicJ(mainDir,sc=[10,10,1],fermi=0,eta=0.01,emptySpace=5, spinList=[], MAE=0):
    '''
    mainDir: the position of Hr file
    fermi: assumed Fermi level
    eta: eta of retarded Green's function
    emptySpace: preserved distance from band edge for the integration of Green's function 
    '''
    global bondList, supercellSize
    # read in Hr
    hr=reader.Hr()
    fileList=aux.ls(mainDir)

    # Try to read existed J.txt
    if fileio.readJ(mainDir,bondList):
        print('%sJ.txt has been read succesfully'%mainDir)
        for bondType in bondList:
            bondType[0].broadCastExchange(verbose=True)
        writeMCSolverIsingFile(mainDir)
        calcSpinWaveSpectra(mainDir,spinList,MAE)
        calcSpinWaveDOS(mainDir,spinList,MAE)
        return
    
    hrFileList=[mainDir+fileName for fileName in fileList if 'hr' in fileName]
    if len(hrFileList)==2:
        hr.readDoubleFiles(hrFileList[0],hrFileList[1],ignoreImagPart=True)
    elif len(hrFileList)==1:
        hr.readSingleFile(hrFileList[0],ignoreImagPart=True)
    elif len(hrFileList)==0:
        print("Error: cannot find Hr file in %s, pls. provide Hr file with name hr(.up or .dn) in %s"%(mainDir,mainDir))
        exit()
    else:
        print("Error: found more than two hr file in %s, pls. clean up tge directory"%mainDir)
        exit()
        
    # create Green's function
    gr=hamilton.Green()
    kmesh=pR.createKPOINTS(provideSize=supercellSize,getKMesh=True)
    emin,emax,fermi_adjusted=gr.calcFullDOS(hr,kmesh=kmesh,fermi=fermi,eta=eta,mainDir=mainDir,saveFig=True)
    fermi_adjusted=fermi
    gr.calcFullBands(hr,fermi=fermi_adjusted,mainDir=mainDir,saveFig=True)

    # first obtain kmesh and Hk 
    nkpt,c_kmesh_addr,c_Hk_addr=gr.getCTypeKMeshAndHk(hr,Lx=sc[0],Ly=sc[1],Lz=sc[2])
    # now the c part, enjoy :-P
    for ibondType, bondType in enumerate(bondList):
        bond=bondType[0]
        print("Now evaluate the bondType%d orb%d-orb%d [%d %d %d]"%(ibondType,bond.sourceOrb.id,bond.targetOrb.id,*bond.overLat))
        J=gr.calcJFourState_CDLL(hr,nkpt,c_kmesh_addr,c_Hk_addr,sourceOrbList=bond.sourceOrb.wannIDList,
                                  targetOrbList=bond.targetOrb.wannIDList,targetR=bond.overLat,
                                  theta=np.pi*0.01,eta=eta,energyRange=(emin-emptySpace,fermi_adjusted),
                                  Lx=sc[0],Ly=sc[1],Lz=sc[2],
                                  nthreads=param.NCORE)
        bond.exchangeEnergy=J
        bond.broadCastExchange(type='Ising')
    gr.freeKMeshAndHk(c_kmesh_addr,c_Hk_addr)
    
    fileio.writeJ(mainDir, bondList, sc)
    writeMCSolverIsingFile(mainDir)
    calcSpinWaveSpectra(mainDir,spinList,MAE)
    calcSpinWaveDOS(mainDir,spinList,MAE)
    
def calcAnisotropicJ(mainDir,fermi=0,eta=0.0001,emptySpace=5, ini_S=[], MAE=0.1):
    '''
    mainDir: the position of Hr file
    fermi: assumed Fermi level
    eta: eta of retarded Green's function
    emptySpace: preserved distance from band edge for the integration of Green's function 
    '''
    global bondList, supercellSize
    # read in Hr
    hr=reader.Hr()
    fileList=aux.ls(mainDir)

    # Try to read existed J.txt
    if fileio.readJ(mainDir,bondList):
        print('%sJ.txt has been read succesfully'%mainDir)
        for bondType in bondList:
            bondType[0].broadCastExchange(verbose=True)
        writeMCSolverIsingFile(mainDir)
        calcSpinWaveSpectra(mainDir,ini_S,MAE)
        
        return
    
    hrFileList=[mainDir+fileName for fileName in fileList if 'hr' in fileName]
    if len(hrFileList)==1:
        hr.readSingleFile(hrFileList[0],ignoreImagPart=False)
    elif len(hrFileList)==0:
        print("Error: cannot find Hr file in %s, pls. provide Hr file with name hr in %s"%(mainDir,mainDir))
        exit()
    else:
        print("Error: found more than two hr file in %s, pls. clean up tge directory"%mainDir)
        exit()
        
    # create Green's function
    gr=hamilton.Green()
    kmesh=pR.createKPOINTS(provideSize=supercellSize,getKMesh=True)
    emin,emax,fermi_adjusted=gr.calcFullDOS(hr,kmesh=kmesh,fermi=fermi,eta=eta,mainDir=mainDir,saveFig=True)
    gr.calcFullBands(hr,fermi=fermi_adjusted,mainDir=mainDir,saveFig=True)

    for ibondType, bondType in enumerate(bondList):
        bond=bondType[0]
        print("Now evaluate the bondType%d orb%d-orb%d [%d %d %d]"%(ibondType,bond.sourceOrb.id,bond.targetOrb.id,*bond.overLat))
        J=gr.calcJ(hr,sourceOrbList=bond.sourceOrb.wannIDList,
                   targetOrbList=bond.targetOrb.wannIDList,targetR=bond.overLat,
                   eta=eta,energyRange=(emin-emptySpace,fermi_adjusted),
                   Lx=supercellSize[0],Ly=supercellSize[1],Lz=supercellSize[2],
                   linearPerturb=True)
        bond.exchangeEnergy=J
        bond.broadCastExchange(type='Ising')
    
    fileio.writeJ(mainDir, bondList, supercellSize)
    writeMCSolverIsingFile(mainDir)
    calcSpinWaveSpectra(mainDir,ini_S,MAE)

def __updateSource(source_list,ini_mag,ini_S,size=2):
    outSource=[]
    outMag=[]
    outS=[]
    expand=size
    print('now size: %d'%size)
    for source in source_list:
        before_sites=source*expand
        for i in range(expand):
            outSource.append(before_sites+i)
    for mag in ini_mag:
        for i in range(expand):
            outMag.append(mag)
    for S in ini_S:
        for i in range(expand):
            outS.append(S)
    return outSource,outMag,outS
          
def writeMCSolverIsingFile(path):
    magOrbList=[orb for orb in pR.orbList if abs(orb.spin)>0]
    for iorb, orb in enumerate(magOrbList):
        orb.index_tmp=iorb
    with open(path+pR.getUniqueName()+'.txt','w') as f:
        f.write("This is mcsolver's save file, version: 2.2\n")
        f.write('Lattice:\n')
        for vec in pR.lat:
            f.write('%.9f %.9f %.9f\n'%(vec[0],vec[1],vec[2]))
        f.write('Supercell used in MC simulations:\n')
        if param.PBC==3:
            f.write('16 16 16\n')
        elif param.PBC==2:
            f.write('16 16 1\n')
        elif param.PBC==1:
            f.write('16 1 1\n')
        elif param.PBC==0:
            f.write('1 1 1\n')
        else:
            print('cannot write mcsolver file for ndimension>3')
            return
        f.write('Orbitals in cell:\n')
        f.write('%d\n'%len(magOrbList))
        f.write('Postions, initial spin states and onsite-anisotropy of every orbitals:\n')
        for orb in magOrbList:
            factor=param.EV2K/abs(orb.spin)
            f.write('orb %d : type %d spin %.1f pos [%.9f %.9f %.9f] Dz: %.9f Dx: %.9f Dy: %.9f\n'%(
                orb.index_tmp,orb.type,orb.spin*orb.magmom/abs(orb.magmom),*orb.pos,orb.D[2]*factor,orb.D[0]*factor,orb.D[1]*factor
            ))
        f.write('Bonds:\n')
        f.write('%d\n'%len(pR.bondList))
        f.write('id, source, target, overLat, Jz, Jx, Jy of each bond:\n')
        for bond in pR.bondList:
            s1=abs(bond.sourceOrb.spin)
            s2=abs(bond.targetOrb.spin)
            factor=param.EV2K/s1/s2
            f.write('bond %d: Jz %.9f Jx %.9f Jy %.9f Jxy %.9f Jxz %.9f Jyz %.9f Jyx %.9f Jzx %.9f Jzy %.9f orb %d to orb %d over [%d %d %d]\n'%(
                bond.id, bond.exchangeEnergy*factor, bond.exchangeEnergy*factor, bond.exchangeEnergy*factor, 0, 0, 0, 0, 0, 0,
                bond.sourceOrb.id,bond.targetOrb.id,*bond.overLat
            ))

def calcSpinWaveDOS(path,SList,MAE=0):
    magOrbID=[iorb for iorb in range(len(SList)) if SList[iorb]>0.1]
    avgMAE=MAE/len(magOrbID)

    # set half Brillouin zone
    kmesh=[]
    dk=1./param.SWNKPT
    for ikptx in range(param.SWNKPT):
        for ikpty in range(int(param.SWNKPT/2)):
            kmesh.append(np.array([ikptx*dk,ikpty*dk,0]))
    
    eigList=[]
    for ikpt, kpt in enumerate(kmesh):
        _ham=hamilton.Ham()
        for bond in pR.bondList:
            S=np.sqrt(SList[bond.sourceOrb.id]*SList[bond.targetOrb.id])
            t=bond.exchangeEnergy/S
            # exchange 
            _ham.addHopping(bond.sourceOrb.id,bond.targetOrb.id,t,bond.overLat,kpt)
            _ham.addHopping(bond.targetOrb.id,bond.sourceOrb.id,t,-bond.overLat,kpt)
            # onsite
            _ham.addHopping(bond.sourceOrb.id,bond.sourceOrb.id,-t,np.zeros(3),kpt)
            _ham.addHopping(bond.targetOrb.id,bond.targetOrb.id,-t,np.zeros(3),kpt)
        for iorb in magOrbID:
            _ham.addHopping(iorb,iorb,avgMAE/SList[iorb],np.zeros(3),kpt)

        eigList.append(_ham.sovle_exact())

    eigList=np.array(eigList)
    if(np.min(eigList)>0):
        Sigma=0
        for eig in eigList:
            Sigma+=1./eig
        Tc_=(S+1)*len(eigList)/3./Sigma
        print('mean field estimation for Tc: %.3f Kelvin\n'%(Tc_*1000*11.58875))

def calcSpinWaveSpectra(path,SList,MAE=0):
    kmesh=[]
    with open(path+'KPOINTS','r') as fin:
        data=fin.read().split('\n')
        nkpts=int(data[1])
        for iline in range(3,3+nkpts):
            k1,k2,k3,weight=data[iline].split()
            kmesh.append(np.array([float(k1),float(k2),float(k3)]))
    
    eigList,kaxis=[],[]
    for ikpt, kpt in enumerate(kmesh):
        _ham=hamilton.Ham()
        for bond in pR.bondList:
            S=np.sqrt(SList[bond.sourceOrb.id]*SList[bond.targetOrb.id])
            t=bond.exchangeEnergy/S
            # exchange 
            _ham.addHopping(bond.sourceOrb.id,bond.targetOrb.id,t,bond.overLat,kpt)
            _ham.addHopping(bond.targetOrb.id,bond.sourceOrb.id,t,-bond.overLat,kpt)
            # onsite
            _ham.addHopping(bond.sourceOrb.id,bond.sourceOrb.id,-t,np.zeros(3),kpt)
            _ham.addHopping(bond.targetOrb.id,bond.targetOrb.id,-t,np.zeros(3),kpt)



        eigList.append(_ham.sovle_exact())
        kaxis.append(ikpt)

    eigList=np.array(eigList)
    nbands=len(eigList[0])
    plt.figure()
    for ibands in range(nbands):
        plt.plot(kaxis,eigList[:,ibands],'-',c='blue')
    plt.xlim(0,ikpt)
    plt.xticks([])
    plt.xlabel('K')
    plt.ylabel('Energy')
    plt.savefig(path+'SpinWaveBands.png')
    plt.show()
    plt.close()