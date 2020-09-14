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
                bond.couplingNumber=len(bondList[0])
                if len(bondLength)>1: # there is other possible link
                    secondNNisOutOfConsideration=(bondLength[1]>bondLengthMax+param.SYMPREC)
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
            print('There are %d links with the same length as bond%d in SC'%(bond.couplingNumber,bond.id))
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
                    if bond.couplingNumber>1:
                        allBondTypesAreConfined=False
                        print("bond%d in bondType%d have %d>1 links in the supercell"%(bond.id,bond.couplingNumber,ibondType))
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

def calcIsotropicJ(mainDir,fermi=0,eta=0.01,emptySpace=5):
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
    # check if the J.txt exsited

    
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
    emin,emax=gr.calcFullDOS(hr,kmesh=kmesh,fermi=fermi,eta=eta,mainDir=mainDir,saveFig=True)
    
    for ibondType, bondType in enumerate(bondList):
        print("Now evaluate the bondType%d"%ibondType)
        bond=bondType[0]
        J=gr.calcJ(hr,sourceOrbList=bond.sourceOrb.wannIDList,
                   targetOrbList=bond.targetOrb.wannIDList,targetR=bond.overLat,
                   eta=eta,energyRange=(emin-emptySpace,fermi),
                   Lx=supercellSize[0],Ly=supercellSize[1],Lz=supercellSize[2],
                   linearPerturb=True)
        bond.exchangeEnergy=J
        bond.broadCastExchange(type='Ising')
    
    with open(mainDir+'J.txt','w') as fJ:
        fJ.write('Isotropic exchange energy J*S^2 up to %dNN calculated by Greens function\n'%(param.CALRANGE))
        
        for ibondType, bondType in enumerate(bondList):
            bond=bondType[0]
            fJ.write('%dNN ion%d - ion%d [%d %d %d] J=%.6f\n'%(
                (ibondType+1), bond.sourceOrb.id, bond.targetOrb.id, *bond.overLat, bond.exchangeEnergy))
    
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
          