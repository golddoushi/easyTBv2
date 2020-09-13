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

'''
global dftscript_std, dftscript_ncl
global couplingDir
global anistropyDir
global banStrategy
global FMEnergy
'''

def mainLoop(mainDir, source_list, ini_mag, ini_S, completelyConfine=True):
    #global couplingDir, dftscript_std, dftscript_ncl, banStrategy, FMEnergy
    #couplingDir=mainDir+'coupling/'
    #anistropyDir=mainDir+'anistropy/'
    # create dir for coupling and anistropy calc. and create POTCAR
    #aux.createDir(couplingDir)
    #aux.createDir(anistropyDir)
    #filepath=mainDir+'POSCAR'
    pR.clean()
    pR.readPOS(mainDir+'POSCAR',ini_mag=ini_mag,ini_S=ini_S)
    #rotations,translations=pR.getWholeSymmetryOperator()
    
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
    pR.readPOS(mainDir+'POSCAR',ini_mag=ini_mag,ini_S=ini_S)
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
    
    '''
    # start calc. in supercell
    scDir=couplingDir+'sc%dx%dx%d/'%(sizeL,sizeW,sizeH)
    aux.createDir(scDir)
    shutil.move(mainDir+'POSCAR.tmp',scDir+'POSCAR')
    aux.cp(mainDir+"POTCAR",scDir+'POTCAR')
    if param.MAGTYPE=='ISING':
        print('>>>>>> start calculations for each bond type with ISING mode (scalar J) <<<<<<')
        status=calcIsingBond(scDir,orbList,bondList)
    
    if param.MAGTYPE=='FULL':
        print('>>>>>> start calculations for each bond type with FULL mode (J with nine independent matrix elements) <<<<<<')
        status=calcFullAnisotropicBond(scDir,orbList,bondList)
    
    if status: cleanAllPOTCAR(scDir) # clean file
    bondListWholeFlatten=[bond for bond in bondType for bondType in bondListWithClassification]
    return status, orbList, bondListWholeFlatten
    '''
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
          