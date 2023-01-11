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
from scipy.special.orthogonal import p_roots
import time
#import WannierKit
import hrReader as reader
import ham as hamilton
import fileio
import matplotlib.pyplot as plt

global bondList, supercellSize

def classifyBonds(mainDir, source_list, wann_list, ini_S):
    global bondList, supercellSize
    pR.clean()
    ini_mag=[S*2 for S in ini_S]
    pR.readPOS(mainDir+'POSCAR',ini_mag=ini_mag,ini_S=ini_S,wann_ID=wann_list)
    
    '''
    phase=0
    while(True):
        if not checkSupercell: break
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
    '''
    # construct representative orbitals and bonds
    #pR.clean()
    #pR.readPOS(mainDir+'POSCAR',ini_mag=ini_mag,ini_S=ini_S,wann_ID=wann_list)
    # loop to find the proper supercell size
    def quenchNonperiodicPBC(sizeL,sizeW,sizeH):
        if param.PBC==2: sizeH=1
        if param.PBC==1: sizeW=1
        if param.PBC==0: sizeL=1
        return sizeL,sizeW,sizeH
    sizeL,sizeW,sizeH=quenchNonperiodicPBC(param.MINCELL,param.MINCELL,param.MINCELL)
    bondListWithClassification=pR.findBondListAndClassifyThemViaLength(source_list,maxRange=np.max([sizeL,sizeW,sizeH]),tolerance=param.SYMPREC)
    for iserch in range(param.MAXCELL):
        if len(bondListWithClassification)>0: break
        sizeL,sizeW,sizeH=quenchNonperiodicPBC(sizeL+1,sizeW+1,sizeH+1)
        bondListWithClassification=pR.findBondListAndClassifyThemViaLength(source_list,maxRange=np.max([sizeL,sizeW,sizeH]),tolerance=param.SYMPREC)
    else:
        print("Error: cannot found enough bond within supercell size below MAXCELL=%d"%param.MAXCELL)
        exit()
    
    print('following bonds are to be evaluated')
    bondList=[]
    for ibondType, bondType in enumerate(bondListWithClassification):
        if ibondType>=param.CALRANGE:
            break
        bond=bondType[0]
        #if checkSupercell:
        #    print('bond%d, linking orb%d (%s-%s) and orb%d (%s-%s) over (%d %d %d)'%(
        #    bond.id,bond.sourceOrb.id,bond.sourceOrb.label,bond.sourceOrb.wyckoffs,
        #    bond.targetOrb.id,bond.targetOrb.label,bond.targetOrb.wyckoffs,
        #    bond.overLat[0],bond.overLat[1],bond.overLat[2]
        #    ))
        #else:
        #    bond.report(short=True)
        bond.report(short=True)
        bondList.append(bondType)
    
    #bondList=bondListWithClassification
    supercellSize=[sizeL,sizeW,sizeH]
    #if param.MAGTYPE=='ISING':
    #    calcIsotropicJ(mainDir,bondList=bondListWithClassification,supercellSize=[sizeL,sizeW,sizeH])

def calcIsotropicJ(mainDir,sc=[10,10,1],fermi=0,eta=0.0001,erange=(-500,500),emptySpace=5,autoFixFermi=False,spinList=[], MAE=0, hrcutoff=-0.01):
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
        #calcSpinWaveSpectra(mainDir,spinList,MAE)
        #calcSpinWaveDOS(mainDir,spinList,MAE)
        return
    
    hrFileList=[mainDir+fileName for fileName in fileList if 'hr' in fileName]
    if len(hrFileList)==2:
        hr.readDoubleFiles(hrFileList[0],hrFileList[1],ignoreImagPart=True,cutoff=hrcutoff)
    elif len(hrFileList)==1:
        hr.readSingleFile(hrFileList[0],ignoreImagPart=True,cutoff=hrcutoff)
    elif len(hrFileList)==0:
        print("Error: cannot find Hr file in %s, pls. provide Hr file with name hr(.up or .dn) in %s"%(mainDir,mainDir))
        exit()
    else:
        print("Error: found more than two hr file in %s, pls. clean up tge directory"%mainDir)
        exit()
        
    # create Green's function
    gr=hamilton.Green()
    # obtain kmesh and Hk 
    nkpt,c_kmesh_addr,c_Hk_addr=gr.getCTypeKMeshAndHk(hr,Lx=sc[0],Ly=sc[1],Lz=sc[2],GammaMode=1,nthreads=param.NCORE)
    if autoFixFermi:
        emin,emax,fermi_adjusted=gr.calcFullDOS_CDLL(hr,c_Hk_addr,nkpt,hr.norb,fermi=fermi,eta=eta,mainDir=mainDir,saveFig=True,nthreads=param.NCORE)
    else:
        emin,emax,fermi_adjusted=-1000,1000,fermi
    emin=np.max([emin,erange[0]])
    emax=np.min([emax,erange[1]])
    print("Energy window for bands: [%f, %f]"%(emin,emax))
    #gr.calcFullBands_CDLL(hr,fermi=fermi,optimalFermi=fermi_adjusted,mainDir=mainDir,saveFig=True,nthreads=param.NCORE)
    
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
    #calcSpinWaveSpectra(mainDir,spinList,MAE)
    #calcSpinWaveDOS(mainDir,spinList,MAE)

def calcIsotropicJ_GaussianQuadrature(mainDir,sc=[10,10,1],
      fermi=0,eta=0.0001,erange=(-500,500),extraEnergyEmpty=5.,hrcutoff=-0.01,Jcvg=1e-3,
      lowOrder=5,highOrder=20,nthreads=param.NCORE):
    '''
    mainDir: the position of Hr file
    sc : k-grid used for BZ integration
    fermi: assumed Fermi level
    eta: eta of retarded Green's function
    Jcvg: convergence criterial for J, no convergence check if Jcvg<0 
    lowOrder, highOrder: the order used for Gaussian quadrature
    '''
    global bondList, supercellSize
    # read in Hr
    hr=reader.Hr()
    fileList=aux.ls(mainDir)

    # Try to read existed J.txt
    if False and fileio.readJ(mainDir,bondList):
        print('%sJ.txt has been read succesfully'%mainDir)
        for bondType in bondList:
            bondType[0].broadCastExchange(verbose=True)
        writeMCSolverIsingFile(mainDir)
        #calcSpinWaveSpectra(mainDir,spinList,MAE)
        #calcSpinWaveDOS(mainDir,spinList,MAE)
        return
    
    hrFileList=[mainDir+fileName for fileName in fileList if 'hr' in fileName]
    if len(hrFileList)==2:
        hr.readDoubleFiles(hrFileList[0],hrFileList[1],ignoreImagPart=True,cutoff=hrcutoff)
    elif len(hrFileList)==1:
        hr.readSingleFile(hrFileList[0],ignoreImagPart=True,cutoff=hrcutoff)
    elif len(hrFileList)==0:
        print("Error: cannot find Hr file in %s, pls. provide Hr file with name hr(.up or .dn) in %s"%(mainDir,mainDir))
        exit()
    else:
        print("Error: found more than two hr file in %s, pls. clean up tge directory"%mainDir)
        exit()
        
    # create Green's function
    # firstly, obtain all overLat
    overLat_set={(0,0,0)}
    for bondType in bondList: 
        overLat_set.add(tuple(bondType[0].overLat))
        overLat_set.add(tuple(-bondType[0].overLat))
    
    overLat_id_dic={}
    iOverLat, overlat_list=0, []
    for item in overLat_set:
        overLat_id_dic[item]=iOverLat
        overlat_list.append(item)
        iOverLat+=1
    # then, search all relevant orbs
    relevantOrb_set=set()
    for bondType in bondList:
        for iorb in bondType[0].sourceOrb.wannIDList:
            relevantOrb_set.add(iorb)
            relevantOrb_set.add(iorb+hr.norb//2)
        for iorb in bondType[0].targetOrb.wannIDList:
            relevantOrb_set.add(iorb)
            relevantOrb_set.add(iorb+hr.norb//2)
    relevantOrbID_list=[item for item in relevantOrb_set]
    sorted_relevantOrbID=np.sort(relevantOrbID_list)
    orbIDMapFromStandardToRelevant=np.zeros(hr.norb,dtype=int)-1
    for iRelevantOrb, orbID in enumerate(sorted_relevantOrbID):
        orbIDMapFromStandardToRelevant[orbID]=iRelevantOrb
    #print(orbIDMapFromStandardToRelevant)
    #exit()
    gr=hamilton.Green()
    
    print("Diagnolizing Hamiltonian")
    eigs_addr, vecs_addr=gr.calcEigsAndVecsInMemory_simple(hr,kgrid=sc,nthreads=nthreads)
    res=hamilton.Response()
    eigs=res.redOutQuantitiesOnKGrid(eigs_addr,data_dim=hr.norb,kmesh=sc,data_type='double')
    bandEnergyMin, bandEnergyMax= eigs.flatten().min(), eigs.flatten().max()
    print("Bands energy range: %.6f ~ %.6f"%(bandEnergyMin,bandEnergyMax))
    emin=np.max([erange[0],bandEnergyMin])-extraEnergyEmpty
    emax=np.min([erange[1],fermi])
    print("Energy window: [%f, %f]"%(emin,emax))
    
    print("Main loop for exchange calculation")
    J_list_old=np.zeros(len(bondList))
    J_list_new=np.zeros(len(bondList))
    iteration_num=0
    for gaussianOrder in range(lowOrder,highOrder):
        points, weights=p_roots(gaussianOrder)
        npoints=len(points)
        print("----- Iteration %d -----"%iteration_num)
        z,dz=aux.semiCirle(emin,emax,points)
        rG_w_lat=gr.calcLatticeGreenFunction_CDLL(
            hr,eigs_addr,vecs_addr,sorted_relevantOrbID,
            kgrid=sc,omega_list=z,lat_vecs=overlat_list,eta=eta,spinor=False,
            nthreads=nthreads)
        #print(rG_w_lat[-1,overLat_id_dic[(-2,0,0)]])
        #exit()
        # calc. charge density and spin density
        diag_list=np.array([np.diag(rG_w_lat[iw][overLat_id_dic[(0,0,0)]])*dz[iw]*weights[iw] for iw in range(npoints)]).imag/2
        rho_of_each_orb=sum(diag_list)
        print("Charges of source atoms: %.6f"%sum(rho_of_each_orb))
        print("Mag.mom of source atoms: %.6f"%(
            sum(rho_of_each_orb[np.arange(len(relevantOrbID_list)//2,dtype=int)])-
            sum(rho_of_each_orb[len(relevantOrbID_list)//2+np.arange(len(relevantOrbID_list)//2,dtype=int)])))
        
        # calc. exchange coupling energies for each bond
        for ibondType, bondType in enumerate(bondList):
            bond=bondType[0]
            i_overlat_p=overLat_id_dic[tuple(bond.overLat)]
            i_overlat_n=overLat_id_dic[tuple(-bond.overLat)]
            orbID,doubleCenter,center1,center2=gr.getFourStateScatterMatrix_x(hr,
                sourceOrbList=bond.sourceOrb.wannIDList,targetOrbList=bond.targetOrb.wannIDList,targetR=bond.overLat,
                Lx=sc[0],Ly=sc[1],Lz=sc[2])
            orbID=orbIDMapFromStandardToRelevant[orbID]
            #print('overlat')
            #print(bond.overLat)
            #print("scatter matrices")
            #print(center1)
            #print(center2)
            #print(doubleCenter)
            #print('orbID:',orbID)
            #print("G_lat:")
            #print(rG_w_lat[-1][i_overlat_p])
            #print(rG_w_lat[-1][i_overlat_n])
            #exit()
            J=np.zeros(len(orbID),dtype=complex)
            for iw in range(npoints):
                rG_w_p=rG_w_lat[iw][i_overlat_p][orbID][:,orbID]
                rG_w_n=rG_w_lat[iw][i_overlat_n][orbID][:,orbID]
                #contr_double=np.diag(doubleCenter@rG_w_p@doubleCenter@rG_w_n)
                #contr_single1=np.diag(center1@rG_w_p@center1@rG_w_n)
                #contr_single2=np.diag(center2@rG_w_p@center2@rG_w_n)
                contr=np.diag(center1@rG_w_p@center2@rG_w_n)
                #J+=(contr_double-contr_single1-contr_single2)*dz[iw]*weights[iw]
                J+=2*contr*dz[iw]*weights[iw]
            bond.exchangeEnergy=sum(J).imag/4
            dmi=sum(J).real/4
            #bond.broadCastExchange(type='Ising')
            print("bondType%d orb%d-orb%d [%d %d %d] len: %.4f J=%.6f (%.6f)"%(
                ibondType,bond.sourceOrb.id,bond.targetOrb.id,*bond.overLat,bond.bondLen,bond.exchangeEnergy,dmi))
            J_list_new[ibondType]=bond.exchangeEnergy
            #exit()
        
        # compute averaged J_list_new
        J_list_new=(J_list_old*iteration_num+J_list_new)/(iteration_num+1)
        # check convergence
        if Jcvg<0:
            print("Warning: no convergence check. Use the results at your own risk!")
            break
        dJ=J_list_new-J_list_old
        dJ_max=abs(dJ).max()
        print("Max dJ= %.6f"%dJ_max)
        if iteration_num>0 and dJ_max<Jcvg:
            print("----- Accuracy reached -----")
            break
        J_list_old=J_list_new.copy()
        iteration_num+=1
    else:
        print("Warning: Accuracy is not reached. Use the results at your own risk!")
    #J_list_old=(J_list_old*iteration_num+J_list_new)/(iteration_num+1)
    
    for J, bondType in zip(J_list_new,bondList):
        bondType[0].exchangeEnergy=J
        bondType[0].broadCastExchange(type='Ising')
    fileio.writeJ(mainDir, bondList, sc)
    writeMCSolverIsingFile(mainDir)
    #calcSpinWaveSpectra(mainDir,spinList,MAE)
    #calcSpinWaveDOS(mainDir,spinList,MAE)

def calcJFull_GaussianQuadrature(mainDir,sc=[10,10,1],
      fermi=0,eta=0.0001,erange=(-500,500),extraEnergyEmpty=5.,hrcutoff=-0.01,Jcvg=1e-3,
      lowOrder=5,highOrder=20,nthreads=param.NCORE):
    '''
    mainDir: the position of Hr file
    sc : k-grid used for BZ integration
    fermi: assumed Fermi level
    eta: eta of retarded Green's function
    Jcvg: convergence criterial for J, no convergence check if Jcvg<0 
    lowOrder, highOrder: the order used for Gaussian quadrature
    '''
    global bondList, supercellSize
    # read in Hr
    hr=reader.Hr()
    fileList=aux.ls(mainDir)

    # Try to read existed J.txt
    if False and fileio.readJ(mainDir,bondList):
        print('%sJ.txt has been read succesfully'%mainDir)
        for bondType in bondList:
            bondType[0].broadCastExchange(verbose=True)
        writeMCSolverIsingFile(mainDir)
        return
    
    hrFileList=[mainDir+fileName for fileName in fileList if 'hr' in fileName]
    if len(hrFileList)==2:
        hr.readDoubleFiles(hrFileList[0],hrFileList[1],ignoreImagPart=False,cutoff=hrcutoff)
    elif len(hrFileList)==1:
        hr.readSingleFile(hrFileList[0],ignoreImagPart=False,cutoff=hrcutoff)
    elif len(hrFileList)==0:
        print("Error: cannot find Hr file in %s, pls. provide Hr file with name hr(.up or .dn) in %s"%(mainDir,mainDir))
        exit()
    else:
        print("Error: found more than two hr file in %s, pls. clean up tge directory"%mainDir)
        exit()
        
    # create Green's function
    # firstly, obtain all overLat
    overLat_set={(0,0,0)}
    for bondType in bondList:
        # since we are to calc. DMI which depends on bond orientation, 
        # we have to include all overlat
        for bond in bondType:
            overLat_set.add(tuple(bond.overLat))
            overLat_set.add(tuple(-bond.overLat))
    
    overLat_id_dic={}
    iOverLat, overlat_list=0, []
    for item in overLat_set:
        overLat_id_dic[item]=iOverLat
        overlat_list.append(item)
        iOverLat+=1
        
    # then, search all relevant orbs
    relevantOrb_set=set()
    for bondType in bondList:
        for bond in bondType:
            for iorb in bond.sourceOrb.wannIDList:
                relevantOrb_set.add(iorb)
                relevantOrb_set.add(iorb+hr.norb//2)
            for iorb in bondType[0].targetOrb.wannIDList:
                relevantOrb_set.add(iorb)
                relevantOrb_set.add(iorb+hr.norb//2)
    relevantOrbID_list=[item for item in relevantOrb_set]
    sorted_relevantOrbID=np.sort(relevantOrbID_list)
    orbIDMapFromStandardToRelevant=np.zeros(hr.norb,dtype=int)-1
    for iRelevantOrb, orbID in enumerate(sorted_relevantOrbID):
        orbIDMapFromStandardToRelevant[orbID]=iRelevantOrb
        
    gr=hamilton.Green()
    
    print("Diagnolizing Hamiltonian")
    eigs_addr, vecs_addr=gr.calcEigsAndVecsInMemory_simple(hr,kgrid=sc,nthreads=nthreads)
    res=hamilton.Response()
    eigs=res.redOutQuantitiesOnKGrid(eigs_addr,data_dim=hr.norb,kmesh=sc,data_type='double')
    bandEnergyMin, bandEnergyMax= eigs.flatten().min(), eigs.flatten().max()
    print("Bands energy range: %.6f ~ %.6f"%(bandEnergyMin,bandEnergyMax))
    emin=np.max([erange[0],bandEnergyMin])-extraEnergyEmpty
    emax=np.min([erange[1],fermi])
    print("Energy window: [%f, %f]"%(emin,emax))
    
    print("Main loop for exchange calculation")
    J_list_old=np.zeros((len(bondList),3,3))
    J_list_new=np.zeros((len(bondList),3,3))
    iteration_num=0
    for gaussianOrder in range(lowOrder,highOrder):
        points, weights=p_roots(gaussianOrder)
        npoints=len(points)
        print("----- Iteration %d -----"%iteration_num)
        z,dz=aux.semiCirle(emin,emax,points)
        rG_w_lat=gr.calcLatticeGreenFunction_CDLL(
            hr,eigs_addr,vecs_addr,sorted_relevantOrbID,
            kgrid=sc,omega_list=z,lat_vecs=overlat_list,eta=eta,spinor=True,
            nthreads=nthreads)
        
        # calc. charge density and spin density
        diag_list=np.array([np.diag(rG_w_lat[iw][overLat_id_dic[(0,0,0)]])*dz[iw]*weights[iw] for iw in range(npoints)]).imag/2
        rho_of_each_orb=sum(diag_list)
        print("Charges of source atoms: %.6f"%sum(rho_of_each_orb))
        print("Mag.mom of source atoms: %.6f"%(
            sum(rho_of_each_orb[np.arange(len(relevantOrbID_list)//2,dtype=int)])-
            sum(rho_of_each_orb[len(relevantOrbID_list)//2+np.arange(len(relevantOrbID_list)//2,dtype=int)])))
        
        # calc. exchange coupling energies for each bond
        for ibondType, bondType in enumerate(bondList):
            for bond in bondType:
                i_overlat_p=overLat_id_dic[tuple(bond.overLat)]
                i_overlat_n=overLat_id_dic[tuple(-bond.overLat)]
            
                sourceOrbList=bond.sourceOrb.wannIDList
                targetOrbList=bond.targetOrb.wannIDList
                p_source=hr.getAtomicSelfEnergy(sourceOrbList)
                p_target=hr.getAtomicSelfEnergy(targetOrbList)
            
                orbID_source_up=list(orbIDMapFromStandardToRelevant[sourceOrbList])
                orbID_source_dn=list(orbIDMapFromStandardToRelevant[[i+hr.norb//2 for i in sourceOrbList]])
                orbID_target_up=list(orbIDMapFromStandardToRelevant[targetOrbList])
                orbID_target_dn=list(orbIDMapFromStandardToRelevant[[i+hr.norb//2 for i in targetOrbList]])
                orbID_source=orbID_source_up+orbID_source_dn
                orbID_target=orbID_target_up+orbID_target_dn
                
                J=0j
                J_a_xx=0j
                J_a_yy=0j
                J_a_xy=0j
                DMI=np.zeros(3,dtype=complex)
                for iw in range(npoints):
                    rG_w_p=rG_w_lat[iw][i_overlat_p][orbID_source][:,orbID_target]
                    rG_w_p_pauli_block =aux.PauliBlock(rG_w_p)

                    rG_w_n=rG_w_lat[iw][i_overlat_n][orbID_target][:,orbID_source]
                    rG_w_n_pauli_block =aux.PauliBlock(rG_w_n)

                    AMat=np.zeros((4,4),dtype=complex)
                    for i in range(4):
                        for j in range(4):
                            AMat[i,j]=np.trace(p_source@rG_w_p_pauli_block[i]@p_target@rG_w_n_pauli_block[j])
                
                    J+=2*(AMat[0,0]-AMat[1,1]-AMat[2,2]-AMat[3,3])*dz[iw]*weights[iw]
                    J_a_xx+=2*(AMat[1,1]+AMat[1,1])*dz[iw]*weights[iw]
                    J_a_yy+=2*(AMat[2,2]+AMat[2,2])*dz[iw]*weights[iw]
                    J_a_xy+=2*(AMat[1,2]+AMat[2,1])*dz[iw]*weights[iw]
                    DMI[0]+=2j*(AMat[0,1]-AMat[1,0])*dz[iw]*weights[iw]
                    DMI[1]+=2j*(AMat[0,2]-AMat[2,0])*dz[iw]*weights[iw]
                    #DMI[2]+=2j*(a0z-az0)*dz[iw]*weights[iw]

                bond.exchangeEnergy=J.imag/2
                bond.Jani=np.array([[J_a_xx,J_a_xy],[J_a_xy,J_a_yy]]).imag/2
                bond.DMI=DMI[:2].imag/2
                
                print("bondType%d orb%d-orb%d[%2d%2d%2d] len: %.2f J=%.4f Jax=%.4f Jay=%.4f Jaxy=%.4f Dx=%.4f Dy=%.4f"%(
                    ibondType,bond.sourceOrb.id,bond.targetOrb.id,*bond.overLat,bond.bondLen,
                    J.imag/4*1000,J_a_xx.imag/4*1000,
                    J_a_yy.imag/4*1000,J_a_xy.imag/4*1000,DMI[0].imag/4*1000,DMI[1].imag/4*1000))
            J_list_new[ibondType,:,:]=bond.exchangeEnergyMatrix
        
        # compute averaged J_list_new
        J_list_new=(J_list_old*iteration_num+J_list_new)/(iteration_num+1)
        # check convergence
        if Jcvg<0:
            print("Warning: no convergence check. Use the results at your own risk!")
            break
        dJ=J_list_new-J_list_old
        dJ_max=abs(dJ).max()
        print("Max dJ= %.6f"%dJ_max)
        if iteration_num>0 and dJ_max<Jcvg:
            print("----- Accuracy reached -----")
            break
        J_list_old=J_list_new.copy()
        iteration_num+=1
    else:
        print("Warning: Accuracy is not reached. Use the results at your own risk!")
    
    #for Jmat, bondType in zip(J_list_new,bondList):
    #    bondType[0].exchangeEnergyMatrix=Jmat
    #    bondType[0].broadCastExchange(type='Full')
    fileio.writeJmat(mainDir, bondList, sc)
    #writeMCSolverIsingFile(mainDir)

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
        #calcSpinWaveSpectra(mainDir,ini_S,MAE)
        
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
    #calcSpinWaveSpectra(mainDir,ini_S,MAE)

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
        f.write("This is mcsolver's save file, version: 2.3\n")
        f.write('Lattice:\n')
        for vec in pR.lat:
            f.write('%.9f %.9f %.9f\n'%(vec[0],vec[1],vec[2]))
        f.write('Supercell used in MC simulations:\n')
        if param.PBC==3:
            f.write('8 8 8\n')
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
        f.write('Positions, initial spin states and onsite-anisotropy of every orbital:\n')
        for orb in magOrbList:
            factor=param.EV2K/abs(orb.spin)**2
            f.write('orb %d : type %d spin %.2f pos [%.9f %.9f %.9f] Dz: %.9f Dx: %.9f Dy: %.9f\n'%(
                orb.index_tmp,orb.type,abs(orb.spin)*orb.magmom/abs(orb.magmom),*orb.pos,orb.D[2]*factor,orb.D[0]*factor,orb.D[1]*factor
            ))
        f.write('Bonds:\n')
        f.write('%d\n'%len(pR.bondList))
        f.write('id, source, target, overLat, Jz, Jx, Jy of each bond:\n')
        for bond in pR.bondList:
            s1=bond.sourceOrb.spin
            s2=bond.targetOrb.spin
            factor=param.EV2K/s1/s2 # NOTE: exchange energy obtained from perturbation has sign dependence on spin already
            f.write('bond %d: Jz %.9f Jx %.9f Jy %.9f Jxy %.9f Jxz %.9f Jyz %.9f Jyx %.9f Jzx %.9f Jzy %.9f orb %d to orb %d over [%d %d %d]\n'%(
                bond.id, bond.exchangeEnergy*factor, bond.exchangeEnergy*factor, bond.exchangeEnergy*factor, 0, 0, 0, 0, 0, 0,
                bond.sourceOrb.id,bond.targetOrb.id,*bond.overLat
            ))
        f.write('Dipole long range correction:\n')
        f.write('alpha 0.0\n')
        f.write('Temperature scanning region:\n')
        f.write('Tmin 100 Tmax 500 nT 8\n')
        f.write('Field scanning region:\n')
        f.write('Hmin 0 Hmax 0.1 nH 1\n')
        f.write('Measurement:\n')
        f.write('measure the correlation function between orb0 and orb0 over [0 0 0]\n')
        f.write('OrbGroup: 1\n')
        f.write('Supergroup\n')
        f.write('group0 orb0-orb0\n')
        f.write('Sweeps for thermalization and statistics, and relaxiation step for each sweep:\n')
        f.write('80000 640000 0\n')
        f.write('Model type:\n')
        f.write('Ising\n')
        f.write('Algorithm:\n')
        f.write('Metropolis\n')
        f.write('XAxis type:\n')
        f.write('T\n')
        f.write('Distribution output frame: 0 \n')
        f.write('Ncores:\n')
        f.write('4\n')

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
