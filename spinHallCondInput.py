import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from multiprocessing import freeze_support

import paramReader
paramReader.readGlobalConst()
import numpy as np
from system import System_omx, System_w90, System_tb
from task import BZGridTask, BZLineTask
#+------------+
#| Bulk tasks |
#+------------+
from task import solvePDOS, solveVelocity, solveVelocity_diag
from task import solveSHC, solveSHC_LG_DM, solveSHC_diag, solveOHC, solveStGC, solveAHC, solveAHC_LG_DM
from task import solveSGC, solveSGC_diag, solveOGC, solveOGC_diag
from task import solveSGC_LG_DM, solveOGC_LG_DM, solveStGC_LG_DM
from task import solveBPVE_TDSE, solveBPVE_VG, solveBPVE_VG_DM, solveBPVE_TDSE_LG, solveBPVE_LG_Shift, solveBPVE_LG_Inject, solveBPVE_LG_DM
from task import solveIFE_TDSE, solveIFE_BlochAcceleration_TDSE, solveIFE_VG, solveIFE_VG_DM, solveIFE_LG, solveIFE_LG_DM
from task import solveIFTorque_TDSE, solveIFTorque_VG_DM, solveIFTorque_LG_DM
from task import solveINPHE_BCP
#+------------+
#| Line tasks |
#+------------+
from task import solveSurfaceSpectraTask, solveBandStructure
#+---------------------------+
#| Tasks free from interface |
#+---------------------------+
from calculator import Chern_FHS, Z2_FHS, CollisionSpecification
from auxiliary import showElapsedTime, getCBM, getVBM
from fileio import (DOSCAR, writeVolumeData, output3rdFreqDenpendentData, outputIFEcoeffs, outputIFTcoeffs,
                    output4thData, outputBZDistributionData, outputIFEResponseOnBZ, outputIFTResponseOnBZ)

def run():
    # Pbene, spin hall conductance
    mainDir="./mcalc/FeCl2/"
    model=System_w90(
        mainDir=mainDir,
        #hamR="omx",
        #hamR="BN_fm_rashiba",
        #hamR="wannier90_mx",
        cutoff=-0.002,
        poscar="POSCAR",
        projections={
            "B":(0,),
            "N":(0,),
            "Ge":(0,0,1,2),
            #"Mn":(0,0,0,1,1,2,),
            "Mn":(0,1,2,),
            "Sn":(0,1,),
            #"Sn":(0,0,0,1,1,2,),
            "Pb":(0,1,),
            #"Pb":(0,0,0,1,1,2,2,),
            "Pt":(0,1,2,),
            #"Pt":(0,0,1,1,2,2,),
            "C":(0,1,),
            "Fe":(0,1,2,),
            "Mo":(2,),
            "Nb":(0,2,),
            "S":(1,),
            "Bi":(0,1,),
            "Te":(0,1,),
            #"Fe":(0,0,1,1,2,2,),
            "Cl":(1,),
            "Se":(1,),
            #"Te":(1,),
            #"Cl":(0,0,1,1,2,3),
            "H":(0,),
            #"H":(0,0,1,),
            "Au":(0,0,0,1,1,2,2,),
            "Sr":(0,0,0,1,1,2,2,),
            "Ti":(0,0,0,1,1,2,),
             "O":(0,0,1,1,2,),
        },
        ispin=2,
        spinSeparated=True,
    )
    #'''
    # Symmetrize the direction-independent collinear and orbital-SOI
    # ingredients once, then assemble any requested magnetization direction.
    model.symmerizeCollinearAndSOI()
    model.getExchangeR(provenance="pauli estimate")
    magnetizationTheta ,magnetizationPhi = 0/180*np.pi, 0.0
    model.includeSOCFromSOI(theta=magnetizationTheta,phi=magnetizationPhi,strength=1,keepRegularSpinAxis=True)
    #model.rotateExchangeDirection(theta=magnetizationTheta,phi=magnetizationPhi)
    #model.getExchangeR(provenance="explicit")
    #'''

    # include el-ph self energy
    #model.includeElectronSelfEnergy(f"{mainDir}/vaspout.h5",temperature=300.,)

    # For an independent noncollinear VASP model, rotate its SAXIS frame manually
    # NOTE: this is only for independent vasp_ncl + wannier90 scheme, DO NOT apply for vasp_std + V_soi scheme
    #model.rotateSpinBasisToGlobal(alpha=np.pi/2, beta=np.pi/2) # SAXIS = 0 1 0
    #model.rotateSpinBasisToGlobal(alpha=0, beta=np.pi/2) # SAXIS = 1 0 0
    
    '''
    # global symmetrization
    # usage: regularized independent noncollinear model, collinear model
    # NOT for vasp_std + V_soi scheme, use the symmerizeCollinearAndSOI rather
    atomicMagMomList=[
        np.array([1, 0, 0]),
    ]*1+[
        np.array([0, 0, 0]),
    ]*2
    model.symmerizeHamAndOverlap(
        atomicMagMomList=atomicMagMomList,spinor=True,print_row=(0,1,9,10,),print_col=(0,1,9,10,),
    )
    #'''

    # modulate the SOC, exchange fields, baised fields, etc.
    #magnetizationTheta ,magnetizationPhi = 0.0, 0.0
    #model.getExchangeR(provenance="explicit")
    #model.scaleSOC(lam=0)
    #model.rotateExchangeField(projection=[(0,(0,)),(1,(0,))], theta=np.pi, phi=0.)
    #model.applyElectrostaticPotentialAtOnsites([0., 0., 0.,]) # in eV/angstrom
    '''
    # create supercell model and cut into nanoribbon
    Tmatrix=[
        [10, 0, 0], # The A1 of the supercell in the frame of primitive cell
        [ 1, 2, 0], # The A2 of the supercell in the frame of primitive cell
        [ 0, 0, 1]  # The A3 of the supercell in the frame of primitive cell
    ]
    model.make_supercell(Tmatrix)
    surfaceDir=0
    model.shiftOrigin(surfaceDir=surfaceDir,)
    model.cutEdge(cutDir=surfaceDir,)
    # add exchange splitting for edge atoms
    #model.addExchangeSplitting(edgeRegion1,[ 0.0, 0.5, 0.0])
    #model.addExchangeSplitting(edgeRegion2,[ 0.0,-0.5, 0.0])
    #exit()
    #'''
    #fermi=0  # OPNEMX generated wannier model always displays zero fermi energy
    #fermi=-0.167662613970*27.2113845 # OPENMX Ge
    #fermi=-3.85                       # OPENMX Sn topological insulator
    #fermi=-4.2                         # OPENMX Pb topological insulator
    #fermi=-0.17020527415105*27.2113845 # OPENMX PbSnH topological insulator
    #fermi=-3.89                         # OPENMX PbSn offset topological insulator
    #fermi=-0.16790565257604*27.2113845 # OPENMX Pt
    #fermi=-0.19908723865958*27.2113845 # OPENMX FeCl2
    #fermi=-0.22660759670266*27.2113845 # OPENMX AuH 111
    #fermi=-0.19580173375979*27.2113845 # OPENMX Au 111
    #fermi=-2.41861676 # VASP Sn
    #fermi=-2.5 # VASP Pb
    #fermi= 5.65354180 # VASP Ta
    #fermi= 7.72024255 # VASP W
    #fermi=10.96516405 # VASP Pt
    #fermi=-2.63611399 # VASP PbSnH
    #fermi=-2.345       # VASP PbSn
    fermi=-3.0         # VASP 2H-FeCl2 U=2.0
    #fermi=0.95         # VASP SL-MnBi2Te4 U=4.0
    #fermi=6.44577045         # VASP Mn3Sn
    #fermi=6.65112400         # VASP Mn3SnH
    #fermi=-0.09570791         # VASP Mn3Sn_2layer
    #fermi=-0.32817881         # VASP Mn3Sn_2layer_Vsn
    #fermi=-0.27313134         # VASP Mn3Sn_2layer_Vmn
    #fermi=9.35224895           # VASP FePt
    #fermi=9.04361448           # VASP FePtH
    #fermi=-0.20789123834373*27.2113845  # OPENMX FePt
    #fermi=-0.20789123834373*27.2113845  # OPENMX FePtH
    #fermi=DOSCAR(mainDir+"DOSCAR").fermi   # Fermi from DOSCAR
    #fermi=0            # model hamiltonian

    fermiAxis=np.linspace(fermi-1.0,fermi+1.0,21)
    #fermiAxis=np.array([fermi,])
    #omegaAxis=np.arange(0.1,3.05,1.0)
    omegaAxis=[0]
    model.setFermi(fermi)
    
    '''
    # check band intepolation
    spin_proj, orbital_proj= True, True
    calc=BZLineTask(model, solveBandStructure, KPOINTS=mainDir+"/KPOINTS", ncores=8,
                   args={"spin proj":spin_proj, "orbital proj":orbital_proj, "degen":1e-2},
                   )
    energies=np.zeros((calc.nkpt,model.norbs),dtype=float)
    if spin_proj: spin_eig=np.zeros((calc.nkpt,3,model.norbs),dtype=float)
    if orbital_proj: orbital_eig = np.zeros((calc.nkpt,3,model.norbs),dtype=float)
    for result in calc.results:
        energies[result['indices']]   =result['energies']
        if spin_proj: spin_eig[result['indices']]   =np.diagonal(result['spin_eig'],axis1=-2,axis2=-1)
        if orbital_proj: orbital_eig[result['indices']]=np.diagonal(result['orbital_eig'],axis1=-2,axis2=-1)
    with open(mainDir+"./tbbands.txt","w") as fout:
        fout.write("#ikpt Energy(eV) ")
        if spin_proj: fout.write("<Sx> <Sy> <Sz> ")
        if orbital_proj: fout.write("<Lx> <Ly> <Lz> ")
        fout.write("\n")
        for iband in range(model.norbs):
            for ik, energy in enumerate(energies[:,iband]):
                fout.write(f"{ik:5d}{energy:15.6E}")
                if spin_proj: fout.write(f"{spin_eig[ik,0,iband]:15.6E}{spin_eig[ik,1,iband]:15.6E}{spin_eig[ik,2,iband]:15.6E}")
                if orbital_proj: fout.write(f"{orbital_eig[ik,0,iband]:15.6E}{orbital_eig[ik,1,iband]:15.6E}{orbital_eig[ik,2,iband]:15.6E}")
                fout.write("\n")
            fout.write("\n")
    exit()#'''
    '''
    # check surface spectra
    Tmatrix=[
        [ 1, 0, 0], # The A1 of the supercell in the frame of primitive cell
        [ 1, 2, 0], # The A2 of the supercell in the frame of primitive cell
        [ 0, 0, 1]  # The A3 of the supercell in the frame of primitive cell
    ]
    surfaceDir=0 # select the lattice vector normal to surface (from the transformed cell via Tmatrix)
    model.make_supercell(Tmatrix, omitRHam=True)
    model.shiftOrigin(surfaceDir=surfaceDir, omitRHam=True)
    model.constructPrincipleLayer(surfaceDir=surfaceDir, omitRHam=True)
    calc=BZLineTask(model, solveSurfaceSpectraTask, KPOINTS=mainDir+"/KPOINTS.rect", ncores=8, 
                    args={"fermiAxis": fermiAxis, "n_iter": 10, "cutoff":0.01, "eta":0.01, "surfaceDir":surfaceDir,
                            },)
    spectral=np.zeros((3,calc.nkpt,len(fermiAxis)),dtype=float) # ikind, ik, iE
    for result in calc.results: 
        spectral[0,result['indices']]=result['A_bulk']    # bulk
        spectral[1,result['indices']]=result['A_surf_L']  # left surface
        spectral[2,result['indices']]=result['A_surf_R']  # right surface
    with open(f"{mainDir}/SurfaceSpectra.txt","w") as fout:
        fout.write("#ik Energy Amplitude_bulk Amplitude_surf_L Amplitude_surf_R\n")
        for ik in range(spectral.shape[1]):
            for ienergy, energy in enumerate(fermiAxis-model.fermi):
                fout.write(f"{ik:5d}{energy:15.6E}{spectral[0,ik,ienergy]:15.6E}{spectral[1,ik,ienergy]:15.6E}{spectral[2,ik,ienergy]:15.6E}\n")
    exit()#'''
    '''
    # check the topological integer
    #calc=Chern_FHS(model=model, kgrid=(51,51,1), verbose=2, )
    calc=Z2_FHS(model=model, kgrid=(51,51,1), verbose=2, )
    #exit()#'''

    BZGrid=(30,30,1)
    BZSymm=False
    recordOmegaIndexList=[]#[6]
    recordFermiIndexList=[]#[0]
    BZDistributionRepeat=(1,1,0)
    ObservableType="SHC" # "IFE" or "IFT"; must match the selected response calculator
    calc=BZGridTask(model, solveStGC_LG_DM,
                    args={"fermi":fermiAxis, 
                          "projected shells for applying current":[
                              #{"source":bulkRegion, "target":bulkRegion},
                           ],
                          'projected shells for detecting spincurrent':[],
                          'projected shells for detecting spin accumulation':[
                              #{"source":Cl1, "target":Cl1},
                          ],
                          'omega':omegaAxis,
                          'record frequency ids':recordOmegaIndexList,
                          'record fermi ids':recordFermiIndexList,

                          "mag type": "spin", # Type of Inversed Faraday effect, can be spin | orbital | total
                          "full covariant derivative": True, # include full covariant formalism for LG 2nd order density-matrix

                          "finiteTemperatureOccupations": True, "kBT":0.026, "eta":0.06582, # ~10 fs
                          "collisionSpecification": CollisionSpecification(mode="crta", constantLinewidth=0.06582), # collision model, currently only for DM based calculators
                          "degen":1e-1,
                    }, 
                    BZGrid=BZGrid,BZSymm=BZSymm,
                    mode='run',
                    ncores=8)
    #exit()
    #_,_,_,nbands=calc.results[0]["energies"].shape
    #pdos=np.zeros((101,nbands,))
    #energies=np.zeros(tuple(BZGrid)+(nbands,))
    responseFunction = None
    projected_sigma=0
    sigma=0
    for result_q in calc.results:
        #pdos+=result_q['PDOS']
        #energies[result_q['iq'][0]::calc.QGrid[0],result_q['iq'][1]::calc.QGrid[1],result_q['iq'][2]::calc.QGrid[2]]=result_q['energies']
        if False and 'projected response' in result_q and np.prod(result_q['projected response'].shape[6:8]) > 0:
            if responseFunction is None:
                local_response = result_q['projected response']
                responseFunction = np.zeros(local_response.shape[:3] + tuple(BZGrid) + local_response.shape[6:],
                                            dtype=local_response.dtype)
            responseFunction[:, :, :,
                             result_q['iq'][0]::calc.QGrid[0],
                             result_q['iq'][1]::calc.QGrid[1],
                             result_q['iq'][2]::calc.QGrid[2],
                             :, :, :, :, :] = result_q['projected response']
        #continue
        if 'projected sigma' in result_q: projected_sigma += result_q['projected sigma']
        if not BZSymm:  # symmetrization on the BZ is only implemented for the whole tensor
            sigma += result_q['sigma']
        #else:
        #    sigma += model.pointGroup.symmetrize_tensor(result_q['sigma'],Iodd=SpinHallConductance.Iodd,TRodd=SpinHallConductance.Todd,rank=3)
    '''
    pdos/=len(irreducibleTask)
    with open("%s/pdos.txt"%mainDir,"w") as fout:
        fout.write("Energy(eV) ")
        for atom in model.atomicOrbInfoList:
            for L, mList in zip(atom['L'],atom['orbital index']):
                for im, m in enumerate(mList): fout.write(f"{atom['label']}:{model.atomicOrbNameList[L][im]} ")
        fout.write("\n")
        for energy_intepolate, pdos_list in zip(fermiAxis,pdos):
            fout.write(f"{energy_intepolate-fermi:15.6E}")
            for atom in model.atomicOrbInfoList:
                for mList in atom['orbital index']:
                    for m in mList: fout.write(f"{pdos_list[m]:15.6E}")
            fout.write("\n")
    exit()'''
    
    '''
    # output (possibly symmetrized) 3-rank conductance
    sigma/=len(calc.results)
    with open("%s/spinHallConductance.txt"%mainDir,"w") as fout:
        fout.write("Fermi(eV)")
        directionName=['x','y','z']
        for alpha in range(3):
            for beta in range(3):
                for gamma in range(3):
                    fout.write(f"            {directionName[alpha]}{directionName[beta]}{directionName[gamma]}")
        fout.write("\n")
        for f, s in zip(fermiAxis,sigma):
            fout.write(f"{f-fermi:9.6f}")
            for alpha in range(3):
                for beta in range(3):
                    for gamma in range(3):
                        fout.write(f"{s[alpha,beta,gamma]:15.6E}")
            fout.write("\n")
    exit()#'''

    #'''
    # output projected 2-rank conductance
    projected_sigma/=len(calc.results)
    with open("%s/secondOrderTensor.txt"%mainDir,"w") as fout:
        fout.write("Fermi(eV)")
        directionName=['x','y','z']
        for iproj1 in range(projected_sigma.shape[0]):
            for iproj2 in range(projected_sigma.shape[1]):
                for alpha in range(3):
                    for gamma in range(3):
                        fout.write(f"            {directionName[alpha]}{directionName[gamma]}")
        fout.write("\n")
        for ienergy, fermiEnergy in enumerate(fermiAxis):
            fout.write(f"{fermiEnergy-fermi:9.6f}")
            for iproj1 in range(projected_sigma.shape[0]):
                for iproj2 in range(projected_sigma.shape[1]):
                    for alpha in range(3):
                        for gamma in range(3):
                            fout.write(f"{projected_sigma[iproj1,iproj2,ienergy,alpha,gamma]:15.6E}")
            fout.write("\n")
    exit()#'''

    '''
    # output projected 3-ranked, frequency-dependent complex conductance
    projected_sigma_average = projected_sigma/len(calc.results)
    for iproj1 in range(projected_sigma_average.shape[0]):
        for iproj2 in range(projected_sigma_average.shape[1]):
            for iproj3 in range(projected_sigma_average.shape[2]):
                output3rdFreqDenpendentData(projected_sigma_average[iproj1,iproj2,iproj3].real, fermi=fermi, fermiAxis=fermiAxis, omegaAxis=omegaAxis,
                                            filePath=f"{mainDir}/{ObservableType}tensor.proj{iproj1}-{iproj2}-{iproj3}.real.txt")
                output3rdFreqDenpendentData(projected_sigma_average[iproj1,iproj2,iproj3].imag, fermi=fermi, fermiAxis=fermiAxis, omegaAxis=omegaAxis,
                                            filePath=f"{mainDir}/{ObservableType}tensor.proj{iproj1}-{iproj2}-{iproj3}.imag.txt")
                if ObservableType.upper() == "IFE":
                    outputIFEcoeffs(projected_sigma_average[iproj1,iproj2,iproj3], 
                                            fermi=fermi, fermiAxis=fermiAxis, omegaAxis=omegaAxis,
                                            filePath=f"{mainDir}/IFEcoeffs.proj{iproj1}-{iproj2}-{iproj3}.txt")
                elif ObservableType.upper() == "IFT":
                    outputIFTcoeffs(projected_sigma_average[iproj1,iproj2,iproj3], 
                                    fermi=fermi, fermiAxis=fermiAxis, omegaAxis=omegaAxis,
                                    theta=magnetizationTheta, phi=magnetizationPhi,
                                    filePath=f"{mainDir}/IFTcoeffs.proj{iproj1}-{iproj2}-{iproj3}.txt")
                else:
                    print("Observable must be 'IFE' or 'IFT'")

    if responseFunction is not None:
        for iproj1 in range(responseFunction.shape[0]):
            for iproj2 in range(responseFunction.shape[1]):
                for iproj3 in range(responseFunction.shape[2]):
                    for irecord_fermi, ifermi in enumerate(recordFermiIndexList):
                        for irecord_omega, iomega in enumerate(recordOmegaIndexList):
                            responseOnBZ = responseFunction[
                                iproj1, iproj2, iproj3, :, :, :, irecord_fermi, irecord_omega, :, :, :]
                            fileStem = (f"{mainDir}/BZDistribution.{ObservableType}."
                                        f"proj{iproj1}-{iproj2}-{iproj3}.fermi{ifermi}.omega{iomega}")
                            figureTitle = (f"EF shift = {fermiAxis[ifermi]-fermi:.4f} eV, "
                                           f"photon energy = {omegaAxis[iomega]:.4f} eV")
                            outputBZDistributionData(responseOnBZ, 
                                                     filePath=f"{mainDir}/BZDistribution.{ObservableType}tensor."
                                                     f"proj{iproj1}-{iproj2}-{iproj3}.fermi{ifermi}.omega{iomega}.txt",
                                                     reciLattice=model.reciLattice, bzRepeat=BZDistributionRepeat)
                            if ObservableType.upper() == "IFE":
                                outputIFEResponseOnBZ(responseOnBZ, filePath=f"{fileStem}.txt",
                                                     reciLattice=model.reciLattice,
                                                     bzRepeat=BZDistributionRepeat, figureTitle=figureTitle)
                            elif ObservableType.upper() == "IFT":
                                outputIFTResponseOnBZ(responseOnBZ, reciLattice=model.reciLattice,
                                                     theta=magnetizationTheta, phi=magnetizationPhi, 
                                                     filePath=f"{fileStem}.txt",
                                                     bzRepeat=BZDistributionRepeat, figureTitle=figureTitle)
                            else:
                                raise ValueError("ObservableType must be 'IFE' or 'IFT'")
    
    exit()#'''  

if __name__ == '__main__': # crucial for multiprocessing in Windows
    freeze_support()
    run()
