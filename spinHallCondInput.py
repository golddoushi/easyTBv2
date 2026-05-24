import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from multiprocessing import freeze_support

import paramReader
paramReader.readGlobalConst()
import numpy as np

from task import BZGridTask, BZLineTask
from system import System_omx, System_w90, System_tb
from calculator import Velocity, Velocity_diag, Energy, SpinBerryCurv, PDOS, VelocityOnPath, BerryCurvOnPath, SpinBerryCurvOnPath
from calculator import AnormalousHallConductance, BulkPhotovoltaicConductance, BulkPhotovoltaicConductance_VG_Full, BulkPhotovoltaicConductance_LG_Shift, BulkPhotovoltaicConductance_LG_Injection, BulkPhotovoltaicConductance_TDSE
from calculator import SpinHallConductance, SpinHallConductance_diag, OrbitalHallConductance
from calculator import SpinGalvanicConductance, SpinGalvanicConductance_diag, OrbitalGalvanicConductance, OrbitalGalvanicConductance_diag
from calculator import SpinTorque, SpintorqueGalvanicConductance
from calculator import EnergyOnPath, SpinExpectationOnPath, SpinOrbitalExpectationOnPath, SurfaceBulkSpectraOnPath
from calculator import Chern_FHS, Z2_FHS
from auxiliary import showElapsedTime, getCBM, getVBM
from fileio import writeVolumeData

def solveSHC(param):
    calc=SpinHallConductance(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],eta=0.06582,degen=1e-6,skip_low=True,
                             fermiList=param['fermi'],
                             currentProjectionList    =param['projected shells for applying current'],
                             spincurrentProjectionList=param['projected shells for detecting spincurrent'],
                             )
    return {"energies":calc.energies,
            #"velocities":calc.velocity_K,
            "spin Berry curv":calc.spinBerryCurv,
            "projected sigma":calc.projected_sigma,"sigma":calc.projected_sigma[0,0],
            "iq":param["iq"]}

def solveSHC_diag(param):
    calc=SpinHallConductance_diag(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],kBT=0.026,eta=0.06582,degen=1e-6,skip_low=True,
                             fermiList=param['fermi'],
                             currentProjectionList    =param['projected shells for applying current'],
                             spincurrentProjectionList=param['projected shells for detecting spincurrent'],
                             )
    return {"energies":calc.energies,
            #"velocities":calc.velocity_K,
            "spin Berry curv":calc.spinvelocityVelocityCorrelation_diag,
            "projected sigma":calc.projected_sigma,"sigma":calc.projected_sigma[0,0],
            "iq":param["iq"]}

def solveOHC(param):
    calc=OrbitalHallConductance(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],eta=0.06582,degen=1e-6,skip_low=True,
                             fermiList=param['fermi'],
                             currentProjectionList    =param['projected shells for applying current'],
                             orbitalcurrentProjectionList=param['projected shells for detecting spincurrent'],
                             )
    return {"energies":calc.energies,
            #"velocities":calc.velocity_K,
            "orbital Berry curv":calc.orbitalBerryCurv,
            "sigma":calc.sigma,"iq":param["iq"]}

def solveStGC(param):
    calc=SpintorqueGalvanicConductance(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],eta=0.06582,degen=1e-6,skip_low=True,
                             fermiList=param['fermi'],
                             spinTorqueProjectionList=param['projected shells for detecting spin accumulation'],
                             currentProjectionList  =param['projected shells for applying current'],
                             )
    return {"energies":calc.energies,
            "projected sigma":calc.projected_sigma,"sigma":calc.projected_sigma[0,0],
            "iq":param["iq"]}

def solveAHC(param):
    calc=AnormalousHallConductance(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],eta=0.06582,degen=1e-6,skip_low=False,
                                   fermiList=param['fermi'],
                                   currentProjectionList    =param['projected shells for applying current'],
                                   )
    return {"energies":calc.energies,
            "projected Berry curv":calc.projected_BerryCurv,
            "projected sigma":calc.projected_sigma,"sigma":calc.projected_sigma[0,0],
            "iq":param["iq"]}

def solveBPVE_VG(param):
    calc=BulkPhotovoltaicConductance_VG_Full(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],eta=0.06582,degen=1e-6,skip_low=False,
                                   fermiList=param['fermi'],omegaList=param['omega'],
                                   responseOmegaIndexList=param['record frequency ids'],
                                   currentProjectionList    =param['projected shells for applying current'],
                                   )
    return {"energies":calc.energies,
            "projected response":calc.projected_response,
            "projected sigma":calc.projected_sigma,"sigma":calc.projected_sigma[0,0,0],
            "iq":param["iq"]}

def solveBPVE_LG_Shift(param):
    calc=BulkPhotovoltaicConductance_LG_Shift(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],eta=0.06582,degen=1e-6,skip_low=False,
                                   fermiList=param['fermi'],omegaList=param['omega'],
                                   responseOmegaIndexList=param['record frequency ids'],
                                   currentProjectionList    =param['projected shells for applying current'],
                                   )
    return {"energies":calc.energies,
            #"projected response":calc.projected_response,
            "projected sigma":calc.projected_sigma,"sigma":calc.projected_sigma[0,0,0],
            "iq":param["iq"]}

def solveBPVE_LG_Inject(param):
    calc=BulkPhotovoltaicConductance_LG_Injection(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],eta=0.06582,degen=1e-6,skip_low=False,
                                   fermiList=param['fermi'],omegaList=param['omega'],
                                   responseOmegaIndexList=param['record frequency ids'],
                                   currentProjectionList    =param['projected shells for applying current'],
                                   )
    return {"energies":calc.energies,
            #"projected response":calc.projected_response,
            "projected sigma":calc.projected_sigma,"sigma":calc.projected_sigma[0,0,0],
            "iq":param["iq"]}

def solveBPVE_TDSE(param):
    calc = BulkPhotovoltaicConductance_TDSE(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],eta=0.06582,skip_low=False,
                                   fermiList=param['fermi'],omegaList=param['omega'],
                                   A0=0.01, # Keep this in the perturbative regime
                                   )
    return {"projected sigma": calc.projected_sigma, "sigma": calc.sigma_TDSE, 
            "iq": param["iq"]}

def solveSGC(param):
    calc=SpinGalvanicConductance(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],eta=0.06582,degen=1e-6,skip_low=True,
                             fermiList=param['fermi'],
                             spinProjectionList=param['projected shells for detecting spin accumulation'],
                             currentProjectionList=param['projected shells for applying current'],
                             )
    return {"energies":calc.energies,
            "projected spin galvanic response":calc.projected_spinVelocityCorrelation,
            "projected sigma":calc.projected_sigma,"sigma":calc.projected_sigma[0,0],
            "iq":param["iq"]}

def solveSGC_diag(param):
    calc=SpinGalvanicConductance_diag(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],kBT=0.026,eta=0.06582,degen=1e-6,skip_low=True,
                             fermiList=param['fermi'],
                             spinProjectionList=param['projected shells for detecting spin accumulation'],
                             currentProjectionList=param['projected shells for applying current'],
                             )
    return {"energies":calc.energies,
            "projected spin galvanic response":calc.projected_spinVelocityCorrelation_diag,
            "projected sigma":calc.projected_sigma,"sigma":calc.projected_sigma[0,0],
            "iq":param["iq"]}

def solveOGC(param):
    calc=OrbitalGalvanicConductance(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],eta=0.06582,degen=1e-6,skip_low=True,
                             fermiList=param['fermi'],
                             orbitalProjectionList=param['projected shells for detecting spin accumulation'],
                             currentProjectionList=param['projected shells for applying current'],
                             )
    return {"energies":calc.energies,
            "projected orbital galvanic response":calc.projected_orbitalVelocityCorrelation,
            "projected sigma":calc.projected_sigma,"sigma":calc.projected_sigma[0,0],
            "iq":param["iq"]}

def solveOGC_diag(param):
    calc=OrbitalGalvanicConductance_diag(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],kBT=0.026,eta=0.06582,degen=1e-6,skip_low=True,
                             fermiList=param['fermi'],
                             orbitalProjectionList=param['projected shells for detecting spin accumulation'],
                             currentProjectionList=param['projected shells for applying current'],
                             )
    return {"energies":calc.energies,
            "projected orbital galvanic response":calc.projected_orbitalVelocityCorrelation_diag,
            "projected sigma":calc.projected_sigma,"sigma":calc.projected_sigma[0,0],
            "iq":param["iq"]}

def solvePDOS(param):
    calc=PDOS(energyAxis=param['fermi'],kgrid=param['FFT grid'],phaseshift=param['phase'],eta=0.026,model=param['model'],)
    return {"energies":calc.energies,"PDOS":calc.pdos,"iq":param["iq"]}

def solveVelocity(param):
    calc=Velocity(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],skip_low=True)
    return {"energies":calc.energies,"velocities":calc.velocity_K,"iq":param["iq"]}

def solveVelocity_diag(param):
    calc=Velocity_diag(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],skip_low=True)
    return {"energies":calc.energies,"velocities":calc.velocity_diag_K,"iq":param["iq"]}

#+--------------------------+
#| factory for BZLine tasks |
#+--------------------------+
def solveSurfaceSpectraTask(param):
    calc=SurfaceBulkSpectraOnPath(model=param['model'], kpath=param['kpath'], verbose=param['verbose'],
                                  surfaceDir=param['surfaceDir'], fermiAxis=param['fermiAxis'], n_iter=param['n_iter'])
    return {
        'indices': param['indices'],
        'A_bulk': calc.A_bulk,
        'A_surf_L': calc.A_surf_L,
        'A_surf_R': calc.A_surf_R,
    }

def solveBandStructure(param):
    calc=SpinOrbitalExpectationOnPath(model=param['model'], kpath=param['kpath'], verbose=param['verbose'])
    return {
        'indices': param['indices'],
        'energies': calc.energies,
        'spin_eig': calc.spin_eig.real,
        'orbital_eig': calc.orbital_eig.real,
    }

def run():
    # Pbene, spin hall conductance
    mainDir="./mcalc/BN/"
    model=System_tb(
        mainDir=mainDir,
        #hamR="wannier90",
        #hamR="omx",
        hamR="BN_fm_soc",
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
            "Fe":(2,),
            "Mo":(2,),
            "S":(1,),
            "Bi":(0,1,),
            "Te":(0,1,),
            #"Fe":(0,0,1,1,2,2,),
            "Cl":(1,),
            #"Cl":(0,0,1,1,2,3),
            "H":(0,),
            #"H":(0,0,1,),
            "Au":(0,0,0,1,1,2,2,),
        },
        ispin=2
    )
    # rotate the spin quantization axis to \sigma^z if SAXIS != 0 0 1
    #model.rotateSpinBasisToGlobal(alpha=np.pi/2, beta=np.pi/2) # SAXIS = 0 1 0
    #model.rotateSpinBasisToGlobal(alpha=0, beta=np.pi/2) # SAXIS = 1 0 0
    #atomicMagMomList=[
    #    np.array([0, 0, 1]),
    #]*1+[
    #    np.array([0, 0, 0]),
    #]*2
    #model.symmerizeHamAndOverlap(
    #    atomicMagMomList=atomicMagMomList,spinor=True,print_row=(0,1,9,10,),print_col=(0,1,9,10,),
    #)
    #exit()
    # modulate the SOC, exchange fields, etc
    #model.scaleSOC(lam=0)
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
    # define the projections for bulk and edge regions, compatible with the format of projection in following
    positionIndex=np.argsort(np.array(model.posinfo['pos ions (frac.)'])[:,surfaceDir])
    nEdgeAtoms=4 # include two atoms for each edge
    nBulkAtoms=int(model.posinfo['tot ions']*0.5) # include 50% central atoms
    edgeRegion1=[(positionIndex[iatom],(0,)) for iatom in range(nEdgeAtoms)]
    edgeRegion2=[(positionIndex[model.posinfo['tot ions']-1-iatom],(0,)) for iatom in range(nEdgeAtoms)]
    bulkRegion=[(positionIndex[(model.posinfo['tot ions']-nBulkAtoms)//2+iatom],(0,)) for iatom in range(nBulkAtoms)]
    print("edge region1:")
    for atom in edgeRegion1:
        pos=model.posinfo['pos ions (frac.)'][atom[0]]
        print(f"atom{atom[0]:4d} {pos[0]:8.4f}{pos[1]:8.4f}{pos[2]:8.4f}")
    print("edge region2:")
    for atom in edgeRegion2:
        pos=model.posinfo['pos ions (frac.)'][atom[0]]
        print(f"atom{atom[0]:4d} {pos[0]:8.4f}{pos[1]:8.4f}{pos[2]:8.4f}")
    print("bulk region:")
    for atom in bulkRegion:
        pos=model.posinfo['pos ions (frac.)'][atom[0]]
        print(f"atom{atom[0]:4d} {pos[0]:8.4f}{pos[1]:8.4f}{pos[2]:8.4f}")
    nBlockAtoms=4
    blockRegionList=[]
    for iblock in range(model.posinfo['tot ions']//nBlockAtoms):
        blockRegionList.append([(positionIndex[iblock*nBlockAtoms+iatom],(0,)) for iatom in range(nBlockAtoms)])
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
    #fermi=-2.1         # VASP 2H-FeCl2
    #fermi=6.44577045         # VASP Mn3Sn
    #fermi=6.65112400         # VASP Mn3SnH
    #fermi=-0.09570791         # VASP Mn3Sn_2layer
    #fermi=-0.32817881         # VASP Mn3Sn_2layer_Vsn
    #fermi=-0.27313134         # VASP Mn3Sn_2layer_Vmn
    #fermi=9.35224895           # VASP FePt
    #fermi=9.04361448           # VASP FePtH
    #fermi=-0.20789123834373*27.2113845  # OPENMX FePt
    #fermi=-0.20789123834373*27.2113845  # OPENMX FePtH
    fermi=0            # model hamiltonian
    
    #fermiAxis=np.linspace(fermi-1.5,fermi+1.5,101)
    fermiAxis=np.array([fermi,])
    omegaAxis=np.linspace(0.5,4.0,51)
    model.setFermi(fermi)
    
    '''
    # check band intepolation
    calc=BZLineTask(model, solveBandStructure, KPOINTS=mainDir+"/KPOINTS", ncores=8,)
    energies,spin_eig,orbital_eig=np.zeros((calc.nkpt,model.norbs),dtype=float),np.zeros((calc.nkpt,3,model.norbs),dtype=float),np.zeros((calc.nkpt,3,model.norbs),dtype=float)
    for result in calc.results:
        energies[result['indices']]   =result['energies']
        spin_eig[result['indices']]   =np.diagonal(result['spin_eig'],axis1=-2,axis2=-1)
        orbital_eig[result['indices']]=np.diagonal(result['orbital_eig'],axis1=-2,axis2=-1)
    with open(mainDir+"./tbbands.txt","w") as fout:
        fout.write("#ikpt Energy(eV) <Sx> <Sy> <Sz> <Lx> <Ly> <Lz>\n")
        for iband in range(model.norbs):
            for ik, (energy,spin,orbital) in enumerate(zip(energies[:,iband],spin_eig[:,:,iband],orbital_eig[:,:,iband])):
                fout.write(f"{ik:5d}{energy:15.6E}{spin[0]:15.6E}{spin[1]:15.6E}{spin[2]:15.6E}{orbital[0]:15.6E}{orbital[1]:15.6E}{orbital[2]:15.6E}\n")
            fout.write("\n")
    exit()#'''
    '''
    # check surface spectra
    Tmatrix=[
        [ 1, 0, 0], # The A1 of the supercell in the frame of primitive cell
        [ 1, 2, 0], # The A2 of the supercell in the frame of primitive cell
        [ 0, 0, 1]  # The A3 of the supercell in the frame of primitive cell
    ]
    model.make_supercell(Tmatrix)
    surfaceDir=0
    model.shiftOrigin(surfaceDir=surfaceDir)
    model.constructPrincipleLayer(surfaceDir=surfaceDir)
    calc=BZLineTask(model, solveSurfaceSpectraTask, KPOINTS=mainDir+"/KPOINTS.rect", ncores=8,
                    args={"fermiAxis": fermiAxis, "surfaceDir": surfaceDir, "n_iter": 10},)
    spectral=np.zeros((calc.nkpt,len(fermiAxis)),dtype=float)
    for result in calc.results: 
        spectral[result['indices']]=(result['A_surf_L']+result['A_surf_R'])/2-result['A_bulk']
    with open(f"{mainDir}/SurfaceSpectra.txt","w") as fout:
        fout.write("#ik Energy Amplitude\n")
        for ik in range(spectral.shape[0]):
            for ienergy, energy in enumerate(fermiAxis):
                fout.write(f"{ik:5d}{energy:15.6E}{spectral[ik,ienergy]:15.6E}\n")
    exit()#'''
    '''
    # check the topological integer
    #calc=Chern_FHS(model=model, kgrid=(51,51,1), verbose=2, )
    calc=Z2_FHS(model=model, kgrid=(51,51,1), verbose=2, )
    #exit()#'''

    BZGrid=(48,48,1)
    BZSymm=False
    calc=BZGridTask(model, solveBPVE_VG, 
                    args={"fermi":fermiAxis, 
                          "projected shells for applying current":[
                              #{"source":bulkRegion, "target":bulkRegion},
                           ],
                          'projected shells for detecting spincurrent':[],
                          'projected shells for detecting spin accumulation':[
                              #{"source":edgeRegion1, "target":edgeRegion1},
                              #{"source":edgeRegion2, "target":edgeRegion2},
                              #{"source":block, "target":block} for block in blockRegionList
                          ],
                          'omega':omegaAxis,
                          'record frequency ids':[],
                    }, 
                    BZGrid=BZGrid,BZSymm=BZSymm,
                    mode='run',
                    ncores=8)
    #exit()
    #_,_,_,nbands=calc.results[0]["energies"].shape
    #pdos=np.zeros((101,nbands,))
    #energies=np.zeros(tuple(BZGrid)+(nbands,))
    #responseFunction = np.zeros(BZGrid+(3,3,3),dtype=float)
    projected_sigma=0
    sigma=0
    for result_q in calc.results:
        #pdos+=result_q['PDOS']
        #energies[result_q['iq'][0]::calc.QGrid[0],result_q['iq'][1]::calc.QGrid[1],result_q['iq'][2]::calc.QGrid[2]]=result_q['energies']
        #responseFunction[result_q['iq'][0]::calc.QGrid[0],result_q['iq'][1]::calc.QGrid[1],result_q['iq'][2]::calc.QGrid[2],:,:]=np.sum(result_q['spin Berry curv']*valence[...,None,None,None,:],axis=(-1,))
        #responseFunction[result_q['iq'][0]::calc.QGrid[0],result_q['iq'][1]::calc.QGrid[1],result_q['iq'][2]::calc.QGrid[2],:,:,:]=result_q['projected response'][0,0,0][:,:,:,0,0,:,:,:]
        #continue
        projected_sigma += result_q['projected sigma']
        if not BZSymm:  # symmetrization on the BZ is only implemented for the whole tensor
            sigma += result_q['sigma']
        else:
            sigma += model.pointGroup.symmetrize_tensor(result_q['sigma'],Iodd=SpinHallConductance.Iodd,TRodd=SpinHallConductance.Todd,rank=3)
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
    #exit()#'''

    '''
    # output projected 2-rank conductance
    projected_sigma/=len(calc.results)
    with open("%s/spinGalvanics.txt"%mainDir,"w") as fout:
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

    #'''
    # output projected 3-ranked, frequency-dependent complex conductance
    projected_sigma=projected_sigma.real
    projected_sigma/=len(calc.results)
    with open("%s/bulkPhotovoltaics.txt"%mainDir,"w") as fout:
        fout.write("Omega(eV) Fermi(eV) ")
        directionName=['x','y','z']
        for iproj1 in range(projected_sigma.shape[0]):
            for iproj2 in range(projected_sigma.shape[1]):
                for iproj2 in range(projected_sigma.shape[2]):
                    for alpha1 in range(3):
                        for alpha2 in range(3):
                            for beta in range(3):
                                fout.write(f"            {directionName[alpha1]}{directionName[alpha2]}{directionName[beta]}")
        fout.write("\n")
        for iomega, omega in enumerate(omegaAxis):
            fout.write(f"{omega:9.6f}")
            for ifermi, fermiEnergy in enumerate(fermiAxis):
                fout.write(f"{fermiEnergy-fermi:9.6f}")
                for iproj1 in range(projected_sigma.shape[0]):
                    for iproj2 in range(projected_sigma.shape[1]):
                        for iproj3 in range(projected_sigma.shape[2]):
                            for alpha1 in range(3):
                                for alpha2 in range(3):
                                    for beta in range(3):
                                        fout.write(f"{projected_sigma[iproj1,iproj2,iproj3,ifermi,iomega,alpha1,alpha2,beta]:15.6E}")
                fout.write("\n")
    exit()#'''

    kcoor=np.array([[[model.reciLattice@(np.array([ik1,ik2,ik3])/BZGrid) for ik3 in range(BZGrid[2])] for ik2 in range(BZGrid[1])] for ik1 in range(BZGrid[0])])
    #'''
    import matplotlib.pyplot as plt
    data_grid=responseFunction[:,:,0, 0,0,0].real
    magnitude=max([abs(data_grid.max()),abs(data_grid.min())])
    print(np.sum(data_grid))
    for G in [(0,0,0,),(-1,0,0,),(0,-1,0,),(-1,-1,0,),]:
        _kcoor=kcoor+model.reciLattice@np.array(G)
        plt.pcolormesh(_kcoor[:,:,0,0],_kcoor[:,:,0,1],data_grid,vmax=magnitude,vmin=-magnitude,cmap="Spectral")
    plt.colorbar()
    plt.axis("equal")
    plt.savefig(f"{mainDir}/BZ_distr1.png",dpi=300)
    plt.show()

    data_grid=responseFunction[:,:,0, 0,0,1].real
    magnitude=max([abs(data_grid.max()),abs(data_grid.min())])
    print(np.sum(data_grid))
    for G in [(0,0,0,),(-1,0,0,),(0,-1,0,),(-1,-1,0,),]:
        _kcoor=kcoor+model.reciLattice@np.array(G)
        plt.pcolormesh(_kcoor[:,:,0,0],_kcoor[:,:,0,1],data_grid,vmax=magnitude,vmin=-magnitude,cmap="Spectral")
    plt.colorbar()
    plt.axis("equal")
    plt.savefig(f"{mainDir}/BZ_distr2.png",dpi=300)
    plt.show()
    #'''

if __name__ == '__main__': # crucial for multiprocessing in Windows
    freeze_support()
    run()
