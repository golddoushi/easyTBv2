import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from multiprocessing import freeze_support

import paramReader
paramReader.readGlobalConst()
import numpy as np

from task import BZGridTask
from system import System_omx, System_w90, System_tb
from calculator import Velocity, Velocity_diag, Energy, SpinBerryCurv, PDOS, VelocityOnPath, BerryCurvOnPath, SpinBerryCurvOnPath
from calculator import AnormalousHallConductance
from calculator import SpinHallConductance, SpinHallConductance_diag, OrbitalHallConductance
from calculator import SpinGalvanicConductance, SpinGalvanicConductance_diag, OrbitalGalvanicConductance, OrbitalGalvanicConductance_diag
from calculator import SpinTorque, SpintorqueGalvanicConductance
from calculator import EnergyOnPath, SpinExpectationOnPath, SpinOrbitalExpectationOnPath
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
    calc=AnormalousHallConductance(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],eta=0.026,fermiList=param['fermi'],skip_low=False)
    return {"energies":calc.energies,
            #"velocities":calc.velocity_K,
            "Berry curv":calc.BerryCurv,
            "sigma":calc.sigma,"iq":param["iq"]}

def solveSGC(param):
    calc=SpinGalvanicConductance(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],eta=0.06582,degen=1e-6,skip_low=True,
                             fermiList=param['fermi'],
                             spinProjectionList=param['projected shells for detecting spin accumulation'],
                             currentProjectionList=param['projected shells for applying current'],
                             )
    return {"energies":calc.energies,
            "spin galvanic response":calc.spinVelocityCorrelation,
            "projected sigma":calc.projected_sigma,"sigma":calc.projected_sigma[0,0],
            "iq":param["iq"]}

def solveSGC_diag(param):
    calc=SpinGalvanicConductance_diag(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],kBT=0.026,eta=0.06582,degen=1e-6,skip_low=True,
                             fermiList=param['fermi'],
                             spinProjectionList=param['projected shells for detecting spin accumulation'],
                             currentProjectionList=param['projected shells for applying current'],
                             )
    return {"energies":calc.energies,
            "spin galvanic response":calc.spinVelocityCorrelation_diag,
            "projected sigma":calc.projected_sigma,"sigma":calc.projected_sigma[0,0],
            "iq":param["iq"]}

def solveOGC(param):
    calc=OrbitalGalvanicConductance(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],eta=0.06582,degen=1e-6,skip_low=True,
                             fermiList=param['fermi'],
                             orbitalProjectionList=param['projected shells for detecting spin accumulation'],
                             currentProjectionList=param['projected shells for applying current'],
                             )
    return {"energies":calc.energies,
            "orbital galvanic response":calc.orbitalVelocityCorrelation,
            "projected sigma":calc.projected_sigma,"sigma":calc.projected_sigma[0,0],
            "iq":param["iq"]}

def solveOGC_diag(param):
    calc=OrbitalGalvanicConductance_diag(model=param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],kBT=0.026,eta=0.06582,degen=1e-6,skip_low=True,
                             fermiList=param['fermi'],
                             orbitalProjectionList=param['projected shells for detecting spin accumulation'],
                             currentProjectionList=param['projected shells for applying current'],
                             )
    return {"energies":calc.energies,
            "orbital galvanic response":calc.orbitalVelocityCorrelation_diag,
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

def run():
    # Pbene, spin hall conductance
    mainDir="./mcalc/BN/"
    model=System_tb(
        mainDir=mainDir,
        hamR="BNsoc",
        #hamR="omx",
        #hamR="BNsoc",
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
            "Te":(1,),
            #"Fe":(0,0,1,1,2,2,),
            "Cl":(1,),
            #"Cl":(0,0,1,1,2,3),
            "H":(0,),
            #"H":(0,0,1,),
            "Au":(0,0,0,1,1,2,2,),
        },
        ispin=2
    )
    
    atomicMagMomList=[
        #np.zeros(3),
        #np.array([0, 0, 1]),
        #np.array([0, 0, 1]),
        #np.zeros(3),
        #np.zeros(3),
        #np.zeros(3),
        np.zeros(3),
    ]
    #model.symmerizeHamAndOverlap(
    #    atomicMagMomList=atomicMagMomList,spinor=True,print_row=(1,2,3,),print_col=(1,2,3,),
    #)
    #exit()
    #'''
    # create supercell model and cut into nanoribbon
    Tmatrix=[
        [ 1, 0, 0], # The A1 of the supercell in the frame of primitive cell
        [10,20, 0], # The A2 of the supercell in the frame of primitive cell
        [ 0, 0, 1]  # The A3 of the supercell in the frame of primitive cell
    ]
    model.make_supercell(Tmatrix)
    #'''
    widthDirection=1
    model.cutEdge(cutDir=widthDirection,)
    # define the projections for bulk and edge regions, compatible with the format of projection in following
    positionIndex=np.argsort(np.array(model.posinfo['pos ions (frac.)'])[:,widthDirection])
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
    #fermi=-2.58287444 # VASP Pb
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
    
    fermiAxis=np.linspace(fermi-1.5,fermi+1.5,101)
    model.setFermi(fermi)
    '''
    # check band intepolation
    calc=SpinOrbitalExpectationOnPath(model=model, KPOINTS=mainDir+"/KPOINTS.rect", verbose=2, )
    with open(mainDir+"./tbbands.txt","w") as fout:
        fout.write("#ikpt Energy(eV) <Sx> <Sy> <Sz> <Lx> <Ly> <Lz>\n")
        for iband in range(model.norbs):
            for ik, (energy,spin,orbital) in enumerate(zip(calc.energies[:,iband],calc.spin_eig[:,:,iband,iband].real,calc.orbital_eig[:,:,iband,iband].real)):
                fout.write(f"{ik:5d}{energy:15.6E}{spin[0]:15.6E}{spin[1]:15.6E}{spin[2]:15.6E}{orbital[0]:15.6E}{orbital[1]:15.6E}{orbital[2]:15.6E}\n")
            fout.write("\n")
    exit()#'''
    BZGrid=(600,1,1)
    BZSymm=False
    calc=BZGridTask(model, solveStGC, 
                    args={"fermi":fermiAxis,
                          "projected shells for applying current":[
                              {"source":bulkRegion, "target":bulkRegion},
                           ],
                          'projected shells for detecting spincurrent':[],
                          'projected shells for detecting spin accumulation':[
                              #{"source":edgeRegion1, "target":edgeRegion1},
                              #{"source":edgeRegion2, "target":edgeRegion2},
                              {"source":block, "target":block} for block in blockRegionList
                          ],
                    }, 
                    BZGrid=BZGrid,BZSymm=BZSymm,
                    mode='run',
                    ncores=8)

    _,_,_,nbands=calc.results[0]["energies"].shape
    #pdos=np.zeros((101,nbands,))
    #energies=np.zeros(tuple(BZGrid)+(nbands,))
    #velocity=np.zeros(tuple(BZGrid)+(3,nbands,))
    #spinVelocity=np.zeros(tuple(BZGrid)+(3,3,nbands,))
    #berryCurv=np.zeros(tuple(BZGrid)+(3,nbands,))
    #spinBerryCurv=np.zeros(BZGrid+(3,3,3))
    #spinBerryCurvOnFermi=np.zeros((BZGrid[0],BZGrid[1],len(fermiAxis),3,3,3))
    #spinRashiba=np.zeros((len(fermiAxis),)+tuple(BZGrid)+(3,3,3,))
    
    #spinTorque=np.zeros(BZGrid+(3,3),dtype=complex)
    projected_sigma=0
    sigma=0
    for result_q in calc.results:
        #pdos+=result_q['PDOS']
        #energies[result_q['iq'][0]::calc.QGrid[0],result_q['iq'][1]::calc.QGrid[1],result_q['iq'][2]::calc.QGrid[2]]=result_q['energies']
        #velocity[result_q['iq'][0]::calc.QGrid[0],result_q['iq'][1]::calc.QGrid[1],result_q['iq'][2]::calc.QGrid[2]]=np.diagonal(result_q['velocities'],axis1=-2,axis2=-1)
        #velocity[result_q['iq'][0]::calc.QGrid[0],result_q['iq'][1]::calc.QGrid[1],result_q['iq'][2]::calc.QGrid[2]]=result_q['velocities'] # obtained by Velocity_diag
        #spinVelocity[result_q['iq'][0]::calc.QGrid[0],result_q['iq'][1]::calc.QGrid[1],result_q['iq'][2]::calc.QGrid[2]]=result_q['spin velocities'] # obtained by spinVelocity_diag
        #spinGalvanic[:,result_q['iq'][0]::calc.QGrid[0],result_q['iq'][1]::calc.QGrid[1],result_q['iq'][2]::calc.QGrid[2],:,:]=result_q['spin galvanic response'] # obtained by spinVelocity_diag
        #spinRashiba[:,result_q['iq'][0]::calc.QGrid[0],result_q['iq'][1]::calc.QGrid[1],result_q['iq'][2]::calc.QGrid[2],:,:]=result_q['spin Rashiba response'] # obtained by SpinRashibaConductance
        #berryCurv[result_q['iq'][0]::calc.QGrid[0],result_q['iq'][1]::calc.QGrid[1],result_q['iq'][2]::calc.QGrid[2]]=result_q['Berry curv']
        #valence=result_q['energies']<=fermi
        #f_E=(1/(result_q['energies']-fermi+0.026j)).imag#-1/0.026*exp/(1+exp)**2
        #_sBC=np.sum(result_q["spin Berry curv"]*valence[...,None,None,None,:],axis=(-1,))
        #spinBerryCurv[result_q['iq'][0]::calc.QGrid[0],result_q['iq'][1]::calc.QGrid[1],result_q['iq'][2]::calc.QGrid[2],:,:,:]=_sBC
        #for ifermi, f in enumerate(fermiAxis):
        #    valence=result_q['energies']<=f
        #    _sBC=np.sum(result_q["spin Berry curv"]*valence[:,:,:,None,None,None,:],axis=(2,-1,))
        #    spinBerryCurvOnFermi[result_q['iq'][0]::calc.QGrid[0],result_q['iq'][1]::calc.QGrid[1],ifermi,:,:,:] += _sBC
        #_sTorque=np.sum(result_q["spin torque"]*valence[...,None,None,:],axis=(-1,))
        #spinTorque[result_q['iq'][0]::calc.QGrid[0],result_q['iq'][1]::calc.QGrid[1],result_q['iq'][2]::calc.QGrid[2]]=_sTorque
        #spinTorque[result_q['iq'][0]::calc.QGrid[0],result_q['iq'][1]::calc.QGrid[1],result_q['iq'][2]::calc.QGrid[2]]=result_q['spin torque']
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
    # output (possibly symmetrized) 2-rank conductance
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

    kcoor=np.array([[[model.reciLattice@(np.array([ik1,ik2,ik3])/BZGrid) for ik3 in range(BZGrid[2])] for ik2 in range(BZGrid[1])] for ik1 in range(BZGrid[0])])
    '''
    #print(f"{energies[:,:,0,27].mean()=}{energies[:,:,0,27].max()=}{energies[:,:,0,27].min()=}")
    with open("%s/energies.txt"%mainDir,"w") as fout:
        for ik1 in range(BZGrid[0]):
            for ik2 in range(BZGrid[1]):
                for ik3 in range(BZGrid[2]):
                    kx,ky,kz=kcoor[ik1,ik2,ik3]
                    fout.write(f"{kx:15.6E}{ky:15.6E}{kz:15.6E}")
                    for ib in range(model.norbs): fout.write(f"{energies[ik1,ik2,ik3,ib]:15.6E}")
                    fout.write("\n")
    with open("%s/vx.txt"%mainDir,"w") as fout:
        for ik1 in range(BZGrid[0]):
            for ik2 in range(BZGrid[1]):
                for ik3 in range(BZGrid[2]):
                    kx,ky,kz=kcoor[ik1,ik2,ik3]
                    fout.write(f"{kx:15.6E}{ky:15.6E}{kz:15.6E}")
                    for ib in range(nbands): fout.write(f"{velocity[ik1,ik2,ik3,0,ib]:15.6E}")
                    fout.write("\n")
    with open("%s/vy.txt"%mainDir,"w") as fout:
        for ik1 in range(BZGrid[0]):
            for ik2 in range(BZGrid[1]):
                for ik3 in range(BZGrid[2]):
                    kx,ky,kz=kcoor[ik1,ik2,ik3]
                    fout.write(f"{kx:15.6E}{ky:15.6E}{kz:15.6E}")
                    for ib in range(nbands): fout.write(f"{velocity[ik1,ik2,ik3,1,ib]:15.6E}")
                    fout.write("\n")
    exit()#'''
    #'''
    import matplotlib.pyplot as plt
    #fig=plt.figure()
    #ax1=fig.add_subplot(1,3,1)
    #ax1.pcolormesh(kcoor[:,:,0,0],kcoor[:,:,0,1],spinBerryCurv[:,:,0,0,1,2],cmap="Spectral")
    #ax1.axis('equal')

    #ax2=fig.add_subplot(1,3,2)
    #ax2.pcolormesh(kcoor[:,:,0,0],kcoor[:,:,0,1],energies[:,:,0,-5],cmap="Spectral")
    #ax2.axis('equal')

    #ax3=fig.add_subplot(1,3,3)
    #ax3.pcolormesh(kcoor[:,:,0,0],kcoor[:,:,0,1],energies[:,:,0,-4],cmap="Spectral")
    #ax3.axis('equal')

    data_grid=spinTorque[:,:,0,0,2].real
    magnitude=max([abs(data_grid.max()),abs(data_grid.min())])
    print(np.sum(data_grid))
    for G in [(0,0,0,),(-1,0,0,),(0,-1,0,),(-1,-1,0,),]:
        _kcoor=kcoor+model.reciLattice@np.array(G)
        plt.pcolormesh(_kcoor[:,:,0,0],_kcoor[:,:,0,1],data_grid,vmax=magnitude,vmin=-magnitude,cmap="Spectral")
    plt.colorbar()
    plt.axis("equal")
    plt.savefig(f"{mainDir}/BZ_distr.png",dpi=300)
    plt.show()
    #'''

if __name__ == '__main__': # crucial for multiprocessing in Windows
    freeze_support()
    run()
