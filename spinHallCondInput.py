import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from multiprocessing import Pool, freeze_support

import paramReader
paramReader.readGlobalConst()
import numpy as np

from task import BZGridTask
from system import System_omx, System_w90
from calculator import Velocity, SpinHallConductance, Energy, SpinBerryCurv, AnormalousHallConductance, EnergyOnPath, VelocityOnPath, BerryCurvOnPath, SpinBerryCurvOnPath, PDOS, SpinHallConductanceRashiba
from auxiliary import showElapsedTime
from symmetry import SYMMETRY_PRECISION
from fileio import writeVolumeData

def solveQ(param):
    calc=SpinHallConductance(param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],eta=0.6582119514E-3,degen=1e-2,fermiList=param['fermi'])
    return {"energies":calc.energies,
            #"velocities":calc.velocity_K,
            "spin Berry curv":calc.spinBerryCurv,
            "sigma":calc.sigma,"iq":param["iq"]}

def solveRashiba(param):
    calc=SpinHallConductanceRashiba(param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],T=300,fermiList=param['fermi'])
    return {"energies":calc.energies,
            "spin Berry curv":calc.spinVeloVelo,
            "sigma":calc.sigma,"iq":param["iq"]}

def solveAHC(param):
    calc=AnormalousHallConductance(param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],eta=0.026,fermiList=param['fermi'])
    return {"energies":calc.energies,
            "velocities":calc.velocity_K,
            "Berry curv":calc.BerryCurv,
            "sigma":calc.sigma,"iq":param["iq"]}

def solvePDOS(param):
    calc=PDOS(param['model'],energyAxis=param['fermi'],kgrid=param['FFT grid'],phaseshift=param['phase'],eta=0.026)
    return {"energies":calc.energies,"PDOS":calc.pdos,"iq":param["iq"]}

def solveVelocity(param):
    calc=Velocity(param['model'],kgrid=param['FFT grid'],)
    return {"energies":calc.energies,"velocities":calc.velocity_K,"iq":param["iq"]}

def getVBM(energies,fermi):
    vb_condition=energies <= fermi
    low_bound = np.min(energies)
    return np.max((energies + low_bound) * vb_condition, axis=(-1,)) - low_bound

def getCBM(energies,fermi):
    cb_condition=energies > fermi
    up_bound = np.max(energies)
    return np.min((energies - up_bound) * cb_condition, axis=(-1,)) + up_bound

def run():
    # Pbene, spin hall conductance
    mainDir="./mcalc/Mn3SnH/"
    model=System_w90(
        mainDir=mainDir,
        hamR="wannier90",
        #hamR="omx",
        cutoff=-0.002,
        poscar="POSCAR",
        projections={
            "Sn":(0,1,),
            #"Mn":(0,0,0,1,1,2,),
            "Mn":(0,1,2,),
            #"Sn":(0,0,0,1,1,2,2,),
            "Pb":(0,1,),
            #"Pt":(0,1,2,),
            "Pt":(0,0,1,1,2,2,),
            "C":(0,1,),
            "Fe":(2,),
            "Cl":(1,),
            "H":(0,),
        },
        ispin=2
    )
    
    a,b=1/2,np.sqrt(3)/2
    atomicMagMomList=[
        np.array([ a, b, 0]),
        np.array([-1, 0, 0]),
        #np.array([ a,-b, 0]),
        #np.array([ a, b, 0]),
        np.array([-1, 0, 0]),
        #np.array([ a,-b, 0]),
        np.array([ a, b, 0]),
        #np.array([-1, 0, 0]),
        np.array([ a,-b, 0]),
        #np.array([ a, b, 0]),
        #np.array([-1, 0, 0]),
        np.array([ a,-b, 0]),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
        #np.zeros(3),
    ]
    model.symmerizeHamAndOverlap(
        atomicMagMomList=atomicMagMomList,spinor=True,print_row=(0,1,2,3,4,),print_col=(0,1,2,3,4,),
    )
    #'''
    #fermi=0  # OPNEMX generated wannier model always displays zero fermi energy
    #fermi=-0.141600218424*27.2113845 # OPENMX Sn
    #fermi=-0.16790565257604*27.2113845 # OPENMX Pt
    #fermi=-2.41861676 # VASP Sn
    #fermi=10.96516405 # VASP Pt
    #fermi=-2.0         # VASP 2H-FeCl2
    #fermi=6.44577045         # VASP Mn3Sn
    fermi=6.65112400         # VASP Mn3SnH
    #fermi=-0.09570791         # VASP Mn3Sn_2layer
    #fermi=-0.32817881         # VASP Mn3Sn_2layer_Vsn
    #fermi=-0.27313134         # VASP Mn3Sn_2layer_Vmn
    
    fermiAxis=np.linspace(fermi-0.5,fermi+0.5,101)
    model.setFermi(fermi)
    '''
    # Bands
    calc=SpinBerryCurvOnPath(model, mainDir+"/KPOINTS", verbose=2)
    spinBerry=-(1/2*calc.spinBerryCurv[:,0,2,2,:] - np.sqrt(3)/2*calc.spinBerryCurv[:,1,2,2,:])/100
    with open(mainDir+"./tbbands.txt","w") as fout:
        fout.write("#ikpt Energy(eV) Vx Vy Vz sBC\n")
        for iband in range(model.norbs):
            for ik, (energy, vx_k, vy_k, vz_k, sBC,) in enumerate(zip(
                    calc.energies[:,iband],
                    calc.velocity_K[:,0,iband,iband],calc.velocity_K[:,1,iband,iband],calc.velocity_K[:,2,iband,iband],
                    spinBerry[:,iband],
                    )):
                fout.write(f"{ik:5d}{energy:15.6E}{vx_k.real:15.6E}{vy_k.real:15.6E}{vz_k.real:15.6E}{sBC:15.6E}\n")
            fout.write("\n")
    with open(mainDir+"./spinBerryCurv.txt","w") as fout:
        #fout.write("#ikpt sBC\n")
        for ifermi, f in enumerate(fermiAxis):
            for ik, sBC in enumerate(np.sum(spinBerry*(calc.energies<=f),axis=(-1,))):
                fout.write(f"{sBC:15.6E}")
            fout.write("\n")
    exit()
    #'''
    BZGrid=(70,70,81)
    BZSymm=False
    calc=BZGridTask(model, solveQ, args={"fermi":fermiAxis}, 
                    BZGrid=BZGrid,BZSymm=BZSymm,
                    ncores=8)

    #pdos=np.zeros((101,model.norbs,))
    #energies=np.zeros(tuple(BZGrid)+(model.norbs,))
    #velocity=np.zeros(tuple(BZGrid)+(3,model.norbs,))
    #berryCurv=np.zeros(tuple(BZGrid)+(3,model.norbs,))
    spinBerryCurv=np.zeros(BZGrid+(3,3,3))
    #spinBerryCurvOnFermi=np.zeros((BZGrid[0],BZGrid[1],len(fermiAxis),3,3,3))
    sigma=np.zeros((len(fermiAxis),3,3,3))
    for result_q in calc.results:
        #pdos+=result_q['PDOS']
        #energies[result_q['iq'][0]::calc.QGrid[0],result_q['iq'][1]::calc.QGrid[1],result_q['iq'][2]::calc.QGrid[2]]=result_q['energies']
        #velocity[result_q['iq'][0]::calc.QGrid[0],result_q['iq'][1]::calc.QGrid[1],result_q['iq'][2]::calc.QGrid[2]]=np.diagonal(result_q['velocities'],axis1=-2,axis2=-1)
        #berryCurv[result_q['iq'][0]::calc.QGrid[0],result_q['iq'][1]::calc.QGrid[1],result_q['iq'][2]::calc.QGrid[2]]=result_q['Berry curv']
        valence=result_q['energies']<=fermi
        #f_E=(1/(result_q['energies']-fermi+0.026j)).imag#-1/0.026*exp/(1+exp)**2
        _sBC=np.sum(result_q["spin Berry curv"]*valence[...,None,None,None,:],axis=(-1,))
        spinBerryCurv[result_q['iq'][0]::calc.QGrid[0],result_q['iq'][1]::calc.QGrid[1],result_q['iq'][2]::calc.QGrid[2],:,:,:]=_sBC
        #for ifermi, f in enumerate(fermiAxis):
        #    valence=result_q['energies']<=f
        #    _sBC=np.sum(result_q["spin Berry curv"]*valence[:,:,:,None,None,None,:],axis=(2,-1,))
        #    spinBerryCurvOnFermi[result_q['iq'][0]::calc.QGrid[0],result_q['iq'][1]::calc.QGrid[1],ifermi,:,:,:] += _sBC
        #continue
        if not BZSymm:
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
    
    #'''
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
    #exit()

    # quantity on BZ
    spinBerry=-(1/2*spinBerryCurv[:,:,:,0,2,2] - np.sqrt(3)/2*spinBerryCurv[:,:,:,1,2,2])/100#/1E15
    print(np.sum(spinBerry,axis=(0,1,2,)))
    writeVolumeData("%s/spinBerryCurv_xzz.vasp"%mainDir, model.reciLattice.T, (BZGrid[0],BZGrid[1],BZGrid[2]), spinBerry)
    spinBerry=-(np.sqrt(3)/2*spinBerryCurv[:,:,:,0,2,2] + 1/2*spinBerryCurv[:,:,:,1,2,2])/100#/1E15
    print(np.sum(spinBerry,axis=(0,1,2,)))
    writeVolumeData("%s/spinBerryCurv_yzz.vasp"%mainDir, model.reciLattice.T, (BZGrid[0],BZGrid[1],BZGrid[2]), spinBerry)
    spinBerry=(0.4330127*spinBerryCurv[:,:,:,0,2,0]-0.75*spinBerryCurv[:,:,:,0,2,1]+0.25*spinBerryCurv[:,:,:,1,2,0]-0.4330127*spinBerryCurv[:,:,:,1,2,1])/100#/1E15
    print(np.sum(spinBerry,axis=(0,1,2,)))
    writeVolumeData("%s/spinBerryCurv_xzy.vasp"%mainDir, model.reciLattice.T, (BZGrid[0],BZGrid[1],BZGrid[2]), spinBerry)
    spinBerry=(0.25*spinBerryCurv[:,:,:,0,2,0]+0.4330127*spinBerryCurv[:,:,:,0,2,1]+0.4330127*spinBerryCurv[:,:,:,1,2,0]+0.75*spinBerryCurv[:,:,:,1,2,1])/100#/1E15
    print(np.sum(spinBerry,axis=(0,1,2,)))
    writeVolumeData("%s/spinBerryCurv_xzx.vasp"%mainDir, model.reciLattice.T, (BZGrid[0],BZGrid[1],BZGrid[2]), spinBerry)
    #vbm=getVBM(energies, fermi)
    #cbm=getCBM(energies, fermi)
    #outputVolumeData("%s/VBM.vasp"%mainDir, model.reciLattice.T, BZGrid, vbm)
    #outputVolumeData("%s/CBM.vasp"%mainDir, model.reciLattice.T, BZGrid, cbm)
    '''
    kcoor=np.array([[[model.reciLattice@(np.array([ik1,ik2,ik3])/BZGrid) for ik3 in range(BZGrid[2])] for ik2 in range(BZGrid[1])] for ik1 in range(BZGrid[0])])
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
                    for ib in range(model.norbs): fout.write(f"{velocity[ik1,ik2,ik3,0,ib]:15.6E}")
                    fout.write("\n")
    with open("%s/vy.txt"%mainDir,"w") as fout:
        for ik1 in range(BZGrid[0]):
            for ik2 in range(BZGrid[1]):
                for ik3 in range(BZGrid[2]):
                    kx,ky,kz=kcoor[ik1,ik2,ik3]
                    fout.write(f"{kx:15.6E}{ky:15.6E}{kz:15.6E}")
                    for ib in range(model.norbs): fout.write(f"{velocity[ik1,ik2,ik3,1,ib]:15.6E}")
                    fout.write("\n")
    exit()'''
    '''
    import matplotlib.pyplot as plt
    fig=plt.figure()
    ax1=fig.add_subplot(1,3,1)
    ax1.pcolormesh(kcoor[:,:,0,0],kcoor[:,:,0,1],np.sum(berryCurv[:,:,0,2,:-4],axis=(-1,)),cmap="Spectral")
    ax1.axis('equal')

    ax2=fig.add_subplot(1,3,2)
    ax2.pcolormesh(kcoor[:,:,0,0],kcoor[:,:,0,1],energies[:,:,0,-5],cmap="Spectral")
    ax2.axis('equal')

    ax3=fig.add_subplot(1,3,3)
    ax3.pcolormesh(kcoor[:,:,0,0],kcoor[:,:,0,1],energies[:,:,0,-4],cmap="Spectral")
    ax3.axis('equal')
    plt.show()
    #'''

if __name__ == '__main__': # crucial for multiprocessing in Windows
    freeze_support()
    run()
