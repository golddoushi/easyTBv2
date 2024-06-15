import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from multiprocessing import Pool, freeze_support

import paramReader
paramReader.readGlobalConst()
import numpy as np

from system import System_omx, System_w90
from calculator import Velocity, SpinHallConductance, Energy, SpinBerryCurv, AnormalousHallConductance, EnergyOnPath
from auxiliary import showElapsedTime
from symmetry import SYMMETRY_PRECISION

ncpu=8

def solveQ(param):
    calc=SpinHallConductance(param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],eta=0.026,fermiList=param['fermi'])
    return {"energies":calc.energies,"velocities":calc.velocity_K,"sigma":calc.sigma,"iq":param["iq"]}

def solveVelocity(param):
    calc=Velocity(param['model'],kgrid=param['FFT grid'],)
    return {"energies":calc.energies,"velocities":calc.velocity_K,"iq":param["iq"]}

def run():
    # Pbene, spin hall conductance
    mainDir="./mcalc/Mn3Sn/"
    model=System_omx(
        mainDir=mainDir,
        #hamR="wannier90",
        hamR="omx",
        cutoff=-0.002,
        poscar="POSCAR",
        projections={
            #"Sn":(0,1,),
            "Mn":(0,0,0,1,1,2,),
            "Sn":(0,0,0,1,1,2,2,),
            "Pb":(0,1,),
            #"Pt":(0,1,2,),
            "Pt":(0,0,1,1,2,2,),
            "C":(0,1,),
        },
        ispin=2
    )
    a,b=1/2,np.sqrt(3)/2
    atomicMagMomList=[
        np.array([ a, b, 0]),
        np.array([-1, 0, 0]),
        np.array([ a,-b, 0]),
        np.array([ a, b, 0]),
        np.array([-1, 0, 0]),
        np.array([ a,-b, 0]),
        np.zeros(3),
        np.zeros(3),
    ]
    model.symmerizeHamAndOverlap(
        atomicMagMomList=atomicMagMomList,spinor=True,print_row=(10,11,12,13,14,15,16,17,),print_col=(10,11,12,13,14,15,16,17,),#(28,29,30,31,32,33,34,35,),
        symOpSelection=[0,],
    )
    #'''
    calc=EnergyOnPath(model, mainDir+"/KPOINTS")
    with open(mainDir+"./tbbands.txt","w") as fout:
        fout.write("#ikpt Energy(eV)\n")
        for iband in range(model.norbs):
            for ik, energy in enumerate(calc.energies[:,iband]):
                fout.write(f"{ik:5d}{energy:15.6E}\n")
            fout.write("\n")
    exit()
    #'''
    BZGrid=np.array([44,44,44])
    FastGrid=np.array(list(model.mp_grid))
    #FastGrid=BZGrid

    for icomp in range(3):
        for extra_size in range(BZGrid[icomp]-FastGrid[icomp]+1):
            if BZGrid[icomp]%FastGrid[icomp]==0: break
            FastGrid[icomp]+=1

    QGrid=BZGrid//FastGrid

    print(f"DFT grid: {model.mp_grid}")
    print(f"Intepolation BZ grid: {BZGrid}")
    print(f"FFT grid: {FastGrid}")
    print(f"Phase grid: {QGrid}")
    
    #fermi=-0.141600218424*27.2113845 # OPENMX Sn
    fermi=-0.16790565257604*27.2113845 # OPENMX Pt
    #fermi=-0.318001699563*27.2113845 # OPENMX C-chain
    #fermi=0 # OPENMX generated wannier model always displays fermi level at zero
    #fermi=-2.41861676 # VASP Sn
    #fermi=10.96516405 # VASP Pt
    
    fermiAxis=np.linspace(fermi-2.25,fermi+2.25,101)
    
    model.setFermi(fermi)
    '''
    from auxiliary import print_matrix
    calc=Velocity(model, FastGrid)
    
    ikc=900
    dk=model.reciLattice[0,0]/calc.kgrid[0]
    # finite diff. 
    Emat_p=calc.E_K_AO[ (ikc+1)%FastGrid[0],0,0]
    U_p   =calc.vectors[(ikc+1)%FastGrid[0],0,0]
    Lam_p=U_p.conj().T @ Emat_p @ U_p
    #print_matrix(Lam_p,rowIndex=(0,1,2,3,),colIndex=(0,1,2,3,))

    Emat_m=calc.E_K_AO[ (ikc-1)%FastGrid[0],0,0]
    U_m   =calc.vectors[(ikc-1)%FastGrid[0],0,0]
    Lam_m=U_m.conj().T @ Emat_m @ U_m
    #print_matrix(Lam,rowIndex=(0,1,2,3,),colIndex=(0,1,2,3,))

    v_fd =(Lam_p - Lam_m)/(2*dk)
    print_matrix(v_fd,rowIndex=(0,1,2,3,),colIndex=(0,1,2,3,))
    
    Emat=calc.E_K_AO[ ikc,0,0]
    U   =calc.vectors[ikc,0,0]
    Lam=U.conj().T @ Emat @ U
    dE_fd = calc.RHam_K[ikc,0,0,0]
    dO_fd = calc.RO_K[ikc,0,0,0]

    eq = -U.conj().T @ dO_fd @ U @ Lam + U.conj().T @ dE_fd @ U
    print_matrix(eq,rowIndex=(0,1,2,3,),colIndex=(0,1,2,3,))
    
    exit()
    '''

    BZSymm=False
    showElapsedTime()
    def distGamma(q):
        shift_corners = np.arange(-2, 3)
        corners = np.array([[x, y, z] for x in shift_corners for y in shift_corners for z in shift_corners])
        return np.linalg.norm(model.reciLattice @ ((q % 1)[:, None] - corners[:,None,:]), axis=1).min()
    def equiv(q1, q2):
        q_diff = model.pointGroup.star(q1)[:,None,:] - model.pointGroup.star(q2)[None,:,:]
        if np.linalg.norm((q_diff - np.round(q_diff)), axis=2).min() < SYMMETRY_PRECISION: return True
        return False
    
    irreducibleTask=[]
    for iq1 in range(QGrid[0]):
        for iq2 in range(QGrid[1]):
            for iq3 in range(QGrid[2]):
                q=np.array([iq1,iq2,iq3])
                qlen=distGamma(q/BZGrid)
                newTask={
                    'iq':q,
                    'q':q/BZGrid,
                    'qlen':qlen,
                    'phase':np.array([np.exp(2j*np.pi*(q/BZGrid@Rvec)) for Rvec in model.iRvec]),
                    'model':model,
                    'FFT grid':FastGrid,
                    'fermi':fermiAxis,
                }
                if not BZSymm: 
                    irreducibleTask.append(newTask)
                    continue
                for itask, oldTask in enumerate(irreducibleTask):
                    if abs(oldTask['qlen'] - qlen) < SYMMETRY_PRECISION:
                        if equiv(oldTask['q'],newTask['q']):
                            break
                        else:
                            irreducibleTask.insert(itask,newTask)
                            break
                    elif oldTask['qlen']>qlen+SYMMETRY_PRECISION:
                        irreducibleTask.insert(itask,newTask)
                        break
                else:
                    irreducibleTask.append(newTask)

    showElapsedTime(extraInfo=f"find {len(irreducibleTask)} irreducible q, {len(irreducibleTask)/np.prod(QGrid)*100:.3f}% the whole BZ")
    
    #'''
    pool=Pool(processes=ncpu)
    results=[]
    for result in pool.imap_unordered(solveQ,irreducibleTask):
        results.append(result)
        showElapsedTime(offset=False,extraInfo=f"{len(results):3d}/{len(irreducibleTask):3d} finished")

    showElapsedTime(extraInfo=f"integration finished")
    #'''
    #results=[solveVelocity(irreducibleTask[0])]

    #energies=np.zeros(tuple(BZGrid)+(model.norbs,))
    #velocity=np.zeros(tuple(BZGrid)+(3,model.norbs,))
    sigma=np.zeros((101,3,3,3))
    for result_q in results:
        #energies[result_q['iq'][0]::QGrid[0],result_q['iq'][1]::QGrid[1],result_q['iq'][2]::QGrid[2]]=result_q['energies']
        #velocity[result_q['iq'][0]::QGrid[0],result_q['iq'][1]::QGrid[1],result_q['iq'][2]::QGrid[2]]=np.diagonal(result_q['velocities'],axis1=-2,axis2=-1)
        #continue
        if not BZSymm:
            sigma += result_q['sigma']
        else:
            sigma += model.pointGroup.symmetrize_tensor(result_q['sigma'],Iodd=SpinHallConductance.Iodd,TRodd=SpinHallConductance.Todd,rank=3)

    #'''
    sigma/=len(irreducibleTask)
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
    #'''
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
    exit()
    import matplotlib.pyplot as plt
    fig=plt.figure()
    ax1=fig.add_subplot(1,3,1)
    ax1.pcolormesh(kcoor[:,:,0,0],kcoor[:,:,0,1],energies[:,:,0,27],cmap="Spectral")
    ax1.axis('equal')

    ax2=fig.add_subplot(1,3,2)
    ax2.pcolormesh(kcoor[:,:,0,0],kcoor[:,:,0,1],velocity[:,:,0,0,27],cmap="Spectral")
    ax2.axis('equal')

    ax3=fig.add_subplot(1,3,3)
    ax3.pcolormesh(kcoor[:,:,0,0],kcoor[:,:,0,1],velocity[:,:,0,1,27],cmap="Spectral")
    ax3.axis('equal')
    plt.show()
    #'''

if __name__ == '__main__': # crucial for multiprocessing in Windows
    freeze_support()
    run()