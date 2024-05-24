import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from multiprocessing import Pool, freeze_support

import paramReader
paramReader.readGlobalConst()
import numpy as np

from system import System_omx, System_w90
from calculator import Velocity, SpinHallConductance, Energy, SpinBerryCurv, AnormalousHallConductance
from auxiliary import showElapsedTime
from symmetry import SYMMETRY_PRECISION

ncpu=8

def solveQ(param):
    calc=SpinHallConductance(param['model'],kgrid=param['FFT grid'],phaseshift=param['phase'],eta=0.026,fermiList=param['fermi'])
    return calc.sigma

def run():
    # Pbene, spin hall conductance
    mainDir="./mcalc/Pt/"
    model=System_w90(
        mainDir=mainDir,
        hamR="wannier90",
        #hamR="omx",
        cutoff=-0.002,
        poscar="POSCAR",
        projections={
            "Sn":(0,1,),
            "Pb":(0,1,),
            "Pt":(0,1,2,),
        },
        ispin=2
    )
    atomicMagMomList=[
        np.zeros(3),
        #np.zeros(3),
    ]
    model.symmerizeHamAndOverlap(
        atomicMagMomList=atomicMagMomList,format="w90",spinor=True,print_row=(0,1,2,3,4),print_col=(0,1,2,3,4),
    )
    model.getRHam_R()
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
    #fermi=0 # OPENMX generated wannier model always displays fermi level at zero
    #fermi=-2.41861676 # VASP Sn
    fermi=10.96516405 # VASP Pt
    fermiAxis=np.linspace(fermi-2.25,fermi+2.25,101)

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
    
    pool=Pool(processes=ncpu)
    results=[]
    for result in pool.imap_unordered(solveQ,irreducibleTask):
        results.append(result)
        showElapsedTime(offset=False,extraInfo=f"{len(results):3d}/{len(irreducibleTask):3d} finished")

    showElapsedTime(extraInfo=f"integration finished")

    sigma=np.zeros((101,3,3,3))
    for result_q, qParallel in zip(results,irreducibleTask):
        if not BZSymm:
            sigma += result_q
        else:
            sigma += model.pointGroup.symmetrize_tensor(result_q,Iodd=SpinHallConductance.Iodd,TRodd=SpinHallConductance.Todd,rank=3)


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

if __name__ == '__main__': # crucial for multiprocessing in Windows
    freeze_support()
    run()