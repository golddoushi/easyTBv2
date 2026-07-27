import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from multiprocessing import freeze_support

import paramReader
paramReader.readGlobalConst()
import numpy as np
from system import System_omx, System_w90
from task import BZGridTask, BZLineTask
#+------------+
#| Bulk tasks |
#+------------+
from task import solvePDOS
from task import solveIFE_TDSE, solveIFE_BlochAcceleration_TDSE, solveIFE_VG, solveIFE_VG_DM, solveIFE_LG, solveIFE_LG_DM
from task import solveIFTorque_TDSE, solveIFTorque_VG_DM, solveIFTorque_LG_DM

from fileio import writeVolumeData, output3rdFreqDenpendentData, output4thData, outputBZDistributionData

def run():
    # Pbene, spin hall conductance
    mainDir="./mcalc/FeCl2/"
    model=System_w90(
        mainDir=mainDir,
        #hamR="omx",
        #hamR="BN_fm_rashiba",
        cutoff=-0.002,
        poscar="POSCAR",
        projections={
            "Fe":(0,1,2,),
            "Cl":(0,1,),
        },
        ispin=2,
        spinSeparated=True,
    )
    # Symmetrize the direction-independent collinear and orbital-SOI
    # ingredients once, then assemble any requested magnetization direction.
    model.symmerizeCollinearAndSOI(
        print_row=(0,1,9,10,),print_col=(0,1,9,10,),
    )

    fermi=-2.5         # VASP 2H-FeCl2 U=2.0
    fermiAxis=np.array([fermi,])
    omegaAxis=np.linspace(0.5,2.0,21)
    model.setFermi(fermi)
    
    BZGrid=(48,48,1)

    thetaAxis=np.linspace(0,np.pi,21)
    phiAxis=np.linspace(0,2*np.pi,30,endpoint=False)
    energyOutput=open("%s/AngleDependentEnergy.txt"%mainDir,"w")
    energyOutput.write("          Theta            Phi    Energy(meV)\n")
    for theta in thetaAxis:
        for phi in phiAxis:
            model.includeSOCFromSOI(theta=theta,phi=phi,strength=1,keepRegularSpinAxis=True)
            calc=BZGridTask(model, solvePDOS,
                    args={"fermi":fermiAxis, 
                          'omega':omegaAxis,
                          "mag type": "total", # Type of Inversed Faraday effect, can be spin | orbital | total
                    }, 
                    BZGrid=BZGrid,BZSymm=False,
                    mode='run',
                    ncores=8)
            _,_,_,nbands=calc.results[0]["energies"].shape
            energies=np.zeros(tuple(BZGrid)+(nbands,))
            projected_sigma=0
            for result_q in calc.results:
                energies[result_q['iq'][0]::calc.QGrid[0],result_q['iq'][1]::calc.QGrid[1],result_q['iq'][2]::calc.QGrid[2]]=result_q['energies']
        
                if 'projected sigma' in result_q: 
                    projected_sigma += result_q['projected sigma']
            occupied=energies<=model.fermi
            energyGroundState=np.sum(energies[occupied]/np.prod(BZGrid))
            energyOutput.write(f"{theta:15.6E}{phi:15.6E}{energyGroundState*1000:18.9E}\n")
    
    '''
    # output projected 3-ranked, frequency-dependent complex conductance
    output3rdFreqDenpendentData(projected_sigma.real/len(calc.results), fermi=fermi, fermiAxis=fermiAxis, omegaAxis=omegaAxis, 
                                filePath="%s/thirdOrderFreqDependent.real.txt"%mainDir)
    output3rdFreqDenpendentData(projected_sigma.imag/len(calc.results), fermi=fermi, fermiAxis=fermiAxis, omegaAxis=omegaAxis, 
                                filePath="%s/thirdOrderFreqDependent.imag.txt"%mainDir)
    #'''

if __name__ == '__main__': # crucial for multiprocessing in Windows
    freeze_support()
    run()
