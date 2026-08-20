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
from task import solveBPVE_TDSE, solveBPVE_VG, solveBPVE_VG_DM, solveBPVE_TDSE_LG, solveBPVE_LG_Shift, solveBPVE_LG_Inject, solveBPVE_LG_DM
from task import solveIFE_TDSE, solveIFE_BlochAcceleration_TDSE, solveIFE_VG, solveIFE_VG_DM, solveIFE_LG, solveIFE_LG_DM
from task import solveIFTorque_TDSE, solveIFTorque_VG_DM, solveIFTorque_LG_DM
from task import solveINPHE_BCP, solve2ndAHC_FermiSea, solve2ndAHC_FermiSurf
#+------------+
#| Line tasks |
#+------------+
from task import solveSurfaceSpectraTask, solveBandStructure
#+---------------------------+
#| Tasks free from interface |
#+---------------------------+
from calculator import Chern_FHS, Z2_FHS
from auxiliary import showElapsedTime, getCBM, getVBM
from fileio import writeVolumeData, output3rdFreqDenpendentData, output4thData, outputBZDistributionData

def run():
    # Pbene, spin hall conductance
    mainDir="./mcalc/2D-rashba-ife/"
    model=System_tb(
        mainDir=mainDir,
        #hamR="omx",
        #hamR="BN_fm_rashiba",
        #cutoff=-0.002,
        poscar="POSCAR",
        projections={
            "B":(0,),
            "N":(0,),
            "Ge":(0,0,1,2),
            "Mn":(0,1,2,),
            "Sn":(0,1,),
            "Pb":(0,1,),
            "Pt":(0,1,2,),
            "C":(0,1,),
            "Fe":(2,),
            "Mo":(2,),
            "S":(1,),
            "Bi":(0,1,),
            "Te":(0,1,),
            "Cl":(1,),
            "Te":(1,),
            "H":(0,),
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
    # modulate the SOC, exchange fields, baised fields, etc.
    #model.scaleSOC(lam=0)
    #model.rotateExchangeField(projection=[(0,(0,)),(1,(0,))], theta=np.pi, phi=0.)
    #model.applyElectrostaticPotentialAtOnsites([0., 0., 0.,]) # in eV/angstrom
    
    fermi=0            # model hamiltonian
    fermiAxis=np.linspace(fermi-2.0,fermi+2.0,101)
    omegaAxis=[0.]
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

    BZGrid=(300,300,1)
    BZSymm=False
    recordFermiIndexList=[50]
    BZDistributionRepeat=(1,1,0)
    calc=BZGridTask(model, solveINPHE_BCP,
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
                          'record fermi ids':recordFermiIndexList,
                          "mag type": "total", # Type of Inversed Faraday effect, can be spin | orbital | total
                    }, 
                    BZGrid=BZGrid,BZSymm=BZSymm,
                    mode='run',
                    ncores=8)
    #exit()
    #_,_,_,nbands=calc.results[0]["energies"].shape
    BCP, BCP_dipole, response = None, None, None
    sigma=0
    for result_q in calc.results:
        if 'BCP' in result_q:
            if BCP is None:
                BCP = np.zeros(tuple(BZGrid) + result_q['BCP'].shape[3:], dtype=result_q['BCP'].dtype)
            BCP[result_q['iq'][0]::calc.QGrid[0],
                             result_q['iq'][1]::calc.QGrid[1],
                             result_q['iq'][2]::calc.QGrid[2],
                             ...] = result_q['BCP']
        if 'BCP dipole' in result_q:
            if BCP_dipole is None:
                BCP_dipole = np.zeros(tuple(BZGrid) + result_q['BCP dipole'].shape[3:], dtype=result_q['BCP dipole'].dtype)
            BCP_dipole[result_q['iq'][0]::calc.QGrid[0],
                             result_q['iq'][1]::calc.QGrid[1],
                             result_q['iq'][2]::calc.QGrid[2],
                             ...] = result_q['BCP dipole']
        if 'response' in result_q:
            if response is None:
                response = np.zeros(tuple(BZGrid) + result_q['response'].shape[3:], dtype=result_q['response'].dtype)
            response[result_q['iq'][0]::calc.QGrid[0],
                             result_q['iq'][1]::calc.QGrid[1],
                             result_q['iq'][2]::calc.QGrid[2],
                             ...] = result_q['response']
        if not BZSymm:  # symmetrization on the BZ is only implemented for the whole tensor
            sigma += result_q['sigma']
        #else:
        #    sigma += model.pointGroup.symmetrize_tensor(result_q['sigma'],Iodd=SpinHallConductance.Iodd,TRodd=SpinHallConductance.Todd,rank=3)
    
    #'''

    '''
    if BCP is not None:
        outputBZDistributionData(
                BCP[..., 0],
                filePath=f"{mainDir}/BCPDistribution.txt",
                reciLattice=model.reciLattice,
                bzRepeat=BZDistributionRepeat
            )
    if BCP_dipole is not None:
        outputBZDistributionData(
                BCP_dipole[..., 0],
                filePath=f"{mainDir}/BCPDipoleDistribution.txt",
                reciLattice=model.reciLattice,
                bzRepeat=BZDistributionRepeat
            )
                
    
    exit()#'''

    # output projected 3-ranked, frequency-dependent complex conductance
    #output3rdFreqDenpendentData(sigma.real[None,None,None,:,None,...]/len(calc.results), fermi=fermi, fermiAxis=fermiAxis, omegaAxis=omegaAxis, 
    #                            filePath="%s/thirdOrderFreqDependent.real.txt"%mainDir)

    # output 4-ranked conductance
    output4thData(sigma/len(calc.results), fermi, fermiAxis, filePath="%s/fourthOrder.txt"%mainDir)
    #print(response.shape)
    #exit()

    kcoor=np.array([[[model.reciLattice@(np.array([ik1,ik2,ik3])/BZGrid) for ik3 in range(BZGrid[2])] for ik2 in range(BZGrid[1])] for ik1 in range(BZGrid[0])])
    #'''
    import matplotlib.pyplot as plt
    data_grid=response[:,:,0, 49, 0,1,1,1].real
    magnitude=max([abs(data_grid.max()),abs(data_grid.min())])
    print(np.sum(data_grid))
    for G in [(0,0,0,),(-1,0,0,),(0,-1,0,),(-1,-1,0,),]:
        _kcoor=kcoor+model.reciLattice@np.array(G)
        plt.pcolormesh(_kcoor[:,:,0,0],_kcoor[:,:,0,1],data_grid,vmax=magnitude,vmin=-magnitude,cmap="Spectral")
    plt.colorbar()
    plt.axis("equal")
    plt.savefig(f"{mainDir}/BZ_distr1.png",dpi=300)
    plt.show()
    exit()

    data_grid=BCP_dipole[:,:,0, 1,0,0, 0].real
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
