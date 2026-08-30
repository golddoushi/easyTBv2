import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from multiprocessing import freeze_support

import numpy as np
from system import System_tb, System_w90
from calculator import ExchangeCoupling
from fileio import DOSCAR, writeJmatV2, writeMCSolverInputFile
from auxiliary import showElapsedTime, solve_chemical_potential
from task import BZGridTask, solveEnergy

def run():
    # 1. define model basic infomation
    mainDir="./mcalc/CrI3/"
    model = System_tb(
            mainDir=mainDir,
            #hamR="BN_afm_soc",
            cutoff=-0.002,
            poscar="POSCAR",
            projections={
                "B":(0,),
                "N":(0,),
                "Fe":(0,1,2),
                "Cr":(0,2,),
                "Cl":(0,1),
                "Ga":(1,),
                "Te":(1,),
                 "I":(1,),
            },
            ispin=2,
            spinSeparated=True,
    )
    ncores=8 # parallel workers for solving both band energies and exchange couplings

    '''
    # 2a. Symmetrize the direction-independent collinear and orbital-SOI
    #     ingredients once, then assemble any requested magnetization direction.
    model.symmerizeCollinearAndSOI()
    model.getExchangeR(provenance="pauli estimate")
    magnetizationTheta ,magnetizationPhi = 0/180*np.pi, 0.0
    model.includeSOCFromSOI(theta=magnetizationTheta,phi=magnetizationPhi,strength=1,keepRegularSpinAxis=True)
    #model.getExchangeR(provenance="explicit")
    #'''

    '''
    # 2b. or enforce the symmetrization on whole SOC model
    atomicMagMomList=[
        np.array([0, 0, 1]),
    ]*1+[
        np.array([0, 0, 0]),
    ]*2
    model.symmerizeHamAndOverlap(
        atomicMagMomList=atomicMagMomList,spinor=True,print_row=(0,1,9,10,),print_col=(0,1,9,10,),
    )
    '''

    # 3. set BZ sampling and fermi energy
    kgrid=(12, 12, 1)
    fermi=-1.87                                # CrI3
    #fermi=DOSCAR(mainDir+"DOSCAR").fermi   # Fermi from DOSCAR
    fermiList=[fermi]

    #'''
    # (optional) determin the exact fermiList
    calc=BZGridTask(model, solveEnergy,  args={}, BZGrid=kgrid, BZSymm=False, mode='run', ncores=ncores)
    _,_,_,nbands=calc.results[0]["energies"].shape
    energies=np.zeros(tuple(kgrid)+(nbands,))
    for result_q in calc.results:
        energies[result_q['iq'][0]::calc.QGrid[0],result_q['iq'][1]::calc.QGrid[1],result_q['iq'][2]::calc.QGrid[2]]=result_q['energies']
    occ_initial = np.sum(1.0*(energies<=fermi))/np.prod(kgrid)
    exact_fermi = solve_chemical_potential(energies=energies, target_electrons_per_cell=np.around(occ_initial), 
                                           number_kpoints=np.prod(kgrid), kBT=0.026)
    fermiList=[exact_fermi]
    del calc
    print(f"{exact_fermi=:.6f}, filling = {np.around(occ_initial)} valence electrons per cell")
    showElapsedTime(extraInfo="Fermi energies obtained") 
    #'''

    # 4. compute the exchange coupling
    pbc=(0, 1, )
    routes=model.classifyExchangeRoutes(sourceAtoms=[0,1,], pbc=pbc, maxRange=6, calRange=6, )
    showElapsedTime(extraInfo="Exchange coupling calc. start")
    exCoup = ExchangeCoupling(model=model, kgrid=kgrid, 
                              fermiList = fermiList, eta=1e-4,
                              exchangeRoutes = routes, symmetricExchangeSign=1,
                              lowOrder=60,highOrder=80,Jcvg=-1e-4,
                              verbose=True,greenFuncOmegaBatch=1, greenFuncNWorkers=ncores) # nworkers > 1, corresponds to parallel runs

    showElapsedTime(extraInfo="Exchange coupling calc. finished")

    # 4. output
    writeJmatV2(mainDir, exCoup, kgrid)
    for ifermi, fermi in enumerate(fermiList):
        for ibondType, ex in enumerate(exCoup.exchangeRoutes):
            print("bondType%d fermi=%.4f orb%d-orb%d[%2d%2d%2d] len: %.2f J=%.4f Jax=%.4f Jay=%.4f Jaxy=%.4f Dx=%.4f Dy=%.4f Dz=%.4f"%(
                        ex["route_type"],fermi,ex["source_atom"],ex["target_atom"],*ex["overlat"],ex["bond_length"],
                        ex["exchangeEnergy"][ifermi]*1000,ex["Jani"][ifermi,0,0]*1000,ex["Jani"][ifermi,1,1]*1000,
                        ex["Jani"][ifermi,0,1]*1000,*ex["DMI"][ifermi,:]*1000,))

    writeMCSolverInputFile(mainDir, exCoup, repetition=16, pbc=pbc, offdiag=False, 
                           singleIonAnisotropy=(0.0, 0.0, 0.0),   # anisotropy energy per atom in eV
                           temperature=(100, 500, 16), field=(0, 0.1, 1),
                           sweeps=(40000, 80000, 0), modelType="Heisenberg",
                           algorithm="Metropolis", ncores=ncores, spin=1.0,
                           spinStates=None, magneticAtoms=None,
                           measurePair=None, filename="MCSolver.txt",
                           fermiIndex=0)

if __name__ == '__main__': # crucial for multiprocessing in Windows
    freeze_support()
    run()
