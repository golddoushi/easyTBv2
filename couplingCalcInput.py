import numpy as np
from system import System_tb
from calculator import ExchangeCoupling
from fileio import writeJmatV2, writeMCSolverInputFile
from auxiliary import showElapsedTime

# 1. define model basic infomation
mainDir="./mcalc/FeCl2/"
model = System_tb(
        mainDir=mainDir,
        #hamR="BN_fm_soc",
        cutoff=-0.002,
        poscar="POSCAR",
        projections={
            "Fe":(0,1,2),
            "Cl":(0,1),
        },
        ispin=2
)

# 2. enforce the symmetrization
atomicMagMomList=[
    np.array([0, 0, 1]),
]*1+[
    np.array([0, 0, 0]),
]*2
model.symmerizeHamAndOverlap(
    atomicMagMomList=atomicMagMomList,spinor=True,print_row=(0,1,9,10,),print_col=(0,1,9,10,),
)

# 3. compute the exchange coupling
kgrid=(56, 56, 1)
pbc=(0, 1, )
fermiList=[-2.5]
routes=model.classifyExchangeRoutes(sourceAtoms=[0,], pbc=pbc, maxRange=5, calRange=5, )
showElapsedTime(extraInfo="Exchange coupling calc. start")
exCoup = ExchangeCoupling(model=model, kgrid=kgrid, 
                          fermiList = fermiList, eta=0.0001,
                          exchangeRoutes = routes, symmetricExchangeSign=1,
                          lowOrder=60,highOrder=80,Jcvg=-1e-4,
                          verbose=True,greenFuncOmegaBatch=1, greenFuncNWorkers=8) # nworkers > 1, corresponds to parallel runs

showElapsedTime(extraInfo="Exchange coupling calc. finished")

# 4. output
writeJmatV2(mainDir, exCoup, kgrid)
for ifermi, fermi in enumerate(fermiList):
    for ibondType, ex in enumerate(exCoup.exchangeRoutes):
        print("bondType%d fermi=%.4f orb%d-orb%d[%2d%2d%2d] len: %.2f J=%.4f Jax=%.4f Jay=%.4f Jaxy=%.4f Dx=%.4f Dy=%.4f Dz=%.4f"%(
                        ex["route_type"],fermi,ex["source_atom"],ex["target_atom"],*ex["overlat"],ex["bond_length"],
                        ex["exchangeEnergy"][ifermi]/2*1000,ex["Jani"][ifermi,0,0]/2*1000,ex["Jani"][ifermi,1,1]/2*1000,
                        ex["Jani"][ifermi,0,1]/2*1000,*ex["DMI"][ifermi,:]/2*1000,))

writeMCSolverInputFile(mainDir, exCoup, repetition=16, pbc=pbc,
                           singleIonAnisotropy=(0.0, 0.0, 0.0),
                           temperature=(100, 500, 16), field=(0, 0.1, 1),
                           sweeps=(40000, 80000, 0), modelType="Heisenberg",
                           algorithm="Metropolis", ncores=8, spin=1.0,
                           spinStates=None, magneticAtoms=None,
                           measurePair=None, filename="MCSolver.txt",
                           fermiIndex=0)