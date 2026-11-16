"""compute bandstructure and / or the Surface band structure using modelSetting setup.
Usage:
1. define the model in modelSetting.py
2. python3 inputBand.py --soc

The k-path is read from the mainDir / KPOINTS
"""
import os
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[name]='1'
import argparse
from pathlib import Path
import numpy as np
from task import BZLineTask
from task import solveSurfaceSpectraTask, solveBandStructure

def run_bandstructure(model,mainDir,options):
    out=Path(mainDir,options.subdir)
    out.mkdir(parents=True,exist_ok=True)

    spin_proj, orbital_proj, soc_proj = options.spin, options.orbital, options.soc
    if soc_proj: spin_proj, orbital_proj = True, True

    calc=BZLineTask(model, solveBandStructure, KPOINTS=mainDir+"/KPOINTS", ncores=options.workers,
                   args={"spin proj":spin_proj, "orbital proj":orbital_proj, "degen":1e-2},
                   )
    energies=np.zeros((calc.nkpt,model.norbs),dtype=float)
    if spin_proj: spin_eig=np.zeros((calc.nkpt,3,model.norbs),dtype=float)
    if orbital_proj: orbital_eig = np.zeros((calc.nkpt,3,model.norbs),dtype=float)
    if soc_proj: spinOrbital_eig = np.zeros((calc.nkpt,model.norbs),dtype=float)
    for result in calc.results:
        energies[result['indices']]   =result['energies']
        if spin_proj: spin_eig[result['indices']]   =np.diagonal(result['spin_eig'],axis1=-2,axis2=-1)
        if orbital_proj: orbital_eig[result['indices']]=np.diagonal(result['orbital_eig'],axis1=-2,axis2=-1)
        if soc_proj: spinOrbital_eig[result['indices']]=np.diagonal(result['spinOrbital_eig'],axis1=-2,axis2=-1)
    with open(out / "tbbands.txt","w") as fout:
        fout.write(" ikpt     Energy(eV) ")
        if spin_proj:    fout.write("          muSx/muB      muSy/muB      muSz/muB ")
        if orbital_proj: fout.write("          muLx/muB      muLy/muB      muLz/muB ")
        if soc_proj:     fout.write("         <L.S>/hbar^2 ")
        fout.write("\n")
        for iband in range(model.norbs):
            for ik, energy in enumerate(energies[:,iband]):
                fout.write(f"{ik:5d}{energy:15.6E}")
                if spin_proj: fout.write(f"{spin_eig[ik,0,iband]:15.6E}{spin_eig[ik,1,iband]:15.6E}{spin_eig[ik,2,iband]:15.6E}")
                if orbital_proj: fout.write(f"{orbital_eig[ik,0,iband]:15.6E}{orbital_eig[ik,1,iband]:15.6E}{orbital_eig[ik,2,iband]:15.6E}")
                if soc_proj: fout.write(f"{spinOrbital_eig[ik,iband]:15.6E}")
                fout.write("\n")
            fout.write("\n")
    return

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subdir',default='bandstructure')
    p.add_argument('--spin',action='store_true',help='compute <S> for each state')
    p.add_argument('--orbital',action='store_true',help='compute <L> for each state')
    p.add_argument('--soc',action='store_true',help='compute dimensionless ACA <L.S>/hbar^2 for each state')
    p.add_argument('--workers',type=int,default=8)
    options=p.parse_args()

    from modelSetting import setModel
    model, mainDir = setModel()
    run_bandstructure(model,mainDir,options)

if __name__=='__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
