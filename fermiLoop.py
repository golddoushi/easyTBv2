import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from multiprocessing import freeze_support

import paramReader
paramReader.readGlobalConst()
import numpy as np
from contourpy import contour_generator
from system import System_omx, System_w90, System_tb
from task import BZGridTask, BZLineTask
from task import solveMagneticMoment, solveBandStructure

def extractPeriodicContours(energy2D:np.ndarray, level:float):
    '''
    Extract ordered constant-energy polylines on a periodic 2D reduced-k
    grid using marching squares.

    The input grid contains points i/N rather than the duplicated periodic
    endpoint. The first row and column are appended at reduced coordinate
    one before contouring, so contours reach the Brillouin-zone boundary
    without an artificial gap.
    '''
    assert energy2D.ndim == 2
    nk1, nk2 = energy2D.shape

    energyPeriodic = np.empty((nk1+1,nk2+1),dtype=energy2D.dtype)
    energyPeriodic[:-1,:-1] = energy2D
    energyPeriodic[-1,:-1] = energy2D[0,:]
    energyPeriodic[:-1,-1] = energy2D[:,0]
    energyPeriodic[-1,-1] = energy2D[0,0]

    k1Reduced = np.arange(nk1+1,dtype=float)/nk1
    k2Reduced = np.arange(nk2+1,dtype=float)/nk2
    generator = contour_generator(
        x=k1Reduced,
        y=k2Reduced,
        z=energyPeriodic.T,
        line_type="Separate",
    )

    contours = []
    for contour in generator.lines(level):
        if len(contour) < 2:
            continue
        keep = np.r_[True,np.any(np.abs(np.diff(contour,axis=0))>1e-14,axis=1)]
        contour = contour[keep]
        if len(contour) >= 2:
            contours.append(contour)
    return contours

def interpolatePeriodicGrid(data:np.ndarray, reducedK:np.ndarray):
    '''
    Bilinearly interpolate data whose first two axes form a periodic grid.

    ``reducedK`` has shape (npoint,2). Any remaining axes of ``data`` are
    retained in the returned array.
    '''
    assert data.ndim >= 2
    assert reducedK.ndim == 2 and reducedK.shape[1] == 2
    nk1, nk2 = data.shape[:2]

    scaled = np.mod(reducedK,1.0)*np.array([nk1,nk2])
    lower = np.floor(scaled).astype(int)
    fraction = scaled-lower
    i0, j0 = lower[:,0]%nk1, lower[:,1]%nk2
    i1, j1 = (i0+1)%nk1, (j0+1)%nk2

    shape = (len(reducedK),)+(1,)*(data.ndim-2)
    fx = fraction[:,0].reshape(shape)
    fy = fraction[:,1].reshape(shape)
    return (
        (1-fx)*(1-fy)*data[i0,j0]
        + fx*(1-fy)*data[i1,j0]
        + (1-fx)*fy*data[i0,j1]
        + fx*fy*data[i1,j1]
    )

def getBZTranslations(bzRepeat):
    '''
    Return integer reduced-coordinate translations for an extended BZ.

    This follows ``outputBZDistributionData``: a positive repeat ``n`` spans
    reduced coordinates from ``-n`` to ``+n`` and therefore contains ``2*n``
    periods. A zero repeat keeps one unshifted period on that axis.
    '''
    if np.isscalar(bzRepeat):
        bzRepeat = (int(bzRepeat),)*3
    else:
        bzRepeat = tuple(int(value) for value in bzRepeat)
    assert len(bzRepeat) == 3, (
        f"expect BZDistributionRepeat to contain three integers but {bzRepeat=}"
    )
    assert all(value >= 0 for value in bzRepeat), (
        f"expect non-negative BZDistributionRepeat but {bzRepeat=}"
    )

    axisTranslations = [
        range(-value,value) if value > 0 else range(1)
        for value in bzRepeat
    ]
    return np.array([
        (shift1,shift2,shift3)
        for shift1 in axisTranslations[0]
        for shift2 in axisTranslations[1]
        for shift3 in axisTranslations[2]
    ],dtype=float)

def run():
    # Pbene, spin hall conductance
    mainDir="./mcalc/AuH111/"
    model=System_omx(
        mainDir=mainDir,
        hamR="omx",
        cutoff=-0.002,
        poscar="POSCAR",
        projections={
            "B":(0,),
            "N":(0,),
            #"H":(0,),
            "Mo":(2,),
            "Te":(1,),
            "Au":(0,0,0,1,1,2,2,),
            "H":(0,0,1,),
            "Sr":(0,0,0,1,1,2,2,),
            "Ti":(0,0,0,1,1,2,),
             "O":(0,0,1,1,2,),
        },
        ispin=2
    )
    fermi=-0.22660759670266*27.2113845
    model.setFermi(fermi)

    BZGrid=(40,40,1)
    BZDistributionRepeat=(1,1,0)
    calc=BZGridTask(model, solveMagneticMoment,
                    args={'spin proj':False,'orbital proj':True,'skip low':True,
                    }, 
                    BZGrid=BZGrid,BZSymm=False,
                    mode='run',
                    ncores=8)
    _,_,_,nbands=calc.results[0]["energies"].shape
    energies=np.zeros(tuple(BZGrid)+(nbands,))
    spin=np.zeros(tuple(BZGrid)+(3, nbands,),dtype=complex)
    orbital_LC=np.zeros(tuple(BZGrid)+(3, nbands,),dtype=complex)
    orbital_IC=np.zeros(tuple(BZGrid)+(3, nbands,),dtype=complex)
    hasSpin, hasOrbitalLC, hasOrbitalIC=False, False, False
    for result_q in calc.results:
        energies[result_q['iq'][0]::calc.QGrid[0],result_q['iq'][1]::calc.QGrid[1],result_q['iq'][2]::calc.QGrid[2]]=result_q['energies']
        if result_q["spin"] is not None:
            hasSpin=True
            spin[result_q['iq'][0]::calc.QGrid[0],result_q['iq'][1]::calc.QGrid[1],result_q['iq'][2]::calc.QGrid[2]]=result_q['spin']
        if result_q["orbital LC"] is not None:
            hasOrbitalLC=True
            orbital_LC[result_q['iq'][0]::calc.QGrid[0],result_q['iq'][1]::calc.QGrid[1],result_q['iq'][2]::calc.QGrid[2]]=result_q['orbital LC']
        if result_q["orbital IC"] is not None:
            hasOrbitalIC=True
            orbital_IC[result_q['iq'][0]::calc.QGrid[0],result_q['iq'][1]::calc.QGrid[1],result_q['iq'][2]::calc.QGrid[2]]=result_q['orbital IC']
    
    assert BZGrid[2] == 1, (
        "Fermi curves require a two-dimensional BZ grid with BZGrid[2] == 1"
    )
    bzTranslations=getBZTranslations(BZDistributionRepeat)
    with open(mainDir+"/fermi_surf.txt","w") as fout:
        fout.write(                 "             kx             ky             kz iband")
        if hasSpin:      fout.write("             Sx             Sy             Sz")
        if hasOrbitalLC: fout.write("          Lx_LC          Ly_LC          Lz_LC")
        if hasOrbitalIC: fout.write("          Lx_IC          Ly_IC          Lz_IC")
        fout.write("\n")

        for iB in range(nbands):
            energy2D=energies[:,:,0,iB]
            if fermi < np.min(energy2D) or fermi > np.max(energy2D):
                continue

            for contour in extractPeriodicContours(energy2D,fermi):
                spinContour=None
                orbitalLCContour=None
                orbitalICContour=None
                if hasSpin:
                    spinContour=interpolatePeriodicGrid(
                        spin[:,:,0,:,iB].real,
                        contour,
                    )
                if hasOrbitalLC:
                    orbitalLCContour=interpolatePeriodicGrid(
                        orbital_LC[:,:,0,:,iB].real,
                        contour,
                    )
                if hasOrbitalIC:
                    orbitalICContour=interpolatePeriodicGrid(
                        orbital_IC[:,:,0,:,iB].real,
                        contour,
                    )

                reducedKBase=np.column_stack((contour,np.zeros(len(contour))))
                for translation in bzTranslations:
                    kCartesian=(reducedKBase+translation)@model.reciLattice.T
                    for ipoint, (kx,ky,kz) in enumerate(kCartesian):
                        fout.write(f"{kx:15.5E}{ky:15.5E}{kz:15.5E}{iB:6d}")
                        if hasSpin:
                            endpoint=spinContour[ipoint]
                            fout.write("".join(f"{value:15.5E}" for value in endpoint))
                        if hasOrbitalLC:
                            endpoint=orbitalLCContour[ipoint]
                            fout.write("".join(f"{value:15.5E}" for value in endpoint))
                        if hasOrbitalIC:
                            endpoint=orbitalICContour[ipoint]
                            fout.write("".join(f"{value:15.5E}" for value in endpoint))
                        fout.write("\n")
                    fout.write("\n")
if __name__ == '__main__': # crucial for multiprocessing in Windows
    freeze_support()
    run()
