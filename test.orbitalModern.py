import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from multiprocessing import freeze_support

import paramReader
paramReader.readGlobalConst()
import numpy as np
from system import System_omx, System_w90, System_tb
from task import BZGridTask, BZLineTask
from task import solveMagneticMoment, solveBandStructure
#+---------------------------+
#| Tasks free from interface |
#+---------------------------+
from fileio import writeVolumeData, output3rdFreqDenpendentData, outputBZDistributionData
from scipy.constants import elementary_charge, electron_mass, hbar, angstrom

def run():
    # Pbene, spin hall conductance
    mainDir="./mcalc/square/"
    model=System_tb(
        mainDir=mainDir,
        hamR="4orbs",
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
            "Bi":(0,1,),
            "Te":(0,1,),
            #"Fe":(0,0,1,1,2,2,),
            "Cl":(1,),
            #"Cl":(0,0,1,1,2,3),
            "H":(0,),
            #"H":(0,0,1,),
            "Au":(0,0,0,1,1,2,2,),
            "Sr":(0,0,0,1,1,2,2,),
            "Ti":(0,0,0,1,1,2,),
             "O":(0,0,1,1,2,),
        },
        ispin=1
    )

    #'''
    # set flux phi1, -(phi1+phi2), phi2, -(phi1+phi2)
    t1=-1
    Phi=np.pi*0.6
    phi1, phi2=2*Phi, 0*Phi
    # left down plaquett, iR = 0 corresponding to R= (0, 0, 0)
    model.ham_R[0,0,1] = t1*np.exp(1j*phi1/4)
    model.ham_R[0,1,2] = t1*np.exp(1j*phi1/4)
    model.ham_R[0,2,3] = t1*np.exp(1j*phi1/4)
    model.ham_R[0,3,0] = t1*np.exp(1j*phi1/4)
    # right up plaquett, iR = 1, 2, 3, 4 corresponding to R= (-1, 0, 0), (0, -1, 0), (1, 0, 0), (0, 1, 0)
    model.ham_R[3,2,3] = t1*np.exp(1j*phi2/4)
    model.ham_R[4,3,0] = t1*np.exp(1j*phi2/4)
    model.ham_R[1,0,1] = t1*np.exp(1j*phi2/4)
    model.ham_R[2,1,2] = t1*np.exp(1j*phi2/4)
    # h.c.
    model.ham_R[0,1,0] = t1*np.exp(-1j*phi1/4)
    model.ham_R[0,2,1] = t1*np.exp(-1j*phi1/4)
    model.ham_R[0,3,2] = t1*np.exp(-1j*phi1/4)
    model.ham_R[0,0,3] = t1*np.exp(-1j*phi1/4)
    
    model.ham_R[1,3,2] = t1*np.exp(-1j*phi2/4)
    model.ham_R[2,0,3] = t1*np.exp(-1j*phi2/4)
    model.ham_R[3,1,0] = t1*np.exp(-1j*phi2/4)
    model.ham_R[4,2,1] = t1*np.exp(-1j*phi2/4)
    
    model._refreshHamiltonianDerivedMatrices()
    #'''

    fermi=-4            # model hamiltonian
    model.setFermi(fermi)

    '''
    # check band intepolation
    spin_proj, orbital_proj= False, False
    calc=BZLineTask(model, solveBandStructure, KPOINTS=mainDir+"/KPOINTS", ncores=8,
                   args={"spin proj":spin_proj, "orbital proj":orbital_proj},
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

    #'''
    BZGrid=(102,102,1)
    BZDistributionRepeat=(1,1,0)
    calc=BZGridTask(model, solveMagneticMoment,
                    args={'spin proj':False,'orbital proj':True,
                    }, 
                    BZGrid=BZGrid,BZSymm=False,
                    mode='run',
                    ncores=8)
    _,_,_,nbands=calc.results[0]["energies"].shape
    energies=np.zeros(tuple(BZGrid)+(nbands,))
    spin=np.zeros(tuple(BZGrid)+(3, nbands,),dtype=complex)
    orbital_LC=np.zeros(tuple(BZGrid)+(3, nbands,),dtype=complex)
    orbital_IC=np.zeros(tuple(BZGrid)+(3, nbands,),dtype=complex)
    for result_q in calc.results:
        energies[result_q['iq'][0]::calc.QGrid[0],result_q['iq'][1]::calc.QGrid[1],result_q['iq'][2]::calc.QGrid[2]]=result_q['energies']
        if result_q["spin"] is not None:
            spin[result_q['iq'][0]::calc.QGrid[0],result_q['iq'][1]::calc.QGrid[1],result_q['iq'][2]::calc.QGrid[2]]=result_q['spin']
        if result_q["orbital LC"] is not None:
            orbital_LC[result_q['iq'][0]::calc.QGrid[0],result_q['iq'][1]::calc.QGrid[1],result_q['iq'][2]::calc.QGrid[2]]=result_q['orbital LC']
        if result_q["orbital IC"] is not None:
            orbital_IC[result_q['iq'][0]::calc.QGrid[0],result_q['iq'][1]::calc.QGrid[1],result_q['iq'][2]::calc.QGrid[2]]=result_q['orbital IC']
        
    
    occupied = energies <= fermi
    Lz_band      = np.sum(orbital_LC[...,2,:][occupied]).real/np.prod(BZGrid)
    Lz_itinerant = np.sum(orbital_IC[...,2,:][occupied].real) / np.prod(BZGrid) 
    Lz=Lz_band+Lz_itinerant
    #print(np.sum(berryCurv[:,:,:,0,1,:][occupied]))
    print(f"Reciprocal-space computations with kgrid {BZGrid[0]}x{BZGrid[1]}x{BZGrid[2]} gives {Lz=:.6E} uB/cell")
    #exit()
    #'''

    # compute orbital angular moment in real-space
    Ncell=21
    Tmatrix=[
        [Ncell, 0, 0], # The A1 of the supercell in the frame of primitive cell
        [ 0,Ncell, 0], # The A2 of the supercell in the frame of primitive cell
        [ 0, 0, 1]  # The A3 of the supercell in the frame of primitive cell
    ]
    model.make_supercell(Tmatrix)
    model.cutEdge(cutDir=0,)
    model.cutEdge(cutDir=1,)

    calc=BZGridTask(model, solveMagneticMoment,
                    args={'spin proj':False,'orbital proj':False,
                    }, 
                    BZGrid=(1,1,1),BZSymm=False,
                    mode='run',
                    ncores=1)
    #print(model.rList)
    energies = calc.results[0]["energies"][0, 0, 0]
    vectors = calc.results[0]["vectors"][0, 0, 0]
    occupied = energies <= fermi
    vectors_occ = vectors[:, occupied]

    rx_occ_all = vectors_occ.conj().T @ (model.rList[0, :, None] * vectors)
    ry_occ_all = vectors_occ.conj().T @ (model.rList[1, :, None] * vectors)

    dE_occ_all = energies[None, :] - energies[occupied, None]
    
    Lz = np.sum(np.imag(rx_occ_all*ry_occ_all.conj())*dE_occ_all)/Ncell**2/(hbar**2/(2*electron_mass*elementary_charge*angstrom**2))
    print(f"Real-space computations on {Ncell}x{Ncell} model gives {Lz=:.6E} uB/cell")

if __name__ == '__main__': # crucial for multiprocessing in Windows
    freeze_support()
    run()
