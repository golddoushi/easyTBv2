import os
os.add_dll_directory("D:/Documents/mingW/mingw64/bin")  # temporary adding path of .dll on Windows
os.add_dll_directory("D:/Documents/fftw3/bin")
os.add_dll_directory("D:/Documents/lapack/bin")
import numpy as np
import matplotlib
matplotlib.use('Agg')
import sys
import paramReader
paramReader.readGlobalConst()
import posReader as pR
import hrReader as reader
import couplingCalc as cc
from auxiliary import obtainProjectionsPlain, obtainProjections

# 1. define the basic structure information
mainDir='./mcalc/BN/'
pR.readPOS(mainDir+'POSCAR')
projections={
    "B":(0,),
    "N":(0,),
}
atomicMagMomList=[np.zeros(3)]*0+[
    np.array([0,0,1]),
    np.array([0,0,0]),
    np.array([0,0,0]),
]
# 2. read in Hamiltonian 
hr=reader.Hr()
hr.readSingleFileV2(mainDir+'BN_fm_soc_hr.dat',ignoreImagPart=False,cutoff=-1e-2,)
# 3. (optional) symmerize Hamiltonian, saxis=(0,0,-1) for w90, saxis=(0,0,-1) for w90
atomicOrbInfoList=obtainProjections(mainDir+'POSCAR',projections,ispin=2)
'''
hr.symmerizeHamAndOverlap(atomicOrbInfoList=atomicOrbInfoList,atomicMagMomList=atomicMagMomList,
                  cutoff=1e-9,symOpSelection=[],
                  spinor=True,saxis=np.array([0,0,-1]),format='openmx',
                  maxPrint=100,print_row=(),print_col=(), # print several local Hams. of bonds in symmertry-set
                  symmerizationMode='E', # directly symmerize the H_R read from file
                  )
#exit()'''
#hr.symmerizeHamAndOverlap(atomicOrbIndexList=atomicOrbIndexList,atomicLList=atomicLList,atomicMagMomList=atomicMagMomList,
#                  cutoff=1e-9,symOpSelection=[0,2,4,]+[1,3,5,],
#                  spinor=True,saxis=np.array([0,0,-1]),format='w90',
#                  maxPrint=4,print_row=(0,1,2,3,4,),print_col=(0,1,2,3,4,5), # print several local Hams. of bonds in symmertry-set
#                  )
#exit()#'''
# 4. assign the orbitals of spin-up tunnels to atoms (no need to change)
wann_list=obtainProjectionsPlain(mainDir+'POSCAR',projections)
# 5. set the iatom of magnetic atoms to source_list, set ini_S for ALL atoms
cc.classifyBonds(mainDir, source_list=[0,], 
           wann_list=wann_list, 
           ini_S=[1]*1+[0]*1,
           includeAllBonds=False)
# 5. set overlap, set supercell size, set fermi level, set erange, etc.
E0=0.0
fermiAxis=[E0]#np.linspace(E0-0.1,E0+0.1,11)
cc.calcJFull_GaussianQuadrature(hr,mainDir,nthreads=8,overlap=True,
    sc=[28,28,1],fermiAxis=fermiAxis,
    eta=0.0001,erange=(-100,100),lowOrder=60,highOrder=80,Jcvg=-1e-4)
