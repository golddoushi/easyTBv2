import os
os.add_dll_directory("C:/Program Files/mingw64/bin")
import numpy as np
import paramReader
paramReader.readGlobalConst()
import posReader as pR
import hrReader as reader
from ham import Green, Response
from illustrator import drawJDOSOnHexagonalBZ
from auxiliary import obtainProjections

mainDir='./mcalc/FeCl2/'
pR.readPOS(mainDir+'POSCAR')
hr=reader.Hr()
hr.readSingleFileV2(mainDir+'wannier90_hr.mz.dat',ignoreImagPart=False,cutoff=1e-2)
#print(hr.onsite)
#exit()
pR.readPOS(mainDir+'POSCAR')
projections=[
    "Fe:d",
    "Cl:p",
]

# data structure atomicOrbIndexList[atom index][Lz][orb index]: give the orb. index corr. to hr.dat
'''
atomicOrbIndexList=obtainProjections(mainDir+"POSCAR",projections=projections,soc=True)
atomicOrbIndexList[0].insert(0,[])
atomicOrbIndexList[0].insert(0,[])
atomicOrbIndexList[1].insert(0,[])
atomicOrbIndexList[2].insert(0,[])
atomicMagMomList=[np.array([0,0,1]), # in z direction
                  np.array([0,0,0]),
                  np.array([0,0,0])]

hr.symmerizeBonds(atomicOrbIndexList=atomicOrbIndexList,symOpSelection=[],
                  spinor=True,saxis=np.array([0,0,-1]),atomicMagMomList=atomicMagMomList)
'''
#'''
atomicOrbIndexList, atomicLList=obtainProjections(mainDir+'POSCAR',projections,soc=True,getLList=True)
atomicMagMomList=[np.array([0,0,1]), # in z direction
                  np.array([0,0,0]),
                  np.array([0,0,0])]
hr.symmerizeHamAndOverlap(atomicOrbIndexList=atomicOrbIndexList,atomicLList=atomicLList,atomicMagMomList=atomicMagMomList,
                  cutoff=1e-9,symOpSelection=[],
                  spinor=True,saxis=np.array([0,0,-1]),format='w90',
                  maxPrint=100,print_row=(),print_col=(), # print several local Hams. of bonds in symmertry-set
                  )
#exit()
#'''
#hr.adjustSOCStrength(socFactor=0.0)
#hr.adjustInterLayerInteraction(
#    [0,1,2,3,4,72,73,74,75,76,5,6,7,8,9,77,78,79,80,81,10,11,12,82,83,84,13,14,15,85,86,87,16,17,18,88,89,90,19,20,21,91,92,93,22,23,24,94,95,96,25,26,27,97,98,99],
#    [],
#    factor=1.0
#)
#print(hr.onsite)
#exit()
gr=Green()
# chose one of the algorithm, calcFullBands_CDLL: directly solve the eigenvals
#gr.calcFullBands_CDLL(hr,fermi=0.0,mainDir=mainDir,saveFig=True,spinComponentForColor="s0",redBands=[0,1],nthreads=10)
# chose one of the algorithm, calcFullBandsWithOverlap_CDLL: solve the eigenvals with overlap matrix (LCAO formula)
gr.calcFullBandsWithOverlap(hr,mainDir=mainDir,ispin=2,fermi=0.0,saveFig=True,spinComponentForColor="s0",nthreads=10)