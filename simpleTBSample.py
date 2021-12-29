import numpy as np
import paramReader
paramReader.readGlobalConst()
import posReader as pR
import hrReader as reader
from ham import Green

pR.readPOS('./mcalc/FeCl2/POSCAR')
#pR.readPOS('./mcalc/MoTe2/POSCAR')
hr=reader.Hr()
#hr.readSingleFile('./mcalc/FeCl2/wannier90.dn_hr.dat',ignoreImagPart=True,cutoff=0.01)
hr.readSingleFile('./mcalc/FeCl2/wannier90_hr.dat',ignoreImagPart=False,cutoff=0.01)
#hr.readSingleFile('./mcalc/MoTe2/wannier90_hr.dat',ignoreImagPart=False,cutoff=0.1,format="QE")
# data structure atomicOrbIndexList[atom index][Lz][orb index]: give the orb. index corr. to hr.dat
atomicOrbIndexList=[[[],[],[0,1,2,3,4,11,12,13,14,15]],
                    [[],[5,6,7,16,17,18]],
                    [[],[8,9,10,19,20,21]],
                   ]
#atomicOrbIndexList=[[[],[],[0,1,2,3,4]],
#                    [[],[5,6,7]],
#                    [[],[8,9,10]],
#                   ]

# data structure atomicMagMomList[atom index]: give the SU(2) rep. of magnetic moments
atomicMagMomList=[[0,1], # in z direction
                  [0,0], # 0 mag.mom.
                  [0,0]  # 0 mag.mom.
                 ]

hr.symmerizeBonds(atomicOrbIndexList=atomicOrbIndexList,symOpSelection=[],
                  spinor=True,atomicMagMomList=atomicMagMomList)
hr.adjustSOCStrength(socFactor=0.0)
#print(hr.onsite)
#exit()
gr=Green()
gr.calcFullBands_CDLL(hr,fermi=-2.1,mainDir='./mcalc/FeCl2/',saveFig=True,redBands=[17,18])
#gr.calcFullBands_CDLL(hr,fermi=0.0,mainDir='./mcalc/MoTe2/',saveFig=True,redBands=[13,14])