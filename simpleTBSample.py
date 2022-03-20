import numpy as np
import paramReader
paramReader.readGlobalConst()
import posReader as pR
import hrReader as reader
from ham import Green

#pR.readPOS('./mcalc/FeCl2/POSCAR')
#pR.readPOS('./mcalc/MoTe2/POSCAR')
#pR.readPOS('./mcalc/MoTe2CrI3/POSCAR')
#pR.readPOS('./mcalc/Cr2P2Se6/POSCAR')
pR.readPOS('./mcalc/FePS3/POSCAR')
hr=reader.Hr()
#hr.readSingleFile('./mcalc/FeCl2/wannier90.dn_hr.dat',ignoreImagPart=True,cutoff=0.01)
#hr.readSingleFile('./mcalc/FeCl2/wannier90_hr.dat',ignoreImagPart=False,cutoff=0.01)
#hr.readSingleFile('./mcalc/MoTe2/wannier90_hr.dat',ignoreImagPart=False,cutoff=-0.1,format="QE")
#hr.readSingleFile('./mcalc/MoTe2CrI3/wannier90_hr.mz.dat',ignoreImagPart=False,cutoff=-0.1,format="QE")
#hr.readDoubleFiles('./mcalc/Cr2P2Se6/wannier90.up_hr.dat','./mcalc/Cr2P2Se6/wannier90.dn_hr.dat',cutoff=0.01)
#hr.readSingleFile('./mcalc/Cr2P2Se6/wannier90.up_hr.dat',cutoff=-0.01)
hr.readSingleFile('./mcalc/FePS3/wannier90_hr.dat',ignoreImagPart=False,cutoff=-0.1)
# data structure atomicOrbIndexList[atom index][Lz][orb index]: give the orb. index corr. to hr.dat
#atomicOrbIndexList=[[[],[],[0,1,2,3,4,11,12,13,14,15]],
#                    [[],[5,6,7,16,17,18]],
#                    [[],[8,9,10,19,20,21]],
#                   ]
#atomicOrbIndexList=[[[],[],[0,1,2,3,4]],
#                    [[],[5,6,7]],
#                    [[],[8,9,10]],
#                   ]
atomicOrbIndexList=[[[],[],[0,1,2,3,4,68,69,70,71,72]],       # Fe1
                    [[],[],[5,6,7,8,9,73,74,75,76,77]],       # Fe2
                    [[],[],[10,11,12,13,14,78,79,80,81,82]],  # Fe3
                    [[],[],[15,16,17,18,19,83,84,85,86,87]],  # Fe4
                    [[],[20,21,22,88,89,90]],                 # P1
                    [[],[23,24,25,91,92,93]],                 # P2
                    [[],[26,27,28,94,95,96]],                 # P3
                    [[],[29,30,31,97,98,99]],                 # P4
                    [[],[32,33,34,100,101,102]],                 # S1
                    [[],[35,36,37,103,104,105]],                 # S2
                    [[],[38,39,40,106,107,108]],                 # S3
                    [[],[41,42,43,109,110,111]],                 # S4
                    [[],[44,45,46,112,113,114]],                 # S5
                    [[],[47,48,49,115,116,117]],                 # S6
                    [[],[50,51,52,118,119,120]],                 # S7
                    [[],[53,54,55,121,122,123]],                 # S8
                    [[],[56,57,58,124,125,126]],                 # S9
                    [[],[59,60,61,127,128,129]],                 # S10
                    [[],[62,63,64,130,131,132]],                 # S11
                    [[],[65,66,67,133,134,135]],                 # S12
                   ]
#atomicOrbIndexList=[[[],[],[0,1,2,3,4,34,35,36,37,38]],
#                    [[],[],[5,6,7,8,9,39,40,41,42,43]],
#                    [[],[10,11,12,44,45,46]],
#                    [[],[13,14,15,47,48,49]],
#                    [[],[16,17,18,50,51,52]],
#                    [[],[19,20,21,53,54,55]],
#                    [[],[22,23,24,56,57,58]],
#                    [[],[25,26,27,59,60,61]],
#                    [[],[28,29,30,62,63,64]],
#                    [[],[31,32,33,65,66,67]],
#                   ]

# data structure atomicMagMomList[atom index]: give the SU(2) rep. of magnetic moments
#atomicMagMomList=[[0,0], # in z direction
#                  [0,0], # 0 mag.mom.
#                  [0,0]  # 0 mag.mom.
#                 ]
atomicMagMomList=[[0,1], # in z direction
                  [1,0], # in -z direction
                  [0,1], # in z direction
                  [1,0], # in -z direction
                  [0,0], # 0 mag.mom.
                  [0,0], # 0 mag.mom.
                  [0,0], # 0 mag.mom.
                  [0,0], # 0 mag.mom.
                  [0,0], # 0 mag.mom.
                  [0,0], # 0 mag.mom.
                  [0,0], # 0 mag.mom.
                  [0,0], # 0 mag.mom.
                  [0,0], # 0 mag.mom.
                  [0,0], # 0 mag.mom.
                  [0,0], # 0 mag.mom.
                  [0,0], # 0 mag.mom.
                  [0,0], # 0 mag.mom.
                  [0,0], # 0 mag.mom.
                  [0,0], # 0 mag.mom.
                  [0,0], # 0 mag.mom.
                 ]


hr.symmerizeBonds(atomicOrbIndexList=atomicOrbIndexList,symOpSelection=[],
                  spinor=True,atomicMagMomList=atomicMagMomList)
#exit()
#hr.adjustSOCStrength(socFactor=0.0)
#hr.adjustInterLayerInteraction(
#    [0,1,2,3,4,72,73,74,75,76,5,6,7,8,9,77,78,79,80,81,10,11,12,82,83,84,13,14,15,85,86,87,16,17,18,88,89,90,19,20,21,91,92,93,22,23,24,94,95,96,25,26,27,97,98,99],
#    [],
#    factor=1.0
#)
#print(hr.onsite)
#exit()
gr=Green()
#gr.calcFullBands_CDLL(hr,fermi=-2.1,mainDir='./mcalc/FeCl2/',saveFig=True,redBands=[17,18])
#gr.calcFullBands_CDLL(hr,fermi=0.0,mainDir='./mcalc/MoTe2/',saveFig=True,redBands=[13,14])
gr.calcFullBands_CDLL(hr,fermi=-2.5,mainDir='./mcalc/FePS3/',saveFig=True,redBands=[91,92])
#gr.calcFullBands_CDLL(hr,fermi=-1.6042,mainDir='./mcalc/Cr2P2Se6/',saveFig=True,redBands=[41,42])