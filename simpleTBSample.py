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
pR.readPOS('./mcalc/MnPS3/POSCAR')
#pR.readPOS('./mcalc/BN/POSCAR')
hr=reader.Hr()
#hr.readSingleFile('./mcalc/FeCl2/wannier90.dn_hr.dat',ignoreImagPart=True,cutoff=0.01)
#hr.readSingleFile('./mcalc/FeCl2/wannier90_hr.dat',ignoreImagPart=False,cutoff=0.01)
#hr.readSingleFile('./mcalc/MoTe2/wannier90_hr.dat',ignoreImagPart=False,cutoff=-0.1,format="QE")
#hr.readSingleFile('./mcalc/MoTe2CrI3/wannier90_hr.mz.dat',ignoreImagPart=False,cutoff=-0.1,format="QE")
#hr.readDoubleFiles('./mcalc/Cr2P2Se6/wannier90.up_hr.dat','./mcalc/Cr2P2Se6/wannier90.dn_hr.dat',cutoff=0.01)
#hr.readSingleFile('./mcalc/Cr2P2Se6/wannier90.up_hr.dat',cutoff=-0.01)
hr.readSingleFile('./mcalc/MnPS3/wannier90_hr.dat',ignoreImagPart=False,cutoff=-0.1)
#hr.readSingleFile('./mcalc/BN/hr.soc.dat',ignoreImagPart=False,cutoff=-0.1)
# data structure atomicOrbIndexList[atom index][Lz][orb index]: give the orb. index corr. to hr.dat
#atomicOrbIndexList=[[[0,2]],
#                    [[1,3]],
#                   ]
#'''
atomicOrbIndexList=[[[],[],[0,1,2,3,4,34,35,36,37,38]],       # Mn1
                    [[],[],[5,6,7,8,9,39,40,41,42,43]],       # Mn2
                    [[],[10,11,12,44,45,46]],                 # P1
                    [[],[13,14,15,47,48,49]],                 # P2
                    [[],[16,17,18,50,51,52]],                 # S1
                    [[],[19,20,21,53,54,55]],                 # S2
                    [[],[22,23,24,56,57,58]],                 # S3
                    [[],[25,26,27,59,60,61]],                 # S4
                    [[],[28,29,30,62,63,64]],                 # S5
                    [[],[31,32,33,65,66,67]],                 # S6
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
#'''
# data structure atomicMagMomList[atom index]: give the SU(2) rep. of magnetic moments
#atomicMagMomList=[[0,1], # in z direction
#                  [1,0], # in -z
#                 ]
#'''
atomicMagMomList=[[0,1], # in z direction
                  [1,0], # in -z direction
                  [0,0], # 0 mag.mom.
                  [0,0], # 0 mag.mom.
                  [0,0], # 0 mag.mom.
                  [0,0], # 0 mag.mom.
                  [0,0], # 0 mag.mom.
                  [0,0], # 0 mag.mom.
                  [0,0], # 0 mag.mom.
                  [0,0], # 0 mag.mom.
                 ]
#'''

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
gr.calcFullBands_CDLL(hr,fermi=-2.5,mainDir='./mcalc/MnPS3/',saveFig=True,redBands=[42,43])
#gr.calcFullBands_CDLL(hr,fermi=-1.6042,mainDir='./mcalc/Cr2P2Se6/',saveFig=True,redBands=[41,42])
#gr.calcFullBands_CDLL(hr,fermi=0,mainDir='./mcalc/BN/',saveFig=True,redBands=[1,2])