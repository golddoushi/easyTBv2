import paramReader
paramReader.readGlobalConst()
import posReader as pR
import hrReader as reader
from ham import Green, Response
import numpy as np
import matplotlib.pyplot as plt


pR.readPOS('./mcalc/Pt/POSCAR')
hr=reader.Hr()
hr.readSingleFile('./mcalc/Pt/wannier90_hr.dat',ignoreImagPart=False,cutoff=-0.1)
res=Response()
# data structure atomicOrbIndexList[atom index][Lz][orb index]: give the orb. index corr. to hr.dat
#'''
# Pt
atomicOrbIndexList=[[[0,9],[1,2,3,10,11,12],[4,5,6,7,8,13,14,15,16,17]],
                   ]
atomicMagMomList=[[0,0]  # 0 mag.mom.
                 ]
#'''
'''
# MoTe2
atomicOrbIndexList=[[[],[],[0,1,2,3,4,11,12,13,14,15]],       # Mo
                    [[],[5,6,7,16,17,18]],                 # Te1
                    [[],[8,9,10,19,20,21]],                 # Te2
                   ]
atomicMagMomList=[[0,0], # 0 mag.mom.
                  [0,0], # 0 mag.mom.
                  [0,0], # 0 mag.mom.
                 ]
'''
'''
# MnPS3
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
'''
'''
# CrI3
atomicOrbIndexList=[[[],[],[0,1,2,3,4,56,57,58,59,60]],       # Cr1
                    [[],[],[5,6,7,8,9,61,62,63,64,65]],       # Cr2
                    [[],[],[10,11,12,13,14,66,67,68,69,70]],       # Cr3
                    [[],[],[15,16,17,18,19,71,72,73,74,75]],       # Cr4
                    [[],[20,21,22,76,77,78]],                 # I1
                    [[],[23,24,25,79,80,81]],                 # I2
                    [[],[26,27,28,82,83,84]],                 # I3
                    [[],[29,30,31,85,86,87]],                 # I4
                    [[],[32,33,34,88,89,90]],                 # I5
                    [[],[35,36,37,91,92,93]],                 # I6
                    [[],[38,39,40,94,95,96]],                 # I7
                    [[],[41,42,43,97,98,99]],                 # I8
                    [[],[44,45,46,100,101,102]],                 # I9
                    [[],[47,48,49,103,104,105]],                 # I10
                    [[],[50,51,52,106,107,108]],                 # I11
                    [[],[53,54,55,109,110,111]],                 # I12
                   ]
atomicMagMomList=[[0,1], # in z direction
                  [1,0], # in z direction
                  [0,1], # in -z direction
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
                 ]
'''

#hr.symmerizeBonds(atomicOrbIndexList=atomicOrbIndexList,symOpSelection=[],
#                  spinor=True,saxis=np.array([0,0,-1]),atomicMagMomList=atomicMagMomList)
#exit()
#atomicOrbIndexList=hr.make_supercell([1,0,0],[1,2,0],[0,0,1],atomicOrbIndexList=atomicOrbIndexList,soc=False)
#hr.adjustSOCStrength(socFactor=1.0)
# NOTE check the fermi in GlobalConst first
res.calcProjectiveSpinHallConductor_CDLL(hr,
    mainDir='./mcalc/Pt/',atomicOrbIndexList=atomicOrbIndexList,kgrid=[20,20,20],fermiList=np.linspace(8.9469,12.9469,21),erange=[-100,100],
    proj_op=[
        [[0],[0]], # full vel
        ],
    scatter_route=[
        [0,0,0],
        ],
    ndivide=40,nthreads=4)