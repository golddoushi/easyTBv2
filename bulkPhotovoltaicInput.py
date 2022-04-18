import paramReader
paramReader.readGlobalConst()
import posReader as pR
import hrReader as reader
from ham import Green, Response
import numpy as np
import matplotlib.pyplot as plt

pR.readPOS('./mcalc/BN/POSCAR')
#pR.readPOS('./mcalc/BN/POSCAR.square.vasp')
#pR.readPOS('./mcalc/FeCl2/POSCAR')
#pR.readPOS('./mcalc/MoTe2/POSCAR')
#pR.readPOS('./mcalc/Cr2P2Se6/POSCAR')
hr=reader.Hr()
hr.readSingleFile('./mcalc/BN/hr.dat',ignoreImagPart=True,cutoff=-0.01)
#hr.readSingleFile('./mcalc/BN/hr.square.dat',ignoreImagPart=True,cutoff=-0.01)
#hr.readSingleFile('./mcalc/FeCl2/wannier90.dn_hr.dat',ignoreImagPart=True,cutoff=-0.01)#,format="QE")
#hr.readSingleFile('./mcalc/FeCl2/wannier90_hr.dat',ignoreImagPart=False,cutoff=-0.1)
#hr.readSingleFile('./mcalc/MoTe2/wannier90_hr.dat',ignoreImagPart=False,cutoff=0.01,format="QE")
#hr.readDoubleFiles('./mcalc/Cr2P2Se6/wannier90.up_hr.dat','./mcalc/Cr2P2Se6/wannier90.dn_hr.dat',cutoff=-0.1)

res=Response()
# data structure atomicOrbIndexList[atom index][Lz][orb index]: give the orb. index corr. to hr.dat
#'''
# BN
atomicOrbIndexList=[
    [[0,2]],
    [[1]]
]
atomicMagMomList=[[0,0], # 0 mag.mom.
                  [0,0], # 0 mag.mom.
                 ]
#'''
'''
# square BN
atomicOrbIndexList=[
    [[0,4]],
    [[1,5]],
    [[2]],
    [[3]]
]
atomicMagMomList=[[0,0], # 0 mag.mom.
                  [0,0], # 0 mag.mom.
                  [0,0], # 0 mag.mom.
                  [0,0], # 0 mag.mom.
                 ]
'''
'''
# FeCl2 down spin
atomicOrbIndexList=[[[],[],[0,1,2,3,4]],
                    [[],[5,6,7]],
                    [[],[8,9,10]],
                   ]
atomicMagMomList=[[0,1], # in z direction
                  [0,0], # 0 mag.mom.
                  [0,0]  # 0 mag.mom.
                 ]
'''
'''
# FeCl2
atomicOrbIndexList=[[[],[],[0,1,2,3,4,11,12,13,14,15]],
                    [[],[5,6,7,16,17,18]],
                    [[],[8,9,10,19,20,21]],
                   ]
atomicMagMomList=[[0,1], # in z direction
                  [0,0], # 0 mag.mom.
                  [0,0]  # 0 mag.mom.
                 ]
'''
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
#                  spinor=False,saxis=np.array([0,0,1]),atomicMagMomList=atomicMagMomList)
#atomicOrbIndexList=hr.make_supercell([1,0,0],[1,2,0],[0,0,1],atomicOrbIndexList=atomicOrbIndexList,soc=False)
#hr.adjustSOCStrength(socFactor=1.0)
# NOTE check the fermi in GlobalConst first
#res.calcBulkPhotovoltaicConductor_CDLL(hr,mainDir='./mcalc/FeCl2/',atomicOrbIndexList=atomicOrbIndexList,kgrid=[40,40,1],omegaList=np.linspace(0.5,1.2,29),tau=0.02,erange=[-80,80],ndivide=4,nthreads=4)
#res.calcBulkPhotovoltaicConductor_CDLL(hr,mainDir='./mcalc/Cr2P2Se6/',atomicOrbIndexList=atomicOrbIndexList,kgrid=[50,50,1],omegaList=np.linspace(0.3,1.2,20),tau=0.01,erange=[-80,80],nthreads=4)
#res.calcSpinProjectiveBulkPhotovoltaicConductor_CDLL(hr,mainDir='./mcalc/FeCl2/',atomicOrbIndexList=atomicOrbIndexList,proj_atomID=[[0,1,2],[0,1,2]],kgrid=[50,50,1],omegaList=np.linspace(0.5,1.2,20),tau=0.02,erange=[-80,80],nthreads=4)
#res.calcShiftPhotovoltaicConductor_CDLL(hr,mainDir='./mcalc/MoTe2/',atomicOrbIndexList=atomicOrbIndexList,kgrid=[50,50,1],omegaList=np.linspace(0.8,2.0,20),tau=0.02,erange=[-80,80],nthreads=4)
res.calcBulkPhotovoltaicConductor_CDLL(hr,mainDir='./mcalc/BN/',atomicOrbIndexList=atomicOrbIndexList,kgrid=[50,50,1],omegaList=np.linspace(0.5,2.5,29),tau=0.02,erange=[-80,80],ndivide=1,nthreads=4)
exit()

res.calcProjectiveBulkPhotovoltaicConductor_CDLL(hr,
    mainDir='./mcalc/FeCl2/',atomicOrbIndexList=atomicOrbIndexList,kgrid=[50,50,1],omegaList=np.linspace(0.5,1.2,29),tau=0.02,erange=[-80,80],
    proj_op=[
        [[0,1,2],[0,1,2]], # full vel
        [[0],[0]], # Fe-Fe vel
        [[0],[1,2]], # Fe-Cl vel
        [[1,2],[1,2]], # Cl-Cl vel
        ],
    scatter_route=[
        [2,0,3],
        [0,2,3],
        ],
    ndivide=4,nthreads=4)