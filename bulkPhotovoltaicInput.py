import paramReader
paramReader.readGlobalConst()
import posReader as pR
import hrReader as reader
from ham import Green, Response
import numpy as np
import matplotlib.pyplot as plt

pR.readPOS('./mcalc/FeCl2/POSCAR')
#pR.readPOS('./mcalc/Cr2P2Se6/POSCAR')
hr=reader.Hr()
#hr.readSingleFile('./mcalc/FeCl2/wannier90.dn_hr.dat',ignoreImagPart=True,cutoff=-0.01)#,format="QE")
hr.readSingleFile('./mcalc/FeCl2/wannier90_hr.dat',ignoreImagPart=False,cutoff=-0.01)
#hr.readSingleFile('./mcalc/MoTe2/wannier90_hr.dat',ignoreImagPart=False,cutoff=0.01,format="QE")
#hr.readDoubleFiles('./mcalc/Cr2P2Se6/wannier90.up_hr.dat','./mcalc/Cr2P2Se6/wannier90.dn_hr.dat',cutoff=-0.1)

res=Response()
# data structure atomicOrbIndexList[atom index][Lz][orb index]: give the orb. index corr. to hr.dat
atomicOrbIndexList=[[[],[],[0,1,2,3,4,11,12,13,14,15]],
                    [[],[5,6,7,16,17,18]],
                    [[],[8,9,10,19,20,21]],
                   ]
'''
atomicOrbIndexList=[[[],[],[0,1,2,3,4,34,35,36,37,38]],
                    [[],[],[5,6,7,8,9,39,40,41,42,43]],
                    [[],[10,11,12,44,45,46]],
                    [[],[13,14,15,47,48,49]],
                    [[],[16,17,18,50,51,52]],
                    [[],[19,20,21,53,54,55]],
                    [[],[22,23,24,56,57,58]],
                    [[],[25,26,27,59,60,61]],
                    [[],[28,29,30,62,63,64]],
                    [[],[31,32,33,65,66,67]],
                   ]
'''
# data structure atomicMagMomList[atom index]: give the SU(2) rep. of magnetic moments
atomicMagMomList=[[0,1], # in z direction
                  [0,0], # 0 mag.mom.
                  [0,0]  # 0 mag.mom.
                 ]

hr.symmerizeBonds(atomicOrbIndexList=atomicOrbIndexList,symOpSelection=[],
                  spinor=True,atomicMagMomList=atomicMagMomList)
#hr.adjustSOCStrength(socFactor=1.0)
#exit()
res.calcBulkPhotovoltaicConductor_CDLL(hr,mainDir='./mcalc/FeCl2/',atomicOrbIndexList=atomicOrbIndexList,kgrid=[100,100,1],omegaList=np.linspace(0.5,1.2,20),tau=0.001,erange=[-80,80],nthreads=4)
#res.calcBulkPhotovoltaicConductor_CDLL(hr,mainDir='./mcalc/Cr2P2Se6/',atomicOrbIndexList=atomicOrbIndexList,kgrid=[50,50,1],omegaList=np.linspace(0.3,1.2,20),tau=0.01,erange=[-80,80],nthreads=4)