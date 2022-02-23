import paramReader
paramReader.readGlobalConst()
import posReader as pR
import hrReader as reader
from ham import Green, Response
import numpy as np
import matplotlib.pyplot as plt

pR.readPOS('./mcalc/FeCl2/POSCAR')
hr=reader.Hr()
#hr.readSingleFile('./mcalc/FeCl2/wannier90.dn_hr.dat',ignoreImagPart=True,cutoff=-0.01)#,format="QE")
hr.readSingleFile('./mcalc/FeCl2/wannier90_hr.dat',ignoreImagPart=False,cutoff=-0.01)
#hr.readSingleFile('./mcalc/MoTe2/wannier90_hr.dat',ignoreImagPart=False,cutoff=0.01,format="QE")

res=Response()
# data structure atomicOrbIndexList[atom index][Lz][orb index]: give the orb. index corr. to hr.dat
atomicOrbIndexList=[[[],[],[0,1,2,3,4,11,12,13,14,15]],
                    [[],[5,6,7,16,17,18]],
                    [[],[8,9,10,19,20,21]],
                   ]
# data structure atomicMagMomList[atom index]: give the SU(2) rep. of magnetic moments
atomicMagMomList=[[0,1], # in z direction
                  [0,0], # 0 mag.mom.
                  [0,0]  # 0 mag.mom.
                 ]

hr.symmerizeBonds(atomicOrbIndexList=atomicOrbIndexList,symOpSelection=[],
                  spinor=True,atomicMagMomList=atomicMagMomList)
hr.adjustSOCStrength(socFactor=1.0)
#exit()
res.calcBulkPhotovoltaicConductor_CDLL(hr,mainDir='./mcalc/FeCl2/',atomicOrbIndexList=atomicOrbIndexList,kgrid=[100,100,1],omegaList=np.linspace(0.5,1.2,20),tau=0.01,erange=[-80,80],nthreads=4)