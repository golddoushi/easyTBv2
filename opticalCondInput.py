import paramReader
paramReader.readGlobalConst()
import posReader as pR
import hrReader as reader
from ham import Response
import numpy as np

pR.readPOS('./mcalc/FeCl2/POSCAR')
hr=reader.Hr()
#hr.readSingleFile('./mcalc/FeCl2/wannier90.dn_hr.dat',ignoreImagPart=True,cutoff=-0.01)#,format="QE")
hr.readSingleFile('./mcalc/FeCl2/wannier90_hr.dat',ignoreImagPart=False,cutoff=-0.01)

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
#symOpSelection=[0,1,2,3,4,5] # chose several sym. ops.
hr.symmerizeBonds(atomicOrbIndexList=atomicOrbIndexList,symOpSelection=[],
                  spinor=True,pgFilePath='./DPGREP/27.txt',atomicMagMomList=atomicMagMomList)

res.calcOpticalHoppingTensor_CDLL(hr,mainDir='./mcalc/FeCl2/',fermi=-2.3,nk=100,atomicOrbIndexList=atomicOrbIndexList,ibandv=17,ibandc=18,nthreads=4)
#res.calcOpticalHoppingTensor_Py(hr,mainDir='./mcalc/FeCl2/',fermi=-2.3,nk=50,atomicOrbIndexList=atomicOrbIndexList,ibandv=6,ibandc=7,nthreads=1)