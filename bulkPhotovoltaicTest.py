import paramReader
paramReader.readGlobalConst()
import posReader as pR
import hrReader as reader
from ham import Green, Response
import numpy as np
import matplotlib.pyplot as plt
from illustrator import drawHotmapOnHexagonalBZ

pR.readPOS('./mcalc/BN/POSCAR')
hr=reader.Hr()
hr.readSingleFile('./mcalc/BN/hr.dat',ignoreImagPart=False,cutoff=-0.01)

res=Response()
# data structure atomicOrbIndexList[atom index][Lz][orb index]: give the orb. index corr. to hr.dat
atomicOrbIndexList=[[[0]],
                    [[1,2]],
                   ]

# data structure atomicMagMomList[atom index]: give the SU(2) rep. of magnetic moments
atomicMagMomList=[
#    [0,1], # in z direction
#    [0,0], # 0 mag.mom.
#    [0,0]  # 0 mag.mom.
                 ]

#hr.symmerizeBonds(atomicOrbIndexList=atomicOrbIndexList,symOpSelection=[],
#                  spinor=False,atomicMagMomList=atomicMagMomList)
#hr.adjustSOCStrength(socFactor=1.0)

Lk=50
eigs_addr,vecs_addr=res.calcEigsAndVecsInMemory_CDLL(hr,kmesh=[Lk,Lk,1],nthreads=1)
dEdkx_addr,dEdky_addr,dEdkz_addr=res.derivateQuantitiesOnKGrid_CDLL(eigs_addr,hr.norb,[Lk,Lk,1],1)
#exit()
data=res.redOutQuantitiesOnKGrid(dEdkx_addr,hr.norb,[Lk,Lk,1])
#exit()
drawHotmapOnHexagonalBZ('./mcalc/BN/',data[:,-1],[Lk,Lk,1])
exit()
#res.calcInjectPhotovoltaicConductor_CDLL(hr,mainDir='./mcalc/BN/',atomicOrbIndexList=atomicOrbIndexList,kgrid=[50,50,1],omegaList=np.linspace(1.0,3.0,20),tau=0.01,erange=[-80,80],nthreads=1)
#res.calcShiftPhotovoltaicConductor_CDLL(hr,mainDir='./mcalc/BN/',atomicOrbIndexList=atomicOrbIndexList,kgrid=[50,50,1],omegaList=np.linspace(1.0,3.0,20),tau=0.01,erange=[-80,80],nthreads=1)
#res.calcBulkPhotovoltaicConductor_CDLL(hr,mainDir='./mcalc/BN/',atomicOrbIndexList=atomicOrbIndexList,kgrid=[50,50,1],omegaList=np.linspace(1.0,3.0,20),tau=0.01,erange=[-80,80],nthreads=1)
