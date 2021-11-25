import paramReader
paramReader.readGlobalConst()
import posReader as pR
import hrReader as reader
from ham import Green, Response
import numpy as np
import matplotlib.pyplot as plt

pR.readPOS('./mcalc/FeCl2/POSCAR')
hr=reader.Hr()
hr.readSingleFile('./mcalc/FeCl2/wannier90.dn_hr.dat',ignoreImagPart=False,cutoff=0.01)#,format="QE")
hr.readSingleRFile('./mcalc/FeCl2/wannier90.dn_r.dat',ignoreImagPart=True,cutoff=-0.01)#,format="QE")

res=Response()
orbCoorList=np.array([
[ 0/3,0/3,0 ],
[ 0/3,0/3,0 ],
[ 0/3,0/3,0 ],
[ 0/3,0/3,0 ],
[ 0/3,0/3,0 ],
[ 2/3,1/3,0.1 ],
[ 2/3,1/3,0.1 ],
[ 2/3,1/3,0.1 ],
[ 2/3,1/3,-0.1 ],
[ 2/3,1/3,-0.1 ],
[ 2/3,1/3,-0.1 ],
])
# data structure atomicOrbIndexList[atom index][l][orb index]
atomicOrbIndexList=[[[],[],[0,1,2,3,4]],
                    [[],[5,6,7]],
                    [[],[8,9,10]]
                   ]

hr.symmerizeBonds(orbCoorList=orbCoorList,atomicOrbIndexList=atomicOrbIndexList)
res.calcBulkPhotovoltaicConductor_CDLL(hr,mainDir='./mcalc/FeCl2/',orbCorrList=orbCoorList,kgrid=[100,100,1],omegaList=np.linspace(0.5,1.2,20),tau=0.01,erange=[-80,80],nthreads=4)