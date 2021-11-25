import numpy as np
import paramReader
paramReader.readGlobalConst()
import posReader as pR
import hrReader as reader
from ham import Green

pR.readPOS('./mcalc/FeCl2/POSCAR')
hr=reader.Hr()
hr.readSingleFile('./mcalc/FeCl2/wannier90.dn_hr.dat',ignoreImagPart=False,cutoff=0.01)
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
#print(hr.onsite)
#exit()
gr=Green()
gr.calcFullBands_CDLL(hr,fermi=-2.2,mainDir='./mcalc/FeCl2/',saveFig=True)