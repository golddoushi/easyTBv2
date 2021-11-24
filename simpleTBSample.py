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
[ 2/3,1/3,0 ],
[ 2/3,1/3,0 ],
[ 2/3,1/3,0 ],
[ 2/3,1/3,0 ],
[ 2/3,1/3,0 ],
[ 2/3,1/3,0 ],
])
O1=np.array([ # rotate by 2pi/3
                    [ 0, 1, 0],
                    [-1,-1, 0],
                    [ 0, 0, 1],
                    ])
O2=np.dot(O1,O1) # rotate by 4pi/3
hr.symmerizeBonds(orbCoorList=orbCoorList,symOpList=[O1,O2])
#print(hr.onsite)
#exit()
gr=Green()
gr.calcFullBands_CDLL(hr,fermi=-2.2,mainDir='./mcalc/FeCl2/',saveFig=True)