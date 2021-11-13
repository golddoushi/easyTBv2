import paramReader
paramReader.readGlobalConst()
import posReader as pR
import hrReader as reader
from ham import Green, Response
import numpy as np
import matplotlib.pyplot as plt

pR.readPOS('./mcalc/MoTe2/POSCAR')
hr=reader.Hr()
hr.readSingleFile('./mcalc/MoTe2/wannier90_hr.dat',ignoreImagPart=False,cutoff=-0.01)#,format="QE")
#hr.readSingleRFile('./mcalc/MoTe2/wannier90_r.dat',ignoreImagPart=True,cutoff=-0.01)#,format="QE")

res=Response()
orbCoorList=np.array([
[ 1/3,2/3,0 ],
[ 1/3,2/3,0 ],
[ 1/3,2/3,0 ],
[ 1/3,2/3,0 ],
[ 1/3,2/3,0 ],
[ 2/3,1/3,0 ],
[ 2/3,1/3,0 ],
[ 2/3,1/3,0 ],
[ 2/3,1/3,0 ],
[ 2/3,1/3,0 ],
[ 2/3,1/3,0 ],
[ 2/3,1/3,0 ],
[ 2/3,1/3,0 ],
])
#orbCoorList=np.array(hr.rList).dot(np.linalg.inv(pR.lat))
#for orb in orbCoorList:
#    print(orb)
res.calcBulkPhotovoltaicConductor_CDLL(hr,mainDir='./mcalc/MoTe2/',orbCorrList=orbCoorList,kgrid=[200,200,1],omegaList=np.linspace(0.5,3,20),tau=0.01,erange=[-80,80],nthreads=4)