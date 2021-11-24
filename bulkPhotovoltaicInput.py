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
[ 2/3,1/3,0 ],
[ 2/3,1/3,0 ],
[ 2/3,1/3,0 ],
[ 2/3,1/3,0 ],
[ 2/3,1/3,0 ],
[ 2/3,1/3,0 ],
])
hr.symmerizeBonds(orbCoorList=orbCoorList)
#print(hr.onsite)
#exit()
#orbCoorList=np.array(hr.rList).dot(np.linalg.inv(pR.lat))
#for orb in orbCoorList:
#    print(orb)
#exit()
#exit()
#orbCoorList=np.array(hr.rList).dot(np.linalg.inv(pR.lat))
#for orb in orbCoorList:
#    print(orb)
res.calcBulkPhotovoltaicConductor_CDLL(hr,mainDir='./mcalc/FeCl2/',orbCorrList=orbCoorList,kgrid=[100,100,1],omegaList=np.linspace(0.5,1.2,20),tau=0.01,erange=[-80,80],nthreads=4)