import paramReader
paramReader.readGlobalConst()
import posReader as pR
import hrReader as reader
from ham import Response
import numpy as np

pR.readPOS('./mcalc/FeCl2/POSCAR')
hr=reader.Hr()
hr.readSingleFile('./mcalc/FeCl2/wannier90.dn_hr.dat',ignoreImagPart=False,cutoff=-0.01)#,format="QE")
hr.readSingleRFile('./mcalc/FeCl2/wannier90.dn_r.dat',ignoreImagPart=True,cutoff=-0.01)#,format="QE")

res=Response()
orbCoorList=np.array([
[ 2/3,1/3,0 ],
[ 2/3,1/3,0 ],
[ 2/3,1/3,0 ],
[ 1/3,2/3,0 ],
[ 1/3,2/3,0 ],
[ 1/3,2/3,0 ],
])

orbCoorList=np.array(hr.rList).dot(np.linalg.inv(pR.lat))
res.calcOpticalHoppingTensor_CDLL(hr,mainDir='./mcalc/FeCl2/',fermi=-2.3,nk=50,orbCorrList=orbCoorList,ibandv=6,ibandc=7,nthreads=4)