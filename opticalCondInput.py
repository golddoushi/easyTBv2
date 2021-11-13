import paramReader
paramReader.readGlobalConst()
import posReader as pR
import hrReader as reader
from ham import Response
import numpy as np

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
res.calcOpticalHoppingTensor_CDLL(hr,mainDir='./mcalc/MoTe2/',fermi=0,nk=50,orbCorrList=orbCoorList,ibandv=8,ibandc=9,nthreads=4)