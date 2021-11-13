import paramReader
paramReader.readGlobalConst()
import posReader as pR
import hrReader as reader
from ham import Response
import numpy as np

pR.readPOS('./mcalc/BN/POSCAR')
hr=reader.Hr()
hr.readSingleFile('./mcalc/BN/hr_biorbitals.dat',ignoreImagPart=False,cutoff=-0.01)

res=Response()
orbCoorList=np.array([
[2/3,1/3,0],
[0,0,0]
])
res.calcOpticalHoppingTensor_CDLL(hr,mainDir='./mcalc/BN/',fermi=0,nk=50,orbCorrList=orbCoorList,ibandv=0,ibandc=1,nthreads=4)