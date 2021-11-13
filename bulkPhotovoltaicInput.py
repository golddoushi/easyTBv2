import paramReader
paramReader.readGlobalConst()
import posReader as pR
import hrReader as reader
from ham import Green, Response
import numpy as np
import matplotlib.pyplot as plt

pR.readPOS('./mcalc/BN/POSCAR')
hr=reader.Hr()
hr.readSingleFile('./mcalc/BN/hr.dat',ignoreImagPart=False,cutoff=-0.01)#,format="QE")

res=Response()
orbCorrList=np.array([
[ 2/3,1/3,0 ],
[ 0/3,0/3,0 ],
[ 0/3,0/3,0 ],
])
res.calcBulkPhotovoltaicConductor_CDLL(hr,mainDir='./mcalc/BN/',orbCorrList=orbCorrList,kgrid=[100,100,1],omegaList=np.linspace(0.5,3,20),tau=0.02,erange=[-80,80],nthreads=4)