import hrReader as reader
import ham as hamilton
import numpy as np
import posReader as pR
import paramReader
paramReader.readGlobalConst()
import matplotlib.pyplot as plt

pR.readPOS('./spg156FeTeIU0.0.vasp')
#print(pR.lat)
#print(pR.autoGenerateKpath2D(onlyGetKPTList=True))
#exit()
kpath=pR.autoGenerateKpath2D(onlyGetKPTList=True)
#eigs=[]
hr1=reader.Hr('./wannier90.dn_hr.dat')
for ikpt, kpt in enumerate(kpath):
    ham = hamilton.Ham()
    ham.buildHkFromHr(hr1,kpt)
    eigs=ham.sovle_exact(vec=False)
    plt.scatter([ikpt]*len(eigs),eigs,c='black')
#kdistance=np.array([i for i in range(len(kpath))])
#eigs=np.array(eigs)
#plt.ylim(-4,1)
plt.show()
