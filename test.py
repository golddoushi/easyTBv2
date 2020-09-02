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
#kpath=pR.autoGenerateKpath2D(onlyGetKPTList=True)
#eigs=[]
pR.createKPOINTS('./',getKMesh=True)
hr1=reader.Hr('./wannier90.dn_hr.dat')
kmesh=pR.createKPOINTS(getKMesh=True)

gr=hamilton.Green()
gr.calcBulkSusceptibility(hr1,kmesh,fermi=-0.72,a=0,b=5,R=[-1,0,0])
