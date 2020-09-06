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
#pR.createKPOINTS('./',getKMesh=True)
#hr1=reader.Hr('./wannier90.dn_hr.dat')
#hr2=reader.Hr('./wannier90.up_hr.dat')
#kmesh=pR.createKPOINTS(getKMesh=True)

#gr=hamilton.Green()
#gr.calcJ(hr1,hr2,fermi=-0.72,orbList=[0,1,2,3,4],R=[2,0,0])

#gr.calcDOS(hr1,orbList=[0,1,2,3,4],fermi=-0.72,eta=0.05,resolution=90)

hr_s=reader.Hr('./simpleHR.txt')
kmesh=pR.createKPOINTS('./',getKMesh=True)
#print(kmesh)
#gr.testG(hr_s,orbList=[0,1],fermi=0,eta=0.05)
gr=hamilton.Green()
#gr.calcBands(hr_s,orbList=[0,1],fermi=0,eta=0.1,resolution=200)
#gr.calcDOSViaFourier(hr_s,orbList=[0,1],eta=0.01,NEDOS=500)
#gr.calcDOSViaDirectInversion(hr_s,eta=0.01,energyRange=(-3.5,3.5),NEDOS=500,Lx=5,Ly=5)
gr.calcPerturbDOSViaFourier(hr_s,delta=-1,eta=0.01,NEDOS=500)