import hrReader as reader
import ham as hamilton
import numpy as np
import posReader as pR
import paramReader
paramReader.readGlobalConst()
import couplingCalc as cc
import matplotlib.pyplot as plt

cc.classifyBonds(mainDir='./mcalc/T-VTe2/HF/', source_list=[0], wann_list=[[0,1,2,3,4]], ini_S=[1.5,0,0],completelyConfine=True)
cc.calcIsotropicJ(mainDir='./mcalc/T-VTe2/HF/',fermi=0.2647,eta=0.0001,emptySpace=5)
exit()

pR.readPOS('./spg156FeTeIU0.0.vasp')
hr_g=reader.Hr()
hr_g.readDoubleFiles('./mcalc/FeSBr/wannier90.dn_hr.dat','./mcalc/FeSBr/wannier90.up_hr.dat')
#hr_g.readSingleFile('./simpleGraphene.txt')
#hr_g.readDoubleFiles('./CrI3.dn_hr.dat','./CrI3.up_hr.dat')
#gr.testG(hr_s,orbList=[0,1],fermi=0,eta=0.05)
#gr=hamilton.Green()
#gr.calcBands(hr_g,orbList=[],spin_polarized=True,fermi=-2.7017,eta=0.1,resolution=200)
#exit()
#scatterList=[[[np.array([0,0,0]),0],[np.array([0,0,0]),1], 0.5],
#             [[np.array([0,0,0]),1],[np.array([0,0,0]),0], 0.5],
#             [[np.array([0,0,0]),0],[np.array([1,0,0]),1], 0.5],
#             [[np.array([1,0,0]),1],[np.array([0,0,0]),0], 0.5],
#             [[np.array([0,0,0]),0],[np.array([0,-1,0]),1], 0.5],
#             [[np.array([0,-1,0]),1],[np.array([0,0,0]),0], 0.5],
#             ]
#scatterList=[[[np.array([0,0,0]),0],[np.array([0,0,0]),2],0.5],
#             [[np.array([0,0,0]),2],[np.array([0,0,0]),0],0.5],
#             [[np.array([1,0,0]),1],[np.array([1,0,0]),3],-0.5],
#             [[np.array([1,0,0]),3],[np.array([1,0,0]),1],-0.5],
#             ]

#gr.calcMatrixPerturbatedDOSViaExact(hr_gd,scatterList,eta=0.01,energyRange=(-3.5,3.5),NEDOS=500,Lx=5,Ly=5,Lz=1)
#gr.calcMatrixPerturbatedDOSViaFourier(hr_gd,scatterList,eta=0.01,energyRange=(-3.5,3.5),NEDOS=500,Lx=5,Ly=5,Lz=1,globalCutOrder=12,diverseCutOrder=2,skipOdds=True)
#E_exact=gr.calcMatrixPerturbatedEnergyViaExact(hr_g,scatterList,eta=0.0001,energyRange=(-10.,0.),Lx=5,Ly=5,Lz=1)
#print(E_exact)
#E_exact=gr.calcMatrixPerturbatedEnergyViaFourier(hr_g,scatterList,eta=0.0001,energyRange=(-5.,0.),Lx=10,Ly=10,Lz=1,globalCutOrder=12,diverseCutOrder=2,skipOdds=True,onlyPerturbatedEnergy=True)
#print(E_exact)

J=gr.calcJ(hr_g,sourceOrbList=[0],sourceR=[0,0,0],targetOrbList=[1],targetR=[0,0,0],eta=0.0001,energyRange=(-5.0,0.),Lx=5,Ly=5,Lz=1,globalCutOrder=8,diverseCutOrder=2,linearPerturb=True)
#J=gr.calcJExact(hr_g,sourceOrbList=[0],sourceR=[0,0,0],targetOrbList=[1],targetR=[0,0,0],eta=0.0001,energyRange=(-5.0,0.),Lx=5,Ly=5,Lz=1)
print(J)

