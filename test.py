import hrReader as reader
import ham as hamilton
import numpy as np
import posReader as pR
import paramReader
paramReader.readGlobalConst()
import couplingCalc as cc
import matplotlib.pyplot as plt

#cc.classifyBonds(mainDir='./mcalc/FeTeI/a_0.05U3/', source_list=[0], wann_list=[[0,1,2,3,4]], ini_S=[1.5,0,0],completelyConfine=True)
#cc.calcIsotropicJ(mainDir='./mcalc/FeTeI/a_0.05U3/',fermi=-0.0353,eta=0.0001,emptySpace=5)
#exit()

pR.readPOS('./mcalc/CrI3/soc/scf_mz/POSCAR')
hr_g=reader.Hr()
#hr_g.readDoubleFiles('./mcalc/spinchain/hr.up.dat','./mcalc/spinchain/hr.dn.dat')
hr_g.readSingleFile('./mcalc/CrI3/soc/scf_mz/wannier90_hr.dat')
gr=hamilton.Green()
#gr.calcBands(hr_g,orbList=[],spin_polarized=True,fermi=0.0,eta=0.05,mainDir='./mcalc/spinchain/',resolution=100)
#L=20
#kmesh=[np.array([k1,0,0]) for k1 in np.linspace(0,1,L,endpoint=False)]
#gr.calcFullDOS(hr_g,fermi=0.7,eta=0.0001,NEDOS=1000,kmesh=kmesh,mainDir='./mcalc/spinchain/')
#gr.calcFullBands(hr_g,fermi=0.7,mainDir='./mcalc/spinchain/')
#exit()
#scatterList=[[[np.array([0,0,0]),0],[np.array([0,0,0]),1], 0.5],
#             [[np.array([0,0,0]),1],[np.array([0,0,0]),0], 0.5],
#             [[np.array([0,0,0]),0],[np.array([1,0,0]),1], 0.5],
#             [[np.array([1,0,0]),1],[np.array([0,0,0]),0], 0.5],
#             [[np.array([0,0,0]),0],[np.array([0,-1,0]),1], 0.5],
#             [[np.array([0,-1,0]),1],[np.array([0,0,0]),0], 0.5],
#             ]
#theta=np.pi/2
#scatterList=[[[np.array([0,0,0]),0],[np.array([0,0,0]),2],1.0*np.sin(theta)],
#             [[np.array([0,0,0]),2],[np.array([0,0,0]),0],1.0*np.sin(theta)],
#             [[np.array([0,0,0]),0],[np.array([0,0,0]),0],1.0*(np.cos(theta)-1)],
#             [[np.array([0,0,0]),2],[np.array([0,0,0]),2],-1.0*(np.cos(theta)-1)],
#             [[np.array([1,0,0]),0],[np.array([1,0,0]),2],1.0*np.sin(theta)],
#             [[np.array([1,0,0]),2],[np.array([1,0,0]),0],1.0*np.sin(theta)],
#             [[np.array([1,0,0]),0],[np.array([1,0,0]),0],1.0*(np.cos(theta)-1)],
#             [[np.array([1,0,0]),2],[np.array([1,0,0]),2],-1.0*(np.cos(theta)-1)],
#             ]

#gr.calcMatrixPerturbatedDOSViaExact(hr_g,scatterList,eta=0.01,energyRange=(-7.5,0),NEDOS=500,Lx=5,Ly=5,Lz=1)
#gr.calcMatrixPerturbatedDOSViaFourier(hr_gd,scatterList,eta=0.01,energyRange=(-3.5,3.5),NEDOS=500,Lx=5,Ly=5,Lz=1,globalCutOrder=12,diverseCutOrder=2,skipOdds=True)
#E_exact=gr.calcMatrixPerturbatedEnergyViaExact(hr_g,scatterList,eta=0.0001,energyRange=(-10.,0.),Lx=5,Ly=5,Lz=1)
#print(E_exact)
E_fourier=gr.calcMatrixPerturbatedEnergyViaFourier(hr_g,[],eta=0.0001,energyRange=(-12.,-1.8),Lx=6,Ly=6,Lz=1,globalCutOrder=0,diverseCutOrder=2,skipOdds=False,onlyPerturbatedEnergy=False)
print(E_fourier)

#J=gr.calcJ(hr_g,sourceOrbList=[0,1,2,3,4],sourceR=[0,0,0],targetOrbList=[5,6,7,8,9],targetR=[0,0,0],eta=0.0001,energyRange=(-12.0,-1.8),Lx=5,Ly=5,Lz=1,globalCutOrder=8,diverseCutOrder=2,linearPerturb=False)
#J=gr.calcJExact(hr_g,sourceOrbList=[0],sourceR=[0,0,0],targetOrbList=[0],targetR=[2,0,0],eta=0.0001,energyRange=(-5.0,0),Lx=30,Ly=1,Lz=1)
#print(J*1000)

'''
L=10
JList=[]
for a in range(1,6):
    J=gr.calcJExact(hr_g,sourceOrbList=[0],sourceR=[0,0,0],targetOrbList=[0],targetR=[a,0,0],eta=0.0001,energyRange=(-5.0,0.7),Lx=L,Ly=1,Lz=1)
    JList.append(J*1000)
for iJ, J in enumerate(JList):
    print('%d %.6f'%(iJ+1,J))
'''
'''
#J=gr.calcJExact(hr_g,sourceOrbList=[0],sourceR=[0,0,0],targetOrbList=[0],targetR=[1,0,0],eta=0.0001,energyRange=(-5.0,0.7),Lx=20,Ly=1,Lz=1)
#print(J*1000)
#J=gr.calcJExactWithTheta(hr_g,sourceOrbList=[0],sourceR=[0,0,0],targetOrbList=[0],targetR=[1,0,0],theta=np.pi/10,eta=0.0001,energyRange=(-5.0,0.7),Lx=20,Ly=1,Lz=1)
J=gr.calcJExactFourStateWithTheta(hr_g,sourceOrbList=[0],sourceR=[0,0,0],targetOrbList=[0],targetR=[1,0,0],theta=np.pi/2,eta=0.0001,energyRange=(-5.0,0.7),Lx=20,Ly=1,Lz=1)
exit()
JList=[]
NTheta=10
dtheta=np.pi/NTheta
for a in range(1,NTheta):
    J=gr.calcJExactFourStateWithTheta(hr_g,sourceOrbList=[0],sourceR=[0,0,0],targetOrbList=[0],targetR=[1,0,0],theta=a*dtheta,eta=0.0001,energyRange=(-5.0,0.7),Lx=20,Ly=1,Lz=1)
    JList.append(J*1000)
for iJ, J in enumerate(JList):
    print('%d %.6f'%(iJ+1,J))
'''
