import hrReader as reader
import ham as hamilton
import numpy as np
import posReader as pR
import paramReader
paramReader.readGlobalConst()


pR.readPOS('./mcalc/spinchain/POSCAR')
hr_g=reader.Hr()
hr_g.readDoubleFiles('./mcalc/spinchain/hr.up.dat','./mcalc/spinchain/hr.dn.dat')
gr=hamilton.Green()
gr.calcBands(hr_g,orbList=[0,1],spin_polarized=True,fermi=0.0,eta=0.05,mainDir='./mcalc/spinchain/',resolution=100)

#L=20
#scatterList=[[[np.array([0,0,0]),0],[np.array([0,0,0]),2], 0.5],
#             [[np.array([0,0,0]),1],[np.array([0,0,0]),3], 0.5],
#             [[np.array([1,0,0]),0],[np.array([1,0,0]),2], 0.5],
#             [[np.array([1,0,0]),1],[np.array([1,0,0]),3], 0.5],
             #[[np.array([0,0,0]),0],[np.array([0,-1,0]),1], 0.5],
             #[[np.array([0,-1,0]),1],[np.array([0,0,0]),0], 0.5],
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
#E_exact=gr.calcMatrixPerturbatedEnergyViaExact(hr_g,scatterList,eta=0.0001,energyRange=(-5.,0.7),Lx=10,Ly=1,Lz=1)
#print(E_exact)
#E_fourier=gr.calcMatrixPerturbatedEnergyViaFourier(hr_g,[],eta=0.0001,energyRange=(-12.,-1.8),Lx=6,Ly=6,Lz=1,globalCutOrder=0,diverseCutOrder=2,skipOdds=False,onlyPerturbatedEnergy=False)
#print(E_fourier)

#J=gr.calcJ(hr_g,sourceOrbList=[0],sourceR=[0,0,0],targetOrbList=[0],targetR=[1,0,0],eta=0.0001,energyRange=(-10.0,0.0),Lx=10,Ly=1,Lz=1,globalCutOrder=8,diverseCutOrder=2,linearPerturb=True)
#J=gr.calcJFourState(hr_g,sourceOrbList=[0],sourceR=[0,0,0],targetOrbList=[0],targetR=[1,0,0],theta=np.pi/2,eta=0.0001,energyRange=(-10.0,0.0),Lx=10,Ly=1,Lz=1)
#J=gr.calcJFourState_Numba(hr_g,sourceOrbList=[0],sourceR=[0,0,0],targetOrbList=[0],targetR=[1,0,0],theta=np.pi,eta=0.0001,energyRange=(-15.0,0.0),Lx=20,Ly=20,Lz=1)
nkpt,c_kmesh,c_Hk_addr=gr.getCTypeKMeshAndHk(hr_g,Lx=10,Ly=1,Lz=1)
J=gr.calcJFourState_CDLL(hr_g,nkpt,c_kmesh,c_Hk_addr,sourceOrbList=[0],sourceR=[0,0,0],targetOrbList=[0],targetR=[1,0,0],theta=np.pi/2,eta=0.0001,energyRange=(-10.0,0.0),Lx=10,Ly=1,Lz=1)

#J=gr.calcJExact(hr_g,sourceOrbList=[0],sourceR=[0,0,0],targetOrbList=[0],targetR=[1,0,0],eta=0.0001,energyRange=(-10.0,0.0),Lx=10,Ly=1,Lz=1)
#J=gr.calcAnisotropicJ(hr_g,sourceOrbList=[0,1,2,3,4],sourceR=[0,0,0],targetOrbList=[5,6,7,8,9],targetR=[0,0,0],eta=0.0001,energyRange=(-12.0,-1.8),Lx=6,Ly=6,Lz=1)
#print(J*1000)

