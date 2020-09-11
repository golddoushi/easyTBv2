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

#hr_s=reader.Hr('./simpleHR.txt')
#hr_d=reader.Hr('./simpleHR_dn.txt')
hr_g=reader.Hr('./simpleGraphene.txt')
#gr.testG(hr_s,orbList=[0,1],fermi=0,eta=0.05)
gr=hamilton.Green()
#gr.calcBands(hr_s,orbList=[0,1],fermi=0,eta=0.1,resolution=200)
#gr.calcDOSViaFourier(hr_s,orbList=[0,1],eta=0.01,NEDOS=500)
#gr.calcDOSViaDirectInversion(hr_s,eta=0.01,energyRange=(-3.5,3.5),NEDOS=500,Lx=5,Ly=5)
#gr.calcPerturbDOSViaFourier(hr_s,delta=-1,eta=0.01,NEDOS=500)
#gr.calcSpinPolarizedDOSViaDirectInversion(hr_s,hr_d,eta=0.01,energyRange=(-3.5,3.5),NEDOS=500,Lx=5,Ly=5,Lz=1)
#gr.calcSpinPolarizedDOSViaFourier(hr_s,hr_d,eta=0.01,energyRange=(-3.5,3.5),NEDOS=500)
#gr.calcHorizontallyMagneticPerturbatedSpinPolarizedDOSViaDirectInversion(hr_s,hr_d,delta=1.,eta=0.01,energyRange=(-3.5,3.5),NEDOS=500,Lx=5,Ly=5,Lz=1)
#gr.calcHorizontallyMagneticPerturbatedDOSViaFourier(hr_s,hr_d,delta=1.,eta=0.01,energyRange=(-3.5,3.5),NEDOS=500)
#gr.calcPerturbDOSViaDirectInversionDoubleCenters(hr_g,scatter1=(np.array([0,0,0]),0,-0.3),scatter2=(np.array([2,2,0]),0,-0.3),eta=0.01,energyRange=(-3.5,3.5),NEDOS=500,Lx=5,Ly=5,Lz=1)
#gr.calcPerturbDOSViaFourierDoubleCenters(hr_g,scatter1=(np.array([0,0,0]),0,-0.3),scatter2=(np.array([2,2,0]),0,-0.3),order=15,eta=0.01,energyRange=(-3.5,3.5),NEDOS=500)
#gr.calcHorizontallyMagneticPerturbatedDOSViaDirectInversionDoubleCenters(hr_s,hr_d,scatter1=(np.array([0,0,0]),0,-0.3),scatter2=(np.array([0,0,0]),1,-0.3),eta=0.035,energyRange=(-3.5,3.5),NEDOS=500,Lx=5,Ly=5,Lz=1)
#gr.calcHorizontallyMagneticPerturbatedDOSViaFourierDoubleCenters(hr_s,hr_d,scatter1=(np.array([0,0,0]),0,-0.3),scatter2=(np.array([0,0,0]),1,-0.3),order=15,eta=0.035,energyRange=(-3.5,3.5),NEDOS=500)


#scatterList=[[[np.array([0,0,0]),0],[np.array([0,0,0]),1], -0.2],
#             [[np.array([0,0,0]),1],[np.array([0,0,0]),0], -0.2],
             #[[np.array([0,0,0]),0],[np.array([1,0,0]),1], 0.1],
             #[[np.array([1,0,0]),1],[np.array([0,0,0]),0], 0.1],
             #[[np.array([0,0,0]),0],[np.array([0,-1,0]),1], 0.1],
             #[[np.array([0,-1,0]),1],[np.array([0,0,0]),0], 0.1],
#             ]
scatterList=[[[np.array([0,0,0]),0],[np.array([0,0,0]),2],0.0],
             [[np.array([0,0,0]),2],[np.array([0,0,0]),0],0.0],
             [[np.array([0,0,0]),1],[np.array([0,0,0]),3],-0.0],
             [[np.array([0,0,0]),3],[np.array([0,0,0]),1],-0.0],
             ]

#gr.calcMatrixPerturbatedDOSViaExact(hr_g,scatterList,eta=0.01,energyRange=(-3.5,3.5),NEDOS=500,Lx=5,Ly=5,Lz=1)
#gr.calcMatrixPerturbatedDOSViaFourier(hr_g,scatterList,eta=0.01,energyRange=(-3.5,3.5),NEDOS=500,Lx=5,Ly=5,Lz=1,globalCutOrder=12,diverseCutOrder=2,skipOdds=True)

gr.calcMatrixPerturbatedEnergyViaExact(hr_g,scatterList,eta=0.1,energyRange=(-5,5),Lx=5,Ly=5,Lz=1)
#gr.calcMatrixPerturbatedEnergyViaFourier(hr_g,scatterList,eta=0.001,energyRange=(-5.,0.),NEDOS=500,Lx=5,Ly=5,Lz=1)



