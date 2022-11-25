import matplotlib
matplotlib.use('Agg')
import paramReader
paramReader.readGlobalConst()
import posReader as pR
import hrReader as reader
from ham import Green, Response
import numpy as np
import matplotlib.pyplot as plt
from auxiliary import RToSpinor

pR.readPOS('./mcalc/CrI3Tube/POSCAR')

res=Response()
# data structure atomicOrbIndexList[atom index][Lz][orb index]: give the orb. index corr. to hr.dat
# CrI3tube
#atomicOrbIndexList=[[[],[],[0,1,2,3,4,11,12,13,14,15]], # Fe
#                    [[],[5,6,7,16,17,18]],              # Cl1
#                    [[],[8,9,10,19,20,21]],             # Cl2
#                   ]
nCr, nI= pR.nions
atomicOrbIndexList=[]
iorb, norb= 0, 224
for iCr in range(nCr):
    atomicOrbIndexList.append([[],[],[iorb,iorb+1,iorb+2,iorb+3,iorb+4,iorb+norb,iorb+1+norb,iorb+2+norb,iorb+3+norb,iorb+4+norb]])
    iorb+=5
for iI in range(nI):
    atomicOrbIndexList.append([[],[iorb,iorb+1,iorb+2,iorb+norb,iorb+1+norb,iorb+2+norb]])
    iorb+=3

atomicMagMomList=[]
for iCr in range(nCr):
    pos_relative = pR.pos_ions[iCr][:2]-np.array([0.5,0.5])
    pos_n=pos_relative/np.sqrt(pos_relative@pos_relative)
    mag=np.array([pos_n[0],pos_n[1],0])
    atomicMagMomList.append(mag)
    #print(mag)
#exit()

atomicMagMomList+=[np.zeros(3) for iI in range(nI)]

hr=reader.Hr()
hr.readSingleFile('./mcalc/CrI3Tube/wannier90_hr.dat',ignoreImagPart=False,cutoff=-0.1)
hr.symmerizeBonds(atomicOrbIndexList=atomicOrbIndexList,symOpSelection=[],
                  spinor=True,saxis=np.array([0,0,-1]),atomicMagMomList=atomicMagMomList)
#exit()
gr=Green()
gr.calcFullBands_CDLL(hr,fermi=0,mainDir='./mcalc/CrI3Tube/',saveFig=True,redBands=[1,2])
exit()
#atomicOrbIndexList=hr.make_supercell([2,0,0],[0,2,0],[0,0,1],atomicOrbIndexList=atomicOrbIndexList,soc=True)
#hr.adjustSOCStrength(socFactor=1.0)
# NOTE check the fermi in GlobalConst first
res.calcBulkPhotovoltaicConductor_CDLL(hr,mainDir='./mcalc/CrI3Tube/',
        atomicOrbIndexList=atomicOrbIndexList,
        kgrid=[1,1,800],omegaList=np.linspace(0.5,3.5,101),tau=0.003,erange=[-8,3],
        show=False,ndivide=5,nthreads=16,full_mat=True)
exit()
'''
res.calcProjectiveBulkPhotovoltaicConductor_CDLL(hr,
    mainDir='./mcalc/FeCl2/',atomicOrbIndexList=atomicOrbIndexList,kgrid=[1600,1600,1],omegaList=np.linspace(1.5,3.5,100),tau=0.001,erange=[-8,3],
    proj_op=[
        [[0,1,2],[0,1,2]], # full vel
        ],
    scatter_route=[
        [0,0,0],
        ],
    ndivide=40,nthreads=16)
exit()
'''
'''
res.calcOrbitalProjectiveBulkPhotovoltaicConductor_CDLL(hr,
    mainDir='./mcalc/FeCl2/',atomicOrbIndexList=atomicOrbIndexList,kgrid=[1600,1600,1],omegaList=np.linspace(1.5,3.5,100),tau=0.001,erange=[-8,3],
    proj_op=[
        [[0,1,2],[0,1,2]], # full vel
        ],
    scatter_route=[
        [0,0,0],
        ],
    ndivide=40,nthreads=16)
exit()
'''
res.calcSpinProjectiveBulkPhotovoltaicConductor_CDLL(hr,
    mainDir='./mcalc/FeCl2/',atomicOrbIndexList=atomicOrbIndexList,kgrid=[1600,1600,1],omegaList=np.linspace(1.5,3.5,100),tau=0.001,erange=[-8,3],
    proj_op=[
        [[0,1,2],[0,1,2]], # full vel
        ],
    scatter_route=[
        [0,0,0],
        ],
    ndivide=40,nthreads=16)
