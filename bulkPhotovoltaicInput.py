import paramReader
paramReader.readGlobalConst()
import posReader as pR
import hrReader as reader
from ham import Green, Response
import numpy as np
from auxiliary import obtainProjections

mainDir='./mcalc/FeCl2/'
pR.readPOS(mainDir+'POSCAR')
projections=[
    "Fe:d",
    "Cl:p"
]
atomicOrbIndexList=obtainProjections(mainDir+"POSCAR",projections=projections,soc=True)
atomicMagMomList=[np.array([0,0,1]), # in z direction
                  np.zeros(3),       # 0 mag.mom.
                  np.zeros(3)        # 0 mag.mom.
                 ]
hr=reader.Hr()
hr.readSingleFile(mainDir+'wannier90_hr.dat',ignoreImagPart=False,cutoff=-0.1)
hr.symmerizeBonds(atomicOrbIndexList=atomicOrbIndexList,symOpSelection=[],
                  spinor=True,saxis=np.array([0,0,-1]),atomicMagMomList=atomicMagMomList)

res=Response()
#atomicOrbIndexList=hr.make_supercell([1,0,0],[1,2,0],[0,0,1],atomicOrbIndexList=atomicOrbIndexList,soc=False)
#hr.adjustSOCStrength(socFactor=1.0)
# NOTE check the fermi in GlobalConst first
res.calcBulkPhotovoltaicConductor_CDLL(hr,mainDir,atomicOrbIndexList=atomicOrbIndexList,
    kgrid=[100,100,1],omegaList=np.linspace(0.5,1.2,21),
    tau=0.02,erange=(-8,3),
    ndivide=1,nthreads=4,full_mat=True)
exit()

#'''
res.calcModernOrbitalProjectiveBulkPhotovoltaicConductor_CDLL(hr,
    mainDir='./mcalc/FeCl2/',atomicOrbIndexList=atomicOrbIndexList,kgrid=[50,50,1],omegaList=np.linspace(0.5,1.2,29),tau=0.02,erange=[-5,0],
    proj_op=[
        [[0,1,2],[0,1,2]], # full vel
        ],
    scatter_route=[
        [0,0,0],
        ],
    ndivide=4,nthreads=4)
exit()
#'''
res.calcOrbitalProjectiveBulkPhotovoltaicConductor_CDLL(hr,
    mainDir='./mcalc/FeCl2/',atomicOrbIndexList=atomicOrbIndexList,kgrid=[50,50,1],omegaList=np.linspace(0.5,1.2,29),tau=0.02,erange=[-5,0],
    proj_op=[
        [[0,1,2],[0,1,2]], # full vel
        ],
    scatter_route=[
        [0,0,0],
        ],
    ndivide=4,nthreads=4)
exit()
res.calcSpinProjectiveBulkPhotovoltaicConductor_CDLL(hr,
    mainDir='./mcalc/FeCl2/',atomicOrbIndexList=atomicOrbIndexList,kgrid=[50,50,1],omegaList=np.linspace(0.5,1.2,29),tau=0.02,erange=[-5,0],
    proj_op=[
        [[0,1,2],[0,1,2]], # full vel
        ],
    scatter_route=[
        [0,0,0],
        ],
    ndivide=4,nthreads=4)