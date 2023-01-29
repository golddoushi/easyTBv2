import paramReader
paramReader.readGlobalConst()
import posReader as pR
import hrReader as reader
from ham import Green, Response
import numpy as np
from auxiliary import obtainProjections

# Pt
mainDir="./mcalc/Pt/"
pR.readPOS(mainDir+'POSCAR')
hr=reader.Hr()
hr.readSingleFile(mainDir+'wannier90_hr.dat',ignoreImagPart=False,cutoff=-0.1)
projections=[
    "Pt:s;p;d"
]
atomicOrbIndexList=obtainProjections(mainDir+"POSCAR",projections=projections,soc=True)
atomicMagMomList=[
    np.zeros(3),  # 0 mag.mom.
]

hr.symmerizeBonds(atomicOrbIndexList=atomicOrbIndexList,symOpSelection=[],
                  spinor=True,saxis=np.array([0,0,-1]),atomicMagMomList=atomicMagMomList)
#exit()
#atomicOrbIndexList=hr.make_supercell([1,0,0],[1,2,0],[0,0,1],atomicOrbIndexList=atomicOrbIndexList,soc=False)
#hr.adjustSOCStrength(socFactor=1.0)
# NOTE check the fermi in GlobalConst first
paramReader.FERMI=10.96914665
res=Response()
res.calcProjectiveSpinHallConductor_CDLL(hr,
    mainDir=mainDir,atomicOrbIndexList=atomicOrbIndexList,kgrid=[50,50,50],fermiList=np.linspace(8.9469,12.9469,21),erange=[-100,100],
    proj_op=[
        [[0],[0]], # full vel
        ],
    scatter_route=[
        [0,0,0],
        ],
    ndivide=40,nthreads=4)