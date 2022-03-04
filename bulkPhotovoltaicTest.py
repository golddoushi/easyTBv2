import paramReader
paramReader.readGlobalConst()
import posReader as pR
import hrReader as reader
from ham import Green, Response
import numpy as np
import matplotlib.pyplot as plt
from illustrator import drawHotmapOnHexagonalBZ

pR.readPOS('./mcalc/BN/POSCAR')
hr=reader.Hr()
hr.readSingleFile('./mcalc/BN/hr.dat',ignoreImagPart=False,cutoff=-0.01)

res=Response()
# data structure atomicOrbIndexList[atom index][Lz][orb index]: give the orb. index corr. to hr.dat
atomicOrbIndexList=[[[0]],
                    [[1,2]],
                   ]

# data structure atomicMagMomList[atom index]: give the SU(2) rep. of magnetic moments
atomicMagMomList=[
#    [0,1], # in z direction
#    [0,0], # 0 mag.mom.
#    [0,0]  # 0 mag.mom.
                 ]

#hr.symmerizeBonds(atomicOrbIndexList=atomicOrbIndexList,symOpSelection=[],
#                  spinor=False,atomicMagMomList=atomicMagMomList)
#hr.adjustSOCStrength(socFactor=1.0)
'''
Lk=51
nocc=2
kmesh=[Lk,Lk,1]
eigs_addr,vecs_addr,DHkDk1_addr,DHkDk2_addr,DHkDk3_addr=res.calcEigsAndVecsInMemory_CDLL(hr,atomicOrbIndexList,kmesh,nthreads=1)
#dEdkx_addr,dEdky_addr,dEdkz_addr=res.derivateQuantitiesOnKGrid_CDLL(eigs_addr,hr.norb,kmesh,data_type="double",nthreads=1)
#data=res.redOutQuantitiesOnKGrid(dEdkx_addr,hr.norb,kmesh,data_type='double')
#drawHotmapOnHexagonalBZ('./mcalc/BN/',data[:,0],kmesh)
#exit()
dvdkx_addr,dvdky_addr,dvdkz_addr=res.derivateEigvectorsOnKGrid_CDLL(vecs_addr,DHkDk1_addr,DHkDk2_addr,DHkDk3_addr,eigs_addr,hr.norb,nocc,kmesh,nthreads=1)
eigs=res.redOutQuantitiesOnKGrid(eigs_addr,hr.norb,kmesh,data_type='double')
vecs=res.redOutQuantitiesOnKGrid(vecs_addr,hr.norb*hr.norb,kmesh,data_type='complex')
DHkDk1=res.redOutQuantitiesOnKGrid(DHkDk1_addr,hr.norb*hr.norb,kmesh,data_type='complex')
DHkDk2=res.redOutQuantitiesOnKGrid(DHkDk2_addr,hr.norb*hr.norb,kmesh,data_type='complex')
dvecs_dkx=res.redOutQuantitiesOnKGrid(dvdkx_addr,hr.norb*hr.norb,kmesh,data_type='complex')
dvecs_dky=res.redOutQuantitiesOnKGrid(dvdky_addr,hr.norb*hr.norb,kmesh,data_type='complex')
Berry_curv=[]
for ikpt in range(Lk*Lk):
    if ikpt%Lk==Lk-1: continue
    if ikpt>=(Lk-1)*Lk:continue
    eig=eigs[ikpt]
    v=vecs[ikpt].reshape((hr.norb,hr.norb))
    dvdkx=dvecs_dkx[ikpt].reshape((hr.norb,hr.norb))
    dvdky=dvecs_dky[ikpt].reshape((hr.norb,hr.norb))
    rx=1j*v.T.conjugate().dot(dvdkx)
    ry=1j*v.T.conjugate().dot(dvdky)
    curv_z=2*(rx[0,2]*ry[2,0]+rx[1,2]*ry[2,1]).imag
    p_plus=rx[1,2]+1j*ry[1,2]
    p_minus=rx[1,2]-1j*ry[1,2]
    c_polar=(abs(p_plus)**2-abs(p_minus)**2)/(abs(p_plus)**2+abs(p_minus)**2)

    
    # analytic solution
    _dHkdk1=DHkDk1[ikpt].reshape((hr.norb,hr.norb))
    _dHkdk2=DHkDk2[ikpt].reshape((hr.norb,hr.norb))
    _dHkdkx=_dHkdk1-_dHkdk2/2
    _dHkdky=_dHkdk2*np.sqrt(3)/2
    px=v.T.conjugate().dot(_dHkdkx).dot(v)
    py=v.T.conjugate().dot(_dHkdky).dot(v)
    p_plus=px[1,2]+1j*py[1,2]
    p_minus=px[1,2]-1j*py[1,2]
    c_polar=(abs(p_plus)**2-abs(p_minus)**2)/(abs(p_plus)**2+abs(p_minus)**2)
    
    Berry_curv.append(c_polar)
drawHotmapOnHexagonalBZ('./mcalc/BN/',np.array(Berry_curv),[Lk-1,Lk-1,1])
#exit()
'''
#res.calcInjectPhotovoltaicConductor_CDLL(hr,mainDir='./mcalc/BN/',atomicOrbIndexList=atomicOrbIndexList,kgrid=[50,50,1],omegaList=np.linspace(1.0,3.0,20),tau=0.01,erange=[-80,80],nthreads=1)
res.calcShiftPhotovoltaicConductor_CDLL(hr,mainDir='./mcalc/BN/',atomicOrbIndexList=atomicOrbIndexList,kgrid=[50,50,1],omegaList=np.linspace(0.5,3.0,20),tau=0.05,erange=[-80,80],nthreads=1)
#res.calcBulkPhotovoltaicConductor_CDLL(hr,mainDir='./mcalc/BN/',atomicOrbIndexList=atomicOrbIndexList,kgrid=[50,50,1],omegaList=np.linspace(0.5,3.0,20),tau=0.06,erange=[-80,80],nthreads=1)
