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
hr.readSingleFile('./mcalc/BN/hr_biorbitals.dat',ignoreImagPart=False,cutoff=-0.01)

res=Response()
# data structure atomicOrbIndexList[atom index][Lz][orb index]: give the orb. index corr. to hr.dat
atomicOrbIndexList=[[[0]],
                    [[1]],
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

Lk=101
eigs_addr,vecs_addr,DHkDk1_addr,DHkDk2_addr,DHkDk3_addr=res.calcEigsAndVecsInMemory_CDLL(hr,atomicOrbIndexList,kmesh=[Lk,Lk,1],nthreads=1)
#dEdkx_addr,dEdky_addr,dEdkz_addr=res.derivateQuantitiesOnKGrid_CDLL(eigs_addr,hr.norb,[Lk,Lk,1],data_type="double",nthreads=1)
#data=res.redOutQuantitiesOnKGrid(dEdkx_addr,hr.norb,[Lk,Lk,1],data_type='double')
#drawHotmapOnHexagonalBZ('./mcalc/BN/',data[:,0],[Lk,Lk,1])
#exit()
dvdkx_addr,dvdky_addr,dvdkz_addr=res.derivateQuantitiesOnKGrid_CDLL(vecs_addr,hr.norb,[Lk,Lk,1],data_type="eigenvector",nthreads=1)
eigs=res.redOutQuantitiesOnKGrid(eigs_addr,hr.norb,[Lk,Lk,1],data_type='double')
vecs=res.redOutQuantitiesOnKGrid(vecs_addr,hr.norb*hr.norb,[Lk,Lk,1],data_type='complex')
DHkDk1=res.redOutQuantitiesOnKGrid(DHkDk1_addr,hr.norb*hr.norb,[Lk,Lk,1],data_type='complex')
DHkDk2=res.redOutQuantitiesOnKGrid(DHkDk2_addr,hr.norb*hr.norb,[Lk,Lk,1],data_type='complex')
dvecs_dkx=res.redOutQuantitiesOnKGrid(dvdkx_addr,hr.norb*hr.norb,[Lk,Lk,1],data_type='complex')
dvecs_dky=res.redOutQuantitiesOnKGrid(dvdky_addr,hr.norb*hr.norb,[Lk,Lk,1],data_type='complex')
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
    p_plus=rx[0,1]+1j*ry[0,1]
    p_minus=rx[0,1]-1j*ry[0,1]
    c_polar=(abs(p_plus)**2-abs(p_minus)**2)/(abs(p_plus)**2+abs(p_minus)**2)
    
    Berry_curv.append(c_polar)
    #print(curv_z.real)
drawHotmapOnHexagonalBZ('./mcalc/BN/',np.array(Berry_curv),[Lk-1,Lk-1,1])
#exit()
#res.calcInjectPhotovoltaicConductor_CDLL(hr,mainDir='./mcalc/BN/',atomicOrbIndexList=atomicOrbIndexList,kgrid=[50,50,1],omegaList=np.linspace(1.0,3.0,20),tau=0.01,erange=[-80,80],nthreads=1)
#res.calcShiftPhotovoltaicConductor_CDLL(hr,mainDir='./mcalc/BN/',atomicOrbIndexList=atomicOrbIndexList,kgrid=[50,50,1],omegaList=np.linspace(1.0,3.0,20),tau=0.01,erange=[-80,80],nthreads=1)
#res.calcBulkPhotovoltaicConductor_CDLL(hr,mainDir='./mcalc/BN/',atomicOrbIndexList=atomicOrbIndexList,kgrid=[50,50,1],omegaList=np.linspace(1.0,3.0,20),tau=0.01,erange=[-80,80],nthreads=1)
