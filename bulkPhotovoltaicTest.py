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

Lk=101
nocc=2
vbm,cbm=0,2
kmesh=[Lk,Lk,1]
eigs_addr,vecs_addr,DHkDkx_addr,DHkDky_addr,DHkDkz_addr,DHkDkxDkx_addr,DHkDkyDky_addr,DHkDkxDky_addr=res.calcEigsAndVecsInMemory_CDLL(hr,atomicOrbIndexList,kmesh,nthreads=1)
#dEdkx_addr,dEdky_addr,dEdkz_addr=res.derivateQuantitiesOnKGrid_CDLL(eigs_addr,hr.norb,kmesh,data_type="double",nthreads=1)
#data=res.redOutQuantitiesOnKGrid(dEdkx_addr,hr.norb,kmesh,data_type='double')
#drawHotmapOnHexagonalBZ('./mcalc/BN/',data[:,0],kmesh)
#exit()
#dvdkx_addr,dvdky_addr,dvdkz_addr=res.derivateEigvectorsOnKGrid_CDLL(vecs_addr,DHkDkx_addr,DHkDky_addr,DHkDkz_addr,eigs_addr,hr.norb,nocc,kmesh,nthreads=1)
eigs=res.redOutQuantitiesOnKGrid(eigs_addr,hr.norb,kmesh,data_type='double')
vecs=res.redOutQuantitiesOnKGrid(vecs_addr,hr.norb*hr.norb,kmesh,data_type='complex')
DHkDkx=res.redOutQuantitiesOnKGrid(DHkDkx_addr,hr.norb*hr.norb,kmesh,data_type='complex')
DHkDky=res.redOutQuantitiesOnKGrid(DHkDky_addr,hr.norb*hr.norb,kmesh,data_type='complex')
DHkDkxDkx=res.redOutQuantitiesOnKGrid(DHkDkxDkx_addr,hr.norb*hr.norb,kmesh,data_type='complex')
DHkDkyDky=res.redOutQuantitiesOnKGrid(DHkDkyDky_addr,hr.norb*hr.norb,kmesh,data_type='complex')
DHkDkxDky=res.redOutQuantitiesOnKGrid(DHkDkxDky_addr,hr.norb*hr.norb,kmesh,data_type='complex')
#dvecs_dkx=res.redOutQuantitiesOnKGrid(dvdkx_addr,hr.norb*hr.norb,kmesh,data_type='complex')
#dvecs_dky=res.redOutQuantitiesOnKGrid(dvdky_addr,hr.norb*hr.norb,kmesh,data_type='complex')
Berry_curv=[]
for ikpt in range(Lk*Lk):
    if ikpt%Lk==Lk-1: continue
    if ikpt>=(Lk-1)*Lk:continue
    eig=eigs[ikpt]
    veck=vecs[ikpt].reshape((hr.norb,hr.norb))
    
    # analytic solution
    _dHkdkx=DHkDkx[ikpt].reshape((hr.norb,hr.norb))
    _dHkdky=DHkDky[ikpt].reshape((hr.norb,hr.norb))
    _dHkdkxdkx=DHkDkxDkx[ikpt].reshape((hr.norb,hr.norb))
    _dHkdkydky=DHkDkyDky[ikpt].reshape((hr.norb,hr.norb))
    _dHkdkxdky=DHkDkxDky[ikpt].reshape((hr.norb,hr.norb))
    vx=veck.T.conjugate().dot(_dHkdkx).dot(veck)
    vy=veck.T.conjugate().dot(_dHkdky).dot(veck)
    vxx=veck.T.conjugate().dot(_dHkdkxdkx).dot(veck)
    vyy=veck.T.conjugate().dot(_dHkdkydky).dot(veck)
    vxy=veck.T.conjugate().dot(_dHkdkxdky).dot(veck)
    #p_plus=px[1,2]+1j*py[1,2]
    #p_minus=px[1,2]-1j*py[1,2]
    #c_polar=(abs(p_plus)**2-abs(p_minus)**2)/(abs(p_plus)**2+abs(p_minus)**2)

    Sigma_xx,Sigma_xy,Sigma_yx,Sigma_yy=0,0,0,0
    for p in range(hr.norb):
        if p==vbm or p==cbm:continue
        Sigma_xx+=vx[vbm][p]*vx[p][cbm]/(eig[p]-eig[vbm])+vx[vbm][p]*vx[p][cbm]/(eig[cbm]-eig[p])
        Sigma_xy+=vx[vbm][p]*vy[p][cbm]/(eig[p]-eig[vbm])+vy[vbm][p]*vx[p][cbm]/(eig[cbm]-eig[p])
        Sigma_yx+=vy[vbm][p]*vx[p][cbm]/(eig[p]-eig[vbm])+vx[vbm][p]*vy[p][cbm]/(eig[cbm]-eig[p])
        Sigma_yy+=vy[vbm][p]*vy[p][cbm]/(eig[p]-eig[vbm])+vy[vbm][p]*vy[p][cbm]/(eig[cbm]-eig[p])
    
    rxx=1j/(eig[vbm]-eig[cbm])*((vx[vbm,cbm]*(vx[vbm,vbm]-vx[cbm,cbm])+vx[vbm,cbm]*(vx[vbm,vbm]-vx[cbm,cbm]))/(eig[vbm]-eig[cbm])-vxx[vbm,cbm]+Sigma_xx)
    rxy=1j/(eig[vbm]-eig[cbm])*((vx[vbm,cbm]*(vy[vbm,vbm]-vy[cbm,cbm])+vy[vbm,cbm]*(vx[vbm,vbm]-vx[cbm,cbm]))/(eig[vbm]-eig[cbm])-vxy[vbm,cbm]+Sigma_xy)
    ryx=1j/(eig[vbm]-eig[cbm])*((vy[vbm,cbm]*(vx[vbm,vbm]-vx[cbm,cbm])+vx[vbm,cbm]*(vy[vbm,vbm]-vy[cbm,cbm]))/(eig[vbm]-eig[cbm])-vxy[vbm,cbm]+Sigma_yx)
    ryy=1j/(eig[vbm]-eig[cbm])*((vy[vbm,cbm]*(vy[vbm,vbm]-vy[cbm,cbm])+vy[vbm,cbm]*(vy[vbm,vbm]-vy[cbm,cbm]))/(eig[vbm]-eig[cbm])-vyy[vbm,cbm]+Sigma_yy)
    
    Berry_curv.append(rxx.real)
drawHotmapOnHexagonalBZ('./mcalc/BN/',np.array(Berry_curv),[Lk-1,Lk-1,1])
#exit()

#res.calcInjectPhotovoltaicConductor_CDLL(hr,mainDir='./mcalc/BN/',atomicOrbIndexList=atomicOrbIndexList,kgrid=[50,50,1],omegaList=np.linspace(1.0,3.0,20),tau=0.01,erange=[-80,80],nthreads=1)
#res.calcShiftPhotovoltaicConductor_CDLL(hr,mainDir='./mcalc/BN/',atomicOrbIndexList=atomicOrbIndexList,kgrid=[50,50,1],omegaList=np.linspace(0.5,3.0,20),tau=0.05,erange=[-80,80],nthreads=1)
#res.calcBulkPhotovoltaicConductor_CDLL(hr,mainDir='./mcalc/BN/',atomicOrbIndexList=atomicOrbIndexList,kgrid=[50,50,1],omegaList=np.linspace(0.5,3.0,20),tau=0.06,erange=[-80,80],nthreads=1)
