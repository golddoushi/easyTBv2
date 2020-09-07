from scipy.sparse import coo_matrix
from scipy import integrate
import scipy.sparse.linalg as linalg
import numpy as np
np.set_printoptions(precision=6,suppress=True)
import time
import posReader as pR
import matplotlib.pyplot as plt

class Ham:
    def __init__(self):
        self.sourceList=[]
        self.targetList=[]
        self.tList=[]

        self.imagHopping=False

    def addHopping(self,sourceID,targetID,t,overLat=np.array([0,0,0]),kpt=np.array([0,0,0])):
        self.sourceList.append(sourceID)
        self.targetList.append(targetID)
        tk=t*np.exp(-2j*np.pi*np.dot(overLat,kpt))
        self.tList.append(tk)
        return

    def buildHkFromHr(self,Hr,kpt,getHk=False):
        for bondDigital in Hr.bondList:
            self.addHopping(bondDigital.source,bondDigital.target,bondDigital.t,overLat=bondDigital.overlat,kpt=kpt)
        if getHk:
            return coo_matrix((self.tList,(self.sourceList,self.targetList)),dtype=complex).toarray()
        return

    def buildSpinPolarizedHk(self,Hr_up,Hr_dn,kpt,getHk=False):
        assert Hr_up.norb==Hr_dn.norb, 'the file of spin up and down should involve the same quantity of orbitals'
        norb=Hr_up.norb
        # spin up
        for bondUpDigital in Hr_up.bondList:
            self.addHopping(bondUpDigital.source,bondUpDigital.target,bondUpDigital.t,overLat=bondUpDigital.overlat,kpt=kpt)
        # spin down
        for bondDnDigital in Hr_dn.bondList:
            self.addHopping(bondDnDigital.source+norb,bondDnDigital.target+norb,bondDnDigital.t,overLat=bondDnDigital.overlat,kpt=kpt)
        
        if getHk:
            return coo_matrix((self.tList,(self.sourceList,self.targetList)),dtype=complex).toarray()
        return

    def buildWholeHam(self,Hr,Lx=1,Ly=1,Lz=1,getHam=True):
        for bondDigital in Hr.bondList:
            for Rx in range(Lx):
                for Ry in range(Ly):
                    for Rz in range(Lz):
                        overLat_tmp=(bondDigital.overlat+np.array([Rx,Ry,Rz]))
                        Rx_, Ry_, Rz_=np.array([overLat_tmp[0]%Lx,overLat_tmp[1]%Ly,overLat_tmp[2]%Lz])
                        #print(overLat_fixed)
                        sourceID=bondDigital.source +  Rx*Hr.norb +  Ry*Lx*Hr.norb +  Rz*Lx*Ly*Hr.norb
                        targetID=bondDigital.target + Rx_*Hr.norb + Ry_*Lx*Hr.norb + Rz_*Lx*Ly*Hr.norb
                        self.addHopping(sourceID,targetID,bondDigital.t)
        if getHam:
            return coo_matrix((self.tList,(self.sourceList,self.targetList)),dtype=complex).toarray()
        return
    
    def buildWholeSpinPolarizedHam(self,Hr_up,Hr_dn,Lx=1,Ly=1,Lz=1,getHam=True):
        totCell=Lx*Ly*Lz
        totOrb=Hr_up.norb*totCell
        # construct the part of spin up
        for bondUpDigital in Hr_up.bondList:
            for Rx in range(Lx):
                for Ry in range(Ly):
                    for Rz in range(Lz):
                        overLat_tmp=(bondUpDigital.overlat+np.array([Rx,Ry,Rz]))
                        Rx_, Ry_, Rz_=np.array([overLat_tmp[0]%Lx,overLat_tmp[1]%Ly,overLat_tmp[2]%Lz])
                        #print(overLat_fixed)
                        sourceID_up=bondUpDigital.source +  Rx*Hr_up.norb +  Ry*Lx*Hr_up.norb +  Rz*Lx*Ly*Hr_up.norb
                        targetID_up=bondUpDigital.target + Rx_*Hr_up.norb + Ry_*Lx*Hr_up.norb + Rz_*Lx*Ly*Hr_up.norb
                        self.addHopping(sourceID_up,targetID_up,bondUpDigital.t)
        # construct the part of spin dn
        for bondDnDigital in Hr_dn.bondList:
            for Rx in range(Lx):
                for Ry in range(Ly):
                    for Rz in range(Lz):
                        overLat_tmp=(bondDnDigital.overlat+np.array([Rx,Ry,Rz]))
                        Rx_, Ry_, Rz_=np.array([overLat_tmp[0]%Lx,overLat_tmp[1]%Ly,overLat_tmp[2]%Lz])
                        #print(overLat_fixed)
                        sourceID_dn=totOrb + bondDnDigital.source +  Rx*Hr_up.norb +  Ry*Lx*Hr_up.norb +  Rz*Lx*Ly*Hr_up.norb
                        targetID_dn=totOrb + bondDnDigital.target + Rx_*Hr_up.norb + Ry_*Lx*Hr_up.norb + Rz_*Lx*Ly*Hr_up.norb
                        self.addHopping(sourceID_dn,targetID_dn,bondDnDigital.t)

        if getHam:
            return coo_matrix((self.tList,(self.sourceList,self.targetList)),dtype=complex).toarray()
        return
    
    # exact diagonal
    def sovle_exact(self,vec=False):
        self.imagHopping=True
        self.buildCsrMatrix(parserMethod=False)
        _eig, _vec = np.linalg.eigh(self.csr,'U')
        if vec: return _eig, _vec
        return _eig

    # partially solve using parse matrix
    def solve(self,k=1,which='LM',v0=None,maxiter=None,tol=1e-5,vec=False):
        self.buildCsrMatrix(parserMethod=True)
        return linalg.eigsh(self.csr,k=k,sigma=0,which='LM',v0=v0,maxiter=maxiter,tol=tol,return_eigenvectors=vec)
        #return np.linalg.eigh(self.csr)

    def buildCsrMatrix(self,parserMethod=True):
        if self.imagHopping:
            coo=coo_matrix((self.tList,(self.sourceList,self.targetList)),dtype=complex)
        else:
            coo=coo_matrix((self.tList,(self.sourceList,self.targetList)),dtype=float)
        self.csr=coo.tocsr() if parserMethod else coo.toarray()


class Green:

    def __init__(self):
        pass

    def calcBulkSusceptibility(self,hr,kmesh,a=0,b=2,R=[-1,0,0],fermi=0.,eta=0.01):
        '''Linear response approximation
           \chi = Int_-inf^fermi{dz G(aR,b0,z)G(b0,aR,z)}
           using Gaussian integration along semi-circle in upper complex plane
        '''
        self.norbital=hr.norb
        self.nkpts=len(kmesh)
        self.HkList=[]
        #find bottom of eigenvalues.
        eig_min=100.
        time0=time.time()
        for kpt in kmesh:
            _ham=Ham()
            Hk=_ham.buildHkFromHr(hr,kpt,getHk=True)
            self.HkList.append(Hk)
            eig=np.min(_ham.sovle_exact())
            eig_min=eig if eig < eig_min else eig_min
        print('Hk constructed in %.3fs, found band edge locates at %.6f'%(time.time()-time0,eig_min))

        eig_min-=5
        ox=(fermi+eig_min)/2
        r=(fermi-eig_min)/2
        
        #fourierBasis=np.array([np.exp(np.pi*2j*np.dot(kpt,np.array(R))) for kpt in kmesh])
        
        def getGG(zlist=0.):
            val=[]
            for z in zlist:
                G_aRb0=0.j
                G_b0aR=0.j
                for ikpt, kpt in enumerate(kmesh):
                    Hk=self.HkList[ikpt]
                    Gk=np.linalg.inv((z+eta*1j)*np.eye(self.norbital)-Hk)
                    G_aRb0+=np.exp(np.pi*2j*np.dot(kpt,np.array(R)))*Gk[a][b]/self.nkpts
                    G_b0aR+=np.exp(np.pi*2j*np.dot(kpt,-np.array(R)))*Gk[b][a]/self.nkpts
                val.append(G_aRb0*G_b0aR)
            return np.array(val)
        
        def onCircle(theta):
            z=r*np.exp(1j*theta,dtype=complex)+ox
            #print(len(z))
            return getGG(z).real*r
        
        time0=time.time()
        inte,err=integrate.quadrature(onCircle,a=np.pi,b=0,maxiter=200)
        print(inte,err)
        print('Integration finished in %.3fs'%(time.time()-time0))
        #print(onCircle(np.array([np.pi])))
        return inte

    def calcBands(self,hr,orbList=[0],fermi=0,eta=0.1,resolution=100):
        norb=hr.norb
        HkList=[]
        
        time0=time.time()
        kmesh=pR.autoGenerateKpath2D(onlyGetKPTList=True)
        for ikpt, kpt in enumerate(kmesh):
            _ham=Ham()
            Hk=_ham.buildHkFromHr(hr,kpt,getHk=True)
            HkList.append(Hk)
            
        print('Hk constructed in %.3fs'%(time.time()-time0))
        energyAXIS=np.linspace(fermi-3,fermi+2,resolution)
        DOS=[]
        for z in energyAXIS:
            _dos_k=[]
            for _hk in HkList:
                gk=np.linalg.inv((z+eta*1j)*np.eye(norb)-_hk)
                _dos_k.append(sum(np.diagonal(gk)[orbList]).imag/(-np.pi))
            DOS.append(_dos_k)
        plt.figure()
        plt.imshow(DOS,extent=(-5,5,-5,5),origin='lower')
        plt.show()
                
    def testG(self,hr,orbList=[0],fermi=0,eta=0.1):
        norb=hr.norb
        HkList=[]
        
        time0=time.time()
        kmesh=pR.createKPOINTS(getKMesh=True)
        nkpts=len(kmesh)
        for ikpt, kpt in enumerate(kmesh):
            _ham=Ham()
            Hk=_ham.buildHkFromHr(hr,kpt,getHk=True)
            HkList.append(Hk)
        print('Hk constructed in %.3fs with %d kpoints'%(time.time()-time0,len(kmesh)))

        RvecList=[np.array([2,2,0]),np.array([0,1,0]),np.array([0,1,0])]
        # fourier transformation for Hr
        for Rvec in RvecList:
            print('Bravais Vector %d %d %d'%(*Rvec,))
            hr_tmp=np.zeros((norb,norb),dtype=complex)
            for kpt, Hk in zip(kmesh,HkList):
                hr_tmp+=np.exp(2j*np.pi*np.dot(kpt,Rvec))*Hk
            hr_tmp/=nkpts
            print(hr_tmp.T)
            exit()
        
    def calcJ(self,hrUP,hrDN,orbList=[0],R=[-1,0,0],fermi=0.,eta=0.0001):
        '''Linear response approximation
           \chi = Int_-inf^fermi{dz G(aR,b0,z)G(b0,aR,z)}
           using Gaussian integration along semi-circle in upper complex plane
        '''
        #exit()
        assert hrUP.norb==hrDN.norb, 'the real space Hamiltonian should be in the same dimension'
        self.norbital=hrUP.norb
        self.HkUPList=[]
        self.HkDNList=[]
        self.deltaList=[hrUP.onsite[i]-hrDN.onsite[i] for i in orbList]
        delta2=np.array(self.deltaList)*np.array(self.deltaList)
        print('Exchange splitting are:')
        print(self.deltaList)
        #exit()
        #find bottom of eigenvalues.
        eigUPMin=100.
        eigDNMin=100.
        time0=time.time()
        kmesh=pR.createKPOINTS(getKMesh=True,forceBoder=False)
        self.nkpts=len(kmesh)
        irkmesh, weight, equalIndex=pR.getIrreducibleKmesh(kmesh,DNA=[])
        print('with symmetry considerations, found %d/%d irreducible kpoints'%(len(irkmesh),len(kmesh)))
        for ikpt, kpt in enumerate(kmesh):
            _hamUP=Ham()
            _hamDN=Ham()
            HkUP=_hamUP.buildHkFromHr(hrUP,kpt,getHk=True)
            HkDN=_hamDN.buildHkFromHr(hrDN,kpt,getHk=True)
            self.HkUPList.append(HkUP)
            self.HkDNList.append(HkDN)
            if equalIndex[ikpt]>=0:
                continue
            eigUPList=_hamUP.sovle_exact()
            eigDNList=_hamDN.sovle_exact()
            _eigUPMin=np.min(eigUPList)
            _eigDNMin=np.min(eigDNList)
            #print(_eigUPMin,_eigDNMin)
            #eig_irkmesh.append(eigUPList)
            eigUPMin=_eigUPMin if _eigUPMin < eigUPMin else eigUPMin
            eigDNMin=_eigDNMin if _eigDNMin < eigDNMin else eigDNMin
        print('Hk constructed in %.3fs, found band edge locates at %.6f and %.6f for upspin and dnspin'%(time.time()-time0,eigUPMin,eigDNMin))
        eig_min=np.min([eigUPMin,eigDNMin])-10
        ox=(fermi+eig_min)/2.
        r=(fermi-eig_min)/2.
        
        def getGG(z):
            G_aRb0=np.zeros(len(orbList),dtype=complex)
            G_b0aR=np.zeros(len(orbList),dtype=complex)
            for ikpt, kpt in enumerate(kmesh):
                HkUP, HkDN=self.HkUPList[ikpt], self.HkDNList[ikpt]
                #eigup, _ =np.linalg.eig(HkUP)
                #eigdn, _ =np.linalg.eig(HkDN)
                #print(eigup,eigdn)
                #exit()
                GkUP=np.linalg.inv((z+eta*1j)*np.eye(self.norbital)-HkUP)
                GkDN=np.linalg.inv((z+eta*1j)*np.eye(self.norbital)-HkDN)
                
                uptunnel=np.diagonal(GkUP)[orbList]
                dntunnel=np.diagonal(GkDN)[orbList]
                G_aRb0+=np.exp(np.pi*2j*np.dot(kpt,np.array(R)))*uptunnel#/self.nkpts
                G_b0aR+=np.exp(np.pi*2j*np.dot(kpt,-np.array(R)))*dntunnel#/self.nkpts
                #print('kpt %.3f %.3f %.3f phase:%.6f 2pi'%(*kpt,np.dot(kpt,np.array(R))))
                #print('uptunnel contribution:',uptunnel)
                #print('dntunnel contribution:',dntunnel)
            
            #exit()
            #print(z)
            #print('R: %d %d %d'%(*R,))
            #print(sum(np.diagonal(np.dot(G_aRb0,G_b0aR))[orbList]))
            #print(G_aRb0)
            #print(G_b0aR)
            #print(sum(G_aRb0*G_b0aR))
            return G_aRb0*G_b0aR/self.nkpts/self.nkpts
        #getGG(-0.72+0.001j)
        #exit()
        
        def onCircle(theta):
            z=r*np.exp(1j*theta,dtype=complex)
            return (sum(getGG(z+ox)*delta2)*1j*z).imag/np.pi*1000
        
        time0=time.time()
        inte,err=integrate.quad(onCircle,a=0,b=np.pi)
        print(inte,err)
        print('Integration finished in %.3fs'%(time.time()-time0))
        #print(onCircle(np.array([np.pi])))
        return inte

    # spin compensated
    def calcDOSViaFourier(self,hr,orbList=[0],eta=0.1,NEDOS=500):
        norb=hr.norb
        kmesh=pR.createKPOINTS(getKMesh=True)
        irkmesh, weight, equalIndex=pR.getIrreducibleKmesh(kmesh)
        totalWeight=sum(weight)
        print('via symmetry considerations, only %d/%d kpoints involved'%(len(irkmesh),len(kmesh)))

        HkList=[]
        eigmin, eigmax= 100, -100
        for kpt in irkmesh:
            _ham=Ham()
            Hk=_ham.buildHkFromHr(hr,kpt,getHk=True)
            eigList=_ham.sovle_exact()
            _eigmin, _eigmax=np.min(eigList), np.max(eigList)
            eigmin = _eigmin if _eigmin < eigmin else eigmin
            eigmax = _eigmax if _eigmax > eigmax else eigmax
            HkList.append(Hk)
        print('Bond range: %.6f ~ %.6f'%(eigmin,eigmax))

        energyAxis=np.linspace(eigmin-0.5,eigmax+0.5,NEDOS)
        dosAxis=[]
        for z in energyAxis:
            rou=0.j
            for _hk, _weight in zip(HkList,weight):
                _gk=np.linalg.inv((z+eta*1j)*np.eye(norb)-_hk)
                rou+=sum(np.diagonal(_gk)[orbList])*_weight/totalWeight
            dosAxis.append(rou.imag/(-np.pi))

        plt.figure()
        plt.plot(energyAxis,dosAxis)
        plt.show()

    def calcPerturbDOSViaFourier(self,hr,delta=-1,eta=0.1,NEDOS=500):
        norb=hr.norb
        kmesh=pR.createKPOINTS(getKMesh=True)
        irkmesh, weight, equalIndex=pR.getIrreducibleKmesh(kmesh)
        totalWeight=sum(weight)
        print('via symmetry considerations, only %d/%d kpoints involved'%(len(irkmesh),len(kmesh)))

        HkList=[]
        eigmin, eigmax= 100, -100
        for kpt in irkmesh:
            _ham=Ham()
            Hk=_ham.buildHkFromHr(hr,kpt,getHk=True)
            eigList=_ham.sovle_exact()
            _eigmin, _eigmax=np.min(eigList), np.max(eigList)
            eigmin = _eigmin if _eigmin < eigmin else eigmin
            eigmax = _eigmax if _eigmax > eigmax else eigmax
            HkList.append(Hk)
        print('Bond range: %.6f ~ %.6f'%(eigmin,eigmax))

        energyAxis=np.linspace(eigmin-0.5,eigmax+0.5,NEDOS)
        dosAxis=[]
        for z in energyAxis:
            rou=0.j
            rou_perturb=0.j
            g00=0.j
            for _hk, _weight in zip(HkList,weight):
                _gk=np.linalg.inv((z+eta*1j)*np.eye(norb)-_hk)
                g00+=_gk[0,0]*_weight/totalWeight
                rou+=sum(np.diagonal(_gk))*_weight/totalWeight
                for iorb in range(norb):
                    rou_perturb+=_gk[iorb,0]*_gk[0,iorb]*_weight/totalWeight
            
            factor=delta/(1-delta*g00)
            rou+=rou_perturb*factor/totalWeight
            
            dosAxis.append(rou.imag/(-np.pi))

        plt.figure()
        plt.plot(energyAxis,dosAxis)
        plt.show()

    def calcDOSViaDirectInversion(self,hr,eta=0.1,energyRange=(-3.5,3.5),NEDOS=500,Lx=1,Ly=1,Lz=1):
        totCell=Lx*Ly*Lz
        norb=hr.norb*totCell
        _ham=Ham().buildWholeHam(hr,Lx=Lx,Ly=Ly,Lz=Lz,getHam=True)
        energyAxis=np.linspace(energyRange[0],energyRange[1],NEDOS)
        dosAxis=[]
        for z in energyAxis:
            G=np.linalg.inv((z+eta*1j)*np.eye(norb)-_ham)
            dosAxis.append(sum(np.diagonal(G)).imag/(-np.pi)/totCell)

        plt.figure()
        plt.plot(energyAxis,dosAxis)
        plt.show()
        
    def calcPerturbDOSViaDirectInversion(self,hr,deta=-1,eta=0.1,energyRange=(-3.5,3.5),NEDOS=500,Lx=1,Ly=1,Lz=1):
        totCell=Lx*Ly*Lz
        norb=hr.norb*totCell
        _ham=Ham().buildWholeHam(hr,Lx=Lx,Ly=Ly,Lz=Lz,getHam=True)
        _ham[0][0]=-deta
        energyAxis=np.linspace(energyRange[0],energyRange[1],NEDOS)
        dosAxis=[]
        for z in energyAxis:
            G=np.linalg.inv((z+eta*1j)*np.eye(norb)-_ham)
            dosAxis.append(sum(np.diagonal(G)).imag/(-np.pi)/totCell)

        plt.figure()
        plt.plot(energyAxis,dosAxis)
        plt.show()

    # spin polarized
    def calcSpinPolarizedDOSViaDirectInversion(self,hr_up,hr_dn,eta=0.1,energyRange=(-3.5,3.5),NEDOS=500,Lx=1,Ly=1,Lz=1):
        assert hr_up.norb==hr_dn.norb, 'The norb is inconsistent in the two hr file'
        totCell=Lx*Ly*Lz
        norb=hr_up.norb*totCell*2
        halfNorb=int(norb/2)
        _ham=Ham().buildWholeSpinPolarizedHam(hr_up,hr_dn,Lx=Lx,Ly=Ly,Lz=Lz,getHam=True)
        energyAxis=np.linspace(energyRange[0],energyRange[1],NEDOS)
        dosUpAxis, dosDnAxis=[], []
        for z in energyAxis:
            G=np.linalg.inv((z+eta*1j)*np.eye(norb)-_ham)
            #print(type(np.diagonal(G)),halfNorb)
            #exit()
            dosUpAxis.append(sum(np.diagonal(G)[:halfNorb]).imag/(-np.pi)/totCell)
            dosDnAxis.append(sum(np.diagonal(G)[halfNorb:]).imag/(np.pi)/totCell)

        plt.figure()
        plt.plot(energyAxis,dosUpAxis)
        plt.plot(energyAxis,dosDnAxis)
        plt.show()
    
    def calcHorizontallyMagneticPerturbatedSpinPolarizedDOSViaDirectInversion(self,hr_up,hr_dn,delta=1.,eta=0.1,energyRange=(-3.5,3.5),NEDOS=500,Lx=1,Ly=1,Lz=1):
        assert hr_up.norb==hr_dn.norb, 'The norb is inconsistent in the two hr file'
        totCell=Lx*Ly*Lz
        norb=hr_up.norb*totCell*2
        halfNorb=int(norb/2)
        _ham=Ham().buildWholeSpinPolarizedHam(hr_up,hr_dn,Lx=Lx,Ly=Ly,Lz=Lz,getHam=True)
        _ham[0,halfNorb]=delta
        _ham[halfNorb,0]=delta
        energyAxis=np.linspace(energyRange[0],energyRange[1],NEDOS)
        dosUpAxis, dosDnAxis=[], []
        for z in energyAxis:
            G=np.linalg.inv((z+eta*1j)*np.eye(norb)-_ham)
            #print(type(np.diagonal(G)),halfNorb)
            #exit()
            dosUpAxis.append(sum(np.diagonal(G)[:halfNorb]).imag/(-np.pi)/totCell)
            dosDnAxis.append(sum(np.diagonal(G)[halfNorb:]).imag/(np.pi)/totCell)

        plt.figure()
        plt.plot(energyAxis,dosUpAxis)
        plt.plot(energyAxis,dosDnAxis)
        plt.show()

    def calcSpinPolarizedDOSViaFourier(self,hr_up,hr_dn,eta=0.1,energyRange=(-3.5,3.5),NEDOS=500):
        norb=int(hr_up.norb)
        kmesh=pR.createKPOINTS(getKMesh=True)
        irkmesh, weight, _=pR.getIrreducibleKmesh(kmesh)
        totalWeight=sum(weight)
        print('via symmetry considerations, only %d/%d kpoints involved'%(len(irkmesh),len(kmesh)))

        HkList=[]
        time0=time.time()
        for kpt in irkmesh:
            _ham=Ham()
            Hk=_ham.buildSpinPolarizedHk(hr_up,hr_dn,kpt,getHk=True)
            HkList.append(Hk)
        print('Hk constructured in %.3fs'%(time.time()-time0))

        energyAxis=np.linspace(energyRange[0],energyRange[1],NEDOS)
        dosUpAxis, dosDnAxis=[], []
        for z in energyAxis:
            rou_up=0.j
            rou_dn=0.j
            for _hk, _weight in zip(HkList,weight):
                _gk=np.linalg.inv((z+eta*1j)*np.eye(norb*2)-_hk)
                rou_up+=sum(np.diagonal(_gk)[:norb])*_weight/totalWeight
                rou_dn+=sum(np.diagonal(_gk)[norb:])*_weight/totalWeight
            dosUpAxis.append(rou_up.imag/(-np.pi))
            dosDnAxis.append(rou_dn.imag/(np.pi))

        plt.figure()
        plt.plot(energyAxis,dosUpAxis)
        plt.plot(energyAxis,dosDnAxis)
        plt.show()

    def calcHorizontallyMagneticPerturbatedDOSViaFourier(self,hr_up,hr_dn,delta=1.,eta=0.1,energyRange=(-3.5,3.5),NEDOS=500):
        norb=int(hr_up.norb)
        kmesh=pR.createKPOINTS(getKMesh=True)
        irkmesh, weight, equalIndex=pR.getIrreducibleKmesh(kmesh)
        totalWeight=sum(weight)
        print('via symmetry considerations, only %d/%d kpoints involved'%(len(irkmesh),len(kmesh)))

        HkList=[]
        time0=time.time()
        for kpt in irkmesh:
            _ham=Ham()
            Hk=_ham.buildSpinPolarizedHk(hr_up,hr_dn,kpt,getHk=True)
            HkList.append(Hk)
        print('Hk constructured in %.3fs'%(time.time()-time0))

        energyAxis=np.linspace(energyRange[0],energyRange[1],NEDOS)
        dosUpAxis, dosDnAxis=[], []
        for z in energyAxis:
            rou_up, rou_dn=0.j, 0.j
            rou_up_perturb, rou_dn_perturb=0.j, 0.j
            g00_up, g00_dn =0.j, 0.j
            for _hk, _weight in zip(HkList,weight):
                _gk=np.linalg.inv((z+eta*1j)*np.eye(2*norb)-_hk)

                g00_up+=_gk[0,0]*_weight/totalWeight
                g00_dn+=_gk[norb,norb]*_weight/totalWeight

                rou_up+=sum(np.diagonal(_gk)[:norb])*_weight/totalWeight
                rou_dn+=sum(np.diagonal(_gk)[norb:])*_weight/totalWeight
                for i in range(norb):
                    rou_up_perturb+=_gk[i,0]*_gk[0,i]*_weight/totalWeight
                    rou_dn_perturb+=_gk[norb+i,norb]*_gk[norb,norb+i]*_weight/totalWeight
            
            factor=delta*delta/(1-delta*delta*g00_up*g00_dn)
            rou_up+=rou_up_perturb*factor*g00_dn/totalWeight
            rou_dn+=rou_dn_perturb*factor*g00_up/totalWeight

            dosUpAxis.append(rou_up.imag/(-np.pi))
            dosDnAxis.append(rou_dn.imag/(np.pi))

        plt.figure()
        plt.plot(energyAxis,dosUpAxis)
        plt.plot(energyAxis,dosDnAxis)
        plt.show()