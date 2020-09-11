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
                        # index style: orbID + (Rx,Ry,Rz) \dot (norb,Lx*norb,Lx*Ly*norb)
                        sourceID=bondDigital.source +  Rx*Hr.norb +  Ry*Lx*Hr.norb +  Rz*Lx*Ly*Hr.norb
                        targetID=bondDigital.target + Rx_*Hr.norb + Ry_*Lx*Hr.norb + Rz_*Lx*Ly*Hr.norb
                        #print(sourceID,targetID,bondDigital.t)
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

    # spin compensated single scattering center
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

    # double scattering centers
    def calcPerturbDOSViaDirectInversionDoubleCenters(self,hr,scatter1,scatter2,eta=0.01,energyRange=(-3.5,3.5),NEDOS=500,Lx=1,Ly=1,Lz=1):
        '''
        scatter[0], scatter[1]: R, l, corresponding to the position of scattering centers
        scatter[1]: delta, corresponding to the potential of responsible scattering
        '''
        totCell=Lx*Ly*Lz
        norb=hr.norb*totCell

        BravaisIndex=np.array([hr.norb,Lx*hr.norb,Lx*Ly*hr.norb])
        iorb1=scatter1[1] + np.dot(scatter1[0]%np.array([Lx,Ly,Lz]),BravaisIndex)
        iorb2=scatter2[1] + np.dot(scatter2[0]%np.array([Lx,Ly,Lz]),BravaisIndex)
        print('impose horizontal magnetic impurities on orb%d and orb%d'%(iorb1,iorb2))

        _ham=Ham().buildWholeHam(hr,Lx=Lx,Ly=Ly,Lz=Lz,getHam=True)
        _ham[iorb1][iorb1]=scatter1[2]
        _ham[iorb2][iorb2]=scatter2[2]
        #print(_ham)
        #exit()
        energyAxis=np.linspace(energyRange[0],energyRange[1],NEDOS)
        dosAxis=[]
        for z in energyAxis:
            G=np.linalg.inv((z+eta*1j)*np.eye(norb)-_ham)
            dosAxis.append(sum(np.diagonal(G)).imag/(-np.pi)/totCell)

        plt.figure()
        plt.plot(energyAxis,dosAxis)
        plt.show()

    def calcPerturbDOSViaFourierDoubleCenters(self,hr,scatter1=(np.array([0,0,0]),0,-1),scatter2=(np.array([2,1,0]),1,-1),order=3,eta=0.01,energyRange=(-3.5,3.5),NEDOS=500):
        '''
        scatter[0]: R, scatter[1]: l, these two define the position of scattering centers
        scatter[1]: delta, corresponding to the potential of responsible scattering
        '''
        def calcPerturbationOnRoute(scatterRouteList,scatterPotential,glineMatrix,plineMatrix):
            totalContribution=0.j
            for scatterRoute in scatterRouteList:
                #print('route:',scatterRoute)
                routeLength=len(scatterRoute)
                plineID0=scatterRoute[0]
                plineID1=scatterRoute[routeLength-1]
                pline=plineMatrix[plineID0][plineID1]
                #print('external line contribution:',pline)

                gline=1.
                for id in range(routeLength-1):
                    gline*=glineMatrix[scatterRoute[id]][scatterRoute[id+1]]
                #print('internal line contribution:',gline)

                potential=1.
                for id in scatterRoute:
                    potential*=scatterPotential[id]
                
                # include the geometric series
                # decide the largest term

                onLineOriginalContribution=pline*gline*potential
                #print('route contribution before series fixing:',onLineOriginalContribution)
                onLineFullContribution=onLineOriginalContribution
                '''
                Now I cannot find a way to solve this bowling up
                '''
                #for id in scatterRoute:
                #    onLineFullContribution-=onLineOriginalContribution
                #    onLineFullContribution+=pline*gline*potential/(1-scatterPotential[id]*glineMatrix[id][id])
                #print('route contribution:',onLineFullContribution)
                totalContribution+=onLineFullContribution#pline*gline*potential
            #print('total contribution:',totalContribution)
            return totalContribution
        
        norb=hr.norb
        kmesh=pR.createKPOINTS(getKMesh=True)
        irkmesh=kmesh
        weight=[1 for i in irkmesh]
        #irkmesh, weight, equalIndex=pR.getIrreducibleKmesh(kmesh)
        totalWeight=sum(weight)
        #print('via symmetry considerations, only %d/%d kpoints involved'%(len(irkmesh),len(kmesh)))

        HkList=[]
        for kpt in irkmesh:
            _ham=Ham()
            Hk=_ham.buildHkFromHr(hr,kpt,getHk=True)
            HkList.append(Hk)

        energyAxis=np.linspace(energyRange[0],energyRange[1],NEDOS)
        dosAxis=[]
        for z in energyAxis:
            #z=-2.5180360721442887
            rou=0.j
            g11,g12,g21,g22=0.j,0.j,0.j,0.j
            p11,p12,p21,p22=0.j,0.j,0.j,0.j

            #g00=0.j
            #rou_perturb_ref=0.j
            # calc scattering elements
            for kpt, _hk, _weight in zip(irkmesh,HkList,weight):
                _gk=np.linalg.inv((z+eta*1j)*np.eye(norb)-_hk)
                # internal line
                g11+=_gk[scatter1[1],scatter1[1]]*_weight/totalWeight
                g22+=_gk[scatter2[1],scatter2[1]]*_weight/totalWeight
                g12+=np.exp(2j*np.pi*np.dot(kpt,scatter1[0]-scatter2[0]))*_gk[scatter1[1],scatter2[1]]*_weight/totalWeight
                g21+=np.exp(2j*np.pi*np.dot(kpt,scatter2[0]-scatter1[0]))*_gk[scatter2[1],scatter1[1]]*_weight/totalWeight
                
                # external line
                p11+=np.dot(_gk[:,scatter1[1]],_gk[scatter1[1],:])*_weight/totalWeight
                p22+=np.dot(_gk[:,scatter2[1]],_gk[scatter2[1],:])*_weight/totalWeight
                p12+=np.exp(-2j*np.pi*np.dot(kpt,scatter1[0]-scatter2[0]))*np.dot(_gk[:,scatter1[1]],_gk[scatter2[1],:])*_weight/totalWeight
                p21+=np.exp(-2j*np.pi*np.dot(kpt,scatter2[0]-scatter1[0]))*np.dot(_gk[:,scatter2[1]],_gk[scatter1[1],:])*_weight/totalWeight

                # unperturbated dos
                rou+=sum(np.diagonal(_gk))*_weight/totalWeight

                #g00+=_gk[scatter1[1],scatter1[1]]*_weight/totalWeight
                #rou_perturb_ref+=np.dot(_gk[:,scatter1[1]],_gk[scatter1[1],:])*_weight/totalWeight
            
            #rou_perturb_ref_tmp=rou_perturb_ref
            #rou_perturb_ref=rou_perturb_ref*scatter1[2]/(1-scatter1[2]*g00)/totalWeight
            '''
            print('accurate perturb before series fixing:',rou_perturb_ref_tmp*scatter1[2]/totalWeight)
            print('accurate perturb:',rou_perturb_ref)
            print('delta:',scatter1[2])
            print('g00',g00)
            print('G0 dot G0:',rou_perturb_ref)
            print('delta dot g00 |r|:%.6f'%abs(scatter1[2]*g00),scatter1[2]*g00)
            '''
            # calc the modification due to perturbation
            scatterIDList=[[0],[1]]
            scatterPotential=[scatter1[2],scatter2[2]]
            gline=[[g11,g12],
                   [g21,g22]]
            pline=[[p11/totalWeight,p12/totalWeight],
                   [p21/totalWeight,p22/totalWeight]]
            perturb_old=calcPerturbationOnRoute(scatterIDList,scatterPotential,gline,pline)
            #perturb0=perturb_old
            '''
            print('now the paramter of iterative')
            print('gline')
            print(gline)
            print('pline')
            print(pline)
            print('potential:')
            print(scatterPotential)
            print('0th order:',perturb_old)
            '''
            # iteratively calc the perturbation itself
            rou_perturb=0.j
            norder=0
            perturb_tmp=[perturb_old]
            cnt=0
            while(True):
                norder+=1
                if norder>order:
                    break
                rou_perturb+=perturb_old
                scatterIDList_update=[]
                for scatterRoute in scatterIDList:
                    #print(scatterRoute)
                    route1=list(scatterRoute)
                    route2=list(scatterRoute)
                    route1.extend([0])
                    route2.extend([1])
                    scatterIDList_update.append(route1)
                    scatterIDList_update.append(route2)
                    #if scatterRoute[len(scatterRoute)-1]==0:
                    #    scatterRoute.append(1)
                    #else:
                    #    scatterRoute.append(0)
                scatterIDList=scatterIDList_update
                #print(scatterIDList)
                #exit()
                perturb_new=calcPerturbationOnRoute(scatterIDList,scatterPotential,gline,pline)
                perturb_tmp.append(perturb_new)
                #print('%dth order'%norder,perturb_new)
                #if abs(perturb_new.imag)<np.min([abs(rou.imag*0.1),0.001]) and cnt==0:
                    #print(z)
                #    break
                #if abs(perturb_new)>abs(perturb_old) or cnt>0:
                #print(abs(perturb_new/perturb_old))
                cnt+=1
                if cnt==8:
                    #if abs(perturb_new/perturb_old)>1:
                    #    print('Bowling up at energy:%.3f with r_new/r_old'%z)
                    perturb_new_list=list(perturb_tmp)
                    zerothPerturb=perturb_new_list.pop(0)
                    perturb_old_list=list(perturb_tmp)
                    perturb_old_list.pop(len(perturb_old_list)-1)
                    divide=np.array(perturb_new_list)/np.array(perturb_old_list)
                    #print(abs(divide))
                    #print(divide)
                    meanDivide=np.mean(divide)
                    print('r: %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f last %.6f'%(*abs(divide),perturb_tmp[7].imag))
                    if abs(meanDivide)>1:
                        print('***Bowling up at energy:%.3f'%z)
                        rou_perturb=zerothPerturb/(1-meanDivide)
                        break
                    rou_perturb=sum(perturb_tmp[:7])+perturb_tmp[7]/(1-divide[7])#zerothPerturb/(1-meanDivide)
                    break
                    #break
                perturb_old=perturb_new
                
            #print(z,rou_perturb,rou_perturb_ref)
            rou+=rou_perturb
            dosAxis.append(rou.imag/(-np.pi))

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
                rou_up_perturb+=np.dot(_gk[:norb,0],_gk[0,:norb])*_weight/totalWeight
                rou_dn_perturb+=np.dot(_gk[norb:,norb],_gk[norb,norb:])*_weight/totalWeight
                #for i in range(norb):
                #    rou_up_perturb+=_gk[i,0]*_gk[0,i]*_weight/totalWeight
                #    rou_dn_perturb+=_gk[norb+i,norb]*_gk[norb,norb+i]*_weight/totalWeight
            
            factor=delta*delta/(1-delta*delta*g00_up*g00_dn)
            rou_up+=rou_up_perturb*factor*g00_dn/totalWeight
            rou_dn+=rou_dn_perturb*factor*g00_up/totalWeight

            dosUpAxis.append(rou_up.imag/(-np.pi))
            dosDnAxis.append(rou_dn.imag/(np.pi))

        plt.figure()
        plt.plot(energyAxis,dosUpAxis)
        plt.plot(energyAxis,dosDnAxis)
        plt.show()

    def calcHorizontallyMagneticPerturbatedDOSViaDirectInversionDoubleCenters(self,hr_up,hr_dn,scatter1,scatter2,eta=0.1,energyRange=(-3.5,3.5),NEDOS=500,Lx=1,Ly=1,Lz=1):
        assert hr_up.norb==hr_dn.norb, 'The norb is inconsistent in the two hr file'
        totCell=Lx*Ly*Lz
        norb=hr_up.norb*totCell*2
        halfNorb=int(norb/2)
        
        BravaisIndex=np.array([hr_up.norb,Lx*hr_up.norb,Lx*Ly*hr_up.norb])
        iorb1=scatter1[1] + np.dot(scatter1[0]%np.array([Lx,Ly,Lz]),BravaisIndex)
        iorb2=scatter2[1] + np.dot(scatter2[0]%np.array([Lx,Ly,Lz]),BravaisIndex)
        print('impose horizontal magnetic impurities on orb%d and orb%d'%(iorb1,iorb2))

        _ham=Ham().buildWholeSpinPolarizedHam(hr_up,hr_dn,Lx=Lx,Ly=Ly,Lz=Lz,getHam=True)
        _ham[iorb1,iorb1+halfNorb]=scatter1[2]
        _ham[iorb1+halfNorb,iorb1]=scatter1[2]
        _ham[iorb2,iorb2+halfNorb]=scatter2[2]
        _ham[iorb2+halfNorb,iorb2]=scatter2[2]

        energyAxis=np.linspace(energyRange[0],energyRange[1],NEDOS)
        dosUpAxis, dosDnAxis=[], []
        for z in energyAxis:
            G=np.linalg.inv((z+eta*1j)*np.eye(norb)-_ham)
            dosUpAxis.append(sum(np.diagonal(G)[:halfNorb]).imag/(-np.pi)/totCell)
            dosDnAxis.append(sum(np.diagonal(G)[halfNorb:]).imag/(np.pi)/totCell)

        plt.figure()
        plt.plot(energyAxis,dosUpAxis)
        plt.plot(energyAxis,dosDnAxis)
        plt.show()

    def calcHorizontallyMagneticPerturbatedDOSViaFourierDoubleCenters(self,hr_up,hr_dn,scatter1=(np.array([0,0,0]),0,-1),scatter2=(np.array([2,1,0]),1,-1),order=3,eta=0.01,energyRange=(-3.5,3.5),NEDOS=500):
        '''
        scatter[0]: R, scatter[1]: l, these two define the position of scattering centers
        scatter[1]: delta, corresponding to the potential of responsible scattering
        '''
        def calcPerturbationOnRoute(scatterRouteList,scatterPotential,glineMatrix,plineMatrix):
            if len(scatterRouteList[0])%2==1:
                return [0, 0]
            
            totalContributionList=[]
            for i in range(2): # for two spin tunnels
                iele=i
                totalContribution=0.j
                for scatterRoute in scatterRouteList:
                    #print('route:',scatterRoute)
                    routeLength=len(scatterRoute)
                    plineID0=scatterRoute[0]
                    plineID1=scatterRoute[routeLength-1]
                    pline=plineMatrix[iele%2][plineID0][plineID1]
                    #print('external line contribution:',pline)
                    iele+=1
                    gline=1.
                    for id in range(routeLength-1):
                        gline*=glineMatrix[iele%2][scatterRoute[id]][scatterRoute[id+1]]
                        iele+=1
                
                    potential=1.
                    for id in scatterRoute:
                        potential*=scatterPotential[id]
                
                    totalContribution+=pline*gline*potential
                totalContributionList.append(totalContribution)
            return totalContributionList
        
        norb=int(hr_up.norb)
        kmesh=pR.createKPOINTS(getKMesh=True)
        weight=[1 for i in kmesh]
        totalWeight=sum(weight)

        HkList=[]
        for kpt in kmesh:
            Hk=Ham().buildSpinPolarizedHk(hr_up,hr_dn,kpt,getHk=True)
            HkList.append(Hk)

        #oz=(energyRange[0]+energyRange[1])/2
        #r=abs(energyRange[1]-energyRange[0])/2
        #thetaAxis=np.linspace(1,0,NEDOS)
        #energyAxis=[r*np.exp(1j*np.pi*theta)+oz for theta in thetaAxis]
        energyAxis=np.linspace(energyRange[0],energyRange[1],NEDOS)
        dosUpAxis,dosDnAxis=[],[]
        for z in energyAxis:
            rou_up, rou_dn=0.j, 0.j
            g11_up,g12_up,g21_up,g22_up=0.j,0.j,0.j,0.j
            g11_dn,g12_dn,g21_dn,g22_dn=0.j,0.j,0.j,0.j
            p11_up,p12_up,p21_up,p22_up=0.j,0.j,0.j,0.j
            p11_dn,p12_dn,p21_dn,p22_dn=0.j,0.j,0.j,0.j

            # calc scattering elements
            for kpt, _hk, _weight in zip(kmesh,HkList,weight):
                _gk=np.linalg.inv((z+eta*1j)*np.eye(norb*2)-_hk)
                # internal line
                g11_up+=_gk[scatter1[1],scatter1[1]]*_weight/totalWeight
                g22_up+=_gk[scatter2[1],scatter2[1]]*_weight/totalWeight
                g12_up+=np.exp(2j*np.pi*np.dot(kpt,scatter1[0]-scatter2[0]))*_gk[scatter1[1],scatter2[1]]*_weight/totalWeight
                g21_up+=np.exp(2j*np.pi*np.dot(kpt,scatter2[0]-scatter1[0]))*_gk[scatter2[1],scatter1[1]]*_weight/totalWeight

                g11_dn+=_gk[scatter1[1]+norb,scatter1[1]+norb]*_weight/totalWeight
                g22_dn+=_gk[scatter2[1]+norb,scatter2[1]+norb]*_weight/totalWeight
                g12_dn+=np.exp(2j*np.pi*np.dot(kpt,scatter1[0]-scatter2[0]))*_gk[scatter1[1]+norb,scatter2[1]+norb]*_weight/totalWeight
                g21_dn+=np.exp(2j*np.pi*np.dot(kpt,scatter2[0]-scatter1[0]))*_gk[scatter2[1]+norb,scatter1[1]+norb]*_weight/totalWeight
                
                # external line
                p11_up+=np.dot(_gk[:,scatter1[1]],_gk[scatter1[1],:])*_weight/totalWeight
                p22_up+=np.dot(_gk[:,scatter2[1]],_gk[scatter2[1],:])*_weight/totalWeight
                p12_up+=np.exp(-2j*np.pi*np.dot(kpt,scatter1[0]-scatter2[0]))*np.dot(_gk[:,scatter1[1]],_gk[scatter2[1],:])*_weight/totalWeight
                p21_up+=np.exp(-2j*np.pi*np.dot(kpt,scatter2[0]-scatter1[0]))*np.dot(_gk[:,scatter2[1]],_gk[scatter1[1],:])*_weight/totalWeight

                p11_dn+=np.dot(_gk[:,scatter1[1]+norb],_gk[scatter1[1]+norb,:])*_weight/totalWeight
                p22_dn+=np.dot(_gk[:,scatter2[1]+norb],_gk[scatter2[1]+norb,:])*_weight/totalWeight
                p12_dn+=np.exp(-2j*np.pi*np.dot(kpt,scatter1[0]-scatter2[0]))*np.dot(_gk[:,scatter1[1]+norb],_gk[scatter2[1]+norb,:])*_weight/totalWeight
                p21_dn+=np.exp(-2j*np.pi*np.dot(kpt,scatter2[0]-scatter1[0]))*np.dot(_gk[:,scatter2[1]+norb],_gk[scatter1[1]+norb,:])*_weight/totalWeight

                # unperturbated dos
                rou_up+=sum(np.diagonal(_gk)[:norb])*_weight/totalWeight
                rou_dn+=sum(np.diagonal(_gk)[norb:])*_weight/totalWeight

            # calc the modification due to perturbation
            scatterIDList=[[0,0],[0,1],[1,0],[1,0]]
            scatterPotential=[scatter1[2],scatter2[2]]
            gline_up=[[g11_up,g12_up],
                      [g21_up,g22_up]]
            gline_dn=[[g11_dn,g12_dn],
                      [g21_dn,g22_dn]]
            gline=[gline_up,gline_dn]
            pline_up=[[p11_up/totalWeight,p12_up/totalWeight],
                      [p21_up/totalWeight,p22_up/totalWeight]]
            pline_dn=[[p11_dn/totalWeight,p12_dn/totalWeight],
                      [p21_dn/totalWeight,p22_dn/totalWeight]]
            pline=[pline_up,pline_dn]

            perturb_up_old, perturb_dn_old=calcPerturbationOnRoute(scatterIDList,scatterPotential,gline,pline)
            
            # iteratively calc the perturbation itself
            rou_up_perturb, rou_dn_perturb=0.j, 0.j
            norder=0
            perturb_up_tmp, perturb_dn_tmp=[perturb_up_old], [perturb_dn_old]
            cnt=0
            while(True):
                norder+=1
                if norder>order:
                    break
                rou_up_perturb+=perturb_up_old
                rou_dn_perturb+=perturb_dn_old
                scatterIDList_update=[]
                for scatterRoute in scatterIDList:
                    #print(scatterRoute)
                    route1=list(scatterRoute)
                    route2=list(scatterRoute)
                    route3=list(scatterRoute)
                    route4=list(scatterRoute)
                    route1.extend([0,0])
                    route2.extend([0,1])
                    route3.extend([1,0])
                    route4.extend([1,1])
                    scatterIDList_update.append(route1)
                    scatterIDList_update.append(route2)
                    scatterIDList_update.append(route3)
                    scatterIDList_update.append(route4)
                scatterIDList=scatterIDList_update
                #print(scatterIDList)
                #exit()
                perturb_up_new,perturb_dn_new=calcPerturbationOnRoute(scatterIDList,scatterPotential,gline,pline)
                perturb_up_tmp.append(perturb_up_new)
                perturb_dn_tmp.append(perturb_dn_new)
                cnt+=1
                if cnt==4:
                    perturb_up_new_list, perturb_dn_new_list=list(perturb_up_tmp),list(perturb_dn_tmp)
                    zerothPerturbUp,zerothPerturbDn=perturb_up_new_list.pop(0),perturb_dn_new_list.pop(0)
                    perturb_up_old_list,perturb_dn_old_list=list(perturb_up_tmp),list(perturb_dn_tmp)
                    perturb_up_old_list.pop(len(perturb_up_old_list)-1)
                    perturb_dn_old_list.pop(len(perturb_dn_old_list)-1)
                    divide_up=np.array(perturb_up_new_list)/np.array(perturb_up_old_list)
                    divide_dn=np.array(perturb_dn_new_list)/np.array(perturb_dn_old_list)
                    meanDivideUp, meanDivideDn=np.mean(divide_up), np.mean(divide_dn)
                    print('up: %.6f %.6f %.6f %.6f theta: %.6f %.6f %.6f %.6f last %.6f'%(*abs(divide_up),*(np.log(divide_up/abs(divide_up),dtype=complex).imag),perturb_up_tmp[3].imag))
                    print('dn: %.6f %.6f %.6f %.6f theta: %.6f %.6f %.6f %.6f last %.6f'%(*abs(divide_dn),*(np.log(divide_dn/abs(divide_dn),dtype=complex).imag),perturb_dn_tmp[3].imag))
                    if abs(meanDivideUp)>1:
                        print('***Spin up is bowling up at energy:%.3f'%z)
                        rou_up_perturb=zerothPerturbUp/(1-meanDivideUp)
                    else:
                        rou_up_perturb=sum(perturb_up_tmp[:3])+perturb_up_tmp[3]/(1-divide_up[3])
                    if abs(meanDivideDn)>1:
                        print('***Spin dn is bowling up at energy:%.3f'%z)
                        rou_dn_perturb=zerothPerturbDn/(1-meanDivideDn)
                    else:
                        rou_dn_perturb=sum(perturb_dn_tmp[:3])+perturb_dn_tmp[3]/(1-divide_dn[3])
                    
                    break
                    
                perturb_up_old, perturb_dn_old=perturb_up_new, perturb_dn_new
                
            rou_up+=rou_up_perturb
            rou_dn+=rou_dn_perturb
            dosUpAxis.append(rou_up.imag/(-np.pi))
            dosDnAxis.append(rou_dn.imag/np.pi)

        plt.figure()
        plt.plot(energyAxis,dosUpAxis)
        plt.plot(energyAxis,dosDnAxis)
        plt.show()

    # the general form
    def calcMatrixPerturbatedDOSViaExact(self,hr,scatterList,eta=0.01,energyRange=(-3.5,3.5),NEDOS=500,Lx=1,Ly=1,Lz=1):
        '''
        in scatterList
        scatter[0]: R_source, iorb_source
        scatter[1]: R_target, iorb_target
        scatter[2]: new matrix element
        '''
        totCell=Lx*Ly*Lz
        norb=hr.norb#*totCell
        totOrb=totCell*norb
        BravaisIndex=np.array([norb,Lx*norb,Lx*Ly*norb])
        _ham=Ham().buildWholeHam(hr,Lx=Lx,Ly=Ly,Lz=Lz,getHam=True)
        for scatter in scatterList:

            iorb_source=scatter[0][1] + np.dot(scatter[0][0]%np.array([Lx,Ly,Lz]),BravaisIndex)
            iorb_target=scatter[1][1] + np.dot(scatter[1][0]%np.array([Lx,Ly,Lz]),BravaisIndex)
            #print(_ham[iorb_source][iorb_target])
            _ham[iorb_source][iorb_target]+=scatter[2]
            #print(scatter[0][0]%np.array([Lx,Ly,Lz]),scatter[1][0]%np.array([Lx,Ly,Lz]))
            #print(iorb_source,iorb_target,scatter[2])
        
        energyAxis=np.linspace(energyRange[0],energyRange[1],NEDOS)
        dosAxis=[]
        for z in energyAxis:
            G=np.linalg.inv((z+eta*1j)*np.eye(totOrb)-_ham)
            dosAxis.append(sum(np.diagonal(G)).imag/(-np.pi)/totCell)

        plt.figure()
        plt.plot(energyAxis,dosAxis)
        plt.show()

    def calcMatrixPerturbatedDOSViaFourier(self,hr,scatterList,eta=0.01,energyRange=(-3.5,3.5),NEDOS=500,Lx=1,Ly=1,Lz=1,globalCutOrder=8,diverseCutOrder=2,skipOdds=False):
        # analyze scattering list
        # construct scattering matrix with all irrelevant zeros suppressed
        relevantOrbList=[]
        scSize=np.array([Lx,Ly,Lz])
        def findOrbIndexInRecordList(orb):
            for iexistedOrb, existedOrb in enumerate(relevantOrbList):
                if (existedOrb[0]==orb[0]%scSize).all() and existedOrb[1]==orb[1]:
                    return iexistedOrb
            return -1
        # build scattering basis
        for scatter in scatterList:
            # standarize Bravais vectors
            scatter[0][0]%=scSize
            scatter[1][0]%=scSize
            if findOrbIndexInRecordList(scatter[0])<0:
                relevantOrbList.append(scatter[0])
            if findOrbIndexInRecordList(scatter[1])<0:
                relevantOrbList.append(scatter[1])
        scatteringOrder=len(relevantOrbList)
        print('find scattering region with %d-dimension'%scatteringOrder)
        print(relevantOrbList)
        # build scattering matrix
        scatteringCenter=np.zeros((scatteringOrder,scatteringOrder))
        for scatter in scatterList:
            sourceID=findOrbIndexInRecordList(scatter[0])
            targetID=findOrbIndexInRecordList(scatter[1])
            scatteringCenter[sourceID,targetID]=scatter[2]
        print('scattering matrix is:')
        print(scatteringCenter)
        
        norb=hr.norb
        kmesh=pR.createKPOINTS(getKMesh=True,provideSize=[Lx,Ly,Lz])
        weight=[1 for i in kmesh]
        totalWeight=sum(weight)
        print(totalWeight)
        #exit()

        HkList=[]
        for kpt in kmesh:
            _ham=Ham()
            Hk=_ham.buildHkFromHr(hr,kpt,getHk=True)
            HkList.append(Hk)

        energyAxis=np.linspace(energyRange[0],energyRange[1],NEDOS)
        dosAxis=[]
        for z in energyAxis:
            rou=0.j
            g_internal=np.zeros((scatteringOrder,scatteringOrder),dtype=complex)
            g_external_contracted=np.zeros((scatteringOrder,scatteringOrder),dtype=complex)
            
            for kpt, _hk in zip(kmesh,HkList):
                _gk=np.linalg.inv((z+eta*1j)*np.eye(norb)-_hk)
                for irow in range(scatteringOrder):
                    R_left, orbID_left=relevantOrbList[irow]
                    for icol in range(scatteringOrder):
                        R_right, orbID_right=relevantOrbList[icol]
                        #print('row%d col%d --- scatterLeft (%d %d %d) orb%d scatterRight (%d %d %d) orb%d'%(irow,icol,*R_left,orbID_left,*R_right,orbID_right))
                        # internal line
                        g_internal[irow,icol]+=np.exp(2j*np.pi*np.dot(kpt,R_left-R_right))*_gk[orbID_left,orbID_right]/totalWeight
                        # contracted external line
                        g_external_contracted[irow,icol]+=np.exp(-2j*np.pi*np.dot(kpt,R_left-R_right))*np.dot(_gk[orbID_right,:],_gk[:,orbID_left])/totalWeight
               
                # unperturbated dos
                rou+=sum(np.diagonal(_gk))
                #exit()
            
            perturb_matrix=np.dot(g_external_contracted,scatteringCenter)
            perturb_tmp=[]
            cnt=0
            while(True):
                if skipOdds:
                    perturb_matrix=perturb_matrix.dot(g_internal).dot(scatteringCenter)
                perturb_matrix_trace=sum(np.diagonal(perturb_matrix))
                perturb_tmp.append(perturb_matrix_trace)
                if cnt==globalCutOrder:
                    perturb_new_list=list(perturb_tmp)
                    perturb_old_list=list(perturb_tmp)
                    perturb_new_list.pop(0)
                    perturb_old_list.pop(globalCutOrder)
                    divide=np.array(perturb_new_list)/np.array(perturb_old_list)
                    meanDivide=np.mean(divide)
                    
                    if abs(divide[globalCutOrder-1])>1 and abs(perturb_tmp[globalCutOrder].imag/totalWeight)>10:
                        print('***Bowling up at energy:%.3f, mean factor %.6f'%(z,abs(meanDivide)))
                        print('r:',(' %.6f'*globalCutOrder)%(*abs(divide),), 'last %.6f'%(perturb_tmp[globalCutOrder].imag))
                        rou_perturb=sum(perturb_tmp[:diverseCutOrder])+perturb_tmp[diverseCutOrder]/(1-meanDivide)#zerothPerturb/(1-meanDivide)
                    else:
                        #print(len(perturb_tmp),globalCutOrder)
                        #print(perturb_tmp[:globalCutOrder])
                        #exit()
                        rou_perturb=sum(perturb_tmp[:(globalCutOrder-1)])+perturb_tmp[(globalCutOrder-1)]/(1-divide[(globalCutOrder-1)])
                    break
                cnt+=1
                perturb_matrix=perturb_matrix.dot(g_internal).dot(scatteringCenter)
                #perturb_matrix_trace=sum(np.diagonal(perturb_matrix))
                #perturb_tmp.append(perturb_matrix_trace)

            rou+=rou_perturb
            dosAxis.append(rou.imag/(-np.pi)/totalWeight) # DOS per cell

        plt.figure()
        plt.plot(energyAxis,dosAxis)
        plt.show()

    # energy calculator
    def calcMatrixPerturbatedEnergyViaExact(self,hr,scatterList,eta=0.01,energyRange=(-5.0,0),Lx=1,Ly=1,Lz=1):
        '''
        in scatterList
        scatter[0]: R_source, iorb_source
        scatter[1]: R_target, iorb_target
        scatter[2]: new matrix element
        '''
        totCell=Lx*Ly*Lz
        norb=hr.norb#*totCell
        totOrb=totCell*norb
        BravaisIndex=np.array([norb,Lx*norb,Lx*Ly*norb])
        _ham=Ham().buildWholeHam(hr,Lx=Lx,Ly=Ly,Lz=Lz,getHam=True)
        for scatter in scatterList:
            iorb_source=scatter[0][1] + np.dot(scatter[0][0]%np.array([Lx,Ly,Lz]),BravaisIndex)
            iorb_target=scatter[1][1] + np.dot(scatter[1][0]%np.array([Lx,Ly,Lz]),BravaisIndex)
            _ham[iorb_source][iorb_target]+=scatter[2]
        
        r=(energyRange[1]-energyRange[0])/2
        o=(energyRange[0]+energyRange[1])/2
        
        def getEOnCircle(theta):
            z=r*np.exp(1j*theta)
            E=z+o
            G=np.linalg.inv((E+eta*1j)*np.eye(totOrb)-_ham)
            dos=sum(np.diagonal(G))/(-np.pi)/totCell
            totalEnergy=E*dos
            return (totalEnergy*1j*z).imag
        
        inte,err=integrate.quad(getEOnCircle,a=np.pi,b=0.)
        return inte

    def calcMatrixPerturbatedEnergyViaFourier(self,hr,scatterList,eta=0.01,energyRange=(-5.0,0),NEDOS=500,Lx=1,Ly=1,Lz=1,globalCutOrder=8,diverseCutOrder=2,skipOdds=False):
        # analyze scattering list
        # construct scattering matrix with all irrelevant zeros suppressed
        relevantOrbList=[]
        scSize=np.array([Lx,Ly,Lz])
        def findOrbIndexInRecordList(orb):
            for iexistedOrb, existedOrb in enumerate(relevantOrbList):
                if (existedOrb[0]==orb[0]%scSize).all() and existedOrb[1]==orb[1]:
                    return iexistedOrb
            return -1
        # build scattering basis
        for scatter in scatterList:
            # standarize Bravais vectors
            scatter[0][0]%=scSize
            scatter[1][0]%=scSize
            if findOrbIndexInRecordList(scatter[0])<0:
                relevantOrbList.append(scatter[0])
            if findOrbIndexInRecordList(scatter[1])<0:
                relevantOrbList.append(scatter[1])
        scatteringOrder=len(relevantOrbList)
        print('find scattering region with %d-dimension'%scatteringOrder)
        print(relevantOrbList)
        # build scattering matrix
        scatteringCenter=np.zeros((scatteringOrder,scatteringOrder))
        for scatter in scatterList:
            sourceID=findOrbIndexInRecordList(scatter[0])
            targetID=findOrbIndexInRecordList(scatter[1])
            scatteringCenter[sourceID,targetID]=scatter[2]
        print('scattering matrix is:')
        print(scatteringCenter)
        
        norb=hr.norb
        kmesh=pR.createKPOINTS(getKMesh=True,provideSize=[Lx,Ly,Lz])
        weight=[1 for i in kmesh]
        totalWeight=sum(weight)

        HkList=[]
        for kpt in kmesh:
            _ham=Ham()
            Hk=_ham.buildHkFromHr(hr,kpt,getHk=True)
            HkList.append(Hk)

        r=(energyRange[1]-energyRange[0])/2
        o=(energyRange[0]+energyRange[1])/2

        def getEOnCircle(theta):
            z=r*np.exp(1j*theta)
            E=z+o
            rou=0.j
            g_internal=np.zeros((scatteringOrder,scatteringOrder),dtype=complex)
            g_external_contracted=np.zeros((scatteringOrder,scatteringOrder),dtype=complex)
            
            for kpt, _hk in zip(kmesh,HkList):
                _gk=np.linalg.inv((E+eta*1j)*np.eye(norb)-_hk)
                for irow in range(scatteringOrder):
                    R_left, orbID_left=relevantOrbList[irow]
                    for icol in range(scatteringOrder):
                        R_right, orbID_right=relevantOrbList[icol]
                        #print('row%d col%d --- scatterLeft (%d %d %d) orb%d scatterRight (%d %d %d) orb%d'%(irow,icol,*R_left,orbID_left,*R_right,orbID_right))
                        # internal line
                        g_internal[irow,icol]+=np.exp(2j*np.pi*np.dot(kpt,R_left-R_right))*_gk[orbID_left,orbID_right]/totalWeight
                        # contracted external line
                        g_external_contracted[irow,icol]+=np.exp(-2j*np.pi*np.dot(kpt,R_left-R_right))*np.dot(_gk[orbID_right,:],_gk[:,orbID_left])/totalWeight
               
                # unperturbated dos
                rou+=sum(np.diagonal(_gk))
                #exit()
            
            perturb_matrix=np.dot(g_external_contracted,scatteringCenter)
            perturb_tmp=[]
            cnt=0
            while(True):
                if skipOdds:
                    perturb_matrix=perturb_matrix.dot(g_internal).dot(scatteringCenter)
                perturb_matrix_trace=sum(np.diagonal(perturb_matrix))
                perturb_tmp.append(perturb_matrix_trace)
                if cnt==globalCutOrder:
                    perturb_new_list=list(perturb_tmp)
                    perturb_old_list=list(perturb_tmp)
                    perturb_new_list.pop(0)
                    perturb_old_list.pop(globalCutOrder)
                    divide=np.array(perturb_new_list)/np.array(perturb_old_list)
                    meanDivide=np.mean(divide)
                    
                    if abs(divide[globalCutOrder-1])>1 and abs(perturb_tmp[globalCutOrder].imag/totalWeight)>10:
                        print('***Bowling up at complex energy:(%.3f %.3f), mean factor %.6f'%(E.real,E.imag,abs(meanDivide)))
                        print('r:',(' %.6f'*globalCutOrder)%(*abs(divide),), 'last %.6f'%(perturb_tmp[globalCutOrder].imag))
                        rou_perturb=sum(perturb_tmp[:diverseCutOrder])+perturb_tmp[diverseCutOrder]/(1-meanDivide)#zerothPerturb/(1-meanDivide)
                    else:
                        rou_perturb=sum(perturb_tmp[:(globalCutOrder-1)])+perturb_tmp[(globalCutOrder-1)]/(1-divide[(globalCutOrder-1)])
                    break
                cnt+=1
                perturb_matrix=perturb_matrix.dot(g_internal).dot(scatteringCenter)

            rou+=rou_perturb
            dos=rou/(-np.pi)/totalWeight # DOS per cell
            totalEnergy=E*dos
            return (totalEnergy*1j*z).imag
        
        inte,err=integrate.quad(getEOnCircle,a=np.pi,b=0.)
        return inte

