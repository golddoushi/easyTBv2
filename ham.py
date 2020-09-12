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

    # DOS calculator
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
                        g_internal[irow,icol]+=np.exp(-2j*np.pi*np.dot(kpt,R_left-R_right))*_gk[orbID_left,orbID_right]/totalWeight
                        # contracted external line
                        g_external_contracted[irow,icol]+=np.exp(2j*np.pi*np.dot(kpt,R_left-R_right))*np.dot(_gk[orbID_right,:],_gk[:,orbID_left])/totalWeight
               
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
        
        
        #for theta in np.linspace(np.pi,0,501):
        #    print('%.6f %.6f'%(theta, getEOnCircle(theta)))
        inte,err=integrate.quad(getEOnCircle,a=np.pi,b=0.)
        return inte
        
    def calcMatrixPerturbatedEnergyViaFourier(self,hr,scatterList,eta=0.01,energyRange=(-5.0,0),Lx=1,Ly=1,Lz=1,globalCutOrder=8,diverseCutOrder=2,skipOdds=False):
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
                        g_internal[irow,icol]+=np.exp(-2j*np.pi*np.dot(kpt,R_left-R_right))*_gk[orbID_left,orbID_right]/totalWeight
                        # contracted external line
                        g_external_contracted[irow,icol]+=np.exp(2j*np.pi*np.dot(kpt,R_left-R_right))*np.dot(_gk[orbID_right,:],_gk[:,orbID_left])/totalWeight
               
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
        
        #for theta in np.linspace(np.pi,0,501):
        #    print('%.6f %.6f'%(theta, getEOnCircle(theta)))
        
        inte,err=integrate.quad(getEOnCircle,a=np.pi,b=0.)
        return inte


