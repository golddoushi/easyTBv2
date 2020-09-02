from scipy.sparse import coo_matrix
from scipy import integrate
import scipy.sparse.linalg as linalg
import numpy as np
import time

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
        if sourceID==targetID: # diagonal elements
            return
        # hermit conjugate
        self.sourceList.append(targetID)
        self.targetList.append(sourceID)
        self.tList.append(tk.conjugate())
        return

    def buildHkFromHr(self,Hr,kpt,getHk=False):
        for bondDigital in Hr.bondList:
            #print(bondDigital.source,bondDigital.target,bondDigital.overlat)
            if bondDigital.source<=bondDigital.target: # upper triangular matrix
                self.addHopping(bondDigital.source,bondDigital.target,bondDigital.t,overLat=bondDigital.overlat,kpt=kpt)
        if getHk:
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