from scipy.sparse import coo_matrix
import scipy.sparse.linalg as linalg
import numpy as np

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

    def buildHkFromHr(self,Hr,kpt):
        for bondDigital in Hr.bondList:
            #print(bondDigital.source,bondDigital.target,bondDigital.overlat)
            if bondDigital.source<=bondDigital.target: # upper triangular matrix
                self.addHopping(bondDigital.source,bondDigital.target,bondDigital.t,overLat=bondDigital.overlat,kpt=kpt)

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



def getGreen_E(orbList,bondList,E=0.,filling=0.5,eta=0.1):
    totOrb=len(orbList)
    ham=Ham()
    for bond in bondList:
        ham.addHopping(bond.sourceOrb.id,bond.targetOrb.id,bond.t)
    for i in range(totOrb):
        ham.addHopping(i,i,0)

    eigList=ham.solve(k=int(totOrb*filling),which='SM')
    #print(eigList)
    #print(ham.csr)
    #exit()
    return sum([1./(E-eig+1j*eta) for eig in eigList])
    
    

