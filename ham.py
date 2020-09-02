from scipy.sparse import coo_matrix
import scipy.sparse.linalg as linalg
import numpy as np

class Ham:
    def __init__(self):
        self.sourceList=[]
        self.targetList=[]
        self.tList=[]

    def addHopping(self,sourceID,targetID,t):
        if True:#sourceID<=targetID:
            self.sourceList.append(sourceID)
            self.targetList.append(targetID)
        #else:
            self.sourceList.append(targetID)
            self.targetList.append(sourceID)
        self.tList.append(t)
        self.tList.append(t)

    def solve(self,k=1,which='LM',v0=None,maxiter=None,tol=1e-5,vec=False):
        self.buildCsrMatrix()
        return linalg.eigsh(self.csr,k=k,sigma=0,which='LM',v0=v0,maxiter=maxiter,tol=tol,return_eigenvectors=vec)
        #return np.linalg.eigh(self.csr)

    def buildCsrMatrix(self):
        realHopping=True
        for t in self.tList:
            if type(t).__name__=='complex':
                realHopping=False
                break
        if realHopping:
            coo=coo_matrix((self.tList,(self.sourceList,self.targetList)),dtype=float)
        else:
            coo=coo_matrix((self.tList,(self.sourceList,self.targetList)),dtype=complex)
        self.csr=coo.tocsr()

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
    
    

