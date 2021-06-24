from scipy.sparse import coo_matrix
from scipy import integrate
import scipy.sparse.linalg as linalg
import numpy as np
np.set_printoptions(precision=6,suppress=True)
import time
import posReader as pR
import paramReader
import matplotlib.pyplot as plt
import auxiliary
from ctypes import c_double, c_int, CDLL, py_object, c_double, c_longlong

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
        self.sourceList=[]
        self.targetList=[]
        self.tList=[]
        for bondDigital in Hr.bondList:
            self.addHopping(bondDigital.source,bondDigital.target,bondDigital.t,overLat=bondDigital.overlat,kpt=kpt)
        if getHk:
            return coo_matrix((self.tList,(self.sourceList,self.targetList)),dtype=complex).toarray()
        return

    def build_Hk_And_DHk_FromHr(self,Hr,kpt):
        sourceList=[];targetList=[]
        t_Hk_List=[];t_DHk_Dk1_List=[];t_DHk_Dk2_List=[];t_DHk_Dk3_List=[]
        for bondDigital in Hr.bondList:
            sourceList.append(bondDigital.source)
            targetList.append(bondDigital.target)
            t_Hk=bondDigital.t*np.exp(-2j*np.pi*np.dot(bondDigital.overlat,kpt))
            t_DHk_Dk1=t_Hk*(-2j*np.pi)*bondDigital.overlat[0]
            t_DHk_Dk2=t_Hk*(-2j*np.pi)*bondDigital.overlat[1]
            t_DHk_Dk3=t_Hk*(-2j*np.pi)*bondDigital.overlat[2]
            t_Hk_List.append(t_Hk)
            t_DHk_Dk1_List.append(t_DHk_Dk1)
            t_DHk_Dk2_List.append(t_DHk_Dk2)
            t_DHk_Dk3_List.append(t_DHk_Dk3)
        
        Hk=coo_matrix((t_Hk_List,(sourceList,targetList)),dtype=complex).toarray()
        DHk_Dk1=coo_matrix((t_DHk_Dk1_List,(sourceList,targetList)),dtype=complex).toarray()
        DHk_Dk2=coo_matrix((t_DHk_Dk2_List,(sourceList,targetList)),dtype=complex).toarray()
        DHk_Dk3=coo_matrix((t_DHk_Dk3_List,(sourceList,targetList)),dtype=complex).toarray()
        return Hk,DHk_Dk1,DHk_Dk2,DHk_Dk3
    
    def calcBerryCurvature(self,Hk,DHk_D1,DHk_D2,occupiedBandIndex=[0],unoccupiedBandIndex=[1]):
        eig, vec=np.linalg.eigh(Hk,'U')
        #print(eig)
        sortIndex=auxiliary.quicksort(eig)
        curvature=0
        #print(Hk)
        #print(DHk_D1)
        #print(DHk_D2)
        #print(eig,sortIndex)
        # refer to notes Berry curvature method for more details
        for iocc in occupiedBandIndex:
            state_n=vec[:,sortIndex[iocc]]
            energy_n=eig[iocc]
            for iunocc in unoccupiedBandIndex:
                state_m=vec[:,sortIndex[iunocc]]
                energy_m=eig[iunocc]
                d_energy=energy_m-energy_n
                assert abs(d_energy)>1e-5, "cannot compute Berry curvature at degenerated point"
                    
                curvature+=-2*(state_n.dot(DHk_D1).dot(state_m)*state_m.dot(DHk_D2).dot(state_n)).imag/(d_energy*d_energy)
        return curvature

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

class Response:
    def __init__(self) -> None:
        pass

    def calcBerryCurvBands_CDLL(self,hr,mainDir=None,fermi=0,saveFig=True,kplane=[[1,0,0],[0,1,0]],nthreads=1):
        # prepare input parameters for cdll
        kaxis=[]
        time0=time.time()
        if mainDir!=None: # read in kpath
            #kmesh=[]
            with open(mainDir+'KPOINTS','r') as fin:
                data=fin.read().split('\n')
                nkpts=int(data[1])
                c_kmesh=(c_double*(nkpts*3))()
                for iline in range(3,3+nkpts):
                    ikpt=iline-3
                    kaxis.append(ikpt)
                    k1,k2,k3,weight=data[iline].split()
                    #kmesh.append(np.array([float(k1),float(k2),float(k3)]))
                    c_kmesh[ikpt*3+0],c_kmesh[ikpt*3+1],c_kmesh[ikpt*3+2]=c_double(float(k1)),c_double(float(k2)),c_double(float(k3))
        else:
            kmesh=pR.autoGenerateKpath2D(onlyGetKPTList=True)
            nkpts=len(kmesh)
            c_kmesh=(c_double*(nkpts*3))()
            for ikpt, kpt in enumerate(kmesh):
                kaxis.append(ikpt)
                k1,k2,k3=kpt
                c_kmesh[ikpt*3+0],c_kmesh[ikpt*3+1],c_kmesh[ikpt*3+2]=c_double(k1),c_double(k2),c_double(k3)
        
        nhoppings=len(hr.bondList)
        c_sourceID=(c_int*nhoppings)()
        c_targetID=(c_int*nhoppings)()
        c_overLat=(c_int*(nhoppings*3))()
        c_hopping_real=(c_double*nhoppings)()
        c_hopping_imag=(c_double*nhoppings)()
        for ibond, bond in enumerate(hr.bondList):
            c_sourceID[ibond]=c_int(bond.source)
            c_targetID[ibond]=c_int(bond.target)
            for i in range(3):
                c_overLat[ibond*3+i]=c_int(bond.overlat[i])
            c_hopping_real[ibond]=c_double(bond.t.real)
            c_hopping_imag[ibond]=c_double(bond.t.imag)
        
        hamDLL=CDLL("./ham.dll")
        hamDLL.calcFullBandsAndVecs.restype=py_object
        result=hamDLL.calcFullBandsAndVecs(c_int(nkpts),c_kmesh,
                        c_int(hr.norb),c_int(nhoppings),
                        c_sourceID,c_targetID,c_overLat,c_hopping_real,c_hopping_imag,
                        c_int(nthreads))
        #print(result.shape)
        _eigList,_vec_r,_vec_i,Hk_r,Hk_i,Hk_dk1_r,Hk_dk1_i,Hk_dk2_r,Hk_dk2_i,Hk_dk3_r,Hk_dk3_i=result
        
        eigList=np.array(_eigList).reshape(nkpts,hr.norb)
        vec=(np.array(_vec_r)+1j*np.array(_vec_i)).reshape(nkpts,hr.norb,hr.norb)
        Hk=(np.array(Hk_r)+1j*np.array(Hk_i)).reshape(nkpts,hr.norb,hr.norb)
        DHk_dk1=(np.array(Hk_dk1_r)+1j*np.array(Hk_dk1_i)).reshape(nkpts,hr.norb,hr.norb)
        DHk_dk2=(np.array(Hk_dk2_r)+1j*np.array(Hk_dk2_i)).reshape(nkpts,hr.norb,hr.norb)
        DHk_dk3=(np.array(Hk_dk3_r)+1j*np.array(Hk_dk3_i)).reshape(nkpts,hr.norb,hr.norb)

        bandmin, bandmax=np.min(eigList), np.max(eigList)
        print('Spectra solved in %.3fs with %d thread(s), band range: %.4f~%.4f'%(time.time()-time0,nthreads,bandmin,bandmax))
        
        # construct Pauli matrix
        sigma_x, sigma_y, sigma_z= np.zeros((hr.norb,hr.norb),dtype=complex), np.zeros((hr.norb,hr.norb),dtype=complex), np.zeros((hr.norb,hr.norb),dtype=complex)
        for iorb in range(hr.norb//2):
            sigma_x[iorb,iorb+hr.norb//2],sigma_x[iorb+hr.norb//2,iorb]=1, 1
            sigma_y[iorb,iorb+hr.norb//2],sigma_y[iorb+hr.norb//2,iorb]=-1j, 1j
            sigma_z[iorb,iorb],sigma_z[iorb+hr.norb//2,iorb+hr.norb//2]=1, -1

        # transform DHk_dk to xyz coordinates
        
        kvec1=np.array(kplane[0])
        kvec2=np.array(kplane[1])
        kvec3=np.zeros(3)
        kvec3[0]=kvec1[1]*kvec2[2]-kvec1[2]*kvec2[1]
        kvec3[1]=kvec1[2]*kvec2[0]-kvec1[0]*kvec2[2]
        kvec3[2]=kvec1[0]*kvec2[1]-kvec1[1]*kvec2[0]
        kvec3_len=np.sqrt(np.dot(kvec3,kvec3))
        assert kvec3_len>1e-5, "Error, kplane was expanded by parallel vectros"
        kvec3/=kvec3_len

        print("kplane in cartesian coordinates:")
        print("a1=(%11.6f %11.6f %11.6f)"%(*kvec1,))
        print("a2=(%11.6f %11.6f %11.6f)"%(*kvec2,))
        print("a3=(%11.6f %11.6f %11.6f)"%(*kvec3,))

        transFrame=np.dot(pR.lat,np.array([kvec1,kvec2,kvec3]).transpose())
        
        a1, a2, a3=transFrame[0], transFrame[1], transFrame[2]
        S=np.linalg.det(transFrame[:2,:2])
        
        DHk_dkx=DHk_dk1#DHk_dk1*a1[0]+DHk_dk2*a2[0]+DHk_dk3*a3[0]
        DHk_dky=DHk_dk2#DHk_dk1*a1[1]+DHk_dk2*a2[1]+DHk_dk3*a3[1]

        kvecn=kvec3
        sigma_i=kvecn[0]*sigma_x+kvecn[1]*sigma_y+kvecn[2]*sigma_z
        berryCurv=[]
        for ikpt in range(nkpts):
            c_xyz=0
            DHk_dkx_commute_sigma=sigma_i.dot(DHk_dkx[ikpt])+DHk_dkx[ikpt].dot(sigma_i)
            for band_n in range(hr.norb): # loop over side band n
                En=eigList[ikpt,band_n]
                if En>fermi: continue
                vn=vec[ikpt][:,band_n]
                for band_m in range(hr.norb): # loop over intermedial band m
                    Em=eigList[ikpt,band_m]
                    if Em<=fermi: continue
                    #if abs(Em-En)<1e-5: continue
                    vm=vec[ikpt][:,band_m]
                    vnm_x_z=(vn.conjugate()).dot(DHk_dkx_commute_sigma).dot(vm)
                    vmn_y=(vm.conjugate()).dot(DHk_dky[ikpt]).dot(vn)
                    c_xyz+=((vnm_x_z*vmn_y).imag/(Em-En)**2)/2
            c_xyz/=(2*np.pi*S)
            #plt.scatter(ikpt,c_xyz,c='black')
            berryCurv.append(c_xyz)
        #plt.plot([paramReader.FERMI-paramReader.DFERMI,paramReader.FERMI+paramReader.DFERMI],[0,0],'--',linewidth=3)
        
        plt.figure()
        plt.plot([ikpt for ikpt in range(nkpts)],berryCurv)
        plt.xlim(0,nkpts-1)
        plt.xlabel(r'kpath')
        plt.ylabel(r'$e^2/\hbar$')
        plt.show()

        return
    
    def calcSpinHallConductor_CDLL_pyAchive(self,hr,mainDir=None,kmesh=[20,20,1],kplane=[[1,0,0],[0,1,0]],nthreads=1):
        time0=time.time()
        # prepare input parameters for cdll
        kxaxis=np.linspace(0,1,kmesh[0],endpoint=False)
        kyaxis=np.linspace(0,1,kmesh[1],endpoint=False)
        kzaxis=np.linspace(0,1,kmesh[2],endpoint=False)
        kmesh=np.array([[k1,k2,k3] for k3 in kzaxis for k2 in kyaxis for k1 in kxaxis])
        nkpts=len(kmesh)
        c_kmesh=(c_double*(nkpts*3))()
        for ikpt, kpt in enumerate(kmesh):
            k1,k2,k3=kpt
            c_kmesh[ikpt*3+0],c_kmesh[ikpt*3+1],c_kmesh[ikpt*3+2]=c_double(k1),c_double(k2),c_double(k3)
        
        nhoppings=len(hr.bondList)
        c_sourceID=(c_int*nhoppings)()
        c_targetID=(c_int*nhoppings)()
        c_overLat=(c_int*(nhoppings*3))()
        c_hopping_real=(c_double*nhoppings)()
        c_hopping_imag=(c_double*nhoppings)()
        for ibond, bond in enumerate(hr.bondList):
            c_sourceID[ibond]=c_int(bond.source)
            c_targetID[ibond]=c_int(bond.target)
            for i in range(3):
                c_overLat[ibond*3+i]=c_int(bond.overlat[i])
            c_hopping_real[ibond]=c_double(bond.t.real)
            c_hopping_imag[ibond]=c_double(bond.t.imag)
        
        hamDLL=CDLL("./ham.dll")
        hamDLL.calcFullBandsAndVecs.restype=py_object
        
        result=hamDLL.calcFullBandsAndVecs(c_int(nkpts),c_kmesh,
                        c_int(hr.norb),c_int(nhoppings),
                        c_sourceID,c_targetID,c_overLat,c_hopping_real,c_hopping_imag,
                        c_int(nthreads))
        #print(result.shape)
        _eigList,_vec_r,_vec_i,Hk_r,Hk_i,Hk_dk1_r,Hk_dk1_i,Hk_dk2_r,Hk_dk2_i,Hk_dk3_r,Hk_dk3_i=result
        
        eigList=np.array(_eigList).reshape(nkpts,hr.norb)
        vec=(np.array(_vec_r)+1j*np.array(_vec_i)).reshape(nkpts,hr.norb,hr.norb)
        Hk=(np.array(Hk_r)+1j*np.array(Hk_i)).reshape(nkpts,hr.norb,hr.norb)
        DHk_dk1=(np.array(Hk_dk1_r)+1j*np.array(Hk_dk1_i)).reshape(nkpts,hr.norb,hr.norb)
        DHk_dk2=(np.array(Hk_dk2_r)+1j*np.array(Hk_dk2_i)).reshape(nkpts,hr.norb,hr.norb)
        DHk_dk3=(np.array(Hk_dk3_r)+1j*np.array(Hk_dk3_i)).reshape(nkpts,hr.norb,hr.norb)

        bandmin, bandmax=np.min(eigList), np.max(eigList)
        print('Spectra solved in %.3fs with %d thread(s), band range: %.4f~%.4f'%(time.time()-time0,nthreads,bandmin,bandmax))
        
        # construct Pauli matrix
        sigma_x, sigma_y, sigma_z= np.zeros((hr.norb,hr.norb),dtype=complex), np.zeros((hr.norb,hr.norb),dtype=complex), np.zeros((hr.norb,hr.norb),dtype=complex)
        for iorb in range(hr.norb//2):
            sigma_x[iorb,iorb+hr.norb//2],sigma_x[iorb+hr.norb//2,iorb]=1, 1
            sigma_y[iorb,iorb+hr.norb//2],sigma_y[iorb+hr.norb//2,iorb]=-1j, 1j
            sigma_z[iorb,iorb],sigma_z[iorb+hr.norb//2,iorb+hr.norb//2]=1, -1

        # transform DHk_dk to xyz coordinates
        print("lattice:")
        print("a1=(%11.6f %11.6f %11.6f)"%(*pR.lat[0],))
        print("a2=(%11.6f %11.6f %11.6f)"%(*pR.lat[1],))
        print("a3=(%11.6f %11.6f %11.6f)"%(*pR.lat[2],))

        kvec1=np.array(kplane[0])
        kvec2=np.array(kplane[1])
        kvec3=np.zeros(3)
        kvec3[0]=kvec1[1]*kvec2[2]-kvec1[2]*kvec2[1]
        kvec3[1]=kvec1[2]*kvec2[0]-kvec1[0]*kvec2[2]
        kvec3[2]=kvec1[0]*kvec2[1]-kvec1[1]*kvec2[0]
        kvec3_len=np.sqrt(np.dot(kvec3,kvec3))
        assert kvec3_len>1e-5, "Error, kplane was expanded by parallel vectros"
        kvec3/=kvec3_len

        print("kplane in cartesian coordinates:")
        print("a1=(%11.6f %11.6f %11.6f)"%(*kvec1,))
        print("a2=(%11.6f %11.6f %11.6f)"%(*kvec2,))
        print("a3=(%11.6f %11.6f %11.6f)"%(*kvec3,))

        Vbz=np.linalg.det(np.linalg.inv(pR.lat))

        transFrame=np.dot(pR.lat,np.array([kvec1,kvec2,kvec3]).transpose())
        
        a1, a2, a3=transFrame[0], transFrame[1], transFrame[2]
        #S=np.linalg.det(transFrame[:2,:2])
        DHk_dkx=DHk_dk1*a1[0]+DHk_dk2*a2[0]+DHk_dk3*a3[0]
        DHk_dky=DHk_dk1*a1[1]+DHk_dk2*a2[1]+DHk_dk3*a3[1]

        kvecn=kvec3
        sigma_i=kvecn[0]*sigma_x+kvecn[1]*sigma_y+kvecn[2]*sigma_z
        fermiList=[]
        condList=[]
        for ifermi, fermi in enumerate(np.linspace(paramReader.FERMI-paramReader.DFERMI,
                            paramReader.FERMI+paramReader.DFERMI,
                            paramReader.NFERMI,endpoint=False)): # loop over fermi energy
            #print(ifermi)
            #fermi=0
            fermiList.append(fermi-paramReader.FERMI)
            c_xyz=0
            for ikpt in range(nkpts): # loop over kpoints
                DHk_dkx_commute_sigma=sigma_i.dot(DHk_dkx[ikpt])+DHk_dkx[ikpt].dot(sigma_i)
                #print("kpt%d [%.1f %.1f %.1f]"%(ikpt,*kmesh[ikpt]))
                #print("DHk/Dy:")
                #print(DHk_dky[ikpt])
                #print("S dot DHk/Dx:")
                #print(DHk_dkx_commute_sigma)
                for band_n in range(hr.norb): # loop over side band n
                    En=eigList[ikpt,band_n]
                    if En>fermi: continue
                    vn=vec[ikpt][:,band_n]
                    #print("valence band %d, energy: %.3f, vec: "%(band_n,En),vn)
                    VnDotDhDx=vn.conjugate().dot(DHk_dkx_commute_sigma)
                    DhDyDotVn=DHk_dky[ikpt].dot(vn)
                    #print("n dot SDh/Dx",VnDotDhDx)
                    #print("Dh/Dy dot n",DhDyDotVn)
                    for band_m in range(hr.norb): # loop over intermedial band m
                        Em=eigList[ikpt,band_m]
                        if Em<=fermi: continue
                        if abs(Em-En)<1e-2: continue
                        vm=vec[ikpt][:,band_m]
                        #print("conduct band %d, energy: %.3f, vec: "%(band_m,Em),vm)
                        vnm_x_z=VnDotDhDx.dot(vm)#(vn.conjugate()).dot(DHk_dkx_commute_sigma).dot(vm)
                        vmn_y=vm.conjugate().dot(DhDyDotVn)#(vm.conjugate()).dot(DHk_dky[ikpt]).dot(vn)
                        #print(vnm_x_z,vmn_y)
                        c_xyz+=((vnm_x_z*vmn_y).imag/(Em-En)**2)/2
            c_xyz*=(1./nkpts/np.pi/2)#/S)
            #print(c_xyz)
            condList.append(c_xyz)
        
        plt.figure()
        plt.scatter(fermiList,condList,c='black')
        plt.plot([-paramReader.DFERMI,paramReader.DFERMI],[0,0],'--',linewidth=3)
        plt.xlabel(r'$E-E_f\ (eV)$')
        plt.ylabel(r'$e^2/\hbar$')
        plt.show()

        filePath=mainDir+'spinHallCond.txt' if mainDir else './spinHallCond.txt'
        with open(filePath,'w') as fout:
            fout.write("#Fermi  SPC\n")
            for fermi, cond in zip(fermiList, condList):
                fout.write("%.6f %.6f \n"%(fermi,cond))

        return
    
    def calcSpinHallConductor_CDLL(self,hr,mainDir=None,kgrid=[20,20,1],kplane=[[1,0,0],[0,1,0]],nMaxRefine=8,summedCvgCriteria=0.1,nthreads=1):
        time0=time.time()
        # prepare input parameters for cdll
        kxaxis=np.linspace(0,1,kgrid[0],endpoint=False)
        kyaxis=np.linspace(0,1,kgrid[1],endpoint=False)
        kzaxis=np.linspace(0,1,kgrid[2],endpoint=False)
        kmesh=np.array([[k1,k2,k3] for k3 in kzaxis for k2 in kyaxis for k1 in kxaxis])

        # transform DHk_dk to xyz coordinates
        print("lattice:")
        print("a1=(%11.6f %11.6f %11.6f)"%(*pR.lat[0],))
        print("a2=(%11.6f %11.6f %11.6f)"%(*pR.lat[1],))
        print("a3=(%11.6f %11.6f %11.6f)"%(*pR.lat[2],))

        kvec1=np.array(kplane[0])
        kvec2=np.array(kplane[1])
        kvec3=np.zeros(3)
        kvec3[0]=kvec1[1]*kvec2[2]-kvec1[2]*kvec2[1]
        kvec3[1]=kvec1[2]*kvec2[0]-kvec1[0]*kvec2[2]
        kvec3[2]=kvec1[0]*kvec2[1]-kvec1[1]*kvec2[0]
        kvec3_len=np.sqrt(np.dot(kvec3,kvec3))
        assert kvec3_len>1e-5, "Error, kplane was expanded by parallel vectros"
        kvec3/=kvec3_len

        print("kplane in cartesian coordinates:")
        print("a1=(%11.6f %11.6f %11.6f)"%(*kvec1,))
        print("a2=(%11.6f %11.6f %11.6f)"%(*kvec2,))
        print("a3=(%11.6f %11.6f %11.6f)"%(*kvec3,))

        Vbz=np.linalg.det(np.linalg.inv(pR.lat))

        transFrame=np.dot(pR.lat,np.array([kvec1,kvec2,kvec3]).transpose())
        kvecn=kvec3
        c_transFrame=(c_double*9)()
        for i in range(3):
            c_transFrame[i]=c_double(transFrame[0][i])
            c_transFrame[i+3]=c_double(transFrame[1][i])
            c_transFrame[i+6]=c_double(kvecn[i])

        nkpts=len(kmesh)
        c_kmesh=(c_double*(nkpts*3))()
        for ikpt, kpt in enumerate(kmesh):
            k1,k2,k3=kpt
            c_kmesh[ikpt*3+0],c_kmesh[ikpt*3+1],c_kmesh[ikpt*3+2]=c_double(k1),c_double(k2),c_double(k3)
        
        c_dk=(c_double*3)()
        for ik, nk in enumerate(kgrid):
            c_dk[ik]=c_double(1/nk)
        #exit()

        nhoppings=len(hr.bondList)
        c_sourceID=(c_int*nhoppings)()
        c_targetID=(c_int*nhoppings)()
        c_overLat=(c_int*(nhoppings*3))()
        c_hopping_real=(c_double*nhoppings)()
        c_hopping_imag=(c_double*nhoppings)()
        for ibond, bond in enumerate(hr.bondList):
            c_sourceID[ibond]=c_int(bond.source)
            c_targetID[ibond]=c_int(bond.target)
            for i in range(3):
                c_overLat[ibond*3+i]=c_int(bond.overlat[i])
            c_hopping_real[ibond]=c_double(bond.t.real)
            c_hopping_imag[ibond]=c_double(bond.t.imag)
        
        c_fermi_list=(c_double*paramReader.NFERMI)()
        fermiList=[]
        for ifermi, fermi in enumerate(np.linspace(paramReader.FERMI-paramReader.DFERMI,
                                paramReader.FERMI+paramReader.DFERMI,
                                paramReader.NFERMI,endpoint=False)):
            fermiList.append(fermi-paramReader.FERMI)
            c_fermi_list[ifermi]=c_double(fermi)

        hamDLL=CDLL("./ham.dll")
        hamDLL.calcSpinHallCond.restype=py_object

        degenerateThreshold=0.01
        result=hamDLL.calcSpinHallCond(c_int(nkpts),c_kmesh,c_dk,c_int(nMaxRefine),
                        c_int(hr.norb),c_int(nhoppings),
                        c_sourceID,c_targetID,c_overLat,c_hopping_real,c_hopping_imag,
                        c_int(paramReader.NFERMI),
                        c_fermi_list,
                        c_transFrame,
                        c_double(degenerateThreshold),
                        c_double(summedCvgCriteria),
                        c_int(nthreads))
            
        condList=[]
        spinBerryCurv=np.array(result).reshape(paramReader.NFERMI,nkpts)
        condList=[np.sum(spinBerryCurv_onFermi)/nkpts/np.pi/2 for spinBerryCurv_onFermi in spinBerryCurv]
        print('Spin Berry curv. solved in %.3fs with %d thread(s)'%(time.time()-time0,nthreads))
        
        plt.figure()
        plt.scatter(fermiList,condList,c='black')
        plt.plot([-paramReader.DFERMI,paramReader.DFERMI],[0,0],'--',linewidth=3)
        plt.xlabel(r'$E-E_f\ (eV)$')
        plt.ylabel(r'$e^2/\hbar$')
        plt.show()

        filePath=mainDir+'spinHallCond.txt' if mainDir else './spinHallCond.txt'
        with open(filePath,'w') as fout:
            fout.write("#Fermi  SPC\n")
            for fermi, cond in zip(fermiList, condList):
                fout.write("%.6f %.6f \n"%(fermi,cond))

        return
class Green:

    def __init__(self):
        pass

    def calcBands(self,hr,orbList=[0],spin_polarized=False,fermi=0,eta=0.1,resolution=100,mainDir=None):
        norb=hr.norb
        if len(orbList)==0:
            orbList=[i for i in range(norb)]
        if spin_polarized:
            orbUpList=orbList[:len(orbList)//2]
            orbDnList=orbList[len(orbList)//2:]
        HkList=[]
        
        time0=time.time()
        if mainDir!=None: # read in kpath
            kmesh=[]
            with open(mainDir+'KPOINTS','r') as fin:
                data=fin.read().split('\n')
                nkpts=int(data[1])
                for iline in range(3,3+nkpts):
                    k1,k2,k3,weight=data[iline].split()
                    kmesh.append(np.array([float(k1),float(k2),float(k3)]))
        else:
            kmesh=pR.autoGenerateKpath2D(onlyGetKPTList=True)
            nkpts=len(kmesh)

        for ikpt, kpt in enumerate(kmesh):
            _ham=Ham()
            Hk=_ham.buildHkFromHr(hr,kpt,getHk=True)
            HkList.append(Hk)
            
        print('Hk constructed in %.3fs'%(time.time()-time0))
        energyAXIS=np.linspace(fermi-2,fermi+2,resolution)
        '''
        if spin_polarized:
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
            return
        '''
        DOS=[]
        for z in energyAXIS:
            _dos_k=[]
            for _hk in HkList:
                gk=np.linalg.inv((z+eta*1j)*np.eye(norb)-_hk)
                if spin_polarized:
                    _dos_k.append(sum(np.diagonal(gk)[orbUpList]-np.diagonal(gk)[orbDnList]).imag/(-np.pi))
                else:
                    _dos_k.append(sum(np.diagonal(gk)[orbList]).imag/(-np.pi))
            DOS.append(_dos_k)
        plt.figure()
        plt.imshow(DOS,extent=(-5,5,-5,5),origin='lower')
        plt.savefig(mainDir+'KDOS.png')
        plt.show()
        plt.close()
                
    def calcFullDOS(self,hr,fermi=0,eta=0.01,NEDOS=1000,kmesh=[],mainDir=None,saveFig=True):
        norb=hr.norb
        EkList=[]
        time0=time.time()
        kmesh=pR.createKPOINTS(getKMesh=True) if len(kmesh)==0 else kmesh
        for ikpt, kpt in enumerate(kmesh):
            _ham=Ham()
            _ham.buildHkFromHr(hr,kpt,getHk=True)
            eigs=_ham.sovle_exact()
            EkList.extend(eigs)
        Emin, Emax = np.min(EkList), np.max(EkList)

        eigs_flat=list(EkList)
        auxiliary.quicksort(eigs_flat)
        #print(eigs_flat)
        for ieig, eig in enumerate(eigs_flat):
            if eig>fermi:
                break
        occupationNumber=int(np.around(1.0*ieig/(ikpt+1)))*(ikpt+1)
        fermi_adjusted=(eigs_flat[occupationNumber]+eigs_flat[occupationNumber-1])/2

        print('Spectra solved in %.3fs'%(time.time()-time0))
        print('band range: %.6f-%.6f'%(Emin,Emax))
        print('level%d %.4f < fermi: %.4f < level%d %.4f'%(ieig,eigs_flat[ieig-1],fermi,ieig+1,eigs_flat[ieig]))
        print('(not use now) occupied %.3f states per kpt --> modified to %d occupation --> %.4f < adjust fermi: %.4f < %.4f'%(1.0*ieig/(ikpt+1),occupationNumber,eigs_flat[occupationNumber-1],fermi_adjusted,eigs_flat[occupationNumber]))
        #exit()

        enhancementForEnergyRange=(Emax-Emin)*0.05
        energyAXIS=np.linspace(Emin-enhancementForEnergyRange,Emax+enhancementForEnergyRange,NEDOS)
        DOS=[]
        for z in energyAXIS:
            _dos_tmp=0
            for energy in EkList:
                gk=1./(z-energy+eta*1j)
                _dos_tmp+=gk.imag/(-np.pi)
            DOS.append(_dos_tmp)
        ymax=np.max(DOS)*1.05
        plt.figure()
        plt.plot(energyAXIS,DOS)
        plt.plot([fermi,fermi],[0,ymax],'--')
        plt.xlim(Emin-enhancementForEnergyRange,Emax+enhancementForEnergyRange)
        plt.ylim(0,ymax)
        plt.yticks([])
        plt.ylabel(r'DOS')
        plt.xlabel(r'Energy')
        if saveFig:
            assert mainDir!=None, 'We need path or directory to save the figure'
            plt.savefig(mainDir+'DOS.png')
        plt.show()
        plt.close()
        return Emin, Emax, fermi_adjusted
    
    def calcFullDOS_CDLL(self,hr,c_Hk_address,nkpt,norb,fermi=0,eta=0.01,NEDOS=1000,mainDir=None,saveFig=True,nthreads=1):
        time0=time.time()
        # show time for c
        hamDLL=CDLL("./ham.dll")
        hamDLL.calcFullDOS.restype=py_object
        result=hamDLL.calcFullDOS(c_longlong(c_Hk_address),nkpt,c_int(norb),c_int(NEDOS),c_double(eta),c_double(fermi),c_int(nthreads))
        energyAXIS, dosAxis, optimal_fermi=result
        Emin,Emax=energyAXIS[0],energyAXIS[NEDOS-1]
        print("cdll calc in %.6fs Emin=%.6f, Emax=%.6f, optimal fermi=%.6f"%(time.time()-time0,Emin,Emax,optimal_fermi))
        # graphical work
        ymax=np.max(dosAxis)*1.05
        plt.figure()
        plt.plot(energyAXIS,dosAxis)
        plt.plot([fermi,fermi],[0,ymax],'--',label=r'original')
        plt.plot([optimal_fermi,optimal_fermi],[0,ymax],'--',label=r'optimal')
        plt.ylim(0,ymax)
        plt.yticks([])
        plt.ylabel(r'DOS')
        plt.xlabel(r'Energy')
        plt.legend()
        if saveFig:
            assert mainDir!=None, 'We need path or directory to save the figure'
            plt.savefig(mainDir+'DOS.png')
        plt.show()
        plt.close()
        return Emin, Emax, optimal_fermi

    def calcFullBands(self,hr,fermi=0,mainDir=None,saveFig=True):
        eigList=[]
        time0=time.time()
        if mainDir!=None: # read in kpath
            kmesh=[]
            with open(mainDir+'KPOINTS','r') as fin:
                data=fin.read().split('\n')
                nkpts=int(data[1])
                for iline in range(3,3+nkpts):
                    k1,k2,k3,weight=data[iline].split()
                    kmesh.append(np.array([float(k1),float(k2),float(k3)]))
        else:
            kmesh=pR.autoGenerateKpath2D(onlyGetKPTList=True)
            nkpts=len(kmesh)
        kaxis=[]
        for ikpt, kpt in enumerate(kmesh):
            _ham=Ham()
            _ham.buildHkFromHr(hr,kpt,getHk=True)
            eigList.append(_ham.sovle_exact())
            kaxis.append(ikpt)
        
        eigList=np.array(eigList)
        print('Hk solved in %.3fs'%(time.time()-time0))
        nbands=len(eigList[0])
        plt.figure()
        for ibands in range(nbands):
            plt.plot(kaxis,eigList[:,ibands],'-',c='blue')
        plt.plot([0,ikpt],[fermi,fermi],'--',c='orange')
        plt.xlim(0,ikpt)
        plt.ylim(fermi-3,fermi+2)
        plt.xticks([])
        plt.xlabel('K')
        plt.ylabel('Energy')
        if saveFig:
            assert mainDir!=None, 'We need path or directory to save the figure'
            plt.savefig(mainDir+'Bands.png')

            with open(mainDir+'tbBands.txt','w') as fout:
                fout.write('#k energy\n')
                for ibands in range(nbands):
                    for ikpt in range(nkpts):
                        fout.write('%d %.6f\n'%(ikpt,eigList[ikpt,ibands]))
                    fout.write('\n')
        plt.show()
        plt.close()

    def calcFullBands_CDLL(self,hr,fermi=0,optimalFermi=-100,mainDir=None,saveFig=True,nthreads=1):
        # prepare input parameters for cdll
        kaxis=[]
        time0=time.time()
        if mainDir!=None: # read in kpath
            #kmesh=[]
            with open(mainDir+'KPOINTS','r') as fin:
                data=fin.read().split('\n')
                nkpts=int(data[1])
                c_kmesh=(c_double*(nkpts*3))()
                for iline in range(3,3+nkpts):
                    ikpt=iline-3
                    kaxis.append(ikpt)
                    k1,k2,k3,weight=data[iline].split()
                    #kmesh.append(np.array([float(k1),float(k2),float(k3)]))
                    c_kmesh[ikpt*3+0],c_kmesh[ikpt*3+1],c_kmesh[ikpt*3+2]=c_double(float(k1)),c_double(float(k2)),c_double(float(k3))
        else:
            kmesh=pR.autoGenerateKpath2D(onlyGetKPTList=True)
            nkpts=len(kmesh)
            c_kmesh=(c_double*(nkpts*3))()
            for ikpt, kpt in enumerate(kmesh):
                kaxis.append(ikpt)
                k1,k2,k3=kpt
                c_kmesh[ikpt*3+0],c_kmesh[ikpt*3+1],c_kmesh[ikpt*3+2]=c_double(k1),c_double(k2),c_double(k3)
        
        nhoppings=len(hr.bondList)
        c_sourceID=(c_int*nhoppings)()
        c_targetID=(c_int*nhoppings)()
        c_overLat=(c_int*(nhoppings*3))()
        c_hopping_real=(c_double*nhoppings)()
        c_hopping_imag=(c_double*nhoppings)()
        for ibond, bond in enumerate(hr.bondList):
            #print("%3d %3d %3d %3d %3d %9.6f %9.6f"%(*bond.overlat,bond.source+1,bond.target+1,bond.t.real,bond.t.imag))
            c_sourceID[ibond]=c_int(bond.source)
            c_targetID[ibond]=c_int(bond.target)
            for i in range(3):
                c_overLat[ibond*3+i]=c_int(bond.overlat[i])
            c_hopping_real[ibond]=c_double(bond.t.real)
            c_hopping_imag[ibond]=c_double(bond.t.imag)
        
        hamDLL=CDLL("./ham.dll")
        hamDLL.calcFullBandsAndVecs.restype=py_object
        result=hamDLL.calcFullBandsAndVecs(c_int(nkpts),c_kmesh,
                        c_int(hr.norb),c_int(nhoppings),
                        c_sourceID,c_targetID,c_overLat,c_hopping_real,c_hopping_imag,
                        c_int(nthreads))
        _eigList,_vec_r,_vec_i,Hk_r,Hk_i,Hk_dk1_r,Hk_dk1_i,Hk_dk2_r,Hk_dk2_i,Hk_dk3_r,Hk_dk3_i=result
        eigList=np.array(_eigList).reshape(nkpts,hr.norb)
        print('Spectra solved in %.3fs with %d threads band range: [%.6f,%.6f]'%(time.time()-time0,nthreads,np.max(eigList),np.min(eigList)))
        #nbands=eigList.shape[1]
        plt.figure()
        for ibands in range(hr.norb):
            plt.plot(kaxis,eigList[:,ibands],'-',c='blue')
        plt.plot([0,ikpt],[fermi,fermi],'--',c='orange')
        if(optimalFermi>-100): plt.plot([0,ikpt],[optimalFermi,optimalFermi],'--',c='green')
        plt.xlim(0,ikpt)
        plt.ylim(fermi-paramReader.DFERMI,fermi+paramReader.DFERMI)
        plt.xticks([])
        plt.xlabel('K')
        plt.ylabel('Energy')
        if saveFig:
            assert mainDir!=None, 'We need path or directory to save the figure'
            plt.savefig(mainDir+'Bands.png')

            with open(mainDir+'tbBands.txt','w') as fout:
                fout.write('#k energy\n')
                for ibands in range(hr.norb):
                    for ikpt in range(nkpts):
                        fout.write('%d %.6f\n'%(ikpt,eigList[ikpt,ibands]))
                    fout.write('\n')
        plt.show()
        plt.close()

    def calcOpticalHoppingTensor_CDLL(self,hr,mainDir=None,fermi=0,nk=100,ibandv=0,ibandc=1,nthreads=1):
        time0=time.time()
        # prepare input parameters for cdll
        kaxis=np.linspace(0,1,nk,endpoint=False)
        kmesh=np.array([[k1,k2,0] for k2 in kaxis for k1 in kaxis])
        nkpts=len(kmesh)
        c_kmesh=(c_double*(nkpts*3))()
        for ikpt, kpt in enumerate(kmesh):
            k1,k2,k3=kpt
            c_kmesh[ikpt*3+0],c_kmesh[ikpt*3+1],c_kmesh[ikpt*3+2]=c_double(k1),c_double(k2),c_double(k3)
        
        nhoppings=len(hr.bondList)
        c_sourceID=(c_int*nhoppings)()
        c_targetID=(c_int*nhoppings)()
        c_overLat=(c_int*(nhoppings*3))()
        c_hopping_real=(c_double*nhoppings)()
        c_hopping_imag=(c_double*nhoppings)()
        for ibond, bond in enumerate(hr.bondList):
            c_sourceID[ibond]=c_int(bond.source)
            c_targetID[ibond]=c_int(bond.target)
            for i in range(3):
                c_overLat[ibond*3+i]=c_int(bond.overlat[i])
            c_hopping_real[ibond]=c_double(bond.t.real)
            c_hopping_imag[ibond]=c_double(bond.t.imag)
        
        hamDLL=CDLL("./ham.dll")
        hamDLL.calcFullBandsAndVecs.restype=py_object
        result=hamDLL.calcFullBandsAndVecs(c_int(nkpts),c_kmesh,
                        c_int(hr.norb),c_int(nhoppings),
                        c_sourceID,c_targetID,c_overLat,c_hopping_real,c_hopping_imag,
                        c_int(nthreads))
        #print(result.shape)
        _eigList,_vec_r,_vec_i,Hk_r,Hk_i,Hk_dk1_r,Hk_dk1_i,Hk_dk2_r,Hk_dk2_i,Hk_dk3_r,Hk_dk3_i=result
        eigList=np.array(_eigList).reshape(nkpts,hr.norb)
        vec=(np.array(_vec_r)+1j*np.array(_vec_i)).reshape(nkpts,hr.norb,hr.norb)
        Hk=(np.array(Hk_r)+1j*np.array(Hk_i)).reshape(nkpts,hr.norb,hr.norb)
        DHk_dk1=(np.array(Hk_dk1_r)+1j*np.array(Hk_dk1_i)).reshape(nkpts,hr.norb,hr.norb)
        DHk_dk2=(np.array(Hk_dk2_r)+1j*np.array(Hk_dk2_i)).reshape(nkpts,hr.norb,hr.norb)
        DHk_dk3=(np.array(Hk_dk3_r)+1j*np.array(Hk_dk3_i)).reshape(nkpts,hr.norb,hr.norb)

        print('Spectra solved in %.3fs (serial mode)'%(time.time()-time0))
        
        # transform DHk_dk to xyz coordinates
        print("lattice:")
        print("a1=(%9.6f %9.6f %9.6f)"%(*pR.lat[0],))
        print("a2=(%9.6f %9.6f %9.6f)"%(*pR.lat[1],))
        print("a3=(%9.6f %9.6f %9.6f)"%(*pR.lat[2],))

        transFrame=pR.lat # trans matrix = Lattice matrix .dot. [k1_cart, k2_cart, k3_cart]
                          # k1_cart is the column vector of first direction of kplane in cart. coor., etc.
        a1, a2, a3=transFrame[0], transFrame[1], transFrame[2]
        DHk_dkx=DHk_dk1*a1[0]+DHk_dk2*a2[0]+DHk_dk3*a3[0]
        DHk_dky=DHk_dk1*a1[1]+DHk_dk2*a2[1]+DHk_dk3*a3[1]
        DHk_dkz=DHk_dk1*a1[2]+DHk_dk2*a2[2]+DHk_dk3*a3[2]

        time0=time.time()
        m,n=ibandv,ibandc
        filePath=mainDir+'opticalOverlap.txt' if mainDir else './opticalOverlap.txt'
        with open(filePath,'w') as fout:
            for ikpt, kpt in enumerate(kmesh):
                Em, En=eigList[ikpt][m], eigList[ikpt][n]

                vmn_x=vec[ikpt][:,m].conjugate().dot(DHk_dkx[ikpt]).dot(vec[ikpt][:,n])
                vmn_y=vec[ikpt][:,m].conjugate().dot(DHk_dky[ikpt]).dot(vec[ikpt][:,n])
                vmn_z=vec[ikpt][:,m].conjugate().dot(DHk_dkz[ikpt]).dot(vec[ikpt][:,n])

                vmm_x=vec[ikpt][:,m].conjugate().dot(DHk_dkx[ikpt]).dot(vec[ikpt][:,m])
                vmm_y=vec[ikpt][:,m].conjugate().dot(DHk_dky[ikpt]).dot(vec[ikpt][:,m])
                vmm_z=vec[ikpt][:,m].conjugate().dot(DHk_dkz[ikpt]).dot(vec[ikpt][:,m])

                vnn_x=vec[ikpt][:,n].conjugate().dot(DHk_dkx[ikpt]).dot(vec[ikpt][:,n])
                vnn_y=vec[ikpt][:,n].conjugate().dot(DHk_dky[ikpt]).dot(vec[ikpt][:,n])
                vnn_z=vec[ikpt][:,n].conjugate().dot(DHk_dkz[ikpt]).dot(vec[ikpt][:,n])

                fout.write("%.3f %.3f %.3f %.6f %.3f %.6f %.3f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n"%(
                    *kpt,Em,1.,En,0.,
                    vmn_x.real,vmn_x.imag,vmn_y.real,vmn_y.imag,vmn_z.real,vmn_z.imag,
                    vmm_x.real,vmm_x.imag,vmm_y.real,vmm_y.imag,vmm_z.real,vmm_z.imag,
                    vnn_x.real,vnn_x.imag,vnn_y.real,vnn_y.imag,vnn_z.real,vnn_z.imag,))
                
        print('overlap solved in %.3fs, data saved in %s'%(time.time()-time0,filePath))

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
            #print(_ham[iorb_source][iorb_target])
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
    def calcMatrixPerturbatedEnergyViaExact(self,hr,scatterList,eta=0.01,energyRange=(-5.0,0),Lx=1,Ly=1,Lz=1,getErr=False):
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
            #print('Ham_%d_%d = %.6f'%(iorb_source,iorb_target,_ham[iorb_source][iorb_target]))
            _ham[iorb_source][iorb_target]+=scatter[2]
            #print('Ham_%d_%d = %.6f'%(iorb_source,iorb_target,_ham[iorb_source][iorb_target]))
        
        r=(energyRange[1]-energyRange[0])/2
        o=(energyRange[0]+energyRange[1])/2
        def getEOnCircle(theta):
            z=r*np.exp(1j*theta)
            E=z+o
            G=np.linalg.inv((E+eta*1j)*np.eye(totOrb)-_ham)
            dos=sum(np.diagonal(G))/(-np.pi)#/totCell
            totalEnergy=E*dos
            return (totalEnergy*1j*z).imag
        
        
        #for theta in np.linspace(np.pi,0,501):
        #    print('%.6f %.6f'%(theta, getEOnCircle(theta)))
        inte,err=integrate.quad(getEOnCircle,a=np.pi,b=0.)
        if getErr:
            return inte, err
        return inte
        
    def calcMatrixPerturbatedEnergyViaFourier(self,hr,scatterList,eta=0.01,energyRange=(-5.0,0),Lx=1,Ly=1,Lz=1,globalCutOrder=8,diverseCutOrder=2,skipOdds=False,onlyPerturbatedEnergy=False,getErr=False):
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
        print('find scattering region with %d-dimension, basis:'%scatteringOrder)
        print(relevantOrbList)
        # build scattering matrix
        scatteringCenter=np.zeros((scatteringOrder,scatteringOrder))
        for scatter in scatterList:
            sourceID=findOrbIndexInRecordList(scatter[0])
            targetID=findOrbIndexInRecordList(scatter[1])
            scatteringCenter[sourceID,targetID]=scatter[2]
        #print('scattering matrix is:')
        #print(scatteringCenter)
        
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
                if globalCutOrder==0: # G0
                    rou+=sum(np.diagonal(_gk))
                    continue
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
                if not onlyPerturbatedEnergy:
                    rou+=sum(np.diagonal(_gk))
                #exit()
            
            if globalCutOrder==0: # G0 exact
                dos=rou/(-np.pi)#/totalWeight # DOS per cell
                totalEnergy=E*dos
                return (totalEnergy*1j*z).imag

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
            dos=rou/(-np.pi)#/totalWeight # DOS per cell
            totalEnergy=E*dos
            return (totalEnergy*1j*z).imag
        
        #for theta in np.linspace(np.pi,0,501):
        #    print('%.6f %.6f'%(theta, getEOnCircle(theta)))
        
        inte,err=integrate.quad(getEOnCircle,a=np.pi,b=0.)
        if getErr:
            return inte,err
        return inte

    def calcJ(self,hr,sourceOrbList=[0],sourceR=[0,0,0],targetOrbList=[0],targetR=[1,0,0],eta=0.0001,energyRange=(-5.0,0),Lx=5,Ly=5,Lz=1,globalCutOrder=8,diverseCutOrder=2,linearPerturb=True):
        '''Linear response approximation
           using Gaussian integration along semi-circle in upper complex plane
        '''
        assert len(sourceOrbList)==len(targetOrbList), 'Now, the coupling ions is assumed to be symmtric to each other'
        S=len(sourceOrbList)/2.
        norb=hr.norb
        halfNOrb=int(norb/2)
        onsite=np.array(hr.onsite)
        deltaList=onsite[:halfNOrb]-onsite[halfNOrb:]
        print('S=%.1f'%S)
        print('Exchange splitting are:')
        print(deltaList)
        
        # create scattering route
        # create scattering route
        scatterTheSameDirectionList=[]
        scatterTheOppoDirectionList=[]
        for iorb in sourceOrbList:
            scatterTheSameDirectionList.append([[np.array(sourceR),iorb],[np.array(sourceR),iorb+halfNOrb],deltaList[iorb]/2.])
            scatterTheSameDirectionList.append([[np.array(sourceR),iorb+halfNOrb],[np.array(sourceR),iorb],deltaList[iorb]/2.])
            scatterTheOppoDirectionList.append([[np.array(sourceR),iorb],[np.array(sourceR),iorb+halfNOrb],deltaList[iorb]/2.])
            scatterTheOppoDirectionList.append([[np.array(sourceR),iorb+halfNOrb],[np.array(sourceR),iorb],deltaList[iorb]/2.])
        
        for iorb in targetOrbList:
            scatterTheSameDirectionList.append([[np.array(targetR),iorb],[np.array(targetR),iorb+halfNOrb],deltaList[iorb]/2.])
            scatterTheSameDirectionList.append([[np.array(targetR),iorb+halfNOrb],[np.array(targetR),iorb],deltaList[iorb]/2.])
            scatterTheOppoDirectionList.append([[np.array(targetR),iorb],[np.array(targetR),iorb+halfNOrb],-deltaList[iorb]/2.])
            scatterTheOppoDirectionList.append([[np.array(targetR),iorb+halfNOrb],[np.array(targetR),iorb],-deltaList[iorb]/2.])
        
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
        for scatter in scatterTheSameDirectionList:
            # standarize Bravais vectors
            scatter[0][0]%=scSize
            scatter[1][0]%=scSize
            if findOrbIndexInRecordList(scatter[0])<0:
                relevantOrbList.append(scatter[0])
            if findOrbIndexInRecordList(scatter[1])<0:
                relevantOrbList.append(scatter[1])
        scatteringOrder=len(relevantOrbList)
        print('find scattering region with %d-dimension, basis consist of'%scatteringOrder)
        for basis in relevantOrbList:
            print('orb%d at [%d %d %d]'%(basis[1],*basis[0]))
        # build scattering matrix
        scatteringSameCenter=np.zeros((scatteringOrder,scatteringOrder))
        for scatter in scatterTheSameDirectionList:
            sourceID=findOrbIndexInRecordList(scatter[0])
            targetID=findOrbIndexInRecordList(scatter[1])
            scatteringSameCenter[sourceID,targetID]=scatter[2]

        scatteringOppoCenter=np.zeros((scatteringOrder,scatteringOrder))
        for scatter in scatterTheOppoDirectionList:
            sourceID=findOrbIndexInRecordList(scatter[0])
            targetID=findOrbIndexInRecordList(scatter[1])
            scatteringOppoCenter[sourceID,targetID]=scatter[2]
        #print('scattering matrix is:')
        #print(scatteringCenter)
        
        norb=hr.norb
        kmesh=pR.createKPOINTS(getKMesh=True,provideSize=[Lx,Ly,Lz])
        totalWeight=len(kmesh)

        time0=time.time()
        HkList=[]
        for kpt in kmesh:
            _ham=Ham()
            Hk=_ham.buildHkFromHr(hr,kpt,getHk=True)
            HkList.append(Hk)
        print('Hk constructed in %.3fs'%(time.time()-time0))
        r=(energyRange[1]-energyRange[0])/2
        o=(energyRange[0]+energyRange[1])/2

        #scatteringCenter=scatteringSameCenter
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
                
                #rou+=sum(np.diagonal(_gk))

            perturb_matrix=np.dot(g_external_contracted,scatteringSameCenter)
            perturb_oppo_matrix=np.dot(g_external_contracted,scatteringOppoCenter)
            perturb_tmp=[]
            cnt=0
            while(True):
                if True:  #!!!!!!!!!!!!!!!!! Maybe not proper for SOC case
                    perturb_matrix=perturb_matrix.dot(g_internal).dot(scatteringSameCenter)
                    perturb_oppo_matrix=perturb_oppo_matrix.dot(g_internal).dot(scatteringOppoCenter)
                perturb_matrix_trace=sum(np.diagonal(perturb_matrix))-sum(np.diagonal(perturb_oppo_matrix))
                if linearPerturb: # linear perturbation
                    rou_perturb=perturb_matrix_trace
                    break
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
                perturb_matrix=perturb_matrix.dot(g_internal).dot(scatteringSameCenter)
                perturb_oppo_matrix=perturb_oppo_matrix.dot(g_internal).dot(scatteringOppoCenter)

            rou+=rou_perturb
            dos=rou/(-np.pi) # DOS of super cell
            totalEnergy=E*dos
            return (totalEnergy*1j*z).imag
        
        time0=time.time()
        E_plus, E_plus_err =integrate.quad(getEOnCircle,a=np.pi,b=0.,epsabs=paramReader.EDIFF)
        print("Energy difference obtained in %.3fs ===> %.6f +- %.6f"%(time.time()-time0,E_plus,E_plus_err))
        
        J=E_plus/2.#/(S**2)
        return J

    def calcJExact(self,hr,sourceOrbList=[0],sourceR=[0,0,0],targetOrbList=[0],targetR=[1,0,0],eta=0.0001,energyRange=(-5.0,0),Lx=5,Ly=5,Lz=1):
        '''Linear response approximation
           using Gaussian integration along semi-circle in upper complex plane
        '''
        assert len(sourceOrbList)==len(targetOrbList), 'Now, the coupling ions is assumed to be symmtric to each other'
        S=len(sourceOrbList)/2.
        norb=hr.norb
        halfNOrb=int(norb/2)
        onsite=np.array(hr.onsite)
        deltaList=onsite[:halfNOrb]-onsite[halfNOrb:]
        print('S=%.1f'%S)
        print('Exchange splitting are:')
        print(deltaList)
        
        # create scattering route
        scatterTheSameDirectionList=[]
        scatterTheOppoDirectionList=[]
        for iorb in sourceOrbList:
            scatterTheSameDirectionList.append([[np.array(sourceR),iorb],[np.array(sourceR),iorb+halfNOrb],deltaList[iorb]/2.])
            scatterTheSameDirectionList.append([[np.array(sourceR),iorb+halfNOrb],[np.array(sourceR),iorb],deltaList[iorb]/2.])

            scatterTheOppoDirectionList.append([[np.array(sourceR),iorb],[np.array(sourceR),iorb+halfNOrb],deltaList[iorb]/2.])
            scatterTheOppoDirectionList.append([[np.array(sourceR),iorb+halfNOrb],[np.array(sourceR),iorb],deltaList[iorb]/2.])
        
        for iorb in targetOrbList:
            scatterTheSameDirectionList.append([[np.array(targetR),iorb],[np.array(targetR),iorb+halfNOrb],deltaList[iorb]/2.])
            scatterTheSameDirectionList.append([[np.array(targetR),iorb+halfNOrb],[np.array(targetR),iorb],deltaList[iorb]/2.])

            scatterTheOppoDirectionList.append([[np.array(targetR),iorb],[np.array(targetR),iorb+halfNOrb],-deltaList[iorb]/2.])
            scatterTheOppoDirectionList.append([[np.array(targetR),iorb+halfNOrb],[np.array(targetR),iorb],-deltaList[iorb]/2.])
        
        
        time0=time.time()
        E_plus, E_plus_err =self.calcMatrixPerturbatedEnergyViaExact(hr,scatterTheSameDirectionList,eta=eta,energyRange=energyRange,Lx=Lx,Ly=Ly,Lz=Lz,getErr=True)
        #E_plus, E_plus_err =self.calcMatrixPerturbatedEnergyViaFourier(hr,scatterTheSameDirectionList,eta=eta,energyRange=energyRange,Lx=Lx,Ly=Ly,Lz=Lz,getErr=True,skipOdds=True)
        print("Energy obtained in %.3fs ===> same biasing: %.6f +- %.6f"%(time.time()-time0,E_plus,E_plus_err))
        
        time0=time.time()
        E_minus,E_minus_err=self.calcMatrixPerturbatedEnergyViaExact(hr,scatterTheOppoDirectionList,eta=eta,energyRange=energyRange,Lx=Lx,Ly=Ly,Lz=Lz,getErr=True)
        #E_minus,E_minus_err=self.calcMatrixPerturbatedEnergyViaFourier(hr,scatterTheOppoDirectionList,eta=eta,energyRange=energyRange,Lx=Lx,Ly=Ly,Lz=Lz,getErr=True,skipOdds=True)
        print("Energy obtained in %.3fs ===> opposite biasing: %.6f +- %.6f"%(time.time()-time0,E_minus,E_minus_err))
        
        J=(E_plus-E_minus)/2.#/(S**2)
        return J

    def calcJExactWithTheta(self,hr,sourceOrbList=[0],sourceR=[0,0,0],targetOrbList=[0],targetR=[1,0,0],theta=np.pi/2,eta=0.0001,energyRange=(-5.0,0),Lx=5,Ly=5,Lz=1):
        '''Linear response approximation
           using Gaussian integration along semi-circle in upper complex plane
        '''
        assert len(sourceOrbList)==len(targetOrbList), 'Now, the coupling ions is assumed to be symmtric to each other'
        S=len(sourceOrbList)/2.
        norb=hr.norb
        halfNOrb=int(norb/2)
        onsite=np.array(hr.onsite)
        deltaList=onsite[:halfNOrb]-onsite[halfNOrb:]
        print('S=%.1f'%S)
        print('Exchange splitting are:')
        print(deltaList)
        
        #print(theta,np.sin(theta),np.cos(theta)-1,1./(1-np.cos(2*theta)))
        #exit()
        # create scattering route
        scatterTheSameDirectionList=[]
        scatterTheOppoDirectionList=[]
        for iorb in sourceOrbList:
            scatterTheSameDirectionList.append([[np.array(sourceR),iorb],[np.array(sourceR),iorb+halfNOrb],deltaList[iorb]/2.*np.sin(theta)])
            scatterTheSameDirectionList.append([[np.array(sourceR),iorb+halfNOrb],[np.array(sourceR),iorb],deltaList[iorb]/2.*np.sin(theta)])
            #scatterTheSameDirectionList.append([[np.array(sourceR),iorb],[np.array(sourceR),iorb],deltaList[iorb]/2.*(np.cos(theta)-1)])
            #scatterTheSameDirectionList.append([[np.array(sourceR),iorb+halfNOrb],[np.array(sourceR),iorb+halfNOrb],-deltaList[iorb]/2.*(np.cos(theta)-1)])

            scatterTheOppoDirectionList.append([[np.array(sourceR),iorb],[np.array(sourceR),iorb+halfNOrb],deltaList[iorb]/2.*np.sin(theta)])
            scatterTheOppoDirectionList.append([[np.array(sourceR),iorb+halfNOrb],[np.array(sourceR),iorb],deltaList[iorb]/2.*np.sin(theta)])
            #scatterTheOppoDirectionList.append([[np.array(sourceR),iorb],[np.array(sourceR),iorb],deltaList[iorb]/2.*(np.cos(theta)-1)])
            #scatterTheOppoDirectionList.append([[np.array(sourceR),iorb+halfNOrb],[np.array(sourceR),iorb+halfNOrb],-deltaList[iorb]/2.*(np.cos(theta)-1)])
        
        for iorb in targetOrbList:
            scatterTheSameDirectionList.append([[np.array(targetR),iorb],[np.array(targetR),iorb+halfNOrb],deltaList[iorb]/2.*np.sin(theta)])
            scatterTheSameDirectionList.append([[np.array(targetR),iorb+halfNOrb],[np.array(targetR),iorb],deltaList[iorb]/2.*np.sin(theta)])
            #scatterTheSameDirectionList.append([[np.array(targetR),iorb],[np.array(targetR),iorb],deltaList[iorb]/2.*(np.cos(theta)-1)])
            #scatterTheSameDirectionList.append([[np.array(targetR),iorb+halfNOrb],[np.array(targetR),iorb+halfNOrb],-deltaList[iorb]/2.*(np.cos(theta)-1)])

            scatterTheOppoDirectionList.append([[np.array(targetR),iorb],[np.array(targetR),iorb+halfNOrb],-deltaList[iorb]/2.*np.sin(theta)])
            scatterTheOppoDirectionList.append([[np.array(targetR),iorb+halfNOrb],[np.array(targetR),iorb],-deltaList[iorb]/2.*np.sin(theta)])
            #scatterTheOppoDirectionList.append([[np.array(targetR),iorb],[np.array(targetR),iorb],deltaList[iorb]/2.*(np.cos(theta)-1)])
            #scatterTheOppoDirectionList.append([[np.array(targetR),iorb+halfNOrb],[np.array(targetR),iorb+halfNOrb],-deltaList[iorb]/2.*(np.cos(theta)-1)])
        
        
        time0=time.time()
        E_plus, E_plus_err =self.calcMatrixPerturbatedEnergyViaExact(hr,scatterTheSameDirectionList,eta=eta,energyRange=energyRange,Lx=Lx,Ly=Ly,Lz=Lz,getErr=True)
        #E_plus, E_plus_err =self.calcMatrixPerturbatedEnergyViaFourier(hr,scatterTheSameDirectionList,eta=eta,energyRange=energyRange,Lx=Lx,Ly=Ly,Lz=Lz,getErr=True,skipOdds=True)
        print("Energy obtained in %.3fs ===> same biasing: %.6f +- %.6f"%(time.time()-time0,E_plus,E_plus_err))
        
        time0=time.time()
        E_minus,E_minus_err=self.calcMatrixPerturbatedEnergyViaExact(hr,scatterTheOppoDirectionList,eta=eta,energyRange=energyRange,Lx=Lx,Ly=Ly,Lz=Lz,getErr=True)
        #E_minus,E_minus_err=self.calcMatrixPerturbatedEnergyViaFourier(hr,scatterTheOppoDirectionList,eta=eta,energyRange=energyRange,Lx=Lx,Ly=Ly,Lz=Lz,getErr=True,skipOdds=True)
        print("Energy obtained in %.3fs ===> opposite biasing: %.6f +- %.6f"%(time.time()-time0,E_minus,E_minus_err))
        
        J=(E_plus-E_minus)/(1-np.cos(2*theta))
        print(J*1000)
        #exit()
        return J

    def calcJExactFourStateWithTheta(self,hr,sourceOrbList=[0],sourceR=[0,0,0],targetOrbList=[0],targetR=[1,0,0],theta=np.pi/2,eta=0.0001,energyRange=(-5.0,0),Lx=5,Ly=5,Lz=1):
        '''Linear response approximation
           using Gaussian integration along semi-circle in upper complex plane
        '''
        assert len(sourceOrbList)==len(targetOrbList), 'Now, the coupling ions is assumed to be symmtric to each other'
        S=len(sourceOrbList)/2.
        norb=hr.norb
        halfNOrb=int(norb/2)
        onsite=np.array(hr.onsite)
        deltaList=onsite[:halfNOrb]-onsite[halfNOrb:]
        print('S=%.1f'%S)
        print('Exchange splitting are:')
        print(deltaList)
        
        #print(theta,np.sin(theta),np.cos(theta)-1,1./(1-np.cos(2*theta)))
        #exit()
        # create scattering route
        #theta=np.pi/2
        scatterDoubleList=[]
        scatterSingleList=[]
        for iorb in sourceOrbList:
            scatterDoubleList.append([[np.array(sourceR),iorb],[np.array(sourceR),iorb+halfNOrb],deltaList[iorb]/2.*np.sin(theta)])
            scatterDoubleList.append([[np.array(sourceR),iorb+halfNOrb],[np.array(sourceR),iorb],deltaList[iorb]/2.*np.sin(theta)])
            scatterDoubleList.append([[np.array(sourceR),iorb],[np.array(sourceR),iorb],deltaList[iorb]/2.*(np.cos(theta)-1)])
            scatterDoubleList.append([[np.array(sourceR),iorb+halfNOrb],[np.array(sourceR),iorb+halfNOrb],-deltaList[iorb]/2.*(np.cos(theta)-1)])

            scatterSingleList.append([[np.array(sourceR),iorb],[np.array(sourceR),iorb+halfNOrb],deltaList[iorb]/2.*np.sin(theta)])
            scatterSingleList.append([[np.array(sourceR),iorb+halfNOrb],[np.array(sourceR),iorb],deltaList[iorb]/2.*np.sin(theta)])
            scatterSingleList.append([[np.array(sourceR),iorb],[np.array(sourceR),iorb],deltaList[iorb]/2.*(np.cos(theta)-1)])
            scatterSingleList.append([[np.array(sourceR),iorb+halfNOrb],[np.array(sourceR),iorb+halfNOrb],-deltaList[iorb]/2.*(np.cos(theta)-1)])
        
        for iorb in targetOrbList:
            scatterDoubleList.append([[np.array(targetR),iorb],[np.array(targetR),iorb+halfNOrb],deltaList[iorb]/2.*np.sin(theta)])
            scatterDoubleList.append([[np.array(targetR),iorb+halfNOrb],[np.array(targetR),iorb],deltaList[iorb]/2.*np.sin(theta)])
            scatterDoubleList.append([[np.array(targetR),iorb],[np.array(targetR),iorb],deltaList[iorb]/2.*(np.cos(theta)-1)])
            scatterDoubleList.append([[np.array(targetR),iorb+halfNOrb],[np.array(targetR),iorb+halfNOrb],-deltaList[iorb]/2.*(np.cos(theta)-1)])
        
        
        time0=time.time()
        E_0, E_0_err =self.calcMatrixPerturbatedEnergyViaExact(hr,[],eta=eta,energyRange=energyRange,Lx=Lx,Ly=Ly,Lz=Lz,getErr=True)
        #E_plus, E_plus_err =self.calcMatrixPerturbatedEnergyViaFourier(hr,scatterTheSameDirectionList,eta=eta,energyRange=energyRange,Lx=Lx,Ly=Ly,Lz=Lz,getErr=True,skipOdds=True)
        print("Energy obtained in %.3fs ===>   ground state: %.6f +- %.6f"%(time.time()-time0,E_0,E_0_err))

        time0=time.time()
        E_plus, E_plus_err =self.calcMatrixPerturbatedEnergyViaExact(hr,scatterDoubleList,eta=eta,energyRange=energyRange,Lx=Lx,Ly=Ly,Lz=Lz,getErr=True)
        #E_plus, E_plus_err =self.calcMatrixPerturbatedEnergyViaFourier(hr,scatterTheSameDirectionList,eta=eta,energyRange=energyRange,Lx=Lx,Ly=Ly,Lz=Lz,getErr=True,skipOdds=True)
        print("Energy obtained in %.3fs ===> double biasing: %.6f +- %.6f"%(time.time()-time0,E_plus,E_plus_err))
        
        time0=time.time()
        E_minus,E_minus_err=self.calcMatrixPerturbatedEnergyViaExact(hr,scatterSingleList,eta=eta,energyRange=energyRange,Lx=Lx,Ly=Ly,Lz=Lz,getErr=True)
        #E_minus,E_minus_err=self.calcMatrixPerturbatedEnergyViaFourier(hr,scatterTheOppoDirectionList,eta=eta,energyRange=energyRange,Lx=Lx,Ly=Ly,Lz=Lz,getErr=True,skipOdds=True)
        print("Energy obtained in %.3fs ===> single biasing: %.6f +- %.6f"%(time.time()-time0,E_minus,E_minus_err))
        
        J=(E_plus+E_0-2*E_minus)/2./(1-np.cos(theta))
        print(J*1000)
        #exit()
        return J

    def calcAnisotropicJ(self,hr,sourceOrbList=[0],sourceR=[0,0,0],targetOrbList=[0],targetR=[1,0,0],eta=0.0001,energyRange=(-5.0,0),Lx=5,Ly=5,Lz=1):
        '''Linear response approximation
           using Gaussian integration along semi-circle in upper complex plane
        '''
        assert len(sourceOrbList)==len(targetOrbList), 'Now, the coupling ions is assumed to be symmtric to each other'
        S=len(sourceOrbList)/2.
        norb=hr.norb
        halfNOrb=int(norb/2)
        onsite=np.array(hr.onsite)
        deltaList=onsite[:halfNOrb]-onsite[halfNOrb:]
        print('S=%.1f'%S)
        print('Exchange splitting are:')
        print(deltaList)
        
        # create scattering route
        # create scattering route
        scatterTheSameXDirectionList=[]
        scatterTheOppoXDirectionList=[]
        scatterTheSameYDirectionList=[]
        scatterTheOppoYDirectionList=[]
        for iorb in sourceOrbList:
            scatterTheSameXDirectionList.append([[np.array(sourceR),iorb],[np.array(sourceR),iorb+halfNOrb], deltaList[iorb]/2.])
            scatterTheSameXDirectionList.append([[np.array(sourceR),iorb+halfNOrb],[np.array(sourceR),iorb], deltaList[iorb]/2.])
            scatterTheOppoXDirectionList.append([[np.array(sourceR),iorb],[np.array(sourceR),iorb+halfNOrb], deltaList[iorb]/2.])
            scatterTheOppoXDirectionList.append([[np.array(sourceR),iorb+halfNOrb],[np.array(sourceR),iorb], deltaList[iorb]/2.])

            scatterTheSameYDirectionList.append([[np.array(sourceR),iorb],[np.array(sourceR),iorb+halfNOrb],-1j*deltaList[iorb]/2.])
            scatterTheSameYDirectionList.append([[np.array(sourceR),iorb+halfNOrb],[np.array(sourceR),iorb], 1j*deltaList[iorb]/2.])
            scatterTheOppoYDirectionList.append([[np.array(sourceR),iorb],[np.array(sourceR),iorb+halfNOrb],-1j*deltaList[iorb]/2.])
            scatterTheOppoYDirectionList.append([[np.array(sourceR),iorb+halfNOrb],[np.array(sourceR),iorb], 1j*deltaList[iorb]/2.])
        
        for iorb in targetOrbList:
            scatterTheSameXDirectionList.append([[np.array(targetR),iorb],[np.array(targetR),iorb+halfNOrb],   deltaList[iorb]/2.])
            scatterTheSameXDirectionList.append([[np.array(targetR),iorb+halfNOrb],[np.array(targetR),iorb],   deltaList[iorb]/2.])
            scatterTheOppoXDirectionList.append([[np.array(targetR),iorb],[np.array(targetR),iorb+halfNOrb],-1*deltaList[iorb]/2.])
            scatterTheOppoXDirectionList.append([[np.array(targetR),iorb+halfNOrb],[np.array(targetR),iorb],-1*deltaList[iorb]/2.])

            scatterTheSameYDirectionList.append([[np.array(targetR),iorb],[np.array(targetR),iorb+halfNOrb],-1j*deltaList[iorb]/2.])
            scatterTheSameYDirectionList.append([[np.array(targetR),iorb+halfNOrb],[np.array(targetR),iorb], 1j*deltaList[iorb]/2.])
            scatterTheOppoYDirectionList.append([[np.array(targetR),iorb],[np.array(targetR),iorb+halfNOrb], 1j*deltaList[iorb]/2.])
            scatterTheOppoYDirectionList.append([[np.array(targetR),iorb+halfNOrb],[np.array(targetR),iorb],-1j*deltaList[iorb]/2.])
        
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
        for scatter in scatterTheSameXDirectionList:
            # standarize Bravais vectors
            scatter[0][0]%=scSize
            scatter[1][0]%=scSize
            if findOrbIndexInRecordList(scatter[0])<0:
                relevantOrbList.append(scatter[0])
            if findOrbIndexInRecordList(scatter[1])<0:
                relevantOrbList.append(scatter[1])
        scatteringOrder=len(relevantOrbList)
        print('find scattering region with %d-dimension, basis consist of'%scatteringOrder)
        for basis in relevantOrbList:
            print('orb%d at [%d %d %d]'%(basis[1],*basis[0]))
        # build scattering matrix
        scatteringSameXCenter=np.zeros((scatteringOrder,scatteringOrder),dtype=complex)
        for scatter in scatterTheSameXDirectionList:
            sourceID=findOrbIndexInRecordList(scatter[0])
            targetID=findOrbIndexInRecordList(scatter[1])
            scatteringSameXCenter[sourceID,targetID]=scatter[2]

        scatteringOppoXCenter=np.zeros((scatteringOrder,scatteringOrder),dtype=complex)
        for scatter in scatterTheOppoXDirectionList:
            sourceID=findOrbIndexInRecordList(scatter[0])
            targetID=findOrbIndexInRecordList(scatter[1])
            scatteringOppoXCenter[sourceID,targetID]=scatter[2]

        scatteringSameYCenter=np.zeros((scatteringOrder,scatteringOrder),dtype=complex)
        for scatter in scatterTheSameYDirectionList:
            sourceID=findOrbIndexInRecordList(scatter[0])
            targetID=findOrbIndexInRecordList(scatter[1])
            scatteringSameYCenter[sourceID,targetID]=scatter[2]

        scatteringOppoYCenter=np.zeros((scatteringOrder,scatteringOrder),dtype=complex)
        for scatter in scatterTheOppoYDirectionList:
            sourceID=findOrbIndexInRecordList(scatter[0])
            targetID=findOrbIndexInRecordList(scatter[1])
            scatteringOppoYCenter[sourceID,targetID]=scatter[2]
        #print('scattering matrix is:')
        #print(scatteringCenter)
        
        norb=hr.norb
        kmesh=pR.createKPOINTS(getKMesh=True,provideSize=[Lx,Ly,Lz])
        totalWeight=len(kmesh)

        time0=time.time()
        HkList=[]
        for kpt in kmesh:
            _ham=Ham()
            Hk=_ham.buildHkFromHr(hr,kpt,getHk=True)
            HkList.append(Hk)
        print('Hk constructed in %.3fs'%(time.time()-time0))
        r=(energyRange[1]-energyRange[0])/2
        o=(energyRange[0]+energyRange[1])/2

        #scatteringCenter=scatteringSameCenter
        def getEOnCircle(theta):
            z=r*np.exp(1j*theta)
            E=z+o
            g_internal=np.zeros((scatteringOrder,scatteringOrder),dtype=complex)
            g_external_contracted=np.zeros((scatteringOrder,scatteringOrder),dtype=complex)
            rou0=0.j
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
                
                rou0+=sum(np.diagonal(_gk))

            x_perturb_matrix_order_1=np.dot(g_external_contracted,scatteringSameXCenter)
            x_perturb_oppo_matrix_order_1=np.dot(g_external_contracted,scatteringOppoXCenter)

            y_perturb_matrix_order_1=np.dot(g_external_contracted,scatteringSameYCenter)
            y_perturb_oppo_matrix_order_1=np.dot(g_external_contracted,scatteringOppoYCenter)

            #x_rou_perturb_order_1=sum(np.diagonal(x_perturb_matrix_order_1))-sum(np.diagonal(x_perturb_oppo_matrix_order_1))
            #y_rou_perturb_order_1=sum(np.diagonal(y_perturb_matrix_order_1))-sum(np.diagonal(y_perturb_oppo_matrix_order_1))

            x_perturb_matrix_order_2=x_perturb_matrix_order_1.dot(g_internal).dot(scatteringSameXCenter)
            x_perturb_oppo_matrix_order_2=x_perturb_oppo_matrix_order_1.dot(g_internal).dot(scatteringOppoXCenter)
            
            y_perturb_matrix_order_2=y_perturb_matrix_order_1.dot(g_internal).dot(scatteringSameYCenter)
            y_perturb_oppo_matrix_order_2=y_perturb_oppo_matrix_order_1.dot(g_internal).dot(scatteringOppoYCenter)
            
            x_rou_perturb_order_2=sum(np.diagonal(x_perturb_matrix_order_2))-sum(np.diagonal(x_perturb_oppo_matrix_order_2))
            y_rou_perturb_order_2=sum(np.diagonal(y_perturb_matrix_order_2))-sum(np.diagonal(y_perturb_oppo_matrix_order_2))

            #dos_order_0=rou0/(-np.pi)
            #dos_order_1=rou_perturb_order_1/(-np.pi) # DOS of super cell
            dos_x_order_2=x_rou_perturb_order_2/(-np.pi)
            dos_y_order_2=y_rou_perturb_order_2/(-np.pi)
            x_totalEnergy2, y_totalEnergy2=E*dos_x_order_2, E*dos_y_order_2
            return (np.array([x_totalEnergy2, y_totalEnergy2])*1j*z).imag
        
        time0=time.time()
        E_diff, E_diff_err =integrate.quad_vec(getEOnCircle,a=np.pi,b=0.,epsabs=paramReader.EDIFF)
        print(E_diff,E_diff_err)
        #print("Energy difference obtained in %.3fs ===> E^0: %.6f E^1: %.6f E^2: %.6f +- %.6f %.6f %.6f"%(time.time()-time0,*E_diff,*E_diff_err))
        exit()
        #J=E_plus/2.#/(S**2)
        #return J

    def calcJFourState(self,hr,sourceOrbList=[0],sourceR=[0,0,0],targetOrbList=[0],targetR=[1,0,0],theta=np.pi,eta=0.0001,energyRange=(-5.0,0),Lx=5,Ly=5,Lz=1):
        '''Linear response approximation
           using Gaussian integration along semi-circle in upper complex plane
        '''
        assert len(sourceOrbList)==len(targetOrbList), 'Now, the coupling ions is assumed to be symmtric to each other'
        S=len(sourceOrbList)/2.
        norb=hr.norb
        halfNOrb=int(norb/2)
        onsite=np.array(hr.onsite)
        deltaList=onsite[:halfNOrb]-onsite[halfNOrb:]
        print('S=%.1f'%S)
        print('Exchange splitting are:')
        print(deltaList)
        
        # create scattering route
        # create scattering route
        scatterDoubleList=[]
        scatterSingle1List=[]
        scatterSingle2List=[]
        for iorb in sourceOrbList:
            scatterDoubleList.append([[np.array(sourceR),iorb],[np.array(sourceR),iorb+halfNOrb],deltaList[iorb]/2.*np.sin(theta)])
            scatterDoubleList.append([[np.array(sourceR),iorb+halfNOrb],[np.array(sourceR),iorb],deltaList[iorb]/2.*np.sin(theta)])
            scatterDoubleList.append([[np.array(sourceR),iorb],[np.array(sourceR),iorb],deltaList[iorb]*(np.cos(theta)-1)])
            scatterDoubleList.append([[np.array(sourceR),iorb+halfNOrb],[np.array(sourceR),iorb+halfNOrb],-deltaList[iorb]*(np.cos(theta)-1)])

            scatterSingle1List.append([[np.array(sourceR),iorb],[np.array(sourceR),iorb+halfNOrb],deltaList[iorb]/2.*np.sin(theta)])
            scatterSingle1List.append([[np.array(sourceR),iorb+halfNOrb],[np.array(sourceR),iorb],deltaList[iorb]/2.*np.sin(theta)])
            scatterSingle1List.append([[np.array(sourceR),iorb],[np.array(sourceR),iorb],deltaList[iorb]*(np.cos(theta)-1)])
            scatterSingle1List.append([[np.array(sourceR),iorb+halfNOrb],[np.array(sourceR),iorb+halfNOrb],-deltaList[iorb]*(np.cos(theta)-1)])

            scatterSingle2List.append([[np.array(sourceR),iorb],[np.array(sourceR),iorb+halfNOrb],0])
            scatterSingle2List.append([[np.array(sourceR),iorb+halfNOrb],[np.array(sourceR),iorb],0])
            scatterSingle2List.append([[np.array(sourceR),iorb],[np.array(sourceR),iorb],0])
            scatterSingle2List.append([[np.array(sourceR),iorb+halfNOrb],[np.array(sourceR),iorb+halfNOrb],0])
        
        for iorb in targetOrbList:
            scatterDoubleList.append([[np.array(targetR),iorb],[np.array(targetR),iorb+halfNOrb],deltaList[iorb]/2.*np.sin(theta)])
            scatterDoubleList.append([[np.array(targetR),iorb+halfNOrb],[np.array(targetR),iorb],deltaList[iorb]/2.*np.sin(theta)])
            scatterDoubleList.append([[np.array(targetR),iorb],[np.array(targetR),iorb],deltaList[iorb]*(np.cos(theta)-1)])
            scatterDoubleList.append([[np.array(targetR),iorb+halfNOrb],[np.array(targetR),iorb+halfNOrb],-deltaList[iorb]*(np.cos(theta)-1)])

            scatterSingle1List.append([[np.array(targetR),iorb],[np.array(targetR),iorb+halfNOrb],0])
            scatterSingle1List.append([[np.array(targetR),iorb+halfNOrb],[np.array(targetR),iorb],0])
            scatterSingle1List.append([[np.array(targetR),iorb],[np.array(targetR),iorb],0])
            scatterSingle1List.append([[np.array(targetR),iorb+halfNOrb],[np.array(targetR),iorb+halfNOrb],0])

            scatterSingle2List.append([[np.array(targetR),iorb],[np.array(targetR),iorb+halfNOrb],deltaList[iorb]/2.*np.sin(theta)])
            scatterSingle2List.append([[np.array(targetR),iorb+halfNOrb],[np.array(targetR),iorb],deltaList[iorb]/2.*np.sin(theta)])
            scatterSingle2List.append([[np.array(targetR),iorb],[np.array(targetR),iorb],deltaList[iorb]*(np.cos(theta)-1)])
            scatterSingle2List.append([[np.array(targetR),iorb+halfNOrb],[np.array(targetR),iorb+halfNOrb],-deltaList[iorb]*(np.cos(theta)-1)])
        
        # analyze scattering list
        # construct scattering matrix with all irrelevant zeros suppressed
        relevantOrbList=[]
        scSize=np.array([Lx,Ly,Lz])
        def findOrbIndexInRecordList(orb):
            for iexistedOrb, existedOrb in enumerate(relevantOrbList):
                if (existedOrb[0]==orb[0]%scSize).all() and existedOrb[1]==orb[1]:
                    return iexistedOrb
            return -1
        # build double scattering basis
        for scatter in scatterDoubleList:
            # standarize Bravais vectors
            scatter[0][0]%=scSize
            scatter[1][0]%=scSize
            if findOrbIndexInRecordList(scatter[0])<0:
                relevantOrbList.append(scatter[0])
            if findOrbIndexInRecordList(scatter[1])<0:
                relevantOrbList.append(scatter[1])
        scatteringOrder=len(relevantOrbList)
        print('find scattering region with %d-dimension, basis consist of'%scatteringOrder)
        for basis in relevantOrbList:
            print('orb%d at [%d %d %d]'%(basis[1],*basis[0]))
        # build scattering matrix
        scatteringDoubleCenter=np.zeros((scatteringOrder,scatteringOrder))
        for scatter in scatterDoubleList:
            sourceID=findOrbIndexInRecordList(scatter[0])
            targetID=findOrbIndexInRecordList(scatter[1])
            scatteringDoubleCenter[sourceID,targetID]=scatter[2]
        #print(scatteringDoubleCenter)
        scatteringSingle1Center=np.zeros((scatteringOrder,scatteringOrder))
        for scatter in scatterSingle1List:
            sourceID=findOrbIndexInRecordList(scatter[0])
            targetID=findOrbIndexInRecordList(scatter[1])
            scatteringSingle1Center[sourceID,targetID]=scatter[2]
        
        scatteringSingle2Center=np.zeros((scatteringOrder,scatteringOrder))
        for scatter in scatterSingle2List:
            sourceID=findOrbIndexInRecordList(scatter[0])
            targetID=findOrbIndexInRecordList(scatter[1])
            scatteringSingle2Center[sourceID,targetID]=scatter[2]
        #print('scattering matrix is:')
        #print(scatteringCenter)
        
        norb=hr.norb
        kmesh=pR.createKPOINTS(getKMesh=True,provideSize=[Lx,Ly,Lz])
        totalWeight=len(kmesh)

        time0=time.time()
        HkList=[]
        for kpt in kmesh:
            _ham=Ham()
            Hk=_ham.buildHkFromHr(hr,kpt,getHk=True)
            HkList.append(Hk)
        print('Hk constructed in %.3fs'%(time.time()-time0))
        r=(energyRange[1]-energyRange[0])/2
        o=(energyRange[0]+energyRange[1])/2

        #scatteringCenter=scatteringSameCenter
        #@jit(nopython=True)
        def getEOnCircle(theta):
            z=r*np.exp(1j*theta)
            E=z+o
            g_internal=np.zeros((scatteringOrder,scatteringOrder),dtype=complex)
            g_external_contracted=np.zeros((scatteringOrder,scatteringOrder),dtype=complex)
            
            #istep=0
            for kpt, _hk in zip(kmesh,HkList):
                #print("kpt (%.2f %.2f %.2f)"%(*kpt,))
                _gk=np.linalg.inv((E+eta*1j)*np.eye(norb)-_hk)
                #print("hk:")
                #for line in _hk:
                #    print("(%.6f, %.6f) (%.6f, %.6f) (%.6f, %.6f) (%.6f, %.6f)"%(line[0].real,line[0].imag,line[1].real,line[1].imag,line[2].real,line[2].imag,line[3].real,line[3].imag))
                #print("Gk:")
                #for line in _gk:
                #    print("(%.6f, %.6f) (%.6f, %.6f) (%.6f, %.6f) (%.6f, %.6f)"%(line[0].real,line[0].imag,line[1].real,line[1].imag,line[2].real,line[2].imag,line[3].real,line[3].imag))
                #print("g_internal before step%d"%istep)
                #for line in g_internal:
                #    print("(%.6f, %.6f) (%.6f, %.6f) (%.6f, %.6f) (%.6f, %.6f)"%(line[0].real,line[0].imag,line[1].real,line[1].imag,line[2].real,line[2].imag,line[3].real,line[3].imag))
                #print("g_external before step%d"%istep)
                #for line in g_external_contracted:
                #    print("(%.6f, %.6f) (%.6f, %.6f) (%.6f, %.6f) (%.6f, %.6f)"%(line[0].real,line[0].imag,line[1].real,line[1].imag,line[2].real,line[2].imag,line[3].real,line[3].imag))
                #istep+=1

                for irow in range(scatteringOrder):
                    R_left, orbID_left=relevantOrbList[irow]
                    for icol in range(scatteringOrder):
                        R_right, orbID_right=relevantOrbList[icol]
                        #print('row%d col%d --- scatterLeft (%d %d %d) orb%d scatterRight (%d %d %d) orb%d'%(irow,icol,*R_left,orbID_left,*R_right,orbID_right))
                        # internal line
                        g_internal[irow,icol]+=np.exp(-2j*np.pi*np.dot(kpt,R_left-R_right))*_gk[orbID_left,orbID_right]/totalWeight
                        # contracted external line
                        g_external_contracted[irow,icol]+=np.exp(2j*np.pi*np.dot(kpt,R_left-R_right))*np.dot(_gk[orbID_right,:],_gk[:,orbID_left])/totalWeight
                
                #rou+=sum(np.diagonal(_gk))
            '''print("left tensor")
            for line in g_external_contracted:
                print("(%.6f, %.6f) (%.6f, %.6f) (%.6f, %.6f) (%.6f, %.6f)"%(line[0].real,line[0].imag,line[1].real,line[1].imag,line[2].real,line[2].imag,line[3].real,line[3].imag))
            print("right tensor")
            for line in scatteringDoubleCenter:
                print("(%.6f, %.6f) (%.6f, %.6f) (%.6f, %.6f) (%.6f, %.6f)"%(line[0].real,line[0].imag,line[1].real,line[1].imag,line[2].real,line[2].imag,line[3].real,line[3].imag))
            print("result tensor")
            for line in np.dot(g_external_contracted,scatteringDoubleCenter):
                print("(%.6f, %.6f) (%.6f, %.6f) (%.6f, %.6f) (%.6f, %.6f)"%(line[0].real,line[0].imag,line[1].real,line[1].imag,line[2].real,line[2].imag,line[3].real,line[3].imag))
            '''
            # 2nd order perturbation
            perturb_double_matrix=np.dot(g_external_contracted,scatteringDoubleCenter).dot(g_internal).dot(scatteringDoubleCenter)
            perturb_single1_matrix=np.dot(g_external_contracted,scatteringSingle1Center).dot(g_internal).dot(scatteringSingle1Center)
            perturb_single2_matrix=np.dot(g_external_contracted,scatteringSingle2Center).dot(g_internal).dot(scatteringSingle2Center)
            # tracet to get rou
            perturb_matrix_trace=np.sum(np.diagonal(perturb_double_matrix))-np.sum(np.diagonal(perturb_single1_matrix))-np.sum(np.diagonal(perturb_single2_matrix))
            delta_rou=perturb_matrix_trace

            dos=delta_rou/(-np.pi) # delta DOS of super cell
            totalEnergy=E*dos
            return (totalEnergy*1j*z).imag
        
        #a=getEOnCircle(np.pi)
        #print(a)
        #exit()

        time0=time.time()
        E_coupling, E_coupling_err =integrate.quad(getEOnCircle,a=np.pi,b=0.,epsabs=paramReader.EDIFF)
        print("Coupling energy obtained in %.3fs ===> %.6f +- %.6f"%(time.time()-time0,E_coupling,E_coupling_err))
        
        J=E_coupling/(-2*(np.cos(theta)-1))
        return J
    
    # func series to calc J with CDLL
    def getCTypeKMeshAndHk(self,hr,Lx=10,Ly=10,Lz=1,GammaMode=1,nthreads=1):
        time0=time.time()
        print('generating kmesh, Hk...')
        hamDLL=CDLL("./ham.dll")
        hamDLL.getKMesh.restype=c_longlong
        #norb=hr.norb
        #kmesh=np.array(pR.createKPOINTS(getKMesh=True,provideSize=[Lx,Ly,Lz]))
        nkpts=Lx*Ly*Lz
        c_kmesh_address=hamDLL.getKMesh(c_int(GammaMode),c_int(Lx),c_int(Ly),c_int(Lz))#(c_double*(3*kmesh.shape[0]))()
        print('in python, memory address of kmesh is %s'%hex(c_kmesh_address))
        #c_Hk=(c_double*(kmesh.shape[0]*norb*norb*2))()
        #Hk_ref=[]
        #for ikpt, kpt in enumerate(kmesh):
            # store kpts
        #    for i in range(3):
        #        c_kmesh[ikpt*3+i]=c_double(kpt[i])
            
        #    _ham=Ham()
        #    Hk=_ham.buildHkFromHr(hr,kpt,getHk=True)
        #    Hk_ref.append(Hk)
        #    # store Hk
        #    for i in range(norb):
        #        for j in range(norb):
        #            c_Hk[ikpt*norb*norb*2+i*norb*2+j*2]=c_double(Hk[i][j].real)
        #            c_Hk[ikpt*norb*norb*2+i*norb*2+j*2+1]=c_double(Hk[i][j].imag)
        nhoppings=len(hr.bondList)
        c_sourceID=(c_int*nhoppings)()
        c_targetID=(c_int*nhoppings)()
        c_overLat=(c_int*(nhoppings*3))()
        c_hopping_real=(c_double*nhoppings)()
        c_hopping_imag=(c_double*nhoppings)()
        for ibond, bond in enumerate(hr.bondList):
            c_sourceID[ibond]=c_int(bond.source)
            c_targetID[ibond]=c_int(bond.target)
            for i in range(3):
                c_overLat[ibond*3+i]=c_int(bond.overlat[i])
            c_hopping_real[ibond]=c_double(bond.t.real)
            c_hopping_imag[ibond]=c_double(bond.t.imag)
            
        hamDLL.getHk.restype=c_longlong
        '''
        ikpt_print=7
        print('python Hk at %dth kpt'%ikpt_print)
        for line in Hk_ref[ikpt_print]:
            print(("(%.3f %.3f) "*len(line))%(tuple(np.array([[ele.real, ele.imag] for ele in line]).flatten())))
        '''
        c_Hk_address=hamDLL.getHk(c_int(nkpts),c_longlong(c_kmesh_address),
                        c_int(hr.norb),c_int(nhoppings),
                        c_sourceID,c_targetID,c_overLat,c_hopping_real,c_hopping_imag,
                        c_int(nthreads)
                        #c_int(ikpt_print)
                        )
        #HkArray=np.array(HkList,dtype=np.complex128)
        print('Hk constructed in %.3fs (with %d threads)'%(time.time()-time0,nthreads))
        print('in python, memory address of Hk: %s'%hex(c_Hk_address))
        #exit()
        return nkpts, c_kmesh_address, c_Hk_address

    def freeKMeshAndHk(self,c_kmesh_address,c_Hk_address):
        hamDLL=CDLL("./ham.dll")
        hamDLL.cleanKMeshAndHk(c_longlong(c_kmesh_address),c_longlong(c_Hk_address))

    def calcJFourState_CDLL(self,hr,nkpt,c_kmesh_address,c_Hk_address,sourceOrbList=[0],sourceR=[0,0,0],targetOrbList=[0],targetR=[1,0,0],theta=np.pi,eta=0.0001,energyRange=(-5.0,0),Lx=5,Ly=5,Lz=1,nthreads=4):
        '''Linear response approximation
           using Gaussian integration along semi-circle in upper complex plane
        '''
        assert len(sourceOrbList)==len(targetOrbList), 'Now, the coupling ions is assumed to be symmtric to each other'
        S=len(sourceOrbList)/2.
        norb=hr.norb
        halfNOrb=int(norb/2)
        onsite=np.array(hr.onsite)
        deltaList=onsite[:halfNOrb]-onsite[halfNOrb:]
        print('S=%.1f'%S)
        print('Exchange splitting are:')
        print(deltaList)
        
        # create scattering route
        # create scattering route
        scatterDoubleList=[]
        scatterSingle1List=[]
        scatterSingle2List=[]
        for iorb in sourceOrbList:
            scatterDoubleList.append([[np.array(sourceR),iorb],[np.array(sourceR),iorb+halfNOrb],deltaList[iorb]/2.])#*np.sin(theta)])
            scatterDoubleList.append([[np.array(sourceR),iorb+halfNOrb],[np.array(sourceR),iorb],deltaList[iorb]/2.])#*np.sin(theta)])
            scatterDoubleList.append([[np.array(sourceR),iorb],[np.array(sourceR),iorb],0])#deltaList[iorb]*(np.cos(theta)-1)])
            scatterDoubleList.append([[np.array(sourceR),iorb+halfNOrb],[np.array(sourceR),iorb+halfNOrb],0])#-deltaList[iorb]*(np.cos(theta)-1)])

            scatterSingle1List.append([[np.array(sourceR),iorb],[np.array(sourceR),iorb+halfNOrb],deltaList[iorb]/2.])#*np.sin(theta)])
            scatterSingle1List.append([[np.array(sourceR),iorb+halfNOrb],[np.array(sourceR),iorb],deltaList[iorb]/2.])#*np.sin(theta)])
            scatterSingle1List.append([[np.array(sourceR),iorb],[np.array(sourceR),iorb],0])#deltaList[iorb]*(np.cos(theta)-1)])
            scatterSingle1List.append([[np.array(sourceR),iorb+halfNOrb],[np.array(sourceR),iorb+halfNOrb],0])#-deltaList[iorb]*(np.cos(theta)-1)])

            scatterSingle2List.append([[np.array(sourceR),iorb],[np.array(sourceR),iorb+halfNOrb],0])
            scatterSingle2List.append([[np.array(sourceR),iorb+halfNOrb],[np.array(sourceR),iorb],0])
            scatterSingle2List.append([[np.array(sourceR),iorb],[np.array(sourceR),iorb],0])
            scatterSingle2List.append([[np.array(sourceR),iorb+halfNOrb],[np.array(sourceR),iorb+halfNOrb],0])
        
        for iorb in targetOrbList:
            scatterDoubleList.append([[np.array(targetR),iorb],[np.array(targetR),iorb+halfNOrb],deltaList[iorb]/2.])#*np.sin(theta)])
            scatterDoubleList.append([[np.array(targetR),iorb+halfNOrb],[np.array(targetR),iorb],deltaList[iorb]/2.])#*np.sin(theta)])
            scatterDoubleList.append([[np.array(targetR),iorb],[np.array(targetR),iorb],0])#deltaList[iorb]*(np.cos(theta)-1)])
            scatterDoubleList.append([[np.array(targetR),iorb+halfNOrb],[np.array(targetR),iorb+halfNOrb],0])#-deltaList[iorb]*(np.cos(theta)-1)])

            scatterSingle1List.append([[np.array(targetR),iorb],[np.array(targetR),iorb+halfNOrb],0])
            scatterSingle1List.append([[np.array(targetR),iorb+halfNOrb],[np.array(targetR),iorb],0])
            scatterSingle1List.append([[np.array(targetR),iorb],[np.array(targetR),iorb],0])
            scatterSingle1List.append([[np.array(targetR),iorb+halfNOrb],[np.array(targetR),iorb+halfNOrb],0])

            scatterSingle2List.append([[np.array(targetR),iorb],[np.array(targetR),iorb+halfNOrb],deltaList[iorb]/2.*np.sin(theta)])
            scatterSingle2List.append([[np.array(targetR),iorb+halfNOrb],[np.array(targetR),iorb],deltaList[iorb]/2.*np.sin(theta)])
            scatterSingle2List.append([[np.array(targetR),iorb],[np.array(targetR),iorb],deltaList[iorb]*(np.cos(theta)-1)])
            scatterSingle2List.append([[np.array(targetR),iorb+halfNOrb],[np.array(targetR),iorb+halfNOrb],-deltaList[iorb]*(np.cos(theta)-1)])
        
        # analyze scattering list
        # construct scattering matrix with all irrelevant zeros suppressed
        relevantOrbList=[]
        scSize=np.array([Lx,Ly,Lz])
        def findOrbIndexInRecordList(orb):
            for iexistedOrb, existedOrb in enumerate(relevantOrbList):
                if (existedOrb[0]==orb[0]%scSize).all() and existedOrb[1]==orb[1]:
                    return iexistedOrb
            return -1
        # build double scattering basis
        for scatter in scatterDoubleList:
            # standarize Bravais vectors
            scatter[0][0]%=scSize
            scatter[1][0]%=scSize
            if findOrbIndexInRecordList(scatter[0])<0:
                relevantOrbList.append(scatter[0])
            if findOrbIndexInRecordList(scatter[1])<0:
                relevantOrbList.append(scatter[1])
        scatteringOrder=len(relevantOrbList)
        print('find scattering region with %d-dimension, basis consist of'%scatteringOrder)
        for basis in relevantOrbList:
            print('orb%d at [%d %d %d]'%(basis[1],*basis[0]))
        # build scattering matrix
        scatteringDoubleCenter=np.zeros((scatteringOrder,scatteringOrder),dtype=np.complex128)
        for scatter in scatterDoubleList:
            sourceID=findOrbIndexInRecordList(scatter[0])
            targetID=findOrbIndexInRecordList(scatter[1])
            scatteringDoubleCenter[sourceID,targetID]=scatter[2]
        
        scatteringSingle1Center=np.zeros((scatteringOrder,scatteringOrder),dtype=np.complex128)
        for scatter in scatterSingle1List:
            sourceID=findOrbIndexInRecordList(scatter[0])
            targetID=findOrbIndexInRecordList(scatter[1])
            scatteringSingle1Center[sourceID,targetID]=scatter[2]
        
        scatteringSingle2Center=np.zeros((scatteringOrder,scatteringOrder),dtype=np.complex128)
        for scatter in scatterSingle2List:
            sourceID=findOrbIndexInRecordList(scatter[0])
            targetID=findOrbIndexInRecordList(scatter[1])
            scatteringSingle2Center[sourceID,targetID]=scatter[2]
        #print('scattering matrix is:')
        #print(scatteringDoubleCenter)

        c_scatteringDoubleCenter=(c_double*(scatteringOrder*scatteringOrder*2))()
        c_scatteringSingleCenter=(c_double*(scatteringOrder*scatteringOrder*2))()
        for i in range(scatteringOrder):
            for j in range(scatteringOrder):
                c_scatteringDoubleCenter[i*scatteringOrder*2+j*2]=c_double(scatteringDoubleCenter[i][j].real)
                c_scatteringDoubleCenter[i*scatteringOrder*2+j*2+1]=c_double(scatteringDoubleCenter[i][j].imag)

                c_scatteringSingleCenter[i*scatteringOrder*2+j*2]=c_double(scatteringSingle1Center[i][j].real)
                c_scatteringSingleCenter[i*scatteringOrder*2+j*2+1]=c_double(scatteringSingle1Center[i][j].imag)
        
        c_r=c_double((energyRange[1]-energyRange[0])/2)
        c_o=c_double((energyRange[0]+energyRange[1])/2)

        # split different data type and prepare for numba
        c_relevantOrbR=(c_int*(scatteringOrder*3))()
        c_relevantOrbIndex=(c_int*scatteringOrder)()
        for iorb, orb in enumerate(relevantOrbList):
            c_relevantOrbIndex[iorb]=c_int(orb[1])
            for i in range(3):
                c_relevantOrbR[iorb*3+i]=c_int(orb[0][i])
        
        BZIntegrationDLL=CDLL("./BZIntegration.dll")
        cfunc=BZIntegrationDLL.getEOnCircle
        cfunc.restype=py_object
        
        #print(type(hex(c_Hk_address)))
        #xit()
        #print("check the memory address is %s"%(hex(c_Hk_address),c_Hk_address,c_longlong(c_Hk_address)))
        def getEOnCircle(theta):
            return cfunc(c_int(nthreads),
                         c_double(theta), c_r, c_o, c_double(eta),
                         c_int(nkpt), c_longlong(c_kmesh_address),
                         c_int(norb), c_longlong(c_Hk_address),
                         c_int(scatteringOrder),
                         c_relevantOrbR,c_relevantOrbIndex,
                         c_scatteringDoubleCenter,
                         c_scatteringSingleCenter)
        
        #a=getEOnCircle(np.pi)
        #print(a)
        #exit()

        print('prepared for integration for J calculation')
        time0=time.time()
        E_coupling, E_coupling_err =integrate.quad(getEOnCircle,a=np.pi,b=0.,epsabs=paramReader.EDIFF)
        print("Coupling energy obtained in %.3fs (with cdll paralleled with %d threads) ===> %.6f +- %.6f"%(time.time()-time0,nthreads,E_coupling,E_coupling_err))
        
        J=E_coupling#/(-2*(np.cos(theta)-1))
        return J