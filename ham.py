import re
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

    def calcBerryCurvBands_CDLL(
        self,hr,atomicOrbIndexList:list,
        mainDir=None,fermi=0,saveFig=True,kplane=[[1,0,0],[0,1,0]],nthreads=1):
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
        
        c_lat=(c_double*9)()
        for i in range(3):
            c_lat[i]=c_double(pR.lat[0][i])
            c_lat[i+3]=c_double(pR.lat[1][i])
            c_lat[i+6]=c_double(pR.lat[2][i])

        assert len(atomicOrbIndexList)==sum(pR.nions), 'pls. provide correctly formatted atomicOrbIndexList[i_atom][i_L]:orb index'
        orbCorrList=[np.zeros(3) for i in range(hr.norb)]
        for atom, pos in zip(atomicOrbIndexList,pR.pos_ions):
            for L in atom:
                for iorb in L: orbCorrList[iorb]=pos
        #assert len(orbCorrList)==hr.norb, "number of wannier centers: %d inconsistent with %d orbs in hr.dat"%(len(orbCorrList),hr.norb)
        c_orbCorrList=(c_double*(3*hr.norb))()
        for i in range(hr.norb):
            c_orbCorrList[3*i+0]=c_double(orbCorrList[i][0])
            c_orbCorrList[3*i+1]=c_double(orbCorrList[i][1])
            c_orbCorrList[3*i+2]=c_double(orbCorrList[i][2])
        
        hamDLL=CDLL("./ham.dll")
        print("Warning: test is not enough, be cautious!")
        hamDLL.calcBandsVecsVelocityOps.restype=py_object
        result=hamDLL.calcBandsVecsVelocityOps(
            c_int(nkpts),c_kmesh,
            c_int(hr.norb),c_int(nhoppings),
            c_sourceID,c_targetID,c_overLat,c_hopping_real,c_hopping_imag,
            c_lat,c_orbCorrList,
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
        if not saveFig: return
        plt.figure()
        plt.plot([ikpt for ikpt in range(nkpts)],berryCurv)
        plt.xlim(0,nkpts-1)
        plt.xlabel(r'kpath')
        plt.ylabel(r'$e^2/\hbar$')
        plt.show()

        return
    
    def calcSpinHallConductor_CDLL(self,hr,mainDir=None,kgrid=[20,20,1],kplane=[[1,0,0],[0,1,0]],nMaxRefine=8,summedCvgCriteria=0.1,tau=0.01,nthreads=1):
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

        #tau=0.05
        result=hamDLL.calcSpinHallCond(c_int(nkpts),c_kmesh,c_dk,c_int(nMaxRefine),
                        c_int(hr.norb),c_int(nhoppings),
                        c_sourceID,c_targetID,c_overLat,c_hopping_real,c_hopping_imag,
                        c_int(paramReader.NFERMI),
                        c_fermi_list,
                        c_transFrame,
                        c_double(tau),
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

    def calcProjectiveSpinHallConductor_CDLL(self,
            hr,mainDir:str=None,kgrid=[20,20,20],
            atomicOrbIndexList=[], # give the correspondence of atom id to wannier id
            proj_op=[[[],[]]], # proj_op[iproj][0,1] give the atom id for source and target projection 
            scatter_route=[[0,0,0]], # scatter_route[iroute][istage=0,1,2] gives the id of projection  
            fermiList=[],erange=[-1000.,1000.],
            ndivide=1,nthreads=1,show=True):
        time0=time.time()
        assert mainDir!=None,'mainDir is required in projective BPVE calc.'
        # prepare input parameters for cdll
        kxaxis=np.linspace(0,1,kgrid[0],endpoint=False)
        kyaxis=np.linspace(0,1,kgrid[1],endpoint=False)
        kzaxis=np.linspace(0,1,kgrid[2],endpoint=False)
        kmesh=np.array([[k1,k2,k3] for k3 in kzaxis for k2 in kyaxis for k1 in kxaxis])

        print("lattice:")
        print("a1=(%11.6f %11.6f %11.6f)"%(*pR.lat[0],))
        print("a2=(%11.6f %11.6f %11.6f)"%(*pR.lat[1],))
        print("a3=(%11.6f %11.6f %11.6f)"%(*pR.lat[2],))

        c_lat=(c_double*9)()
        for i in range(3):
            c_lat[i]=c_double(pR.lat[0][i])
            c_lat[i+3]=c_double(pR.lat[1][i])
            c_lat[i+6]=c_double(pR.lat[2][i])

        assert len(atomicOrbIndexList)==sum(pR.nions), 'pls. provide correctly formatted atomicOrbIndexList[i_atom][i_L]:orb index'
        orbCorrList=[np.zeros(3) for i in range(hr.norb)]
        for atom, pos in zip(atomicOrbIndexList,pR.pos_ions):
            for L in atom:
                for iorb in L: orbCorrList[iorb]=pos
        #assert sum(orbCorrList)==hr.norb, "number of wannier centers: %d inconsistent with %d orbs in hr.dat"%(len(orbCorrList),hr.norb)
        c_orbCorrList=(c_double*(3*hr.norb))()
        for i in range(hr.norb):
            c_orbCorrList[3*i+0]=c_double(orbCorrList[i][0])
            c_orbCorrList[3*i+1]=c_double(orbCorrList[i][1])
            c_orbCorrList[3*i+2]=c_double(orbCorrList[i][2])

        # construct the orbital weights for projection
        nProj=len(proj_op)
        c_orbProj1WeightList=(c_int*(nProj*hr.norb))()
        c_orbProj2WeightList=(c_int*(nProj*hr.norb))()
        for iproj, proj in enumerate(proj_op):
            orbProj1WeightList=[0]*hr.norb
            for atomID in proj[0]:
                for L in atomicOrbIndexList[atomID]:
                    for iorb in L: orbProj1WeightList[iorb]=1
            orbProj2WeightList=[0]*hr.norb
            for atomID in proj[1]:
                for L in atomicOrbIndexList[atomID]:
                    for iorb in L: orbProj2WeightList[iorb]=1
            for i in range(hr.norb):
                c_orbProj1WeightList[iproj*hr.norb+i]=c_int(orbProj1WeightList[i])
                c_orbProj2WeightList[iproj*hr.norb+i]=c_int(orbProj2WeightList[i])
        # construct scatter routes
        nScatter=len(scatter_route)
        c_scatterRoute=(c_int*(nScatter*3))()
        for iscatter, scatter in enumerate(scatter_route):
            for i in range(3):c_scatterRoute[iscatter*3+i]=scatter[i]

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
        
        assert len(fermiList)>=1, 'omegaList is not defined or empty'
        c_fermi_list=(c_double*len(fermiList))()
        for ifermi, fermi in enumerate(fermiList):
            c_fermi_list[ifermi]=c_double(fermi)
        
        c_erange=(c_double*2)()
        c_erange[0]=c_double(erange[0])
        c_erange[1]=c_double(erange[1])
        #exit()
        hamDLL=CDLL("./ham.dll")
        time0=time.time()

        hamDLL.calcSpinHallCond_memoryReduce.restype=py_object
        result=hamDLL.calcSpinHallCond_memoryReduce(
                        c_int(nkpts), c_kmesh,
                        c_int(hr.norb), c_int(nhoppings),
                        c_sourceID, c_targetID, c_overLat,
                        c_hopping_real, c_hopping_imag,

                        c_int(len(fermiList)), c_fermi_list,
                        c_lat,
                        c_orbCorrList,
                        c_erange,

                        c_int(nProj), c_orbProj1WeightList, c_orbProj2WeightList,
                        c_int(nScatter),c_scatterRoute,

                        c_int(ndivide),
                        c_int(nthreads))
                        
        condFull_scatter_sx,condFull_scatter_sy,condFull_scatter_sz=np.array(result).reshape(3,nScatter,9,len(fermiList))
        for condFull_scatter, sname in zip([condFull_scatter_sx,condFull_scatter_sy,condFull_scatter_sz],['sx','sy','sz']):
            for iscatter in range(nScatter):
                cxx,cyx,czx,cxy,cyy,czy,cxz,cyz,czz=condFull_scatter[iscatter]
                with open(mainDir+'spinHallConductanceVsFermi_proj%d_%s.txt'%(iscatter,sname),'w') as fout:
                    fout.write("Fermi Sx_x Sy_x Sz_x Sx_y Sy_y Sz_y Sx_z Sy_z Sz_z\n")
                    for omega, _cxx,_cyx,_czx,_cxy,_cyy,_czy,_cxz,_cyz,_czz in zip(fermiList,cxx,cyx,czx,cxy,cyy,czy,cxz,cyz,czz):
                        fout.write(" %.4f %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E\n"%(
                            omega,_cxx,_cyx,_czx,_cxy,_cyy,_czy,_cxz,_cyz,_czz))

            
            if not show: continue
            #plt.figure()
            fig,ax=plt.subplots(3,3,sharey=True)
            ax[0,0].scatter(fermiList,cxx,label=r'$\sigma^{x,S}_{x}$')
            ax[0,1].scatter(fermiList,cyx,label=r'$\sigma^{y,S}_{x}$')
            ax[0,2].scatter(fermiList,czx,label=r'$\sigma^{z,S}_{x}$')
            ax[1,0].scatter(fermiList,cxx,label=r'$\sigma^{x,S}_{y}$')
            ax[1,1].scatter(fermiList,cyx,label=r'$\sigma^{y,S}_{y}$')
            ax[1,2].scatter(fermiList,czx,label=r'$\sigma^{z,S}_{y}$')
            ax[2,0].scatter(fermiList,cxx,label=r'$\sigma^{x,S}_{z}$')
            ax[2,1].scatter(fermiList,cyx,label=r'$\sigma^{y,S}_{z}$')
            ax[2,2].scatter(fermiList,czx,label=r'$\sigma^{z,S}_{z}$')
            for row in ax:
                for _ax in row:
                    _ax.legend()
                    _ax.plot([fermiList[0],fermiList[-1]],[0,0],'--',linewidth=1)
            plt.savefig(mainDir+'spin_%s_hallconductance.png'%sname,dpi=300)
            plt.show()
        print('Projective spin hall conductance solved in %.3fs with %d thread(s)'%(time.time()-time0,nthreads))
        return

    def calcBulkPhotovoltaicConductor_CDLL(self,hr,mainDir=None,
            kgrid=[20,20,1],atomicOrbIndexList=[],omegaList=[],tau=0.01,
            erange=[-1000.,1000.],ndivide=1,nthreads=1,show=True,
            full_mat=False):
        time0=time.time()
        # prepare input parameters for cdll
        kxaxis=np.linspace(0,1,kgrid[0],endpoint=False)
        kyaxis=np.linspace(0,1,kgrid[1],endpoint=False)
        kzaxis=np.linspace(0,1,kgrid[2],endpoint=False)
        kmesh=np.array([[k1,k2,k3] for k3 in kzaxis for k2 in kyaxis for k1 in kxaxis])

        c_lat=(c_double*9)()
        for i in range(3):
            c_lat[i]=c_double(pR.lat[0][i])
            c_lat[i+3]=c_double(pR.lat[1][i])
            c_lat[i+6]=c_double(pR.lat[2][i])

        assert len(atomicOrbIndexList)==sum(pR.nions), 'pls. provide correctly formatted atomicOrbIndexList[i_atom][i_L]:orb index'
        orbCorrList=[np.zeros(3) for i in range(hr.norb)]
        for atom, pos in zip(atomicOrbIndexList,pR.pos_ions):
            for L in atom:
                for iorb in L: orbCorrList[iorb]=pos
        #for pos in orbCorrList: print(pos)
        #exit()
        #assert sum(orbCorrList)==hr.norb, "number of wannier centers: %d inconsistent with %d orbs in hr.dat"%(len(orbCorrList),hr.norb)
        c_orbCorrList=(c_double*(3*hr.norb))()
        for i in range(hr.norb):
            c_orbCorrList[3*i+0]=c_double(orbCorrList[i][0])
            c_orbCorrList[3*i+1]=c_double(orbCorrList[i][1])
            c_orbCorrList[3*i+2]=c_double(orbCorrList[i][2])

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
        
        assert len(omegaList)>=1, 'omegaList is not defined or empty'
        c_omega_list=(c_double*len(omegaList))()
        for iomega, omega in enumerate(omegaList):
            c_omega_list[iomega]=c_double(omega)
        
        c_erange=(c_double*2)()
        c_erange[0]=c_double(erange[0])
        c_erange[1]=c_double(erange[1])
        #exit()
        hamDLL=CDLL("./ham.dll")
        myFunc=hamDLL.calcBulkPhotovoltaicCond_memoryReduce
        if full_mat:myFunc=hamDLL.calcBulkPhotovoltaicCond_FullMatrices_memoryReduce
        myFunc.restype=py_object

        #tau=0.05
        result=myFunc(c_int(nkpts),c_kmesh,
                    c_int(hr.norb),c_int(nhoppings),
                    c_sourceID,c_targetID,c_overLat,c_hopping_real,c_hopping_imag,
                    c_int(len(omegaList)),
                    c_omega_list,
                    c_double(paramReader.FERMI),
                    c_lat,
                    c_orbCorrList,
                    c_double(tau),
                    c_erange,
                    c_int(ndivide),
                    c_int(nthreads))
        print('Bulk photovoltaic conductance solved in %.3fs with %d thread(s)'%(time.time()-time0,nthreads))
        
        condFull=(np.array(result[0])+1j*np.array(result[1]))
        if mainDir!=None:
            if len(omegaList)==1:
                if full_mat:
                    cxxx,cxxy,cyyx,cyyy,cxyx,cxyy,cyxx,cyxy, czxx,czxy,czyx,czyy, cxzx,cxzy,cyzx,cyzy, cxxz,cxyz,cyxz,cyyz, czzx,czzy,czxz,czyz,cxzz,cyzz, czzz=condFull.reshape(27,nkpts)
                else:
                    cxxx,cxxy,cyyx,cyyy,cxyx,cxyy,cyxx,cyxy=condFull.reshape(8,nkpts)
                with open(mainDir+'photoCondTensor.txt','w') as fout:
                    for kpt, _cxxx,_cxxy,_cyyx,_cyyy,_cxyx,_cxyy,_cyxx,_cyxy in zip(kmesh,cxxx,cxxy,cyyx,cyyy,cxyx,cxyy,cyxx,cyxy):
                        fout.write(" %.4f %.4f %.4f %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E\n"%(*kpt,_cxxx.real,_cxxx.imag,_cyxx.real,_cyxx.imag,_cxyy.real,_cxyy.imag,_cyyy.real,_cyyy.imag,_cxxy.real,_cxxy.imag,_cyxy.real,_cyxy.imag,_cxyx.real,_cxyx.imag,_cyyx.real,_cyyx.imag))
                return
            else:
                if full_mat:
                    cxxx,cxxy,cyyx,cyyy,cxyx,cxyy,cyxx,cyxy, czxx,czxy,czyx,czyy, cxzx,cxzy,cyzx,cyzy, cxxz,cxyz,cyxz,cyyz, czzx,czzy,czxz,czyz,cxzz,cyzz, czzz=condFull.reshape(27,len(omegaList),nkpts)
                    with open(mainDir+'photoCondTensorVsOmega.txt','w') as fout:
                        fout.write("Omega Sxx_x_r Sxx_x_i Sxx_y_r Sxx_y_i Syy_x_r Syy_x_i Syy_y_r Syy_y_i Sxy_x_r Sxy_x_i Sxy_y_r Sxy_y_i Syx_x_r Syx_x_i Syx_y_r Syx_y_i Szx_x_r Szx_x_i Szx_y_r Szx_y_i Szy_x_r Szy_x_i Szy_y_r Szy_y_i Sxz_x_r Sxz_x_i Sxz_y_r Sxz_y_i Syz_x_r Syz_x_i Syz_y_r Syz_y_i Sxx_z_r Sxx_z_i Sxy_z_r Sxy_z_i Syx_z_r Syx_z_i Syy_z_r Syy_z_i Szz_x_r Szz_x_i Szz_y_r Szz_y_i Szx_z_r Szx_z_i Szy_z_r Szy_z_i Sxz_z_r Sxz_z_i Syz_z_r Syz_z_i Szz_z_r Szz_z_i\n")
                        for omega, _cxxx,_cxxy,_cyyx,_cyyy,_cxyx,_cxyy,_cyxx,_cyxy,_czxx,_czxy,_czyx,_czyy, _cxzx,_cxzy,_cyzx,_cyzy, _cxxz,_cxyz,_cyxz,_cyyz, _czzx,_czzy,_czxz,_czyz,_cxzz,_cyzz, _czzz in zip(omegaList,cxxx,cxxy,cyyx,cyyy,cxyx,cxyy,cyxx,cyxy,czxx,czxy,czyx,czyy, cxzx,cxzy,cyzx,cyzy, cxxz,cxyz,cyxz,cyyz, czzx,czzy,czxz,czyz,cxzz,cyzz, czzz):
                            fout.write(" %.4f %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E\n"%(
                                omega,
                                sum(_cxxx/nkpts).real,sum(_cxxx/nkpts).imag,
                                sum(_cxxy/nkpts).real,sum(_cxxy/nkpts).imag,
                                sum(_cyyx/nkpts).real,sum(_cyyx/nkpts).imag,
                                sum(_cyyy/nkpts).real,sum(_cyyy/nkpts).imag,
                                sum(_cxyx/nkpts).real,sum(_cxyx/nkpts).imag,
                                sum(_cxyy/nkpts).real,sum(_cxyy/nkpts).imag,
                                sum(_cyxx/nkpts).real,sum(_cyxx/nkpts).imag,
                                sum(_cyxy/nkpts).real,sum(_cyxy/nkpts).imag,

                                sum(_czxx/nkpts).real,sum(_czxx/nkpts).imag,
                                sum(_czxy/nkpts).real,sum(_czxy/nkpts).imag,
                                sum(_czyx/nkpts).real,sum(_czyx/nkpts).imag,
                                sum(_czyy/nkpts).real,sum(_czyy/nkpts).imag,

                                sum(_cxzx/nkpts).real,sum(_cxzx/nkpts).imag,
                                sum(_cxzy/nkpts).real,sum(_cxzy/nkpts).imag,
                                sum(_cyzx/nkpts).real,sum(_cyzx/nkpts).imag,
                                sum(_cyzy/nkpts).real,sum(_cyzy/nkpts).imag,

                                sum(_cxxz/nkpts).real,sum(_cxxz/nkpts).imag,
                                sum(_cxyz/nkpts).real,sum(_cxyz/nkpts).imag,
                                sum(_cyxz/nkpts).real,sum(_cyxz/nkpts).imag,
                                sum(_cyyz/nkpts).real,sum(_cyyz/nkpts).imag,

                                sum(_czzx/nkpts).real,sum(_czzx/nkpts).imag,
                                sum(_czzy/nkpts).real,sum(_czzy/nkpts).imag,
                                sum(_czxz/nkpts).real,sum(_czxz/nkpts).imag,
                                sum(_czyz/nkpts).real,sum(_czyz/nkpts).imag,
                                sum(_cxzz/nkpts).real,sum(_cxzz/nkpts).imag,
                                sum(_cyzz/nkpts).real,sum(_cyzz/nkpts).imag,

                                sum(_czzz/nkpts).real,sum(_czzz/nkpts).imag,
                                ))
                    for iomega, omega in enumerate(omegaList):
                        with open(mainDir+'photoCondTensor%d_E%.3f.txt'%(iomega,omega),'w') as fout:
                            for kpt, _cxxx,_cxxy,_cyyx,_cyyy,_cxyx,_cxyy,_cyxx,_cyxy,_czxx,_czxy,_czyx,_czyy, _cxzx,_cxzy,_cyzx,_cyzy, _cxxz,_cxyz,_cyxz,_cyyz, _czzx,_czzy,_czxz,_czyz,_cxzz,_cyzz, _czzz in zip(kmesh,cxxx[iomega],cxxy[iomega],cyyx[iomega],cyyy[iomega],cxyx[iomega],cxyy[iomega],cyxx[iomega],cyxy[iomega],czxx[iomega],czxy[iomega],czyx[iomega],czyy[iomega], cxzx[iomega],cxzy[iomega],cyzx[iomega],cyzy[iomega], cxxz[iomega],cxyz[iomega],cyxz[iomega],cyyz[iomega], czzx[iomega],czzy[iomega],czxz[iomega],czyz[iomega],cxzz[iomega],cyzz[iomega], czzz[iomega]):
                                fout.write(" %.4f %.4f %.4f %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E\n"%(
                                    *kpt,_cxxx.real,_cxxx.imag,_cyxx.real,_cyxx.imag,_cxyy.real,_cxyy.imag,_cyyy.real,_cyyy.imag,_cxxy.real,_cxxy.imag,_cyxy.real,_cyxy.imag,_cxyx.real,_cxyx.imag,_cyyx.real,_cyyx.imag,
                                    _czxx.real,_czxx.imag,_czxy.real,_czxy.imag,_czyx.real,_czyx.imag,_czyy.real,_czyy.imag,_cxzx.real,_cxzx.imag,_cxzy.real,_cxzy.imag,_cyzx.real,_cyzx.imag,_cyzy.real,_cyzy.imag,_cxxz.real,_cxxz.imag,_cxyz.real,_cxyz.imag,_cyxz.real,_cyxz.imag,_cyyz.real,_cyyz.imag,_czzx.real,_czzx.imag,_czzy.real,_czzy.imag,_czxz.real,_czxz.imag,_czyz.real,_czyz.imag,_cxzz.real,_cxzz.imag,_cyzz.real,_cyzz.imag,_czzz.real,_czzz.imag,
                                    ))
                else:
                    cxxx,cyxx,cxyy,cyyy,cxxy,cyxy,cxyx,cyyx=condFull.reshape(8,len(omegaList),nkpts)
                    with open(mainDir+'photoCondTensorVsOmega.txt','w') as fout:
                        fout.write("Omega Sxx_x_r Sxx_x_i Sxx_y_r Sxx_y_i Syy_x_r Syy_x_i Syy_y_r Syy_y_i Sxy_x_r Sxy_x_i Sxy_y_r Sxy_y_i Syx_x_r Syx_x_i Syx_y_r Syx_y_i\n")
                        for omega, _cxxx,_cxxy,_cyyx,_cyyy,_cxyx,_cxyy,_cyxx,_cyxy in zip(omegaList,cxxx,cxxy,cyyx,cyyy,cxyx,cxyy,cyxx,cyxy):
                            fout.write(" %.4f %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E\n"%(
                                omega,
                                sum(_cxxx/nkpts).real,sum(_cxxx/nkpts).imag,
                                sum(_cxxy/nkpts).real,sum(_cxxy/nkpts).imag,
                                sum(_cyyx/nkpts).real,sum(_cyyx/nkpts).imag,
                                sum(_cyyy/nkpts).real,sum(_cyyy/nkpts).imag,
                                sum(_cxyx/nkpts).real,sum(_cxyx/nkpts).imag,
                                sum(_cxyy/nkpts).real,sum(_cxyy/nkpts).imag,
                                sum(_cyxx/nkpts).real,sum(_cyxx/nkpts).imag,
                                sum(_cyxy/nkpts).real,sum(_cyxy/nkpts).imag))
                    for iomega, omega in enumerate(omegaList):
                        with open(mainDir+'photoCondTensor%d_E%.3f.txt'%(iomega,omega),'w') as fout:
                            for kpt, _cxxx,_cxxy,_cyyx,_cyyy,_cxyx,_cxyy,_cyxx,_cyxy in zip(kmesh,cxxx[iomega],cxxy[iomega],cyyx[iomega],cyyy[iomega],cxyx[iomega],cxyy[iomega],cyxx[iomega],cyxy[iomega]):
                                fout.write(" %.4f %.4f %.4f %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E\n"%(*kpt,_cxxx.real,_cxxx.imag,_cyxx.real,_cyxx.imag,_cxyy.real,_cxyy.imag,_cyyy.real,_cyyy.imag,_cxxy.real,_cxxy.imag,_cyxy.real,_cyxy.imag,_cxyx.real,_cxyx.imag,_cyyx.real,_cyyx.imag))
        if not show:
            return
        condcxxxList=np.array([np.sum(cond_on_omega/nkpts) for cond_on_omega in cxxx])
        condcxxyList=np.array([np.sum(cond_on_omega/nkpts) for cond_on_omega in cxxy])
        condcyyxList=np.array([np.sum(cond_on_omega/nkpts) for cond_on_omega in cyyx])
        condcyyyList=np.array([np.sum(cond_on_omega/nkpts) for cond_on_omega in cyyy])
        condcxyxList=np.array([np.sum(cond_on_omega/nkpts) for cond_on_omega in cxyx])
        condcxyyList=np.array([np.sum(cond_on_omega/nkpts) for cond_on_omega in cxyy])
        condcyxxList=np.array([np.sum(cond_on_omega/nkpts) for cond_on_omega in cyxx])
        condcyxyList=np.array([np.sum(cond_on_omega/nkpts) for cond_on_omega in cyxy])
        
        #exit()
        fig,ax=plt.subplots(2,4,sharey=True)
        ax[0,0].scatter(omegaList,condcxxxList.real,label=r'$Re\ \sigma^x_{xx}$')
        ax[0,1].scatter(omegaList,condcxxyList.real,label=r'$Re\ \sigma^y_{xx}$')
        ax[0,2].scatter(omegaList,condcyyxList.real,label=r'$Re\ \sigma^x_{yy}$')
        ax[0,3].scatter(omegaList,condcyyyList.real,label=r'$Re\ \sigma^y_{yy}$')
        ax[1,0].scatter(omegaList,condcxyxList.real,label=r'$Re\ \sigma^x_{xy}$')
        ax[1,1].scatter(omegaList,condcxyyList.real,label=r'$Re\ \sigma^y_{xy}$')
        ax[1,2].scatter(omegaList,condcyxxList.real,label=r'$Re\ \sigma^x_{yx}$')
        ax[1,3].scatter(omegaList,condcyxyList.real,label=r'$Re\ \sigma^y_{yx}$')
        for row in ax:
            #row.legend()
            #row.plot([omegaList[0],omegaList[-1]],[0,0],'--',linewidth=1)
            for _ax in row:
                _ax.legend()
                _ax.plot([omegaList[0],omegaList[-1]],[0,0],'--',linewidth=1)
        if mainDir!=None: plt.savefig(mainDir+'photoconductance.png',dpi=300)
        plt.show()
        return
    
    def calcBulkPhotovoltaicConductor_mixFFT(self,hr,mainDir=None,
            kgrid=[20,20,1],atomicOrbIndexList=[],omegaList=[],tau=0.01,
            erange=[-1000.,1000.],ndivide=1,nthreads=1,show=True,
            full_mat=False):
        
        time0=time.time()
        gr=Green()
        eig_addr, vec_addr, velo_addr=gr.calcEigsAndVecsInMemory_mixedFourier(hr,kgrid,atomicOrbIndexList,nthreads)
        print("velocity finished in %.3f"%(time.time()-time0))

        #time0=time.time()
        kxaxis=np.linspace(0,1,kgrid[0],endpoint=False)
        kyaxis=np.linspace(0,1,kgrid[1],endpoint=False)
        kzaxis=np.linspace(0,1,kgrid[2],endpoint=False)
        kmesh=np.array([[k1,k2,k3] for k3 in kzaxis for k2 in kyaxis for k1 in kxaxis])

        # prepare input parameters for cdll
        nkpts=np.prod(kgrid)
        assert len(omegaList)>=1, 'omegaList is not defined or empty'
        c_omega_list=(c_double*len(omegaList))()
        for iomega, omega in enumerate(omegaList):
            c_omega_list[iomega]=c_double(omega)
        
        c_erange=(c_double*2)()
        c_erange[0]=c_double(erange[0])
        c_erange[1]=c_double(erange[1])
        #exit()
        hamDLL=CDLL("./ham.dll")
        myFunc=hamDLL.calcBulkPhotovoltaicCond_FullMatrices_mixFFT
        if full_mat:myFunc=hamDLL.calcBulkPhotovoltaicCond_FullMatrices_mixFFT
        myFunc.restype=py_object

        #tau=0.05
        result=myFunc(c_longlong(eig_addr), c_longlong(velo_addr),
                    c_int(nkpts),c_int(hr.norb),
                    c_int(len(omegaList)),
                    c_omega_list,
                    c_double(paramReader.FERMI),
                    c_double(tau),
                    c_erange,
                    c_int(nthreads))
        print('Bulk photovoltaic conductance solved in %.3fs with %d thread(s)'%(time.time()-time0,nthreads))

        condFull=(np.array(result[0])+1j*np.array(result[1]))
        if mainDir!=None:
            if len(omegaList)==1:
                if full_mat:
                    cxxx,cxxy,cyyx,cyyy,cxyx,cxyy,cyxx,cyxy, czxx,czxy,czyx,czyy, cxzx,cxzy,cyzx,cyzy, cxxz,cxyz,cyxz,cyyz, czzx,czzy,czxz,czyz,cxzz,cyzz, czzz=condFull.reshape(27,nkpts)
                else:
                    cxxx,cxxy,cyyx,cyyy,cxyx,cxyy,cyxx,cyxy=condFull.reshape(8,nkpts)
                with open(mainDir+'photoCondTensor.txt','w') as fout:
                    for kpt, _cxxx,_cxxy,_cyyx,_cyyy,_cxyx,_cxyy,_cyxx,_cyxy in zip(kmesh,cxxx,cxxy,cyyx,cyyy,cxyx,cxyy,cyxx,cyxy):
                        fout.write(" %.4f %.4f %.4f %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E\n"%(*kpt,_cxxx.real,_cxxx.imag,_cyxx.real,_cyxx.imag,_cxyy.real,_cxyy.imag,_cyyy.real,_cyyy.imag,_cxxy.real,_cxxy.imag,_cyxy.real,_cyxy.imag,_cxyx.real,_cxyx.imag,_cyyx.real,_cyyx.imag))
                return
            else:
                if full_mat:
                    cxxx,cxxy,cyyx,cyyy,cxyx,cxyy,cyxx,cyxy, czxx,czxy,czyx,czyy, cxzx,cxzy,cyzx,cyzy, cxxz,cxyz,cyxz,cyyz, czzx,czzy,czxz,czyz,cxzz,cyzz, czzz=condFull.reshape(27,len(omegaList),nkpts)
                    with open(mainDir+'photoCondTensorVsOmega.txt','w') as fout:
                        fout.write("Omega Sxx_x_r Sxx_x_i Sxx_y_r Sxx_y_i Syy_x_r Syy_x_i Syy_y_r Syy_y_i Sxy_x_r Sxy_x_i Sxy_y_r Sxy_y_i Syx_x_r Syx_x_i Syx_y_r Syx_y_i Szx_x_r Szx_x_i Szx_y_r Szx_y_i Szy_x_r Szy_x_i Szy_y_r Szy_y_i Sxz_x_r Sxz_x_i Sxz_y_r Sxz_y_i Syz_x_r Syz_x_i Syz_y_r Syz_y_i Sxx_z_r Sxx_z_i Sxy_z_r Sxy_z_i Syx_z_r Syx_z_i Syy_z_r Syy_z_i Szz_x_r Szz_x_i Szz_y_r Szz_y_i Szx_z_r Szx_z_i Szy_z_r Szy_z_i Sxz_z_r Sxz_z_i Syz_z_r Syz_z_i Szz_z_r Szz_z_i\n")
                        for omega, _cxxx,_cxxy,_cyyx,_cyyy,_cxyx,_cxyy,_cyxx,_cyxy,_czxx,_czxy,_czyx,_czyy, _cxzx,_cxzy,_cyzx,_cyzy, _cxxz,_cxyz,_cyxz,_cyyz, _czzx,_czzy,_czxz,_czyz,_cxzz,_cyzz, _czzz in zip(omegaList,cxxx,cxxy,cyyx,cyyy,cxyx,cxyy,cyxx,cyxy,czxx,czxy,czyx,czyy, cxzx,cxzy,cyzx,cyzy, cxxz,cxyz,cyxz,cyyz, czzx,czzy,czxz,czyz,cxzz,cyzz, czzz):
                            fout.write(" %.4f %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E\n"%(
                                omega,
                                sum(_cxxx/nkpts).real,sum(_cxxx/nkpts).imag,
                                sum(_cxxy/nkpts).real,sum(_cxxy/nkpts).imag,
                                sum(_cyyx/nkpts).real,sum(_cyyx/nkpts).imag,
                                sum(_cyyy/nkpts).real,sum(_cyyy/nkpts).imag,
                                sum(_cxyx/nkpts).real,sum(_cxyx/nkpts).imag,
                                sum(_cxyy/nkpts).real,sum(_cxyy/nkpts).imag,
                                sum(_cyxx/nkpts).real,sum(_cyxx/nkpts).imag,
                                sum(_cyxy/nkpts).real,sum(_cyxy/nkpts).imag,

                                sum(_czxx/nkpts).real,sum(_czxx/nkpts).imag,
                                sum(_czxy/nkpts).real,sum(_czxy/nkpts).imag,
                                sum(_czyx/nkpts).real,sum(_czyx/nkpts).imag,
                                sum(_czyy/nkpts).real,sum(_czyy/nkpts).imag,

                                sum(_cxzx/nkpts).real,sum(_cxzx/nkpts).imag,
                                sum(_cxzy/nkpts).real,sum(_cxzy/nkpts).imag,
                                sum(_cyzx/nkpts).real,sum(_cyzx/nkpts).imag,
                                sum(_cyzy/nkpts).real,sum(_cyzy/nkpts).imag,

                                sum(_cxxz/nkpts).real,sum(_cxxz/nkpts).imag,
                                sum(_cxyz/nkpts).real,sum(_cxyz/nkpts).imag,
                                sum(_cyxz/nkpts).real,sum(_cyxz/nkpts).imag,
                                sum(_cyyz/nkpts).real,sum(_cyyz/nkpts).imag,

                                sum(_czzx/nkpts).real,sum(_czzx/nkpts).imag,
                                sum(_czzy/nkpts).real,sum(_czzy/nkpts).imag,
                                sum(_czxz/nkpts).real,sum(_czxz/nkpts).imag,
                                sum(_czyz/nkpts).real,sum(_czyz/nkpts).imag,
                                sum(_cxzz/nkpts).real,sum(_cxzz/nkpts).imag,
                                sum(_cyzz/nkpts).real,sum(_cyzz/nkpts).imag,

                                sum(_czzz/nkpts).real,sum(_czzz/nkpts).imag,
                                ))
                    for iomega, omega in enumerate(omegaList):
                        with open(mainDir+'photoCondTensor%d_E%.3f.txt'%(iomega,omega),'w') as fout:
                            for kpt, _cxxx,_cxxy,_cyyx,_cyyy,_cxyx,_cxyy,_cyxx,_cyxy,_czxx,_czxy,_czyx,_czyy, _cxzx,_cxzy,_cyzx,_cyzy, _cxxz,_cxyz,_cyxz,_cyyz, _czzx,_czzy,_czxz,_czyz,_cxzz,_cyzz, _czzz in zip(kmesh,cxxx[iomega],cxxy[iomega],cyyx[iomega],cyyy[iomega],cxyx[iomega],cxyy[iomega],cyxx[iomega],cyxy[iomega],czxx[iomega],czxy[iomega],czyx[iomega],czyy[iomega], cxzx[iomega],cxzy[iomega],cyzx[iomega],cyzy[iomega], cxxz[iomega],cxyz[iomega],cyxz[iomega],cyyz[iomega], czzx[iomega],czzy[iomega],czxz[iomega],czyz[iomega],cxzz[iomega],cyzz[iomega], czzz[iomega]):
                                fout.write(" %.4f %.4f %.4f %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E\n"%(
                                    *kpt,_cxxx.real,_cxxx.imag,_cyxx.real,_cyxx.imag,_cxyy.real,_cxyy.imag,_cyyy.real,_cyyy.imag,_cxxy.real,_cxxy.imag,_cyxy.real,_cyxy.imag,_cxyx.real,_cxyx.imag,_cyyx.real,_cyyx.imag,
                                    _czxx.real,_czxx.imag,_czxy.real,_czxy.imag,_czyx.real,_czyx.imag,_czyy.real,_czyy.imag,_cxzx.real,_cxzx.imag,_cxzy.real,_cxzy.imag,_cyzx.real,_cyzx.imag,_cyzy.real,_cyzy.imag,_cxxz.real,_cxxz.imag,_cxyz.real,_cxyz.imag,_cyxz.real,_cyxz.imag,_cyyz.real,_cyyz.imag,_czzx.real,_czzx.imag,_czzy.real,_czzy.imag,_czxz.real,_czxz.imag,_czyz.real,_czyz.imag,_cxzz.real,_cxzz.imag,_cyzz.real,_cyzz.imag,_czzz.real,_czzz.imag,
                                    ))
                else:
                    cxxx,cyxx,cxyy,cyyy,cxxy,cyxy,cxyx,cyyx=condFull.reshape(8,len(omegaList),nkpts)
                    with open(mainDir+'photoCondTensorVsOmega.txt','w') as fout:
                        fout.write("Omega Sxx_x_r Sxx_x_i Sxx_y_r Sxx_y_i Syy_x_r Syy_x_i Syy_y_r Syy_y_i Sxy_x_r Sxy_x_i Sxy_y_r Sxy_y_i Syx_x_r Syx_x_i Syx_y_r Syx_y_i\n")
                        for omega, _cxxx,_cxxy,_cyyx,_cyyy,_cxyx,_cxyy,_cyxx,_cyxy in zip(omegaList,cxxx,cxxy,cyyx,cyyy,cxyx,cxyy,cyxx,cyxy):
                            fout.write(" %.4f %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E\n"%(
                                omega,
                                sum(_cxxx/nkpts).real,sum(_cxxx/nkpts).imag,
                                sum(_cxxy/nkpts).real,sum(_cxxy/nkpts).imag,
                                sum(_cyyx/nkpts).real,sum(_cyyx/nkpts).imag,
                                sum(_cyyy/nkpts).real,sum(_cyyy/nkpts).imag,
                                sum(_cxyx/nkpts).real,sum(_cxyx/nkpts).imag,
                                sum(_cxyy/nkpts).real,sum(_cxyy/nkpts).imag,
                                sum(_cyxx/nkpts).real,sum(_cyxx/nkpts).imag,
                                sum(_cyxy/nkpts).real,sum(_cyxy/nkpts).imag))
                    for iomega, omega in enumerate(omegaList):
                        with open(mainDir+'photoCondTensor%d_E%.3f.txt'%(iomega,omega),'w') as fout:
                            for kpt, _cxxx,_cxxy,_cyyx,_cyyy,_cxyx,_cxyy,_cyxx,_cyxy in zip(kmesh,cxxx[iomega],cxxy[iomega],cyyx[iomega],cyyy[iomega],cxyx[iomega],cxyy[iomega],cyxx[iomega],cyxy[iomega]):
                                fout.write(" %.4f %.4f %.4f %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E\n"%(*kpt,_cxxx.real,_cxxx.imag,_cyxx.real,_cyxx.imag,_cxyy.real,_cxyy.imag,_cyyy.real,_cyyy.imag,_cxxy.real,_cxxy.imag,_cyxy.real,_cyxy.imag,_cxyx.real,_cxyx.imag,_cyyx.real,_cyyx.imag))
        if not show:
            return
        condcxxxList=np.array([np.sum(cond_on_omega/nkpts) for cond_on_omega in cxxx])
        condcxxyList=np.array([np.sum(cond_on_omega/nkpts) for cond_on_omega in cxxy])
        condcyyxList=np.array([np.sum(cond_on_omega/nkpts) for cond_on_omega in cyyx])
        condcyyyList=np.array([np.sum(cond_on_omega/nkpts) for cond_on_omega in cyyy])
        condcxyxList=np.array([np.sum(cond_on_omega/nkpts) for cond_on_omega in cxyx])
        condcxyyList=np.array([np.sum(cond_on_omega/nkpts) for cond_on_omega in cxyy])
        condcyxxList=np.array([np.sum(cond_on_omega/nkpts) for cond_on_omega in cyxx])
        condcyxyList=np.array([np.sum(cond_on_omega/nkpts) for cond_on_omega in cyxy])
        #exit()
        fig,ax=plt.subplots(2,4,sharey=True)
        ax[0,0].scatter(omegaList,condcxxxList.real,label=r'$Re\ \sigma^x_{xx}$')
        ax[0,1].scatter(omegaList,condcxxyList.real,label=r'$Re\ \sigma^y_{xx}$')
        ax[0,2].scatter(omegaList,condcyyxList.real,label=r'$Re\ \sigma^x_{yy}$')
        ax[0,3].scatter(omegaList,condcyyyList.real,label=r'$Re\ \sigma^y_{yy}$')
        ax[1,0].scatter(omegaList,condcxyxList.real,label=r'$Re\ \sigma^x_{xy}$')
        ax[1,1].scatter(omegaList,condcxyyList.real,label=r'$Re\ \sigma^y_{xy}$')
        ax[1,2].scatter(omegaList,condcyxxList.real,label=r'$Re\ \sigma^x_{yx}$')
        ax[1,3].scatter(omegaList,condcyxyList.real,label=r'$Re\ \sigma^y_{yx}$')
        for row in ax:
            #row.legend()
            #row.plot([omegaList[0],omegaList[-1]],[0,0],'--',linewidth=1)
            for _ax in row:
                _ax.legend()
                _ax.plot([omegaList[0],omegaList[-1]],[0,0],'--',linewidth=1)
        if mainDir!=None: plt.savefig(mainDir+'photoconductance.png',dpi=300)
        plt.show()
        return
    
    def calcShiftPhotovoltaicConductor_CDLL(self,hr,mainDir=None,kgrid=[20,20,1],atomicOrbIndexList=[],omegaList=[],tau=0.01,erange=[-1000.,1000.],nthreads=1):
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

        Vbz=np.linalg.det(np.linalg.inv(pR.lat))

        c_lat=(c_double*9)()
        for i in range(3):
            c_lat[i]=c_double(pR.lat[0][i])
            c_lat[i+3]=c_double(pR.lat[1][i])
            c_lat[i+6]=c_double(pR.lat[2][i])

        assert len(atomicOrbIndexList)==sum(pR.nions), 'pls. provide correctly formatted atomicOrbIndexList[i_atom][i_L]:orb index'
        orbCorrList=[np.zeros(3) for i in range(hr.norb)]
        for atom, pos in zip(atomicOrbIndexList,pR.pos_ions):
            for L in atom:
                for iorb in L: orbCorrList[iorb]=pos
        #assert sum(orbCorrList)==hr.norb, "number of wannier centers: %d inconsistent with %d orbs in hr.dat"%(len(orbCorrList),hr.norb)
        c_orbCorrList=(c_double*(3*hr.norb))()
        for i in range(hr.norb):
            c_orbCorrList[3*i+0]=c_double(orbCorrList[i][0])
            c_orbCorrList[3*i+1]=c_double(orbCorrList[i][1])
            c_orbCorrList[3*i+2]=c_double(orbCorrList[i][2])

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
        
        assert len(omegaList)>=1, 'omegaList is not defined or empty'
        c_omega_list=(c_double*len(omegaList))()
        for iomega, omega in enumerate(omegaList):
            c_omega_list[iomega]=c_double(omega)
        
        c_erange=(c_double*2)()
        c_erange[0]=c_double(erange[0])
        c_erange[1]=c_double(erange[1])
        #exit()
        hamDLL=CDLL("./ham.dll")
        hamDLL.calcShiftCond.restype=py_object

        #tau=0.05
        result=hamDLL.calcShiftCond(c_int(nkpts),c_kmesh,c_dk,
                        c_int(hr.norb),c_int(nhoppings),
                        c_sourceID,c_targetID,c_overLat,c_hopping_real,c_hopping_imag,
                        c_int(len(omegaList)),
                        c_omega_list,
                        c_double(paramReader.FERMI),
                        c_lat,
                        c_orbCorrList,
                        c_double(tau),
                        c_erange,
                        c_int(nthreads))
                        
        condFull=(np.array(result[0])+1j*np.array(result[1]))#.reshape(8,len(omegaList)*nkpts)
        if mainDir!=None:
            if len(omegaList)==1:
                cxxx,cxxy,cyyx,cyyy,cxyx,cxyy,cyxx,cyxy=condFull.reshape(8,nkpts)
                with open(mainDir+'shiftCondTensor.txt','w') as fout:
                    for kpt, _cxxx,_cxxy,_cyyx,_cyyy,_cxyx,_cxyy,_cyxx,_cyxy in zip(kmesh,cxxx,cxxy,cyyx,cyyy,cxyx,cxyy,cyxx,cyxy):
                        fout.write(" %.4f %.4f %.4f %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E\n"%(*kpt,_cxxx.real,_cxxx.imag,_cyxx.real,_cyxx.imag,_cxyy.real,_cxyy.imag,_cyyy.real,_cyyy.imag,_cxxy.real,_cxxy.imag,_cyxy.real,_cyxy.imag,_cxyx.real,_cxyx.imag,_cyyx.real,_cyyx.imag))
                return
            else:
                cxxx,cyxx,cxyy,cyyy,cxxy,cyxy,cxyx,cyyx=condFull.reshape(8,len(omegaList),nkpts)
                with open(mainDir+'shiftCondTensorVsOmega.txt','w') as fout:
                    fout.write("Omega Sxx_x_r Sxx_x_i Sxx_y_r Sxx_y_i Syy_x_r Syy_x_i Syy_y_r Syy_y_i Sxy_x_r Sxy_x_i Sxy_y_r Sxy_y_i Syx_x_r Syx_x_i Syx_y_r Syx_y_i\n")
                    for omega, _cxxx,_cxxy,_cyyx,_cyyy,_cxyx,_cxyy,_cyxx,_cyxy in zip(omegaList,cxxx,cxxy,cyyx,cyyy,cxyx,cxyy,cyxx,cyxy):
                        fout.write(" %.4f %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E\n"%(
                            omega,
                            sum(_cxxx/nkpts).real,sum(_cxxx/nkpts).imag,
                            sum(_cxxy/nkpts).real,sum(_cxxy/nkpts).imag,
                            sum(_cyyx/nkpts).real,sum(_cyyx/nkpts).imag,
                            sum(_cyyy/nkpts).real,sum(_cyyy/nkpts).imag,
                            sum(_cxyx/nkpts).real,sum(_cxyx/nkpts).imag,
                            sum(_cxyy/nkpts).real,sum(_cxyy/nkpts).imag,
                            sum(_cyxx/nkpts).real,sum(_cyxx/nkpts).imag,
                            sum(_cyxy/nkpts).real,sum(_cyxy/nkpts).imag))
                for iomega, omega in enumerate(omegaList):
                    with open(mainDir+'shiftCondTensor%d_E%.3f.txt'%(iomega,omega),'w') as fout:
                        for kpt, _cxxx,_cxxy,_cyyx,_cyyy,_cxyx,_cxyy,_cyxx,_cyxy in zip(kmesh,cxxx[iomega],cxxy[iomega],cyyx[iomega],cyyy[iomega],cxyx[iomega],cxyy[iomega],cyxx[iomega],cyxy[iomega]):
                            fout.write(" %.4f %.4f %.4f %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E\n"%(*kpt,_cxxx.real,_cxxx.imag,_cyxx.real,_cyxx.imag,_cxyy.real,_cxyy.imag,_cyyy.real,_cyyy.imag,_cxxy.real,_cxxy.imag,_cyxy.real,_cyxy.imag,_cxyx.real,_cxyx.imag,_cyyx.real,_cyyx.imag))

        condcxxxList=np.array([np.sum(cond_on_omega/nkpts) for cond_on_omega in cxxx])
        condcxxyList=np.array([np.sum(cond_on_omega/nkpts) for cond_on_omega in cxxy])
        condcyyxList=np.array([np.sum(cond_on_omega/nkpts) for cond_on_omega in cyyx])
        condcyyyList=np.array([np.sum(cond_on_omega/nkpts) for cond_on_omega in cyyy])
        condcxyxList=np.array([np.sum(cond_on_omega/nkpts) for cond_on_omega in cxyx])
        condcxyyList=np.array([np.sum(cond_on_omega/nkpts) for cond_on_omega in cxyy])
        condcyxxList=np.array([np.sum(cond_on_omega/nkpts) for cond_on_omega in cyxx])
        condcyxyList=np.array([np.sum(cond_on_omega/nkpts) for cond_on_omega in cyxy])
        print('Shift photovoltaic conductance solved in %.3fs with %d thread(s)'%(time.time()-time0,nthreads))
        #exit()
        fig,ax=plt.subplots(2,4,sharey=True)
        ax[0,0].scatter(omegaList,condcxxxList.real,label=r'$Re\ \sigma^x_{xx}$')
        ax[0,1].scatter(omegaList,condcxxyList.real,label=r'$Re\ \sigma^y_{xx}$')
        ax[0,2].scatter(omegaList,condcyyxList.real,label=r'$Re\ \sigma^x_{yy}$')
        ax[0,3].scatter(omegaList,condcyyyList.real,label=r'$Re\ \sigma^y_{yy}$')
        ax[1,0].scatter(omegaList,condcxyxList.real,label=r'$Re\ \sigma^x_{xy}$')
        ax[1,1].scatter(omegaList,condcxyyList.real,label=r'$Re\ \sigma^y_{xy}$')
        ax[1,2].scatter(omegaList,condcyxxList.real,label=r'$Re\ \sigma^x_{yx}$')
        ax[1,3].scatter(omegaList,condcyxyList.real,label=r'$Re\ \sigma^y_{yx}$')
        for row in ax:
            #row.legend()
            #row.plot([omegaList[0],omegaList[-1]],[0,0],'--',linewidth=1)
            for _ax in row:
                _ax.legend()
                _ax.plot([omegaList[0],omegaList[-1]],[0,0],'--',linewidth=1)
        if mainDir!=None: plt.savefig(mainDir+'shiftconductance.png',dpi=300)
        plt.show()
        return
    
    def calcInjectPhotovoltaicConductor_CDLL(self,hr,mainDir=None,kgrid=[20,20,1],atomicOrbIndexList=[],omegaList=[],tau=0.01,erange=[-1000.,1000.],nthreads=1):
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

        Vbz=np.linalg.det(np.linalg.inv(pR.lat))

        c_lat=(c_double*9)()
        for i in range(3):
            c_lat[i]=c_double(pR.lat[0][i])
            c_lat[i+3]=c_double(pR.lat[1][i])
            c_lat[i+6]=c_double(pR.lat[2][i])

        assert len(atomicOrbIndexList)==sum(pR.nions), 'pls. provide correctly formatted atomicOrbIndexList[i_atom][i_L]:orb index'
        orbCorrList=[np.zeros(3) for i in range(hr.norb)]
        for atom, pos in zip(atomicOrbIndexList,pR.pos_ions):
            for L in atom:
                for iorb in L: orbCorrList[iorb]=pos
        #assert sum(orbCorrList)==hr.norb, "number of wannier centers: %d inconsistent with %d orbs in hr.dat"%(len(orbCorrList),hr.norb)
        c_orbCorrList=(c_double*(3*hr.norb))()
        for i in range(hr.norb):
            c_orbCorrList[3*i+0]=c_double(orbCorrList[i][0])
            c_orbCorrList[3*i+1]=c_double(orbCorrList[i][1])
            c_orbCorrList[3*i+2]=c_double(orbCorrList[i][2])

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
        
        assert len(omegaList)>=1, 'omegaList is not defined or empty'
        c_omega_list=(c_double*len(omegaList))()
        for iomega, omega in enumerate(omegaList):
            c_omega_list[iomega]=c_double(omega)
        
        c_erange=(c_double*2)()
        c_erange[0]=c_double(erange[0])
        c_erange[1]=c_double(erange[1])
        #exit()
        hamDLL=CDLL("./ham.dll")
        hamDLL.calcInjectCond.restype=py_object

        #tau=0.05
        result=hamDLL.calcInjectCond(c_int(nkpts),c_kmesh,c_dk,
                        c_int(hr.norb),c_int(nhoppings),
                        c_sourceID,c_targetID,c_overLat,c_hopping_real,c_hopping_imag,
                        c_int(len(omegaList)),
                        c_omega_list,
                        c_double(paramReader.FERMI),
                        c_lat,
                        c_orbCorrList,
                        c_double(tau),
                        c_erange,
                        c_int(nthreads))
                        
        condFull=(np.array(result[0])+1j*np.array(result[1]))#.reshape(8,len(omegaList)*nkpts)
        if mainDir!=None:
            if len(omegaList)==1:
                cxxx,cxxy,cyyx,cyyy,cxyx,cxyy,cyxx,cyxy=condFull.reshape(8,nkpts)
                with open(mainDir+'injectionCondTensor.txt','w') as fout:
                    for kpt, _cxxx,_cxxy,_cyyx,_cyyy,_cxyx,_cxyy,_cyxx,_cyxy in zip(kmesh,cxxx,cxxy,cyyx,cyyy,cxyx,cxyy,cyxx,cyxy):
                        fout.write(" %.4f %.4f %.4f %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E\n"%(*kpt,_cxxx.real,_cxxx.imag,_cyxx.real,_cyxx.imag,_cxyy.real,_cxyy.imag,_cyyy.real,_cyyy.imag,_cxxy.real,_cxxy.imag,_cyxy.real,_cyxy.imag,_cxyx.real,_cxyx.imag,_cyyx.real,_cyyx.imag))
                return
            else:
                cxxx,cyxx,cxyy,cyyy,cxxy,cyxy,cxyx,cyyx=condFull.reshape(8,len(omegaList),nkpts)
                with open(mainDir+'injectionCondTensorVsOmega.txt','w') as fout:
                    fout.write("Omega Sxx_x_r Sxx_x_i Sxx_y_r Sxx_y_i Syy_x_r Syy_x_i Syy_y_r Syy_y_i Sxy_x_r Sxy_x_i Sxy_y_r Sxy_y_i Syx_x_r Syx_x_i Syx_y_r Syx_y_i\n")
                    for omega, _cxxx,_cxxy,_cyyx,_cyyy,_cxyx,_cxyy,_cyxx,_cyxy in zip(omegaList,cxxx,cxxy,cyyx,cyyy,cxyx,cxyy,cyxx,cyxy):
                        fout.write(" %.4f %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E\n"%(
                            omega,
                            sum(_cxxx/nkpts).real,sum(_cxxx/nkpts).imag,
                            sum(_cxxy/nkpts).real,sum(_cxxy/nkpts).imag,
                            sum(_cyyx/nkpts).real,sum(_cyyx/nkpts).imag,
                            sum(_cyyy/nkpts).real,sum(_cyyy/nkpts).imag,
                            sum(_cxyx/nkpts).real,sum(_cxyx/nkpts).imag,
                            sum(_cxyy/nkpts).real,sum(_cxyy/nkpts).imag,
                            sum(_cyxx/nkpts).real,sum(_cyxx/nkpts).imag,
                            sum(_cyxy/nkpts).real,sum(_cyxy/nkpts).imag))
                for iomega, omega in enumerate(omegaList):
                    with open(mainDir+'injectionCondTensor%d_E%.3f.txt'%(iomega,omega),'w') as fout:
                        for kpt, _cxxx,_cxxy,_cyyx,_cyyy,_cxyx,_cxyy,_cyxx,_cyxy in zip(kmesh,cxxx[iomega],cxxy[iomega],cyyx[iomega],cyyy[iomega],cxyx[iomega],cxyy[iomega],cyxx[iomega],cyxy[iomega]):
                            fout.write(" %.4f %.4f %.4f %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E\n"%(*kpt,_cxxx.real,_cxxx.imag,_cyxx.real,_cyxx.imag,_cxyy.real,_cxyy.imag,_cyyy.real,_cyyy.imag,_cxxy.real,_cxxy.imag,_cyxy.real,_cyxy.imag,_cxyx.real,_cxyx.imag,_cyyx.real,_cyyx.imag))

        condcxxxList=np.array([np.sum(cond_on_omega/nkpts) for cond_on_omega in cxxx])
        condcxxyList=np.array([np.sum(cond_on_omega/nkpts) for cond_on_omega in cxxy])
        condcyyxList=np.array([np.sum(cond_on_omega/nkpts) for cond_on_omega in cyyx])
        condcyyyList=np.array([np.sum(cond_on_omega/nkpts) for cond_on_omega in cyyy])
        condcxyxList=np.array([np.sum(cond_on_omega/nkpts) for cond_on_omega in cxyx])
        condcxyyList=np.array([np.sum(cond_on_omega/nkpts) for cond_on_omega in cxyy])
        condcyxxList=np.array([np.sum(cond_on_omega/nkpts) for cond_on_omega in cyxx])
        condcyxyList=np.array([np.sum(cond_on_omega/nkpts) for cond_on_omega in cyxy])
        print('Injection photovoltaic conductance solved in %.3fs with %d thread(s)'%(time.time()-time0,nthreads))
        #exit()
        fig,ax=plt.subplots(2,4,sharey=True)
        ax[0,0].scatter(omegaList,condcxxxList.real,label=r'$Re\ \sigma^x_{xx}$')
        ax[0,1].scatter(omegaList,condcxxyList.real,label=r'$Re\ \sigma^y_{xx}$')
        ax[0,2].scatter(omegaList,condcyyxList.real,label=r'$Re\ \sigma^x_{yy}$')
        ax[0,3].scatter(omegaList,condcyyyList.real,label=r'$Re\ \sigma^y_{yy}$')
        ax[1,0].scatter(omegaList,condcxyxList.real,label=r'$Re\ \sigma^x_{xy}$')
        ax[1,1].scatter(omegaList,condcxyyList.real,label=r'$Re\ \sigma^y_{xy}$')
        ax[1,2].scatter(omegaList,condcyxxList.real,label=r'$Re\ \sigma^x_{yx}$')
        ax[1,3].scatter(omegaList,condcyxyList.real,label=r'$Re\ \sigma^y_{yx}$')
        for row in ax:
            #row.legend()
            #row.plot([omegaList[0],omegaList[-1]],[0,0],'--',linewidth=1)
            for _ax in row:
                _ax.legend()
                _ax.plot([omegaList[0],omegaList[-1]],[0,0],'--',linewidth=1)
        if mainDir!=None: plt.savefig(mainDir+'shiftconductance.png',dpi=300)
        plt.show()
        return

    def calcJDOS_CDLL(self,hr,mainDir=None,kgrid=[20,20,1],atomicOrbIndexList=[],omegaList=[],tau=0.01,erange=[-1000.,1000.],nthreads=1):
        time0=time.time()
        # prepare input parameters for cdll
        kxaxis=np.linspace(0,1,kgrid[0],endpoint=False)
        kyaxis=np.linspace(0,1,kgrid[1],endpoint=False)
        kzaxis=np.linspace(0,1,kgrid[2],endpoint=False)
        kmesh=np.array([[k1,k2,k3] for k3 in kzaxis for k2 in kyaxis for k1 in kxaxis])

        c_lat=(c_double*9)()
        for i in range(3):
            c_lat[i]=c_double(pR.lat[0][i])
            c_lat[i+3]=c_double(pR.lat[1][i])
            c_lat[i+6]=c_double(pR.lat[2][i])

        assert len(atomicOrbIndexList)==sum(pR.nions), 'pls. provide correctly formatted atomicOrbIndexList[i_atom][i_L]:orb index'
        orbCorrList=[np.zeros(3) for i in range(hr.norb)]
        for atom, pos in zip(atomicOrbIndexList,pR.pos_ions):
            for L in atom:
                for iorb in L: orbCorrList[iorb]=pos
        #assert sum(orbCorrList)==hr.norb, "number of wannier centers: %d inconsistent with %d orbs in hr.dat"%(len(orbCorrList),hr.norb)
        c_orbCorrList=(c_double*(3*hr.norb))()
        for i in range(hr.norb):
            c_orbCorrList[3*i+0]=c_double(orbCorrList[i][0])
            c_orbCorrList[3*i+1]=c_double(orbCorrList[i][1])
            c_orbCorrList[3*i+2]=c_double(orbCorrList[i][2])

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
        
        assert len(omegaList)>=1, 'omegaList is not defined or empty'
        c_omega_list=(c_double*len(omegaList))()
        for iomega, omega in enumerate(omegaList):
            c_omega_list[iomega]=c_double(omega)
        
        c_erange=(c_double*2)()
        c_erange[0]=c_double(erange[0])
        c_erange[1]=c_double(erange[1])
        #exit()
        hamDLL=CDLL("./ham.dll")
        hamDLL.calcJointDOS.restype=py_object

        #tau=0.05
        result=hamDLL.calcJointDOS(c_int(nkpts),c_kmesh,
                        c_int(hr.norb),c_int(nhoppings),
                        c_sourceID,c_targetID,c_overLat,c_hopping_real,c_hopping_imag,
                        c_int(len(omegaList)),
                        c_omega_list,
                        c_double(paramReader.FERMI),
                        c_lat,
                        c_orbCorrList,
                        c_double(tau),
                        c_erange,
                        c_int(nthreads))
                        
        JDOSFull=np.array(result).reshape(len(omegaList),nkpts)
        for iomega, omega in enumerate(omegaList):
            with open(mainDir+'JDOS%d_E%.3f.txt'%(iomega,omega),'w') as fout:
                for kpt, _jdos in zip(kmesh,JDOSFull[iomega]):
                    fout.write(" %.4f %.4f %.4f %13.6E\n"%(*kpt,_jdos))
        print('JDOS solved in %.3fs with %d thread(s)'%(time.time()-time0,nthreads))
        return

    def calcProjectiveBulkPhotovoltaicConductor_CDLL(self,
            hr,mainDir:str=None,kgrid=[20,20,1],
            atomicOrbIndexList=[], # give the correspondence of atom id to wannier id
            proj_op=[[[],[]]], # proj_op[iproj][0,1] give the atom id for source and target projection 
            scatter_route=[[0,0,0]], # scatter_route[iroute][istage=0,1,2] gives the id of projection  
            omegaList=[],tau=0.01,erange=[-1000.,1000.],
            ndivide=1,nthreads=1,show=True):
        time0=time.time()
        assert mainDir!=None,'mainDir is required in projective BPVE calc.'
        # prepare input parameters for cdll
        kxaxis=np.linspace(0,1,kgrid[0],endpoint=False)
        kyaxis=np.linspace(0,1,kgrid[1],endpoint=False)
        kzaxis=np.linspace(0,1,kgrid[2],endpoint=False)
        kmesh=np.array([[k1,k2,k3] for k3 in kzaxis for k2 in kyaxis for k1 in kxaxis])

        print("lattice:")
        print("a1=(%11.6f %11.6f %11.6f)"%(*pR.lat[0],))
        print("a2=(%11.6f %11.6f %11.6f)"%(*pR.lat[1],))
        print("a3=(%11.6f %11.6f %11.6f)"%(*pR.lat[2],))

        c_lat=(c_double*9)()
        for i in range(3):
            c_lat[i]=c_double(pR.lat[0][i])
            c_lat[i+3]=c_double(pR.lat[1][i])
            c_lat[i+6]=c_double(pR.lat[2][i])

        assert len(atomicOrbIndexList)==sum(pR.nions), 'pls. provide correctly formatted atomicOrbIndexList[i_atom][i_L]:orb index'
        orbCorrList=[np.zeros(3) for i in range(hr.norb)]
        for atom, pos in zip(atomicOrbIndexList,pR.pos_ions):
            for L in atom:
                for iorb in L: orbCorrList[iorb]=pos
        #assert sum(orbCorrList)==hr.norb, "number of wannier centers: %d inconsistent with %d orbs in hr.dat"%(len(orbCorrList),hr.norb)
        c_orbCorrList=(c_double*(3*hr.norb))()
        for i in range(hr.norb):
            c_orbCorrList[3*i+0]=c_double(orbCorrList[i][0])
            c_orbCorrList[3*i+1]=c_double(orbCorrList[i][1])
            c_orbCorrList[3*i+2]=c_double(orbCorrList[i][2])

        # construct the orbital weights for projection
        nProj=len(proj_op)
        c_orbProj1WeightList=(c_int*(nProj*hr.norb))()
        c_orbProj2WeightList=(c_int*(nProj*hr.norb))()
        for iproj, proj in enumerate(proj_op):
            orbProj1WeightList=[0]*hr.norb
            for atomID in proj[0]:
                for L in atomicOrbIndexList[atomID]:
                    for iorb in L: orbProj1WeightList[iorb]=1
            orbProj2WeightList=[0]*hr.norb
            for atomID in proj[1]:
                for L in atomicOrbIndexList[atomID]:
                    for iorb in L: orbProj2WeightList[iorb]=1
            for i in range(hr.norb):
                c_orbProj1WeightList[iproj*hr.norb+i]=c_int(orbProj1WeightList[i])
                c_orbProj2WeightList[iproj*hr.norb+i]=c_int(orbProj2WeightList[i])
        # construct scatter routes
        nScatter=len(scatter_route)
        c_scatterRoute=(c_int*(nScatter*3))()
        for iscatter, scatter in enumerate(scatter_route):
            for i in range(3):c_scatterRoute[iscatter*3+i]=scatter[i]

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
        
        assert len(omegaList)>=1, 'omegaList is not defined or empty'
        c_omega_list=(c_double*len(omegaList))()
        for iomega, omega in enumerate(omegaList):
            c_omega_list[iomega]=c_double(omega)
        
        c_erange=(c_double*2)()
        c_erange[0]=c_double(erange[0])
        c_erange[1]=c_double(erange[1])
        hamDLL=CDLL("./ham.dll")
        time0=time.time()

        hamDLL.calcProjectiveBulkPhotovoltaicCond_memoryReduce.restype=py_object
        result=hamDLL.calcProjectiveBulkPhotovoltaicCond_memoryReduce(
                        c_int(nkpts), c_kmesh,
                        c_int(hr.norb), c_int(nhoppings),
                        c_sourceID, c_targetID, c_overLat,
                        c_hopping_real, c_hopping_imag,

                        c_int(len(omegaList)), c_omega_list,
                        c_double(paramReader.FERMI),
                        c_lat,
                        c_orbCorrList,
                        c_double(tau),
                        c_erange,

                        c_int(nProj), c_orbProj1WeightList, c_orbProj2WeightList,
                        c_int(nScatter),c_scatterRoute,

                        c_int(ndivide),
                        c_int(nthreads))
                        
        condFull_scatter=(np.array(result[0])+1j*np.array(result[1])).reshape(nScatter,8,len(omegaList))
        for iscatter in range(nScatter):
            cxxx,cyxx,cxyy,cyyy,cxxy,cyxy,cxyx,cyyx=condFull_scatter[iscatter]
            with open(mainDir+'photoCondTensorVsOmega_proj%d.txt'%iscatter,'w') as fout:
                fout.write("Omega Sxx_x_r Sxx_x_i Sxx_y_r Sxx_y_i Syy_x_r Syy_x_i Syy_y_r Syy_y_i Sxy_x_r Sxy_x_i Sxy_y_r Sxy_y_i Syx_x_r Syx_x_i Syx_y_r Syx_y_i\n")
                for omega, _cxxx,_cxxy,_cyyx,_cyyy,_cxyx,_cxyy,_cyxx,_cyxy in zip(omegaList,cxxx,cxxy,cyyx,cyyy,cxyx,cxyy,cyxx,cyxy):
                    fout.write(" %.4f %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E\n"%(
                        omega,
                        _cxxx.real,_cxxx.imag,
                        _cxxy.real,_cxxy.imag,
                        _cyyx.real,_cyyx.imag,
                        _cyyy.real,_cyyy.imag,
                        _cxyx.real,_cxyx.imag,
                        _cxyy.real,_cxyy.imag,
                        _cyxx.real,_cyxx.imag,
                        _cyxy.real,_cyxy.imag))

        print('Projective Bulk photovoltaic conductance solved in %.3fs with %d thread(s)'%(time.time()-time0,nthreads))
        if not show:
            return
        
        fig,ax=plt.subplots(2,4,sharey=True)
        ax[0,0].scatter(omegaList,cxxx.real,label=r'$Re\ \sigma^x_{xx}$')
        ax[0,1].scatter(omegaList,cxxy.real,label=r'$Re\ \sigma^y_{xx}$')
        ax[0,2].scatter(omegaList,cyyx.real,label=r'$Re\ \sigma^x_{yy}$')
        ax[0,3].scatter(omegaList,cyyy.real,label=r'$Re\ \sigma^y_{yy}$')
        ax[1,0].scatter(omegaList,cxyx.real,label=r'$Re\ \sigma^x_{xy}$')
        ax[1,1].scatter(omegaList,cxyy.real,label=r'$Re\ \sigma^y_{xy}$')
        ax[1,2].scatter(omegaList,cyxx.real,label=r'$Re\ \sigma^x_{yx}$')
        ax[1,3].scatter(omegaList,cyxy.real,label=r'$Re\ \sigma^y_{yx}$')
        for row in ax:
            for _ax in row:
                _ax.legend()
                _ax.plot([omegaList[0],omegaList[-1]],[0,0],'--',linewidth=1)
        plt.savefig(mainDir+'projPhotoconductance.png',dpi=300)
        plt.show()
        return

    def calcModernOrbitalProjectiveBulkPhotovoltaicConductor_CDLL(self,
            hr,mainDir:str=None,kgrid=[20,20,1],
            atomicOrbIndexList=[], # give the correspondence of atom id to wannier id
            proj_op=[[[],[]]], # proj_op[iproj][0,1] give the atom id for source and target projection 
            scatter_route=[[0,0,0]], # scatter_route[iroute][istage=0,1,2] gives the id of projection  
            omegaList=[],tau=0.01,erange=[-1000.,1000.],
            ndivide=1,nthreads=1,show=True):
        time0=time.time()
        assert mainDir!=None,'mainDir is required in projective BPVE calc.'
        # prepare input parameters for cdll
        kxaxis=np.linspace(0,1,kgrid[0],endpoint=False)
        kyaxis=np.linspace(0,1,kgrid[1],endpoint=False)
        kzaxis=np.linspace(0,1,kgrid[2],endpoint=False)
        kmesh=np.array([[k1,k2,k3] for k3 in kzaxis for k2 in kyaxis for k1 in kxaxis])

        print("lattice:")
        print("a1=(%11.6f %11.6f %11.6f)"%(*pR.lat[0],))
        print("a2=(%11.6f %11.6f %11.6f)"%(*pR.lat[1],))
        print("a3=(%11.6f %11.6f %11.6f)"%(*pR.lat[2],))

        c_lat=(c_double*9)()
        for i in range(3):
            c_lat[i]=c_double(pR.lat[0][i])
            c_lat[i+3]=c_double(pR.lat[1][i])
            c_lat[i+6]=c_double(pR.lat[2][i])

        assert len(atomicOrbIndexList)==sum(pR.nions), 'pls. provide correctly formatted atomicOrbIndexList[i_atom][i_L]:orb index'
        orbCorrList=[np.zeros(3) for i in range(hr.norb)]
        for atom, pos in zip(atomicOrbIndexList,pR.pos_ions):
            for L in atom:
                for iorb in L: orbCorrList[iorb]=pos
        #assert sum(orbCorrList)==hr.norb, "number of wannier centers: %d inconsistent with %d orbs in hr.dat"%(len(orbCorrList),hr.norb)
        c_orbCorrList=(c_double*(3*hr.norb))()
        for i in range(hr.norb):
            c_orbCorrList[3*i+0]=c_double(orbCorrList[i][0])
            c_orbCorrList[3*i+1]=c_double(orbCorrList[i][1])
            c_orbCorrList[3*i+2]=c_double(orbCorrList[i][2])

        # construct the orbital weights for projection
        nProj=len(proj_op)
        c_orbProj1WeightList=(c_int*(nProj*hr.norb))()
        c_orbProj2WeightList=(c_int*(nProj*hr.norb))()
        for iproj, proj in enumerate(proj_op):
            orbProj1WeightList=[0]*hr.norb
            for atomID in proj[0]:
                for L in atomicOrbIndexList[atomID]:
                    for iorb in L: orbProj1WeightList[iorb]=1
            orbProj2WeightList=[0]*hr.norb
            for atomID in proj[1]:
                for L in atomicOrbIndexList[atomID]:
                    for iorb in L: orbProj2WeightList[iorb]=1
            for i in range(hr.norb):
                c_orbProj1WeightList[iproj*hr.norb+i]=c_int(orbProj1WeightList[i])
                c_orbProj2WeightList[iproj*hr.norb+i]=c_int(orbProj2WeightList[i])
        # construct scatter routes
        nScatter=len(scatter_route)
        c_scatterRoute=(c_int*(nScatter*3))()
        for iscatter, scatter in enumerate(scatter_route):
            for i in range(3):c_scatterRoute[iscatter*3+i]=scatter[i]

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
        
        assert len(omegaList)>=1, 'omegaList is not defined or empty'
        c_omega_list=(c_double*len(omegaList))()
        for iomega, omega in enumerate(omegaList):
            c_omega_list[iomega]=c_double(omega)
        
        c_erange=(c_double*2)()
        c_erange[0]=c_double(erange[0])
        c_erange[1]=c_double(erange[1])
        #exit()
        hamDLL=CDLL("./ham.dll")
        time0=time.time()

        hamDLL.calcModernOrbitalProjectiveBulkPhotovoltaicCond_memoryReduce.restype=py_object
        result=hamDLL.calcModernOrbitalProjectiveBulkPhotovoltaicCond_memoryReduce(
                        c_int(nkpts), c_kmesh,
                        c_int(hr.norb), c_int(nhoppings),
                        c_sourceID, c_targetID, c_overLat,
                        c_hopping_real, c_hopping_imag,

                        c_int(len(omegaList)), c_omega_list,
                        c_double(paramReader.FERMI),
                        c_lat,
                        c_orbCorrList,
                        c_double(tau),
                        c_erange,

                        c_int(nProj), c_orbProj1WeightList, c_orbProj2WeightList,
                        c_int(nScatter),c_scatterRoute,

                        c_int(ndivide),
                        c_int(nthreads))
                        
        condFull_scatter_lc,condFull_scatter_ic=(np.array(result[0])+1j*np.array(result[1])).reshape(2,nScatter,8,len(omegaList))
        for condFull_scatter, lname in zip([condFull_scatter_lc,condFull_scatter_ic],['LC','IC']):
            for iscatter in range(nScatter):
                cxxx,cyxx,cxyy,cyyy,cxxy,cyxy,cxyx,cyyx=condFull_scatter[iscatter]
                with open(mainDir+'photoCondTensorVsOmega_proj%d_%s_Lz.txt'%(iscatter,lname),'w') as fout:
                    fout.write("Omega Sxx_x_r Sxx_x_i Sxx_y_r Sxx_y_i Syy_x_r Syy_x_i Syy_y_r Syy_y_i Sxy_x_r Sxy_x_i Sxy_y_r Sxy_y_i Syx_x_r Syx_x_i Syx_y_r Syx_y_i\n")
                    for omega, _cxxx,_cxxy,_cyyx,_cyyy,_cxyx,_cxyy,_cyxx,_cyxy in zip(omegaList,cxxx,cxxy,cyyx,cyyy,cxyx,cxyy,cyxx,cyxy):
                        fout.write(" %.4f %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E\n"%(
                            omega,
                            _cxxx.real,_cxxx.imag,
                            _cxxy.real,_cxxy.imag,
                            _cyyx.real,_cyyx.imag,
                            _cyyy.real,_cyyy.imag,
                            _cxyx.real,_cxyx.imag,
                            _cxyy.real,_cxyy.imag,
                            _cyxx.real,_cyxx.imag,
                            _cyxy.real,_cyxy.imag))

        print('Projective Modern orbital bulk photovoltaic conductance solved in %.3fs with %d thread(s)'%(time.time()-time0,nthreads))
        if not show:
            return
        
        _,ax=plt.subplots(2,4,sharey=True)
        ax[0,0].plot(omegaList,condFull_scatter_lc[0,0,:].real,label=r'$\sigma^{x:xx}_{LC}$')
        ax[0,1].plot(omegaList,condFull_scatter_lc[0,1,:].real,label=r'$\sigma^{y:xx}_{LC}$')
        ax[0,2].plot(omegaList,condFull_scatter_lc[0,2,:].real,label=r'$\sigma^{x:yy}_{LC}$')
        ax[0,3].plot(omegaList,condFull_scatter_lc[0,3,:].real,label=r'$\sigma^{y:yy}_{LC}$')
        ax[1,0].plot(omegaList,condFull_scatter_lc[0,4,:].real,label=r'$\sigma^{x:xy}_{LC}$')
        ax[1,1].plot(omegaList,condFull_scatter_lc[0,5,:].real,label=r'$\sigma^{y:xy}_{LC}$')
        ax[1,2].plot(omegaList,condFull_scatter_lc[0,6,:].real,label=r'$\sigma^{x:yx}_{LC}$')
        ax[1,3].plot(omegaList,condFull_scatter_lc[0,7,:].real,label=r'$\sigma^{y:yx}_{LC}$')
        ax[0,0].plot(omegaList,condFull_scatter_ic[0,0,:].real,label=r'$\sigma^{x:xx}_{IC}$')
        ax[0,1].plot(omegaList,condFull_scatter_ic[0,1,:].real,label=r'$\sigma^{y:xx}_{IC}$')
        ax[0,2].plot(omegaList,condFull_scatter_ic[0,2,:].real,label=r'$\sigma^{x:yy}_{IC}$')
        ax[0,3].plot(omegaList,condFull_scatter_ic[0,3,:].real,label=r'$\sigma^{y:yy}_{IC}$')
        ax[1,0].plot(omegaList,condFull_scatter_ic[0,4,:].real,label=r'$\sigma^{x:xy}_{IC}$')
        ax[1,1].plot(omegaList,condFull_scatter_ic[0,5,:].real,label=r'$\sigma^{y:xy}_{IC}$')
        ax[1,2].plot(omegaList,condFull_scatter_ic[0,6,:].real,label=r'$\sigma^{x:yx}_{IC}$')
        ax[1,3].plot(omegaList,condFull_scatter_ic[0,7,:].real,label=r'$\sigma^{y:yx}_{IC}$')
        for row in ax:
            for _ax in row:
                _ax.legend()
                _ax.plot([omegaList[0],omegaList[-1]],[0,0],'--',linewidth=1)
        plt.savefig(mainDir+'projPhotoconductance_Lz.png',dpi=300)
        plt.show()
        return

    def calcSpinProjectiveBulkPhotovoltaicConductor_CDLL(self,
            hr,mainDir:str=None,kgrid=[20,20,1],
            atomicOrbIndexList=[], # give the correspondence of atom id to wannier id
            proj_op=[[[],[]]], # proj_op[iproj][0,1] give the atom id for source and target projection 
            scatter_route=[[0,0,0]], # scatter_route[iroute][istage=0,1,2] gives the id of projection  
            omegaList=[],tau=0.01,erange=[-1000.,1000.],
            ndivide=1,nthreads=1,show=True):
        time0=time.time()
        assert mainDir!=None,'mainDir is required in projective BPVE calc.'
        # prepare input parameters for cdll
        kxaxis=np.linspace(0,1,kgrid[0],endpoint=False)
        kyaxis=np.linspace(0,1,kgrid[1],endpoint=False)
        kzaxis=np.linspace(0,1,kgrid[2],endpoint=False)
        kmesh=np.array([[k1,k2,k3] for k3 in kzaxis for k2 in kyaxis for k1 in kxaxis])

        print("lattice:")
        print("a1=(%11.6f %11.6f %11.6f)"%(*pR.lat[0],))
        print("a2=(%11.6f %11.6f %11.6f)"%(*pR.lat[1],))
        print("a3=(%11.6f %11.6f %11.6f)"%(*pR.lat[2],))

        c_lat=(c_double*9)()
        for i in range(3):
            c_lat[i]=c_double(pR.lat[0][i])
            c_lat[i+3]=c_double(pR.lat[1][i])
            c_lat[i+6]=c_double(pR.lat[2][i])

        assert len(atomicOrbIndexList)==sum(pR.nions), 'pls. provide correctly formatted atomicOrbIndexList[i_atom][i_L]:orb index'
        orbCorrList=[np.zeros(3) for i in range(hr.norb)]
        for atom, pos in zip(atomicOrbIndexList,pR.pos_ions):
            for L in atom:
                for iorb in L: orbCorrList[iorb]=pos
        #assert sum(orbCorrList)==hr.norb, "number of wannier centers: %d inconsistent with %d orbs in hr.dat"%(len(orbCorrList),hr.norb)
        c_orbCorrList=(c_double*(3*hr.norb))()
        for i in range(hr.norb):
            c_orbCorrList[3*i+0]=c_double(orbCorrList[i][0])
            c_orbCorrList[3*i+1]=c_double(orbCorrList[i][1])
            c_orbCorrList[3*i+2]=c_double(orbCorrList[i][2])

        # construct the orbital weights for projection
        nProj=len(proj_op)
        c_orbProj1WeightList=(c_int*(nProj*hr.norb))()
        c_orbProj2WeightList=(c_int*(nProj*hr.norb))()
        for iproj, proj in enumerate(proj_op):
            orbProj1WeightList=[0]*hr.norb
            for atomID in proj[0]:
                for L in atomicOrbIndexList[atomID]:
                    for iorb in L: orbProj1WeightList[iorb]=1
            orbProj2WeightList=[0]*hr.norb
            for atomID in proj[1]:
                for L in atomicOrbIndexList[atomID]:
                    for iorb in L: orbProj2WeightList[iorb]=1
            for i in range(hr.norb):
                c_orbProj1WeightList[iproj*hr.norb+i]=c_int(orbProj1WeightList[i])
                c_orbProj2WeightList[iproj*hr.norb+i]=c_int(orbProj2WeightList[i])
        # construct scatter routes
        nScatter=len(scatter_route)
        c_scatterRoute=(c_int*(nScatter*3))()
        for iscatter, scatter in enumerate(scatter_route):
            for i in range(3):c_scatterRoute[iscatter*3+i]=scatter[i]

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
        
        assert len(omegaList)>=1, 'omegaList is not defined or empty'
        c_omega_list=(c_double*len(omegaList))()
        for iomega, omega in enumerate(omegaList):
            c_omega_list[iomega]=c_double(omega)
        
        c_erange=(c_double*2)()
        c_erange[0]=c_double(erange[0])
        c_erange[1]=c_double(erange[1])
        #exit()
        hamDLL=CDLL("./ham.dll")
        time0=time.time()

        hamDLL.calcSpinProjectiveBulkPhotovoltaicCond_memoryReduce.restype=py_object
        result=hamDLL.calcSpinProjectiveBulkPhotovoltaicCond_memoryReduce(
                        c_int(nkpts), c_kmesh,
                        c_int(hr.norb), c_int(nhoppings),
                        c_sourceID, c_targetID, c_overLat,
                        c_hopping_real, c_hopping_imag,

                        c_int(len(omegaList)), c_omega_list,
                        c_double(paramReader.FERMI),
                        c_lat,
                        c_orbCorrList,
                        c_double(tau),
                        c_erange,

                        c_int(nProj), c_orbProj1WeightList, c_orbProj2WeightList,
                        c_int(nScatter),c_scatterRoute,

                        c_int(ndivide),
                        c_int(nthreads))
                        
        condFull_scatter_sx,condFull_scatter_sy,condFull_scatter_sz=(np.array(result[0])+1j*np.array(result[1])).reshape(3,nScatter,8,len(omegaList))
        for condFull_scatter, sname in zip([condFull_scatter_sx,condFull_scatter_sy,condFull_scatter_sz],['sx','sy','sz']):
            for iscatter in range(nScatter):
                cxxx,cyxx,cxyy,cyyy,cxxy,cyxy,cxyx,cyyx=condFull_scatter[iscatter]
                with open(mainDir+'photoCondTensorVsOmega_proj%d_%s.txt'%(iscatter,sname),'w') as fout:
                    fout.write("Omega Sxx_x_r Sxx_x_i Sxx_y_r Sxx_y_i Syy_x_r Syy_x_i Syy_y_r Syy_y_i Sxy_x_r Sxy_x_i Sxy_y_r Sxy_y_i Syx_x_r Syx_x_i Syx_y_r Syx_y_i\n")
                    for omega, _cxxx,_cxxy,_cyyx,_cyyy,_cxyx,_cxyy,_cyxx,_cyxy in zip(omegaList,cxxx,cxxy,cyyx,cyyy,cxyx,cxyy,cyxx,cyxy):
                        fout.write(" %.4f %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E\n"%(
                            omega,
                            _cxxx.real,_cxxx.imag,
                            _cxxy.real,_cxxy.imag,
                            _cyyx.real,_cyyx.imag,
                            _cyyy.real,_cyyy.imag,
                            _cxyx.real,_cxyx.imag,
                            _cxyy.real,_cxyy.imag,
                            _cyxx.real,_cyxx.imag,
                            _cyxy.real,_cyxy.imag))

            
            if not show: continue
            #plt.figure()
            fig,ax=plt.subplots(2,4,sharey=True)
            ax[0,0].scatter(omegaList,cxxx.real,label=r'$Re\ \sigma^x_{xx}$')
            ax[0,1].scatter(omegaList,cxxy.real,label=r'$Re\ \sigma^y_{xx}$')
            ax[0,2].scatter(omegaList,cyyx.real,label=r'$Re\ \sigma^x_{yy}$')
            ax[0,3].scatter(omegaList,cyyy.real,label=r'$Re\ \sigma^y_{yy}$')
            ax[1,0].scatter(omegaList,cxyx.real,label=r'$Re\ \sigma^x_{xy}$')
            ax[1,1].scatter(omegaList,cxyy.real,label=r'$Re\ \sigma^y_{xy}$')
            ax[1,2].scatter(omegaList,cyxx.real,label=r'$Re\ \sigma^x_{yx}$')
            ax[1,3].scatter(omegaList,cyxy.real,label=r'$Re\ \sigma^y_{yx}$')
            for row in ax:
                for _ax in row:
                    _ax.legend()
                    _ax.plot([omegaList[0],omegaList[-1]],[0,0],'--',linewidth=1)
            plt.savefig(mainDir+'spin_%s_photoconductance.png'%sname,dpi=300)
            plt.show()
        print('Spin projective Bulk photovoltaic conductance solved in %.3fs with %d thread(s)'%(time.time()-time0,nthreads))
        return

    def calcOrbitalProjectiveBulkPhotovoltaicConductor_CDLL(self,
            hr,mainDir:str=None,kgrid=[20,20,1],
            atomicOrbIndexList=[], # give the correspondence of atom id to wannier id
            proj_op=[[[],[]]], # proj_op[iproj][0,1] give the atom id for source and target projection 
            scatter_route=[[0,0,0]], # scatter_route[iroute][istage=0,1,2] gives the id of projection  
            omegaList=[],tau=0.01,erange=[-1000.,1000.],
            ndivide=1,nthreads=1,show=True):
        time0=time.time()
        assert mainDir!=None,'mainDir is required in projective BPVE calc.'
        # prepare input parameters for cdll
        kxaxis=np.linspace(0,1,kgrid[0],endpoint=False)
        kyaxis=np.linspace(0,1,kgrid[1],endpoint=False)
        kzaxis=np.linspace(0,1,kgrid[2],endpoint=False)
        kmesh=np.array([[k1,k2,k3] for k3 in kzaxis for k2 in kyaxis for k1 in kxaxis])

        print("lattice:")
        print("a1=(%11.6f %11.6f %11.6f)"%(*pR.lat[0],))
        print("a2=(%11.6f %11.6f %11.6f)"%(*pR.lat[1],))
        print("a3=(%11.6f %11.6f %11.6f)"%(*pR.lat[2],))

        c_lat=(c_double*9)()
        for i in range(3):
            c_lat[i]=c_double(pR.lat[0][i])
            c_lat[i+3]=c_double(pR.lat[1][i])
            c_lat[i+6]=c_double(pR.lat[2][i])

        assert len(atomicOrbIndexList)==sum(pR.nions), 'pls. provide correctly formatted atomicOrbIndexList[i_atom][i_L]:orb index'
        orbCorrList=[np.zeros(3) for i in range(hr.norb)]
        for atom, pos in zip(atomicOrbIndexList,pR.pos_ions):
            for L in atom:
                for iorb in L: orbCorrList[iorb]=pos
        #assert sum(orbCorrList)==hr.norb, "number of wannier centers: %d inconsistent with %d orbs in hr.dat"%(len(orbCorrList),hr.norb)
        c_orbCorrList=(c_double*(3*hr.norb))()
        for i in range(hr.norb):
            c_orbCorrList[3*i+0]=c_double(orbCorrList[i][0])
            c_orbCorrList[3*i+1]=c_double(orbCorrList[i][1])
            c_orbCorrList[3*i+2]=c_double(orbCorrList[i][2])

        # canonical transformation from tb orbitals to Lz orbitals
        tTBLz=np.zeros((hr.norb,hr.norb),dtype=complex)
        Lz=np.zeros(hr.norb,dtype=int)
        #cnt=0
        for atom in atomicOrbIndexList:
            for iL, L in enumerate(atom):
                if iL==0: # s orbitals
                    for iorb in L:
                        tTBLz[iorb,iorb]=1
                        Lz[iorb]=0
                elif iL==1: # pz, px, py
                    for iorb in range(0,len(L),3):
                        ipz, ipx, ipy= L[iorb], L[iorb+1], L[iorb+2]
                        tTBLz[ipz,ipz]=1 # Lz=0
                        tTBLz[ipx,ipx],tTBLz[ipx,ipy]=1/np.sqrt(2), 1j/np.sqrt(2) # Lz=+1
                        tTBLz[ipy,ipx],tTBLz[ipy,ipy]=1/np.sqrt(2),-1j/np.sqrt(2) # Lz=-1
                        Lz[ipz],Lz[ipx],Lz[ipy]=0,1,-1
                elif iL==2: # dz2, dxz, dyz, dx2-y2, dxy
                    for iorb in range(0,len(L),5):
                        idz2, idxz, idyz, idx2, idxy= L[iorb], L[iorb+1], L[iorb+2], L[iorb+3], L[iorb+4]
                        tTBLz[idz2,idz2]=1 # Lz=0
                        tTBLz[idxz,idxz],tTBLz[idxz,idyz]=1/np.sqrt(2), 1j/np.sqrt(2) # Lz=+1
                        tTBLz[idyz,idxz],tTBLz[idyz,idyz]=1/np.sqrt(2),-1j/np.sqrt(2) # Lz=-1
                        tTBLz[idx2,idx2],tTBLz[idx2,idxy]=1/np.sqrt(2), 1j/np.sqrt(2) # Lz=+2
                        tTBLz[idxy,idx2],tTBLz[idxy,idxy]=1/np.sqrt(2),-1j/np.sqrt(2) # Lz=-2
                        Lz[idz2],Lz[idxz],Lz[idyz],Lz[idx2],Lz[idxy]=0,1,-1,2,-2
                elif iL==3: # fz3, fxz2, fyz2, fz(x2-y2), fxyz, fx(x2-3y2), fy(3x2-y2)
                    for iorb in range(0,len(L),7):
                        ifz3, ifxz2, ifyz2, ifzx2, ifzxy, ifx3, ify3= L[iorb], L[iorb+1], L[iorb+2], L[iorb+3], L[iorb+4], L[iorb+5], L[iorb+6]
                        tTBLz[ifz3,ifz3]=1 # Lz=0
                        tTBLz[ifxz2,ifxz2],tTBLz[ifxz2,ifyz2]=1/np.sqrt(2), 1j/np.sqrt(2) # Lz=+1
                        tTBLz[ifyz2,ifxz2],tTBLz[ifyz2,ifyz2]=1/np.sqrt(2),-1j/np.sqrt(2) # Lz=-1
                        tTBLz[ifzx2,ifzx2],tTBLz[ifzx2,ifzxy]=1/np.sqrt(2), 1j/np.sqrt(2) # Lz=+2
                        tTBLz[ifzxy,ifzx2],tTBLz[ifzxy,ifzxy]=1/np.sqrt(2),-1j/np.sqrt(2) # Lz=-2
                        tTBLz[ ifx3, ifx3],tTBLz[ ifx3, ify3]=1/np.sqrt(2), 1j/np.sqrt(2) # Lz=+3
                        tTBLz[ ify3, ifx3],tTBLz[ ify3, ify3]=1/np.sqrt(2),-1j/np.sqrt(2) # Lz=-3
                        Lz[ifz3], Lz[ifxz2], Lz[ifyz2], Lz[ifzx2], Lz[ifzxy], Lz[ifx3], Lz[ify3]=0,1,-1,2,-2,3,-3
        #print(Lz)
        #print(tTBLz)
        #print(tTBLz.dot(tTBLz.T.conjugate()))
        #exit()
        c_tTBLz_real=(c_double*(hr.norb*hr.norb))()
        c_tTBLz_imag=(c_double*(hr.norb*hr.norb))()
        c_Lz=(c_int*(hr.norb*hr.norb))()
        for i in range(hr.norb):
            for j in range(hr.norb):
                c_tTBLz_real[i*hr.norb+j]=c_double(tTBLz.real[i][j])
                c_tTBLz_imag[i*hr.norb+j]=c_double(tTBLz.imag[i][j])
                c_Lz[i*hr.norb+j]=c_int(0)
                if i==j:
                    c_Lz[i*hr.norb+j]=c_int(Lz[i])#c_int(1)
                    #c_tTBLz_real[i*hr.norb+j]=c_double(1.)
        #exit()
        # construct the orbital weights for projection
        nProj=len(proj_op)
        c_orbProj1WeightList=(c_int*(nProj*hr.norb))()
        c_orbProj2WeightList=(c_int*(nProj*hr.norb))()
        for iproj, proj in enumerate(proj_op):
            orbProj1WeightList=[0]*hr.norb
            for atomID in proj[0]:
                for L in atomicOrbIndexList[atomID]:
                    for iorb in L: orbProj1WeightList[iorb]=1
            orbProj2WeightList=[0]*hr.norb
            for atomID in proj[1]:
                for L in atomicOrbIndexList[atomID]:
                    for iorb in L: orbProj2WeightList[iorb]=1
            for i in range(hr.norb):
                c_orbProj1WeightList[iproj*hr.norb+i]=c_int(orbProj1WeightList[i])
                c_orbProj2WeightList[iproj*hr.norb+i]=c_int(orbProj2WeightList[i])
        # construct scatter routes
        nScatter=len(scatter_route)
        c_scatterRoute=(c_int*(nScatter*3))()
        for iscatter, scatter in enumerate(scatter_route):
            for i in range(3):c_scatterRoute[iscatter*3+i]=scatter[i]

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
        
        assert len(omegaList)>=1, 'omegaList is not defined or empty'
        c_omega_list=(c_double*len(omegaList))()
        for iomega, omega in enumerate(omegaList):
            c_omega_list[iomega]=c_double(omega)
        
        c_erange=(c_double*2)()
        c_erange[0]=c_double(erange[0])
        c_erange[1]=c_double(erange[1])
        #exit()
        hamDLL=CDLL("./ham.dll")
        time0=time.time()

        hamDLL.calcOrbitalProjectiveBulkPhotovoltaicCond_memoryReduce.restype=py_object
        result=hamDLL.calcOrbitalProjectiveBulkPhotovoltaicCond_memoryReduce(
                        c_int(nkpts), c_kmesh,
                        c_int(hr.norb), c_int(nhoppings),
                        c_sourceID, c_targetID, c_overLat,
                        c_hopping_real, c_hopping_imag,

                        c_int(len(omegaList)), c_omega_list,
                        c_double(paramReader.FERMI),
                        c_lat,
                        c_orbCorrList,
                        c_double(tau),
                        c_erange,

                        c_int(nProj), c_orbProj1WeightList, c_orbProj2WeightList,
                        c_int(nScatter),c_scatterRoute,

                        c_tTBLz_real, c_tTBLz_imag, c_Lz,

                        c_int(ndivide),
                        c_int(nthreads))
                        
        condFull_scatter_Lz=(np.array(result[0])+1j*np.array(result[1])).reshape(nScatter,8,len(omegaList))
        for iscatter in range(nScatter):
            cxxx,cyxx,cxyy,cyyy,cxxy,cyxy,cxyx,cyyx=condFull_scatter_Lz[iscatter]
            with open(mainDir+'photoCondTensorVsOmega_proj%d_Lz.txt'%iscatter,'w') as fout:
                fout.write("Omega Sxx_x_r Sxx_x_i Sxx_y_r Sxx_y_i Syy_x_r Syy_x_i Syy_y_r Syy_y_i Sxy_x_r Sxy_x_i Sxy_y_r Sxy_y_i Syx_x_r Syx_x_i Syx_y_r Syx_y_i\n")
                for omega, _cxxx,_cxxy,_cyyx,_cyyy,_cxyx,_cxyy,_cyxx,_cyxy in zip(omegaList,cxxx,cxxy,cyyx,cyyy,cxyx,cxyy,cyxx,cyxy):
                    fout.write(" %.4f %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E\n"%(
                        omega,
                        _cxxx.real,_cxxx.imag,
                        _cxxy.real,_cxxy.imag,
                        _cyyx.real,_cyyx.imag,
                        _cyyy.real,_cyyy.imag,
                        _cxyx.real,_cxyx.imag,
                        _cxyy.real,_cxyy.imag,
                        _cyxx.real,_cyxx.imag,
                        _cyxy.real,_cyxy.imag))

            
        if not show: return
        #plt.figure()
        fig,ax=plt.subplots(2,4,sharey=True)
        ax[0,0].scatter(omegaList,cxxx.real,label=r'$Re\ \sigma^x_{xx}$')
        ax[0,1].scatter(omegaList,cxxy.real,label=r'$Re\ \sigma^y_{xx}$')
        ax[0,2].scatter(omegaList,cyyx.real,label=r'$Re\ \sigma^x_{yy}$')
        ax[0,3].scatter(omegaList,cyyy.real,label=r'$Re\ \sigma^y_{yy}$')
        ax[1,0].scatter(omegaList,cxyx.real,label=r'$Re\ \sigma^x_{xy}$')
        ax[1,1].scatter(omegaList,cxyy.real,label=r'$Re\ \sigma^y_{xy}$')
        ax[1,2].scatter(omegaList,cyxx.real,label=r'$Re\ \sigma^x_{yx}$')
        ax[1,3].scatter(omegaList,cyxy.real,label=r'$Re\ \sigma^y_{yx}$')
        for row in ax:
            for _ax in row:
                _ax.legend()
                _ax.plot([omegaList[0],omegaList[-1]],[0,0],'--',linewidth=1)
        plt.savefig(mainDir+'orbital_Lz_photoconductance.png',dpi=300)
        plt.show()
        print('Orbital projective Bulk photovoltaic conductance solved in %.3fs with %d thread(s)'%(time.time()-time0,nthreads))
        return

    def calcOpticalHoppingTensor_CDLL(self,hr,mainDir=None,fermi=0,nk=100,atomicOrbIndexList=[],ibandv=0,ibandc=1,nthreads=1):
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

        c_lat=(c_double*9)()
        for i in range(3):
            c_lat[i]=c_double(pR.lat[0][i])
            c_lat[i+3]=c_double(pR.lat[1][i])
            c_lat[i+6]=c_double(pR.lat[2][i])

        assert len(atomicOrbIndexList)==sum(pR.nions), 'pls. provide correctly formatted atomicOrbIndexList[i_atom][i_L]:orb index'
        orbCorrList=[np.zeros(3) for i in range(hr.norb)]
        for atom, pos in zip(atomicOrbIndexList,pR.pos_ions):
            for L in atom:
                for iorb in L: orbCorrList[iorb]=pos
        #assert len(orbCorrList)==hr.norb, "number of wannier centers: %d inconsistent with %d orbs in hr.dat"%(len(orbCorrList),hr.norb)
        c_orbCorrList=(c_double*(3*hr.norb))()
        for i in range(hr.norb):
            c_orbCorrList[3*i+0]=c_double(orbCorrList[i][0])
            c_orbCorrList[3*i+1]=c_double(orbCorrList[i][1])
            c_orbCorrList[3*i+2]=c_double(orbCorrList[i][2])
        
        hamDLL=CDLL("./ham.dll")
        hamDLL.calcBandsVecsVelocityOps.restype=py_object
        result=hamDLL.calcBandsVecsVelocityOps(c_int(nkpts),c_kmesh,
                        c_int(hr.norb),c_int(nhoppings),
                        c_sourceID,c_targetID,c_overLat,c_hopping_real,c_hopping_imag,
                        c_lat,c_orbCorrList,
                        c_int(nthreads))
        #print(result.shape)
        _eigList,_vec_r,_vec_i,Hk_r,Hk_i,Hk_dAx_r,Hk_dAx_i,Hk_dAy_r,Hk_dAy_i,Hk_dAz_r,Hk_dAz_i=result
        eigList=np.array(_eigList).reshape(nkpts,hr.norb)
        vec=(np.array(_vec_r)+1j*np.array(_vec_i)).reshape(nkpts,hr.norb,hr.norb)
        Hk=(np.array(Hk_r)+1j*np.array(Hk_i)).reshape(nkpts,hr.norb,hr.norb)
        DHk_dAx=(np.array(Hk_dAx_r)+1j*np.array(Hk_dAx_i)).reshape(nkpts,hr.norb,hr.norb)
        DHk_dAy=(np.array(Hk_dAy_r)+1j*np.array(Hk_dAy_i)).reshape(nkpts,hr.norb,hr.norb)
        DHk_dAz=(np.array(Hk_dAz_r)+1j*np.array(Hk_dAz_i)).reshape(nkpts,hr.norb,hr.norb)

        print('Spectra solved in %.3fs (serial mode)'%(time.time()-time0))
        
        # transform DHk_dk to xyz coordinates
        print("lattice:")
        print("a1=(%9.6f %9.6f %9.6f)"%(*pR.lat[0],))
        print("a2=(%9.6f %9.6f %9.6f)"%(*pR.lat[1],))
        print("a3=(%9.6f %9.6f %9.6f)"%(*pR.lat[2],))
        reciLat=np.linalg.inv(pR.lat)
        print("reciprocal lattice:")
        print("g1=(%9.6f %9.6f %9.6f)"%(*reciLat[:,0],))
        print("g2=(%9.6f %9.6f %9.6f)"%(*reciLat[:,1],))
        print("g3=(%9.6f %9.6f %9.6f)"%(*reciLat[:,2],))
        print("unit of velocity matrix: (h_bar^-1 eV A.)")

        time0=time.time()
        m,n=ibandv,ibandc
        filePath=mainDir+'opticalOverlap.txt' if mainDir else './opticalOverlap.txt'
        vx,vy,vz=[],[],[]
        with open(filePath,'w') as fout:
            for ikpt, kpt in enumerate(kmesh):
                Em, En=eigList[ikpt][m], eigList[ikpt][n]

                vmn_x=vec[ikpt][:,m].conjugate().dot(DHk_dAx[ikpt]).dot(vec[ikpt][:,n])
                vmn_y=vec[ikpt][:,m].conjugate().dot(DHk_dAy[ikpt]).dot(vec[ikpt][:,n])
                vmn_z=vec[ikpt][:,m].conjugate().dot(DHk_dAz[ikpt]).dot(vec[ikpt][:,n])

                vmm_x=vec[ikpt][:,m].conjugate().dot(DHk_dAx[ikpt]).dot(vec[ikpt][:,m])
                vmm_y=vec[ikpt][:,m].conjugate().dot(DHk_dAy[ikpt]).dot(vec[ikpt][:,m])
                vmm_z=vec[ikpt][:,m].conjugate().dot(DHk_dAz[ikpt]).dot(vec[ikpt][:,m])

                vnn_x=vec[ikpt][:,n].conjugate().dot(DHk_dAx[ikpt]).dot(vec[ikpt][:,n])
                vnn_y=vec[ikpt][:,n].conjugate().dot(DHk_dAy[ikpt]).dot(vec[ikpt][:,n])
                vnn_z=vec[ikpt][:,n].conjugate().dot(DHk_dAz[ikpt]).dot(vec[ikpt][:,n])

                fout.write("%.3f %.3f %.3f %.6f %.3f %.6f %.3f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n"%(
                    *kpt,Em,1.,En,0.,
                    vmn_x.real,vmn_x.imag,vmn_y.real,vmn_y.imag,vmn_z.real,vmn_z.imag,
                    vmm_x.real,vmm_x.imag,vmm_y.real,vmm_y.imag,vmm_z.real,vmm_z.imag,
                    vnn_x.real,vnn_x.imag,vnn_y.real,vnn_y.imag,vnn_z.real,vnn_z.imag,))
                vx.append(vmn_x);vy.append(vmn_y);vz.append(vmn_z)
                
        print('overlap solved in %.3fs, data saved in %s'%(time.time()-time0,filePath))
        return np.array(vx), np.array(vy), np.array(vz)

    def calcTransverseFieldResponse_CDLL(self,hr,mainDir=None,fermi=0,Lx=10,Ly=10,Lz=1,omegaAxis=[0],eta=0.001,nthreads=1):
        halfOrb=hr.norb//2
        assert halfOrb*2==hr.norb, 'num. orbs is odd, magnetic field response cannot be implemented'
        time0=time.time()
        # prepare input parameters for cdll
        #kaxis=np.linspace(0,1,nk,endpoint=False)
        kmesh=np.array([[k1,k2,k3] for k3 in np.linspace(0,1,Lz,endpoint=False) for k2 in np.linspace(0,1,Ly,endpoint=False) for k1 in np.linspace(0,1,Lx,endpoint=False)])
        nkpts=len(kmesh)
        c_kmesh=(c_double*(nkpts*3))()
        for ikpt, kpt in enumerate(kmesh):
            k1,k2,k3=kpt
            c_kmesh[ikpt*3+0],c_kmesh[ikpt*3+1],c_kmesh[ikpt*3+2]=c_double(k1),c_double(k2),c_double(k3)
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
        
        hamDLL=CDLL("./ham.dll")
        hamDLL.calcFullBandsAndVecs.restype=py_object
        result=hamDLL.calcFullBandsAndVecs(c_int(nkpts),c_kmesh,
                        c_int(hr.norb),c_int(nhoppings),
                        c_sourceID,c_targetID,c_overLat,c_hopping_real,c_hopping_imag,
                        c_int(nthreads))
        #print(result.shape)
        _eigList,_vec_r,_vec_i=result
        
        eigList=np.array(_eigList).reshape(Lx,Ly,Lz,hr.norb)
        vec=(np.array(_vec_r)+1j*np.array(_vec_i)).reshape(Lx,Ly,Lz,hr.norb,hr.norb)

        bandmin, bandmax=np.min(eigList), np.max(eigList)
        print('Spectra solved in %.3fs with %d thread(s), band range: %.4f~%.4f'%(time.time()-time0,nthreads,bandmin,bandmax))
        time0=time.time()
        # structure chi_data[iorb][iomega][iqx][iqy][iqz]: gives transverse field response factor
        chi_data=np.zeros((halfOrb,len(omegaAxis),Lx,Ly,Lz),dtype=complex)
        for iorb in range(halfOrb):
            for k1x in range(Lx):
                for k1y in range(Ly):
                    for k1z in range(Lz):
                        for n in range(hr.norb):
                            En_d=eigList[k1x,k1y,k1z,n]
                            S_n_d=vec[k1x,k1y,k1z]
                            S_dagger_n_d=S_n_d.conjugate().T
                            factor_d=S_dagger_n_d[iorb,n]*S_n_d[n,iorb]
                            fn_d=1 if En_d<=fermi else 0
                            for qx in range(Lx):
                                for qy in range(Ly):
                                    for qz in range(Lz):
                                        k2x,k2y,k2z=(k1x+qx)%Lx,(k1y+qy)%Ly,(k1z+qz)%Lz
                                        for m in range(hr.norb):
                                            Em_u=eigList[k2x,k2y,k2z,m]
                                            fm_u=1 if Em_u<=fermi else 0
                                            df=fn_d-fm_u
                                            if df==0: continue
                                            S_m_u=vec[k2x,k2y,k2z]
                                            S_dagger_m_u=S_m_u.conjugate().T
                                            factor_u=S_dagger_m_u[iorb+halfOrb,m]*S_m_u[m,iorb+halfOrb]
                                            #print(S_m_u)
                                            #print(factor_u,factor_d,)
                                            factor_combine=factor_u*factor_d
                                            if abs(factor_combine)<paramReader.SYMPREC: continue
                                            for iomega, omega in enumerate(omegaAxis):
                                                kernal=df/(omega+1j*eta-(Em_u-En_d))
                                                chi_data[iorb,iomega,qx,qy,qz]+=factor_d*factor_u*kernal
        print("Response solved in %.3fs"%(time.time()-time0))
        return chi_data

    '''
    PEAD region
    '''
    def calcEigsAndVecsInMemory_CDLL(self,hr,atomicOrbIndexList:list,kmesh:list=[50,50,1],kpt_list:list=[],nthreads=1):
        '''
        solve the Hk in give k-points
        there are two way to define the k-points list, via kpt_list or kmesh
        when use kpt_list, the frac. coor. of kpts (in units of 2\pi/a) should be expressed as tuples e.g., kpt_list=[(1/3,1/3,0), ...]
        when use kmesh, the grid size should be give e.g., kmesh=[5,5,1]
        If both are give, the kpt_list is prefered
        '''
        # prepare input parameters for cdll
        time0=time.time()
        if len(kpt_list)>0:
            nkpts=len(kpt_list)
            c_kmesh=(c_double*(nkpts*3))()
            for ikpt, kpt in enumerate(kpt_list):
                c_kmesh[ikpt*3+0],c_kmesh[ikpt*3+1],c_kmesh[ikpt*3+2]=c_double(kpt[0]),c_double(kpt[1]),c_double(kpt[2])
        else:
            nkpts=kmesh[0]*kmesh[1]*kmesh[2]
            c_kmesh=(c_double*(nkpts*3))()
            ikpt=0
            # k3 run fastest
            for k1 in np.linspace(0,1,kmesh[0],endpoint=True):
                for k2 in np.linspace(0,1,kmesh[1],endpoint=True):
                    for k3 in np.linspace(0,1,kmesh[2],endpoint=True):
                        c_kmesh[ikpt*3+0],c_kmesh[ikpt*3+1],c_kmesh[ikpt*3+2]=c_double(k1),c_double(k2),c_double(k3)
                        ikpt+=1
                        #print(k1,k2,k3)
        c_lat=(c_double*9)()
        for i in range(3):
            c_lat[i]=c_double(pR.lat[0][i])
            c_lat[i+3]=c_double(pR.lat[1][i])
            c_lat[i+6]=c_double(pR.lat[2][i])
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
        
        assert len(atomicOrbIndexList)==sum(pR.nions), 'pls. provide correctly formatted atomicOrbIndexList[i_atom][i_L]:orb index'
        orbCorrList=[np.zeros(3) for i in range(hr.norb)]
        for atom, pos in zip(atomicOrbIndexList,pR.pos_ions):
            for L in atom:
                for iorb in L: orbCorrList[iorb]=pos
        #assert len(orbCorrList)==hr.norb, "number of wannier centers: %d inconsistent with %d orbs in hr.dat"%(len(orbCorrList),hr.norb)
        c_orbCorrList=(c_double*(3*hr.norb))()
        for i in range(hr.norb):
            c_orbCorrList[3*i+0]=c_double(orbCorrList[i][0])
            c_orbCorrList[3*i+1]=c_double(orbCorrList[i][1])
            c_orbCorrList[3*i+2]=c_double(orbCorrList[i][2])
        
        hamDLL=CDLL("./ham.dll")
        hamDLL.calcFullBandsAndVecsInMemory.restype=py_object
        eigs_addr,vecs_addr,DHkDkx_addr,DHkDky_addr,DHkDkz_addr,DHkDkxDkx_addr,DHkDkyDky_addr,DHkDkxDky_addr=hamDLL.calcFullBandsAndVecsInMemory(
                        c_int(nkpts),c_kmesh,
                        c_int(hr.norb),c_int(nhoppings),
                        c_sourceID,c_targetID,c_overLat,c_hopping_real,c_hopping_imag,
                        c_orbCorrList,c_lat,
                        c_int(nthreads))
        print('Ham solved in %.3fs with %d threads'%(time.time()-time0,nthreads))
        print('pls. check the address of vecs and eigs',hex(eigs_addr),hex(vecs_addr))
        return eigs_addr,vecs_addr,DHkDkx_addr,DHkDky_addr,DHkDkz_addr,DHkDkxDkx_addr,DHkDkyDky_addr,DHkDkxDky_addr
    
    def derivateQuantitiesOnKGrid_CDLL(self,data_addr:int,data_dim:int,kmesh:list,data_type='double',nthreads=1):
        time0=time.time()
        c_kmesh=(3*c_int)()
        c_kmesh[0],c_kmesh[1],c_kmesh[2]=c_int(kmesh[0]),c_int(kmesh[1]),c_int(kmesh[2])
        c_lat=(c_double*9)()
        for i in range(3):
            c_lat[i]=c_double(pR.lat[0][i])
            c_lat[i+3]=c_double(pR.lat[1][i])
            c_lat[i+6]=c_double(pR.lat[2][i])
        hamDLL=CDLL("./ham.dll")
        if data_type=='double':
            c_func=hamDLL.derivateOnKMesh_double
        elif data_type=='complex':
            c_func=hamDLL.derivateOnKMesh_complex
        else:
            print('Error: data_type should either double or complex')
            exit()
        c_func.restype=py_object
        dData_dkx_addr,dData_dky_addr,dData_dkz_addr=c_func(c_longlong(data_addr),c_int(data_dim),
                        c_kmesh,c_lat,c_int(nthreads))
        print('Derivates solved in %.3fs with %d threads'%(time.time()-time0,nthreads))
        print('pls. check the address of vecs and eigs',hex(dData_dkx_addr),hex(dData_dky_addr),hex(dData_dkz_addr))
        return dData_dkx_addr,dData_dky_addr,dData_dkz_addr
    
    def derivateEigvectorsOnKGrid_CDLL(self,vec_addr:int,DHkDk1_addr:int,DHkDk2_addr:int,DHkDk3_addr:int,eigs_addr:int,norb:int,nocc:int,kmesh:list,nthreads=1):
        time0=time.time()
        c_kmesh=(3*c_int)()
        c_kmesh[0],c_kmesh[1],c_kmesh[2]=c_int(kmesh[0]),c_int(kmesh[1]),c_int(kmesh[2])
        c_lat=(c_double*9)()
        for i in range(3):
            c_lat[i]=c_double(pR.lat[0][i])
            c_lat[i+3]=c_double(pR.lat[1][i])
            c_lat[i+6]=c_double(pR.lat[2][i])
        hamDLL=CDLL("./ham.dll")
        c_func=hamDLL.derivateOnKMesh_eigenvector
        c_func.restype=py_object
        dv_dkx_addr,dv_dky_addr,dv_dkz_addr=c_func(c_longlong(vec_addr),
                        c_longlong(DHkDk1_addr),c_longlong(DHkDk2_addr),c_longlong(DHkDk3_addr),c_longlong(eigs_addr),
                        c_int(norb),c_int(nocc),c_kmesh,c_lat,c_int(nthreads))
        print('Derivates solved in %.3fs with %d threads'%(time.time()-time0,nthreads))
        print('pls. check the address',hex(dv_dkx_addr),hex(dv_dky_addr),hex(dv_dkz_addr))
        return dv_dkx_addr,dv_dky_addr,dv_dkz_addr
    
    def redOutQuantitiesOnKGrid(self,data_addr:int,data_dim:int,kmesh:list,data_type='double'):
        c_kmesh=(3*c_int)()
        c_kmesh[0],c_kmesh[1],c_kmesh[2]=c_int(kmesh[0]),c_int(kmesh[1]),c_int(kmesh[2])
        hamDLL=CDLL("./ham.dll")
        if data_type=='double':
            c_func=hamDLL.readOutGridDataOnKMesh_double
        elif data_type=='complex':
            c_func=hamDLL.readOutGridDataOnKMesh_complex
        else:
            print('Error: data_type should either double or complex')
            exit()
        c_func.restype=py_object
        result=c_func(c_longlong(data_addr),c_int(data_dim),
                        c_kmesh)
        # reconstruct the complex type
        if data_type=='complex':
            outData=np.array(result[0])+1j*np.array(result[1])
        elif data_type=='double':
            outData=np.array(result)
        
        # check the dimension
        if data_dim==1:
            return outData
        return outData.reshape((kmesh[0]*kmesh[1]*kmesh[2],data_dim))

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
        energyAXIS=np.linspace(fermi-3,fermi+3,resolution)
        
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

    def calcFullBands_CDLL(self,hr,fermi=0,erange=(0,0),optimalFermi=-100,mainDir=None,saveFig=True,nthreads=1,redBands=[]):
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
        _eigList,_vec_r,_vec_i=result
        eigList=np.array(_eigList).reshape(nkpts,hr.norb)
        print('Spectra solved in %.3fs with %d threads band range: [%.6f,%.6f]'%(time.time()-time0,nthreads,np.max(eigList),np.min(eigList)))
        #nbands=eigList.shape[1]
        plt.figure()
        for ibands in range(hr.norb):
            plt.plot(kaxis,eigList[:,ibands],'-',c='blue')
        for ibands in redBands:
            plt.plot(kaxis,eigList[:,ibands],'-',c='red')
        plt.plot([0,ikpt],[fermi,fermi],'--',c='orange')
        if(optimalFermi>-100): plt.plot([0,ikpt],[optimalFermi,optimalFermi],'--',c='green')
        plt.xlim(0,ikpt)
        if erange[1]!=erange[0]:
            plt.ylim(fermi+erange[0],fermi+erange[1])
        else:    
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
    
    def calcEigsAndVecsInMemory_simple(self,hr,kgrid:list=[1,1,1],nthreads=1):
        # prepare input parameters for cdll
        time0=time.time()
        c_kgrid=(c_int*3)()
        for i in range(3):c_kgrid[i]=kgrid[i]
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
        hamDLL.calcHam_simple.restype=py_object
        result=hamDLL.calcHam_simple(
                        # Hamilton part
                        c_kgrid, 
                        c_int(hr.norb),c_int(nhoppings),
                        c_sourceID,c_targetID,c_overLat,c_hopping_real,c_hopping_imag,
                        c_int(nthreads)
                        )
        eig_addr, vec_addr=result
        print('Hk constructed in %.3fs (with %d threads)'%(time.time()-time0,nthreads))
        print('in python, memory address of eigs.: %s, vecs.: %s'%(hex(eig_addr),hex(vec_addr)))
        #exit()
        return eig_addr, vec_addr

    def calcEigsAndVecsInMemory_mixedFourier(self,hr,kgrid:list=[1,1,1],
            atomicOrbIndexList=[], # give the correspondence of atom id to wannier id
            nthreads=1):
        # prepare input parameters for cdll
        time0=time.time()
        
        nhoppings=len(hr.bondList)
        c_sourceID=(c_int*nhoppings)()
        c_targetID=(c_int*nhoppings)()
        c_overLat=(c_int*(nhoppings*3))()
        c_hopping_real=(c_double*nhoppings)()
        c_hopping_imag=(c_double*nhoppings)()
        R_max,R_min=np.zeros(3,dtype=int),np.zeros(3,dtype=int)
        for ibond, bond in enumerate(hr.bondList):
            #print("%3d %3d %3d %3d %3d %9.6f %9.6f"%(*bond.overlat,bond.source+1,bond.target+1,bond.t.real,bond.t.imag))
            c_sourceID[ibond]=c_int(bond.source)
            c_targetID[ibond]=c_int(bond.target)
            for i in range(3):
                c_overLat[ibond*3+i]=c_int(bond.overlat[i])
                if bond.overlat[i]>R_max[i]: R_max[i]=bond.overlat[i]
                if bond.overlat[i]<R_min[i]: R_min[i]=bond.overlat[i]
            c_hopping_real[ibond]=c_double(bond.t.real)
            c_hopping_imag[ibond]=c_double(bond.t.imag)
        
        # calc real-space grid
        Rgrid=R_max+abs(R_min)+np.ones(3,dtype=int)
        assert (np.array(kgrid)>=Rgrid).all(), "k-grid %dx%dx%d is smaller than Rgrid %dx%dx%d, mix FFT cannot be done"%(*np.array(kgrid),*Rgrid)
        
        for icomp in range(3):
            while(kgrid[icomp]%Rgrid[icomp]!=0): Rgrid[icomp]+=1
        Qgrid=np.array(kgrid)//Rgrid
        assert (Rgrid*Qgrid==np.array(kgrid)).all()
        print(f"Fast (ab inito) Rgrid:{Rgrid}, Slow Qgrid:{Qgrid}")
        
        c_kgrid,c_rgrid,c_qgrid=(c_int*3)(),(c_int*3)(),(c_int*3)()
        for i in range(3):
            c_kgrid[i]=kgrid[i]
            c_rgrid[i]=Rgrid[i]
            c_qgrid[i]=Qgrid[i]
        
        c_lat=(c_double*9)()
        for i in range(3):
            c_lat[i]=c_double(pR.lat[0][i])
            c_lat[i+3]=c_double(pR.lat[1][i])
            c_lat[i+6]=c_double(pR.lat[2][i])

        assert len(atomicOrbIndexList)==sum(pR.nions), 'pls. provide correctly formatted atomicOrbIndexList[i_atom][i_L]:orb index'
        orbCorrList=[np.zeros(3) for i in range(hr.norb)]
        for atom, pos in zip(atomicOrbIndexList,pR.pos_ions):
            for L in atom:
                for iorb in L: orbCorrList[iorb]=pos
        #assert sum(orbCorrList)==hr.norb, "number of wannier centers: %d inconsistent with %d orbs in hr.dat"%(len(orbCorrList),hr.norb)
        c_orbCorrList=(c_double*(3*hr.norb))()
        for i in range(hr.norb):
            c_orbCorrList[3*i+0]=c_double(orbCorrList[i][0])
            c_orbCorrList[3*i+1]=c_double(orbCorrList[i][1])
            c_orbCorrList[3*i+2]=c_double(orbCorrList[i][2])
        
        hamDLL=CDLL("./ham.dll")
        hamDLL.calcHam_mixFFT.restype=py_object
        result=hamDLL.calcHam_mixFFT(
                        # Hamilton part
                        c_kgrid,c_rgrid,c_qgrid,
                        c_int(hr.norb),c_int(nhoppings),
                        c_sourceID,c_targetID,c_overLat,c_hopping_real,c_hopping_imag,
                        c_lat,c_orbCorrList,
                        c_int(nthreads)
                        )
        #print(result)
        #exit()
        eig_addr, vec_addr, velo_addr=result
        print('Hk solved in %.3fs (with %d threads)'%(time.time()-time0,nthreads))
        print('in python, memory address of eigs.: %s, vecs.: %s'%(hex(eig_addr),hex(vec_addr)))
        #exit()
        return eig_addr, vec_addr, velo_addr

    def calcLatticeGreenFunction_CDLL(self,hr,eigs_addr,vecs_addr,relevantOrbIDList,
                    kgrid:list=[1,1,1],eta=1e-4,
                    omega_list=np.zeros(1,dtype=complex),lat_vecs=np.zeros((1,3),dtype=int),
                    spinor=False,
                    nthreads=1):
        # prepare input parameters for cdll
        time0=time.time()

        nOrbInG_w=len(relevantOrbIDList)
        c_orbID_map=(c_int*nOrbInG_w)()
        for iOrbID, orbID in enumerate(relevantOrbIDList):
            c_orbID_map[iOrbID]=c_int(orbID)
            
        c_kgrid=(c_int*3)()
        for i in range(3):c_kgrid[i]=kgrid[i]
        
        nOmega=len(omega_list)
        c_omega_real=(c_double*nOmega)()
        c_omega_imag=(c_double*nOmega)()
        for iomega, omega in enumerate(omega_list):
            c_omega_real[iomega]=c_double(omega.real)
            c_omega_imag[iomega]=c_double(omega.imag)
        
        nLatVec=len(lat_vecs)
        c_latVec=(c_int*(nLatVec*3))()
        for ivec, vec in enumerate(lat_vecs):
            for i in range(3): c_latVec[ivec*3+i]=vec[i]
        spinor_tag=c_int(0)
        if spinor: spinor_tag=c_int(1)
        hamDLL=CDLL("./ham.dll")
        hamDLL.calcRetardedLatticeGreensFunction.restype=py_object
        #print(hex(eigs_addr),hex(vecs_addr),nthreads)
        #exit()
        result=hamDLL.calcRetardedLatticeGreensFunction(
                        c_longlong(eigs_addr),c_longlong(vecs_addr),
                        c_kgrid, 
                        c_int(hr.norb),c_int(nOrbInG_w),c_orbID_map,
                        c_int(nOmega),c_omega_real,c_omega_imag,
                        c_double(eta),
                        c_int(nLatVec),c_latVec,
                        spinor_tag,
                        c_int(nthreads))
        #exit()
        rG_w_lat_real,rG_w_lat_imag=np.array(result).reshape(2,nOmega,nLatVec,nOrbInG_w,nOrbInG_w)
        rG_w_lat=rG_w_lat_real+1j*rG_w_lat_imag
        print('Retarded lattice Greens\' s function solved in %.3fs with %d threads'%(time.time()-time0,nthreads))
        return rG_w_lat

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
        #exit()
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
        
        J=E_coupling/(2*(1-np.cos(theta)))
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
        #assert len(sourceOrbList)==len(targetOrbList), 'Now, the coupling ions is assumed to be symmtric to each other'
        #S=1#len(sourceOrbList)/2.
        assert hr.norb%2==0, 'spin hr must have even orbs'
        norb=hr.norb
        halfNOrb=int(norb/2)
        onsite=np.array(hr.onsite)
        deltaList=onsite[:halfNOrb]-onsite[halfNOrb:]
        #print('S=%.1f'%S)
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

            scatterSingle2List.append([[np.array(targetR),iorb],[np.array(targetR),iorb+halfNOrb],deltaList[iorb]/2.])#*np.sin(theta)])
            scatterSingle2List.append([[np.array(targetR),iorb+halfNOrb],[np.array(targetR),iorb],deltaList[iorb]/2.])#*np.sin(theta)])
            scatterSingle2List.append([[np.array(targetR),iorb],[np.array(targetR),iorb],0])#deltaList[iorb]*(np.cos(theta)-1)])
            scatterSingle2List.append([[np.array(targetR),iorb+halfNOrb],[np.array(targetR),iorb+halfNOrb],0])#-deltaList[iorb]*(np.cos(theta)-1)])
        
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
        c_scatteringSingle1Center=(c_double*(scatteringOrder*scatteringOrder*2))()
        c_scatteringSingle2Center=(c_double*(scatteringOrder*scatteringOrder*2))()
        for i in range(scatteringOrder):
            for j in range(scatteringOrder):
                c_scatteringDoubleCenter[i*scatteringOrder*2+j*2]=c_double(scatteringDoubleCenter[i][j].real)
                c_scatteringDoubleCenter[i*scatteringOrder*2+j*2+1]=c_double(scatteringDoubleCenter[i][j].imag)

                c_scatteringSingle1Center[i*scatteringOrder*2+j*2]=c_double(scatteringSingle1Center[i][j].real)
                c_scatteringSingle1Center[i*scatteringOrder*2+j*2+1]=c_double(scatteringSingle1Center[i][j].imag)

                c_scatteringSingle2Center[i*scatteringOrder*2+j*2]=c_double(scatteringSingle2Center[i][j].real)
                c_scatteringSingle2Center[i*scatteringOrder*2+j*2+1]=c_double(scatteringSingle2Center[i][j].imag)
        
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
                         c_scatteringSingle1Center,
                         c_scatteringSingle2Center)
        
        #a=getEOnCircle(np.pi)
        #print(a)
        #exit()

        print('prepared for integration for J calculation')
        time0=time.time()
        E_coupling, E_coupling_err =integrate.quad(getEOnCircle,a=np.pi,b=0.,epsabs=paramReader.EDIFF)
        print("Coupling energy obtained in %.3fs (with cdll paralleled with %d threads) ===> %.6f +- %.6f"%(time.time()-time0,nthreads,E_coupling,E_coupling_err))
        
        J=E_coupling#/(-2*(np.cos(theta)-1))
        return J

    def getFourStateScatterMatrix_x(self,hr,sourceOrbList=[0],sourceR=[0,0,0],targetOrbList=[0],targetR=[1,0,0],Lx=5,Ly=5,Lz=1):
        '''
        Construct scatter matrix due to spin derivate to x-direction
        '''
        #assert len(sourceOrbList)==len(targetOrbList), 'Now, the coupling ions is assumed to be symmtric to each other'
        #S=1#len(sourceOrbList)/2.
        assert hr.norb%2==0, 'spin hr must have even orbs'
        norb=hr.norb
        halfNOrb=int(norb/2)
        onsite=np.array(hr.onsite)
        deltaList=onsite[:halfNOrb]-onsite[halfNOrb:]
        #print('S=%.1f'%S)
        #print('Exchange splitting are:')
        #print(deltaList)
        
        # create scattering route
        scatterDoubleList=[]
        scatterSingle1List=[]
        scatterSingle2List=[]
        for iorb in sourceOrbList:
            scatterDoubleList.append([[np.array(sourceR),iorb],[np.array(sourceR),iorb+halfNOrb],deltaList[iorb]/2.])
            scatterDoubleList.append([[np.array(sourceR),iorb+halfNOrb],[np.array(sourceR),iorb],deltaList[iorb]/2.])
            scatterDoubleList.append([[np.array(sourceR),iorb],[np.array(sourceR),iorb],0])
            scatterDoubleList.append([[np.array(sourceR),iorb+halfNOrb],[np.array(sourceR),iorb+halfNOrb],0])

            scatterSingle1List.append([[np.array(sourceR),iorb],[np.array(sourceR),iorb+halfNOrb],deltaList[iorb]/2.])
            scatterSingle1List.append([[np.array(sourceR),iorb+halfNOrb],[np.array(sourceR),iorb],deltaList[iorb]/2.])
            scatterSingle1List.append([[np.array(sourceR),iorb],[np.array(sourceR),iorb],0])
            scatterSingle1List.append([[np.array(sourceR),iorb+halfNOrb],[np.array(sourceR),iorb+halfNOrb],0])

            scatterSingle2List.append([[np.array(sourceR),iorb],[np.array(sourceR),iorb+halfNOrb],0])
            scatterSingle2List.append([[np.array(sourceR),iorb+halfNOrb],[np.array(sourceR),iorb],0])
            scatterSingle2List.append([[np.array(sourceR),iorb],[np.array(sourceR),iorb],0])
            scatterSingle2List.append([[np.array(sourceR),iorb+halfNOrb],[np.array(sourceR),iorb+halfNOrb],0])
        
        for iorb in targetOrbList:
            scatterDoubleList.append([[np.array(targetR),iorb],[np.array(targetR),iorb+halfNOrb],deltaList[iorb]/2.])
            scatterDoubleList.append([[np.array(targetR),iorb+halfNOrb],[np.array(targetR),iorb],deltaList[iorb]/2.])
            scatterDoubleList.append([[np.array(targetR),iorb],[np.array(targetR),iorb],0])
            scatterDoubleList.append([[np.array(targetR),iorb+halfNOrb],[np.array(targetR),iorb+halfNOrb],0])

            scatterSingle1List.append([[np.array(targetR),iorb],[np.array(targetR),iorb+halfNOrb],0])
            scatterSingle1List.append([[np.array(targetR),iorb+halfNOrb],[np.array(targetR),iorb],0])
            scatterSingle1List.append([[np.array(targetR),iorb],[np.array(targetR),iorb],0])
            scatterSingle1List.append([[np.array(targetR),iorb+halfNOrb],[np.array(targetR),iorb+halfNOrb],0])

            scatterSingle2List.append([[np.array(targetR),iorb],[np.array(targetR),iorb+halfNOrb],deltaList[iorb]/2.])
            scatterSingle2List.append([[np.array(targetR),iorb+halfNOrb],[np.array(targetR),iorb],deltaList[iorb]/2.])
            scatterSingle2List.append([[np.array(targetR),iorb],[np.array(targetR),iorb],0])
            scatterSingle2List.append([[np.array(targetR),iorb+halfNOrb],[np.array(targetR),iorb+halfNOrb],0])
        
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
        #print('find scattering region with %d-dimension, basis consist of'%scatteringOrder)
        relevantOrbIdList=[]
        for basis in relevantOrbList:
            relevantOrbIdList.append(basis[1])
            #print('orb%d at [%d %d %d]'%(basis[1],*basis[0]))
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
        return relevantOrbIdList, scatteringDoubleCenter, scatteringSingle1Center, scatteringSingle2Center

    def getFourStateScatterMatrix_y(self,hr,sourceOrbList=[0],sourceR=[0,0,0],targetOrbList=[0],targetR=[1,0,0],Lx=5,Ly=5,Lz=1):
        '''
        Construct scatter matrix due to spin derivate to y-direction
        '''
        assert hr.norb%2==0, 'spin hr must have even orbs'
        norb=hr.norb
        halfNOrb=int(norb/2)
        onsite=np.array(hr.onsite)
        deltaList=onsite[:halfNOrb]-onsite[halfNOrb:]
        
        # create scattering route
        scatterDoubleList=[]
        scatterSingle1List=[]
        scatterSingle2List=[]
        for iorb in sourceOrbList:
            scatterDoubleList.append([[np.array(sourceR),iorb],[np.array(sourceR),iorb+halfNOrb],-1j*deltaList[iorb]/2.])
            scatterDoubleList.append([[np.array(sourceR),iorb+halfNOrb],[np.array(sourceR),iorb],1j*deltaList[iorb]/2.])
            scatterDoubleList.append([[np.array(sourceR),iorb],[np.array(sourceR),iorb],0])
            scatterDoubleList.append([[np.array(sourceR),iorb+halfNOrb],[np.array(sourceR),iorb+halfNOrb],0])

            scatterSingle1List.append([[np.array(sourceR),iorb],[np.array(sourceR),iorb+halfNOrb],-1j*deltaList[iorb]/2.])
            scatterSingle1List.append([[np.array(sourceR),iorb+halfNOrb],[np.array(sourceR),iorb],1j*deltaList[iorb]/2.])
            scatterSingle1List.append([[np.array(sourceR),iorb],[np.array(sourceR),iorb],0])
            scatterSingle1List.append([[np.array(sourceR),iorb+halfNOrb],[np.array(sourceR),iorb+halfNOrb],0])

            scatterSingle2List.append([[np.array(sourceR),iorb],[np.array(sourceR),iorb+halfNOrb],0])
            scatterSingle2List.append([[np.array(sourceR),iorb+halfNOrb],[np.array(sourceR),iorb],0])
            scatterSingle2List.append([[np.array(sourceR),iorb],[np.array(sourceR),iorb],0])
            scatterSingle2List.append([[np.array(sourceR),iorb+halfNOrb],[np.array(sourceR),iorb+halfNOrb],0])
        
        for iorb in targetOrbList:
            scatterDoubleList.append([[np.array(targetR),iorb],[np.array(targetR),iorb+halfNOrb],-1j*deltaList[iorb]/2.])
            scatterDoubleList.append([[np.array(targetR),iorb+halfNOrb],[np.array(targetR),iorb],1j*deltaList[iorb]/2.])
            scatterDoubleList.append([[np.array(targetR),iorb],[np.array(targetR),iorb],0])
            scatterDoubleList.append([[np.array(targetR),iorb+halfNOrb],[np.array(targetR),iorb+halfNOrb],0])

            scatterSingle1List.append([[np.array(targetR),iorb],[np.array(targetR),iorb+halfNOrb],0])
            scatterSingle1List.append([[np.array(targetR),iorb+halfNOrb],[np.array(targetR),iorb],0])
            scatterSingle1List.append([[np.array(targetR),iorb],[np.array(targetR),iorb],0])
            scatterSingle1List.append([[np.array(targetR),iorb+halfNOrb],[np.array(targetR),iorb+halfNOrb],0])

            scatterSingle2List.append([[np.array(targetR),iorb],[np.array(targetR),iorb+halfNOrb],-1j*deltaList[iorb]/2.])
            scatterSingle2List.append([[np.array(targetR),iorb+halfNOrb],[np.array(targetR),iorb],1j*deltaList[iorb]/2.])
            scatterSingle2List.append([[np.array(targetR),iorb],[np.array(targetR),iorb],0])
            scatterSingle2List.append([[np.array(targetR),iorb+halfNOrb],[np.array(targetR),iorb+halfNOrb],0])
        
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
        #print('find scattering region with %d-dimension, basis consist of'%scatteringOrder)
        relevantOrbIdList=[]
        for basis in relevantOrbList:
            relevantOrbIdList.append(basis[1])
            #print('orb%d at [%d %d %d]'%(basis[1],*basis[0]))
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
        return relevantOrbIdList, scatteringDoubleCenter, scatteringSingle1Center, scatteringSingle2Center