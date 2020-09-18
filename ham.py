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

    def calcBands(self,hr,orbList=[0],spin_polarized=False,fermi=0,eta=0.1,resolution=100):
        norb=hr.norb
        if len(orbList)==0:
            orbList=[i for i in range(norb)]
        if spin_polarized:
            orbUpList=orbList[:len(orbList)//2]
            orbDnList=orbList[len(orbList)//2:]
        HkList=[]
        
        time0=time.time()
        kmesh=pR.autoGenerateKpath2D(onlyGetKPTList=True)
        for ikpt, kpt in enumerate(kmesh):
            _ham=Ham()
            Hk=_ham.buildHkFromHr(hr,kpt,getHk=True)
            HkList.append(Hk)
            
        print('Hk constructed in %.3fs'%(time.time()-time0))
        energyAXIS=np.linspace(fermi-3,fermi+2,resolution)
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
        plt.show()
                
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
        return Emin, Emax, fermi#fermi_adjusted

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
        print('Hk constructed in %.3fs'%(time.time()-time0))
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
                    fout.wirte('\n')
        plt.show()
        plt.close()

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
            _ham[iorb_source][iorb_target]+=scatter[2]
        
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
        
        #scatteringCenter=scatteringOppoCenter
        #time0=time.time()
        #E_minus,E_minus_err=integrate.quad(getEOnCircle,a=np.pi,b=0.)
        #print("Energy obtained in %.3fs ===> opposite biasing: %.6f +- %.6f"%(time.time()-time0,E_minus,E_minus_err))
        
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
        #E_plus, E_plus_err =self.calcMatrixPerturbatedEnergyViaExact(hr,scatterTheSameDirectionList,eta=eta,energyRange=energyRange,Lx=Lx,Ly=Ly,Lz=Lz,getErr=True)
        E_plus, E_plus_err =self.calcMatrixPerturbatedEnergyViaFourier(hr,scatterTheSameDirectionList,eta=eta,energyRange=energyRange,Lx=Lx,Ly=Ly,Lz=Lz,getErr=True,skipOdds=True)
        print("Energy obtained in %.3fs ===> same biasing: %.6f +- %.6f"%(time.time()-time0,E_plus,E_plus_err))
        
        time0=time.time()
        #E_minus,E_minus_err=self.calcMatrixPerturbatedEnergyViaExact(hr,scatterTheOppoDirectionList,eta=eta,energyRange=energyRange,Lx=Lx,Ly=Ly,Lz=Lz,getErr=True)
        E_minus,E_minus_err=self.calcMatrixPerturbatedEnergyViaFourier(hr,scatterTheOppoDirectionList,eta=eta,energyRange=energyRange,Lx=Lx,Ly=Ly,Lz=Lz,getErr=True,skipOdds=True)
        print("Energy obtained in %.3fs ===> opposite biasing: %.6f +- %.6f"%(time.time()-time0,E_minus,E_minus_err))
        
        J=(E_plus-E_minus)/2.#/(S**2)
        return J



