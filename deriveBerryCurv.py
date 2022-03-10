import paramReader
paramReader.readGlobalConst()
import posReader as pR
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from ctypes import c_double, c_int, CDLL, py_object, c_char, c_longlong

class VASPProcessor:
    def __init__(self,rpath) -> None:
        self.rpath=rpath
        self.nkpt=-1
        self.nbnd=-1
        self.nelect=-1
        self.nspin=-1

    def readEIGENVAL(self) -> None:
        pathToEIGENVAL=os.path.join(self.rpath,'EIGENVAL')
        self.vkpt=[]
        self.weights=[]
        time0=time.time()
        with open(pathToEIGENVAL,'r') as fin:
            EIGENVAL=fin.read().split('\n')
            self.nelect, self.nkpt, self.nbnd =[int(x) for x in EIGENVAL[5].split()]
            data=EIGENVAL[8].split()
            if len(data)==3:
                self.nspin=1
                self.eigs=[[]]
                self.occupations=[[]]
            elif len(data)==5:
                self.nspin=2
                self.eigs=[[],[]]
                self.occupations=[[],[]]
            else:
                print("unknow format in %s: %s"%(pathToEIGENVAL,EIGENVAL[8]))
                exit()
            print("NELECT: %d, NBANDS: %d, NKPTS: %d, NSPIN: %d"%(self.nelect,self.nbnd,self.nkpt,self.nspin))
            for i in range(6): EIGENVAL.pop(0)
            for line in EIGENVAL:
                linedata=line.split()
                if len(linedata)==0: # empty line
                    continue
                elif len(linedata)==4: # counter k-point line
                    self.vkpt.append(np.array([float(linedata[0]),float(linedata[1]),float(linedata[2])]))
                    self.weights.append(float(linedata[3]))
                    self.eigs[0].append([])  # constructions for spin-up
                    self.occupations[0].append([])
                    if self.nspin==2: # extra constructions for spin-dn
                        self.eigs[1].append([])
                        self.occupations[1].append([])
                elif len(linedata)==3:
                    ibnd,energy,occupation=int(linedata[0]),float(linedata[1]),float(linedata[2])
                    self.eigs[0][-1].append(energy)
                    self.occupations[0][-1].append(occupation)
                elif len(linedata)==5:
                    ibnd,energy_up,energy_dn,occupation_up,occupation_dn=\
                        int(linedata[0]),float(linedata[1]),float(linedata[2]),float(linedata[3]),float(linedata[4])
                    self.eigs[0][-1].append(energy_up)
                    self.eigs[1][-1].append(energy_dn)
                    self.occupations[0][-1].append(occupation_up)
                    self.occupations[1][-1].append(occupation_dn)
                else:
                    print("unknow format by countering:",line)
        print("EIGENVAL read in %.3fs"%(time.time()-time0))
    
    def getFullKPT(self,kmesh,rotations,timeReversalSym=True) -> None:
        '''
        getFullKPT(kmesh: 3d-list, rotations: list of symmetry rotations, timeReversalSym: bool to switch the TRS)
        VASP only calculated properties on symmetry-reduced k-points
        This func. intends to found out how the reduced kpts related to full k-mesh
        '''
        triggerSign=np.zeros((kmesh[0],kmesh[1],kmesh[2]),dtype=int) # sign of correspondence
        dk_f=np.array([1/kmesh[0],1/kmesh[1],1/kmesh[2]]) # dk step implicated by k-mesh size

        rotation_frac_reciprocal=[]
        for rotate in rotations:
            rotation_frac_reciprocal.append(rotate.T)
            # time-reversal-symmetry involved operation
            if timeReversalSym:rotation_frac_reciprocal.append(-rotate.T)
        
        kpt_index_map=np.zeros((kmesh[0],kmesh[1],kmesh[2]),dtype=int) # kpt_index_map[ik1,ik2,ik3]=index of reduced kpt
        kpt_rotation_map=[[[np.eye(3) for kz in range(kmesh[2])] for ky in range(kmesh[1])] for kx in range(kmesh[0])]
        for ikpt, kpt in enumerate(self.vkpt):
            weight=self.weights[ikpt]
            # kpt folds indicated by VASP, note that the gamma pt always presents 1-fold
            nfolds=int(np.around(weight/self.weights[0]))
            # total folds, before consider this reduced-kpt
            nSetting_old=sum(sum(sum(triggerSign)))
            equal_kpt_list=[]
            for rotation_reci in rotation_frac_reciprocal:
                kpt_transform=(rotation_reci.dot(kpt))%1
                # ik1,ik2,ik3 correspond to the position index of transformed kpt in full kmesh
                ik1,ik2,ik3=[int(x)%50 for x in np.around(kpt_transform/dk_f)]
                if triggerSign[ik1,ik2,ik3]==0: # found a new kpt in full kmesh
                    kpt_index_map[ik1,ik2,ik3]=ikpt
                    kpt_rotation_map[ik1][ik2][ik3]=rotation_reci
                    triggerSign[ik1,ik2,ik3]=1
                equal_kpt_list.append({'origin kpt':kpt,'rotation':rotation_reci,'trans kpt':kpt_transform,'position':[ik1,ik2,ik3]})
            nSetting_new=sum(sum(sum(triggerSign)))
            if nSetting_new-nSetting_old!=nfolds:
                #print(nfolds,nSetting_new-nSetting_old)
                for info in equal_kpt_list:
                    print(info['rotation'])
                    print(info['trans kpt'],info['position'])
                exit()
        print(sum(sum(sum(triggerSign))))
        return

    def readVelocity(self) -> None:
        assert self.nspin>0 and self.nkpt>0 and self.nbnd>0,'unknow nspin|nkpt|nbnd, may be read EIGENVAL first'
        time0=time.time()
        pathToVelocity=list(os.path.join(self.rpath,'VELOCITY'))
        c_path=(c_char*len(pathToVelocity))()
        for ic, c in enumerate(pathToVelocity):
            c_path[ic]=c_char(c.encode())
        mylib=CDLL('./auxiliary.dll')
        readVelocity=mylib.readVelocity
        readVelocity.restype=py_object
        velocity_real,velocity_imag,self.velocity_addr=readVelocity(c_path,c_int(self.nspin),c_int(self.nkpt),c_int(self.nbnd))
        print('velocity address in python: %s'%hex(self.velocity_addr))
        self.velocities=(np.array(velocity_real)+1j*np.array(velocity_imag)).reshape((3,self.nspin,self.nkpt,self.nbnd,self.nbnd))
        print('velocity established in %.3fs'%(time.time()-time0))

    def calcCircularPolarization(self,iv,ic):
        time0=time.time()
        px=np.zeros((self.nspin,self.nkpt),dtype=complex)
        py=np.zeros((self.nspin,self.nkpt),dtype=complex)
        for ikpt in range(self.nkpt):
            for ispin in range(self.nspin):
                assert self.occupations[ispin][ikpt][iv]>=0.8,'occupation of band%d is %d<0.8'%(iv,self.occupations[ispin][ikpt][iv])
                assert self.occupations[ispin][ikpt][ic]<=0.8,'occupation of band%d is %d>0.2'%(ic,self.occupations[ispin][ikpt][ic])
                px[ispin,ikpt]=self.velocities[0,ispin,ikpt,iv,ic]
                py[ispin,ikpt]=self.velocities[1,ispin,ikpt,iv,ic]
        print('Circular polarization calculated in %.3fs'%(time.time()-time0))
        return px,py
    def calcBerryCurv(self):
        time0=time.time()
        curv_x=np.zeros(self.nkpt,dtype=complex)
        curv_y=np.zeros(self.nkpt,dtype=complex)
        curv_z=np.zeros(self.nkpt,dtype=complex)
        for ikpt in range(self.nkpt):
            for ispin in range(self.nspin):
                for n1 in range(self.nbnd):
                    if self.occupations[ispin][ikpt][n1]<0.8:continue
                    e1=self.eigs[ispin][ikpt][n1]
                    for n2 in range(self.nbnd):
                        if self.occupations[ispin][ikpt][n2]>0.2: continue
                        e2=self.eigs[ispin][ikpt][n2]
                        curv_x[ikpt]+=self.velocities[1,ispin,ikpt,n1,n2]*self.velocities[2,ispin,ikpt,n2,n1]/(e1-e2)**2
                        curv_y[ikpt]+=self.velocities[2,ispin,ikpt,n1,n2]*self.velocities[0,ispin,ikpt,n2,n1]/(e1-e2)**2
                        curv_z[ikpt]+=self.velocities[0,ispin,ikpt,n1,n2]*self.velocities[1,ispin,ikpt,n2,n1]/(e1-e2)**2
        print('Berry curv calculated in %.3fs'%(time.time()-time0))
        return curv_x, curv_y, curv_z
    
    def calcBPVC(self,EphotonList:np.ndarray,erange=[-50,50],tau=0.066,type='bulk',mainDir:str=None,nthreads:int=1):
        time0=time.time()
        #'''
        c_erange=(2*c_double)()
        c_erange[0],c_erange[1]=c_double(erange[0]),c_double(erange[1])
        nphoton=len(EphotonList)
        c_EphotonList=(nphoton*c_double)()
        for iomega, omega in enumerate(EphotonList):
            c_EphotonList[iomega]=c_double(omega)
        
        c_eigs=(self.nspin*self.nkpt*self.nbnd*c_double)()
        c_occupations=(self.nspin*self.nkpt*self.nbnd*c_double)()
        cnt=0
        for ispin in range(self.nspin):
            for ikpt in range(self.nkpt):
                for iband in range(self.nbnd):
                    c_eigs[cnt]=c_double(self.eigs[ispin][ikpt][iband])
                    c_occupations[cnt]=c_double(self.occupations[ispin][ikpt][iband])
                    cnt+=1
        mylib=CDLL('./auxiliary.dll')
        if type=='bulk':
            calcBPVC=mylib.calcBPVC
        elif type=='shift':
            calcBPVC=mylib.calcShiftBPVC
        else:
            print("Error: type should be either bulk or shift")
            exit()
        calcBPVC.restype=py_object
        data=calcBPVC(c_int(nphoton),c_EphotonList,c_erange,c_double(tau),c_longlong(self.velocity_addr),
                      c_int(self.nspin),c_int(self.nkpt),c_int(self.nbnd),c_eigs,c_occupations,c_int(nthreads))
        cx_xx_r, cx_xx_i, cy_xx_r, cy_xx_i, cx_yy_r, cx_yy_i, cy_yy_r, cy_yy_i, cx_xy_r, cx_xy_i, cy_xy_r, cy_xy_i, cx_yx_r, cx_yx_i, cy_yx_r, cy_yx_i=np.array(data).reshape((16,nphoton,self.nkpt))
        cx_xx=(np.array(cx_xx_r)+1j*np.array(cx_xx_i))#.reshape((nphoton,self.nkpt))
        cy_xx=(np.array(cy_xx_r)+1j*np.array(cy_xx_i))#.reshape((nphoton,self.nkpt))
        cx_yy=(np.array(cx_yy_r)+1j*np.array(cx_yy_i))#.reshape((nphoton,self.nkpt))
        cy_yy=(np.array(cy_yy_r)+1j*np.array(cy_yy_i))#.reshape((nphoton,self.nkpt))
        cx_xy=(np.array(cx_xy_r)+1j*np.array(cx_xy_i))#.reshape((nphoton,self.nkpt))
        cy_xy=(np.array(cy_xy_r)+1j*np.array(cy_xy_i))#.reshape((nphoton,self.nkpt))
        cx_yx=(np.array(cx_yx_r)+1j*np.array(cx_yx_i))#.reshape((nphoton,self.nkpt))
        cy_yx=(np.array(cy_yx_r)+1j*np.array(cy_yx_i))#.reshape((nphoton,self.nkpt))
        #'''
        print('Conductances calculated in %.3fs'%(time.time()-time0))

        if mainDir!=None:
            #pathToFile=os.path.join(mainDir,'photoCondTensorVsOmega.txt')
            #if type=='shift':pathToFile=os.path.join(mainDir,'shiftCondTensorVsOmega.txt')
            
            if nphoton==1:
                pathToFile=os.path.join(mainDir,'photoCondTensor.txt')
                if type=='shift':pathToFile=os.path.join(mainDir,'shiftCondTensor.txt')
                with open(pathToFile,'w') as fout:
                    for kpt, _cxxx,_cxxy,_cyyx,_cyyy,_cxyx,_cxyy,_cyxx,_cyxy in zip(self.vkpt,cx_xx[0],cy_xx[0],cx_yy[0],cy_yy[0],cx_xy[0],cy_xy[0],cx_yx[0],cy_yx[0]):
                        fout.write(" %.4f %.4f %.4f %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E\n"%(*kpt,_cxxx.real,_cxxx.imag,_cyxx.real,_cyxx.imag,_cxyy.real,_cxyy.imag,_cyyy.real,_cyyy.imag,_cxxy.real,_cxxy.imag,_cyxy.real,_cyxy.imag,_cxyx.real,_cxyx.imag,_cyyx.real,_cyyx.imag))
                return
            else:
                pathToFile=os.path.join(mainDir,'photoCondTensorVsOmega.txt')
                if type=='shift':pathToFile=os.path.join(mainDir,'shiftCondTensorVsOmega.txt')
                with open(pathToFile,'w') as fout:
                    fout.write("Omega Sxx_x_r Sxx_x_i Sxx_y_r Sxx_y_i Syy_x_r Syy_x_i Syy_y_r Syy_y_i Sxy_x_r Sxy_x_i Sxy_y_r Sxy_y_i Syx_x_r Syx_x_i Syx_y_r Syx_y_i\n")
                    for omega, _cxxx,_cxxy,_cyyx,_cyyy,_cxyx,_cxyy,_cyxx,_cyxy in zip(EphotonList,cx_xx,cy_xx,cx_yy,cy_yy,cx_xy,cy_xy,cx_yx,cy_yx):
                        fout.write(" %.4f %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E\n"%(
                            omega,
                            sum(_cxxx/self.nkpt).real,sum(_cxxx/self.nkpt).imag,
                            sum(_cxxy/self.nkpt).real,sum(_cxxy/self.nkpt).imag,
                            sum(_cyyx/self.nkpt).real,sum(_cyyx/self.nkpt).imag,
                            sum(_cyyy/self.nkpt).real,sum(_cyyy/self.nkpt).imag,
                            sum(_cxyx/self.nkpt).real,sum(_cxyx/self.nkpt).imag,
                            sum(_cxyy/self.nkpt).real,sum(_cxyy/self.nkpt).imag,
                            sum(_cyxx/self.nkpt).real,sum(_cyxx/self.nkpt).imag,
                            sum(_cyxy/self.nkpt).real,sum(_cyxy/self.nkpt).imag))
                for iomega, omega in enumerate(EphotonList):
                    pathToFile=os.path.join(mainDir,'photoCondTensor%d_E%.3f.txt'%(iomega,omega))
                    if type=='shift':pathToFile=os.path.join(mainDir,'shiftCondTensor%d_E%.3f.txt'%(iomega,omega))
                    with open(pathToFile,'w') as fout:
                        for kpt, _cxxx,_cxxy,_cyyx,_cyyy,_cxyx,_cxyy,_cyxx,_cyxy in zip(self.vkpt,cx_xx[iomega],cy_xx[iomega],cx_yy[iomega],cy_yy[iomega],cx_xy[iomega],cy_xy[iomega],cx_yx[iomega],cy_yx[iomega]):
                            fout.write(" %.4f %.4f %.4f %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E %13.6E\n"%(*kpt,_cxxx.real,_cxxx.imag,_cyxx.real,_cyxx.imag,_cxyy.real,_cxyy.imag,_cyyy.real,_cyyy.imag,_cxxy.real,_cxxy.imag,_cyxy.real,_cyxy.imag,_cxyx.real,_cxyx.imag,_cyyx.real,_cyyx.imag))
            

        condcxxxList=np.array([np.sum(cond_on_omega/self.nkpt) for cond_on_omega in cx_xx])
        condcxxyList=np.array([np.sum(cond_on_omega/self.nkpt) for cond_on_omega in cy_xx])
        condcyyxList=np.array([np.sum(cond_on_omega/self.nkpt) for cond_on_omega in cx_yy])
        condcyyyList=np.array([np.sum(cond_on_omega/self.nkpt) for cond_on_omega in cy_yy])
        condcxyxList=np.array([np.sum(cond_on_omega/self.nkpt) for cond_on_omega in cx_xy])
        condcxyyList=np.array([np.sum(cond_on_omega/self.nkpt) for cond_on_omega in cy_xy])
        condcyxxList=np.array([np.sum(cond_on_omega/self.nkpt) for cond_on_omega in cx_yx])
        condcyxyList=np.array([np.sum(cond_on_omega/self.nkpt) for cond_on_omega in cy_yx])

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
            for _ax in row:
                _ax.legend()
                _ax.plot([EphotonList[0],EphotonList[-1]],[0,0],'--',linewidth=1)
        plt.show()

        return cx_xx, cy_xx, cx_yy, cy_yy, cx_xy, cy_xy, cx_yx, cy_yx

def makeGrid(kmesh,reciLat,curv_z):
    # expand grid to include border, needed for pcolormesh plotting
    X=np.linspace(0,1,kmesh[0]+1,endpoint=True)
    Y=np.linspace(0,1,kmesh[1]+1,endpoint=True)
    Z=np.linspace(0,0,kmesh[2],endpoint=True)
    kx_grid_f,ky_grid_f,kz_grid_f=np.meshgrid(X,Y,Z)
    #print(x.shape,y.shape,z.shape)
    #exit()
    kpts_f=np.array([kx_grid_f.flatten(),ky_grid_f.flatten(),kz_grid_f.flatten()])
    kpts_cart=np.dot(reciLat,kpts_f)
    X_cart=kpts_cart[0]
    Y_cart=kpts_cart[1]

    curv_z_grid_origin=curv_z.reshape((kmesh[1],kmesh[0]))
    curv_z_grid= []
    for row in curv_z_grid_origin:
        row_add=list(row)
        row_add.append(row[0])
        curv_z_grid.append(row_add)
    curv_z_grid.append(curv_z_grid[0])

    kx_grid=X_cart.reshape((kmesh[1]+1,kmesh[0]+1))
    ky_grid=Y_cart.reshape((kmesh[1]+1,kmesh[0]+1))
    return kx_grid, ky_grid, np.array(curv_z_grid)

def drawPatternOnHexagonalBZ(rpath,kx_grid,ky_grid,data_grid,reciLat):
    g1,g2=reciLat[:2,0],reciLat[:2,1]
    maxValue=np.max(abs(data_grid))
    plt.figure(figsize=(6,6))

    plt.pcolormesh(kx_grid,ky_grid,data_grid,shading='nearest',cmap="Spectral",vmin=-maxValue,vmax=maxValue)
    plt.pcolormesh(kx_grid-g1[0],ky_grid-g1[1],data_grid,shading='nearest',cmap="Spectral",vmin=-maxValue,vmax=maxValue)
    plt.pcolormesh(kx_grid-g2[0],ky_grid-g2[1],data_grid,shading='nearest',cmap="Spectral",vmin=-maxValue,vmax=maxValue)
    plt.pcolormesh(kx_grid-(g1+g2)[0],ky_grid-(g1+g2)[1],data_grid,shading='nearest',cmap="Spectral",vmin=-maxValue,vmax=maxValue)
    
    #plt.colorbar()
    # shift grid for contour plotting
    shiftx=(reciLat[0][0]+reciLat[0][1])/(2*kx_grid.shape[0])
    shifty=(reciLat[1][0]+reciLat[1][1])/(2*kx_grid.shape[1])
    _kx_grid=kx_grid+shiftx
    _ky_grid=ky_grid+shifty
    #print(_kx_grid.shape)
    sub1=plt.contour(_kx_grid,_ky_grid,data_grid,linewidths=1,linestyles="dashed",colors="black")
    sub2=plt.contour(_kx_grid-g1[0],_ky_grid-g1[1],data_grid,linewidths=1,linestyles="dashed",colors="black")
    sub3=plt.contour(_kx_grid-g2[0],_ky_grid-g2[1],data_grid,linewidths=1,linestyles="dashed",colors="black")
    sub4=plt.contour(_kx_grid-(g1+g2)[0],_ky_grid-(g1+g2)[1],data_grid,linewidths=1,linestyles="dashed",colors="black")

    if np.dot(reciLat[0],reciLat[1])>0: # g1 cross g2 by 60 deg.
        K1=(g1+g2)/3
        K2=((g1-g2)+g1)/3
        K3=(-g2+(g1-g2))/3
        K4=-(g1+g2)/3
        K5=-((g1-g2)+g1)/3
        K6=-(-g2+(g1-g2))/3
    else: # g1 cross g2 by 120 deg.
        K1=(g1+2*g2)/3
        K2=(-g1+g2)/3
        K3=(-2*g1-g2)/3
        K4=-(g1+2*g2)/3
        K5=-(-g1+g2)/3
        K6=-(-2*g1-g2)/3

    K=np.array([K1,K2,K3,K4,K5,K6,K1])
    corr_max=np.max([np.max(K[:,0]),np.max(K[:,1])])*1.1
    plt.plot(K[:,0]+shiftx,K[:,1]+shifty,linewidth=3,c="black")
    plt.xlim(-corr_max,corr_max)
    plt.ylim(-corr_max,corr_max)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(rpath+"BerryCurv_z.png",dpi=300)
    plt.show()


pR.readPOS('./mcalc/BN/POSCAR')
reciLat=np.linalg.inv(pR.lat)*np.pi*2
kmesh=[100,100,1]

vaspPro=VASPProcessor('./mcalc/BN')
vaspPro.readEIGENVAL()
#rotations,translations=pR.getWholeSymmetryOperator()
#vaspPro.getFullKPT(kmesh,rotations)
vaspPro.readVelocity()
#for ikpt in range(vaspPro.nkpt):
#    print(vaspPro.velocities[0,0,ikpt,3,3].real)
#exit()
omegaList=np.linspace(3.0,6.0,31,endpoint=True)
cx_xx, cy_xx, cx_yy, cy_yy, cx_xy, cy_xy, cx_yx, cy_yx=vaspPro.calcBPVC(
    EphotonList=omegaList,tau=0.02,type='bulk',mainDir='./mcalc/BN/',nthreads=4)
exit()

#curv_x_flat,curv_y_flat,curv_z_flat=vaspPro.calcBerryCurv()
px,py=vaspPro.calcCircularPolarization(3,4)
pr,pl=px+1j*py,px-1j*py
curv_z_flat=(abs(pr)**2-abs(pl)**2)/(abs(pr)**2+abs(pl)**2)
#curv_z_flat=np.array([v[4,4] for v in vaspPro.velocities[1][0]]).real
#curv_z_flat=cy_xx[0].imag#np.array([eigk[4]-eigk[3] for eigk in vaspPro.eigs[0]])
print(np.min(curv_z_flat))
kx_grid, ky_grid, curv_z= makeGrid(kmesh,reciLat,curv_z_flat)
drawPatternOnHexagonalBZ('./mcalc/BN/',kx_grid,ky_grid,curv_z,reciLat)