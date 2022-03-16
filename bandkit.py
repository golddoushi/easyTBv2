import os
import re
import numpy as np
import time

class ProEval:
    """Projected band structure class
    "public variables"
    nkp=-1
    nbnd=-1
    nion=-1
    spinor=True
    orb_f=False
    spn_ndg=False
    fermi=0.
    eval_max=0.0
    eval_min=0.0
    kpt_path=[]
    "teval[ikpt][ibnd], teval_dn is prepared for spin non-degenerated solution"
    teval=[]; teval_dn=[]
    "wkpt[ikpt] store the weight of each k-points"
    wkpt=[]
    "tpro[ikpt][ibnd][orbital], orbital index: s, py, pz, px, dxy, dyz, dz2, dxz, dx2"
    tpro=[]; tpro_dn=[]
    '''
    ipro[ikpt][ibnd][iion], store the ion proportion for each state
    '''
    ipro=[]; ipro_dn=[]
    #itpro[ikpt][ibnd][iion][orbital] store the orbital content for each ion
    itpro=[]; itpro_dn=[]
    '''
    spro[ikpt][ibnd](spin direction), spin direction index:
    0: x, 1: y, 2: z
    z direction was defined by the keyword AXIS in INCAR file
    each data is an np.array vector
    '''
    spro=[]; spro_dn=[]
    '''
    phase[ikpt][ibnd][iion][iorb], phase factor for each site orbits
    '''
    phase=[]; phase_dn=[]
    
    '''
    private max_orb: if f orbit is considered the max_orb = 17 
    '''
    max_orb = 10
    """
    def __init__(self) -> None:
        self.nkp=-1
        self.nbnd=-1
        self.nion=-1
        self.spinor=True
        self.orb_f=False
        self.spn_ndg=False
        self.fermi=0.
        self.eval_max=0.0
        self.eval_min=0.0
        self.kpt_path=[]
        self.teval=[]; self.teval_dn=[]
        self.tocc=[]; self.tocc_dn=[]
        self.wkpt=[]
        self.tpro=[]; self.tpro_dn=[]
        self.ipro=[]; self.ipro_dn=[]
        self.itpro=[]; self.itpro_dn=[]
        self.spro=[]; self.spro_dn=[]
        self.phase=[]; self.phase_dn=[]
        self.max_orb = 10
    
    def readPROCAR(self,spinor=True,fermi=0.,phase_cal=True,orb_f=False,rpath='./',directPro=True):
        self.directPro=directPro # if we use the data in PROCAR directly or we perform a percentage mapping?
        self.spinor=spinor
        self.fermi=fermi
        self.orb_f=orb_f
        self.max_orb = 17 if orb_f else 10
        self.rpath=rpath
        self.kpt_path=[]
        
        try:
            with open(os.path.join(rpath,'PROCAR'),'r') as f:
                #skip title line
                f.readline()
                line=re.findall('[0-9.E]+',f.readline())
         
                #get all size
                (self.nkp, self.nbnd, self.nion) = [int(l) for l in line]
        
                #read the rest of the file
                data=f.read()
        except IOError as e:
            msg=str(e)
            raise IOError(msg)

        #split date into lines, ignore blank lines
        data=[entry for entry in data.split('\n') if (entry and entry!=' ')]

        #eval is the total data array, teval[ikpt][ibnd]
        data_index=0
        self.teval, self.tocc, self.tpro, self.ipro, self.itpro, self.spro, self.phase, data_index = \
        self.readdata(data,data_index,fermi,phase_cal)
        #print data_index, len(data)
        if len(data) > 2*data_index:
            self.spn_ndg=True
            data_index+=1
            self.teval_dn, self.tocc_dn, self.tpro_dn, self.ipro_dn, self.itpro_dn, self.spro_dn, self.phase_dn, data_index= \
            self.readdata(data,data_index,fermi,phase_cal)

        print('maximum eigenvalue', self.eval_max)
        print('minimum eigenvalue', self.eval_min)
        f.close()
    
    def readdata(self,data,data_index,fermi,phase_cal):
        r_teval=[]
        r_tocc=[]
        r_tpro=[]
        r_ipro=[]
        r_itpro=[]
        r_spro=[]
        r_phase=[]

        for ikp in range(self.nkp):
            #decomposed data for each k-point:
            #keval store eigenvalues, kpro_tmp for orbital percentages
            #ipro_tmp for ion's proportion, spro_tmp for spinor data
            #phase_tmp for phase factor at each k-point
            keval=[]
            kocc=[]
            kpro_tmp=[]
            ipro_tmp=[]
            itpro_tmp=[]
            spro_tmp=[]
            phase_tmp=[]
            
            if self.spn_ndg:
                data_index+=1
            else:
                data_line=re.findall('\-?[0-9.]+',data[data_index])
            
                _, kx, ky, kz, w =[float(l) for l in data_line]
                self.kpt_path.append([kx,ky,kz])
                self.wkpt.append(w)
                data_index+=1
            
            #data_index+=1
            for ibnd in range(self.nbnd):
                #input energy eigenvalue
                data_line=re.findall('[0-9.E\-]+',data[data_index])
                #print data[data_index]
                #print 'data index:',data_index
                #print data_line
                _, eval_tmp, _, occ_tmp= data_line
                eval_tmp,occ_tmp=float(eval_tmp)-fermi,float(occ_tmp)
                keval.append(eval_tmp)
                kocc.append(occ_tmp)
                data_index+=1
                
                #update max and min eigenvalues
                self.eval_max=eval_tmp if eval_tmp>self.eval_max else self.eval_max
                self.eval_min=eval_tmp if eval_tmp<self.eval_min else self.eval_min
                        
                #get projecting data
                data_index+=1
                #firstly get the ion proportion
                pro_sum=0.
                pro_tmp=[]
                orb_factor=[]
                ion_tmp=0.
                for ipro in range(self.nion):
                    data_line=re.findall('[0-9.]+',data[data_index])
                    orb_factor.append([float(data_line[l]) for l in range(1,self.max_orb)])
                    ion_tmp=float(data_line[self.max_orb])
                    pro_tmp.append(ion_tmp)
                    pro_sum+=ion_tmp
                    data_index+=1
                #for ipro in range(self.nion):
                #    pro_tmp[ipro] = pro_tmp[ipro]/pro_sum if pro_sum>0 else 0
                    
                ipro_tmp.append(pro_tmp)
                itpro_tmp.append(orb_factor)

                if self.nion==1:
                    data_index-=1
                    data_line=re.findall(r'[0-9]+\.[0-9]+',data[data_index])
                    data_index+=1
                else:
                    data_line=re.findall(r'[0-9.E\-]+',data[data_index])
                    data_index+=1
                pro_tmp=[]
                for ipro in range(self.max_orb-1):
                    pro_tmp.append(float(data_line[ipro]))
                #calculate the proportion
                #pro_sum=0.
                #for ipro in range(self.max_orb-1):
                #    pro_sum+=pro_tmp[ipro]
                #pro_sum=1 if self.directPro else pro_sum
                #for ipro in range(self.max_orb-1):
                #    pro_tmp[ipro]= pro_tmp[ipro]/pro_sum if pro_sum>0 else 0
                
                
                kpro_tmp.append(pro_tmp)
                
                #skip rest spinor component data unless spinor case
                if self.spinor:
                    pro_tmp=[]
                    for ispin in range(3):
                        data_index += self.nion
                        data_line=re.findall(r'\-?[0-9.]+',data[data_index])
                        data_index += 1
                        pro_tmp.append(float(data_line[self.max_orb-1]))
                    spro_tmp.append(np.array(pro_tmp))
                    
                #get phase information for phase calculations
                # designed only for vasp 5.4.4 format, the real and imag part are in single line
                # note that in 5.4.1 and below the imag and real part are in neighbor lines, thus
                # is not compactible with this reader 
                if phase_cal:
                    pro_tmp=[]
                    data_index += 1
                    for iion in range(self.nion):
                        data_line=re.findall(r'[0-9.-]+',data[data_index])
                        data_index += 1
                        #data_line_imag=re.findall('[0-9.-]+',data[data_index])
                        #data_index += 1
                        pro_ion_tmp=[]
                        for iorb in range(self.max_orb-1):
                            #note we should skip the first index num, thus iorb+1 not iorb
                            realpart = float(data_line[2*iorb+1])
                            imagpart = float(data_line[2*iorb+2])
                            z=complex(realpart,imagpart)
                            #z=z if abs(z)==0 else orb_factor[iion][iorb]*z/abs(z)
                            pro_ion_tmp.append(z)
                        pro_tmp.append(pro_ion_tmp)
                    phase_tmp.append(pro_tmp)
                    if self.nion>1: data_index+=1
                    
            r_teval.append(keval)
            r_tocc.append(kocc)
            r_tpro.append(kpro_tmp)
            r_ipro.append(ipro_tmp)
            r_itpro.append(itpro_tmp)
            r_spro.append(spro_tmp)
            r_phase.append(phase_tmp)
        
        return np.array(r_teval), np.array(r_tocc), np.array(r_tpro), np.array(r_ipro), np.array(r_itpro), np.array(r_spro), np.array(r_phase), data_index

    """
    def correctPDOS(self,iion_list=[0],emin=-3,emax=2,ngrid=64,localFrame=np.diag([1,1,1])):
        time0=time.time()
        mylib=CDLL("./harmonics.so")
        generateCubicOrbital=mylib.generateCubicOrbital
        generateCubicOrbital.restype=py_object
        original_frame=(c_double*9)()
        transformed_frame=(c_double*9)()
        icnt=0
        invLocal=localFrame
        for i in range(3):
            for j in range(3):
                if i==j:
                    original_frame[icnt]=c_double(1)
                else:
                    original_frame[icnt]=c_double(0)
                transformed_frame[icnt]=c_double(invLocal[i][j])
                icnt+=1

        px_std=generateCubicOrbital(c_int(2), c_int(1), b'x', c_int(ngrid), c_double(80), original_frame)
        py_std=generateCubicOrbital(c_int(2), c_int(1), b'y', c_int(ngrid), c_double(80), original_frame)
        pz_std=generateCubicOrbital(c_int(2), c_int(1), b'z', c_int(ngrid), c_double(80), original_frame)

        dxy_std=generateCubicOrbital(c_int(3), c_int(2), b'xy', c_int(ngrid), c_double(80), original_frame)
        dyz_std=generateCubicOrbital(c_int(3), c_int(2), b'yz', c_int(ngrid), c_double(80), original_frame)
        dxz_std=generateCubicOrbital(c_int(3), c_int(2), b'xz', c_int(ngrid), c_double(80), original_frame)
        dx2_std=generateCubicOrbital(c_int(3), c_int(2), b'x2', c_int(ngrid), c_double(80), original_frame)
        dz2_std=generateCubicOrbital(c_int(3), c_int(2), b'z2', c_int(ngrid), c_double(80), original_frame)
        ao_basis_std=np.array([py_std,pz_std,px_std,dxy_std,dyz_std,dz2_std,dxz_std,dx2_std])
        if self.orb_f:
            fz3_std=generateCubicOrbital(c_int(4), c_int(3), b'z3', c_int(ngrid), c_double(80), original_frame)
            fxz2_std=generateCubicOrbital(c_int(4), c_int(3), b'xz2', c_int(ngrid), c_double(80), original_frame)
            fyz2_std=generateCubicOrbital(c_int(4), c_int(3), b'yz2', c_int(ngrid), c_double(80), original_frame)
            fxyz_std=generateCubicOrbital(c_int(4), c_int(3), b'xyz', c_int(ngrid), c_double(80), original_frame)
            fz_x2_y2_std=generateCubicOrbital(c_int(4), c_int(3), b'z(x2-y2)', c_int(ngrid), c_double(80), original_frame)
            fx_x2_3y2_std=generateCubicOrbital(c_int(4), c_int(3), b'x(x2-3y2)', c_int(ngrid), c_double(80), original_frame)
            fy_3x2_y2_std=generateCubicOrbital(c_int(4), c_int(3), b'y(3x2-y2)', c_int(ngrid), c_double(80), original_frame)
            ao_basis_std=np.array([py_std,pz_std,px_std,dxy_std,dyz_std,dz2_std,dxz_std,dx2_std,
                                   fy_3x2_y2_std,fxyz_std,fyz2_std,fz3_std,fxz2_std,fz_x2_y2_std,fx_x2_3y2_std])

        px_trans=generateCubicOrbital(c_int(2), c_int(1), b'x', c_int(ngrid), c_double(80), transformed_frame)
        py_trans=generateCubicOrbital(c_int(2), c_int(1), b'y', c_int(ngrid), c_double(80), transformed_frame)
        pz_trans=generateCubicOrbital(c_int(2), c_int(1), b'z', c_int(ngrid), c_double(80), transformed_frame)

        dxy_trans=generateCubicOrbital(c_int(3), c_int(2), b'xy', c_int(ngrid), c_double(80), transformed_frame)
        dyz_trans=generateCubicOrbital(c_int(3), c_int(2), b'yz', c_int(ngrid), c_double(80), transformed_frame)
        dxz_trans=generateCubicOrbital(c_int(3), c_int(2), b'xz', c_int(ngrid), c_double(80), transformed_frame)
        dx2_trans=generateCubicOrbital(c_int(3), c_int(2), b'x2', c_int(ngrid), c_double(80), transformed_frame)
        dz2_trans=generateCubicOrbital(c_int(3), c_int(2), b'z2', c_int(ngrid), c_double(80), transformed_frame)
        ao_basis_trans=np.array([py_trans,pz_trans,px_trans,dxy_trans,dyz_trans,dz2_trans,dxz_trans,dx2_trans])

        if self.orb_f:
            fz3_trans=generateCubicOrbital(c_int(4), c_int(3), b'z3', c_int(ngrid), c_double(80), transformed_frame)
            fxz2_trans=generateCubicOrbital(c_int(4), c_int(3), b'xz2', c_int(ngrid), c_double(80), transformed_frame)
            fyz2_trans=generateCubicOrbital(c_int(4), c_int(3), b'yz2', c_int(ngrid), c_double(80), transformed_frame)
            fxyz_trans=generateCubicOrbital(c_int(4), c_int(3), b'xyz', c_int(ngrid), c_double(80), transformed_frame)
            fz_x2_y2_trans=generateCubicOrbital(c_int(4), c_int(3), b'z(x2-y2)', c_int(ngrid), c_double(80), transformed_frame)
            fx_x2_3y2_trans=generateCubicOrbital(c_int(4), c_int(3), b'x(x2-3y2)', c_int(ngrid), c_double(80), transformed_frame)
            fy_3x2_y2_trans=generateCubicOrbital(c_int(4), c_int(3), b'y(3x2-y2)', c_int(ngrid), c_double(80), transformed_frame)
            ao_basis_trans=np.array([py_trans,pz_trans,px_trans,dxy_trans,dyz_trans,dz2_trans,dxz_trans,dx2_trans,
                                   fy_3x2_y2_trans,fxyz_trans,fyz2_trans,fz3_trans,fxz2_trans,fz_x2_y2_trans,fx_x2_3y2_trans])

        transMatrix=np.dot(ao_basis_trans,ao_basis_std.T)
        np.set_printoptions(formatter={'float':'{:.1f}'.format})
        print("rotation matrix constructed in %.3f s"%(time.time()-time0))
        print("check orthorgonality: std basis")
        print(np.dot(ao_basis_std,ao_basis_std.T))
        print("rotated basis:")
        print(np.dot(ao_basis_trans,ao_basis_trans.T))
        time0=time.time()
        
        itpro=list(self.itpro)
        for ikpt in range(self.nkp):
            for ibnd in range(self.nbnd):
                eigenvalue=self.teval[ikpt][ibnd]
                if eigenvalue<emin:
                    continue
                if eigenvalue>emax:
                    break
                for iion in iion_list:
                    wfn=np.dot(self.phase[ikpt][ibnd][iion][1:],transMatrix)
                    itpro[ikpt][ibnd][iion][1:]=(wfn*wfn.conjugate()).real
        self.itpro=itpro
        
        if not self.spn_ndg:
            print("transformation completed in %.3f s"%(time.time()-time0))
            return

        itpro_dn=list(self.itpro_dn)
        for ikpt in range(self.nkp):
            for ibnd in range(self.nbnd):
                eigenvalue=self.teval_dn[ikpt][ibnd]
                if eigenvalue<emin:
                    continue
                if eigenvalue>emax:
                    break
                for iion in iion_list:
                    wfn=np.dot(self.phase_dn[ikpt][ibnd][iion][1:],transMatrix)
                    itpro_dn[ikpt][ibnd][iion][1:]=(wfn*wfn.conjugate()).real
        self.itpro_dn=itpro_dn
        print("transformation completed in %.3f s"%(time.time()-time0))
        return
    """
