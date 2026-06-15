import auxiliary as aux
import paramReader as param
import numpy as np
import fortio, scipy.io

global version
version='0.2'

def readPOSCAR(filePath="./POSCAR"):
    '''
    Read in the POSCAR file and structualize the data in a dictionary\n
    '''
    resultDic={}
    with open(filePath,'r') as fin:
        data=fin.read().split("\n")

    resultDic['lattice']=np.array([[float(x) for x in data[iline].split()] for iline in range(2,5)])
    resultDic['label list']=data[5].split()
    resultDic['nions']=[int(x) for x in data[6].split()]
    resultDic['itype list']=[]
    for itype, num_atom in enumerate(resultDic['nions']):
        resultDic['itype list']+=[itype]*num_atom
    resultDic['tot ions']=sum(resultDic['nions'])
    resultDic['type']=data[7][0]
    iline_shift=0
    if resultDic['type']=='S':
        print(f"POSCAR with Selective dynamics {resultDic['type']}")
        iline_shift=1
        resultDic['type']=data[8][0]
        
    if resultDic['type']=='D':
        resultDic['pos ions (frac.)']=np.loadtxt(data[8+iline_shift:8+iline_shift+resultDic['tot ions']],usecols=(0,1,2,))
        resultDic['pos ions (cart.)']=resultDic['pos ions (frac.)']@resultDic['lattice']
    elif resultDic['type']=='C':
        resultDic['pos ions (cart.)']=np.loadtxt(data[8+iline_shift:8+iline_shift+resultDic['tot ions']],usecols=(0,1,2,))
        resultDic['pos ions (frac.)']=resultDic['pos ions (cart.)']@np.linalg.inv(resultDic['lattice'])
    else:
        print(f"Error: coordinates type should be either D or C, now is {resultDic['type']}")
        exit()
    if len(resultDic['pos ions (frac.)'].shape)==1:
        resultDic['pos ions (frac.)']=resultDic['pos ions (frac.)'].reshape((1,3))
        resultDic['pos ions (cart.)']=resultDic['pos ions (cart.)'].reshape((1,3))
    return resultDic

def readHrDat(filepath,ignoreImagPart=False,overlap=False,format="VASP"):
    aux.showElapsedTime(True)
    with open(filepath,'r') as fin:
        headline=fin.readline()
        print('reading file %s, title line:%s'%(filepath,headline))
        norb=int(fin.readline())
        if format=="QE": assert norb%2==0, "format is set to QE, means we are doing soc case, but norb is odd"

        totWaveVectors=int(fin.readline())
        iline=3
        recordedWaveVectors=0
        while(True):
            waveVectorPack=fin.readline().split()
            recordedWaveVectors+=len(waveVectorPack)
            iline+=1
            if recordedWaveVectors>=totWaveVectors:break
            
    data=np.loadtxt(filepath,skiprows=iline)
    aux.showElapsedTime(extraInfo="%s loaded"%filepath)
    
    nlines, ncols= data.shape
    if overlap:
        assert ncols == 14, f"For LCAO case, we need 14 cols. for R(3) iorb(2) t(2) overlap(1) p(3) r(3), but {ncols=} found"
    else:
        assert ncols == 7, f"For w90 case, we need 7 cols. for R(3) iorb(2) t(2), but {ncols=} found"

    iRvecList = []
    iRvecIndex={}
    nR=0
    for Rvec in np.int16(data[:,:3]):
        n=tuple(Rvec)
        if n not in iRvecList:
            iRvecList.append(n)
            iRvecIndex[n]=nR
            nR+=1
    iRvec=np.array(iRvecList)

    sourceList, targetList = np.int16(data[:,3:5].T)-1
    norbs=sourceList.max()+1
    #t_R = data[:,5] + 1j* data[:,6]
    #ham_R=t_R.reshape((nR,norbs,norbs))
    ham_R=np.zeros((nR,norbs,norbs),dtype=complex)
    for Rvec, (isource, itarget), t in zip(np.int16(data[:,:3]),np.int16(data[:,3:5])-1,data[:,5] + 1j* data[:,6]):
        ham_R[iRvecIndex[tuple(Rvec)],isource,itarget]=t
    if ignoreImagPart: ham_R=ham_R.real
    if overlap:
        o, px, py, pz, rx, ry, rz = data[:,7], data[:,8], data[:,9], data[:,10], data[:,11], data[:,12], data[:,13]
        overlap_R, px_R, py_R, pz_R, rx_R, ry_R, rz_R = o.reshape((nR,norbs,norbs)), px.reshape((nR,norbs,norbs)), py.reshape((nR,norbs,norbs)), pz.reshape((nR,norbs,norbs)), rx.reshape((nR,norbs,norbs)), ry.reshape((nR,norbs,norbs)), rz.reshape((nR,norbs,norbs))
    
    Rx_max, Rx_min, Ry_max, Ry_min, Rz_max, Rz_min= iRvec[:,0].max(), iRvec[:,0].min(), iRvec[:,1].max(), iRvec[:,1].min(), iRvec[:,2].max(), iRvec[:,2].min()
    mp_grid=np.array([Rx_max-Rx_min+1,Ry_max-Ry_min+1,Rz_max-Rz_min+1,])
    
    aux.showElapsedTime(extraInfo='\nreal-space grid: %dx%dx%d, ham_R read'%(*ham_R.shape[:3],))
    if overlap:
        return {
        "ham_R":ham_R,
        "overlap":overlap,
        "overlap_R":overlap_R,
        "momentum_R":np.array([px_R,py_R,pz_R]).transpose((1,0,2,3)), # set 1st index to be the Rvec
        "position_R":np.array([rx_R,ry_R,rz_R]).transpose((1,0,2,3)), # set 1st index to be the Rvec
        "iRvec":iRvec,
        "iRvecIndex":iRvecIndex,
        "mp_grid":mp_grid,
    }    
    return {
        "ham_R":ham_R,
        "iRvec":iRvec,
        "iRvecIndex":iRvecIndex,
        "mp_grid":mp_grid,
    }

def readRDat(filepath,ignoreImagPart=False,cutoff=-1,):
    aux.showElapsedTime(True)
    with open(filepath,'r') as fin:
        headline=fin.readline()
        print('reading file %s, title line:%s'%(filepath,headline))
        norb=int(fin.readline())
            
    data=np.loadtxt(filepath,skiprows=2)
    aux.showElapsedTime(extraInfo="%s loaded"%filepath)
    
    nlines, ncols= data.shape
    assert ncols == 11, f"Position matrix is assumed to be 11 cols., including R(3) iorb(2) rx(2) ry(2) rz(2), but {ncols=} found"

    dRList = -np.int16(data[:,:3])
    sourceList, targetList = np.int16(data[:,3:5].T)-1
    rxList, ryList, rzList = data[:,5] + 1j* data[:,6], data[:,7] + 1j* data[:,8], data[:,9] + 1j* data[:,10]
    rList=np.array([rxList,ryList,rzList]).T
    if ignoreImagPart: 
        rList=rList.real
    
    Rx_max, Rx_min, Ry_max, Ry_min, Rz_max, Rz_min= dRList[:,0].max(), dRList[:,0].min(), dRList[:,1].max(), dRList[:,1].min(), dRList[:,2].max(), dRList[:,2].min()
    r_R=np.zeros((Rx_max-Rx_min+1,Ry_max-Ry_min+1,Rz_max-Rz_min+1,3,norb,norb),dtype=complex)

    dRSet=set()
    for ibond, (dR, source, target, r) in enumerate(zip(dRList,sourceList,targetList,rList)):
        if(ibond+1)%((nlines//50+1))==0: print('>',end='') # process bar

        # skip negligible bond
        if np.sum(abs(r)) < cutoff: continue
        # add bond
        r_R[dR[0],dR[1],dR[2],:,source,target]=r

        dRSet.add(tuple(dR))
        
    aux.showElapsedTime(extraInfo='\nreal-space grid: %dx%dx%d, ham_R read'%(*r_R.shape[:3],))
    return {
        "r_R":r_R,
        "dRList":dRSet
    }

def readJ(mainDir, bondList):
    global version
    fileList=aux.ls(mainDir)
    if 'J.txt' in fileList:
        with open(mainDir+'J.txt','r') as fJ:
            data=[line for line in  fJ.read().split('\n') if line]
            if len(data)==0:
                print('File info: J.txt is empty')
                return False
            headline=data[0].split()
            if len(headline)==0:
                print('File info: head line is empty')
                return False
            versionInFile=headline[len(headline)-1]
            if versionInFile!=version:
                print('File info: version does not match')
                return False
            if (len(data)-3)!=sum([len(bond) for bond in bondList]):
                print('File info: exchange truncation dose not match %d/%d'%(len(data)-3,len(bondList)))
                return False
            if len(data[3].split())==8: # isotropic exchange, only one ele of J
                for iline in range(3,len(data)):
                    NN,  distance,  sourceID,  targetID,  Rx,  Ry,  Rz,  J = data[iline].split()
                    iBondType=iline-3
                    bond=bondList[iBondType][0]
                    if bond.sourceOrb.id==int(sourceID) and bond.targetOrb.id==int(targetID) and bond.overLat[0]==int(Rx) and bond.overLat[1]==int(Ry) and bond.overLat[2]==int(Rz):
                        bond.exchangeEnergy=float(J)
                    else:
                        print('File info: bond does not match ion%d-ion%d [%d %d %d] <-> ion%s-ion%s [%s %s %s]'%(
                            bond.sourceOrb.id,bond.targetOrb.id,*bond.overLat,sourceID,targetID,Rx,Ry,Rz))
                        return False
            elif len(data[3].split())==13: # anisotropic exchange
                tag=np.array([False for i in range(len(data)-3)])
                _ibond_abs=0
                for ibondType, bondType in enumerate(bondList):   # scan all bonds should be addressed
                    for ibond_inMem, bond in enumerate(bondType): # scan all bonds should be addressed
                        for ibond_fromTXT in range(len(data)-3):  # scan txt for the data
                            NN,  distance,  sourceID,  targetID,  Rx,  Ry,  Rz,  J, J_a_xx, J_a_yy, J_a_xy, Dx, Dy = data[3+ibond_fromTXT].split()
                            if bond.sourceOrb.id==int(sourceID) and bond.targetOrb.id==int(targetID) and bond.overLat[0]==int(Rx) and bond.overLat[1]==int(Ry) and bond.overLat[2]==int(Rz):
                                bond.exchangeEnergy=float(J)
                                bond.Jani=np.array([[float(J_a_xx),float(J_a_xy)],[float(J_a_xy),float(J_a_yy)]])
                                bond.DMI=np.array([float(Dx),float(Dy)])
                                tag[_ibond_abs]=True
                                data.pop(3+ibond_fromTXT)
                                _ibond_abs+=1
                                break
                        else:
                            print('File info: cannot find the info. of bond NN%d (bond%d) ion%d-ion%d [%d %d %d]'%(
                                ibondType,ibond_inMem,bond.sourceOrb.id,bond.targetOrb.id,*bond.overLat))
                            return False
                if not tag.all():
                    print("Failed to address all bonds",tag)
                    return False
        return True
    print('File info: J.txt does not exist')
    return False
            
def writeJ(mainDir, bondList, supercellSize):
    global version
    with open(mainDir+'J.txt','w') as fJ:
        fJ.write('Isotropic exchange energy J*S^2 up to %dNN calculated by Greens function. version: %s\n'%(param.CALRANGE,version))
        fJ.write('evaluated in %dx%dx%d supercell\n'%(*supercellSize,))
        fJ.write('#NN  distance  sourceID  targetID  Rx  Ry  Rz  nJ  J\n')

        for ibondType, bondType in enumerate(bondList):
            bond=bondType[0]
            fJ.write('%3d%12.6f%3d%3d%3d%3d%3d%3d%12.6f\n'%(
                (ibondType+1), bond.bondLen, bond.sourceOrb.id, bond.targetOrb.id, *bond.overLat, len(bondType), bond.exchangeEnergy))

def writeJmat(mainDir, bondList, supercellSize):
    global version
    with open(mainDir+'J.txt','w') as fJ:
        fJ.write('Anisotropic exchange energy J*S^2 up to %dNN calculated by Greens function. version: %s\n'%(param.CALRANGE,version))
        fJ.write('evaluated in %dx%dx%d supercell\n'%(*supercellSize,))
        fJ.write('#Fermi          NN   dis.  is  it  Rx  Ry  Rz             J          Ja_x          Ja_y         Ja_yx            Dx            Dy\n')

        for ibondType, bondType in enumerate(bondList):
            for bond in bondType:
                nfermi=len(bond.exchangeEnergy)
                for ifermi in range(nfermi):
                    fJ.write('%14.6e%3d%7.3f%4d%4d%4d%4d%4d%14.6e%14.6e%14.6e%14.6e%14.6e%14.6e\n'%(
                        bond.fermi[ifermi], (ibondType+1), bond.bondLen, bond.sourceOrb.id, bond.targetOrb.id, *bond.overLat,  
                        bond.exchangeEnergy[ifermi],bond.Jani[ifermi][0,0],bond.Jani[ifermi][1,1],bond.Jani[ifermi][0,1],bond.DMI[ifermi][0],bond.DMI[ifermi][1]))

def writeJmatV2(mainDir, exch, supercellSize):
    global version
    with open(mainDir+'J.txt','w') as fJ:
        fJ.write(f'Anisotropic exchange energy J*S^2, calculated by Greens function. version: {version}\n')
        fJ.write('evaluated in %dx%dx%d supercell\n'%(*supercellSize,))
        fJ.write('#Fermi          NN   dis.  is  it  Rx  Ry  Rz             J          Ja_x          Ja_y         Ja_yx            Dx            Dy            Dz\n')

        for iroute, route in enumerate(exch.exchangeRoutes):
            for ifermi, fermi in enumerate(exch.fermiList):
                fJ.write('%14.6e%3d%7.3f%4d%4d%4d%4d%4d%14.6e%14.6e%14.6e%14.6e%14.6e%14.6e%14.6e\n'%(
                    fermi, (iroute+1), route["bond_length"], route["source_atom"], route["target_atom"], *route["overlat"],  
                    exch.exchangeEnergy[ifermi,iroute],
                    exch.Jani[ifermi,iroute,0,0],exch.Jani[ifermi,iroute,1,1],exch.Jani[ifermi,iroute,0,1],
                    *exch.DMI[ifermi,iroute,:]))

def writeVolumeData(filePath:str,lat: list | np.ndarray,grid_size:tuple | list | np.ndarray,grid_data:np.ndarray):
    '''
    output volume data in the format of CHGCAR

    Parameters
    ----------
    - filePath : str 
        the target file path
    - lat : array-like, float
        lattice matrix, row major
    - grid_size : size-3 tuple-like, int
        grid size Nx, Ny, Nz in a1, a2, a3 direction
    - grid_data : np.ndarray((Nx,Ny,Nz)), float
        volume data to be outputed
    '''
    assert tuple(grid_size) == grid_data.shape
    with open(filePath,"w") as fout:
        fout.write("Automatic generated volume data\n")
        fout.write("1.0\n")
        for ib in range(3):
            for icomp in range(3): fout.write("%.9f "%lat[ib,icomp])
            fout.write("\n")
        fout.write("H\n")
        fout.write("0\n")
        fout.write("D\n")
        fout.write("\n")
        fout.write("\n")
        fout.write("%5d%5d%5d\n"%(*grid_size,))
        cnt=0
        for ik3 in range(grid_size[2]):
           for ik2 in range(grid_size[1]):
              for ik1 in range(grid_size[0]):
                  fout.write("%18.9E"%grid_data[ik1,ik2,ik3])
                  cnt+=1
                  if cnt%5==0:fout.write("\n")

def output3rdFreqDenpendentData(tensor_data,fermi,fermiAxis,omegaAxis,filePath):
    '''
    Output the fermi-denpendent frequency-dependent 3rd order tensor data

    Parameters
    ----------
    - tensor_data : np.ndarray
        The data to be output
    - fermi : float
        Fermi energy
    - fermiAxis : np.ndarray
        Fermi axis, absolute values, should have the same size with the 4th axis of tensor_data
    - omegaAxis : np.ndarray
        Angle frequency axis, should have the same size with the 5th axis of tensor_data
    - filePath : path
        Position of the file to be output
    '''
    with open("%s"%filePath,"w") as fout:
        fout.write("Omega(eV) Fermi(eV) ")
        directionName=['x','y','z']
        for iproj1 in range(tensor_data.shape[0]):
            for iproj2 in range(tensor_data.shape[1]):
                for iproj3 in range(tensor_data.shape[2]):
                    for alpha1 in range(3):
                        for alpha2 in range(3):
                            for beta in range(3):
                                fout.write(f"            {directionName[alpha1]}{directionName[alpha2]}{directionName[beta]}")
        fout.write("\n")
        for iomega, omega in enumerate(omegaAxis):
            fout.write(f"{omega:9.6f}")
            for ifermi, fermiEnergy in enumerate(fermiAxis):
                fout.write(f"{fermiEnergy-fermi:9.6f}")
                for iproj1 in range(tensor_data.shape[0]):
                    for iproj2 in range(tensor_data.shape[1]):
                        for iproj3 in range(tensor_data.shape[2]):
                            for alpha1 in range(3):
                                for alpha2 in range(3):
                                    for beta in range(3):
                                        fout.write(f"{tensor_data[iproj1,iproj2,iproj3,ifermi,iomega,alpha1,alpha2,beta]:15.6E}")
                fout.write("\n")

def outputBZDistributionData(tensor_data, filePath, reciLattice, componentLabels=None,
                             bzRepeat=(0, 0, 0)):
    '''
    Output the k-resolved distribution of a tensor response on the Brillouin zone.

    Parameters
    ----------
    - filePath : str
        Target output file.
    - reciLattice : np.ndarray(3, 3)
        Reciprocal lattice matrix. The first three axes of tensor_data are
        interpreted as the regular BZ grid, and k = reciLattice @ frac_k.
    - tensor_data : np.ndarray, shape (Nk1, Nk2, Nk3, 3, ..., 3)
        Tensor data at each k-point. Rank-2 and rank-3 tensors are the usual
        use cases, but any positive tensor rank is accepted.
    - componentLabels : list[str] | None
        Optional axis labels. Defaults to ['x', 'y', 'z'].
    - bzRepeat : int | tuple(3, int)
        Number of reciprocal periods to write around Gamma. For example,
        bzRepeat=(1, 1, 0) writes a 2D map from -G1,-G2 to +G1,+G2
        with Gamma at the center. The default (0,0,0) preserves the one-BZ
        output.
    '''
    tensor_data = np.asarray(tensor_data)
    reciLattice = np.asarray(reciLattice)
    assert reciLattice.shape == (3, 3), f"expect reciLattice shape (3, 3) but {reciLattice.shape=}"
    assert tensor_data.ndim >= 4, f"expect tensor_data shape (Nk1, Nk2, Nk3, components...) but {tensor_data.shape=}"
    kgrid = tensor_data.shape[:3]

    tensor_shape = tensor_data.shape[3:]
    assert len(tensor_shape) > 0, "expect at least one tensor/component axis"
    if componentLabels is None:
        componentLabels = ['x', 'y', 'z']
    if np.isscalar(bzRepeat):
        bzRepeat = (int(bzRepeat), int(bzRepeat), int(bzRepeat))
    else:
        bzRepeat = tuple(int(x) for x in bzRepeat)

    component_indices = list(np.ndindex(tensor_shape))
    component_names = [
        ''.join(componentLabels[index] if index < len(componentLabels) else str(index)
                for index in component_index)
        for component_index in component_indices
    ]
    is_complex = np.iscomplexobj(tensor_data)

    with open(filePath, "w") as fout:
        fout.write("kx ky kz")
        for component_name in component_names:
            if is_complex:
                fout.write(f"            Re_{component_name}            Im_{component_name}")
            else:
                fout.write(f"            {component_name}")
        fout.write("\n")

        axis_ranges = []
        for iaxis, ngrid in enumerate(kgrid):
            if bzRepeat[iaxis] > 0:
                axis_ranges.append(range(-bzRepeat[iaxis] * ngrid, bzRepeat[iaxis] * ngrid + 1))
            else:
                axis_ranges.append(range(ngrid))

        for ik_extended in np.ndindex(tuple(len(axis_range) for axis_range in axis_ranges)):
            fractional_index = np.array([axis_ranges[iaxis][ik_extended[iaxis]] for iaxis in range(len(kgrid))])
            ik = tuple(int(fractional_index[iaxis] % kgrid[iaxis]) for iaxis in range(len(kgrid)))
            frac = np.zeros(3)
            for iaxis, ngrid in enumerate(kgrid):
                frac[iaxis] = fractional_index[iaxis] / ngrid
            kpoint = reciLattice @ frac
            fout.write(f"{kpoint[0]:15.6E}{kpoint[1]:15.6E}{kpoint[2]:15.6E}")
            for component_index in component_indices:
                value = tensor_data[ik + component_index]
                if is_complex:
                    fout.write(f"{value.real:15.6E}{value.imag:15.6E}")
                else:
                    fout.write(f"{value:15.6E}")
            fout.write("\n")

class FortranFileR(fortio.FortranFile):

    def __init__(self, filename):
        print("using fortio to read")
        try:
            super().__init__(filename, mode='r', header_dtype='uint32', auto_endian=True, check_file=True)
        except ValueError:
            print("File '{}' contains subrecords - using header_dtype='int32'".format(filename))
            super().__init__(filename, mode='r', header_dtype='int32', auto_endian=True, check_file=True)

class FortranFileW(scipy.io.FortranFile):

    def __init__(self, filename):
        print("using scipy.io to write")
        super().__init__(filename, mode='w')

class WAVECAR:

    def __init__(self,filename='./WAVECAR') -> None:
        try:
            WAV = open(filename, "rb")
        except IOError as e:
            print(e)
            exit()
        
        RECL = 3
        def record(irec, cnt=np.Inf, dtype=float):
            WAV.seek(irec * RECL)
            return np.fromfile(WAV, dtype=dtype, count=min(RECL, cnt))

        RECL, self.ispin, iprec = [int(x) for x in record(0)]
        assert iprec == 45200, 'Double precise WAVECAR is not supported by now, iprec=%d'%iprec

        #print(RECL, self.ispin, iprec)

        self.nkpts, self.nbands = [int(x) for x in record(1, 2)]
        print(f"nkpts: {self.nkpts}, nbands: {self.nbands}, ispin: {self.ispin}, rec len: {RECL}")

        self.uk=[]                                                     # wave functions (on plane wave basis)
        self.kpt_f=np.zeros((self.nkpts,3),dtype=float)                # k-points in fractional coordinates
        self.energies=np.zeros((self.nkpts,self.nbands),dtype=float)   # band energies
        self.occupies=np.zeros((self.nkpts,self.nbands),dtype=float)   # band occupation
        for ik in range(self.nkpts):
            #npw = int(record(2 + ik * (self.nbands + 1), 1))
                                                    # npw, kpt, energy_cplx,  occ
            headBlock = record(2 + ik * (self.nbands + 1), 1 + 3 + self.nbands*2 + self.nbands, dtype=float)
            npw=int(headBlock[0])
            self.kpt_f[ik,:]=headBlock[1:4]
            self.energies[ik,:]=headBlock[4::3]
            self.occupies[ik,:]=headBlock[6::3]
            #if npw%2 != 0:
            #    raise RuntimeError(f"odd number of coefs {npw}")
            #print("k-point {0:3d} : {1:6d} plane waves".format(ik, npw))
            #npw12=npw//2
            #print(record(2 + ik * (self.nbands + 1), 10, dtype=float))

            uk = np.zeros((self.nbands, npw), dtype=complex)
            for iband in range(self.nbands):
                uk[iband, :] = record(3 + ik * (self.nbands + 1) + iband, npw, np.complex64)
            self.uk.append(uk)
            #overlap = WF.conj().T.dot(WF)
            #print(np.diag(abs(overlap)))
            #assert np.max(np.abs(overlap - overlap.T.conj())) < 1e-15

            #if normalize == "norm":
            #    WF = WF / np.sqrt(np.abs(overlap.diagonal()))

            #SIGMA = np.array(
            #    [
            #        [
            #            np.einsum("ki,kj->ij",
            #                      WF.conj()[npw12 * i:npw12 * (i + 1), :], WF[npw12 * j:npw12 * (j + 1), :])
            #            for j in (0, 1)
            #        ] for i in (0, 1)
            #    ])
            #SX = SIGMA[0, 1] + SIGMA[1, 0]
            #SY = -1.j * (SIGMA[0, 1] - SIGMA[1, 0])
            #SZ = SIGMA[0, 0] - SIGMA[1, 1]
            #A = np.array([s[n, m] for m in range(NB) for n in range(m + 1) for s in (SX, SY, SZ)], dtype=np.complex128)

    def resortEigs(self,wave_ref):
        assert wave_ref.nkpts == self.nkpts, 'inconsistent nkpts: wave_ref %d wave_self %d'%(wave_ref.nkpts,self.nkpts)
        assert wave_ref.nbands == self.nbands, 'inconsistent nbands: wave_ref %d wave_self %d'%(wave_ref.nbands,self.nbands)

        for ikpt in range(self.nkpts):
            assert wave_ref.uk[ikpt].shape[1] == self.uk[ikpt].shape[1], 'inconsistent num. of plane-wave basis on kpt%d: wave_ref %d wave_self %d'%(ikpt,wave_ref.uk[ikpt].shape[1],self.uk[ikpt].shape[1])
            overlap=abs(wave_ref.uk[ikpt].conj() @ self.uk[ikpt].T)
            match_index=np.argsort(overlap, axis=1)[:,-1]
            assert len(match_index) == len(set(match_index)), 'multi-covering at kpt%d'%ikpt
            # updating indexes
            self.energies[ikpt,:]=self.energies[ikpt,match_index]
            self.occupies[ikpt,:]=self.occupies[ikpt,match_index]
            self.uk[ikpt][:,:]=self.uk[ikpt][match_index,:]

class W90EIG:
    def __init__(self,seedname='wannier90') -> None:
        filename=seedname+'.eig'
        data=np.loadtxt(filename)
        self.nbands=int(max(data[:,0]))
        self.nkpts=int(max(data[:,1]))
        print(f"energy file {filename} has been read, including:")
        print(f"nbands: {self.nbands}, nkpts: {self.nkpts}")
        #data = data.reshape(self.nkpts, self.nbands, 3)
        self.data=data[:,2].reshape(self.nkpts, self.nbands)#data[:,:,2]

class W90MMN:
    
    def __init__(self,seedname:str,kpt_list:np.ndarray,reci_lat:np.ndarray) -> None:
        r'''
        Read in seedname.mmn file, including the data:

        $M_{mn,b}^{(ab initio) k} = \langle u_{km} | u_{k+b,n} \rangle$

        which is needed for further calcs. on the Berry connection (DFT-gauge):

        $A_{mn,\alpha}^{(ab initio) k} = \langle u_{km} | i\partial_{k_\alpha} u_{kn} \rangle$\n
    
        Finite diffenrece is used to approach the partial derivates (gradients to k):

        $A_{mn,\alpha}^{(ab initio) k} = i \sum w_b b_\alpha (\langle u_{km} | u_{k+bn} \rangle - \delta_{mn})$\n
        
        See the eq. C14 [PRB 56, 12847 (1997)]

        Parameters
        ----------
        - seedname:
            the seedname of w90 files e.g., "wannier90"
        - kpt_list:
            k-points in fractional coordinates, shape (nkpts,3)
        - reci_lat:
            reciprocal lattice matrix, shape (3,3)
    
        Returns
        -------
        - self.Mmn[ ikpt, ib, ibands, ibands]: 
            overlaps between eig.-states at neighbored kpts, \langle u_{km} | u_{k+b,n} \rangle
        - A_DFT[ ikpt, idirection, ibands, ibands]: 
            Berry connections in reciprocal, A_{mn,\alpha}^(ab initio)(k)
        '''
        filename=seedname+'.mmn'
        
        with open(filename,'r') as fin:
            data=fin.read().split("\n")
        info=[int(x) for x in data[1].split()]
        self.nbands,self.nkpts,self.nntot=info[:3]
        assert kpt_list.shape == (self.nkpts,3)
        assert reci_lat.shape == (3,3)

        self.Mmn  =np.zeros((self.nkpts,self.nntot,self.nbands,self.nbands),dtype=complex)
        self.AA_DFT=np.zeros((self.nkpts,3,self.nbands,self.nbands),dtype=complex)
        
        b_block_len=1+self.nbands*self.nbands
        k_block_len=self.nntot*b_block_len
        
        info=np.loadtxt(data[2:self.nkpts*k_block_len:b_block_len],dtype=int)
        self.innkpt=info[:,1].reshape((self.nkpts,self.nntot))-1
        k1,k2,g1,g2,g3=info.T
        kpt1=kpt_list[k1-1]
        kpt2=kpt_list[k2-1]+np.array([g1,g2,g3]).T
        self.bvecs=(kpt2-kpt1).reshape((self.nkpts,self.nntot,3))@reci_lat
        self.wbvecs=self.__getWb(data[2:k_block_len:b_block_len],kpt_list,reci_lat)
        
        for ikpt in range(self.nkpts):
            overlap_data=np.loadtxt(data[2+ikpt*k_block_len:2+(ikpt+1)*k_block_len],usecols=(0,1))
            for ib in range(self.nntot):
                __overlap=overlap_data[1+ib*b_block_len:(ib+1)*b_block_len]
                overlap=(__overlap[:,0]+1j*__overlap[:,1]).reshape((self.nbands,self.nbands))
                self.Mmn[ikpt,ib,...]=overlap.T
        
        print(f"nbands: {self.nbands}, nkpts: {self.nkpts}, nntot: {self.nntot}")
        
    def __getWb(self,b_series:list,kpt_list:np.ndarray,reci_lat:np.ndarray):
        '''
        Calculate the weights for each b-vector, using Eq.B1 [PRB 56, 12847 (1997)]
        '''
        ik1, ik2, g1, g2, g3= np.loadtxt(b_series,dtype=int).T
        bList=(kpt_list[ik2-1]+np.array([g1,g2,g3]).T-kpt_list[ik1-1])@reci_lat
        wbList=np.zeros(self.nntot,dtype=float)
        bLenList=np.sum(bList*bList,axis=1)

        # classify the b-vec via length
        bLen0=bLenList[0]
        seperators=[0] # the index for b-sector which involves b-vecs of the same length
        for ibLen in range(1,self.nntot):
            bLen1=bLenList[ibLen]
            if (bLen1-bLen0)/bLen0>0.1:
                seperators.append(ibLen)
                bLen0=bLen1
        seperators.append(self.nntot)

        # work out the wb using eq. B1 [PRB 56, 12847 (1997)]
        factor=np.array([sum(bList[seperators[igap-1]:seperators[igap]]**2) for igap in range(1,len(seperators))]).T
        wb_kind=np.linalg.pinv(factor)@np.ones(3)
        for igap in range(1,len(seperators)):
            wbList[seperators[igap-1]:seperators[igap]]=wb_kind[igap-1]
        return wbList
    
class W90CHK:
    
    def __init__(self,seedname) -> None:
        '''
        Wannier Model\n
        To process the file seedname.chk to obtain gauge transformation (from Bloch eigenstates to MLWF basis) and achieving reversal gauge transformation
        Two useful functions are provided:

        Methods
        -------
        - wannier_gauge:  
            transform quantity in Hamitonian (eigen) gauge into Wannier gauge
        - get_HH_q:  
            transform eigenvalues on k-grid into Wannier gauge (hopping elements on k-grid)
        - get_AAW_q:  
            compute Berry connection on k-grid with Wannier gauge (position operator on k-grid)

        Properties
        ----------
        - win_min: 
            the lowest DFT band index used for Wannier constructions
        - win_max: 
            the highest DFT band index PLUS ONE used for Wannier constructions, so that the win_min:win_max gives the piece of band indexes used
        '''
        
        FIN = FortranFileR(seedname + '.chk')
        def readstr(): "".join(c.decode('ascii') for c in FIN.read_record('c')).strip()
        def readint(): return FIN.read_record('i4')
        def readfloat(): return FIN.read_record('f8')
        def readcomplex():
            a = readfloat()
            return a[::2] + 1j * a[1::2]

        print('Reading restart information from file ' + seedname + '.chk :')
        comment = readstr()
        self.num_bands = readint()[0]
        num_exclude_bands = readint()[0]
        self.exclude_bands = readint()
        print(f"head line:{comment}\nnbands:{self.num_bands}\nnexcludeBands:{num_exclude_bands}\nexclude_bands:{self.exclude_bands}")
        self.real_lattice = readfloat().reshape((3, 3), order='F')
        self.recip_lattice = readfloat().reshape((3, 3), order='F')
        print(f"real lattice:\n{self.real_lattice}")
        print(f"reciprocal lattice:\n{self.recip_lattice}")

        self.num_kpts:int = readint()[0]
        self.mp_grid = readint()
        self.kpt_list:np.ndarray = readfloat().reshape((self.num_kpts, 3))
        dk=1/self.mp_grid
        self.kpt_index = np.array([np.around(kpt/dk) for kpt in self.kpt_list],dtype=int)
        self.nntot = readint()[0]
        self.num_wann = readint()[0]
        #print(f"nkpts:{self.num_kpts}\nself.mp_grid:{self.mp_grid}\nkpt_index:\n{self.kpt_index.shape}\nnntot:{self.nntot}\nnwann:{self.num_wann}")

        checkpoint = readstr()
        #print(checkpoint)
        have_disentangled = bool(readint()[0])
        #print(f"have_disentangled:{have_disentangled}")
        if have_disentangled:
            omega_invariant = readfloat()[0]
            #print(f"omega_invariant:{omega_invariant}")
            lwindow = np.array(readint().reshape((self.num_kpts, self.num_bands)), dtype=bool)
            ndimwin = readint()
            #print(f"lwindow:\n{lwindow}\nndimwin:\n{ndimwin}")
            u_matrix_opt = readcomplex().reshape((self.num_kpts, self.num_wann, self.num_bands))
            self.win_min = np.array([np.where(lwin)[0].min() for lwin in lwindow])
            self.win_max = np.array([wm + nd for wm, nd in zip(self.win_min, ndimwin)])
        else:
            self.win_min = np.array([0] * self.num_kpts)
            self.win_max = np.array([self.num_wann] * self.num_kpts)
        #print(f"self.win_min:{self.win_min}\nself.win_max:{self.win_max}")

        u_matrix:np.ndarray = readcomplex().reshape((self.num_kpts, self.num_wann, self.num_wann))
        m_matrix:np.ndarray = readcomplex().reshape((self.num_kpts, self.nntot, self.num_wann, self.num_wann))
        
        if have_disentangled:
            self.v_matrix = np.array([u@u_opt[:, :nd] for u, u_opt, nd in zip(u_matrix, u_matrix_opt, ndimwin)])
        else:
            self.v_matrix = np.array([u for u in u_matrix])
        self.wannier_centers:np.ndarray = readfloat().reshape((self.num_wann, 3))
        self.wannier_spreads:np.ndarray = readfloat().reshape((self.num_wann))
        del u_matrix, m_matrix
        print(f"self.v_matrix:{self.v_matrix.shape}")

class DOSCAR:
    def __init__(self,filePath='DOSCAR'):
        '''
        For now, only use to read the Fermi energy from the DOSCAR
        '''
        with open(filePath,"r") as fin:
            data=fin.read().split("\n")
        emax, emin, NEDOS, fermi, _ = data[5].split()
        self.emax, self.emin, self.fermi = float(emax), float(emin), float(fermi)
        self.NEDOS=int(NEDOS)
        print(f"DOSCAR in {filePath} has been read.")
        print(f"Energy window: {self.emin} - {self.emax}")
        print(f"Sampling grid points: {self.NEDOS=}")
        print(f"{self.fermi=}")
        
