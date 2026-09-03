import os
import auxiliary as aux
import paramReader as param
import numpy as np
import fortio, scipy.io, h5py
import warnings

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

def _mpGridFromIRvec(iRvec):
    if len(iRvec) == 0:
        raise ValueError("Cannot build mp_grid from an empty R-vector list")
    Rx_max, Rx_min = iRvec[:,0].max(), iRvec[:,0].min()
    Ry_max, Ry_min = iRvec[:,1].max(), iRvec[:,1].min()
    Rz_max, Rz_min = iRvec[:,2].max(), iRvec[:,2].min()
    return np.array([Rx_max-Rx_min+1, Ry_max-Ry_min+1, Rz_max-Rz_min+1,])

def _applyHrCutoff(hrDat, cutoff=-1, overlap=False):
    if cutoff is None or cutoff < 0:
        return hrDat

    nRvec = len(hrDat["iRvec"])
    ham_amp = np.max(np.abs(hrDat["ham_R"]), axis=(1,2))
    if overlap:
        overlap_amp = np.max(np.abs(hrDat["overlap_R"]), axis=(1,2))
        keep = (ham_amp >= cutoff) | (overlap_amp >= cutoff)
    else:
        keep = ham_amp >= cutoff

    if not np.any(keep):
        raise ValueError(f"cutoff={cutoff} removes all real-space Hamiltonian blocks")
    if np.all(keep):
        return hrDat

    hrDat = hrDat.copy()
    for key, value in list(hrDat.items()):
        if isinstance(value, np.ndarray) and value.shape[:1] == (nRvec,):
            hrDat[key] = value[keep]

    hrDat["iRvecIndex"] = {tuple(Rvec): iR for iR, Rvec in enumerate(hrDat["iRvec"])}
    hrDat["mp_grid"] = _mpGridFromIRvec(hrDat["iRvec"])
    print(f"cutoff={cutoff} keeps {np.count_nonzero(keep)}/{nRvec} real-space Hamiltonian blocks")
    return hrDat

def readHrDat(filepath,ignoreImagPart=False,overlap=False,format="VASP",cutoff=-1,
              expectedNorbs=None,requireUnitDegeneracies=False,requireReal=False,
              requireReciprocal=False,requireComplete=False,allowedRvecs=None):
    """Read one hr-format file, including its Wigner-Seitz degeneracies.

    ``ham_R`` retains the historical raw file entries; degeneracy factors are
    returned as ``Ndegen`` and are not applied implicitly. The optional checks
    are useful for model coupling files while preserving the old Hamiltonian
    reader defaults. ``requireComplete`` requires exactly one matrix entry for
    every (R, source, target) tuple. ``allowedRvecs`` optionally restricts the
    translations to a supplied real-space grid.
    """
    aux.showElapsedTime(True)
    with open(filepath,'r') as fin:
        headline=fin.readline()
        print('reading file %s, title line:%s'%(filepath,headline))
        try:
            norb=int(fin.readline())
            totWaveVectors=int(fin.readline())
        except ValueError as error:
            raise ValueError(f"Invalid orbital/R-count header in {filepath}") from error
        if norb <= 0 or totWaveVectors <= 0:
            raise ValueError(f"Orbital and R-vector counts must be positive in {filepath}")
        if format=="QE" and norb%2:
            raise ValueError("format='QE' requires an even spinor orbital count")
        if expectedNorbs is not None and norb != int(expectedNorbs):
            raise ValueError(f"{filepath} contains {norb} orbitals; expected {int(expectedNorbs)}")

        degeneracies=[]
        while len(degeneracies) < totWaveVectors:
            line=fin.readline()
            if not line:
                raise ValueError(f"Incomplete Wigner-Seitz degeneracy header in {filepath}")
            try:
                degeneracies.extend(int(value) for value in line.split())
            except ValueError as error:
                raise ValueError(f"Invalid Wigner-Seitz degeneracy in {filepath}") from error
        if len(degeneracies) != totWaveVectors:
            raise ValueError(f"Expected {totWaveVectors} Wigner-Seitz degeneracies in {filepath}, got {len(degeneracies)}")
        Ndegen=np.asarray(degeneracies,dtype=int)
        if np.any(Ndegen <= 0):
            raise ValueError(f"Wigner-Seitz degeneracies must be positive in {filepath}")
        if requireUnitDegeneracies and np.any(Ndegen != 1):
            raise ValueError(f"{filepath} requires one unit degeneracy per R block")
        try:
            data=np.loadtxt(fin)
        except ValueError as error:
            raise ValueError(f"Invalid matrix data in {filepath}: {error}") from error
    aux.showElapsedTime(extraInfo="%s loaded"%filepath)

    data=np.atleast_2d(data)
    if data.size == 0 or data.shape[1] == 0 or not np.all(np.isfinite(data)):
        raise ValueError(f"Matrix data in {filepath} must be nonempty and finite")
    nlines, ncols=data.shape
    if overlap:
        if ncols != 14:
            raise ValueError(f"For LCAO case, expected 14 columns, but found {ncols} in {filepath}")
    else:
        if ncols != 7:
            raise ValueError(f"For hr case, expected 7 columns, but found {ncols} in {filepath}")

    integer_columns=data[:,:5]
    if not np.all(integer_columns == np.rint(integer_columns)):
        raise ValueError(f"R vectors and orbital indices must be integers in {filepath}")

    iRvecList = []
    iRvecIndex={}
    nR=0
    for Rvec in np.asarray(data[:,:3],dtype=int):
        n=tuple(Rvec)
        if n not in iRvecIndex:
            iRvecList.append(n)
            iRvecIndex[n]=nR
            nR+=1
    if nR != totWaveVectors:
        raise ValueError(f"Header declares {totWaveVectors} R blocks, but matrix data contains {nR} in {filepath}")
    iRvec=np.asarray(iRvecList,dtype=int)
    if allowedRvecs is not None:
        allowed={tuple(np.asarray(Rvec,dtype=int)) for Rvec in allowedRvecs}
        for Rvec in iRvec:
            if tuple(Rvec) not in allowed:
                raise ValueError(f"Coupling R={tuple(Rvec)} is absent from the Hamiltonian iRvec grid")

    sourceList, targetList=np.asarray(data[:,3:5].T,dtype=int)-1
    if (sourceList.min() < 0 or targetList.min() < 0
            or sourceList.max() >= norb or targetList.max() >= norb):
        raise ValueError(f"Orbital indices in {filepath} must lie between 1 and {norb}")
    records=[(iRvecIndex[tuple(R)], int(i), int(j))
             for R, i, j in zip(np.asarray(data[:,:3],dtype=int),sourceList,targetList)]
    if requireComplete:
        unique_records=set(records)
        expected_records=nR*norb*norb
        if len(records) != expected_records or len(unique_records) != expected_records:
            raise ValueError(f"{filepath} must contain exactly one entry for every R/orbital pair")
    values=data[:,5]+1j*data[:,6]
    if requireReal and np.max(np.abs(values.imag)) > 1e-12:
        raise ValueError(f"Matrix entries in {filepath} must be real")
    ham_R=np.zeros((nR,norb,norb),dtype=complex)
    for (iR,isource,itarget), t in zip(records,values):
        ham_R[iR,isource,itarget]=t
    if requireReciprocal:
        for iR,Rvec in enumerate(iRvec):
            reverse=iRvecIndex.get(tuple(-Rvec))
            partner=np.zeros_like(ham_R[iR]) if reverse is None else ham_R[reverse].conj().T
            if not np.allclose(ham_R[iR],partner,atol=1e-12,rtol=1e-10):
                raise ValueError(f"Matrix data in {filepath} lacks reciprocal bonds at R={tuple(Rvec)}")
    if ignoreImagPart: ham_R=ham_R.real
    if overlap:
        o, px, py, pz, rx, ry, rz = data[:,7], data[:,8], data[:,9], data[:,10], data[:,11], data[:,12], data[:,13]
        matrices=[np.zeros((nR,norb,norb),dtype=float) for _ in range(7)]
        for record,entries in zip(records,zip(o,px,py,pz,rx,ry,rz)):
            for matrix,value in zip(matrices,entries):
                matrix[record]=value
        overlap_R,px_R,py_R,pz_R,rx_R,ry_R,rz_R=matrices
    
    mp_grid=_mpGridFromIRvec(iRvec)
    
    aux.showElapsedTime(extraInfo='\nreal-space grid: %d(nR)x%d(norbs)x%d(norbs), ham_R read'%(*ham_R.shape[:3],))
    if overlap:
        return _applyHrCutoff({
        "ham_R":ham_R,
        "overlap":overlap,
        "overlap_R":overlap_R,
        "momentum_R":np.array([px_R,py_R,pz_R]).transpose((1,0,2,3)), # set 1st index to be the Rvec
        "position_R":np.array([rx_R,ry_R,rz_R]).transpose((1,0,2,3)), # set 1st index to be the Rvec
        "title":headline.rstrip("\r\n"),
        "norbs":norb,
        "Ndegen":Ndegen,
        "iRvec":iRvec,
        "iRvecIndex":iRvecIndex,
        "mp_grid":mp_grid,
    }, cutoff=cutoff, overlap=True)
    return _applyHrCutoff({
        "ham_R":ham_R,
        "title":headline.rstrip("\r\n"),
        "norbs":norb,
        "Ndegen":Ndegen,
        "iRvec":iRvec,
        "iRvecIndex":iRvecIndex,
        "mp_grid":mp_grid,
    }, cutoff=cutoff, overlap=False)

def readFullHrDat(mainDir, hamR, ignoreImagPart, format, ispin, spinSeparated, cutoff=-1,
                  expectedNorbs=None,requireUnitDegeneracies=False,requireReal=False,
                  requireReciprocal=False,requireComplete=False,allowedRvecs=None):
    """Read one hr file or combine separated collinear spin files.

    The result includes the title, declared orbital count, and ``Ndegen`` in
    the returned R-block order. Hamiltonian values remain unweighted for
    backward compatibility. Optional validation is performed during this same
    read, so specialized consumers do not need to parse the file header again.
    See ``readHrDat`` for the validation keyword meanings.
    """
    single_path = '%s/%s_hr.dat'%(mainDir, hamR)
    up_path = '%s/%s.up_hr.dat'%(mainDir, hamR)
    dn_path = '%s/%s.dn_hr.dat'%(mainDir, hamR)
    if spinSeparated is None:
        spinSeparated = (
            ispin == 2
            and not os.path.exists(single_path)
            and os.path.exists(up_path)
            and os.path.exists(dn_path)
        )

    if not spinSeparated:
        return readHrDat(single_path,ignoreImagPart=ignoreImagPart,overlap=False,format=format,cutoff=cutoff,
                         expectedNorbs=expectedNorbs,requireUnitDegeneracies=requireUnitDegeneracies,
                         requireReal=requireReal,requireReciprocal=requireReciprocal,
                         requireComplete=requireComplete,allowedRvecs=allowedRvecs)

    if ispin != 2:
        raise ValueError("separated spin-channel Hamiltonians require ispin=2")
    if not os.path.exists(up_path):
        raise FileNotFoundError(f"cannot find spin-up Hamiltonian file {up_path}")
    if not os.path.exists(dn_path):
        raise FileNotFoundError(f"cannot find spin-down Hamiltonian file {dn_path}")

    expected_spin_norbs=None
    if expectedNorbs is not None:
        if int(expectedNorbs)%2:
            raise ValueError("A separated-spin expectedNorbs value must be even")
        expected_spin_norbs=int(expectedNorbs)//2
    read_options=dict(ignoreImagPart=ignoreImagPart,overlap=False,format=format,
                      expectedNorbs=expected_spin_norbs,
                      requireUnitDegeneracies=requireUnitDegeneracies,
                      requireReal=requireReal,requireReciprocal=requireReciprocal,
                      requireComplete=requireComplete,allowedRvecs=allowedRvecs)
    hr_up=readHrDat(up_path,**read_options)
    hr_dn=readHrDat(dn_path,**read_options)

    if hr_up['ham_R'].shape != hr_dn['ham_R'].shape:
        raise ValueError("spin-up and spin-down Hamiltonians have different shapes: "
                         f"{hr_up['ham_R'].shape} vs {hr_dn['ham_R'].shape}")
    if set(map(tuple,hr_up['iRvec'])) != set(map(tuple,hr_dn['iRvec'])):
        raise ValueError("spin-up and spin-down Hamiltonians do not contain the same R vectors")
    for Rvec,Ndegen in zip(hr_up['iRvec'],hr_up['Ndegen']):
        iR_dn=hr_dn['iRvecIndex'][tuple(Rvec)]
        if Ndegen != hr_dn['Ndegen'][iR_dn]:
            raise ValueError(f"spin channels have different degeneracies at R={tuple(Rvec)}")

    nRvec = len(hr_up['iRvec'])
    norb_spin = hr_up['ham_R'].shape[-1]
    ham_R = np.zeros((nRvec, 2*norb_spin, 2*norb_spin), dtype=complex)
    ham_R[:, :norb_spin, :norb_spin] = hr_up['ham_R']
    for iR, Rvec in enumerate(hr_up['iRvec']):
        iR_dn = hr_dn['iRvecIndex'][tuple(Rvec)]
        ham_R[iR, norb_spin:, norb_spin:] = hr_dn['ham_R'][iR_dn]

    return _applyHrCutoff({
        "ham_R": ham_R,
        "title": (hr_up['title'],hr_dn['title']),
        "norbs": 2*norb_spin,
        "Ndegen": hr_up['Ndegen'],
        "iRvec": hr_up['iRvec'],
        "iRvecIndex": hr_up['iRvecIndex'],
        "mp_grid": hr_up['mp_grid'],
    }, cutoff=cutoff, overlap=False)

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
        fJ.write(f'Anisotropic exchange energy J*S^2 in eV, calculated by Greens function. version: {version}\n')
        fJ.write('evaluated on %dx%dx%d k grid\n'%(*supercellSize,))
        fJ.write('#Fermi          NN   dis.  is  it  Rx  Ry  Rz           Jxx           Jxy           Jxz           Jyx           Jyy           Jyz           Jzx           Jzy           Jzz             J          Ja_x          Ja_y         Ja_yx            Dx            Dy            Dz\n')

        for iroute, route in enumerate(exch.exchangeRoutes):
            for ifermi, fermi in enumerate(exch.fermiList):
                J = exch.exchangeEnergy[ifermi,iroute]
                Jani = exch.Jani[ifermi,iroute]
                Dx, Dy, Dz = exch.DMI[ifermi,iroute,:]
                Jmat = np.array([
                    [ J + Jani[0,0], Jani[0,1] + Dz, -Dy],
                    [Jani[0,1] - Dz,  J + Jani[1,1],  Dx],
                    [            Dy,            -Dx,   J],
                ])
                fJ.write('%14.6e%3d%7.3f%4d%4d%4d%4d%4d%14.6e%14.6e%14.6e%14.6e%14.6e%14.6e%14.6e%14.6e%14.6e%14.6e%14.6e%14.6e%14.6e%14.6e%14.6e%14.6e\n'%(
                    fermi, (iroute+1), route["bond_length"], route["source_atom"], route["target_atom"], *route["overlat"],  
                    Jmat[0,0], Jmat[0,1], Jmat[0,2],
                    Jmat[1,0], Jmat[1,1], Jmat[1,2],
                    Jmat[2,0], Jmat[2,1], Jmat[2,2],
                    J, Jani[0,0], Jani[1,1], Jani[0,1],
                    Dx, Dy, Dz))

def writeMCSolverInputFile(mainDir, exch, repetition=16, pbc=(0,1), offdiag=False,
                           MAE=(0.0, 0.0, 0.0),
                           temperature=(100, 500, 16), field=(0, 0.1, 1),
                           sweeps=(40000, 80000, 0), modelType="Ising",
                           algorithm="Metropolis", ncores=4, spin=1.0,
                           spinStates=None, magneticAtoms=None,
                           measurePair=None, filename="MCSolver.txt",
                           fermiIndex=0):
    """
    Write an MCSOLVER input file from a new-framework ExchangeCoupling object.

    ``MAE=(E_x, E_y, E_z)`` gives the DFT total energies in eV per model cell
    for the reference magnetic structure rotated uniformly along the three
    Cartesian axes. Only energy differences matter.

    With ``offdiag=True``, the full exchange tensors are written and their
    uniform-axis anisotropy is subtracted from ``MAE``. The residual is shared
    equally among the magnetic ions as single-ion anisotropy. With
    ``offdiag=False``, isotropic exchange is written, so the complete ``MAE``
    is represented by single-ion anisotropy. MCSOLVER coefficients are finally
    converted from eV to Kelvin and divided by the corresponding spin norms.
    """
    EV_TO_KELVIN = 11604.518121550082

    def as_axis_tuple(value, name):
        if np.isscalar(value):
            out = np.ones(3, dtype=int)
            out[:] = int(value)
        else:
            out = np.asarray(value, dtype=int).reshape(3)
        out_masked = np.ones(3, dtype=int)
        for axis in pbc:
            out_masked[int(axis)] = int(out[int(axis)])
        return tuple(out_masked)

    def scalar_for_atom(value, atom_id, default):
        if isinstance(value, dict):
            return value.get(atom_id, default)
        if np.isscalar(value):
            return value
        return value[atom_id]

    def spin_for_atom(atom_id):
        val = scalar_for_atom(spin, atom_id, 1.0)
        return float(val)

    def spin_state_for_atom(atom_id):
        if spinStates is None:
            return np.sign(spin_for_atom(atom_id)) or 1.0
        val = scalar_for_atom(spinStates, atom_id, 1.0)
        if isinstance(val, (list, tuple, np.ndarray)):
            val = np.asarray(val, dtype=float)
            if val.shape == (3,):
                return float(val[2])
        return float(val)

    def exchange_tensor(ifermi, iroute):
        J = exch.exchangeEnergy[ifermi, iroute]
        Jani = exch.Jani[ifermi, iroute]
        Dx, Dy, Dz = exch.DMI[ifermi, iroute, :]
        return np.array([
            [ J + Jani[0,0], Jani[0,1] + Dz, -Dy],
            [Jani[0,1] - Dz,  J + Jani[1,1],  Dx],
            [            Dy,            -Dx,   J],
        ])

    repetition = as_axis_tuple(repetition, "repetition")
    model = exch.model
    if magneticAtoms is None:
        magneticAtoms = sorted({
            int(route["source_atom"]) for route in exch.exchangeRoutes
        } | {
            int(route["target_atom"]) for route in exch.exchangeRoutes
        })
    else:
        magneticAtoms = [int(atom_id) for atom_id in magneticAtoms]
    atom_to_mc_index = {atom_id: iorb for iorb, atom_id in enumerate(magneticAtoms)}

    if measurePair is None:
        anti_index = 0
        first_spin = spin_state_for_atom(magneticAtoms[0])
        for atom_id in magneticAtoms:
            if spin_state_for_atom(atom_id) * first_spin < 0:
                anti_index = atom_to_mc_index[atom_id]
                break
        measurePair = (0, anti_index, (0,0,0))

    Tmin, Tmax, nT = temperature
    Hmin, Hmax, nH = field
    thermalization, statistics, relax_step = sweeps
    outpath = mainDir + filename

    written_routes = [
        (iroute, route) for iroute, route in enumerate(exch.exchangeRoutes)
        if int(route["source_atom"]) in atom_to_mc_index and int(route["target_atom"]) in atom_to_mc_index
    ]

    mae_relative = np.asarray(MAE, dtype=float)
    if mae_relative.shape != (3,):
        raise ValueError(f"MAE should be a length-3 sequence (E_x, E_y, E_z), got shape {mae_relative.shape}")
    if not np.all(np.isfinite(mae_relative)):
        raise ValueError(f"MAE should contain finite energies, got {mae_relative}")
    mae_relative = mae_relative - np.min(mae_relative)

    exchange_axis_energy = np.zeros(3, dtype=float)
    if offdiag:
        for iroute, route in written_routes:
            source_atom = int(route["source_atom"])
            target_atom = int(route["target_atom"])
            source_sign = np.sign(spin_state_for_atom(source_atom)) or 1.0
            target_sign = np.sign(spin_state_for_atom(target_atom)) or 1.0
            exchange_axis_energy += source_sign * target_sign * np.diag(
                exchange_tensor(int(fermiIndex), iroute)
            )

    exchange_mae = exchange_axis_energy - np.min(exchange_axis_energy)
    residual_mae = mae_relative - exchange_mae
    residual_mae -= np.min(residual_mae)
    single_ion_energy_per_atom = residual_mae / len(magneticAtoms)

    with open(outpath, 'w') as f:
        f.write("This is mcsolver's save file, version: 3.0\n")
        f.write('Lattice:\n')
        for vec in model.lattice:
            f.write('%.9f %.9f %.9f\n'%(vec[0], vec[1], vec[2]))
        f.write('Supercell used in MC simulations:\n')
        f.write('%d %d %d\n'%repetition)
        f.write('Orbitals in cell:\n')
        f.write('%d\n'%len(magneticAtoms))
        f.write('Positions, initial spin states and onsite-anisotropy of every orbital:\n')
        for atom_id in magneticAtoms:
            iorb = atom_to_mc_index[atom_id]
            atom_type = int(model.posinfo['itype list'][atom_id])
            spin_abs = abs(spin_for_atom(atom_id))
            spin_state = spin_abs * np.sign(spin_state_for_atom(atom_id))
            if spin_state == 0:
                spin_state = spin_abs
            pos = model.posinfo['pos ions (frac.)'][atom_id]
            Dx, Dy, Dz = single_ion_energy_per_atom * EV_TO_KELVIN / spin_abs**2
            f.write('orb %d : type %d spin %.2f pos [%.9f %.9f %.9f] Dx: %.9f Dy: %.9f Dz: %.9f\n'%(
                iorb, atom_type, spin_state, pos[0], pos[1], pos[2], Dx, Dy, Dz
            ))

        f.write('Bonds:\n')
        f.write('%d\n'%len(written_routes))
        f.write('id, source, target, overLat, Jx, Jy, Jz, Jxy, Jxz, Jyz, Jyx, Jzx, Jzy of each bond:\n')
        ifermi = int(fermiIndex)
        for ibond, (iroute, route) in enumerate(written_routes):
            source_atom = int(route["source_atom"])
            target_atom = int(route["target_atom"])
            s1 = abs(spin_for_atom(source_atom))
            s2 = abs(spin_for_atom(target_atom))
            factor = EV_TO_KELVIN / abs(s1 * s2)
            Jmat = exchange_tensor(ifermi, iroute) * factor
            if offdiag:
                f.write('bond %d: Jx %.9f Jy %.9f Jz %.9f Jxy %.9f Jxz %.9f Jyz %.9f Jyx %.9f Jzx %.9f Jzy %.9f orb %d to orb %d over [%d %d %d]\n'%(
                    ibond, Jmat[0,0], Jmat[1,1], Jmat[2,2],
                    Jmat[0,1], Jmat[0,2], Jmat[1,2],
                    Jmat[1,0], Jmat[2,0], Jmat[2,1],
                    atom_to_mc_index[source_atom], atom_to_mc_index[target_atom],
                    *route["overlat"]
                ))
            else:
                f.write('bond %d: Jx %.9f Jy %.9f Jz %.9f Jxy 0 Jxz 0 Jyz 0 Jyx 0 Jzx 0 Jzy 0 orb %d to orb %d over [%d %d %d]\n'%(
                    ibond, Jmat[2,2], Jmat[2,2], Jmat[2,2],
                    atom_to_mc_index[source_atom], atom_to_mc_index[target_atom],
                    *route["overlat"]
                ))

        f.write('Dipole long range correction:\n')
        f.write('alpha 0.0\n')
        f.write('Temperature scanning region:\n')
        f.write('Tmin %.9g Tmax %.9g nT %d\n'%(Tmin, Tmax, int(nT)))
        f.write('Field scanning region:\n')
        f.write('Hmin %.9g Hmax %.9g nH %d\n'%(Hmin, Hmax, int(nH)))
        f.write('Measurement:\n')
        f.write('measure the correlation function between orb%d and orb%d over [%d %d %d]\n'%(
            int(measurePair[0]), int(measurePair[1]), *measurePair[2]
        ))
        f.write('OrbGroup: 1\n')
        f.write('Supergroup\n')
        f.write('group0 orb0-orb0\n')
        f.write('>>>       Topological section      <<<\n')
        f.write('LocalCircuit per cell: 0 (set to 0 to skip the calc. for topo. Q)\n')
        f.write('>>>   End of Topological section   <<<\n')
        f.write('Sweeps for thermalization and statistics, and relaxiation step for each sweep:\n')
        f.write('%d %d %d\n'%(int(thermalization), int(statistics), int(relax_step)))
        f.write('Model type:\n')
        f.write(f'{modelType}\n')
        f.write('Algorithm:\n')
        f.write(f'{algorithm}\n')
        f.write('XAxis type:\n')
        f.write('T\n')
        f.write('Distribution output frame: 0 \n')
        f.write('Ncores:\n')
        f.write('%d\n'%int(ncores))

    return outpath

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

def output3rdFreqDenpendentData(tensor_data,fermi,fermiAxis,omegaAxis,filePath,saveFigures=True,plotFermiIndex=None):
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
        fout.write("   Omega(eV)   Fermi(eV)")
        directionName=['x','y','z']
        for alpha1 in range(3):
            for alpha2 in range(3):
                for beta in range(3):
                    fout.write(f"            {directionName[alpha1]}{directionName[alpha2]}{directionName[beta]}")
        fout.write("\n")
        for iomega, omega in enumerate(omegaAxis):
            for ifermi, fermiEnergy in enumerate(fermiAxis):
                fout.write(f"{omega:12.6f}{fermiEnergy-fermi:12.6f}")
                for alpha1 in range(3):
                    for alpha2 in range(3):
                        for beta in range(3):
                            fout.write(f"{tensor_data[ifermi,iomega,alpha1,alpha2,beta]:15.6E}")
                fout.write("\n")

    if saveFigures:
        from matplotlib.figure import Figure

        if plotFermiIndex is None:
            plotFermiIndex = int(np.argmin(np.abs(fermiAxis-fermi)))
        if not 0 <= plotFermiIndex < len(fermiAxis):
            raise IndexError(f"plotFermiIndex={plotFermiIndex} is outside [0, {len(fermiAxis)})")

        figurePolarizations = (((0,0,), 'xx'), ((1,1,), 'yy'), ((0,1,), 'xy'), ((1,0,), 'yx'))
        componentStyles = (('x', '#d62728', '-'), ('y', '#2ca02c', '--'), ('z', '#4c78a8', '-.'))
        figureBasePath = os.path.splitext(filePath)[0]
        for (ipolarization1, ipolarization2), polarizationName in figurePolarizations:
            figure = Figure(figsize=(6.4, 4.8))
            ax1 = figure.subplots()
            for component, (componentName, color, lineStyle) in enumerate(componentStyles):
                ax1.plot(omegaAxis, tensor_data[plotFermiIndex,:,ipolarization1,ipolarization2,component],
                          color=color, linestyle=lineStyle, linewidth=2.0, label=rf"$O_{{{componentName}}}$")
            ax1.axhline(0.0, color='black', linewidth=0.8)
            ax1.set_xlabel("Photon energy (eV)")
            ax1.set_ylabel(r"$\chi$")
            ax1.set_title(f"{polarizationName} component, EF shift = "
                           f"{fermiAxis[plotFermiIndex]-fermi:.4f} eV")
            ax1.grid(alpha=0.22, linewidth=0.7)
            ax1.legend(frameon=False)
            figure.tight_layout()
            figure.savefig(f"{figureBasePath}.{polarizationName}.png", dpi=300)
            figure.clear()


def outputIFTcoeffs(tensor_data, fermi, fermiAxis, omegaAxis, theta, phi, filePath,
                    saveFigures=True, plotFermiIndex=None, magneticOrderReaction=True):
    '''
    Contract an IFT tensor with common unit-amplitude light polarizations and
    output the induced torque in the spherical frame of the magnetic moment.

    ``tensor_data[ifermi,iomega,a,b,c]`` follows
    ``tau_c = sum_ab E_a E_b^* chi_ab,c`` and stores the electronic exchange
    magnetic-moment torque ``-i[mu_S,H_XC]``. By default, the result is
    multiplied by the reaction minus sign to give the torque on the magnetic
    order used by ``IFTDevice``. Set ``magneticOrderReaction=False`` to output
    the raw electronic torque. The angles theta and phi are in radians and
    define the magnetic direction used for the spherical basis.

    Returns
    -------
    np.ndarray, shape (9, len(fermiAxis), len(omegaAxis), 3)
        Complex torque-rate coefficients in mu_B/fs/(V/angstrom)^2 for x, y,
        z, xy+, xy-, yz+, yz-, xz+, and xz- polarization. The last axis is
        (r, theta, phi).
    '''
    tensor_data = np.asarray(tensor_data)
    fermiAxis = np.asarray(fermiAxis)
    omegaAxis = np.asarray(omegaAxis)
    expectedShape = (len(fermiAxis), len(omegaAxis), 3, 3, 3)
    if tensor_data.shape != expectedShape:
        raise ValueError(f"expect tensor_data shape {expectedShape}, got {tensor_data.shape}")
    if not np.isfinite(theta) or not np.isfinite(phi):
        raise ValueError("theta and phi must be finite angles in radians")

    invSqrt2 = 1.0/np.sqrt(2.0)
    polarizationNames = ('x', 'y', 'z', 'xy+', 'xy-', 'yz+', 'yz-', 'xz+', 'xz-')
    polarizationVectors = np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1j, 0], [1, -1j, 0],
        [0, 1, 1j], [0, 1, -1j],
        [1, 0, 1j], [1, 0, -1j],
    ], dtype=complex)
    polarizationVectors[3:] *= invSqrt2

    torqueCartesian = np.einsum("pa,pb,foabc->pfoc", polarizationVectors,
                                polarizationVectors.conj(), tensor_data, optimize=True)
    sinTheta, cosTheta = np.sin(theta), np.cos(theta)
    sinPhi, cosPhi = np.sin(phi), np.cos(phi)
    sphericalBasis = np.array([
        [sinTheta*cosPhi, sinTheta*sinPhi, cosTheta],
        [cosTheta*cosPhi, cosTheta*sinPhi, -sinTheta],
        [-sinPhi, cosPhi, 0.0],
    ])
    torqueSpherical = np.einsum("sc,pfoc->pfos", sphericalBasis, torqueCartesian, optimize=True)
    reactionSign = -1.0 if magneticOrderReaction else 1.0
    torqueRateSpherical = reactionSign*torqueSpherical/aux.HBAR_EV_FS

    with open(filePath, "w") as fout:
        fout.write("# IFT torque for normalized Jones vectors with |E(Omega)| = 1\n")
        fout.write("# electronic q_c = sum_ab E_a E_b^* chi_ab,c; ab+ = (e_a + i e_b)/sqrt(2)\n")
        convention = "magnetic-order reaction torque" if magneticOrderReaction else "raw electronic exchange torque"
        fout.write(f"# torque convention: {convention}\n")
        fout.write(f"# applied reaction sign relative to q_c: {reactionSign:+.0f}\n")
        fout.write(f"# Values are divided by hbar = {aux.HBAR_EV_FS:.10f} eV fs; unit: mu_B/fs/(V/angstrom)^2\n")
        fout.write(f"# magnetization theta = {theta:.12E} rad ({np.degrees(theta):.9f} deg), "
                   f"phi = {phi:.12E} rad ({np.degrees(phi):.9f} deg)\n")
        fout.write("# spherical components use (e_r, e_theta, e_phi); only their physical real parts are written\n")
        fout.write(f"# maximum discarded imaginary residual = {np.max(np.abs(torqueRateSpherical.imag)):.9E}\n")
        fout.write("   Omega(eV) Fermi-EF(eV)")
        for polarizationName in polarizationNames:
            for componentName in ('r', 'theta', 'phi'):
                fout.write(f" {polarizationName}_tau_{componentName:5s}")
        fout.write("\n")
        for iomega, omega in enumerate(omegaAxis):
            for ifermi, fermiEnergy in enumerate(fermiAxis):
                fout.write(f"{omega:12.6f}{fermiEnergy-fermi:14.6f}")
                for ipolarization in range(len(polarizationNames)):
                    for component in range(3):
                        fout.write(f"{torqueRateSpherical[ipolarization,ifermi,iomega,component].real:15.6E}")
                fout.write("\n")

    if saveFigures:
        from matplotlib.figure import Figure

        if plotFermiIndex is None:
            plotFermiIndex = int(np.argmin(np.abs(fermiAxis-fermi)))
        if not 0 <= plotFermiIndex < len(fermiAxis):
            raise IndexError(f"plotFermiIndex={plotFermiIndex} is outside [0, {len(fermiAxis)})")

        figurePolarizations = ((0, 'x'), (1, 'y'), (3, 'xy+'), (4, 'xy-'))
        componentStyles = (('r', '#4c78a8', '--'), ('theta', '#d62728', '-'), ('phi', '#2ca02c', '-.'))
        figureBasePath = os.path.splitext(filePath)[0]
        for ipolarization, polarizationName in figurePolarizations:
            figure = Figure(figsize=(6.4, 4.8))
            axes = figure.subplots()
            for component, (componentName, color, lineStyle) in enumerate(componentStyles):
                axes.plot(omegaAxis, torqueRateSpherical[ipolarization,plotFermiIndex,:,component].real,
                          color=color, linestyle=lineStyle, linewidth=2.0,
                          label=rf"$\tau_{{{componentName}}}$")
            axes.axhline(0.0, color='black', linewidth=0.8)
            axes.set_xlabel("Photon energy (eV)")
            axes.set_ylabel(r"Torque galvanic efficiency [$\mu_B\,\mathrm{fs}^{-1}/(\mathrm{V}/\mathrm{\AA})^2$]")
            axes.set_title(f"{polarizationName}-polarized light, EF shift = "
                           f"{fermiAxis[plotFermiIndex]-fermi:.4f} eV")
            axes.grid(alpha=0.22, linewidth=0.7)
            axes.legend(frameon=False)
            figure.tight_layout()
            figure.savefig(f"{figureBasePath}.{polarizationName}.png", dpi=300)
            figure.clear()
    return torqueRateSpherical

def outputIFEcoeffs(tensor_data, fermi, fermiAxis, omegaAxis, filePath,
                    saveFigures=True, plotFermiIndex=None):
    '''
    Contract a frequency-domain IFE tensor with common unit-amplitude light
    polarizations and output the induced magnetic moment in Cartesian form.

    ``tensor_data[ifermi,iomega,a,b,c]`` follows
    ``Delta M_c = sum_ab E_a E_b^* chi_ab,c``. The input must use the standard
    complex frequency-domain tensor convention; the specially packed real
    off-diagonal tensor returned by the TDSE calculators is not compatible.

    Returns
    -------
    np.ndarray, shape (9, len(fermiAxis), len(omegaAxis), 3)
        Complex magnetic-moment induction coefficients in
        mu_B/(V/angstrom)^2 for x, y, z, xy+, xy-, yz+, yz-, xz+, and xz-
        polarization. The last axis is Cartesian (x, y, z).
    '''
    tensor_data = np.asarray(tensor_data)
    fermiAxis = np.asarray(fermiAxis)
    omegaAxis = np.asarray(omegaAxis)
    expectedShape = (len(fermiAxis), len(omegaAxis), 3, 3, 3)
    if tensor_data.shape != expectedShape:
        raise ValueError(f"expect tensor_data shape {expectedShape}, got {tensor_data.shape}")

    invSqrt2 = 1.0/np.sqrt(2.0)
    polarizationNames = ('x', 'y', 'z', 'xy+', 'xy-', 'yz+', 'yz-', 'xz+', 'xz-')
    polarizationVectors = np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1j, 0], [1, -1j, 0],
        [0, 1, 1j], [0, 1, -1j],
        [1, 0, 1j], [1, 0, -1j],
    ], dtype=complex)
    polarizationVectors[3:] *= invSqrt2
    momentCartesian = np.einsum("pa,pb,foabc->pfoc", polarizationVectors,
                                polarizationVectors.conj(), tensor_data, optimize=True)

    with open(filePath, "w") as fout:
        fout.write("# IFE magnetic moment for normalized Jones vectors with |E(Omega)| = 1\n")
        fout.write("# Delta M_c = sum_ab E_a E_b^* chi_ab,c; ab+ = (e_a + i e_b)/sqrt(2)\n")
        fout.write("# unit: mu_B/(V/angstrom)^2\n")
        fout.write("# Cartesian components (M_x, M_y, M_z); only their physical real parts are written\n")
        fout.write(f"# maximum discarded imaginary residual = {np.max(np.abs(momentCartesian.imag)):.9E}\n")
        fout.write("   Omega(eV) Fermi-EF(eV)")
        for polarizationName in polarizationNames:
            for componentName in ('x', 'y', 'z'):
                fout.write(f" {polarizationName}_M_{componentName}")
        fout.write("\n")
        for iomega, omega in enumerate(omegaAxis):
            for ifermi, fermiEnergy in enumerate(fermiAxis):
                fout.write(f"{omega:12.6f}{fermiEnergy-fermi:14.6f}")
                for ipolarization in range(len(polarizationNames)):
                    for component in range(3):
                        fout.write(f"{momentCartesian[ipolarization,ifermi,iomega,component].real:15.6E}")
                fout.write("\n")

    if saveFigures:
        from matplotlib.figure import Figure

        if plotFermiIndex is None:
            plotFermiIndex = int(np.argmin(np.abs(fermiAxis-fermi)))
        if not 0 <= plotFermiIndex < len(fermiAxis):
            raise IndexError(f"plotFermiIndex={plotFermiIndex} is outside [0, {len(fermiAxis)})")

        figurePolarizations = ((0, 'x'), (1, 'y'), (3, 'xy+'), (4, 'xy-'))
        componentStyles = (('x', '#d62728', '-'), ('y', '#2ca02c', '--'), ('z', '#4c78a8', '-.'))
        figureBasePath = os.path.splitext(filePath)[0]
        for ipolarization, polarizationName in figurePolarizations:
            figure = Figure(figsize=(6.4, 4.8))
            axes = figure.subplots()
            for component, (componentName, color, lineStyle) in enumerate(componentStyles):
                axes.plot(omegaAxis, momentCartesian[ipolarization,plotFermiIndex,:,component].real,
                          color=color, linestyle=lineStyle, linewidth=2.0, label=rf"$M_{{{componentName}}}$")
            axes.axhline(0.0, color='black', linewidth=0.8)
            axes.set_xlabel("Photon energy (eV)")
            axes.set_ylabel(r"Magnetic-moment induction efficiency [$\mu_B/(\mathrm{V}/\mathrm{\AA})^2$]")
            axes.set_title(f"{polarizationName}-polarized light, EF shift = "
                           f"{fermiAxis[plotFermiIndex]-fermi:.4f} eV")
            axes.grid(alpha=0.22, linewidth=0.7)
            axes.legend(frameon=False)
            figure.tight_layout()
            figure.savefig(f"{figureBasePath}.{polarizationName}.png", dpi=300)
            figure.clear()
    return momentCartesian

def output4thData(tensor_data, fermi, fermiAxis, filePath):
    '''
    Output a Fermi-dependent fourth-rank Cartesian tensor.

    ``tensor_data[ifermi, a, b, c, d]`` and ``fermiAxis[ifermi]`` use
    absolute Fermi energies. The first output column is relative to ``fermi``.
    '''
    tensor_data = np.asarray(tensor_data)
    fermiAxis = np.asarray(fermiAxis)
    expectedShape = (len(fermiAxis), 3, 3, 3, 3)
    assert tensor_data.shape == expectedShape, (
        f"expect tensor_data shape {expectedShape}, got {tensor_data.shape}"
    )

    directionName = ('x', 'y', 'z')
    with open(filePath, "w") as fout:
        fout.write("Fermi(eV) ")
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    for d in range(3):
                        componentName = (
                            directionName[a]+directionName[b]
                            + directionName[c]+directionName[d]
                        )
                        fout.write(f"{componentName:>15s}")
        fout.write("\n")

        for ifermi, fermiEnergy in enumerate(fermiAxis):
            fout.write(f"{fermiEnergy-fermi:9.6f}")
            for component in np.ndindex((3, 3, 3, 3)):
                fout.write(f"{tensor_data[(ifermi,)+component]:15.6E}")
            fout.write("\n")

def outputBZDistributionData(tensor_data, filePath, reciLattice, componentLabels=None, 
                             bzRepeat=(0, 0, 0), headerLines=None):
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
    - headerLines : list[str] | None
        Optional metadata written as comment lines before the column names.
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
        if headerLines is not None:
            for headerLine in headerLines:
                fout.write(f"# {headerLine}\n")
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

def _opticalPolarizationStatesForOutput():
    polarizationNames = ('x', 'y', 'z', 'xy+', 'xy-', 'yz+', 'yz-', 'xz+', 'xz-')
    polarizationVectors = np.array([
        [1, 0, 0], [0, 1, 0], [0, 0, 1],
        [1, 1j, 0], [1, -1j, 0],
        [0, 1, 1j], [0, 1, -1j],
        [1, 0, 1j], [1, 0, -1j],
    ], dtype=complex)
    polarizationVectors[3:] /= np.sqrt(2.0)
    return polarizationNames, polarizationVectors

def _normalizeBZRepeatForOutput(bzRepeat):
    if np.isscalar(bzRepeat):
        bzRepeat = (int(bzRepeat), int(bzRepeat), int(bzRepeat))
    else:
        bzRepeat = tuple(int(value) for value in bzRepeat)
    if len(bzRepeat) != 3 or any(value < 0 for value in bzRepeat):
        raise ValueError(f"bzRepeat must contain three nonnegative integers, got {bzRepeat}")
    return bzRepeat

def _plotVectorResponseOnBZ(response, filePath, reciLattice, bzRepeat, componentNames,
                            unitLabel, figureTitle=None):
    from matplotlib.figure import Figure

    kgrid = response.shape[:3]
    if kgrid[2] != 1 or bzRepeat[2] != 0:
        warnings.warn("BZ response figures are only generated for Nk3=1 and bzRepeat[2]=0")
        return

    axisIndices = []
    for iaxis, ngrid in enumerate(kgrid[:2]):
        if bzRepeat[iaxis] > 0:
            axisIndices.append(np.arange(-bzRepeat[iaxis]*ngrid, bzRepeat[iaxis]*ngrid+1))
        else:
            axisIndices.append(np.arange(ngrid))
    index1, index2 = np.meshgrid(axisIndices[0], axisIndices[1], indexing='ij')
    fractionalCoordinates = np.zeros(index1.shape+(3,))
    fractionalCoordinates[..., 0] = index1/kgrid[0]
    fractionalCoordinates[..., 1] = index2/kgrid[1]
    kCoordinates = np.einsum("ab,...b->...a", reciLattice, fractionalCoordinates, optimize=True)
    wrapped1, wrapped2 = index1 % kgrid[0], index2 % kgrid[1]

    figurePolarizations = ((0, 'x'), (1, 'y'), (3, 'xy+'), (4, 'xy-'))
    figureBasePath = os.path.splitext(filePath)[0]
    for ipolarization, polarizationName in figurePolarizations:
        figure = Figure(figsize=(14.4, 4.4))
        axes = figure.subplots(1, 3)
        for component, componentName in enumerate(componentNames):
            values = response[wrapped1, wrapped2, 0, ipolarization, component].real
            magnitude = np.max(np.abs(values))
            if magnitude == 0.0:
                magnitude = 1.0
            image = axes[component].pcolormesh(kCoordinates[..., 0], kCoordinates[..., 1], values,
                                               shading='auto', cmap='RdBu_r', vmin=-magnitude, vmax=magnitude)
            axes[component].set_xlabel(r"$k_x$ ($\mathrm{\AA}^{-1}$)")
            axes[component].set_ylabel(r"$k_y$ ($\mathrm{\AA}^{-1}$)")
            axes[component].set_title(componentName)
            axes[component].set_aspect('equal', adjustable='box')
            figure.colorbar(image, ax=axes[component], label=unitLabel, fraction=0.046, pad=0.04)
        title = f"{polarizationName}-polarized light"
        if figureTitle:
            title += f", {figureTitle}"
        figure.suptitle(title)
        figure.tight_layout()
        figure.savefig(f"{figureBasePath}.{polarizationName}.png", dpi=300)
        figure.clear()

def outputIFEResponseOnBZ(tensor_data, filePath, reciLattice, bzRepeat=(0, 0, 0),
                          saveFigures=True, figureTitle=None):
    '''
    Contract a k-resolved IFE tensor with common light polarizations.

    ``tensor_data[k1,k2,k3,a,b,c]`` follows
    ``Delta M_c(k) = sum_ab E_a E_b^* chi_ab,c(k)``. The output table contains
    Cartesian (M_x, M_y, M_z) for x, y, z, xy+, xy-, yz+, yz-, xz+, and xz-
    polarization. The input must use the standard complex frequency-domain
    tensor convention rather than the packed real TDSE convention.

    Returns
    -------
    np.ndarray, shape (Nk1, Nk2, Nk3, 9, 3)
        Complex k-resolved moment coefficients in mu_B/(V/angstrom)^2.
    '''
    tensor_data = np.asarray(tensor_data)
    if tensor_data.ndim != 6 or tensor_data.shape[-3:] != (3, 3, 3):
        raise ValueError(f"expect tensor_data shape (Nk1, Nk2, Nk3, 3, 3, 3), got {tensor_data.shape}")
    reciLattice = np.asarray(reciLattice)
    if reciLattice.shape != (3, 3):
        raise ValueError(f"expect reciLattice shape (3, 3), got {reciLattice.shape}")
    bzRepeat = _normalizeBZRepeatForOutput(bzRepeat)
    polarizationNames, polarizationVectors = _opticalPolarizationStatesForOutput()
    momentCartesian = np.einsum("pa,pb,...abc->...pc", polarizationVectors,
                                polarizationVectors.conj(), tensor_data, optimize=True)
    componentNames = tuple(f"{polarizationName}_M_{componentName}"
                           for polarizationName in polarizationNames for componentName in ('x', 'y', 'z'))
    outputBZDistributionData(momentCartesian.real.reshape(momentCartesian.shape[:3]+(-1,)), filePath=filePath,
                             reciLattice=reciLattice, componentLabels=componentNames, bzRepeat=bzRepeat,
                             headerLines=[
                                 "IFE response for normalized Jones vectors with |E(Omega)| = 1",
                                 "Delta M_c(k) = sum_ab E_a E_b^* chi_ab,c(k); ab+ = (e_a + i e_b)/sqrt(2)",
                                 "unit: mu_B/(V/angstrom)^2; only physical real parts are written",
                                 f"maximum discarded imaginary residual = {np.max(np.abs(momentCartesian.imag)):.9E}",
                             ])
    if saveFigures:
        _plotVectorResponseOnBZ(momentCartesian, filePath, reciLattice, bzRepeat, ('M_x', 'M_y', 'M_z'),
                                r"$\mu_B/(\mathrm{V}/\mathrm{\AA})^2$", figureTitle=figureTitle)
    return momentCartesian

def outputIFTResponseOnBZ(tensor_data, theta, phi, filePath, reciLattice, bzRepeat=(0, 0, 0),
                          saveFigures=True, figureTitle=None, magneticOrderReaction=True):
    '''
    Contract a k-resolved IFT tensor and express the torque in a spherical frame.

    ``tensor_data[k1,k2,k3,a,b,c]`` stores the electronic exchange torque and
    follows ``q_c(k) = sum_ab E_a E_b^* chi_ab,c(k)``. By default the reaction
    minus sign and 1/hbar conversion used by ``IFTDevice`` and
    ``outputIFTcoeffs`` are applied. The output components are
    (tau_r, tau_theta, tau_phi).

    Returns
    -------
    np.ndarray, shape (Nk1, Nk2, Nk3, 9, 3)
        Complex k-resolved torque-rate coefficients in
        mu_B/fs/(V/angstrom)^2.
    '''
    tensor_data = np.asarray(tensor_data)
    if tensor_data.ndim != 6 or tensor_data.shape[-3:] != (3, 3, 3):
        raise ValueError(f"expect tensor_data shape (Nk1, Nk2, Nk3, 3, 3, 3), got {tensor_data.shape}")
    reciLattice = np.asarray(reciLattice)
    if reciLattice.shape != (3, 3):
        raise ValueError(f"expect reciLattice shape (3, 3), got {reciLattice.shape}")
    if not np.isfinite(theta) or not np.isfinite(phi):
        raise ValueError("theta and phi must be finite angles in radians")
    bzRepeat = _normalizeBZRepeatForOutput(bzRepeat)
    polarizationNames, polarizationVectors = _opticalPolarizationStatesForOutput()
    torqueCartesian = np.einsum("pa,pb,...abc->...pc", polarizationVectors,
                                polarizationVectors.conj(), tensor_data, optimize=True)
    sinTheta, cosTheta = np.sin(theta), np.cos(theta)
    sinPhi, cosPhi = np.sin(phi), np.cos(phi)
    sphericalBasis = np.array([
        [sinTheta*cosPhi, sinTheta*sinPhi, cosTheta],
        [cosTheta*cosPhi, cosTheta*sinPhi, -sinTheta],
        [-sinPhi, cosPhi, 0.0],
    ])
    torqueSpherical = np.einsum("sc,...pc->...ps", sphericalBasis, torqueCartesian, optimize=True)
    reactionSign = -1.0 if magneticOrderReaction else 1.0
    torqueRateSpherical = reactionSign*torqueSpherical/aux.HBAR_EV_FS
    componentNames = tuple(f"{polarizationName}_tau_{componentName}"
                           for polarizationName in polarizationNames for componentName in ('r', 'theta', 'phi'))
    convention = "magnetic-order reaction torque" if magneticOrderReaction else "raw electronic exchange torque"
    outputBZDistributionData(torqueRateSpherical.real.reshape(torqueRateSpherical.shape[:3]+(-1,)),
                             filePath=filePath, reciLattice=reciLattice, componentLabels=componentNames,
                             bzRepeat=bzRepeat, headerLines=[
                                 "IFT response for normalized Jones vectors with |E(Omega)| = 1",
                                 "q_c(k) = sum_ab E_a E_b^* chi_ab,c(k); ab+ = (e_a + i e_b)/sqrt(2)",
                                 f"torque convention: {convention}; reaction sign = {reactionSign:+.0f}",
                                 f"divided by hbar = {aux.HBAR_EV_FS:.10f} eV fs; unit: mu_B/fs/(V/angstrom)^2",
                                 f"magnetization theta = {theta:.12E} rad, phi = {phi:.12E} rad",
                                 "spherical components (tau_r, tau_theta, tau_phi); only physical real parts are written",
                                 f"maximum discarded imaginary residual = {np.max(np.abs(torqueRateSpherical.imag)):.9E}",
                             ])
    if saveFigures:
        _plotVectorResponseOnBZ(torqueRateSpherical, filePath, reciLattice, bzRepeat,
                                ('tau_r', 'tau_theta', 'tau_phi'),
                                r"$\mu_B\,\mathrm{fs}^{-1}/(\mathrm{V}/\mathrm{\AA})^2$", figureTitle=figureTitle)
    return torqueRateSpherical

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

class W90SOI:
    def __init__(self,filename='wannier90.soi') -> None:
        r'''
        Read Cartesian orbital-SOC matrices written by the VASP
        ``LWRITE_SOI`` patch.

        The stored matrix elements are

        .. math::

            A_{\mathbf{k},nm}^{\sigma\sigma',j}
            = \langle\psi_{\mathbf{k}n\sigma}|\xi L_j|
              \psi_{\mathbf{k}m\sigma'}\rangle .

        Properties
        ----------
        - data:
            Array indexed as ``[ik, j, spin_bra, spin_ket, band_bra,
            band_ket]``.
        - kpt_list:
            Fractional reciprocal coordinates in the output block order.
        - hermiticity_error:
            Maximum error in
            ``A[j,s,s'] == A[j,s',s].conj().T``.
        '''
        if not filename.endswith('.soi'):
            filename += '.soi'

        with open(filename,'r') as fin:
            self.comment=fin.readline().strip()
            assert "VASP PAW orbital SOC v3:" in self.comment, (
                    f"{filename} was written by the legacy LWRITE_SOI "
                    "patch, whose PAW projector contraction and Cartesian "
                    "SOC decomposition were incorrect. Rebuild VASP with "
                    "the current patch and regenerate this file."
            )
                
            header=fin.readline().split()
            assert len(header)==4, f"invalid header in {filename}: {header}"
            self.nbands,self.nkpts,self.ncomponents,self.nspins=map(int,header)
            assert self.ncomponents==3, f"{filename} must contain three Cartesian components, found {self.ncomponents}"
            assert self.nspins==2, f"{filename} must contain two spin channels, found {self.nspins}"

            self.kpt_list=np.empty((self.nkpts,3),dtype=float)
            self.data=np.empty((self.nkpts,self.ncomponents,self.nspins,self.nspins,self.nbands,self.nbands),dtype=complex)

            for ikpt in range(self.nkpts):
                for icomponent in range(self.ncomponents):
                    for ispin_bra in range(self.nspins):
                        for ispin_ket in range(self.nspins):
                            block_header=fin.readline().split()
                            assert len(block_header)==7, f"invalid block header in {filename}: {block_header}"
                            block_index=tuple(map(int,block_header[:4]))
                            expected=(ikpt+1,icomponent+1,ispin_bra+1,ispin_ket+1)
                            assert block_index==expected, f"unexpected block order in {filename}: {block_index}, expected {expected}"
                            kpt=np.array(block_header[4:],dtype=float)
                            if icomponent==0 and ispin_bra==0 and ispin_ket==0:
                                self.kpt_list[ikpt]=kpt
                            else:
                                assert np.allclose(kpt,self.kpt_list[ikpt],atol=1e-12,rtol=0), f"inconsistent k point in block {block_index}"

                            matrix=self.data[ikpt,icomponent,ispin_bra,ispin_ket]
                            for _ in range(self.nbands*self.nbands):
                                fields=fin.readline().split()
                                assert len(fields)==4, f"invalid matrix entry in {filename}: {fields}"
                                ibra,iket=int(fields[0])-1,int(fields[1])-1
                                assert (0<=ibra<self.nbands and 0<=iket<self.nbands), f"band index outside the header range in {filename}"
                                matrix[ibra,iket]=(float(fields[2])+1j*float(fields[3]))

            assert not any(line.strip() for line in fin), (
                f"unexpected trailing data in {filename}"
            )

        hermitian_partner=self.data.conj().transpose(0,1,3,2,5,4)
        self.hermiticity_error=np.max(np.abs(self.data-hermitian_partner))
        print(f"SOI file {filename} has been read: {self.nbands} bands, {self.nkpts} k points; "
              f"Hermiticity error {self.hermiticity_error:.3e} eV")

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

class ElectronSelfEnergy:
    """Read VASP electron-phonon self energies from ``vaspout.h5``.

    VASP stores the selected ``(spin, k, band)`` states through one-based
    sparse record indices. This reader restores dense irreducible-k arrays.
    ``System_w90.includeElectronSelfEnergy`` normally infers the full-to-IBZ
    map from the corresponding full-grid Wannier90 ``.eig`` spectra and the
    VASP symmetry weights. An explicit map extracted from ``vaspelph.h5`` may
    still be supplied for electronically ambiguous cases.

    The relaxation energy used by the density-matrix SERTA implementation is

        Gamma_nk = -2 Im Sigma_Fan^R(nk, epsilon_nk) = hbar / tau_nk.

    Parameters
    ----------
    filePath : str
        VASP ``vaspout.h5`` containing ``self_energy_<calculator>``.
    calculator : int
        One-based self-energy accumulator index.
    kpointMap : str, optional
        ``.npz`` containing ``full_kpoints``, ``irreducible_kpoints`` and
        zero-based ``full_to_irreducible``.  A VASP ``vaspelph.h5`` is also
        accepted. Normally unnecessary when matching Wannier90 ``.eig`` files
        are available.
    """

    def __init__(self, filePath, calculator=1, kpointMap=None):
        self.filePath=os.path.abspath(os.path.normpath(filePath))
        groupName=f"results/electron_phonon/electrons/self_energy_{int(calculator)}"
        with h5py.File(self.filePath,"r") as h5:
            if groupName not in h5:
                raise KeyError(f"cannot find {groupName} in {self.filePath}")
            group=h5[groupName]
            self.band_start=int(group["band_start"][()])
            self.band_stop=int(group["band_stop"][()])
            self.nbands=int(group["nbands"][()])
            self.nbands_sum=int(np.atleast_1d(group["nbands_sum"][()])[0])
            self.temperatures=np.asarray(group["temps"][()],dtype=float)
            self.efermi=np.asarray(group["efermi"][()],dtype=float)
            self.scattering_approximation=group["scattering_approximation"][()].decode().strip()
            record_index=np.asarray(group["bks_idx"][()],dtype=int)-1
            record_energy=np.asarray(group["energies"][()],dtype=float)
            record_fan=np.asarray(group["selfen_fan"][()],dtype=float)
            record_dw=np.asarray(group["selfen_dw"][()],dtype=float)

            electronGroup=h5["results/electron_phonon/electrons/eigenvalues"]
            self.irreducible_kpoints=np.asarray(electronGroup["kpoint_coords"][()],dtype=float)
            self.dft_eigenvalues=np.asarray(electronGroup["eigenvalues"][()],dtype=float)
            self.kpoint_symmetry_weights=np.asarray(electronGroup["kpoints_symmetry_weight"][()],dtype=float)
            kpointInput=h5["input/kpoints_elph"] if "input/kpoints_elph" in h5 else h5["input/kpoints"]
            self.mp_grid=np.array([kpointInput["nkpx"][()],kpointInput["nkpy"][()],
                                   kpointInput["nkpz"][()]],dtype=int)

        if record_fan.shape[-1] != 2:
            raise ValueError(f"expected real/imaginary self-energy pairs, found {record_fan.shape}")
        if record_index.shape[:2] != self.dft_eigenvalues.shape[:2]:
            raise ValueError("self-energy sparse indices and electron eigenvalues use different spin/k grids")
        expected_bands=self.band_stop-self.band_start+1
        if record_index.shape[-1] != expected_bands:
            raise ValueError(f"expected {expected_bands} selected bands, found {record_index.shape[-1]}")

        valid=record_index >= 0
        if np.any(record_index[valid] >= len(record_fan)):
            raise ValueError("self-energy sparse record index exceeds the stored record count")
        denseShape=record_index.shape+record_fan.shape[1:-1]
        self.fan=np.full(denseShape,np.nan+1j*np.nan,dtype=complex)
        self.energies=np.full(record_index.shape+record_energy.shape[1:],np.nan,dtype=float)
        self.debye_waller=np.full(record_index.shape+record_dw.shape[1:],np.nan,dtype=float)
        self.fan[valid]=record_fan[record_index[valid],...,0]+1j*record_fan[record_index[valid],...,1]
        self.energies[valid]=record_energy[record_index[valid]]
        self.debye_waller[valid]=record_dw[record_index[valid]]

        if np.any(~valid):
            missing=int(np.count_nonzero(~valid))
            warnings.warn(f"VASP self-energy file omits {missing} selected spin-k-band states",RuntimeWarning)

        self.full_kpoints=None
        self.full_to_irreducible=None
        if kpointMap is not None:
            self.loadKpointMap(kpointMap)

        print(f"electron self energy read from {self.filePath}: {record_index.shape[0]} spins, "
              f"{record_index.shape[1]} irreducible k points, bands {self.band_start}-{self.band_stop}, "
              f"temperatures {self.temperatures}")

    def loadKpointMap(self, filePath):
        """Load the full-to-irreducible electron-phonon k-point map."""
        filePath=os.path.abspath(os.path.normpath(filePath))
        if filePath.lower().endswith(".npz"):
            with np.load(filePath) as data:
                full=np.asarray(data["full_kpoints"],dtype=float)
                irreducible=np.asarray(data["irreducible_kpoints"],dtype=float)
                mapping=np.asarray(data["full_to_irreducible"],dtype=int)
        else:
            with h5py.File(filePath,"r") as h5:
                full=np.asarray(h5["kpoints/vkpt_kp"][()],dtype=float)
                irreducible=np.asarray(h5["kpoints/vkpt_k"][()],dtype=float)
                mapping=np.asarray(h5["kpoints/indx_fbz2ibz"][()],dtype=int)-1
        if mapping.shape != (len(full),) or mapping.min() < 0 or mapping.max() >= len(irreducible):
            raise ValueError("invalid full-to-irreducible k-point map")
        delta=irreducible-self.irreducible_kpoints
        delta-=np.rint(delta)
        if irreducible.shape != self.irreducible_kpoints.shape or np.max(np.abs(delta)) > 1e-7:
            raise ValueError("k-point map and self-energy file have different irreducible k-point lists")
        self.full_kpoints=full
        self.full_to_irreducible=mapping
        self.kpointMapPath=filePath
        return self

    def inferKpointMap(self, fullKpoints, fullDFTEigenvalues, mp_grid=None,
                       energyTolerance=0.25, ambiguityTolerance=1e-6, batchSize=256):
        """Infer the full-to-IBZ map from DFT-band spectral fingerprints.

        This avoids requiring ``vaspelph.h5`` when the same full-grid DFT
        eigenvalues are available from Wannier90 ``.eig`` files. Every full
        k point is assigned to the IBZ spectrum with the smallest spin/band
        RMS difference. VASP's stored symmetry weights, exact IBZ
        representatives, and best/second-best separation validate the map.
        Ambiguous mappings must use an explicit map instead.
        """
        fullKpoints=np.asarray(fullKpoints,dtype=float)
        fullDFTEigenvalues=np.asarray(fullDFTEigenvalues,dtype=float)
        if fullDFTEigenvalues.ndim == 2:
            fullDFTEigenvalues=fullDFTEigenvalues[None,...]
        expected=(self.dft_eigenvalues.shape[0],len(fullKpoints),self.dft_eigenvalues.shape[-1])
        if fullDFTEigenvalues.shape != expected:
            raise ValueError(f"full-grid DFT eigenvalues must have shape {expected}; "
                             f"found {fullDFTEigenvalues.shape}")
        mp_grid=self.mp_grid if mp_grid is None else np.asarray(mp_grid,dtype=int)
        if tuple(mp_grid) != tuple(self.mp_grid):
            raise ValueError(f"self-energy mesh {tuple(self.mp_grid)} and target mesh {tuple(mp_grid)} differ")
        if len(fullKpoints) != int(np.prod(mp_grid)):
            raise ValueError("spectral map inference requires the complete regular Monkhorst-Pack grid")
        keys=[self._gridKey(kpoint,mp_grid) for kpoint in fullKpoints]
        if len(set(keys)) != len(keys):
            raise ValueError("full k-point list does not contain each Monkhorst-Pack point exactly once")

        mapping=np.empty(len(fullKpoints),dtype=int)
        bestScores=np.empty(len(fullKpoints),dtype=float)
        margins=np.empty(len(fullKpoints),dtype=float)
        for start in range(0,len(fullKpoints),int(batchSize)):
            stop=min(start+int(batchSize),len(fullKpoints))
            difference=(fullDFTEigenvalues[:,start:stop,None,:]
                        -self.dft_eigenvalues[:,None,:,:])
            scores=np.sqrt(np.mean(difference**2,axis=(0,3)))
            order=np.argpartition(scores,1,axis=1)[:,:2]
            firstScores=np.take_along_axis(scores,order,axis=1)
            swap=firstScores[:,1]<firstScores[:,0]
            order[swap]=order[swap,::-1]
            firstScores[swap]=firstScores[swap,::-1]
            mapping[start:stop]=order[:,0]
            bestScores[start:stop]=firstScores[:,0]
            margins[start:stop]=firstScores[:,1]-firstScores[:,0]

        if np.max(bestScores)>float(energyTolerance):
            raise ValueError(f"cannot infer a reliable k-point map: maximum matched spectral RMS error "
                             f"{np.max(bestScores):.3e} eV exceeds {energyTolerance:.3e} eV")
        if np.min(margins)<float(ambiguityTolerance):
            raise ValueError(f"spectral k-point map is ambiguous: minimum best/second-best separation "
                             f"{np.min(margins):.3e} eV; provide an explicit vaspelph map")

        expectedCounts=np.rint(self.kpoint_symmetry_weights*len(fullKpoints)).astype(int)
        if np.max(np.abs(self.kpoint_symmetry_weights*len(fullKpoints)-expectedCounts))>1e-6:
            raise ValueError("VASP IBZ weights are inconsistent with the full k-grid size")
        counts=np.bincount(mapping,minlength=len(self.irreducible_kpoints))
        if not np.array_equal(counts,expectedCounts):
            raise ValueError("spectral map multiplicities disagree with VASP's IBZ symmetry weights; "
                             "provide an explicit vaspelph map")

        for iibz,kpoint in enumerate(self.irreducible_kpoints):
            delta=fullKpoints-kpoint
            delta-=np.rint(delta)
            distances=np.max(np.abs(delta),axis=1)
            if np.min(distances)>1e-7 or mapping[np.argmin(distances)]!=iibz:
                raise ValueError("spectral map does not preserve VASP's explicit IBZ representatives")

        self.full_kpoints=fullKpoints
        self.full_to_irreducible=mapping
        self.kpointMapPath=None
        self.kpointMapInference={
            "maximum_spectral_rms_error":float(np.max(bestScores)),
            "minimum_assignment_margin":float(np.min(margins)),
        }
        print(f"full-to-IBZ map inferred from DFT spectra: maximum RMS error "
              f"{np.max(bestScores):.3e} eV, minimum assignment margin {np.min(margins):.3e} eV")
        return self

    def temperatureIndex(self, temperature):
        index=int(np.argmin(np.abs(self.temperatures-float(temperature))))
        if abs(self.temperatures[index]-float(temperature)) > 1e-7:
            raise ValueError(f"temperature {temperature} K is unavailable; stored values are {self.temperatures}")
        return index

    def selfEnergy(self, temperature, frequencyIndex=0):
        """Return on-shell Fan plus Debye-Waller self energy on the IBZ."""
        itemp=self.temperatureIndex(temperature)
        if not 0 <= int(frequencyIndex) < self.fan.shape[-2]:
            raise IndexError("self-energy frequency index is out of range")
        fan=self.fan[...,int(frequencyIndex),itemp]
        dw=self.debye_waller[...,itemp]
        return fan+dw

    def linewidth(self, temperature, frequencyIndex=0):
        """Return ``Gamma=-2 Im Sigma_Fan`` in eV on the irreducible grid."""
        return -2.0*self.selfEnergy(temperature,frequencyIndex).imag

    @staticmethod
    def _gridKey(kpoint, mp_grid):
        return tuple(np.mod(np.rint(np.asarray(kpoint)*np.asarray(mp_grid)).astype(int),mp_grid))

    def expandToKpoints(self, values, kpoints, mp_grid):
        """Expand IBZ values and reorder them to an arbitrary full MP-grid list."""
        if self.full_kpoints is None:
            raise ValueError("a full k-point map is required to expand irreducible self energies")
        values=np.asarray(values)
        if values.shape[:2] != (len(self.dft_eigenvalues),len(self.irreducible_kpoints)):
            raise ValueError("values must start with (spin, irreducible-k) axes")
        expanded=values[:,self.full_to_irreducible]
        lookup={self._gridKey(kpoint,mp_grid): index for index,kpoint in enumerate(self.full_kpoints)}
        try:
            order=np.array([lookup[self._gridKey(kpoint,mp_grid)] for kpoint in kpoints],dtype=int)
        except KeyError as exc:
            raise ValueError(f"target k point {exc.args[0]} is absent from the VASP full grid") from exc
        if len(np.unique(order)) != len(order):
            raise ValueError("target k-point list contains duplicate Monkhorst-Pack points")
        return expanded[:,order]

    def linewidthOnKpoints(self, kpoints, mp_grid, temperature, frequencyIndex=0):
        return self.expandToKpoints(self.linewidth(temperature,frequencyIndex),kpoints,mp_grid)

    def selfEnergyOnKpoints(self, kpoints, mp_grid, temperature, frequencyIndex=0):
        return self.expandToKpoints(self.selfEnergy(temperature,frequencyIndex),kpoints,mp_grid)


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
        
