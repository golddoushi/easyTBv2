import os
import numpy as np
import functools
import warnings

from numpy.core.multiarray import array as array, zeros as zeros
from system import System
from auxiliary import obtainProjections, obtainProjectionsPlain, showElapsedTime, iterate3dpm, fourier_q_to_R, reindexOrbitalsQE2VASP
from auxiliary import SYMPREC
from fileio import readHrDat, readPOSCAR, readRDat
from fileio import ElectronSelfEnergy, W90CHK, W90MMN, W90EIG, W90SOI

class System_w90(System):

    def __init__(self,
                 mainDir:str,hamR:str="wannier90",ignoreImagPart=False,cutoff=-1.,format="VASP",vaspSPN=None,
                 poscar:str="POSCAR",projections:dict=None,ispin=1,
                 getAA:bool=True,trans_inv=True,
                 spinSeparated=None,
                 ) -> None:
        '''
        The system in basis of wannier orbitals, with convention as below:
        - Basis are normalized and orthorgonal to each other.
        - For case with double spin tunnels, the basis of spin-up are collected in 1st block while spin-down are in the other block.
        - The basis in L=1 shell are in order p_z, p_x, p_y; 
          L=2 are d_{z^2}, d_{xz}, d_{yz}, d_{x^2-y^2}, d_{xy}; 
          L=3 are f_{z^3}, f_{xz^2}, f_{yz^2}, f_{z(x^2-y^2)}, f_{xyz}, f_{x(x^2-3y^2)}, f_{y(3x^2-y^2)}.

        Parameters:
        -----------
        - mainDir : str  
        the path to working directory
        - hamR : str
        seedname of real-space hamiltonian, in w90 format
        - ignoreImagPart : logic
        to skip the imaginary part of hopping and overlap
        - cutoff : float
        to skip the bond with strength below the criteria
        - format : "VASP"|"VASP6"|"QE"
        the spin tunnel order can be in VASP format, which the spin-down are collected, or in QE and VASP format
        - poscarPath : str
        seedname of POSCAR
        - projections : dictionary
        The atomic orbitals of each element. Key name corresponds to the element, item corresponds to the
          sequence of atomic orbitals, e.g.,
        - ispin : int
        The number of spin tunnels. NOTE: set ispin=2 for spinor case
        - spinSeparated : None | bool
        If True, read ``{hamR}.up.chk/eig/mmn`` and
        ``{hamR}.dn.chk/eig/mmn`` as separate collinear spin channels and
        combine them in block order ``[up, down]``. If None, this mode is
        selected automatically only when the single ``{hamR}.chk`` file is
        absent and both separated checkpoints are present.
          projections={
              "Fe":("s","s","p","p","d","d"),
              "Cl":(0,0,1,1,2,3),
          }

          in which 0 is "s", 1 is "p", 2 is "d", 3 is "d" and 4 is "f". By now, only these 4 kinds are supported.
        - trans_inv : bool
        switch the periodic boundary condition
        '''
        super().__init__()
        # wannier basis are normalized and othorgonal
        self.overlap=False
        # order of angular moment
        self.format=format
        # TB approximation
        self.getAA=getAA
        self.mainDir=mainDir
        self.hamR=hamR
        single_seed=os.path.join(mainDir,hamR)
        up_seed=single_seed+'.up'
        dn_seed=single_seed+'.dn'
        if spinSeparated is None:
            spinSeparated=(
                ispin==2
                and not os.path.exists(single_seed+'.chk')
                and os.path.exists(up_seed+'.chk')
                and os.path.exists(dn_seed+'.chk')
            )
        self.spinSeparated=bool(spinSeparated)
        self.soi_R_W=None
        self._soi_filename=None
        self._included_soi_R=None
        self._soi_global_spin_frame=None

        showElapsedTime()

        if self.spinSeparated:
            assert ispin==2, "separated spin-channel Wannier files require ispin=2"
            for seed in (up_seed,dn_seed):
                for extension in ('.chk','.eig'):
                    assert os.path.exists(seed+extension),f"cannot find separated spin-channel file {seed+extension}"

            chk_up=W90CHK(seedname=up_seed)
            chk_dn=W90CHK(seedname=dn_seed)
            self._validateSeparatedCheckpoints(chk_up,chk_dn)
            self._w90_spin_chks=(chk_up,chk_dn)
            self._w90_chks=self._w90_spin_chks
            chk=chk_up
            showElapsedTime(extraInfo="spin-separated chk files read")

            ham_k_DFT_up=W90EIG(seedname=up_seed)
            ham_k_DFT_dn=W90EIG(seedname=dn_seed)
            self._w90_dft_eigenvalues=(ham_k_DFT_up.data,ham_k_DFT_dn.data)
            assert ham_k_DFT_up.nkpts==chk_up.num_kpts
            assert ham_k_DFT_dn.nkpts==chk_dn.num_kpts
            assert ham_k_DFT_up.nbands==chk_up.num_bands
            assert ham_k_DFT_dn.nbands==chk_dn.num_bands
            Ham_k_W_up=np.array([self.toWannierGauge(H_k,ik,ik,chk_up) for ik,H_k in enumerate(ham_k_DFT_up.data)])
            Ham_k_W_dn=np.array([self.toWannierGauge(H_k,ik,ik,chk_dn) for ik,H_k in enumerate(ham_k_DFT_dn.data)])
            nwann_spin=chk_up.num_wann
            self._w90_spin_slices=(slice(0,nwann_spin), slice(nwann_spin,2*nwann_spin))
            Ham_k_W=np.zeros((chk.num_kpts,2*nwann_spin,2*nwann_spin),dtype=complex)
            Ham_k_W[:,self._w90_spin_slices[0],self._w90_spin_slices[0]]=Ham_k_W_up
            Ham_k_W[:,self._w90_spin_slices[1],self._w90_spin_slices[1]]=Ham_k_W_dn
            chk_center=np.concatenate((chk_up.wannier_centers,chk_dn.wannier_centers),axis=0)
            showElapsedTime(extraInfo="spin-separated H_k_DFT read and Wannier-gauge transformation done")
        else:
            chk=W90CHK(seedname=single_seed)
            self._w90_spin_chks=None
            self._w90_spin_slices=None
            self._w90_chks=(chk,)
            showElapsedTime(extraInfo="chk read")
            ham_k_DFT=W90EIG(seedname=single_seed)
            self._w90_dft_eigenvalues=(ham_k_DFT.data,)
            showElapsedTime(extraInfo=f"H_k_DFT: {ham_k_DFT.data.min():.6f} ~ {ham_k_DFT.data.max():.6f} eigs. read")
            Ham_k_W=np.array([self.toWannierGauge(H_k,ik,ik,chk) for ik,H_k in enumerate(ham_k_DFT.data)])
            chk_center=chk.wannier_centers
            showElapsedTime(extraInfo="Wannier gauge trans. done")

        # 5. store lattice info.
        self.norbs=Ham_k_W.shape[-1]
        self.mp_grid=chk.mp_grid
        self.lattice=chk.real_lattice
        self.reciLattice=np.linalg.inv(self.lattice)*np.pi*2
        self.iRvec, self.iRvecIndex, self.Ndegen = self.wigner_seitz(chk.mp_grid)
        self.nRvec=len(self.iRvec)
        self.electronSelfEnergy_R=None
        self.electronLinewidth_R=None
        self.electronLinewidthSqrt_R=None
        self.electronSelfEnergySource=None
        self.electronSelfEnergyTemperature=None

        kpt_mp_grid = [tuple(k) for k in np.array(np.round(chk.kpt_list*np.array(chk.mp_grid)[None,:]), dtype=int)%chk.mp_grid]

        fourier_q_to_R_loc = functools.partial(
            fourier_q_to_R,
            mp_grid=chk.mp_grid,
            kpt_mp_grid=kpt_mp_grid,
            iRvec=self.iRvec,
            ndegen=self.Ndegen)
        self._fourier_q_to_R=fourier_q_to_R_loc
        
        # 6. FFT to obtain H_W in R-space
        self.ham_R=fourier_q_to_R_loc(Ham_k_W)
        if (format=="QE" or format=="VASP6") and not self.spinSeparated:
            self.ham_R=reindexOrbitalsQE2VASP(self.ham_R, axis=-1)
            self.ham_R=reindexOrbitalsQE2VASP(self.ham_R, axis=-2)
            chk_center=reindexOrbitalsQE2VASP(chk_center,axis=0)
        
        '''
        # 7. check the eigs
        ham_R_W=np.zeros(tuple(chk.mp_grid)+(self.norbs,self.norbs),dtype=complex)
        for iR, Rvec in enumerate(self.iRvec):
            ham_R_W[tuple(Rvec)]=self.ham_R[iR]
            
        ham_k_W=np.fft.ifftn(ham_R_W,axes=(0,1,2,))*np.prod(chk.mp_grid)
        ham_k_H, _=np.linalg.eigh(ham_k_W)
        for ik, kpt in enumerate(kpt_mp_grid):
            print(kpt)
            for eig in ham_k_DFT.data[ik,:self.norbs]: print(f"{eig:8.4f}",end='')
            print()
            for eig in ham_k_H[kpt]: print(f"{eig:8.4f}",end='')
            print()
        #exit()'''
        '''
        print(self.ham_R.shape)
        with open("%s/auto_hr.dat"%mainDir,"w") as fout:
            for iR, hR in zip(self.iRvec,self.ham_R):
                for isource in range(self.norbs):
                    for itarget in range(self.norbs):
                        fout.write("%5d%5d%5d%5d%5d%10.6f%10.6f\n"%(*iR,isource+1,itarget+1,hR[isource,itarget].real,hR[isource,itarget].imag))
        exit()'''
        
        # 7. compute Berry connection in wannier gauge
        if getAA:
            if self.spinSeparated:
                AA_k_W_up=self._berryConnectionWannier(up_seed,chk_up,trans_inv)
                AA_k_W_dn=self._berryConnectionWannier(dn_seed,chk_dn,trans_inv)
                AA_k_W=np.zeros((chk.num_kpts,3,self.norbs,self.norbs),dtype=complex)
                AA_k_W[:,:,self._w90_spin_slices[0],self._w90_spin_slices[0]]=AA_k_W_up
                AA_k_W[:,:,self._w90_spin_slices[1],self._w90_spin_slices[1]]=AA_k_W_dn
            else:
                AA_k_W=self._berryConnectionWannier(single_seed,chk,trans_inv)

            # 8. FFT to obtain AA_W in R-space
            self.AA_R_W=fourier_q_to_R_loc(AA_k_W)
            if (format=="QE" or format=="VASP6") and not self.spinSeparated:
                self.AA_R_W=reindexOrbitalsQE2VASP(self.AA_R_W, axis=-1)
                self.AA_R_W=reindexOrbitalsQE2VASP(self.AA_R_W, axis=-2)
            for Rvec, AA in zip(self.iRvec,self.AA_R_W):
            #    print(Rvec)
            #    print(AA[0,:3,:3])
                if Rvec@Rvec==0: self.rList=np.diagonal(AA,axis1=-2,axis2=-1).real
            #exit()
            print("diagonal element of AA_R_W and the wcc produced in .chk")
            for AA_r, wcc in zip(self.rList.T,chk_center): 
                print(f"{AA_r[0].real:10.6f}{AA_r[1].real:10.6f}{AA_r[2].real:10.6f} - {wcc[0]:10.6f}{wcc[1]:10.6f}{wcc[2]:10.6f}")
            showElapsedTime(extraInfo="AA_R_W calc. done")

        # 12. read in the centers of each basis
        print("set wcc to align the chk")
        self.rList=chk_center.T
        self.ispin=ispin
        
        self.posinfo=readPOSCAR('%s/%s'%(mainDir,poscar))
        orb_list=obtainProjectionsPlain('%s/%s'%(mainDir,poscar),projections,ispin=ispin)
        assert self.norbs == sum([len(atom) for atom in orb_list]), f"{self.norbs=} {sum([len(atom) for atom in orb_list])=}"
        print("pls. check the wcc and atom coors.")
        print("At present, the atom coor will be used (for furthur symmerization)")
        reciLat=np.linalg.inv(self.posinfo['lattice'])
        for iatom, (atom_cart, atom_frac, atom_shell) in enumerate(zip(self.posinfo['pos ions (cart.)'],self.posinfo['pos ions (frac.)'],orb_list)):
            orbCenter=np.mean(self.rList[:,atom_shell],axis=1)@reciLat
            atom_frac = atom_frac + np.around(orbCenter - atom_frac)
            atom_cart = atom_frac@self.posinfo['lattice']
            # update poscar
            self.posinfo['pos ions (frac.)'][iatom] = atom_frac
            self.posinfo['pos ions (cart.)'][iatom] = atom_cart
            for iorb in atom_shell:
                wcc=self.rList[:,iorb]
                print(f"orb{iorb:3d} {wcc[0]:6.3f} {wcc[1]:6.3f} {wcc[2]:6.3f} atom{iatom:3d} {atom_cart[0]:6.3f} {atom_cart[1]:6.3f} {atom_cart[2]:6.3f}")
                self.rList[:,iorb]=atom_cart
        
        self.atomicOrbInfoList=obtainProjections('%s/%s'%(mainDir,poscar),projections,ispin=ispin)
        '''
        # tight-binding approach
        hrDat=readHrDat('%s/%s_hr.dat'%(mainDir,hamR),ignoreImagPart=ignoreImagPart,overlap=False,format=format)
        self.iRvec=hrDat['iRvec']
        self.iRvecIndex=hrDat['iRvecIndex']
        self.nRvec=len(self.iRvec)
        self.mp_grid=hrDat['mp_grid']
        self.ham_R=hrDat['ham_R']
        self.norbs=self.ham_R.shape[-1]
        # r_mn of tight-binding model is diagonal
        self.AA_R_W=np.zeros((self.nRvec,3,self.norbs,self.norbs),dtype=complex)
        for iRvec, Rvec in enumerate(self.iRvec):
            if Rvec@Rvec==0: 
                for iorb in range(self.norbs): self.AA_R_W[iRvec,:,iorb,iorb]=self.rList[:,iorb]
        #'''
        # 12. compute the R H_R, R \otimes R H_R and R \otimes R \otimes R H_R, for the further calcs. on velocities 
        self.getRHam_R()
        self.getRRHam_R()
        self.getRRRHam_R()
        self.getSpinMatrice()
        # 13. compute orbital angular moment operator
        self.atomOrbAM=self.orbitalAngularMomentCubic()
        self.getOrbitalMatrice()

    def _validateSeparatedCheckpoints(self,chk_up:W90CHK,chk_dn:W90CHK):
        assert chk_up.num_wann==chk_dn.num_wann, (
            "the current spin-block architecture requires equal numbers of "
            f"up/down Wannier functions, found {chk_up.num_wann} and "
            f"{chk_dn.num_wann}"
        )
        assert chk_up.num_kpts==chk_dn.num_kpts, "spin-up and spin-down checkpoints have different k-point counts"
        assert np.array_equal(chk_up.mp_grid,chk_dn.mp_grid), "spin-up and spin-down checkpoints have different Monkhorst-Pack grids"
        assert np.allclose(chk_up.real_lattice,chk_dn.real_lattice,atol=1e-10,rtol=0), "spin-up and spin-down checkpoints have different real lattices"
        assert np.allclose(chk_up.recip_lattice,chk_dn.recip_lattice,atol=1e-10,rtol=0), "spin-up and spin-down checkpoints have different reciprocal lattices"
        dk=chk_up.kpt_list-chk_dn.kpt_list
        dk-=np.rint(dk)
        assert np.max(np.abs(dk))<1e-10, "spin-up and spin-down checkpoints have differently ordered k points"

        center_delta=chk_up.wannier_centers-chk_dn.wannier_centers
        center_frac=center_delta@np.linalg.inv(chk_up.real_lattice)
        center_frac-=np.rint(center_frac)
        center_distance=np.linalg.norm(center_frac@chk_up.real_lattice, axis=1)
        print(f"paired up/down Wannier-center distance: mean {center_distance.mean():.3e} angstrom, max {center_distance.max():.3e} angstrom")

    def includeElectronSelfEnergy(self, selfEnergy, temperature=300.0, kpointMap=None, frequencyIndex=0,
                                  eigenvalueTolerance=5e-3):
        """Project a VASP on-shell electron self energy into the Wannier basis.

        The diagonal DFT-gauge self energy is transformed with the same
        checkpoint matrices used for ``ham_R`` and Fourier transformed to the
        same Wigner-Seitz real-space grid.  ``electronLinewidth_R`` stores
        ``Gamma=-2 Im Sigma_Fan`` in eV for use by the SERTA collision operator.
        """
        if isinstance(selfEnergy,(str,os.PathLike)):
            selfEnergy=ElectronSelfEnergy(selfEnergy,kpointMap=kpointMap)
        elif not isinstance(selfEnergy,ElectronSelfEnergy):
            raise TypeError("selfEnergy must be an ElectronSelfEnergy object or a vaspout.h5 path")
        elif kpointMap is not None:
            selfEnergy.loadKpointMap(kpointMap)

        nspin_expected=len(self._w90_chks)
        if selfEnergy.dft_eigenvalues.shape[0] != nspin_expected:
            raise ValueError(f"self energy has {selfEnergy.dft_eigenvalues.shape[0]} spin channels, "
                             f"but this Wannier model expects {nspin_expected}")
        for chk in self._w90_chks:
            if selfEnergy.band_start != 1 or selfEnergy.band_stop != chk.num_bands:
                raise ValueError("the first SERTA implementation requires self energies for every DFT band "
                                 f"in each checkpoint (1-{chk.num_bands}); found "
                                 f"{selfEnergy.band_start}-{selfEnergy.band_stop}")

        kpoints=self._w90_chks[0].kpt_list
        if selfEnergy.full_kpoints is None:
            selfEnergy.inferKpointMap(kpoints,np.stack(self._w90_dft_eigenvalues),self.mp_grid)
        linewidth_dft=selfEnergy.linewidthOnKpoints(kpoints,self.mp_grid,temperature,frequencyIndex)
        sigma_dft=selfEnergy.selfEnergyOnKpoints(kpoints,self.mp_grid,temperature,frequencyIndex)
        dft_eigenvalues=selfEnergy.expandToKpoints(selfEnergy.dft_eigenvalues,kpoints,self.mp_grid)
        for ispin,(chk,reference,values) in enumerate(zip(
                self._w90_chks,self._w90_dft_eigenvalues,dft_eigenvalues)):
            raw_error=float(np.max(np.abs(values-reference)))
            projected_error=max(float(np.max(np.abs(self.toWannierGauge(
                values[ik]-reference[ik],ik,ik,chk
            )))) for ik in range(chk.num_kpts))
            print(f"spin {ispin} self-energy eigenvalue mismatch: raw {raw_error:.3e} eV, "
                  f"Wannier-subspace {projected_error:.3e} eV")
            if raw_error > eigenvalueTolerance:
                warnings.warn(
                    f"spin {ispin} raw self-energy and Wannier .eig energies differ by {raw_error:.3e} eV. "
                    "They should agree for identical frozen-potential VASP runs; the smaller projected error "
                    "only shows how much of this mismatch enters the selected Wannier subspace.",
                    RuntimeWarning,
                )
            if projected_error > eigenvalueTolerance:
                raise ValueError(f"spin {ispin} self-energy and Wannier DFT Hamiltonians differ by "
                                 f"{projected_error:.3e} eV after projection; the files do not describe the same "
                                 "electronic calculation")

        sigma_blocks=[]
        linewidth_blocks=[]
        for ispin,chk in enumerate(self._w90_chks):
            sigma_blocks.append(np.array([
                self.toWannierGauge(values,ik,ik,chk) for ik,values in enumerate(sigma_dft[ispin])
            ]))
            linewidth_blocks.append(np.array([
                self.toWannierGauge(values,ik,ik,chk) for ik,values in enumerate(linewidth_dft[ispin])
            ]))

        if self.spinSeparated:
            nk=len(kpoints)
            sigma_k=np.zeros((nk,self.norbs,self.norbs),dtype=complex)
            linewidth_k=np.zeros_like(sigma_k)
            for block,linewidth,orbital_slice in zip(sigma_blocks,linewidth_blocks,self._w90_spin_slices):
                sigma_k[:,orbital_slice,orbital_slice]=block
                linewidth_k[:,orbital_slice,orbital_slice]=linewidth
        else:
            sigma_k=sigma_blocks[0]
            linewidth_k=linewidth_blocks[0]
            if self.format in ("QE","VASP6"):
                sigma_k=reindexOrbitalsQE2VASP(sigma_k,axis=-1)
                sigma_k=reindexOrbitalsQE2VASP(sigma_k,axis=-2)
                linewidth_k=reindexOrbitalsQE2VASP(linewidth_k,axis=-1)
                linewidth_k=reindexOrbitalsQE2VASP(linewidth_k,axis=-2)

        linewidth_k=0.5*(linewidth_k+linewidth_k.conj().swapaxes(-2,-1))
        linewidth_eigenvalues,linewidth_eigenvectors=np.linalg.eigh(linewidth_k)
        min_linewidth=float(linewidth_eigenvalues.min())
        if min_linewidth < -1e-8:
            warnings.warn(f"Wannier-projected linewidth has negative eigenvalue {min_linewidth:.3e} eV",RuntimeWarning)
        linewidth_eigenvalues=np.maximum(linewidth_eigenvalues,0.0)
        linewidth_sqrt_k=(linewidth_eigenvectors*np.sqrt(linewidth_eigenvalues)[...,None,:]) @ linewidth_eigenvectors.conj().swapaxes(-2,-1)
        self.electronSelfEnergy_R=self._fourier_q_to_R(sigma_k)
        self.electronLinewidth_R=self._fourier_q_to_R(linewidth_k)
        self.electronLinewidthSqrt_R=self._fourier_q_to_R(linewidth_sqrt_k)
        boundary_mask=np.any(2*np.abs(np.asarray(self.iRvec))>=np.asarray(self.mp_grid),axis=1)
        def boundaryWeight(quantity_R):
            norm_squared=np.sum(np.abs(quantity_R.reshape(self.nRvec,-1))**2,axis=1)
            total=float(np.sum(norm_squared))
            return float(np.sum(norm_squared[boundary_mask])/total) if total>0 else 0.
        self.electronSelfEnergyBoundaryWeight=boundaryWeight(self.electronSelfEnergy_R)
        self.electronLinewidthBoundaryWeight=boundaryWeight(self.electronLinewidth_R)
        self.electronLinewidthSqrtBoundaryWeight=boundaryWeight(self.electronLinewidthSqrt_R)
        if self.electronLinewidthSqrtBoundaryWeight>1e-3:
            warnings.warn(
                "SERTA linewidth interpolation is poorly localized: "
                f"{self.electronLinewidthSqrtBoundaryWeight:.3%} of the squared norm of sqrt(Gamma)_R lies "
                "on the Wigner-Seitz boundary. Use a denser source self-energy grid before relying on "
                "interpolated lifetimes.",
                RuntimeWarning,
            )
        self.electron_self_energy_R=self.electronSelfEnergy_R
        self.electron_linewidth_R=self.electronLinewidth_R
        self.electron_linewidth_sqrt_R=self.electronLinewidthSqrt_R
        self.electronSelfEnergySource=selfEnergy
        self.electronSelfEnergyTemperature=float(temperature)
        print(f"electron self energy transformed to Wannier R space at {temperature:g} K; "
              f"linewidth range {linewidth_dft.min():.6e} ~ {linewidth_dft.max():.6e} eV; "
              f"boundary weights Sigma/Gamma/sqrt(Gamma) = {self.electronSelfEnergyBoundaryWeight:.3%}/"
              f"{self.electronLinewidthBoundaryWeight:.3%}/{self.electronLinewidthSqrtBoundaryWeight:.3%}")
        return self.electronLinewidth_R

    def _berryConnectionWannier(self,seedname:str,chk:W90CHK,trans_inv:bool):
        M_k_DFT=W90MMN(seedname=seedname, kpt_list=chk.kpt_list, reci_lat=chk.recip_lattice)
        assert M_k_DFT.nbands==chk.num_bands
        assert M_k_DFT.nkpts==chk.num_kpts
        showElapsedTime(extraInfo=f"{seedname}.mmn read")
        Mmn_k_W=np.array([[self.toWannierGauge(Mmnk1k2,ik1,M_k_DFT.innkpt[ik1,ib],chk) 
                           for ib,Mmnk1k2 in enumerate(Mmnk1)]
                           for ik1,Mmnk1 in enumerate(M_k_DFT.Mmn)
        ])
        AA_k_W=np.zeros((chk.num_kpts,3,chk.num_wann,chk.num_wann),dtype=complex)
        delta=np.eye(chk.num_wann)
        diagonal=np.arange(chk.num_wann,dtype=int)
        for ikpt,MMNk in enumerate(Mmn_k_W):
            for bvec,wb,overlap in zip(M_k_DFT.bvecs[ikpt],M_k_DFT.wbvecs,MMNk):
                # Eq. 44, PRB 74, 195118 (2006).
                contribution= 1j*wb*bvec[:,None,None] * (overlap-delta)[None,:,:]
                if trans_inv:
                    # Translationally invariant diagonal, Eq. 13,
                    # PRB 56, 12847 (1997).
                    phase=np.angle(np.diag(overlap))
                    # Im log is defined modulo 2*pi.  Lift every link phase
                    # to the branch nearest the corresponding Wannier center.
                    # This is essential when a center lies on a cell boundary:
                    # tiny numerical noise can otherwise alternate a phase
                    # between +pi and -pi across the k mesh and create large,
                    # spurious AA(R != 0) matrix elements after the FFT.
                    phase_reference=-(chk.wannier_centers@bvec)
                    phase+=2*np.pi*np.rint(
                        (phase_reference-phase)/(2*np.pi)
                    )
                    contribution[:,diagonal,diagonal]=-(
                        wb*bvec[:,None]*
                        phase[None,:]
                    )
                AA_k_W[ikpt]+=contribution
        return AA_k_W

    def _resolveSOIFile(self,soi):
        if soi is None:
            soi=self.hamR+'.soi'
        if not soi.endswith('.soi'):
            soi+='.soi'
        if not os.path.isabs(soi):
            soi=os.path.join(self.mainDir,soi)
        return os.path.abspath(os.path.normpath(soi))

    def loadSOI(self,soi=None,force=False):
        '''
        Read ``wannier90.soi`` and transform its Cartesian orbital-SOC
        matrices into the separated-spin Wannier basis.

        This method does not alter ``ham_R``. The resulting tensor is stored
        in ``soi_R_W`` with indexing ``[iR, j, orbital_bra, orbital_ket]``.
        '''
        assert self.spinSeparated, ("wannier90.soi must be combined with the separated collinear "
                                    "up/down Wannier checkpoints")
        filename=self._resolveSOIFile(soi)
        if not force and self.soi_R_W is not None and self._soi_filename==filename:
            return self.soi_R_W
        assert os.path.exists(filename), f"cannot find SOI file {filename}"

        soi_data=W90SOI(filename)
        assert soi_data.hermiticity_error<1e-8, f"Bloch-gauge SOI matrices are not Hermitian: {soi_data.hermiticity_error:.3e} eV"
        chk_up,chk_dn=self._w90_spin_chks
        for chk_spin in (chk_up,chk_dn):
            assert soi_data.nbands==chk_spin.num_bands, f"{filename} has {soi_data.nbands} bands but the checkpoint has {chk_spin.num_bands}"
            assert soi_data.nkpts==chk_spin.num_kpts, f"{filename} has {soi_data.nkpts} kpoints but the checkpoint has {chk_spin.num_kpts}"
            dk=soi_data.kpt_list-chk_spin.kpt_list
            dk-=np.rint(dk)
            assert np.max(np.abs(dk))<1e-10, f"k-point order in {filename} differs from the checkpoints"

        soi_k_W=np.zeros((soi_data.nkpts,3,self.norbs,self.norbs),dtype=complex)
        spin_chks=(chk_up,chk_dn)
        for ikpt in range(soi_data.nkpts):
            for ispin_bra in range(2):
                bra_slice=self._w90_spin_slices[ispin_bra]
                for ispin_ket in range(2):
                    ket_slice=self._w90_spin_slices[ispin_ket]
                    soi_k_W[ikpt,:,bra_slice,ket_slice]=self.toWannierGaugeCross(
                        soi_data.data[ikpt,:,ispin_bra,ispin_ket],
                        ikpt, ikpt, spin_chks[ispin_bra], spin_chks[ispin_ket],
                    )

        hermiticity_error=float(np.max(np.abs(
            soi_k_W-soi_k_W.conj().swapaxes(-1,-2)
        )))
        assert hermiticity_error<1e-8, f"Wannier-gauge Cartesian SOI matrices are not Hermitian: {hermiticity_error:.3e} eV"

        self.soi_R_W=self._fourier_q_to_R(soi_k_W)
        soi_norm_squared=np.sum(np.abs(self.soi_R_W.reshape(self.nRvec,-1))**2,axis=1)
        boundary_mask=np.any(2*np.abs(np.asarray(self.iRvec))>=np.asarray(self.mp_grid),axis=1)
        total_soi_norm_squared=np.sum(soi_norm_squared)
        self.soi_boundary_weight=(float(np.sum(soi_norm_squared[boundary_mask]) / total_soi_norm_squared) if total_soi_norm_squared>0 else 0.)
        if self.soi_boundary_weight>1e-3:
            warnings.warn(
                "SOI interpolation is poorly localized: "
                f"{self.soi_boundary_weight:.3%} of its squared norm lies "
                "on the Wigner-Seitz boundary. Native-mesh values remain "
                "valid, but interpolation may require a denser mesh or "
                "better aligned up/down Wannier gauges.",
                RuntimeWarning,
            )
        self._soi_filename=filename
        print(
            "Cartesian SOI matrices transformed to Wannier gauge; "
            f"Hermiticity error {hermiticity_error:.3e} eV; "
            f"boundary weight {self.soi_boundary_weight:.3%}"
        )
        return self.soi_R_W

    def symmerizeCollinearAndSOI(self,soi=None,symOpSelection:list=[],print_row=(0,1,),print_col=(0,1,),
        enforceSpinlessTimeReversal=True):
        r'''
        Symmetrize the SOC-free collinear Hamiltonian and Cartesian SOI
        coefficients before selecting a magnetization direction.

        The three matrices in ``soi_R_W[:, j]`` are coefficients of the
        orbital axial vector in

        .. math::

            H_{\rm SO}=\sum_j A_j\sigma_j.

        They therefore cannot be symmetrized as three unrelated scalar
        Hamiltonians.  Their Cartesian axis is transformed as an axial
        vector, including the extra determinant for improper rotations.

        After this routine, repeated calls to ``includeSOCFromSOI`` may use
        arbitrary ``theta`` and ``phi`` without symmetrizing the assembled
        Hamiltonian again.  Each fixed direction retains the magnetic little
        group appropriate to that direction, while Hamiltonians at
        symmetry-related directions transform covariantly.

        ``enforceSpinlessTimeReversal`` imposes the real-space relations
        ``H_R = H_R*`` and ``A_R = -A_R*``.  The SOC-free Hamiltonian is even
        and its orbital axial-vector coefficient is odd under spinless time
        reversal.  These projections remove residual violations between
        independently disentangled up/down Wannier gauges.  Disable them only
        when the collinear reference intentionally contains orbital flux or
        another spin-independent time-reversal-breaking term.

        This routine must act on the collinear model before an SOI
        contribution is included.
        '''
        assert self.spinSeparated, "separate collinear/SOI symmetrization requires the separated up/down Wannier model"
        assert self._included_soi_R is None, "remove the included SOI before symmetrizing its collinear and Cartesian ingredients"
        assert self._soi_global_spin_frame is None

        filename=self._resolveSOIFile(soi)
        if self.soi_R_W is None or self._soi_filename!=filename:
            self.loadSOI(filename)

        nions=int(self.posinfo['tot ions'])
        nonmagnetic_reference=[np.zeros(3) for _ in range(nions)]

        if enforceSpinlessTimeReversal:
            spinless_tr_residual=float(np.max(np.abs(self.ham_R-self.ham_R.conj())))
            self.ham_R=0.5*(self.ham_R+self.ham_R.conj())
            print(f"Projected the collinear Hamiltonian onto spinless time reversal; residual {spinless_tr_residual:.3e} eV")
            spinless_soi_tr_residual=float(np.max(np.abs(self.soi_R_W+self.soi_R_W.conj())))
            self.soi_R_W=0.5*(self.soi_R_W-self.soi_R_W.conj())
            print(f"Projected the orbital-SOI coefficients onto spinless time reversal; residual {spinless_soi_tr_residual:.3e} eV")

        print("Symmetrizing the SOC-free collinear model")
        self.symmerizeHamAndOverlap(spinor=False, symOpSelection=symOpSelection,
            atomicMagMomList=nonmagnetic_reference, print_row=print_row, print_col=print_col,
        )

        print("Symmetrizing Cartesian orbital-SOI coefficients")
        self.soi_R_W=self.symmerizeTensorInAtomicBasis(self.soi_R_W, axisRep=("iRvec","axial","atomic orb","atomic orb"),
            spinor=False,
            symOpSelection=symOpSelection, atomicMagMomList=nonmagnetic_reference,
            format="VASP6" if self.format=="VASP6" else "w90",
            print_row=print_row, print_col=print_col, reportSymOps=False,
        )
        return self.soi_R_W

    def _restoreSOILocalSpinFrame(self):
        """Return an automatically rotated SOI model to its local spin frame."""
        if self._soi_global_spin_frame is None:
            return
        theta,phi=self._soi_global_spin_frame
        self.rotateSpinBasisToGlobal(alpha=phi,beta=-theta,gamma=-phi)
        self._soi_global_spin_frame=None

    def _rotateSOIToGlobalSpinFrame(self,theta,phi):
        """Apply the spin-frame convention used by ``includeSOCFromSOI``."""
        self.rotateSpinBasisToGlobal(alpha=phi,beta=theta,gamma=-phi)
        self._soi_global_spin_frame=(float(theta),float(phi))

    def includeSOCFromSOI(self,soi=None,theta=0.,phi=0.,strength=1.,replace=True,keepRegularSpinAxis=True):
        r'''
        Add the WISOI Hamiltonian for a chosen magnetization direction.

        Following the WISOI construction, ``theta`` and ``phi`` specify the
        local collinear spin basis aligned with the magnetization. The orbital
        SOI coefficients are contracted with Pauli matrices represented in
        that local basis. For ``theta=phi=0`` the SOC blocks are

        .. math::

            \begin{pmatrix}
            A_z^{\uparrow\uparrow} &
            A_x^{\uparrow\downarrow}-iA_y^{\uparrow\downarrow}\\
            A_x^{\downarrow\uparrow}+iA_y^{\downarrow\uparrow} &
            -A_z^{\downarrow\downarrow}
            \end{pmatrix}.

        If ``keepRegularSpinAxis`` is True (the default), the complete model is
        subsequently transformed with

        .. math::

            U=R_z(\phi)R_y(\theta)R_z(-\phi)

        so that ``spin[0:3]`` again denotes the global Cartesian operators.
        This is a change of representation, not an additional physical
        rotation. If False, the model remains in the local spin frame.

        This WISOI construction is distinct from adding the zero-angle SOC and
        then calling ``rotateExchangeDirection``. The latter keeps SOC fixed in
        a global frame and is available as a separate rigid-exchange
        approximation.

        If ``replace`` is True, a contribution previously added by this
        method is removed before the new direction is applied. An automatic
        global-frame rotation from an earlier call is undone before the
        replacement, preventing spin-frame rotations from accumulating.
        '''
        self._restoreSOILocalSpinFrame()
        filename=self._resolveSOIFile(soi)
        if self.soi_R_W is None or self._soi_filename!=filename:
            self.loadSOI(filename)
        strength=float(strength)

        c=np.cos(theta/2)
        s=np.sin(theta/2)
        spin_rotation=np.array([
            [c,-np.exp(-1j*phi)*s],
            [np.exp(1j*phi)*s,c],
        ],dtype=complex)
        pauli=np.array([
            [[0,1],[1,0]],
            [[0,-1j],[1j,0]],
            [[1,0],[0,-1]],
        ],dtype=complex)
        pauli_local=np.array([
            spin_rotation.conj().T@sigma@spin_rotation
            for sigma in pauli
        ])

        soc_R=np.zeros_like(self.ham_R)
        for ispin_bra in range(2):
            bra_slice=self._w90_spin_slices[ispin_bra]
            for ispin_ket in range(2):
                ket_slice=self._w90_spin_slices[ispin_ket]
                soc_R[:,bra_slice,ket_slice]=np.einsum(
                    'j,rjmn->rmn',
                    pauli_local[:,ispin_bra,ispin_ket],
                    self.soi_R_W[:,:,bra_slice,ket_slice],
                    optimize=True,
                )
        soc_R*=strength

        real_space_error=0.
        for iR,Rvec in enumerate(self.iRvec):
            iR_minus=self.iRvecIndex.get(tuple(-np.asarray(Rvec)))
            if iR_minus is not None:
                real_space_error=max(
                    real_space_error,
                    float(np.max(np.abs(
                        soc_R[iR]-soc_R[iR_minus].conj().T
                    ))),
                )
        assert real_space_error<SYMPREC*0.2, f"assembled SOI Hamiltonian violates real-space Hermiticity: {real_space_error:.3e} eV"

        if replace and self._included_soi_R is not None:
            self.ham_R-=self._included_soi_R
            if self.nonexchange_R is not None:
                self.nonexchange_R-=self._included_soi_R
            self._included_soi_R=None
        self.ham_R+=soc_R
        if self.nonexchange_R is not None:
            self.nonexchange_R+=soc_R
        if self._included_soi_R is None:
            self._included_soi_R=soc_R.copy()
        else:
            self._included_soi_R+=soc_R
        returned_soc_R=soc_R
        if keepRegularSpinAxis:
            spin_rotation_full=np.kron(spin_rotation,np.eye(self.norbs//2,dtype=complex))
            returned_soc_R=spin_rotation_full@soc_R@spin_rotation_full.conj().T
            self._rotateSOIToGlobalSpinFrame(theta,phi)
        else:
            self._refreshHamiltonianDerivedMatrices()
        print(
            f"included SOI for theta={theta:.8f}, phi={phi:.8f}, "
            f"strength={strength:.8f}, "
            f"keepRegularSpinAxis={bool(keepRegularSpinAxis)}; "
            f"real-space Hermiticity error "
            f"{real_space_error:.3e} eV"
        )
        return returned_soc_R

    def removeSOCFromSOI(self):
        '''Remove SOC contributions previously added by ``includeSOCFromSOI``.'''
        if self._included_soi_R is None:
            return
        global_spin_frame=self._soi_global_spin_frame
        self._restoreSOILocalSpinFrame()
        self.ham_R-=self._included_soi_R
        if self.nonexchange_R is not None:
            self.nonexchange_R-=self._included_soi_R
        self._included_soi_R=None
        if global_spin_frame is None:
            self._refreshHamiltonianDerivedMatrices()
        else:
            self._rotateSOIToGlobalSpinFrame(*global_spin_frame)
    
    def toWannierGauge(self, mat:np.ndarray, ik1:int, ik2:int, chk:W90CHK):
        '''
        Transform a matrix on DFT gauge into Wannier gauge. 
        mat should be an array in form of ..., NB, NB. 
        The last two axis is to peform gauge trans.
        '''
        if len(mat.shape) == 1: mat = np.diag(mat)
        assert mat.shape[-2:] == (chk.num_bands, chk.num_bands, )
        mat = mat[...,chk.win_min[ik1]:chk.win_max[ik1], chk.win_min[ik2]:chk.win_max[ik2]]
        return chk.v_matrix[ik1].conj() @ mat @ chk.v_matrix[ik2].T

    def toWannierGaugeCross(self,mat:np.ndarray, ik1:int, ik2:int, chk_bra:W90CHK, chk_ket:W90CHK):
        '''
        Transform a matrix whose bra and ket use different Wannier gauges.
        '''
        assert mat.shape[-2:]==(chk_bra.num_bands,chk_ket.num_bands,)
        mat=mat[..., chk_bra.win_min[ik1]:chk_bra.win_max[ik1], chk_ket.win_min[ik2]:chk_ket.win_max[ik2]]
        return  chk_bra.v_matrix[ik1].conj() @ mat @ chk_ket.v_matrix[ik2].T

    def getSpinMatrice(self):
        super().getSpinMatrice()
        return

    def orbitalAngularMomentCubic(self):
        r'''
        Orbital angular moment operator represented in the cubic hamonics (s, px, py, pz, ...)
        in unit of \hbar
        For now, only l=s, p, d, f was implemented

        Return
        ------
        - Lop : list
          Lop[l][idirection][m1, m2]. 
          l=s, p, d, e, denoting the orbital quantum number
          idirection=0, 1, 2, denoting the spatial orientation of x, y, z
          m1, m2 is the magnetic quantum number
        '''
        orbitalAngularMomentSpherical=self.orbitalAngularMomentSpherical()
        w=1/np.sqrt(2)
        def orbitalAngularP():
            '''
            Representation of orbital angular moment in space of cubic harmonics
            For w90 format, p-orbitals are listed as pz, px, py
            For VASP6 format, p-orbitals are listed as  py,pz,px
            '''
            if self.format=="VASP" or self.format=="QE":
                L=np.array([
                [    0, 1,    0],   # pz
                [   -w, 0,    w],   # px 
                [ 1j*w, 0, 1j*w],   # py
                ])
            elif self.format=="VASP6":
                L=np.array([
                [ 1j*w, 0, 1j*w],   # py
                [    0, 1,    0],   # pz
                [   -w, 0,    w],   # px
                ])
            else:
                raise NotImplementedError()
            return L.conj() @ orbitalAngularMomentSpherical[1] @ L.T

        def orbitalAngularD():
            '''
            Representation of orbital angular moment in space of cubic harmonics
            For w90 format, d-orbitals are listed as dz2, dxz, dyz, dx2-y2, dxy
            For VASP6 format, d-orbitals are listed as  dxy, dyz, dz2, dxz, dx2-y2
            '''
            if self.format=="VASP" or self.format=="QE":
                L=np.array([
                [    0,    0, 1,    0,    0],    # dz2
                [    0,   -w, 0,    w,    0],    # dxz
                [    0, 1j*w, 0, 1j*w,    0],    # dyz
                [    w,    0, 0,    0,    w],    # dx2-y2
                [-1j*w,    0, 0,    0, 1j*w],    # dxy
                ])
            elif self.format=="VASP6":
                L=np.array([
                [-1j*w,    0, 0,    0, 1j*w],    # dxy
                [    0, 1j*w, 0, 1j*w,    0],    # dyz
                [    0,    0, 1,    0,    0],    # dz2
                [    0,   -w, 0,    w,    0],    # dxz
                [    w,    0, 0,    0,    w],    # dx2-y2
                ])
            else:
                raise NotImplementedError()
            return L.conj() @ orbitalAngularMomentSpherical[2] @ L.T

        def orbitalAngularF():
            '''
            Representation of orbital angular moment in space of cubic harmonics
            For w90 format, f-orbitals are listed as fz3, fxz2, fyz2, fz(x2-y2), fxyz, fx(x2-3y2), fy(3x2-y2)
            For VASP6 format, f-orbitals are listed as  fy(3x2-y2), fxyz, fyz2, fz3, fxz2, fz(x2-y2), fx(x2-3y2)
            '''
            if self.format=="VASP" or self.format=="QE":
                L=np.array([
                [    0,    0,    0,  1,    0,    0,    0],    # fz3
                [    0,    0,   -w,  0,    w,    0,    0],    # fxz2
                [    0,    0, 1j*w,  0, 1j*w,    0,    0],    # fyz2
                [    0,    w,    0,  0,    0,    w,    0],    # fz(x2-y2)
                [    0,-1j*w,    0,  0,    0, 1j*w,    0],    # fxyz
                [   -w,    0,    0,  0,    0,    0,    w],    # fx(x2-3y2)
                [ 1j*w,    0,    0,  0,    0,    0, 1j*w],    # fy(3x2-y2)
                ])
            elif self.format=="VASP6":
                L=np.array([
                [ 1j*w,    0,    0,  0,    0,    0, 1j*w],    # fy(3x2-y2)
                [    0,-1j*w,    0,  0,    0, 1j*w,    0],    # fxyz
                [    0,    0, 1j*w,  0, 1j*w,    0,    0],    # fyz2
                [    0,    0,    0,  1,    0,    0,    0],    # fz3
                [    0,    0,   -w,  0,    w,    0,    0],    # fxz2
                [    0,    w,    0,  0,    0,    w,    0],    # fz(x2-y2)
                [   -w,    0,    0,  0,    0,    0,    w],    # fx(x2-3y2)
                ])
            else:
                raise NotImplementedError()
            return L.conj() @ orbitalAngularMomentSpherical[3] @ L.T
        return (0, orbitalAngularP(), orbitalAngularD(), orbitalAngularF())
    
    @property
    def get_mp_grid(self): return self.mp_grid

    def wigner_seitz(self, mp_grid):
        ws_search_size = np.ones(3,dtype=int)
        dist_dim = np.prod((ws_search_size + 1) * 2 + 1)
        origin = divmod((dist_dim + 1), 2)[0] - 1
        real_metric = self.lattice @ self.lattice.T
        mp_grid = np.array(mp_grid)
        iRvec = []
        iRvecIndex={}
        ndegen = []
        for n in iterate3dpm(mp_grid * ws_search_size):
            dist = []
            for i in iterate3dpm((1, 1, 1) + ws_search_size):
                ndiff = n - i * mp_grid
                dist.append(ndiff.dot(real_metric.dot(ndiff)))
            dist_min = np.min(dist)
            if abs(dist[origin] - dist_min) < 1.e-7:
                iRvecIndex[tuple(n)]=len(iRvec)
                iRvec.append(np.array(n))
                ndegen.append(np.sum(abs(dist - dist_min) < 1.e-7))
                
        return iRvec, iRvecIndex, np.array(ndegen)
    
    def symmerizeHamAndOverlap(self, spinor=False, saxis: np.ndarray = np.array([0,0,-1]), spin_basis: np.ndarray = np.zeros((2,2)), symOpSelection: list = [], atomicMagMomList: list = [], print_row=(0,1,), print_col=(0,1,)):
        if self.format=="VASP6":
            super().symmerizeHamAndOverlap(spinor, saxis, spin_basis, symOpSelection, atomicMagMomList, 'VASP6', print_row, print_col)
        else:
            super().symmerizeHamAndOverlap(spinor, saxis, spin_basis, symOpSelection, atomicMagMomList, 'w90', print_row, print_col)
        self.getRHam_R()
        self.getRRHam_R()
        self.getRRRHam_R()
