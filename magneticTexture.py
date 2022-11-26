import matplotlib
matplotlib.use('Agg')
import paramReader
paramReader.readGlobalConst()
import posReader as pR
import hrReader as reader
from ham import Green, Response
import numpy as np
from scipy.special.orthogonal import p_roots
from auxiliary import semiCirle
import matplotlib.pyplot as plt

def calcAtomicMagmom(hr:reader.Hr,atomicOrbIndexList:list,
        kgrid=[1,1,1],fermi=0.0,nGaussian=20,eta=0.0001,nthreads=1):
    '''
    Calculate the atomic charges and magnetization, using lattice Green's function
    '''
    gr=Green()
    print("Diagnolizing Hamiltonian")
    eigs_addr, vecs_addr=gr.calcEigsAndVecsInMemory_simple(hr,kgrid=kgrid,nthreads=nthreads)

    res=Response()
    eigs=res.redOutQuantitiesOnKGrid(eigs_addr,data_dim=hr.norb,kmesh=kgrid,data_type='double')
    bandEnergyMin, bandEnergyMax= eigs.flatten().min()-5, np.min([eigs.flatten().max(),fermi])
    print("Bands energy range: %.6f ~ %.6f"%(bandEnergyMin,bandEnergyMax))

    points, weights=p_roots(nGaussian)
    npoints=len(points)
    z,dz=semiCirle(bandEnergyMin,bandEnergyMax,points)

    rG_w_lat=gr.calcLatticeGreenFunction_CDLL(
        hr,eigs_addr,vecs_addr,np.arange(hr.norb),
        kgrid=kgrid,omega_list=z,lat_vecs=[(0,0,0)],eta=0.0001,spinor=True,
        nthreads=nthreads)

    diag_list=np.array([np.diag(rG_w_lat[iw][0])*dz[iw]*weights[iw] for iw in range(npoints)]).imag/2
    rho_of_each_orb=sum(diag_list)
    print("Total Charges: %.6f"%sum(rho_of_each_orb))

    rho_list,mx_list,my_list,mz_list=[],[],[],[]
    for atom in atomicOrbIndexList:
        orbIDList=[]
        for orb in atom:
            orbIDList+=orb

        norbs=len(orbIDList)
        s0,sx,sy,sz=np.zeros((norbs,norbs),dtype=complex),np.zeros((norbs,norbs),dtype=complex),np.zeros((norbs,norbs),dtype=complex),np.zeros((norbs,norbs),dtype=complex)
        for iorb in range(norbs//2):
            sx[iorb,iorb+norbs//2],sx[iorb+norbs//2,iorb]=1, 1
            sy[iorb,iorb+norbs//2],sy[iorb+norbs//2,iorb]=-1j, 1j
            sz[iorb,iorb],sz[iorb+norbs//2,iorb+norbs//2]=1,-1
            s0[iorb,iorb],s0[iorb+norbs//2,iorb+norbs//2]=1,1

        # Gaussian integration, sum here is for integration!
        s0_of_atom=sum(np.array([np.diag(s0@rG_w_lat[iw][0][orbIDList][:,orbIDList])*dz[iw]*weights[iw] for iw in range(npoints)]).imag/2)
        sx_of_atom=sum(np.array([np.diag(sx@rG_w_lat[iw][0][orbIDList][:,orbIDList])*dz[iw]*weights[iw] for iw in range(npoints)]).imag/2)
        sy_of_atom=sum(np.array([np.diag(sy@rG_w_lat[iw][0][orbIDList][:,orbIDList])*dz[iw]*weights[iw] for iw in range(npoints)]).imag/2)
        sz_of_atom=sum(np.array([np.diag(sz@rG_w_lat[iw][0][orbIDList][:,orbIDList])*dz[iw]*weights[iw] for iw in range(npoints)]).imag/2)
        #print("Charge of atom%d: %.6f, m: %.6f %.6f %.6f"%(iatom,sum(s0_of_atom),sum(sx_of_atom),sum(sy_of_atom),sum(sz_of_atom)))
        rho_list.append(sum(s0_of_atom))
        mx_list.append(sum(sx_of_atom))
        my_list.append(sum(sy_of_atom))
        mz_list.append(sum(sz_of_atom))
    return rho_list,mx_list,my_list,mz_list

pR.readPOS('./mcalc/CrI3Tube/POSCAR')

nCr, nI= pR.nions
atomicOrbIndexList=[]
iorb, norb= 0, 224
for iCr in range(nCr):
    atomicOrbIndexList.append([[],[],[iorb,iorb+1,iorb+2,iorb+3,iorb+4,iorb+norb,iorb+1+norb,iorb+2+norb,iorb+3+norb,iorb+4+norb]])
    iorb+=5
for iI in range(nI):
    atomicOrbIndexList.append([[],[iorb,iorb+1,iorb+2,iorb+norb,iorb+1+norb,iorb+2+norb]])
    iorb+=3

atomicMagMomList=[]
for iCr in range(nCr):
    pos_relative = pR.pos_ions[iCr][:2]-np.array([0.5,0.5])
    pos_n=pos_relative/np.sqrt(pos_relative@pos_relative)
    mag=np.array([pos_n[1],-pos_n[0],0])
    atomicMagMomList.append(mag)

atomicMagMomList+=[np.zeros(3) for iI in range(nI)]

hr=reader.Hr()
hr.readSingleFile('./mcalc/CrI3Tube/wannier90_hr.m_theta.dat',ignoreImagPart=False,cutoff=-0.1)
#hr.symmerizeBonds(atomicOrbIndexList=atomicOrbIndexList,symOpSelection=[],
#                  spinor=True,saxis=np.array([0,0,-1]),atomicMagMomList=atomicMagMomList)

rho_list,mx_list,my_list,mz_list=calcAtomicMagmom(hr,atomicOrbIndexList,
            kgrid=[1,1,4],fermi=-2.5,nGaussian=20,eta=0.0001,nthreads=4)
for pos, rho, mx, my, mz in zip(pR.pos_ions,rho_list,mx_list,my_list,mz_list):
    x,y,z=pos@pR.lat
    plt.arrow(x,y,mx,my,width=0.1,length_includes_head=False,fc="blue",ec="blue")
    plt.scatter(x,y,c='black')
plt.axis("equal")
plt.savefig("out.png",dpi=300)
plt.show()


