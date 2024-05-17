import paramReader
paramReader.readGlobalConst()
import numpy as np

from system import System_omx, System_w90
from calculator import Velocity, SpinHallConductance, Energy, SpinBerryCurv, AnormalousHallConductance
from auxiliary import showElapsedTime

# Pbene, spin hall conductance
mainDir="./mcalc/Pt/"
model=System_w90(
    mainDir=mainDir,
    hamR="wannier90",
    #hamR="omx",
    cutoff=-0.002,
    poscar="POSCAR",
    projections={
        "Sn":(0,1,),
        "Pb":(0,1,),
        "Pt":(0,1,2,),
    },
    ispin=2
)
atomicMagMomList=[np.zeros(3),]
model.symmerizeHamAndOverlap(
    atomicMagMomList=atomicMagMomList,format="w90",spinor=True,
    maxPrint=5,                    # To print out the local Hams. of inequal bonds in first maxPrint
    print_row=(8,9,10,11,12,),print_col=(8,9,10,11,12,), # To print specialized block of local Hams.
)
model.getRHam_R()
#exit()
#fermi=-0.141600218424*27.2113845 # OPENMX Sn
#fermi=0 # OPENMX generated wannier model always displays fermi level at zero
#fermi=-2.41861676 # VASP Sn
fermi=10.96516405 # VASP Pt
fermiAxis=np.linspace(fermi-2.25,fermi+2.25,101)

BZGrid=np.array([22,22,22])
FastGrid=np.array(list(model.mp_grid))
#FastGrid=BZGrid

for icomp in range(3):
    for extra_size in range(BZGrid[icomp]-FastGrid[icomp]+1):
        if BZGrid[icomp]%FastGrid[icomp]==0: break
        FastGrid[icomp]+=1

QGrid=BZGrid//FastGrid

print(f"DFT grid: {model.mp_grid}")
print(f"Intepolation BZ grid: {BZGrid}")
print(f"FFT grid: {FastGrid}")
print(f"Phase grid: {QGrid}")
#exit()

#phaseAxis=[np.exp(-2j*np.pi/BZGrid[icomp])**np.arange(BZGrid[icomp]) for icomp in range(3)]
phaseShift=np.zeros(tuple(QGrid)+(model.nRvec,),dtype=complex)
for iq1 in range(QGrid[0]):
    for iq2 in range(QGrid[1]):
        for iq3 in range(QGrid[2]):
            q=np.array([iq1,iq2,iq3])
            for iR, Rvec in enumerate(model.iRvec):
                #phase=iq1*iR1/BZGrid[0]+iq2*iR2/BZGrid[1]+iq3*iR3/BZGrid[2]
                phase=q/BZGrid@Rvec
                phaseShift[iq1,iq2,iq3,iR]=np.exp(2j*np.pi*phase)

energies=np.zeros(tuple(BZGrid)+(model.norbs,))
velocity=np.zeros(tuple(BZGrid)+(3,model.norbs,model.norbs,),dtype=complex)
spinBerry=np.zeros(tuple(BZGrid)+(model.norbs,))
sigma=np.zeros((101,3,3,3))

showElapsedTime(offset=True)
cnt=0
for iq1 in range(QGrid[0]):
    for iq2 in range(QGrid[1]):
        for iq3 in range(QGrid[2]):
            calc=SpinHallConductance(model,kgrid=FastGrid,phaseshift=phaseShift[iq1,iq2,iq3],eta=0.026,fermiList=fermiAxis)
            #calc=Energy(model,kgrid=FastGrid,phaseshift=phaseShift[iq1,iq2,iq3])
            #calc=Velocity(model,kgrid=FastGrid,phaseshift=phaseShift[iq1,iq2,iq3])
            
            energies[iq1:BZGrid[0]:QGrid[0],iq2:BZGrid[1]:QGrid[1],iq3:BZGrid[2]:QGrid[2],:]=calc.energies
            velocity[iq1:BZGrid[0]:QGrid[0],iq2:BZGrid[1]:QGrid[1],iq3:BZGrid[2]:QGrid[2]]=calc.velocity_K
            spinBerry[iq1:BZGrid[0]:QGrid[0],iq2:BZGrid[1]:QGrid[1],iq3:BZGrid[2]:QGrid[2]]=calc.spinBerryCurv[...,0,1,2,:]
            sigma+=calc.sigma

            cnt+=1
            showElapsedTime(offset=False,extraInfo=f"q({iq1:3d},{iq2:3d},{iq3:3d}) {cnt:5d}/{np.prod(QGrid):5d} finished")
            
sigma/=np.prod(QGrid)
'''
with open("velocity.txt","w") as fout:
    for ik1 in range(BZGrid[0]):
        for ik2 in range(BZGrid[1]):
            for ik3 in range(BZGrid[2]):
                fout.write(f"{ik1:5d}{ik2:5d}{ik3:5d}")
                for ib in range(model.norbs):fout.write(f"{velocity[ik1,ik2,ik3,0,ib,(ib+2)%model.norbs].imag:10.6f}")
                #for ib in range(model.norbs):fout.write(f"{energies[ik1,ik2,ik3,ib]:10.6f}")
                fout.write("\n")

#exit()#'''

with open("%s/spinHallConductance.txt"%mainDir,"w") as fout:
    fout.write("Fermi(eV)")
    directionName=['x','y','z']
    for alpha in range(3):
        for beta in range(3):
            for gamma in range(3):
                fout.write(f"            {directionName[alpha]}{directionName[beta]}{directionName[gamma]}")
    fout.write("\n")
    for f, s in zip(fermiAxis,sigma):
        fout.write(f"{f-fermi:9.6f}")
        for alpha in range(3):
            for beta in range(3):
                for gamma in range(3):
                    fout.write(f"{s[alpha,beta,gamma]:15.6E}")
        fout.write("\n")
exit()
#'''
import matplotlib.pyplot as plt
for i, s in enumerate(sigma):
    print(f"{i=:3d} {s[0,1,2]=:.6E}")
plt.plot(fermiAxis-fermi,sigma[:,0,1,2])
plt.show()
exit()#'''

#exit()'''
# below illustrate some quantities on BZ
dataList=[
    #energies[:,:,0,8]-energies[:,:,0,7],
    #velocity[:,:,0, 1, 7,7].real
    spinBerry[:,:,0, 7]+spinBerry[:,:,0, 6],
    #calc.spinBerryCurv[:,:,0, 2,2, 8]
]

# illustrate the BZ quantity
import matplotlib.pyplot as plt
fig, axList = plt.subplots(1,len(dataList))
if len(dataList)==1:
    axList=[axList]

for data_grid, ax in zip(dataList,axList):
    vnorm=max([abs(data_grid.max()),abs(data_grid.min())])
    for G in [(0,0),(-1,0),(-1,-1),(0,-1)]:
        k1axis,k2axis=np.linspace(G[0],G[0]+1,BZGrid[0],endpoint=False),np.linspace(G[1],G[1]+1,BZGrid[1],endpoint=False)
        k1grid,k2grid=np.meshgrid(k1axis,k2axis)
        kfrac=np.array([k1grid.reshape(-1),k2grid.reshape(-1)])
        k1coor,k2coor=2*np.pi*np.linalg.inv(model.lattice[:2,:2])@kfrac
        k1mesh,k2mesh=k1coor.reshape((BZGrid[0],BZGrid[1])),k2coor.reshape((BZGrid[0],BZGrid[1]))

        p=ax.pcolormesh(k1mesh,k2mesh,data_grid,vmin=-vnorm,vmax=vnorm,cmap="Spectral")
        cs=ax.contour(k1mesh,k2mesh,data_grid,vmin=-vnorm,vmax=vnorm,colors='black')
        ax.clabel(cs)
    plt.colorbar(p)
    ax.axis("equal")
plt.show()