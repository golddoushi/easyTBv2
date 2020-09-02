from Lattice import BondDigital, establishHopping, establishLattice, getRadius, generateWrinkle, gaussian, cos2, showRealSpaceLattice
import numpy as np
import ham as sshModel
import matplotlib.pyplot as plt

lat=[[1,0],
     [-0.5,0.5*np.sqrt(3)]]
lat=np.array(lat)
pos=[[1./3,2./3],
     [2./3,1./3]]
bondlist=[BondDigital(0,1,[0,0],1),BondDigital(0,1,[0,1],1),BondDigital(0,1,[-1,0],1)]
Lx,Ly=20,20
orbList, orbList_flat=establishLattice(lat,pos,Lx=Lx,Ly=Ly)
phyBondList=establishHopping(orbList,bondlist,PBC=True)

radius=0.5*getRadius(orbList_flat)

center1=np.dot(np.array([1./3*Lx,2./3*Ly]),lat)
generateWrinkle(orbList_flat,centerPT=center1,wrinkleType=cos2,appliedRadius=radius,peakHeight=10)

center2=np.dot(np.array([2./3*Lx,1./3*Ly]),lat)
generateWrinkle(orbList_flat,centerPT=center2,wrinkleType=cos2,appliedRadius=radius,peakHeight=-10)

#showRealSpaceLattice(orbList_flat)
#exit()
a0=np.array([orbList_flat[0].x,orbList_flat[0].y])-np.array([orbList_flat[1].x,orbList_flat[1].y])
a0_len=np.sqrt(np.dot(a0,a0))
energyAxis=np.linspace(-0.1,0.1,101)
for beta in np.linspace(0,0.1,5):
     for bond in phyBondList:
          bondVec=np.array([bond.sourceOrb.x,bond.sourceOrb.y,bond.sourceOrb.h])-np.array([bond.targetOrb.x,bond.targetOrb.y,bond.targetOrb.h])
          bondLen=np.sqrt(np.dot(bondVec,bondVec))
          bond.t=np.exp(beta*(1-bondLen/a0_len))
     
     G_E=sshModel.getGreen_E(orbList_flat, phyBondList, E=energyAxis, filling=0.2, eta=0.01)
     A=-1./np.pi*G_E.imag

     plt.plot(energyAxis,A,label=beta)

plt.ylim(0,)
plt.legend()
plt.show()

