import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class Orb:
    def __init__(self,id,fx=0,fy=0,x=0,y=0,h=0):
        self.id=id
        self.fx=fx
        self.fy=fy
        self.x=x
        self.y=y
        self.h=0

class BondDigital:
    # bond object for establish linking, attributes are numericals
    def __init__(self,source,target,overlat,t,overlap=0.):
        self.source=source
        self.target=target
        self.overlat=np.array(overlat)
        self.t=t
        self.overlap=overlap

class BondPhysical:
    # bond object related to Orb objects
    def __init__(self,id,sourceOrb,targetOrb,t):
        self.id=id
        self.sourceOrb=sourceOrb
        self.targetOrb=targetOrb
        self.t=t

def establishLattice(lat,pos=[[0,0]],Lx=10,Ly=10):
    orbList=[]
    orbList_flat=[]
    id=0
    for x in range(Lx):
        row=[]
        for y in range(Ly):
            col=[]
            for o in pos:
                orb_fpos=np.array([x,y])+np.array(o)
                orb_cpos=np.dot(orb_fpos,lat)
                orb=Orb(id,fx=orb_fpos[0],fy=orb_fpos[1],x=orb_cpos[0],y=orb_cpos[1])
                col.append(orb)
                orbList_flat.append(orb)
                id+=1
            row.append(col)
        orbList.append(row)
    return orbList, orbList_flat

def establishHopping(orbList,bondList,PBC=False):
    phyBondList=[]
    def checkRedundancy(bond):
        alg_avg2_ref=bond.sourceOrb.id+bond.targetOrb.id
        ham_avg2_ref=bond.sourceOrb.id*bond.targetOrb.id
        for existedBond in phyBondList:
            alg_avg2=existedBond.sourceOrb.id+existedBond.targetOrb.id
            ham_avg2=existedBond.sourceOrb.id*existedBond.targetOrb.id
            if alg_avg2==alg_avg2_ref and ham_avg2==ham_avg2_ref:
                print("WARNING: meet redundant bond")
                return True
        return False

    Lx=len(orbList)
    Ly=len(orbList[0])
    id=0
    for Rx, row in enumerate(orbList):
        for Ry, col in enumerate(row):
            for iorb, orb in enumerate(col):
                for bond in bondList:
                    if bond.source==iorb:
                        targetRx=Rx+bond.overlat[0]
                        targetRy=Ry+bond.overlat[1]
                        targetFounded=False
                        if PBC:
                            targetOrb=orbList[targetRx%Lx][targetRy%Ly][bond.target]
                            targetFounded=True
                        else:
                            if targetRx>=0 and targetRx<Lx and targetRy>=0 and targetRy<Ly:
                                targetOrb=orbList[targetRx][targetRy][bond.target]
                                targetFounded=True
                        if targetFounded:
                            phyBond=BondPhysical(id,orb,targetOrb,bond.t)
                            if not checkRedundancy(phyBond):
                                phyBondList.append(phyBond)
                                id+=1
    return phyBondList

def getRadius(orbListFlat):
    def getDr(pos,center):
        drv=pos-center
        return np.sqrt(np.dot(drv,drv))
    minx, miny, maxx, maxy = 10000, 10000, -10000, -10000
    for orb in orbListFlat:
        minx=orb.x if orb.x<minx else minx
        miny=orb.y if orb.y<miny else miny
        maxx=orb.x if orb.x>maxx else maxx
        maxy=orb.y if orb.y>maxy else maxy
    centerPT=np.array([(minx+maxx),(miny+maxy)])/2
    
    orbOnBorder=[]
    for orb in orbListFlat:
        diff1, diff2, diff3, diff4=abs(orb.x-minx), abs(orb.x-maxx), abs(orb.y-miny), abs(orb.y-maxy)
        
        if diff1<1e-5 or diff2<1e-5 or diff3<1e-5 or diff4<1e-5:
            orbOnBorder.append(orb)
    radiusList=[getDr(np.array([orb.x,orb.y]),centerPT) for orb in orbOnBorder]
    touchedRadius=np.min(radiusList)
    
    return touchedRadius

def cos2(x):
    return (np.cos(np.pi*x)+1)/2

def gaussian(x):
    return np.exp(-(x/0.5)**2)

def generateWrinkle(orbListFlat, centerPT, wrinkleType=cos2, appliedRadius=0, peakHeight=5):
    def getDr(pos,center):
        drv=pos-center
        return np.sqrt(np.dot(drv,drv))
    for orb in orbListFlat:
        dr=getDr(np.array([orb.x,orb.y]),centerPT)
        if dr<=appliedRadius:
            #orb.h=peakHeight*(np.cos(np.pi*dr/appliedRadius)+1)/2
            orb.h+=peakHeight*wrinkleType(dr/appliedRadius)
    return

def showRealSpaceLattice(orbListFlat):
    coordinates=np.array([[orb.x, orb.y, orb.h] for orb in orbListFlat])
    fig = plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    ax.scatter(coordinates[:,0],coordinates[:,1],coordinates[:,2],c=coordinates[:,2])

    xmin=np.min(coordinates[:,0])
    ymin=np.min(coordinates[:,1])
    xmax=np.max(coordinates[:,0])
    ymax=np.max(coordinates[:,1])

    lim=np.max([xmax-xmin,ymax-ymin])
    ax.set_xlim(xmin,xmin+lim)
    ax.set_ylim(ymin,ymin+lim)
    ax.set_zlim(-lim/2,lim/2)
    plt.show()
            

