import hrReader as reader
import ham as hamilton
import numpy as np
import posReader as pR
import paramReader
import auxiliary as aux
paramReader.readGlobalConst()
import matplotlib.pyplot as plt

def calcWannierBands(mainDir,ikaxis_route=0,ikaxis_proj=1,nk_route=10,nk_proj=10,ioccupied_bands=[0]):
    def calcOverlap(eig_k1_list,eig_k2_list):
        # calc eigenstate overlapping between k1 and k2
        dim=eig_k1_list.shape[1]
        #print(eig_k1_list)
        #print(eig_k2_list)
        return np.array([[eig_k1_list[:,iv1].conjugate().dot(eig_k2_list[:,iv2]) for iv2 in range(dim)] for iv1 in range(dim)])
    
    # read Hr file
    hr=reader.Hr()
    fileList=aux.ls(mainDir)
    hrFileList=[mainDir+fileName for fileName in fileList if 'hr' in fileName]
    if len(hrFileList)==2:
        hr.readDoubleFiles(hrFileList[0],hrFileList[1],ignoreImagPart=False)
    elif len(hrFileList)==1:
        hr.readSingleFile(hrFileList[0],ignoreImagPart=False)
    elif len(hrFileList)==0:
        print("Error: cannot find Hr file in %s, pls. provide Hr file with name hr(.up or .dn) in %s"%(mainDir,mainDir))
        exit()
    else:
        print("Error: found more than two hr file in %s, pls. clean up tge directory"%mainDir)
        exit()
    
    dk_route, dk_proj=1.0/nk_route, 1.0/nk_proj
    _ham=hamilton.Ham()
    wcc_list=[]
    for ik_route in range(nk_route): # loop along the route
        wholeEigstateList=[]
        kpt=[0,0,0]
        kpt[0]=ik_route*dk_route
        for ik_proj in range(nk_proj): # contruct list of eig. states
            kpt[1]=ik_proj*dk_proj
            _hk=_ham.buildHkFromHr(hr,kpt,getHk=True)
            eigs, vec = np.linalg.eigh(_hk,'U')
            idx=eigs.argsort()[ioccupied_bands]
            wholeEigstateList.append(vec[:,idx])
        
        wilsonLoopMatrix=np.eye(len(ioccupied_bands),dtype=complex)
        for ik_proj in range(nk_proj): # multiply all matrix to get wilson loop
            overlap=calcOverlap(wholeEigstateList[ik_proj],wholeEigstateList[(ik_proj+1)%nk_proj])
            wilsonLoopMatrix=wilsonLoopMatrix.dot(overlap)
        
        eigs, _ = np.linalg.eig(wilsonLoopMatrix)
        wccVec=[(1j*np.log(z,dtype=complex)).real/(2*np.pi) %1 for z in eigs]
        wcc_list.append(wccVec)
    
    # plot and save fig
    plt.figure()
    for iwccVec, wccVec in enumerate(wcc_list):
        for wcc in wccVec:
            plt.scatter(1.0*iwccVec/nk_route,wcc,c='blue')
    
    plt.xlim(0,1)
    plt.xlabel(r'$k_%d\ (2\pi/a)$'%(ikaxis_route+1))
    plt.ylabel(r'$wcc\ along\ direction\ of\ k_%d$'%(ikaxis_proj+1))
    plt.savefig(mainDir+'wannierBands.png',dpi=300)
    plt.show()
    plt.close()

def plotBerryCurvatureInWholeBZ(hr,nkpt_in_each_direction=50,occupiedBandIndex=[0],unoccupiedBandIndex=[1]):
    kmesh=[np.array([1.0*k1/nkpt_in_each_direction,1.0*k2/nkpt_in_each_direction,0]) for k1 in range(nkpt_in_each_direction) for k2 in range(nkpt_in_each_direction)]
    berryCurv=[]
    for kpt in kmesh:
        _ham=hamilton.Ham()
        hk, d_hk_dk1, d_hk_dk2, d_hk_dk3 = _ham.build_Hk_And_DHk_FromHr(hr,kpt)
        _curv=_ham.calcBerryCurvature(hk,d_hk_dk1,d_hk_dk2,occupiedBandIndex=occupiedBandIndex,unoccupiedBandIndex=unoccupiedBandIndex)
        berryCurv.append(_curv)


