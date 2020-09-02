import numpy as np

def autoGenerateKpath2D(rpath):
    '''
    generate k-path automatically,
    only for 2D case, temporally
    Parameter(s):
        nikpt: represents the number of interval KPs between two high-symmetric KP
    '''
    def genKPath(highSymK,nikpt=20):
        #generate kpath between high-symmetry kpoints
        kpath=[]
        npath=len(highSymK)-1
        for ipath in range(npath):
            dk=(highSymK[ipath+1]-highSymK[ipath])/nikpt[ipath]
            for ikpt in range(nikpt[ipath]):
                kpath.append(ikpt*dk+highSymK[ipath])
        kpath.append(highSymK[npath])
        return kpath

    global lat
    reci_lattice=np.linalg.inv(lat.T)
    b1=reci_lattice[0][:2]/2
    b2=reci_lattice[1][:2]/2
    b1b2=b1+b2
    # calc. coordinates of edge
    # in cartesian coor.
    factor=np.array([b1,b2])
    res=np.array([np.dot(b1,b1),np.dot(b2,b2)])
    K_cart=np.dot(np.linalg.inv(factor),res)
    # in reci-fractional coor.
    K0=np.dot(np.array([K_cart[0],K_cart[1],0.]),np.linalg.inv(reci_lattice))

    # another possible
    factor=np.array([b1,b1b2])
    res=np.array([np.dot(b1,b1),np.dot(b1b2,b1b2)])
    K_cart=np.dot(np.linalg.inv(factor),res)
    K1=np.dot(np.array([K_cart[0],K_cart[1],0.]),np.linalg.inv(reci_lattice))

    # chose the short one
    K=K0[:2] if np.dot(K0,K0)<np.dot(K1,K1) else K1[:2]

    # find the coordinates of bond
    # four possible positions
    M_list=[np.array([0.5,0]),np.array([-0.5,0]),np.array([0.,0.5]),np.array([0.,-0.5])]
    sort_list=aux.quicksort([np.dot(M-K,M-K) for M in M_list])
    # chose the nearest one
    M=M_list[sort_list[0]]
    G=np.array([0.,0.])
        
    # generate kpath
    print('')
    print('auto-generate kpath for bandstructure of 2D system')
    print('gamma:',G)
    print('edge :',K)
    print('bond :',M)
    kpath=genKPath([G,K,M,G],[param.KINTERVAL,param.KINTERVAL,param.KINTERVAL])
    f_kpt=open(rpath+'KPOINTS','w')
    f_kpt.write('Automatic generated kpath for bandstructure calc. of 2D system via XXZ\n')
    f_kpt.write('%5d\n'%len(kpath))
    f_kpt.write('reciprocal\n')
    for kpt in kpath:
        f_kpt.write('{:.6f} {:.6f} {:.6f} {:.2f}\n'.format(kpt[0],kpt[1],0,1))
    f_kpt.close()