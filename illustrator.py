import numpy as np
import matplotlib.pyplot as plt
import posReader as pR

def __makeGrid(data:np.ndarray,kmesh:list,reciLat:np.ndarray):
    # expand grid to include border, needed for pcolormesh plotting
    X=np.linspace(0,1,kmesh[0]+1,endpoint=True)
    Y=np.linspace(0,1,kmesh[1]+1,endpoint=True)
    Z=np.linspace(0,0,kmesh[2],endpoint=True)
    kx_grid_f,ky_grid_f,kz_grid_f=np.meshgrid(X,Y,Z)
    #print(x.shape,y.shape,z.shape)
    #exit()
    # Note that the k1 run fastest in this case
    kpts_f=np.array([kx_grid_f.flatten(),ky_grid_f.flatten(),kz_grid_f.flatten()])
    #for kpt in kpts_f:
    #    print(kpt)
    #exit()
    kpts_cart=np.dot(reciLat,kpts_f)
    X_cart=kpts_cart[0]
    Y_cart=kpts_cart[1]

    data_grid_origin=data.reshape((kmesh[0],kmesh[1])).T # since the grid coor. were reversed
                                                         # we have to reverse the data sorts accordingly
    data_grid= []
    for row in data_grid_origin:
        row_add=list(row)
        row_add.append(row[0])
        data_grid.append(row_add)
    data_grid.append(data_grid[0])

    kx_grid=X_cart.reshape((kmesh[0]+1,kmesh[1]+1))
    ky_grid=Y_cart.reshape((kmesh[0]+1,kmesh[1]+1))
    return kx_grid, ky_grid, np.array(data_grid)

def __drawPatternOnHexagonalBZ(rpath:str,data_grid:np.ndarray,kx_grid:np.ndarray,ky_grid:np.ndarray,reciLat:list,
                              valueSignSym=True,cmap="Spectral"
                              ):
    g1,g2=reciLat[:2,0],reciLat[:2,1]
    maxValue=np.max(abs(data_grid))
    minValue=-maxValue if valueSignSym else 0
    plt.figure(figsize=(6,6))

    plt.pcolormesh(kx_grid,ky_grid,data_grid,shading='nearest',cmap=cmap,vmin=minValue,vmax=maxValue)
    plt.pcolormesh(kx_grid-g1[0],ky_grid-g1[1],data_grid,shading='nearest',cmap=cmap,vmin=minValue,vmax=maxValue)
    plt.pcolormesh(kx_grid-g2[0],ky_grid-g2[1],data_grid,shading='nearest',cmap=cmap,vmin=minValue,vmax=maxValue)
    plt.pcolormesh(kx_grid-(g1+g2)[0],ky_grid-(g1+g2)[1],data_grid,shading='nearest',cmap=cmap,vmin=minValue,vmax=maxValue)
    
    #plt.colorbar()
    # shift grid for contour plotting
    shiftx=(reciLat[0][0]+reciLat[0][1])/(2*kx_grid.shape[0])
    shifty=(reciLat[1][0]+reciLat[1][1])/(2*kx_grid.shape[1])
    _kx_grid=kx_grid+shiftx
    _ky_grid=ky_grid+shifty
    #print(_kx_grid.shape)
    #sub1=plt.contour(_kx_grid,_ky_grid,data_grid,linewidths=1,linestyles="dashed",colors="black")
    #sub2=plt.contour(_kx_grid-g1[0],_ky_grid-g1[1],data_grid,linewidths=1,linestyles="dashed",colors="black")
    #sub3=plt.contour(_kx_grid-g2[0],_ky_grid-g2[1],data_grid,linewidths=1,linestyles="dashed",colors="black")
    #sub4=plt.contour(_kx_grid-(g1+g2)[0],_ky_grid-(g1+g2)[1],data_grid,linewidths=1,linestyles="dashed",colors="black")

    if np.dot(reciLat[0],reciLat[1])>0: # g1 cross g2 by 60 deg.
        K1=(g1+g2)/3
        K2=((g1-g2)+g1)/3
        K3=(-g2+(g1-g2))/3
        K4=-(g1+g2)/3
        K5=-((g1-g2)+g1)/3
        K6=-(-g2+(g1-g2))/3
    else: # g1 cross g2 by 120 deg.
        K1=(g1+2*g2)/3
        K2=(-g1+g2)/3
        K3=(-2*g1-g2)/3
        K4=-(g1+2*g2)/3
        K5=-(-g1+g2)/3
        K6=-(-2*g1-g2)/3

    K=np.array([K1,K2,K3,K4,K5,K6,K1])
    corr_max=np.max([np.max(K[:,0]),np.max(K[:,1])])*1.1
    plt.plot(K[:,0]+shiftx,K[:,1]+shifty,linewidth=2,c="black")
    plt.xlim(-corr_max,corr_max)
    plt.ylim(-corr_max,corr_max)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(rpath+"hotmap.png",dpi=2000)
    plt.show()

def __drawPatternOnSquareBZ(rpath:str,data_grid:np.ndarray,kx_grid:np.ndarray,ky_grid:np.ndarray,reciLat:list,
                              valueSignSym=True,cmap="Spectral"
                              ):
    g1,g2=reciLat[:2,0],reciLat[:2,1]
    maxValue=np.max(abs(data_grid))
    minValue=-maxValue if valueSignSym else 0
    plt.figure(figsize=(6,6))

    plt.pcolormesh(kx_grid,ky_grid,data_grid,shading='nearest',cmap=cmap,vmin=minValue,vmax=maxValue)
    plt.pcolormesh(kx_grid-g1[0],ky_grid-g1[1],data_grid,shading='nearest',cmap=cmap,vmin=minValue,vmax=maxValue)
    plt.pcolormesh(kx_grid-g2[0],ky_grid-g2[1],data_grid,shading='nearest',cmap=cmap,vmin=minValue,vmax=maxValue)
    plt.pcolormesh(kx_grid-(g1+g2)[0],ky_grid-(g1+g2)[1],data_grid,shading='nearest',cmap=cmap,vmin=minValue,vmax=maxValue)
    
    #plt.colorbar()
    # shift grid for contour plotting
    shiftx=(reciLat[0][0]+reciLat[0][1])/(2*kx_grid.shape[0])
    shifty=(reciLat[1][0]+reciLat[1][1])/(2*kx_grid.shape[1])
    _kx_grid=kx_grid+shiftx
    _ky_grid=ky_grid+shifty
    #print(_kx_grid.shape)
    #sub1=plt.contour(_kx_grid,_ky_grid,data_grid,linewidths=1,linestyles="dashed",colors="black")
    #sub2=plt.contour(_kx_grid-g1[0],_ky_grid-g1[1],data_grid,linewidths=1,linestyles="dashed",colors="black")
    #sub3=plt.contour(_kx_grid-g2[0],_ky_grid-g2[1],data_grid,linewidths=1,linestyles="dashed",colors="black")
    #sub4=plt.contour(_kx_grid-(g1+g2)[0],_ky_grid-(g1+g2)[1],data_grid,linewidths=1,linestyles="dashed",colors="black")

    
    K1=(g1+g2)/2
    K2=(g1-g2)/2
    K3=(-g1-g2)/2
    K4=(-g1+g2)/2
    

    K=np.array([K1,K2,K3,K4,K1])
    corr_max=np.max([np.max(K[:,0]),np.max(K[:,1])])*1.1
    plt.plot(K[:,0]+shiftx,K[:,1]+shifty,linewidth=2,c="black")
    plt.xlim(-corr_max,corr_max)
    plt.ylim(-corr_max,corr_max)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(rpath+"hotmap.png",dpi=2000)
    plt.show()

def drawHotmapOnHexagonalBZ(rpath:str,data:np.ndarray,kmesh:list,
                            valueSignSym=True,cmap="Spectral"
                            ):
    reciLat=np.linalg.inv(pR.lat)*np.pi*2
    #print(data)
    #exit()
    kx_grid, ky_grid, data_grid= __makeGrid(data,kmesh,reciLat)
    __drawPatternOnHexagonalBZ(rpath,data_grid,kx_grid,ky_grid,reciLat,
    valueSignSym=valueSignSym,cmap=cmap
    )

def drawHotmapOnSquareBZ(rpath:str,data:np.ndarray,kmesh:list,
                            valueSignSym=True,cmap="Spectral"
                            ):
    reciLat=np.linalg.inv(pR.lat)*np.pi*2
    #print(data)
    #exit()
    kx_grid, ky_grid, data_grid= __makeGrid(data,kmesh,reciLat)
    __drawPatternOnSquareBZ(rpath,data_grid,kx_grid,ky_grid,reciLat,
    valueSignSym=valueSignSym,cmap=cmap
    )

def drawJDOSOnHexagonalBZ(filepath:str,mainDir:str,kmesh:list):
    data=np.loadtxt(filepath)
    kx_f, ky_f, kz_f, jdos= data[:,0], data[:,1], data[:,2], data[:,3]
    drawHotmapOnHexagonalBZ(mainDir,jdos,kmesh,valueSignSym=False,cmap="Purples")