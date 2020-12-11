import hrReader as reader
import ham as hamilton
import numpy as np
import posReader as pR
import paramReader
paramReader.readGlobalConst()
import matplotlib.pyplot as plt


def calcWannierBands(hr,ikaxis=0,nRouteIntersections=10,nWilsonPoints=10):
    pass

def plotBerryCurvatureInWholeBZ(hr,nkpt_in_each_direction=50,occupiedBandIndex=[0],unoccupiedBandIndex=[1]):
    kmesh=[np.array([1.0*k1/nkpt_in_each_direction,1.0*k2/nkpt_in_each_direction,0]) for k1 in range(nkpt_in_each_direction) for k2 in range(nkpt_in_each_direction)]
    berryCurv=[]
    for kpt in kmesh:
        _ham=hamilton.Ham()
        hk, d_hk_dk1, d_hk_dk2, d_hk_dk3 = _ham.build_Hk_And_DHk_FromHr(hr,kpt)
        _curv=_ham.calcBerryCurvature(hk,d_hk_dk1,d_hk_dk2,occupiedBandIndex=occupiedBandIndex,unoccupiedBandIndex=unoccupiedBandIndex)
        berryCurv.append(_curv)


