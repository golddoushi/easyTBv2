import hrReader as reader
import ham as hamilton
import numpy as np
import posReader as pR
import paramReader
paramReader.readGlobalConst()
import matplotlib.pyplot as plt
import topoCalc

topoCalc.calcWannierBands('./mcalc/BN/',ikaxis_route=0,ikaxis_proj=1,nk_route=20,nk_proj=20,ioccupied_bands=[0,1])



