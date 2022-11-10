import matplotlib
matplotlib.use('Agg')
import sys
import paramReader
paramReader.readGlobalConst()
import hrReader as reader
import ham as hamilton
import numpy as np
import posReader as pR
import couplingCalc as cc
import matplotlib.pyplot as plt
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.filename=filename
        self.log = open(filename, 'w')
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.close()
        self.log=open(self.filename,'a')
    
    def flush(self):
        self.terminal.flush()
        self.log=open(self.filename,'w')
        self.log.close()
sys.stdout = Logger('log')
'''
cc.classifyBonds(mainDir='./mcalc/FeCl2/', source_list=[0,], 
           wann_list=[
                      [ 0, 1, 2, 3, 4],
                      [ 5, 6, 7],
                      [ 8, 9,10],
                     ], 
           ini_S=[2,0,0])

cc.calcIsotropicJ_GaussianQuadrature(mainDir='./mcalc/FeCl2/',hrcutoff=-1,nthreads=4,
    sc=[10,10,1],fermi=-2,eta=0.0001,erange=(-12,10),lowOrder=12,highOrder=50,Jcvg=-1e-4)
#cc.calcIsotropicJ(mainDir='./mcalc/FeCl2/',hrcutoff=-1,sc=[10,10,1],fermi=-2,eta=0.0001,
#    erange=(-12,10), emptySpace=0)


exit()
'''
cc.classifyBonds(mainDir='./mcalc/Fe3GeTe2/', source_list=[0,1,2,], 
           wann_list=[
                      [ 0, 1, 2, 3, 4],
                      [ 5, 6, 7, 8, 9],
                      [10,11,12,13,14],
                      [15,16,17],
                      [18,19,20],
                      [21,22,23],
                     ], 
           ini_S=[1.5,1.5,1.0,0,0,0])

cc.calcIsotropicJ_GaussianQuadrature(mainDir='./mcalc/Fe3GeTe2/',hrcutoff=-1,nthreads=4,
    sc=[32,32,1],fermi=0.04758627,eta=0.0001,erange=(-8,10),lowOrder=20,highOrder=50,Jcvg=1e-3)
