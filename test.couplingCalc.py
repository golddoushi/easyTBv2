import matplotlib
matplotlib.use('Agg')
import sys
import paramReader
paramReader.readGlobalConst()
import hrReader as reader
from ham import Green
import posReader as pR
import couplingCalc as cc
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
# check the band structures
pR.readPOS('./mcalc/spinchain/POSCAR')
hr=reader.Hr()
hr.readDoubleFiles('./mcalc/spinchain/hr.up.dat','./mcalc/spinchain/hr.dn.dat',cutoff=-0.01)
gr=Green()
gr.calcFullBands_CDLL(hr,fermi=0,mainDir='./mcalc/spinchain/',saveFig=True,redBands=[0,1])
exit()
'''
'''
cc.classifyBonds(mainDir='./mcalc/spinchain/', source_list=[0,1], 
           wann_list=[
                      [ 0,],
                      [ 1,],
                     ], 
           ini_S=[1,1])

cc.calcIsotropicJ_GaussianQuadrature(mainDir='./mcalc/spinchain/',hrcutoff=-1,nthreads=4,
    sc=[30,1,1],fermi=0,eta=0.0001,erange=(-10,5),lowOrder=20,highOrder=50,Jcvg=-1e-4)
exit()
'''
'''
cc.classifyBonds(mainDir='./mcalc/FeCl2/', source_list=[0,], 
           wann_list=[
                      [ 0, 1, 2, 3, 4],
                      [ 5, 6, 7],
                      [ 8, 9,10],
                     ], 
           ini_S=[2,0,0])

#cc.calcIsotropicJ_GaussianQuadrature(mainDir='./mcalc/FeCl2/',hrcutoff=-1,nthreads=4,
#    sc=[20,20,1],fermi=-2,eta=0.0001,erange=(-12,10),lowOrder=20,highOrder=50,Jcvg=-1e-4)
cc.calcJFull_GaussianQuadrature(mainDir='./mcalc/FeCl2/',hrcutoff=-1,nthreads=4,
    sc=[20,20,1],fermi=-2,eta=0.0001,erange=(-12,10),lowOrder=20,highOrder=50,Jcvg=-1e-4)
exit()
'''
cc.classifyBonds(mainDir='./mcalc/Fe3GeTe2/', source_list=[0,1,2], 
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
    sc=[32,32,1],fermi=0.04758627,eta=0.0001,erange=(-8,10),
    lowOrder=20,highOrder=50,Jcvg=-1e-3)
#cc.calcJFull_GaussianQuadrature(mainDir='./mcalc/Fe3GeTe2/',hrcutoff=-1,nthreads=4,
#    sc=[32,32,1],fermi=0.04758627,eta=0.0001,erange=(-8,10),
#    lowOrder=20,highOrder=50,Jcvg=-1e-4)