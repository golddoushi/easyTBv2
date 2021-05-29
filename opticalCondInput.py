import paramReader
paramReader.readGlobalConst()
import posReader as pR
import hrReader as reader
from ham import Green

pR.readPOS('./mcalc/FeCl2/POSCAR')
hr=reader.Hr()
hr.readSingleTBFile('./mcalc/FeCl2/wannier90.dn_tb.dat')

gr=Green()
gr.calcOpticalHoppingTensor_CDLL(hr,fermi=-2.5,nk=100,ibandv=6,ibandc=7)