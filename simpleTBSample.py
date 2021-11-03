import paramReader
paramReader.readGlobalConst()
import posReader as pR
import hrReader as reader
from ham import Green

pR.readPOS('./mcalc/FeCl2/POSCAR')
hr=reader.Hr()
hr.readSingleFile('./mcalc/FeCl2/wannier90_hr.dat',ignoreImagPart=False,cutoff=-0.01)

gr=Green()
gr.calcFullBands_CDLL(hr,fermi=-2.1,mainDir='./mcalc/FeCl2/',saveFig=True)