import paramReader
paramReader.readGlobalConst()
import posReader as pR
import hrReader as reader
from ham import Green

pR.readPOS('./mcalc/BN/POSCAR')
hr=reader.Hr()
hr.readSingleFile('./mcalc/BN/hr.dat',ignoreImagPart=False,cutoff=-0.01)

gr=Green()
gr.calcFullBands_CDLL(hr,fermi=0,mainDir='./mcalc/BN/',saveFig=True)