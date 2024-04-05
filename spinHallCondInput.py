import os
os.add_dll_directory("C:/Program Files/mingw64/bin")
import paramReader
paramReader.readGlobalConst()
import posReader as pR
import hrReader as reader
from ham import Green, Response
import numpy as np

from system import System_w90
from calculator import SpinHallConductance

mainDir="./mcalc/Pt/"
model=System_w90(
    mainDir=mainDir,
    hamR="wannier90",
    cutoff=-0.002,
    poscar="POSCAR",
    projections={
        "Pt":(0,1,2),
    },
    ispin=2
)

NK=24
fermi=10.96914665
fermiAxis=np.linspace(fermi-2,fermi+2,101)
calc=SpinHallConductance(model,kgrid=(NK,NK,NK),eta=0.026,fermiList=fermiAxis)
import matplotlib.pyplot as plt
plt.plot(fermiAxis,calc.sigma)
plt.show()