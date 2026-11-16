import paramReader
paramReader.readGlobalConst()
import numpy as np
from system import System_omx, System_w90, System_tb
from fileio import DOSCAR


def setModel():
    # Pbene, spin hall conductance
    mainDir="./mcalc/FeCl2/"
    model=System_w90(
        mainDir=mainDir,
        #hamR="omx",
        #hamR="BNcol",
        #hamR="wannier90_mx",
        cutoff=-0.002,
        poscar="POSCAR",
        projections={
            "B":(0,),
            "N":(0,),
            "Ge":(0,0,1,2),
            #"Mn":(0,0,0,1,1,2,),
            "Mn":(0,1,2,),
            "Sn":(0,1,),
            #"Sn":(0,0,0,1,1,2,),
            "Pb":(0,1,),
            #"Pb":(0,0,0,1,1,2,2,),
            "Pt":(0,1,2,),
            #"Pt":(0,0,1,1,2,2,),
            "C":(0,1,),
            "Fe":(0,1,2,),
            "Co":(0,1,2,),
            "Mo":(2,),
            "Nb":(0,2,),
            "S":(1,),
            "Bi":(0,1,),
            "Te":(0,1,),
            #"Fe":(0,0,1,1,2,2,),
            "Cl":(1,),
            "Se":(1,),
            #"Te":(1,),
            #"Cl":(0,0,1,1,2,3),
            "H":(0,),
            #"H":(0,0,1,),
            "Au":(0,0,0,1,1,2,2,),
            "Sr":(0,0,0,1,1,2,2,),
            "Ti":(0,0,0,1,1,2,),
             "O":(0,0,1,1,2,),
        },
        ispin=2,
        spinSeparated=True,
    )
    #'''
    # Symmetrize the direction-independent collinear and orbital-SOI
    # ingredients once, then assemble any requested magnetization direction.
    model.symmerizeCollinearAndSOI()
    model.getExchangeR(provenance="pauli estimate")
    magnetizationTheta ,magnetizationPhi = 5/180*np.pi, 0/180*np.pi
    model.includeSOCFromSOI(theta=magnetizationTheta,phi=magnetizationPhi,strength=1,keepRegularSpinAxis=True)
    #model.rotateExchangeDirection(theta=magnetizationTheta,phi=magnetizationPhi)
    #model.getExchangeR(provenance="explicit")
    #'''

    # include el-ph self energy
    #model.includeElectronSelfEnergy(f"{mainDir}/vaspout.h5",temperature=300.,)

    # For an independent noncollinear VASP model, rotate its SAXIS frame manually
    # NOTE: this is only for independent vasp_ncl + wannier90 scheme, DO NOT apply for vasp_std + V_soi scheme
    #model.rotateSpinBasisToGlobal(alpha=np.pi/2, beta=np.pi/2) # SAXIS = 0 1 0
    #model.rotateSpinBasisToGlobal(alpha=0, beta=np.pi/2) # SAXIS = 1 0 0
    
    #'''
    # global symmetrization
    # usage: regularized independent noncollinear model, collinear model
    # NOT for vasp_std + V_soi scheme, use the symmerizeCollinearAndSOI rather
    atomicMagMomList=[
        np.array([0, 0, 0]),
    ]*4+[
        np.array([0, 0, 1]),
    ]*3
    #model.symmerizeHamAndOverlap(
    #    atomicMagMomList=atomicMagMomList,spinor=True,print_row=(0,1,9,10,),print_col=(0,1,9,10,),
    #)
    #'''

    # modulate the SOC, exchange fields, baised fields, etc.
    #magnetizationTheta ,magnetizationPhi = 0.0, 0.0
    #model.getExchangeR(provenance="explicit")
    #model.scaleSOC(lam=0)
    #model.rotateExchangeField(projection=[(0,(0,)),(1,(0,))], theta=np.pi, phi=0.)
    #model.applyElectrostaticPotentialAtOnsites([0., 0., 0.,]) # in eV/angstrom
    '''
    # create supercell model and cut into nanoribbon
    Tmatrix=[
        [10, 0, 0], # The A1 of the supercell in the frame of primitive cell
        [ 1, 2, 0], # The A2 of the supercell in the frame of primitive cell
        [ 0, 0, 1]  # The A3 of the supercell in the frame of primitive cell
    ]
    model.make_supercell(Tmatrix)
    surfaceDir=0
    model.shiftOrigin(surfaceDir=surfaceDir,)
    model.cutEdge(cutDir=surfaceDir,)
    # add exchange splitting for edge atoms
    #model.addExchangeSplitting(edgeRegion1,[ 0.0, 0.5, 0.0])
    #model.addExchangeSplitting(edgeRegion2,[ 0.0,-0.5, 0.0])
    #exit()
    #'''
    #fermi=0  # OPNEMX generated wannier model always displays zero fermi energy
    #fermi=-0.167662613970*27.2113845 # OPENMX Ge
    #fermi=-3.85                       # OPENMX Sn topological insulator
    #fermi=-4.2                         # OPENMX Pb topological insulator
    #fermi=-0.17020527415105*27.2113845 # OPENMX PbSnH topological insulator
    #fermi=-3.89                         # OPENMX PbSn offset topological insulator
    #fermi=-0.16790565257604*27.2113845 # OPENMX Pt
    #fermi=-0.19908723865958*27.2113845 # OPENMX FeCl2
    #fermi=-0.22660759670266*27.2113845 # OPENMX AuH 111
    #fermi=-0.19580173375979*27.2113845 # OPENMX Au 111
    #fermi=-2.41861676 # VASP Sn
    #fermi=-2.5 # VASP Pb
    #fermi= 5.65354180 # VASP Ta
    #fermi= 7.72024255 # VASP W
    #fermi=10.96516405 # VASP Pt
    #fermi=-2.63611399 # VASP PbSnH
    #fermi=-2.345       # VASP PbSn
    fermi=-3.0         # VASP 2H-FeCl2 U=2.0
    #fermi=0.95         # VASP SL-MnBi2Te4 U=4.0
    #fermi=6.44577045         # VASP Mn3Sn
    #fermi=6.65112400         # VASP Mn3SnH
    #fermi=-0.09570791         # VASP Mn3Sn_2layer
    #fermi=-0.32817881         # VASP Mn3Sn_2layer_Vsn
    #fermi=-0.27313134         # VASP Mn3Sn_2layer_Vmn
    #fermi=9.35224895           # VASP FePt
    #fermi=9.04361448           # VASP FePtH
    #fermi=-0.20789123834373*27.2113845  # OPENMX FePt
    #fermi=-0.20789123834373*27.2113845  # OPENMX FePtH
    #fermi=DOSCAR(mainDir+"DOSCAR").fermi   # Fermi from DOSCAR
    #fermi=0            # model hamiltonian
    model.setFermi(fermi)

    return model, mainDir