from system import System_tb
from calculator import ExchangeCoupling

mainDir="./mcalc/1D-chain-3orb/"
model = System_tb(
        mainDir=mainDir,
        #hamR="BN_fm_soc",
        cutoff=-0.002,
        poscar="POSCAR",
        projections={
            "B":(0,),
            "N":(0,),
            "H":(0,),
        },
        ispin=2
)
fermiList=[0,]
routes=model.classifyExchangeRoutes(sourceAtoms=[2,], pbc=(0,), maxRange=5, calRange=5, )
exCoup = ExchangeCoupling(model=model, kgrid=(28, 1, 1), 
                          fermiList = fermiList, eta=0.0001,
                          exchangeRoutes = routes, symmetricExchangeSign=1,
                          lowOrder=60,highOrder=80,Jcvg=-1e-4)

for ifermi, fermi in enumerate(fermiList):
    print(f"{fermi=:.4f}")
    for ibondType, ex in enumerate(exCoup.exchangeRoutes):
        print("bondType%d ifermi%d orb%d-orb%d[%2d%2d%2d] len: %.2f J=%.4f Jax=%.4f Jay=%.4f Jaxy=%.4f Dx=%.4f Dy=%.4f"%(
                        ex["route_type"],ifermi,ex["source_atom"],ex["target_atom"],*ex["overlat"],ex["bond_length"],
                        ex["exchangeEnergy"][ifermi]/2*1000,ex["Jani"][ifermi,0,0]/2*1000,ex["Jani"][ifermi,1,1]/2*1000,
                        ex["Jani"][ifermi,0,1]/2*1000,ex["DMI"][ifermi,0]/2*1000,ex["DMI"][ifermi,1]/2*1000))