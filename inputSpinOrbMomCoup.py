"""compute SOC coupling (ground state) using modelSetting setup.
Usage:
1. define the model in modelSetting.py
2. python3 inputSpinOrbMomCoup.py --grid 48 48 1  --emin -2 --emax 2 --estep 0.05

grid defines the k-mesh of BZ sampling
The energy window defined by emin, emax is relative to fermi, as associated with model
"""
import os
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[name]='1'
import sys
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from calculator import CollisionSpecification
from task import BZGridTask, solveSpinOrbitalCoupling

def run_moment(model,mainDir,o):
    out=Path(mainDir,o.subdir)
    out.mkdir(parents=True,exist_ok=True)

    fermi=np.arange(model.fermi+o.emin, model.fermi+o.emax, o.estep)
    config=dict(vars(o),fermi_eV=np.asarray(fermi).tolist(),)
    config['model_sha256']={k:hashlib.sha256(np.ascontiguousarray(getattr(model,k)).view(np.uint8)).hexdigest()
                            for k in ['ham_R','AA_R_W']}
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    calc=BZGridTask(model, solveSpinOrbitalCoupling,
                   args={'fermi':np.asarray(fermi),
                         'eta':o.eta,'kBT':o.kbt,
                         'collisionSpecification':CollisionSpecification(mode='crta',constantLinewidth=o.eta),
                         'skip low':False,
                         'record frequency ids':[],'record fermi ids':[]},
                BZGrid=tuple(o.grid),BZSymm=False,
                ncores=o.workers,parallelBackend='thread' if sys.platform=='win32' else 'process',workerThreads=1)
    stored=np.mean([result['spin-orbital coupling'] for result in calc.results],axis=0)
    
    np.savetxt(out/f'spin_orbital_coupling.txt',
                   np.column_stack(
                       (fermi-model.fermi,            stored)),
            header='mu_minus_reference_eV ACA_LdotS_over_hbar2_per_cell')
    return

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--subdir',default='SOC')
    p.add_argument('--grid',type=int,nargs=3,default=[48,48,1])
    p.add_argument('--emin',type=float,default=-2)
    p.add_argument('--emax',type=float,default=+2)
    p.add_argument('--estep',type=float,default=0.05)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--eta',type=float,default=.06582)
    p.add_argument('--kbt',type=float,default=.026)
    options=p.parse_args()

    from modelSetting import setModel
    model, mainDir = setModel()
    run_moment(model,mainDir,options)

if __name__=='__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
