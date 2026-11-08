"""Full-LG finite-field driver reusing the model setup in spinHallCondInput.py."""
import os
for _name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[_name]='1'
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
from task import BZGridTask, solveIFTorque_BlochAcceleration_TDSE, solveIFTorque_LG_DM
from calculator import CollisionSpecification
from calculator.response_contractions import BLOCH_HBAR_EV_FS as HB

def run_bloch_scan(model, fermi, options):
    o=options
    minimum=np.where(np.asarray(model.mp_grid)>1,np.asarray(model.mp_grid)+1,1)
    if np.any(np.asarray(o.grid)<minimum):
        raise ValueError(f'BZGridTask requires --grid at least {tuple(minimum)} for this model')
    if o.benchmark_lg and not o.cw:
        raise ValueError('--benchmark-lg compares steady CW; use --cw')
    out=Path(o.output)
    out.mkdir(parents=True,exist_ok=True)
    config=out/'config.json'
    if config.exists():
        raise FileExistsError(f'{out} already contains a run; choose a new --output directory')
    provenance=dict(vars(o),fermi_eV=np.asarray(fermi).tolist(),
        model_norbs=int(model.norbs),model_mp_grid=np.asarray(model.mp_grid).tolist(),
        lattice_angstrom=np.asarray(model.lattice).tolist())
    provenance['model_sha256']={key:hashlib.sha256(np.ascontiguousarray(getattr(model,key)).view(np.uint8)).hexdigest()
        for key in ('ham_R','AA_R_W','exchange_R','spin')}
    config.write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    times=np.linspace(o.start,o.end,int(np.ceil((o.end-o.start)/o.dt))+1)
    eps=np.array({'x':(1,0,0),'y':(0,1,0),'xy+':(1,1j,0),'xy-':(1,-1j,0)}[o.polarization],complex)
    eps/=np.linalg.norm(eps)
    args=dict(fermi=np.asarray(fermi),omega=np.asarray(o.omega),eta=o.eta,kBT=o.kbt,
              collisionSpecification=CollisionSpecification(mode='crta',constantLinewidth=o.eta),
              finiteTemperatureOccupations=True)
    args['mag type']='spin'
    args['full covariant derivative']=True
    def task(function,extra):
        return BZGridTask(model,function,args={**args,**extra},BZGrid=tuple(o.grid),
            BZSymm=False,ncores=o.workers,parallelBackend='process',workerThreads=1)
    reference=None
    if o.benchmark_lg:
        dm=task(solveIFTorque_LG_DM,{})
        chi=np.mean([r['sigma'] for r in dm.results],axis=0)
        reference=np.einsum('fwabc,a,b->fwc',chi,eps.conj(),eps).real
        np.save(out/'lg_reference.npy',reference)
    summary=[]
    for field in o.fields:
        result=task(solveIFTorque_BlochAcceleration_TDSE,dict(times_fs=times,E0=field,
            polarization=eps,pulse_fwhm_fs=None if o.cw else o.fwhm,
            pulse_center_fs=o.center,phase_pair=True))
        arrays={key:np.mean([r[key] for r in result.results],axis=0)
                for key in ('torque','torque_even','torque_odd','impulse')}
        diagnostic_keys=[key for key in result.results[0]
            if key.startswith(('electron_number','electronic_energy','initial_band_populations'))]
        arrays.update({key:np.mean([r[key] for r in result.results],axis=0) for key in diagnostic_keys})
        stem=f'E{field:.8g}'
        np.savez_compressed(out/f'{stem}.npz',**arrays,times_fs=times,fermi=fermi,
            omega=o.omega,E0=field,polarization=eps,field=result.results[0]['field'],
            k_shift=result.results[0]['k_shift'],
            hermiticity_error=max(r['hermiticity_error'] for r in result.results))
        for im in range(len(fermi)):
            for iw,omega in enumerate(o.omega):
                even=arrays['torque_even'][im,iw]
                raw=arrays['torque'][im,iw]
                np.savetxt(out/f'{stem}_f{im}_w{iw}.txt',np.column_stack((times,raw,even,even/field**2)),
                    header='t_fs raw_x raw_y raw_z even_x even_y even_z even_x_over_E02 even_y_over_E02 even_z_over_E02')
                for suffix in ('','_minus'):
                    keys=['electron_number','electron_number_change','electronic_energy','electronic_energy_change']
                    np.savetxt(out/f'{stem}_f{im}_w{iw}_diagnostics{suffix}.txt',
                        np.column_stack([times]+[arrays[k+suffix][im,iw] for k in keys]),
                        header='t_fs N_electrons_per_cell delta_N energy_eV_per_cell delta_energy_eV_per_cell')
                    pop=arrays['initial_band_populations'+suffix][im,iw]
                    np.savetxt(out/f'{stem}_f{im}_w{iw}_populations{suffix}.txt',
                        np.column_stack((times,pop)),
                        header='t_fs '+' '.join(f'initial_band_{b}' for b in range(pop.shape[-1])))
                row=[field,im,omega,*arrays['impulse'][im,iw],* (arrays['impulse'][im,iw]/field**2)]
                if o.cw:
                    begin=times[-1]-o.average_cycles*2*np.pi*HB/omega
                    if begin<times[0]:
                        raise ValueError('CW duration is shorter than the requested averaging window')
                    window=np.r_[begin,times[times>begin]]
                    data=np.column_stack([np.interp(window,times,even[:,c]) for c in range(3)])
                    integrate=(np.trapezoid if hasattr(np, 'trapezoid') else np.trapz)
                    measured=integrate(data,window,axis=0)/(window[-1]-window[0])/(field**2/4)
                    row.extend(measured)
                    if reference is not None:
                        ref=reference[im,iw]
                        row.extend(ref)
                        row.append(np.linalg.norm(measured-ref)/max(np.linalg.norm(ref),1e-30))
                summary.append(row)
        header='E0 fermi_index photon_eV impulse_x impulse_y impulse_z impulse_x_over_E02 impulse_y_over_E02 impulse_z_over_E02'
        if o.cw: header+=' CW_x_over_E02quarter CW_y_over_E02quarter CW_z_over_E02quarter'
        if reference is not None: header+=' LG_x LG_y LG_z relative_vector_error'
        np.savetxt(out/'summary.txt',summary,header=header)
        print(f'Finished E0={field:g}; results: {out}',flush=True)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fields',type=float,nargs='+',default=[.001])
    p.add_argument('--omega',type=float,nargs='+',default=[.75])
    p.add_argument('--grid',type=int,nargs=3,default=[12,12,1])
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--dt',type=float,default=.1,help='time step, fs; converge by halving')
    p.add_argument('--start',type=float,default=-150.)
    p.add_argument('--end',type=float,default=250.)
    p.add_argument('--center',type=float,default=0.)
    p.add_argument('--fwhm',type=float,default=50.,help='intensity FWHM, fs')
    p.add_argument('--eta',type=float,default=.06582)
    p.add_argument('--kbt',type=float,default=.026)
    p.add_argument('--polarization',choices=['x','y','xy+','xy-'],default='xy+')
    p.add_argument('--cw',action='store_true')
    p.add_argument('--benchmark-lg',action='store_true')
    p.add_argument('--average-cycles',type=int,default=5)
    p.add_argument('--output',default='mcalc/bloch_acceleration')
    return p

if __name__=='__main__':
    from multiprocessing import freeze_support
    freeze_support()
    options=parser().parse_args()
    if any(not np.isfinite(e) or e<=0 for e in options.fields) or len(set(options.fields))!=len(options.fields):
        raise ValueError('--fields must contain distinct positive finite amplitudes')
    from spinHallCondInput import run
    run(bloch_options=options)
