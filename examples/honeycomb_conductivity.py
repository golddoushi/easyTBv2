"""Compare linear dc/ac LG, VG, and real-time conductivity on the honeycomb model.

python examples/honeycomb_conductivity.py --grid 24 --tdse-grid 4
The TDSE and its VG reference always use exactly the same mesh.
"""
import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.honeycomb_four_band import build_model
from calculator import (AnormalousHallConductance_LG_DM, AnormalousHallConductance_VG_DM,
                        Opticalconductivity_LG_DM, Opticalconductivity_VG_DM, Opticalconductivity_TDSE)


def relative_error(value, reference):
    return float(np.max(np.abs(value-reference))/max(np.max(np.abs(reference)),1e-30))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='tmp/honeycomb_conductivity')
    parser.add_argument('--grid', type=int, default=24)
    parser.add_argument('--tdse-grid', type=int, default=4)
    parser.add_argument('--omega', type=float, nargs='+', default=[.6,1.3])
    parser.add_argument('--fermi', type=float, default=-.7)
    parser.add_argument('--eta', type=float, default=.2)
    parser.add_argument('--kBT', type=float, default=.15)
    parser.add_argument('--field', type=float, default=1e-4)
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--skip-tdse', action='store_true')
    parser.add_argument('--snapshots', type=int, default=0,
                        help='Optional frozen-displacement LG ac average; this is not an inelastic EPC collision calculation')
    parser.add_argument('--supercell', type=int, default=2)
    parser.add_argument('--temperature', type=float, default=300.)
    args = parser.parse_args()
    if args.grid < 1 or args.tdse_grid < 1 or args.supercell < 1 or args.snapshots < 0:
        parser.error('grids and supercell must be positive; snapshots must be nonnegative')
    output = Path(args.output)
    with contextlib.redirect_stdout(io.StringIO()):
        model = build_model(output/'model')
    options = dict(model=model, kgrid=(args.grid,args.grid,1),eta=args.eta,kBT=args.kBT,
                   fermiList=np.array([args.fermi]),omegaList=np.array(args.omega),fullCovariantDerivative=True)
    lgdc = AnormalousHallConductance_LG_DM(**options)
    vgdc = AnormalousHallConductance_VG_DM(**options)
    lg = Opticalconductivity_LG_DM(**options)
    vg = Opticalconductivity_VG_DM(**options)
    print(f"{lgdc.sigma=}\n {vgdc.sigma=}\n {lg.sigma=}\n {vg.sigma=}\n")
    arrays = dict(omega_eV=np.array(args.omega),fermi_eV=np.array([args.fermi]),
                  sigma_dc_LG=lgdc.sigma,sigma_dc_VG=vgdc.sigma,
                  sigma_ac_LG=lg.sigma,sigma_ac_VG=vg.sigma,
                  sum_rule_residual_S_per_m_eV=vg.sum_rule_residual)
    summary = dict(units='S/m',axes='fermi, omega (ac only), drive, current',grid=args.grid,
                   eta_eV=args.eta,kBT_eV=args.kBT,dc_relative_error=relative_error(vgdc.sigma,lgdc.sigma),
                   ac_LG_VG_relative_error=relative_error(vg.sigma,lg.sigma))
    if not args.skip_tdse:
        small = dict(options,kgrid=(args.tdse_grid,args.tdse_grid,1))
        reference = Opticalconductivity_VG_DM(**small)
        td = Opticalconductivity_TDSE(E0=args.field,n_steps_per_cycle=args.steps,recordCurrent=True,**small)
        print(f"{td.sigma=}")
        arrays.update(sigma_ac_VG_tdse_mesh=reference.sigma,sigma_ac_TDSE=td.sigma)
        for (fi,wi,a), trace in td.time_traces.items():
            for key, value in trace.items():
                arrays[f'trace_f{fi}_w{wi}_a{a}_{key}'] = value
        summary.update(tdse_grid=args.tdse_grid,tdse_relative_error=relative_error(td.sigma,reference.sigma),
                       field_peak_V_per_A=args.field,integration_info=td.integration_info)
    if args.snapshots:
        sc = copy.deepcopy(model)
        sc.make_supercell(np.diag([args.supercell,args.supercell,1]))
        snapshot_options = dict(options, model=sc)
        arrays['sigma_supercell_equilibrium'] = Opticalconductivity_LG_DM(**snapshot_options).sigma
        samples = []
        displacements = sc.sampleModelDisplacements(args.temperature,nSamples=args.snapshots,seed=2026)
        for u in displacements:
            snapshot_options['model'] = sc.makeDisplacedModel(u)
            samples.append(Opticalconductivity_LG_DM(**snapshot_options).sigma)
        samples = np.asarray(samples)
        arrays.update(sigma_snapshots=samples,sigma_snapshot_mean=samples.mean(axis=0),displacements_A=displacements)
        if args.snapshots > 1:
            arrays['sigma_snapshot_stderr_real'] = samples.real.std(axis=0,ddof=1)/np.sqrt(args.snapshots)
            arrays['sigma_snapshot_stderr_imag'] = samples.imag.std(axis=0,ddof=1)/np.sqrt(args.snapshots)
        print(f"LG_DM sigma of stochastic displaced supercell {args.supercell}x{args.supercell}x1:\n {np.mean(samples,axis=0,)}")
        summary.update(snapshots=args.snapshots,temperature_K=args.temperature,
                       snapshot_model='static frozen SSH disorder plus unchanged CRTA eta')
    np.savez(output/'conductivity.npz',**arrays)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))
    print(f'Outputs: {output.resolve()}')


if __name__ == '__main__':
    main()
