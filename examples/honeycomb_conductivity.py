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
from scipy.constants import Boltzmann, electron_volt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.honeycomb_four_band import build_model
from calculator import (AnormalousHallConductance_LG_DM, AnormalousHallConductance_VG_DM,
                        Opticalconductivity_LG_DM, Opticalconductivity_VG_DM, Opticalconductivity_TDSE)
from calculator import CollisionSpecification, ModelEPCLinewidth


def relative_error(value, reference):
    imax=np.argsort(np.abs(reference.flatten()))[-4:]
    diff=np.abs(value-reference).flatten()[imax]
    return max(diff/np.abs(reference.flatten())[imax] )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='tmp/honeycomb_conductivity')
    parser.add_argument('--grid', type=int, default=24)
    parser.add_argument('--tdse-grid', type=int, default=4)
    parser.add_argument('--omega', type=float, nargs='+', default=[.6,1.3])
    parser.add_argument('--fermi', type=float, default=-.7)
    parser.add_argument('--constantLinewidth', type=float, default=.2)
    parser.add_argument('--kBT', type=float, default=None,
                        help='Electronic kBT in eV; default is Boltzmann*temperature')
    parser.add_argument('--field', type=float, default=1e-4)
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--skip-tdse', action='store_true')
    parser.add_argument('--snapshots', type=int, default=0,
                        help='Optional frozen-displacement LG ac average; this is not an inelastic EPC collision calculation')
    parser.add_argument('--supercell', type=int, default=2)
    parser.add_argument('--temperature', type=float, default=300.)
    parser.add_argument('--serta', choices=('none','static','dynamic','both'), default='both',
                        help='Compute toy-EPC SERTA optical and dc tensors')
    parser.add_argument('--serta-qgrid', type=int, nargs=3, default=(6,6,1))
    parser.add_argument('--epc-delta', type=float, default=.03,
                        help='Lorentzian delta broadening used to construct EPC linewidths, eV')
    parser.add_argument('--tdse-collision', choices=('crta','static','dynamic'), default='crta')
    parser.add_argument('--tdse-min-linewidth', type=float, default=.02,
                        help='Floor in eV for optional SERTA TDSE only; set 0 for unmodified rates')
    args = parser.parse_args()
    output = Path(args.output)
    with contextlib.redirect_stdout(io.StringIO()):
        model = build_model(output/'model')
    kBT = Boltzmann*args.temperature/electron_volt if args.kBT is None else args.kBT
    options = dict(model=model, kgrid=(args.grid,args.grid,1),kBT=kBT,
                   CollisionSpecification=CollisionSpecification(mode="crta",constantLinewidth=args.constantLinewidth),
                   fermiList=np.array([args.fermi]),omegaList=np.array(args.omega),fullCovariantDerivative=True)
    lgdc = AnormalousHallConductance_LG_DM(**options)
    vgdc = AnormalousHallConductance_VG_DM(**options)
    lg = Opticalconductivity_LG_DM(**options)
    vg = Opticalconductivity_VG_DM(**options)
    #print(f"conductivity with CRTA, {args.constantLinewidth=}\n{lgdc.sigma=}\n {vgdc.sigma=}\n {lg.sigma=}\n {vg.sigma=}\n")
    arrays = dict(omega_eV=np.array(args.omega),fermi_eV=np.array([args.fermi]),
                  sigma_dc_LG=lgdc.sigma,sigma_dc_VG=vgdc.sigma,
                  sigma_ac_LG=lg.sigma,sigma_ac_VG=vg.sigma,
                  sum_rule_residual_S_per_m_eV=vg.sum_rule_residual)
    summary = dict(units='S/m',axes='fermi, omega (ac only), drive, current',grid=args.grid,
                   constantLinewidth_eV=args.constantLinewidth,kBT_eV=kBT,
                   dc_LG_VG_relative_error=relative_error(vgdc.sigma,lgdc.sigma),
                   ac_LG_VG_relative_error=relative_error(vg.sigma,lg.sigma),
                   max_ac_LG_conductivity=np.max(abs(lg.sigma)),
                   max_ac_VG_conductivity=np.max(abs(vg.sigma)),
                   )
    requested = (() if args.serta == 'none' else ('static','dynamic') if args.serta == 'both' else (args.serta,))
    serta_sources = {}
    for name in requested:
        source = ModelEPCLinewidth(qgrid=args.serta_qgrid,temperature=args.temperature,
                                   fermi=args.fermi,delta=args.epc_delta,adiabatic=name=='static')
        collision = CollisionSpecification('serta',linewidthSource=source)
        serta_options = dict(options,collisionSpecification=collision)
        slg = Opticalconductivity_LG_DM(**serta_options)
        svg = Opticalconductivity_VG_DM(**serta_options)
        sdc = AnormalousHallConductance_VG_DM(**serta_options)
        gamma = slg.density_matrix_kernel.collision.population_linewidth
        serta_sources[name] = source
        arrays[f'linewidth_{name}_eV'] = gamma
        arrays[f'sigma_ac_LG_SERTA_{name}'] = slg.sigma
        arrays[f'sigma_ac_VG_SERTA_{name}'] = svg.sigma
        arrays[f'sigma_dc_VG_SERTA_{name}'] = sdc.sigma
        summary[f'serta_{name}'] = dict(linewidth_min_eV=float(gamma.min()),
                                          linewidth_max_eV=float(gamma.max()),
                                          ac_LG_VG_relative_error=relative_error(svg.sigma,slg.sigma),
                                          change_from_CRTA=relative_error(slg.sigma,lg.sigma))
    if not args.skip_tdse:
        small = dict(options,kgrid=(args.tdse_grid,args.tdse_grid,1))
        tdse_collision = None
        gamma_floor = None
        if args.tdse_collision != 'crta':
            source = ModelEPCLinewidth(qgrid=args.serta_qgrid,temperature=args.temperature,
                                       fermi=args.fermi,delta=args.epc_delta,
                                       adiabatic=args.tdse_collision=='static')
            owner = Opticalconductivity_LG_DM(**small)
            gamma_floor = np.maximum(source.populationLinewidth(owner),args.tdse_min_linewidth)
            tdse_collision = CollisionSpecification('serta',linewidthSource=gamma_floor)
            small['collisionSpecification'] = tdse_collision
        reference = Opticalconductivity_VG_DM(**small)
        td = Opticalconductivity_TDSE(E0=args.field,n_steps_per_cycle=args.steps,recordCurrent=True,**small)
        #print(f"{td.sigma=}")
        arrays.update(sigma_ac_VG_tdse_mesh=reference.sigma,sigma_ac_TDSE=td.sigma)
        for (fi,wi,a), trace in td.time_traces.items():
            for key, value in trace.items():
                arrays[f'trace_f{fi}_w{wi}_a{a}_{key}'] = value
        summary.update(tdse_grid=args.tdse_grid,ac_tdse_vg_relative_error=relative_error(td.sigma,reference.sigma),
                       tdse_collision=args.tdse_collision,
                       tdse_min_linewidth_eV=(None if gamma_floor is None else float(gamma_floor.min())),
                       field_peak_V_per_A=args.field,integration_info=td.integration_info,
                       max_ac_VG_small_conductivity=np.max(abs(reference.sigma)),
                       max_ac_TD_small_conductivity=np.max(abs(td.sigma)),
                   )
    if args.snapshots:
        sc = copy.deepcopy(model)
        sc.make_supercell(np.diag([args.supercell,args.supercell,1]))
        if args.grid % args.supercell:
            parser.error('--grid must be divisible by --supercell for a commensurate snapshot comparison')
        snapshot_grid=(args.grid//args.supercell,args.grid//args.supercell,1)
        snapshot_options = dict(options, model=sc, kgrid=snapshot_grid)
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
        #print(f"LG_DM sigma of stochastic displaced supercell {args.supercell}x{args.supercell}x1:\n {np.mean(samples,axis=0,)}")
        summary.update(snapshots=args.snapshots,temperature_K=args.temperature,
                       snapshot_kgrid=snapshot_grid,primitive_equivalent_grid=args.grid,
                       snapshot_model='static frozen SSH disorder plus unchanged CRTA eta',
                       supercell_equilibrium_relative_error=relative_error(
                           arrays['sigma_supercell_equilibrium'],lg.sigma),
                       snapshot_change_from_equilibrium=relative_error(samples.mean(axis=0),arrays['sigma_supercell_equilibrium']))
        for name in requested:
            summary[f'snapshot_vs_{name}_serta_relative_difference'] = relative_error(
                samples.mean(axis=0),arrays[f'sigma_ac_LG_SERTA_{name}'])
            combined_collision=CollisionSpecification(
                'serta',linewidthSource=arrays[f'linewidth_{name}_eV']+args.constantLinewidth)
            combined=Opticalconductivity_LG_DM(
                **dict(options,collisionSpecification=combined_collision))
            arrays[f'sigma_ac_LG_{name}_RTA_plus_snapshot_background'] = combined.sigma
            summary[f'snapshot_vs_{name}_rta_matched_background_relative_difference'] = relative_error(
                samples.mean(axis=0),combined.sigma)
            if args.snapshots > 1:
                stderr=np.hypot(arrays['sigma_snapshot_stderr_real'],arrays['sigma_snapshot_stderr_imag'])
                discrepancy=np.abs(samples.mean(axis=0)-arrays[f'sigma_ac_LG_SERTA_{name}'])
                active=stderr>1e-14*np.max(stderr)
                summary['snapshot_max_stderr_relative_to_mean'] = float(
                    np.max(stderr)/max(np.max(np.abs(samples.mean(axis=0))),1e-30))
                summary['snapshot_max_discrepancy_in_standard_errors'] = float(
                    np.max(discrepancy[active]/stderr[active]))
    np.savez(output/'conductivity.npz',**arrays)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))
    print(f'Outputs: {output.resolve()}')


if __name__ == '__main__':
    main()
