"""Compare linear dc/ac LG, VG, and real-time conductivity on the honeycomb model.

python examples/honeycomb_conductivity.py --grid 24 --tdse-grid 4
The TDSE and its VG reference always use equivalent electronic meshes.
For moving ions: --phonon-dynamics (default 4x4x1 supercell, Gamma sampling).
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
from calculator import Opticalconductivity_VG_DM_Phonon
from auxiliary import showElapsedTime

def relative_error(value, reference):
    # for 2D system, the xx, xy, yx, yy components of sigma at specific Fermi specific photon energy
    # can be nonzero in general cases. So, here select the largest 4 components to compute the 
    # relative errors.
    reference_line = np.abs(reference.flatten())
    difference_line= np.abs(value-reference).flatten()
    imax = np.argsort(reference_line)[-4:]
    return tuple(difference_line[imax] / reference_line[imax])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='tmp/honeycomb_conductivity')
    parser.add_argument('--grid', type=int, default=32,
                        help="kgrid of primitive cell")
    parser.add_argument('--tdse-grid', type=int, default=4,
                        help="supercell size used for ion-clamped TDSE")
    parser.add_argument('--omega', type=float, nargs='+', default=[.6,1.3])
    parser.add_argument('--fermi', type=float, default=-.7)
    parser.add_argument('--constantLinewidth', type=float, default=.0658)
    parser.add_argument('--kBT', type=float, default=None,
                        help='Electronic kBT in eV; default is Boltzmann*temperature')
    parser.add_argument('--field', type=float, default=1e-4)
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--skip-tdse', action='store_true')
    parser.add_argument('--snapshots', type=int, default=0,
                        help='Optional frozen-displacement LG ac average; this is not an inelastic EPC collision calculation')
    parser.add_argument('--supercell', type=int, default=4,
                        help="supercell size for stochastic displaced model, and for the ion-dynamic TDSE")
    parser.add_argument('--phonon-dynamics', action='store_true',
                        help='Propagate electrons in a moving harmonic SSH supercell bath')
    parser.add_argument('--phonon-grid', type=int, default=1,
                        help='kgrid of the supercell model in ion-dynamic TDSE')
    parser.add_argument('--phonon-trajectories', type=int, default=4)
    parser.add_argument('--phonon-statistics', choices=('classical', 'wigner'), default='classical')
    parser.add_argument('--phonon-cycles', type=float, default=1.,
                        help='Minimum measurement duration in periods of the slowest phonon')
    parser.add_argument('--phonon-seed', type=int, default=2026)
    parser.add_argument('--phonon-phase-points', type=int, default=33,
                        help='Odd Fourier phase-grid size for the matched frequency-domain Liouville solver')
    parser.add_argument('--phonon-solver-tolerance', type=float, default=1e-9)
    parser.add_argument('--measure-cycles', type=int, default=2)
    parser.add_argument('--transient-lifetimes', type=float, default=12.)
    parser.add_argument('--temperature', type=float, default=300.)
    parser.add_argument('--serta', choices=('none','static','dynamic','both'), default='both',
                        help='Compute toy-EPC SERTA optical and dc tensors')
    parser.add_argument('--serta-qgrid', type=int, default=6)
    parser.add_argument('--epc-delta', type=float, default=.03,
                        help='Lorentzian delta broadening used to construct EPC linewidths, eV')
    parser.add_argument('--epc-markov', choices=('none','quantum','classical','wigner'), default='none',
                        help='Additional stationary Pauli collision calculation beyond SERTA')
    parser.add_argument('--epc-qgrid', type=int, default=4,
                        help='Markov phonon grid; must divide --grid')
    parser.add_argument('--epc-population-solver', choices=('auto','direct','iterative'), default='auto')
    parser.add_argument('--tdse-collision', choices=('crta','static','dynamic'), default='crta')
    parser.add_argument('--tdse-min-linewidth', type=float, default=.02,
                        help='Floor in eV for optional SERTA TDSE only; set 0 for unmodified rates')
    args = parser.parse_args()
    if args.epc_markov != 'none' and (args.epc_qgrid < 1 or args.grid % args.epc_qgrid):
        parser.error('--epc-qgrid must be positive and divide --grid')
    if args.phonon_dynamics and args.skip_tdse:
        parser.error('--phonon-dynamics and --skip-tdse are mutually exclusive')
    if args.phonon_dynamics and args.tdse_collision != 'crta':
        parser.error('--phonon-dynamics uses only CRTA background; explicit EPC plus SERTA would count scattering twice')
    if args.phonon_dynamics and (args.supercell < 1 or args.phonon_grid < 1 or args.phonon_trajectories < 1):
        parser.error('supercell, phonon-grid, and phonon-trajectories must be positive')
    output = Path(args.output)
    with contextlib.redirect_stdout(io.StringIO()):
        model = build_model(output/'model')
    kBT = Boltzmann*args.temperature/electron_volt if args.kBT is None else args.kBT
    options = dict(model=model, kgrid=(args.grid,args.grid,1),kBT=kBT,
                   collisionSpecification=CollisionSpecification(mode="crta",constantLinewidth=args.constantLinewidth),
                   fermiList=np.array([args.fermi]),omegaList=np.array(args.omega),fullCovariantDerivative=True)
    showElapsedTime(extraInfo="model prepared")
    lgdc = AnormalousHallConductance_LG_DM(**options)
    vgdc = AnormalousHallConductance_VG_DM(**options)
    showElapsedTime(extraInfo="CRTA DC conductivity finished")
    lg = Opticalconductivity_LG_DM(**options)
    vg = Opticalconductivity_VG_DM(**options)
    showElapsedTime(extraInfo="CRTA AC conductivity finished")
    arrays = dict(omega_eV=np.array(args.omega),fermi_eV=np.array([args.fermi]),
                  sigma_dc_LG=lgdc.sigma,sigma_dc_VG=vgdc.sigma,
                  sigma_ac_LG=lg.sigma,sigma_ac_VG=vg.sigma,
                  sum_rule_residual_S_per_m_eV=vg.sum_rule_residual)
    summary = dict(units='S/m',axes='fermi, omega (ac only), drive, current',grid=args.grid,
                   constantLinewidth_eV=args.constantLinewidth,kBT_eV=kBT,
                   max_dc_LG_conductivity=np.max(abs(lgdc.sigma)),
                   dc_LG_VG_relative_error=relative_error(vgdc.sigma,lgdc.sigma),
                   max_ac_LG_conductivity=np.max(abs(lg.sigma)),
                   ac_LG_VG_relative_error=relative_error(vg.sigma,lg.sigma),
                   )
    serta_requested = (() if args.serta == 'none' else ('static','dynamic') if args.serta == 'both' else (args.serta,))
    serta_sources = {}
    for name in serta_requested:
        source = ModelEPCLinewidth(qgrid=(args.serta_qgrid,args.serta_qgrid,1),temperature=args.temperature,
                                   fermi=args.fermi,delta=args.epc_delta,adiabatic=name=='static')
        # ModelEPCLinewidth is lazy.  Evaluate it against the already solved
        # electronic grid so this timing reports the actual one-time EPC cost.
        source.populationLinewidth(lgdc)
        collision = CollisionSpecification('serta',linewidthSource=source)
        serta_options = dict(options,collisionSpecification=collision)
        showElapsedTime(extraInfo=f"{name}-SERTA line width and collision finished")
        slgdc = AnormalousHallConductance_LG_DM(**serta_options)
        svgdc = AnormalousHallConductance_VG_DM(**serta_options)
        showElapsedTime(extraInfo=f"{name}-SERTA DC conductivity finished")
        slg = Opticalconductivity_LG_DM(**serta_options)
        svg = Opticalconductivity_VG_DM(**serta_options)
        showElapsedTime(extraInfo=f"{name}-SERTA AC conductivity finished")
        gamma = slg.density_matrix_kernel.collision.population_linewidth
        serta_sources[name] = source
        arrays[f'linewidth_{name}_eV'] = gamma
        arrays[f'sigma_ac_LG_SERTA_{name}'] = slg.sigma
        arrays[f'sigma_ac_VG_SERTA_{name}'] = svg.sigma
        summary[f'serta_{name}'] = dict(linewidth_min_eV=float(gamma.min()),
                                          linewidth_max_eV=float(gamma.max()),
                                          max_dc_LG_conductivity=np.max(abs(slgdc.sigma)),
                                          dc_LG_VG_relative_error=relative_error(svgdc.sigma,slgdc.sigma),
                                          dc_change_from_CRTA=relative_error(slgdc.sigma,lgdc.sigma),
                                          max_ac_LG_conductivity=np.max(abs(slg.sigma)),
                                          ac_LG_VG_relative_error=relative_error(svg.sigma,slg.sigma),
                                          ac_change_from_CRTA=relative_error(slg.sigma,lg.sigma),
                                          )
    markov_vg = None
    if args.epc_markov != 'none':
        markov_specification=CollisionSpecification('epc',qgrid=(args.epc_qgrid,args.epc_qgrid,1),
            phononTemperature=args.temperature,bathStatistics=args.epc_markov,delta=args.epc_delta,
            constantLinewidth=args.constantLinewidth,populationSolver=args.epc_population_solver)
        markov_options=dict(options,collisionSpecification=markov_specification)
        markov_dc_lg=AnormalousHallConductance_LG_DM(**markov_options)
        markov_dc_vg=AnormalousHallConductance_VG_DM(**markov_options)
        markov_lg=Opticalconductivity_LG_DM(**markov_options)
        markov_vg=Opticalconductivity_VG_DM(**markov_options)
        view=markov_vg.density_matrix_kernel.collision.forFermi(args.fermi)
        arrays.update(sigma_dc_LG_Markov=markov_dc_lg.sigma,sigma_dc_VG_Markov=markov_dc_vg.sigma,
            sigma_ac_LG_Markov=markov_lg.sigma,sigma_ac_VG_Markov=markov_vg.sigma,
            markov_stationary_occupations=view.stationary_occupations,
            markov_reservoir_occupations=view.reservoir_occupations)
        summary['epc_markov']=dict(statistics=args.epc_markov,
            qgrid=list(markov_specification.qgrid),stationary_state=view.stationary_info,
            population_solver=args.epc_population_solver,transition_matrix_nnz=view.parent.transition_rates.nnz,
            dc_LG_VG_relative_error=relative_error(markov_dc_vg.sigma,markov_dc_lg.sigma),
            ac_LG_VG_relative_error=relative_error(markov_vg.sigma,markov_lg.sigma))
        showElapsedTime(extraInfo='Stationary Markov EPC DC and AC conductivity finished')
    if not args.skip_tdse and not args.phonon_dynamics:
        small = dict(options,kgrid=(args.tdse_grid,args.tdse_grid,1))
        tdse_collision = None
        gamma_floor = None
        if args.tdse_collision != 'crta':
            source = ModelEPCLinewidth(qgrid=(args.serta_qgrid,args.serta_qgrid,1),temperature=args.temperature,
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
    if args.phonon_dynamics:
        sc = copy.deepcopy(model)
        sc.make_supercell(np.diag([args.supercell, args.supercell, 1]))
        qgrid = (args.supercell, args.supercell, 1)
        supercell_grid = (args.phonon_grid, args.phonon_grid, 1)
        folded_grid = tuple(q*s for q, s in zip(qgrid, supercell_grid))
        matched_options = dict(options, kgrid=folded_grid)
        clean = Opticalconductivity_VG_DM(**matched_options)
        arrays['sigma_ac_VG_phonon_clean'] = clean.sigma
        for statistics in (args.phonon_statistics, 'quantum'):
            source = ModelEPCLinewidth(
                qgrid=qgrid, temperature=args.temperature, fermi=args.fermi,
                delta=args.epc_delta, bathStatistics=statistics)
            linewidth = source.populationLinewidth(clean)
            reference = Opticalconductivity_VG_DM(**dict(
                matched_options, collisionSpecification=CollisionSpecification(
                    'serta', linewidthSource=linewidth+args.constantLinewidth)))
            arrays[f'linewidth_phonon_{statistics}_eV'] = linewidth
            arrays[f'sigma_ac_VG_phonon_{statistics}_RTA'] = reference.sigma
        showElapsedTime(extraInfo='Matched phonon-mesh frequency-domain references finished')
        frequency_dm = Opticalconductivity_VG_DM_Phonon(
            **dict(options, model=sc, kgrid=supercell_grid),
            phononTemperature=args.temperature, phononStatistics=args.phonon_statistics,
            nPhononTrajectories=args.phonon_trajectories, phononSeed=args.phonon_seed,
            phononPhasePoints=args.phonon_phase_points, solverTolerance=args.phonon_solver_tolerance)
        showElapsedTime(extraInfo='Full harmonic VG density-matrix frequency solve finished')
        td = Opticalconductivity_TDSE(**dict(options, model=sc, kgrid=supercell_grid),
            phononDynamics=True, phononTemperature=args.temperature,
            phononStatistics=args.phonon_statistics, nPhononTrajectories=args.phonon_trajectories,
            phononSeed=args.phonon_seed, phononCycles=args.phonon_cycles,
            E0=args.field, n_steps_per_cycle=args.steps,
            n_cycles_measure=args.measure_cycles, transient_lifetimes=args.transient_lifetimes,
            recordCurrent=True)
        showElapsedTime(extraInfo='Moving-phonon supercell TDSE conductivity finished')
        # Apply the same finite real-drive measurement to the stationary
        # spectral current. This exposes sideband leakage separately from
        # numerical disagreement between frequency and time solvers.
        frequency_lockin = np.empty_like(td.sigma_samples)
        for sample in range(args.phonon_trajectories):
            for fi in range(len(options['fermiList'])):
                for wi in range(len(options['omegaList'])):
                    times = td.time_traces[sample, fi, wi, 0]['time']
                    frequency_lockin[sample, fi, wi] = frequency_dm.lockinConductivity(times, sample, fi, wi)
        arrays.update(sigma_ac_phonon_TDSE=td.sigma,
            sigma_ac_phonon_VG_DM=frequency_dm.sigma,
            sigma_ac_phonon_VG_DM_samples=frequency_dm.sigma_samples,
            sigma_ac_phonon_VG_DM_harmonics=frequency_dm.conductivity_harmonics,
            phonon_VG_DM_harmonic_indices=frequency_dm.phonon_harmonics,
            sigma_ac_phonon_VG_DM_lockin=frequency_lockin.mean(axis=0),
            sigma_ac_phonon_VG_DM_lockin_samples=frequency_lockin,
            sigma_ac_phonon_TDSE_samples=td.sigma_samples,
            sigma_ac_phonon_TDSE_stderr_real=td.sigma_stderr_real,
            sigma_ac_phonon_TDSE_stderr_imag=td.sigma_stderr_imag,
            phonon_energies_atom_cartesian_eV=td.phonon_energies,
            phonon_initial_positions_A=td.phonon_initial_positions,
            phonon_initial_velocities_A_per_hbar_over_eV=td.phonon_initial_velocities)
        for (sample, fi, wi, a), trace in td.time_traces.items():
            for key, value in trace.items():
                arrays[f'phonon_trace_s{sample}_f{fi}_w{wi}_a{a}_{key}'] = value
        summary['phonon_dynamics'] = dict(td.phonon_dynamics_info,
            supercell=list(qgrid), phonon_qgrid=list(qgrid),
            electronic_supercell_kgrid=list(supercell_grid),
            electronic_primitive_kgrid=list(folded_grid),
            integration_info=td.integration_info, epc_delta_eV=args.epc_delta,
            frequency_solver=frequency_dm.phonon_dynamics_info,
            frequency_solver_max_relative_residual=max(s['relative_residual'] for s in frequency_dm.solver_info),
            frequency_solver_max_iterations=max(s['iterations'] for s in frequency_dm.solver_info),
            frequency_TDSE_relative_difference=relative_error(frequency_lockin.mean(axis=0), td.sigma),
            frequency_TDSE_carrier_relative_difference=relative_error(frequency_dm.sigma, td.sigma),
            matched_bath_RTA_relative_difference=relative_error(td.sigma,
                arrays[f'sigma_ac_VG_phonon_{args.phonon_statistics}_RTA']),
            quantum_SERTA_relative_difference=relative_error(td.sigma,
                arrays['sigma_ac_VG_phonon_quantum_RTA']),
            clean_relative_change=relative_error(td.sigma, clean.sigma),
            comparison='harmonic VG DM and TDSE solve the same model; linewidth RTA comparisons remain separate approximations')
        if args.epc_markov != 'none':
            if options['kgrid'] != folded_grid or markov_specification.qgrid != qgrid:
                markov_vg=Opticalconductivity_VG_DM(**dict(matched_options,
                    collisionSpecification=CollisionSpecification('epc',qgrid=qgrid,
                        phononTemperature=args.temperature,bathStatistics=args.epc_markov,
                        delta=args.epc_delta,constantLinewidth=args.constantLinewidth,
                        populationSolver=args.epc_population_solver)))
            showElapsedTime(extraInfo='Markov full collision incorporated VG DM conductivity finished')
            arrays['sigma_ac_VG_phonon_Markov']=markov_vg.sigma
            summary['phonon_dynamics']['markov_relative_difference']=relative_error(markov_vg.sigma,td.sigma)
            summary['phonon_dynamics']['markov_statistics']=args.epc_markov
    if args.snapshots:
        sc = copy.deepcopy(model)
        sc.make_supercell(np.diag([args.supercell,args.supercell,1]))
        if args.grid % args.supercell:
            parser.error('--grid must be divisible by --supercell for a commensurate snapshot comparison')
        snapshot_grid=(args.grid//args.supercell,args.grid//args.supercell,1)
        snapshot_options = dict(options, model=sc, kgrid=snapshot_grid)
        showElapsedTime(extraInfo=f"Supercell model prepared")
        arrays['sigma_supercell_equilibrium'] = Opticalconductivity_LG_DM(**snapshot_options).sigma
        showElapsedTime(extraInfo=f"Supercell AC conductivity finished")
        samples = []
        displacements = sc.sampleModelDisplacements(args.temperature,nSamples=args.snapshots,seed=2026)
        showElapsedTime(extraInfo=f"Stochastic displacements generated")
        for u in displacements:
            snapshot_options['model'] = sc.makeDisplacedModel(u)
            samples.append(Opticalconductivity_LG_DM(**snapshot_options).sigma)
        showElapsedTime(extraInfo=f"All random displaced supercell AC conductivity finished")
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
                       snapshot_change_from_equilibrium=relative_error(samples.mean(axis=0),arrays['sigma_supercell_equilibrium']),
                       max_supercell_AC_conductivity=np.max(abs(samples.mean(axis=0))),
                       )
        for name in serta_requested:
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
