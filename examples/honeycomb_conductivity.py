"""Compare CRTA, SERTA, Markov, Fourier-bath, and TDSE conductivities.

Grid names describe what they sample:
  electronic-k-grid: primitive-cell k mesh used by the main DM calculations.
  clamped-tdse-k-grid: smaller primitive-cell k mesh used only by ion-clamped TDSE.
  phonon-supercell: real-space repeats used by explicit moving-ion calculations.
  supercell-k-grid: electronic k mesh of that enlarged moving-ion supercell.
  serta-q-grid / markov-q-grid: phonon q meshes used by those collision models.

For explicit phonons, a supercell of S and supercell k grid K represent a
primitive electronic mesh S*K and a primitive phonon mesh S along every
periodic direction. Deprecated short grid names remain accepted but hidden.
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
    reference_line = np.abs(reference.flatten())
    difference_line = np.abs(value-reference).flatten()
    active = np.flatnonzero(reference_line > max(np.max(reference_line)*1e-12, 1e-30))
    imax = active[np.argsort(reference_line[active])[-4:]] if len(active) else np.array([], dtype=int)
    return tuple(difference_line[imax] / reference_line[imax])

def global_relative_error(value, reference):
    denominator = np.linalg.norm(reference)
    return float(np.linalg.norm(value-reference)/denominator) if denominator else float('inf')


def periodic_grid(points, dimensions):
    return tuple([int(points)]*dimensions+[1]*(3-dimensions))

def periodic_supercell(cells, dimensions):
    return np.diag(periodic_grid(cells, dimensions))

def _hidden_alias(parser, option, destination, value_type=int):
    parser.add_argument(option, dest=destination, type=value_type,
                        default=argparse.SUPPRESS, help=argparse.SUPPRESS)

def run_conductivity_benchmark(model_builder=build_model, periodic_dimensions=2,
                               default_output='tmp/honeycomb_conductivity',
                               default_electronic_k_grid=32, description=__doc__, argv=None):
    """Run the common EPC conductivity benchmark for a 1D or 2D model builder."""
    parser = argparse.ArgumentParser(description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', default=default_output)
    parser.add_argument('--electronic-k-grid', type=int, default=default_electronic_k_grid,
                        help='Primitive-cell k points per periodic reciprocal direction for main DM calculations')
    parser.add_argument('--clamped-tdse-k-grid', type=int, default=4,
                        help='Primitive-cell k points per periodic direction for the ion-clamped TDSE check only')
    parser.add_argument('--omega', type=float, nargs='+', default=[.6, 1.3],
                        help='Photon energies in eV')
    parser.add_argument('--fermi', type=float, default=-.7,
                        help='Electronic chemical potential in eV')
    parser.add_argument('--background-linewidth', type=float, default=.0658,
                        help='Electronic CRTA background linewidth in eV')
    parser.add_argument('--electronic-kbt', type=float, default=None,
                        help='Electronic kBT in eV; default is Boltzmann*temperature')
    parser.add_argument('--optical-field', type=float, default=1e-4,
                        help='Peak optical electric field in V/Angstrom (TDSE trace normalization)')
    parser.add_argument('--tdse-steps-per-cycle', type=int, default=128,
                        help='Minimum electronic TDSE steps per optical period')
    parser.add_argument('--skip-tdse', action='store_true')
    parser.add_argument('--snapshots', type=int, default=0,
                        help='Optional frozen-displacement LG ac average; this is not an inelastic EPC collision calculation')
    parser.add_argument('--phonon-supercell', type=int, default=4,
                        help='Primitive-cell repeats per periodic direction for snapshots and moving-phonon solvers')
    parser.add_argument('--phonon-dynamics', action='store_true',
                        help='Propagate electrons in a moving harmonic SSH supercell bath')
    parser.add_argument('--supercell-k-grid', type=int, default=1,
                        help='Electronic k points per periodic direction of the enlarged moving-phonon supercell')
    parser.add_argument('--phonon-trajectories', type=int, default=4,
                        help='Independent stochastic ionic paths used by both explicit-phonon solvers')
    parser.add_argument('--phonon-statistics', choices=('classical', 'wigner'), default='classical')
    parser.add_argument('--phonon-cycles', type=float, default=1.,
                        help='Minimum measurement duration in periods of the slowest phonon')
    parser.add_argument('--phonon-seed', type=int, default=2026)
    parser.add_argument('--phonon-phase-points', type=int, default=33,
                        help='Odd Fourier phase-grid size for the matched frequency-domain Liouville solver')
    parser.add_argument('--phonon-damping', type=float, default=0.,
                        help='Phonon amplitude damping in eV; zero retains ideal harmonic motion')
    parser.add_argument('--phonon-bath-cycles', type=float, default=8.,
                        help='Period of the stationary damped trajectory in slowest-phonon cycles')
    parser.add_argument('--phonon-solver-tolerance', type=float, default=1e-9)
    parser.add_argument('--measure-cycles', type=int, default=2)
    parser.add_argument('--transient-lifetimes', type=float, default=12.)
    parser.add_argument('--temperature', type=float, default=300.,
                        help='Electronic and phonon temperature in kelvin unless electronic-kbt overrides it')
    parser.add_argument('--serta', choices=('none','static','dynamic','both'), default='both',
                        help='Compute toy-EPC SERTA optical and dc tensors')
    parser.add_argument('--serta-q-grid', type=int, default=6,
                        help='Phonon q points per periodic direction used to construct SERTA linewidths')
    parser.add_argument('--epc-delta', type=float, default=.03,
                        help='Lorentzian delta broadening used to construct EPC linewidths, eV')
    parser.add_argument('--epc-markov', choices=('none','quantum','classical','wigner'), default='none',
                        help='Additional stationary Pauli collision calculation beyond SERTA')
    parser.add_argument('--markov-q-grid', type=int, default=4,
                        help='Phonon q points per periodic direction for Markov rates; must divide electronic-k-grid')
    parser.add_argument('--epc-population-solver', choices=('auto','direct','iterative'), default='auto')
    parser.add_argument('--epc-raw-rates', action='store_true',
                        help='Disable pairwise detailed balance for a diagnostic raw-rate Markov calculation')
    parser.add_argument('--tdse-collision', choices=('crta','static','dynamic'), default='crta')
    parser.add_argument('--tdse-min-linewidth', type=float, default=.02,
                        help='Floor in eV for optional SERTA TDSE only; set 0 for unmodified rates')
    _hidden_alias(parser, '--grid', 'electronic_k_grid')
    _hidden_alias(parser, '--tdse-grid', 'clamped_tdse_k_grid')
    _hidden_alias(parser, '--supercell', 'phonon_supercell')
    _hidden_alias(parser, '--phonon-grid', 'supercell_k_grid')
    _hidden_alias(parser, '--serta-qgrid', 'serta_q_grid')
    _hidden_alias(parser, '--epc-qgrid', 'markov_q_grid')
    _hidden_alias(parser, '--constantLinewidth', 'background_linewidth', float)
    _hidden_alias(parser, '--kBT', 'electronic_kbt', float)
    _hidden_alias(parser, '--field', 'optical_field', float)
    _hidden_alias(parser, '--steps', 'tdse_steps_per_cycle')
    args = parser.parse_args(argv)
    if periodic_dimensions not in (1, 2, 3):
        raise ValueError('periodic_dimensions must be 1, 2, or 3')
    if args.epc_markov != 'none' and (args.markov_q_grid < 1
                                      or args.electronic_k_grid % args.markov_q_grid):
        parser.error('--markov-q-grid must be positive and divide --electronic-k-grid')
    if min(args.electronic_k_grid, args.clamped_tdse_k_grid, args.serta_q_grid) < 1:
        parser.error('electronic-k-grid, clamped-tdse-k-grid, and serta-q-grid must be positive')
    if args.phonon_dynamics and args.skip_tdse:
        parser.error('--phonon-dynamics and --skip-tdse are mutually exclusive')
    if args.phonon_dynamics and args.tdse_collision != 'crta':
        parser.error('--phonon-dynamics uses only CRTA background; explicit EPC plus SERTA would count scattering twice')
    if args.phonon_dynamics and (args.phonon_supercell < 1 or args.supercell_k_grid < 1
                                 or args.phonon_trajectories < 1):
        parser.error('phonon-supercell, supercell-k-grid, and phonon-trajectories must be positive')
    output = Path(args.output)
    with contextlib.redirect_stdout(io.StringIO()):
        model = model_builder(output/'model')
    kBT = (Boltzmann*args.temperature/electron_volt
           if args.electronic_kbt is None else args.electronic_kbt)
    electronic_grid = periodic_grid(args.electronic_k_grid, periodic_dimensions)
    options = dict(model=model, kgrid=electronic_grid, kBT=kBT,
                   collisionSpecification=CollisionSpecification(
                       mode="crta", constantLinewidth=args.background_linewidth),
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
    summary = dict(units='S/m', axes='fermi, omega (ac only), drive, current',
                   periodic_dimensions=periodic_dimensions, electronic_k_grid=list(electronic_grid),
                   background_linewidth_eV=args.background_linewidth, kBT_eV=kBT,
                   max_dc_LG_conductivity=np.max(abs(lgdc.sigma)),
                   dc_LG_VG_relative_error=relative_error(vgdc.sigma,lgdc.sigma),
                   max_ac_LG_conductivity=np.max(abs(lg.sigma)),
                   ac_LG_VG_relative_error=relative_error(vg.sigma,lg.sigma),
                   )
    serta_requested = (() if args.serta == 'none' else ('static','dynamic') if args.serta == 'both' else (args.serta,))
    serta_sources = {}
    for name in serta_requested:
        source = ModelEPCLinewidth(qgrid=periodic_grid(args.serta_q_grid, periodic_dimensions), temperature=args.temperature,
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
        markov_specification=CollisionSpecification('epc',
            qgrid=periodic_grid(args.markov_q_grid, periodic_dimensions),
            phononTemperature=args.temperature,bathStatistics=args.epc_markov,delta=args.epc_delta,
            constantLinewidth=args.background_linewidth, populationSolver=args.epc_population_solver,
            enforceDetailedBalance=not args.epc_raw_rates)
        markov_options=dict(options,collisionSpecification=markov_specification)
        markov_dc_lg=AnormalousHallConductance_LG_DM(**markov_options)
        markov_dc_vg=AnormalousHallConductance_VG_DM(**markov_options)
        markov_lg=Opticalconductivity_LG_DM(**markov_options)
        markov_vg=Opticalconductivity_VG_DM(**markov_options)
        view=markov_vg.density_matrix_kernel.collision.forFermi(args.fermi)
        arrays.update(sigma_dc_LG_Markov=markov_dc_lg.sigma,sigma_dc_VG_Markov=markov_dc_vg.sigma,
            sigma_ac_LG_Markov=markov_lg.sigma,sigma_ac_VG_Markov=markov_vg.sigma,
            markov_stationary_occupations=view.stationary_occupations,
            markov_equilibrium_occupations=view.equilibrium_occupations)
        summary['epc_markov']=dict(statistics=args.epc_markov,
            qgrid=list(markov_specification.qgrid),stationary_state=view.stationary_info,
            fixed_particle_number=True,detailed_balance=markov_specification.enforceDetailedBalance,
            population_solver=args.epc_population_solver,transition_matrix_nnz=view.parent.transition_rates.nnz,
            dc_LG_VG_relative_error=relative_error(markov_dc_vg.sigma,markov_dc_lg.sigma),
            ac_LG_VG_relative_error=relative_error(markov_vg.sigma,markov_lg.sigma))
        showElapsedTime(extraInfo='Stationary Markov EPC DC and AC conductivity finished')
    if not args.skip_tdse and not args.phonon_dynamics:
        clamped_tdse_grid = periodic_grid(args.clamped_tdse_k_grid, periodic_dimensions)
        small = dict(options, kgrid=clamped_tdse_grid)
        tdse_collision = None
        gamma_floor = None
        if args.tdse_collision != 'crta':
            source = ModelEPCLinewidth(qgrid=periodic_grid(args.serta_q_grid, periodic_dimensions),
                                       temperature=args.temperature,
                                       fermi=args.fermi,delta=args.epc_delta,
                                       adiabatic=args.tdse_collision=='static')
            owner = Opticalconductivity_LG_DM(**small)
            gamma_floor = np.maximum(source.populationLinewidth(owner),args.tdse_min_linewidth)
            tdse_collision = CollisionSpecification('serta',linewidthSource=gamma_floor)
            small['collisionSpecification'] = tdse_collision
        reference = Opticalconductivity_VG_DM(**small)
        td = Opticalconductivity_TDSE(E0=args.optical_field,
            n_steps_per_cycle=args.tdse_steps_per_cycle, recordCurrent=True, **small)
        #print(f"{td.sigma=}")
        arrays.update(sigma_ac_VG_tdse_mesh=reference.sigma,sigma_ac_TDSE=td.sigma)
        for (fi,wi,a), trace in td.time_traces.items():
            for key, value in trace.items():
                arrays[f'trace_f{fi}_w{wi}_a{a}_{key}'] = value
        summary.update(clamped_tdse_k_grid=list(clamped_tdse_grid),
                       ac_tdse_vg_relative_error=relative_error(td.sigma,reference.sigma),
                       tdse_collision=args.tdse_collision,
                       tdse_min_linewidth_eV=(None if gamma_floor is None else float(gamma_floor.min())),
                       field_peak_V_per_A=args.optical_field, integration_info=td.integration_info,
                       max_ac_VG_small_conductivity=np.max(abs(reference.sigma)),
                       max_ac_TD_small_conductivity=np.max(abs(td.sigma)),
                   )
    if args.phonon_dynamics:
        sc = copy.deepcopy(model)
        sc.make_supercell(periodic_supercell(args.phonon_supercell, periodic_dimensions))
        qgrid = periodic_grid(args.phonon_supercell, periodic_dimensions)
        supercell_grid = periodic_grid(args.supercell_k_grid, periodic_dimensions)
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
                    'serta', linewidthSource=linewidth+args.background_linewidth)))
            arrays[f'linewidth_phonon_{statistics}_eV'] = linewidth
            arrays[f'sigma_ac_VG_phonon_{statistics}_RTA'] = reference.sigma
        showElapsedTime(extraInfo='Matched phonon-mesh frequency-domain references finished')
        frequency_dm = Opticalconductivity_VG_DM_Phonon(
            **dict(options, model=sc, kgrid=supercell_grid),
            phononTemperature=args.temperature, phononStatistics=args.phonon_statistics,
            nPhononTrajectories=args.phonon_trajectories, phononSeed=args.phonon_seed,
            phononPhasePoints=args.phonon_phase_points, phononDamping=args.phonon_damping,
            phononBathCycles=args.phonon_bath_cycles, solverTolerance=args.phonon_solver_tolerance)
        showElapsedTime(extraInfo='Periodic-phonon VG density-matrix frequency solve finished')
        td = Opticalconductivity_TDSE(**dict(options, model=sc, kgrid=supercell_grid),
            phononDynamics=True, phononTemperature=args.temperature,
            phononStatistics=args.phonon_statistics, nPhononTrajectories=args.phonon_trajectories,
            phononSeed=args.phonon_seed, phononCycles=args.phonon_cycles,
            phononDamping=args.phonon_damping, phononBathCycles=args.phonon_bath_cycles,
            phononPhasePoints=args.phonon_phase_points,
            E0=args.optical_field, n_steps_per_cycle=args.tdse_steps_per_cycle,
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
            phonon_initial_velocities_A_per_hbar_over_eV=td.phonon_initial_velocities,
            phonon_trajectory_cosine_A=td.phonon_trajectory_cosine,
            phonon_trajectory_sine_A=td.phonon_trajectory_sine)
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
            frequency_TDSE_global_relative_difference=global_relative_error(
                frequency_lockin.mean(axis=0), td.sigma),
            frequency_TDSE_carrier_relative_difference=relative_error(frequency_dm.sigma, td.sigma),
            matched_bath_RTA_relative_difference=relative_error(td.sigma,
                arrays[f'sigma_ac_VG_phonon_{args.phonon_statistics}_RTA']),
            quantum_SERTA_relative_difference=relative_error(td.sigma,
                arrays['sigma_ac_VG_phonon_quantum_RTA']),
            clean_relative_change=relative_error(td.sigma, clean.sigma),
            comparison='periodic-phonon VG DM and TDSE solve the same realization; linewidth RTA comparisons remain separate approximations')
        if args.epc_markov != 'none':
            if options['kgrid'] != folded_grid or markov_specification.qgrid != qgrid:
                markov_vg=Opticalconductivity_VG_DM(**dict(matched_options,
                    collisionSpecification=CollisionSpecification('epc',qgrid=qgrid,
                        phononTemperature=args.temperature,bathStatistics=args.epc_markov,
                        delta=args.epc_delta, constantLinewidth=args.background_linewidth,
                        populationSolver=args.epc_population_solver,
                        enforceDetailedBalance=not args.epc_raw_rates)))
            showElapsedTime(extraInfo='Markov full collision incorporated VG DM conductivity finished')
            arrays['sigma_ac_VG_phonon_Markov']=markov_vg.sigma
            summary['phonon_dynamics']['markov_relative_difference']=relative_error(markov_vg.sigma,td.sigma)
            summary['phonon_dynamics']['markov_global_relative_difference'] = global_relative_error(
                markov_vg.sigma, td.sigma)
            summary['phonon_dynamics']['markov_statistics']=args.epc_markov
            summary['phonon_dynamics']['markov_energy_delta_eV']=args.epc_delta
    if args.snapshots:
        sc = copy.deepcopy(model)
        sc.make_supercell(periodic_supercell(args.phonon_supercell, periodic_dimensions))
        if args.electronic_k_grid % args.phonon_supercell:
            parser.error('--electronic-k-grid must be divisible by --phonon-supercell for snapshots')
        snapshot_grid = periodic_grid(
            args.electronic_k_grid//args.phonon_supercell, periodic_dimensions)
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
        summary.update(snapshots=args.snapshots,temperature_K=args.temperature,
                       snapshot_k_grid=list(snapshot_grid),
                       snapshot_primitive_equivalent_k_grid=list(electronic_grid),
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
                'serta', linewidthSource=arrays[f'linewidth_{name}_eV']+args.background_linewidth)
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

def main(argv=None):
    return run_conductivity_benchmark(argv=argv)

if __name__ == '__main__':
    main()
