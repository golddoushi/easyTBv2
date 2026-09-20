"""Controlled snapshot / SERTA / population-BTE comparison on the chain.

Scale the thermal displacement amplitude (and EPC matrix elements) through
the ionic masses, leaving the electronic Hamiltonian and SSH derivatives fixed.
When changing phonon energy, adjust masses to preserve the equal-time quantum
displacement covariance. Antithetic pairs u,-u cancel odd sampling fluctuations.
This is a diagnostic benchmark, not a claim of exact optical aiBTE equivalence.
"""
import argparse
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.constants import Boltzmann, electron_volt
from examples.chain_four_band import build_model
from calculator import (Opticalconductivity_LG_DM, Opticalconductivity_VG_DM,
                        Opticalconductivity_TDSE, Opticalconductivity_VG_DM_Phonon,
                        Opticalconductivity_VG_DM_StaticEPC)
from calculator import CollisionSpecification, ModelEPCLinewidth
from auxiliary import uniform_qmesh
from system.phonons import PhononModeSelection


def run_tdse_alignment(model, epc, args, common, background, base, arrays, meta):
    """Compare moving-phonon dynamics on the same folded electronic grid.

    Only positive frequencies are propagated; DC slots are explicitly NaN.
    A zero-EPC run with identical time controls separates the time-step error
    of the pristine response from the EPC correction.
    """
    tick = time.perf_counter()
    indices = np.flatnonzero(np.asarray(args.omega) > 0)
    nw = len(args.omega)
    sc_epc = epc.supercell(np.diag([args.supercell, 1, 1]))
    sc = sc_epc.electronic_model
    options = dict(common, model=sc, epcModel=sc_epc, kgrid=(args.grid//args.supercell, 1, 1),
                   omegaList=np.asarray(args.omega)[indices], collisionSpecification=background)
    trajectory_statistics = 'classical' if args.phonon_statistics == 'classical' else 'wigner'
    bath = dict(phononTemperature=args.temperature, phononStatistics=trajectory_statistics,
                nPhononTrajectories=args.tdse_trajectories, phononSeed=args.seed,
                phononDamping=args.tdse_phonon_damping, phononBathCycles=args.tdse_bath_cycles,
                phononPhasePoints=args.tdse_phase_points)
    controls = dict(phononDynamics=True, n_steps_per_cycle=args.tdse_steps,
                    n_cycles_measure=args.tdse_measure_cycles,
                    transient_lifetimes=args.tdse_transient_lifetimes,
                    max_phase_step=args.tdse_max_phase_step,
                    phononCycles=args.tdse_phonon_cycles,
                    recordCurrent=args.tdse_record_current)

    def padded(values):
        values = np.asarray(values)
        result = np.full(values.shape[:-1]+(nw,), np.nan,
                         dtype=complex if np.iscomplexobj(values) else float)
        result[..., indices] = values
        return result

    clean_dm = Opticalconductivity_VG_DM(**options)
    arrays['tdse_clean_dm'] = padded(clean_dm.sigma[0, :, 0, 0])
    # Keep every frequency-domain entry in the TDSE comparison in VG. On
    # coarse meshes the pristine LG/VG difference can dwarf the EPC change.
    for name in ('static_serta', 'dynamic_serta', 'balanced_rta'):
        rates = (arrays['balanced_rta_rates_with_background_eV'] if name == 'balanced_rta'
                 else arrays[name+'_rates_eV']+args.background)
        calc = Opticalconductivity_VG_DM(**dict(common, model=model, epcModel=epc,
            collisionSpecification=CollisionSpecification('serta', linewidthSource=rates)))
        arrays[name+'_vg'] = calc.sigma[0, :, 0, 0]
    # The explicit bath has the supercell q grid, even in an intentional
    # convergence scan with a different --qgrid for the other solvers.
    for kind, static in [('static', True), ('dynamic', False)]:
        rates = ModelEPCLinewidth(epc, qgrid=(args.supercell, 1, 1), temperature=args.temperature,
            fermi=args.fermi, delta=args.delta, adiabatic=static,
            bathStatistics=trajectory_statistics).populationLinewidth(base)
        calc = Opticalconductivity_VG_DM(**dict(common, model=model, epcModel=epc,
            collisionSpecification=CollisionSpecification('serta', linewidthSource=rates+args.background)))
        arrays['tdse_bath_'+kind+'_rta'] = calc.sigma[0, :, 0, 0]
        arrays['tdse_bath_'+kind+'_rates_eV'] = rates
    selection = getattr(epc.phonon_model, 'modeSelection', None)
    if args.supercell != args.qgrid and selection is None:
        print('TDSE uses the supercell phonon grid; main SERTA/full qgrid differs.', flush=True)
    if args.tdse_phonon_damping > 0:
        print('Noise-RTA references use undamped phonon poles; only the frequency-check solver matches the damped trajectory spectrum.', flush=True)

    clean_epc = copy.deepcopy(sc_epc)

    clean_model = clean_epc.electronic_model
    clean_epc.alpha_R[:] = 0.
    clean_epc.beta_R[:] = 0.
    clean = Opticalconductivity_TDSE(**dict(options, model=clean_model, epcModel=clean_epc),
        **dict(bath, nPhononTrajectories=1), **dict(controls, recordCurrent=False))
    arrays['tdse_clean'] = padded(clean.sigma[0, :, 0, 0])
    arrays['tdse_clean_minus_dm'] = arrays['tdse_clean']-arrays['tdse_clean_dm']
    print('TDSE clean reference finished', round(time.perf_counter()-tick, 2), flush=True)
    td = Opticalconductivity_TDSE(**options, **bath, **controls)
    arrays['tdse'] = padded(td.sigma[0, :, 0, 0])
    arrays['tdse_samples'] = padded(td.sigma_samples[:, 0, :, 0, 0])
    arrays['tdse_stderr_real'] = padded(td.sigma_stderr_real[0, :, 0, 0])
    arrays['tdse_stderr_imag'] = padded(td.sigma_stderr_imag[0, :, 0, 0])
    arrays['tdse_change'] = arrays['tdse']-arrays['tdse_clean']
    arrays['tdse_minus_clean_dm'] = arrays['tdse']-arrays['tdse_clean_dm']
    arrays['tdse_initial_positions_A'] = td.phonon_initial_positions
    arrays['tdse_initial_velocities_A_per_hbar_over_eV'] = td.phonon_initial_velocities
    arrays['tdse_trajectory_cosine_A'] = td.phonon_trajectory_cosine
    arrays['tdse_trajectory_sine_A'] = td.phonon_trajectory_sine
    arrays['tdse_phonon_energies_eV'] = td.phonon_energies
    for (sample, fi, wi, drive), trace in td.time_traces.items():
        if drive == 0:
            for key, value in trace.items():
                arrays[f'tdse_trace_s{sample}_w{indices[wi]}_{key}'] = value
    meta['tdse'] = dict(td.phonon_dynamics_info, positive_frequency_indices=indices.tolist(),
        electronic_primitive_kgrid=[args.grid, 1, 1],
        electronic_supercell_kgrid=list(options['kgrid']), phonon_qgrid=[args.supercell, 1, 1],
        matches_main_qgrid=(args.supercell == args.qgrid),
        matches_main_phonon_bath=(selection is not None or args.supercell == args.qgrid),
        phonon_selection=None if selection is None else selection.as_dict(),
        integration_info=td.integration_info,
        correction_reference='zero-EPC TDSE with identical time controls',
        background_target='instantaneous f_mu,T(H(u(t),A(t))); exchanges particles/energy with a reservoir',
        snapshot_comparison='Shared equal-time phonon variance. TDSE uses fixed orbital centers; compare hopping-only snapshots/static DM.',
        bath_rta_comparison='same undamped harmonic noise statistics and qgrid, but only Born linewidths; finite delta and dressed vertices differ',
        bath_rta_includes_phonon_damping=False)
    print('Moving-phonon TDSE finished', round(time.perf_counter()-tick, 2), flush=True)

    if args.tdse_frequency_check:
        spectral = Opticalconductivity_VG_DM_Phonon(**options, **bath)
        if not (np.array_equal(spectral.phonon_trajectory_cosine, td.phonon_trajectory_cosine)
                and np.array_equal(spectral.phonon_trajectory_sine, td.phonon_trajectory_sine)):
            raise RuntimeError('Frequency and time solvers did not sample the same phonon trajectories')
        lockin = np.empty((args.tdse_trajectories, len(indices)), complex)
        for wi, info in enumerate(td.integration_info):
            times = (np.arange(info['measurement_steps'])+info['transient_steps']+1)*info['dt']
            for sample in range(args.tdse_trajectories):
                lockin[sample, wi] = spectral.lockinConductivity(times, sample, 0, wi)[0, 0]
        arrays['tdse_frequency_carrier'] = padded(spectral.sigma[0, :, 0, 0])
        arrays['tdse_frequency_lockin_samples'] = padded(lockin)
        arrays['tdse_frequency_lockin'] = padded(lockin.mean(axis=0))
        arrays['tdse_minus_frequency_lockin'] = arrays['tdse']-arrays['tdse_frequency_lockin']
        arrays['tdse_frequency_change'] = arrays['tdse_frequency_lockin']-arrays['tdse_clean_dm']
        arrays['tdse_change_minus_frequency_change'] = arrays['tdse_change']-arrays['tdse_frequency_change']
        meta['tdse']['frequency_check'] = dict(spectral.phonon_dynamics_info,
            maximum_relative_residual=max(s['relative_residual'] for s in spectral.solver_info),
            comparison='same trajectories and finite real-drive lock-in window; carrier reported separately')
        print('Matched frequency-domain bath check finished', round(time.perf_counter()-tick, 2), flush=True)
    meta['tdse']['elapsed_seconds'] = time.perf_counter()-tick


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='tmp/chain_alignment')
    p.add_argument('--grid', type=int, default=256)
    p.add_argument('--qgrid', type=int, default=16)
    p.add_argument('--supercell', type=int, default=16)
    p.add_argument('--phonon-q', type=float, nargs='+',
                   help='Selected fractional chain qx values; -q partners added automatically. Supersedes the uniform q mesh')
    p.add_argument('--phonon-branches', type=int, nargs='+',
                   help='Zero-based primitive Einstein branches: 0=A-x, 1=B-x; default both')
    p.add_argument('--phonon-weights', type=float, nargs='+',
                   help='Weight PER q point, also assigned to its -q partner; one shared value or one per input q. Default 1/qgrid; never renormalized')
    p.add_argument('--pairs', type=int, default=100)
    p.add_argument('--scale', type=float, default=.5)
    p.add_argument('--phonon-energy', type=float, default=.04)
    p.add_argument('--temperature', type=float, default=300.)
    p.add_argument('--phonon-statistics', choices=['quantum', 'wigner', 'classical'], default='quantum',
                   help='Shared bath for every EPC method: quantum (default) uses quantum collision rates and Wigner trajectories; wigner uses symmetric noise rates; classical uses equipartition throughout')
    p.add_argument('--background', type=float, default=.02,
                   help='Physical/background response linewidth in eV; included in all methods')
    p.add_argument('--delta', type=float, default=.02,
                   help='Lorentzian HWHM in eV for EPC energy-conservation integrals; numerical, not an extra lifetime. Converge jointly with qgrid; does not enter snapshots')
    p.add_argument('--fermi', type=float, default=-.7)
    p.add_argument('--omega', type=float, nargs='+', default=[0., .03, .1, .6, 1.5])
    p.add_argument('--seed', type=int, default=20260908)
    p.add_argument('--fixed-centers', action='store_true',
                   help='Also evaluate hopping-only snapshots to isolate moving-center vertices')
    p.add_argument('--curvature', action='store_true',
                   help='Evaluate the Gaussian snapshot average to second order without sampling noise')
    p.add_argument('--curvature-step', type=float, default=.005)
    p.add_argument('--static-epc-dm', action='store_true',
                   help='Complete static Gaussian EPC response through second displacement order, using DM coefficients and CRTA background')
    p.add_argument('--static-epc-kbatch', type=int, default=4,
                   help='Maximum supercell k points per static DM derivative batch')
    p.add_argument('--tdse', action='store_true', help='Also propagate electrons in moving phonon trajectories on --supercell; positive frequencies only')
    p.add_argument('--tdse-trajectories', type=int, default=4)
    p.add_argument('--tdse-steps', type=int, default=128, help='Minimum steps per optical period; max-phase-step may require more')
    p.add_argument('--tdse-max-phase-step', type=float, default=.3)
    p.add_argument('--tdse-transient-lifetimes', type=float, default=12.)
    p.add_argument('--tdse-measure-cycles', type=int, default=2)
    p.add_argument('--tdse-phonon-cycles', type=float, default=2., help='Minimum slowest-phonon periods measured, rounded to optical periods')
    p.add_argument('--tdse-phonon-damping', type=float, default=0., help='Optional oscillator amplitude damping in eV; samples a periodic Langevin spectrum')
    p.add_argument('--tdse-bath-cycles', type=float, default=8.)
    p.add_argument('--tdse-phase-points', type=int, default=33)
    p.add_argument('--tdse-frequency-check', action='store_true', help='Compare with the existing periodic-phonon VG DM solver on identical trajectories')
    p.add_argument('--tdse-record-current', action='store_true', help='Save x-drive time traces as well as ensemble conductivities')
    a = p.parse_args(argv)
    quantum_variance = a.phonon_statistics != 'classical'
    if min(a.grid, a.qgrid, a.supercell) < 1 or a.pairs < 0:
        p.error('grids/supercell must be positive and pairs nonnegative')
    if ((a.phonon_q is None and a.grid % a.qgrid) or a.grid % a.supercell
            or not np.isfinite(a.scale) or a.scale <= 0):
        p.error('qgrid and supercell must divide grid; scale must be positive')
    selection = None
    if a.phonon_q is not None or a.phonon_branches is not None or a.phonon_weights is not None:
        points = (uniform_qmesh((a.qgrid, 1, 1)) if a.phonon_q is None else
                  np.column_stack((a.phonon_q, np.zeros((len(a.phonon_q), 2)))))
        weights = np.full(len(points), 1/a.qgrid)
        if a.phonon_weights is not None:
            try:
                weights = np.broadcast_to(a.phonon_weights, (len(points),)).copy()
            except ValueError:
                p.error('--phonon-weights needs one value or one per input q point')
        try:
            selection = PhononModeSelection(points, a.phonon_branches, weights)
            selection.branch_indices(2)
            selection.mesh_shifts((a.grid, 1, 1))
            selection.check_supercell(np.diag([a.supercell, 1, 1]))
        except ValueError as error:
            p.error(str(error))
    if not np.all(np.isfinite(a.omega)):
        p.error('frequencies must be finite')
    if a.curvature and (not np.isfinite(a.curvature_step) or a.curvature_step <= 0):
        p.error('--curvature-step must be positive and finite')
    if a.static_epc_kbatch < 1:
        p.error('--static-epc-kbatch must be positive')
    if a.tdse_frequency_check and not a.tdse:
        p.error('--tdse-frequency-check requires --tdse')
    if a.tdse:
        if not np.any(np.asarray(a.omega) > 0):
            p.error('--tdse requires at least one positive --omega; it does not compute DC')
        positive = [a.background, a.tdse_max_phase_step, a.tdse_bath_cycles,
                    a.temperature, a.phonon_energy, a.delta]
        nonnegative = [a.tdse_transient_lifetimes, a.tdse_phonon_cycles, a.tdse_phonon_damping]
        if (not np.all(np.isfinite(positive+nonnegative)) or min(positive) <= 0
                or min(nonnegative) < 0 or min(a.tdse_trajectories, a.tdse_steps, a.tdse_measure_cycles) < 1):
            p.error('TDSE requires positive temperature/energies/steps/cycles/trajectories and valid time controls')
        if (a.tdse_frequency_check or a.tdse_phonon_damping > 0) and (a.tdse_phase_points < 5 or a.tdse_phase_points % 2 == 0):
            p.error('--tdse-phase-points must be odd and >= 5')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    kbt = Boltzmann*a.temperature/electron_volt
    # u^2 is proportional to coth(Eph/2kBT)/(M Eph). Keep its value at
    # reference Eph=40 meV, then scale all displacements by a.scale. This
    # mass convention is independent of statistics: do not compensate away
    # the smaller classical variance when changing the bath at fixed model.
    mass_factor = (.04/a.phonon_energy
                   * np.tanh(.04/(2*kbt))/np.tanh(a.phonon_energy/(2*kbt))
                   / a.scale**2)
    with contextlib.redirect_stdout(io.StringIO()):
        model, epc = build_model(out/'model', masses=np.array([12., 14.])*mass_factor,
                            phononEnergy=a.phonon_energy)
    epc.phonon_model.modeSelection = selection
    print('Phonon statistics:', a.phonon_statistics, flush=True)
    if selection is not None:
        print('Selected phonon bath:', json.dumps(selection.as_dict()), flush=True)
    common = dict(kgrid=(a.grid, 1, 1), kBT=kbt, fermiList=np.array([a.fermi]),
                  omegaList=np.array(a.omega), fullCovariantDerivative=True)
    bg = CollisionSpecification('crta', constantLinewidth=a.background)
    def solve(m=model, spec=bg, grid=None, vg=False):
        opts = dict(common, model=m, epcModel=epc if m is model else None, collisionSpecification=spec)
        if grid is not None:
            opts['kgrid'] = (grid, 1, 1)
        cls = Opticalconductivity_VG_DM if vg else Opticalconductivity_LG_DM
        return cls(**opts)
    base = solve()
    xx = lambda calc: calc.sigma[0, :, 0, 0]
    arrays = dict(omega_eV=np.array(a.omega), baseline=xx(base))
    meta = dict(parameters=vars(a), masses_amu=(np.array([12., 14.])*mass_factor).tolist(),
                rms_displacements_A=np.sqrt(epc.phonon_model.meanSquareDisplacements(
                    a.temperature, quantum=quantum_variance)[:, 0]).tolist())
    meta['phonon_statistics'] = dict(
        requested=a.phonon_statistics, collision=a.phonon_statistics,
        static_covariance='quantum' if quantum_variance else 'classical',
        trajectories='wigner' if quantum_variance else 'classical',
        quantum_trajectory_approximation=a.phonon_statistics == 'quantum',
        mass_convention='Fixed quantum reference variance at 40 meV; masses do not depend on statistics',
        full_collision_balance='Full and balanced RTA retain thermal pair balancing; for Wigner/classical noise this is a thermally balanced surrogate')
    meta['phonon_selection'] = None if selection is None else selection.as_dict()
    if selection is not None:
        arrays['phonon_qpoints'] = selection.qpoints
        arrays['phonon_qweights'] = selection.weights
        arrays['phonon_branches'] = selection.branch_indices(2)
    bond_rms = np.sqrt(epc.phonon_model.meanSquareDisplacements(
        a.temperature, quantum=quantum_variance)[:, 0].sum())
    meta['rms_hopping_changes_eV'] = (bond_rms*np.array([.45, .35])).tolist()
    meta['rms_hopping_relative'] = (bond_rms*np.array([.45, .35])/np.array([1., .72])).tolist()
    for name, static in [('static_serta', True), ('dynamic_serta', False)]:
        source = ModelEPCLinewidth(epc, qgrid=(a.qgrid, 1, 1), temperature=a.temperature,
            fermi=a.fermi, delta=a.delta, adiabatic=static, bathStatistics=a.phonon_statistics)
        rates = source.populationLinewidth(base)
        arrays[name+'_rates_eV'] = rates
        arrays[name] = xx(solve(spec=CollisionSpecification('serta', linewidthSource=rates+a.background)))
    full = solve(spec=CollisionSpecification('epc', qgrid=(a.qgrid, 1, 1),
        phononTemperature=a.temperature, delta=a.delta, constantLinewidth=a.background,
        bathStatistics=a.phonon_statistics, populationSolver='direct'), vg=True)
    arrays['full'] = xx(full)
    view = full.density_matrix_kernel.collision.forFermi(a.fermi)
    # Same balanced rates, but omit scattering-in: distinguishes redistribution
    # from the finite-smearing detailed-balance modification of raw SERTA rates.
    arrays['balanced_rta'] = xx(solve(spec=CollisionSpecification('serta',
        linewidthSource=view.population_linewidth)))
    arrays['balanced_rta_rates_with_background_eV'] = view.population_linewidth
    meta['full_stationary'] = view.stationary_info
    arrays['baseline_vg'] = xx(solve(vg=True))
    arrays['baseline_vg_minus_lg'] = arrays['baseline_vg']-arrays['baseline']
    print('Frequency-domain solvers finished', round(time.perf_counter()-start, 2), flush=True)
    if a.tdse:
        run_tdse_alignment(model, epc, a, common, bg, base, arrays, meta)
    if a.pairs or a.curvature or a.static_epc_dm:
        sc_epc = epc.supercell(np.diag([a.supercell, 1, 1]))
        sc = sc_epc.electronic_model
        scgrid = a.grid//a.supercell
        arrays['supercell_equilibrium'] = xx(solve(sc, grid=scgrid))
        if a.static_epc_dm:
            opts = dict(common, model=sc, epcModel=sc_epc, kgrid=(scgrid, 1, 1), collisionSpecification=bg)
            for fixed in ([False, True] if a.fixed_centers else [False]):
                result = Opticalconductivity_VG_DM_StaticEPC(**opts,
                    phononTemperature=a.temperature, phononStatistics=a.phonon_statistics, movingCenters=not fixed,
                    kBatchSize=a.static_epc_kbatch)
                key = 'static_epc_dm_fixed' if fixed else 'static_epc_dm'
                arrays[key] = xx(result)
                arrays[key+'_bare'] = result.sigma_bare[0, :, 0, 0]
                arrays[key+'_change'] = result.sigma_correction[0, :, 0, 0]
                arrays[key+'_change_by_mode'] = result.sigma_correction_by_mode[:, 0, :, 0, 0]
                arrays[key+'_sum_rule_residual'] = result.correction_sum_rule_residual
                for name, value in result.sigma_correction_channels.items():
                    arrays[key+'_'+name] = value[0, :, 0, 0]
                meta[key] = result.static_epc_info
            print('Complete static EPC DM finished', round(time.perf_counter()-start, 2), flush=True)
        if a.curvature and selection is not None:
            # Selected q modes correlate different cells. Contract the full
            # covariance through real normal-coordinate directions, retaining
            # cross-atom/cell Hessian contributions instead of local variances.
            correction = np.zeros(len(a.omega), complex)
            for factor in sc_epc.phonon_model.displacementFactors(a.temperature, quantum=quantum_variance):
                amplitude = np.max(np.abs(factor))
                if amplitude == 0:
                    continue
                pair = []
                for sign in (1, -1):
                    u = sign*a.curvature_step*factor/amplitude
                    pair.append(xx(solve(sc_epc.makeDisplacedModel(u), grid=scgrid)))
                correction += amplitude**2*(sum(pair)-2*arrays['supercell_equilibrium'])/(2*a.curvature_step**2)
            arrays['snapshot_second_order'] = arrays['supercell_equilibrium']+correction
            print('Selected-mode snapshot curvature finished', round(time.perf_counter()-start, 2), flush=True)
        elif a.curvature:
            # Independent Einstein atoms: <sigma> = sigma0 + sum_i <u_i^2>
            # d_i^2 sigma/2 + O(u^4). Translation symmetry makes every cell's
            # derivative identical, leaving only two inequivalent atoms.
            variance = sc_epc.phonon_model.meanSquareDisplacements(a.temperature, quantum=quantum_variance)[:, 0]
            correction = np.zeros(len(a.omega), complex)
            for atom in range(2):
                pair = []
                for sign in (1, -1):
                    u = np.zeros((len(sc_epc.phonon_model.masses), 3))
                    u[atom, 0] = sign*a.curvature_step
                    pair.append(xx(solve(sc_epc.makeDisplacedModel(u), grid=scgrid)))
                correction += a.supercell*variance[atom]*(sum(pair)-2*arrays['supercell_equilibrium'])/(2*a.curvature_step**2)
            arrays['snapshot_second_order'] = arrays['supercell_equilibrium']+correction
            print('Snapshot curvature finished', round(time.perf_counter()-start, 2), flush=True)
    if a.pairs:
        displacements = sc_epc.phonon_model.sampleDisplacements(a.temperature, nSamples=a.pairs, seed=a.seed,
                                                   quantum=quantum_variance)
        arrays['snapshot_displacements_A'] = displacements
        samples, fixed = [], []
        for ip, u in enumerate(displacements):
            pair, fixed_pair = [], []
            for sign in (1, -1):
                displaced = sc_epc.makeDisplacedModel(sign*u)
                pair.append(xx(solve(displaced, grid=scgrid)))
                if a.fixed_centers:
                    displaced.rList = sc.rList.copy()
                    displaced.AA_R_W = sc.AA_R_W.copy()
                    displaced.posinfo = copy.deepcopy(sc.posinfo)
                    displaced._refreshHamiltonianDerivedMatrices()
                    fixed_pair.append(xx(solve(displaced, grid=scgrid)))
            samples.append(np.mean(pair, axis=0))
            if fixed_pair:
                fixed.append(np.mean(fixed_pair, axis=0))
            if (ip+1) % 20 == 0:
                print('Pairs', ip+1, 'elapsed', round(time.perf_counter()-start, 1), flush=True)
        arrays['snapshot_pairs'] = np.array(samples)
        arrays['snapshot'] = np.mean(samples, axis=0)
        if fixed:
            arrays['fixed_center_pairs'] = np.array(fixed)
            arrays['fixed_center_snapshot'] = np.mean(fixed, axis=0)
        for name in ('snapshot', 'fixed_center'):
            key = name+'_pairs'
            if key in arrays and a.pairs > 1:
                values = arrays[key]
                arrays[name+'_stderr_real'] = values.real.std(axis=0, ddof=1)/np.sqrt(a.pairs)
                arrays[name+'_stderr_imag'] = values.imag.std(axis=0, ddof=1)/np.sqrt(a.pairs)
    change_references = {}
    for name in ('static_serta', 'dynamic_serta', 'balanced_rta', 'full', 'snapshot',
                 'fixed_center_snapshot', 'snapshot_second_order', 'static_serta_vg',
                 'dynamic_serta_vg', 'balanced_rta_vg', 'tdse_bath_static_rta', 'tdse_bath_dynamic_rta'):
        if name in arrays:
            reference = ('supercell_equilibrium' if 'snapshot' in name else
                         'baseline_vg' if name == 'full' or name.endswith('_vg') or name.startswith('tdse_bath_')
                         else 'baseline')
            arrays[name+'_change'] = arrays[name]-arrays[reference]
            change_references[name] = reference
    if a.tdse:
        change_references['tdse'] = 'tdse_clean'
    for name in ('static_epc_dm', 'static_epc_dm_fixed'):
        if name in arrays:
            change_references[name] = name+'_bare'
    meta['change_references'] = change_references
    meta['elapsed_seconds'] = time.perf_counter()-start
    np.savez(out/'results.npz', **arrays)
    (out/'metadata.json').write_text(json.dumps(meta, indent=2)+'\n', encoding='utf-8')
    print(f"sigma0: {arrays['baseline'][0].real:12.5E}", flush=True)
    for name in ('static_serta', 'dynamic_serta', 'balanced_rta', 'full', 'snapshot', 'fixed_center_snapshot', 'snapshot_second_order', 'static_epc_dm', 'static_epc_dm_fixed'):
        if name in arrays:
            change       = arrays[name+'_change']
            change_ratio = change/np.abs(arrays[change_references[name]])*100
            print(f"{name:13s} change: {change[0].real:12.5E} {change[0].imag:12.5E}; change/|sigma0| percent: {change_ratio[0].real:6.2f} %", flush=True)
    if a.tdse:
        for wi in meta['tdse']['positive_frequency_indices']:
            print(f'\nTDSE comparison at {a.omega[wi]:g} eV (S/m; changes are complex):', flush=True)
            print('Pristine VG minus LG:', arrays['baseline_vg_minus_lg'][wi], flush=True)
            for name in ('static_serta_vg', 'dynamic_serta_vg', 'balanced_rta_vg', 'full',
                         'tdse_bath_static_rta', 'tdse_bath_dynamic_rta'):
                change = arrays[name+'_change'][wi]
                print(f'{name:25s} change={change.real: .7e} {change.imag:+.7e}j', flush=True)
            for name in ('tdse_change', 'tdse_clean_minus_dm', 'tdse_minus_frequency_lockin',
                         'tdse_change_minus_frequency_change'):
                if name in arrays:
                    value = arrays[name][wi]
                    print(f'{name:38s} {value.real: .7e} {value.imag:+.7e}j', flush=True)
            print(f'TDSE one-SE real/imag: {arrays["tdse_stderr_real"][wi]:.5e}, '
                  f'{arrays["tdse_stderr_imag"][wi]:.5e}', flush=True)
    print('Saved', out.resolve(), flush=True)
    return arrays, meta


if __name__ == '__main__':
    main()
