"""Compare rectified chain BPVE with static/dynamic SERTA, balanced RTA,
full Pauli collision, and antithetic frozen-displacement snapshots.

Outputs are xxx BPVE in the calculators' raw units, normalized per primitive
cell (not SI conductivity). Both frozen and covariant linewidth derivatives
are evaluated. This tests approximations; agreement is not imposed.
"""
from phonon import PhononBath
import argparse
import contextlib
import copy
import csv
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
from calculator import (BulkPhotovoltaicEffect_LG_DM, BulkPhotovoltaicEffect_VG_DM,
                        CollisionSpecification, ModelEPCLinewidth)
from examples.chain_four_band import build_model


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='tmp/chain_BPVE_alignment')
    p.add_argument('--grid', type=int, default=512)
    p.add_argument('--qgrid', type=int, default=16)
    p.add_argument('--supercell', type=int, default=16)
    p.add_argument('--pairs', type=int, default=20, help='Number of independent u,-u pairs (2*pairs snapshots)')
    p.add_argument('--scale', type=float, default=.25, help='Displacement/EPC amplitude relative to the reference covariance')
    p.add_argument('--phonon-energy', type=float, default=.02)
    p.add_argument('--temperature', type=float, default=300.)
    p.add_argument('--background', type=float, default=.00658, help='Background relaxation in eV, present in every method')
    p.add_argument('--delta', type=float, default=.05, help='Numerical Lorentzian HWHM in EPC rates (eV); converge with qgrid')
    p.add_argument('--fermi', type=float, default=-.7)
    p.add_argument('--omega', type=float, nargs='+', default=[.8], help='Positive pump photon energies in eV; output frequency is zero')
    p.add_argument('--seed', type=int, default=20260908)
    p.add_argument('--skip-vg', action='store_true', help='Skip primitive-cell VG cross-checks')
    p.add_argument('--fixed-centers', action='store_true', help='Also sample hopping-only disorder with fixed optical position vertices')
    p.add_argument('--curvature', action='store_true', help='Also compute the deterministic O(u^2) snapshot correction')
    p.add_argument('--curvature-step', type=float, default=.002)
    a = p.parse_args(argv)
    if min(a.grid, a.qgrid, a.supercell) < 1 or a.pairs < 0:
        p.error('grids/supercell must be positive and pairs nonnegative')
    if a.grid % a.qgrid or a.grid % a.supercell:
        p.error('qgrid and supercell must divide grid')
    positive = [a.scale, a.phonon_energy, a.temperature, a.background,
                a.delta, a.curvature_step, *a.omega]
    if not np.all(np.isfinite(positive)) or min(positive) <= 0 or not np.isfinite(a.fermi):
        p.error('scale, phonon energy, temperature, background, delta, step and pump frequencies must be positive and finite')
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    kbt = Boltzmann*a.temperature/electron_volt
    # Preserve the equal-time covariance at reference Eph=40 meV, as in the
    # optical alignment example. Phonon-energy scans then isolate dynamics.
    mass_factor = (.04/a.phonon_energy*np.tanh(.04/(2*kbt))
                   / np.tanh(a.phonon_energy/(2*kbt))/a.scale**2)
    masses = np.array([12., 14.])*mass_factor
    with contextlib.redirect_stdout(io.StringIO()):
        model, epc = build_model(out/'model', masses=masses, phononEnergy=a.phonon_energy)
    common = dict(kgrid=(a.grid, 1, 1), kBT=kbt, fermiList=np.array([a.fermi]),
                  omegaList=np.array(a.omega), fullCovariantDerivative=True,
                  finiteTemperatureOccupations=True)
    bg = CollisionSpecification('crta', constantLinewidth=a.background)
    arrays = {'omega_eV': np.array(a.omega)}
    meta = dict(parameters=vars(a), masses_amu=masses.tolist(), timings_seconds={},
                observable='xxx rectified BPVE (0; -omega, +omega)',
                units='raw DM calculator units per primitive cell; no SI prefactor',
                snapshot_qgrid_matches_rates=(a.supercell == a.qgrid),
                covariance='quantum Einstein covariance held fixed when phonon energy changes',
                balanced_rta='Frozen equilibrium diagonal population damping of full; no scattering-in or collision Hessian. Coherences use the RTA half-sum.',
                limitations=['Linewidth-only paths omit real self-energy/DW and optical vertex renormalization.',
                             'Full is a fixed-number nonlinear secular Pauli model; RTA restores fixed-mu equilibrium.',
                             'Snapshots are static, with the same background; they are not a quantum dynamical bath.'])
    variance = epc.phonon_model.meanSquareDisplacements(a.temperature)[:, 0]
    meta['rms_displacements_A'] = np.sqrt(variance).tolist()
    meta['rms_hopping_relative'] = (np.sqrt(variance.sum())*np.array([.45, .35])/np.array([1., .72])).tolist()

    def solve(m=model, spec=bg, grid=None, vg=False):
        opts = dict(common, model=m, epcModel=epc if m is model else None, collisionSpecification=spec)
        if grid is not None:
            opts['kgrid'] = (grid, 1, 1)
        cls = BulkPhotovoltaicEffect_VG_DM if vg else BulkPhotovoltaicEffect_LG_DM
        return cls(**opts)

    def xxx(calc):
        # Unlike Opticalconductivity, these BPVE DM classes currently return
        # an unnormalized band trace. A supercell contains N primitive cells.
        cells = abs(calc.model.Vcell/model.Vcell)
        return calc.projected_sigma[0, 0, 0, 0, :, 0, 0, 0].copy()/cells

    def record(name, spec):
        tick = time.perf_counter()
        calc = solve(spec=spec)
        arrays[name] = xxx(calc)
        if not a.skip_vg:
            arrays[name+'_vg'] = xxx(solve(spec=spec, vg=True))
            arrays[name+'_vg_minus_lg'] = arrays[name+'_vg']-arrays[name]
        meta['timings_seconds'][name] = time.perf_counter()-tick
        print(f'{name} finished in {meta["timings_seconds"][name]:.3f}s', flush=True)
        return calc

    base = record('baseline', bg)
    # Verify the new derivative machinery independently in the CRTA limit.
    record('baseline_covariant', CollisionSpecification('crta', constantLinewidth=a.background,
           secondOrderLinewidthDerivative='covariant'))
    for name, static in [('static_serta', True), ('dynamic_serta', False)]:
        tick = time.perf_counter()
        rates = ModelEPCLinewidth(epc, fermi=a.fermi, delta=a.delta, adiabatic=static, phononBath=PhononBath(qgrid=(a.qgrid, 1, 1), temperature=a.temperature)).populationLinewidth(
            base,
        )
        meta['timings_seconds'][name+'_rates'] = time.perf_counter()-tick
        arrays[name+'_rates_eV'] = rates
        for derivative in ('frozen', 'covariant'):
            spec = CollisionSpecification('serta', linewidthSource=rates+a.background,
                                         secondOrderLinewidthDerivative=derivative)
            record(name+'_'+derivative, spec)
    full = record(
        'full',
        CollisionSpecification('epc', delta=a.delta, constantLinewidth=a.background, phononBath=PhononBath(qgrid=(a.qgrid, 1, 1), temperature=a.temperature, statistics='quantum')),
    )
    view = full.density_matrix_kernel.collision.forFermi(a.fermi)
    arrays['balanced_rta_rates_with_background_eV'] = view.population_linewidth
    meta['full_stationary'] = view.stationary_info
    for derivative in ('frozen', 'covariant'):
        record('balanced_rta_'+derivative, CollisionSpecification('serta',
            linewidthSource=view.population_linewidth, secondOrderLinewidthDerivative=derivative))

    if a.pairs or a.curvature:
        if a.supercell != a.qgrid:
            print('Note: snapshot phonon mesh and rate qgrid differ; see metadata.', flush=True)
        sc_epc = epc.supercell(np.diag([a.supercell, 1, 1]))
        sc = sc_epc.electronic_model
        scgrid = a.grid//a.supercell
        arrays['supercell_equilibrium'] = xxx(solve(sc, grid=scgrid))
        arrays['supercell_folding_error'] = arrays['supercell_equilibrium']-arrays['baseline']
        print('Equilibrium supercell folding error:', np.max(np.abs(arrays['supercell_folding_error'])), flush=True)
        if a.curvature:
            tick = time.perf_counter()
            correction = np.zeros(len(a.omega), complex)
            for atom in range(2):
                pair = []
                for sign in (1, -1):
                    u = np.zeros((len(sc_epc.phonon_model.masses), 3))
                    u[atom, 0] = sign*a.curvature_step
                    pair.append(xxx(solve(sc_epc.makeDisplacedModel(u), grid=scgrid)))
                correction += a.supercell*variance[atom]*(sum(pair)-2*arrays['supercell_equilibrium'])/(2*a.curvature_step**2)
            arrays['snapshot_second_order'] = arrays['supercell_equilibrium']+correction
            meta['timings_seconds']['curvature'] = time.perf_counter()-tick
            print('Snapshot curvature finished', flush=True)
        if a.pairs:
            tick = time.perf_counter()
            samples, fixed = [], []
            displacements = sc_epc.phonon_model.sampleDisplacements(a.temperature, nSamples=a.pairs, seed=a.seed)
            for ip, u in enumerate(displacements):
                pair, fixed_pair = [], []
                for sign in (1, -1):
                    displaced = sc_epc.makeDisplacedModel(sign*u)
                    pair.append(xxx(solve(displaced, grid=scgrid)))
                    if a.fixed_centers:
                        displaced.rList = sc.rList.copy()
                        displaced.AA_R_W = sc.AA_R_W.copy()
                        displaced.posinfo = copy.deepcopy(sc.posinfo)
                        displaced._refreshHamiltonianDerivedMatrices()
                        fixed_pair.append(xxx(solve(displaced, grid=scgrid)))
                samples.append(np.mean(pair, axis=0))
                if fixed_pair:
                    fixed.append(np.mean(fixed_pair, axis=0))
                print(f'Snapshot pair {ip+1}/{a.pairs}, elapsed {time.perf_counter()-tick:.1f}s', flush=True)
            for name, values in [('snapshot', samples), ('fixed_center_snapshot', fixed)]:
                if not values:
                    continue
                values = np.array(values)
                arrays[name+'_pairs'] = values
                arrays[name] = values.mean(axis=0)
                for part in ('real', 'imag'):
                    arrays[name+'_stderr_'+part] = (getattr(values, part).std(axis=0, ddof=1)/np.sqrt(a.pairs)
                                                    if a.pairs > 1 else np.full(len(a.omega), np.nan))
            meta['timings_seconds']['snapshots'] = time.perf_counter()-tick

    names = [name for name in ('static_serta_frozen', 'static_serta_covariant',
        'dynamic_serta_frozen', 'dynamic_serta_covariant', 'balanced_rta_frozen',
        'balanced_rta_covariant', 'full', 'snapshot', 'fixed_center_snapshot',
        'snapshot_second_order') if name in arrays]
    for name in names:
        reference = arrays['supercell_equilibrium'] if 'snapshot' in name else arrays['baseline']
        arrays[name+'_change'] = arrays[name]-reference
    for name in ('static_serta', 'dynamic_serta', 'balanced_rta'):
        arrays[name+'_derivative_effect'] = arrays[name+'_covariant']-arrays[name+'_frozen']
    with (out/'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['omega_eV', 'method', 'BPVE_real', 'BPVE_imag', 'EPC_change_real',
                         'EPC_change_imag', 'snapshot_stderr_real', 'snapshot_stderr_imag'])
        for iw, omega in enumerate(a.omega):
            print(f'\nPump {omega:g} eV, baseline xxx = {arrays["baseline"][iw].real:.8e}', flush=True)
            for name in names:
                value, change = arrays[name][iw], arrays[name+'_change'][iw]
                errors = [arrays.get(name+'_stderr_'+part, np.full(len(a.omega), np.nan))[iw]
                          for part in ('real', 'imag')]
                writer.writerow([omega, name, value.real, value.imag, change.real, change.imag, *errors])
                print(f'{name:26s} xxx={value.real: .8e}  change={change.real: .8e}'
                      + (f' +/- {errors[0]:.3e} (1 SE)' if np.isfinite(errors[0]) else ''), flush=True)
            for name in ('static_serta', 'dynamic_serta', 'balanced_rta'):
                effect = arrays[name+'_derivative_effect'][iw]
                denominator = abs(arrays[name+'_frozen_change'][iw])
                percent = 100*abs(effect)/denominator if denominator > 1e-12 else np.nan
                print(f'{name:13s} derivative effect={effect.real: .8e}; |effect|/|frozen EPC change|={percent:.3g}%', flush=True)
    meta['elapsed_seconds'] = time.perf_counter()-start
    np.savez(out/'results.npz', **arrays)
    (out/'metadata.json').write_text(json.dumps(meta, indent=2)+'\n', encoding='utf-8')
    print('Saved', out.resolve(), flush=True)
    return arrays, meta


if __name__ == '__main__':
    main()
