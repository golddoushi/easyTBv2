"""Compare model Fan+DW band-gap shifts with Gaussian displaced supercells.

    python examples/honeycomb_self_energy.py

Uses a small, matched k/q mesh for a finite-supercell weak-coupling benchmark.
This does not claim thermodynamic-limit convergence or validate dynamical
phonon lifetimes by static disorder. Each independent MC sample is paired
with its negative to remove odd displacement contributions.
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
from calculator import ModelElectronSelfEnergy
from calculator.model_self_energy import uniform_qmesh
from examples.honeycomb_four_band import build_model, write_hr
from system import SSHModelEPC


def sample_gaps(sc_epc, temperature, pairs, seed):
    """Independent antithetic Gaussian pairs; return their gap shifts in eV."""
    sc = sc_epc.electronic_model
    bare = np.linalg.eigvalsh(sc.ham_R.sum(axis=0))
    occupied = sc.norbs//2
    gap0 = bare[occupied]-bare[occupied-1]
    if gap0 <= 0:
        raise ValueError("The supercell must have a positive half-filling gap")
    displacements = sc_epc.phonon_model.sampleDisplacements(temperature, nSamples=pairs, seed=seed)
    shifts = np.zeros(pairs)
    for sample, u in enumerate(displacements):
        for sign in (-1, 1):
            # Identical electronic Hamiltonian to makeDisplacedModel, without
            # copying/rebuilding position and velocity operators not used here.
            h = sc.ham_R + sc_epc.deltaHamiltonian(sign*u)
            energies = np.linalg.eigvalsh(h.sum(axis=0))
            shifts[sample] += 0.5*(energies[occupied]-energies[occupied-1]-gap0)
    return gap0, shifts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='tmp/honeycomb_self_energy')
    parser.add_argument('--qgrid', nargs=3, type=int, default=[3, 2, 1])
    parser.add_argument('--temperatures', nargs='+', type=float, default=[0, 300])
    parser.add_argument('--pairs', type=int, default=2000)
    parser.add_argument('--scale', type=float, default=0.05,
                        help='scale linear EPC by s and quadratic EPC by s^2')
    parser.add_argument('--beta', type=float, default=0.3,
                        help='quadratic coefficient before scale^2, eV/Angstrom^2')
    parser.add_argument('--eta', type=float, default=1e-7, help='positive retarded regulator in eV')
    parser.add_argument('--seed', type=int, default=5739)
    args = parser.parse_args()
    if args.pairs < 2 or not np.isfinite(args.scale) or args.scale <= 0:
        parser.error('pairs must be at least two and scale positive/finite')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    grid = tuple(args.qgrid)
    kpoints = uniform_qmesh(grid)
    rows, arrays = [], {'kpoints': kpoints}
    for quadratic in (False, True):
        name = 'quadratic' if quadratic else 'linear'
        directory = output/name
        with contextlib.redirect_stdout(io.StringIO()):
            model, epc = build_model(directory, alpha=0.5*args.scale)
            if quadratic:
                beta = np.where(epc.alpha_R[:, :2, :2] != 0, args.beta*args.scale**2, 0)
                write_hr(directory/'quadratic_hr.dat', model.iRvec, beta, 'd2H/d(projected stretch)^2; eV/Angstrom^2')
                epc = SSHModelEPC.from_files(model, epc.phonon_model, betaR='quadratic')
            sc_epc = epc.supercell(np.diag(grid))
            sc = sc_epc.electronic_model
        for it, temperature in enumerate(args.temperatures):
            common = dict(epc_model=epc, kpoints=kpoints, qgrid=grid, temperature=temperature,
                          fermi=0.0, delta=args.eta)
            static = ModelElectronSelfEnergy(adiabatic=True, **common)
            dynamic = ModelElectronSelfEnergy(adiabatic=False, **common)
            vbm = np.unravel_index(np.argmax(static.energies[:, :2]), (len(kpoints), 2))
            cbm0 = np.unravel_index(np.argmin(static.energies[:, 2:]), (len(kpoints), 2))
            cbm = (cbm0[0], cbm0[1]+2)
            fan_shift = float((static.fan_migdal[cbm]-static.fan_migdal[vbm]).real)
            dw_shift = float(static.debye_waller[cbm]-static.debye_waller[vbm])
            gap0, samples = sample_gaps(sc_epc, temperature, args.pairs, args.seed+it)
            average = float(samples.mean())
            error = float(samples.std(ddof=1)/np.sqrt(args.pairs))
            prediction = static.bandGap(2)['gap_shift']
            if not np.isclose(static.bandGap(2)['bare_gap'], gap0, atol=1e-12, rtol=0):
                raise RuntimeError('Primitive and supercell bare gaps disagree')
            row = dict(model=name, temperature_K=temperature, qgrid=list(grid), pairs=args.pairs,
                       scale=args.scale, beta_before_scaling=args.beta if quadratic else 0.0,
                       effective_beta_eV_per_A2=args.beta*args.scale**2 if quadratic else 0.0,
                       bare_gap_eV=gap0, vbm_k=kpoints[vbm[0]].tolist(), cbm_k=kpoints[cbm[0]].tolist(),
                       static_fan_gap_shift_eV=fan_shift, dw_gap_shift_eV=dw_shift,
                       max_abs_dw_band_shift_eV=float(np.max(np.abs(static.debye_waller))),
                       static_total_gap_shift_eV=prediction,
                       dynamic_total_gap_shift_eV=dynamic.bandGap(2)['gap_shift'],
                       mc_gap_shift_eV=average, mc_standard_error_eV=error,
                       discrepancy_in_standard_errors=(average-prediction)/error if error else None)
            rows.append(row)
            prefix = f'{name}_T{it}'
            arrays.update({prefix+'_bare_energies_eV': static.energies,
                           prefix+'_fan_static_eV': static.fan_migdal,
                           prefix+'_fan_dynamic_eV': dynamic.fan_migdal,
                           prefix+'_dw_eV': static.debye_waller,
                           prefix+'_mc_gap_shifts_eV': samples})
            print(json.dumps(row), flush=True)
    np.savez(output/'self_energy_results.npz', **arrays)
    (output/'comparison.json').write_text(json.dumps(rows, indent=2)+'\n', encoding='utf-8')
    print(f'Outputs: {output.resolve()}')


if __name__ == '__main__':
    main()
