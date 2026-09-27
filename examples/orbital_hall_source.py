"""Spin-independent square-lattice s+p source: finite OHC and zero SHC.

Run from any directory: python examples/orbital_hall_source.py
Uses the established LG dc density-matrix calculators, without SOC, exchange,
post-processing cancellation, or a spinless substitute for the spinful model.
See README_orbital_hall_source.md for conventions and interpretation.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from system import System_tb
from calculator import (CollisionSpecification, OrbitalHallConductance_LG_DM,
                        SpinHallConductance_LG_DM)


def build_model(directory, tsp=0.7):
    """Return eight-band System_tb; eV, Angstrom; basis (s,pz,px,py) per spin.

    Fourier convention: H(k)=sum_R H_R exp(+i k.R). The positive-x/y
    s-p hoppings are +tsp, and reverse orbital hoppings are -tsp.
    Thus H_s,px=2i*tsp*sin(kx*a), H_s,py=2i*tsp*sin(ky*a).
    pz has no hybridization but completes the p shell for later atomic SOC.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'POSCAR').write_text(
        'Square s+p orbital Hall source\n1.0\n3 0 0\n0 3 0\n0 0 10\n'
        'X\n1\nCartesian\n0 0 0\n', encoding='utf-8')
    rvec = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
    h = np.zeros((5, 4, 4), dtype=complex)
    h[0] = np.diag([-2.0, 0.0, 0.0, 0.0])
    # ts=-0.5, t_sigma=0.5, t_pi=-0.1; pz uses t_pi in both directions.
    h[1] = np.diag([-0.5, -0.1, 0.5, -0.1])
    h[3] = np.diag([-0.5, -0.1, -0.1, 0.5])
    h[1, 0, 2], h[1, 2, 0] = tsp, -tsp
    h[3, 0, 3], h[3, 3, 0] = tsp, -tsp
    h[2], h[4] = h[1].conj().T, h[3].conj().T
    ham = np.zeros((5, 8, 8), dtype=complex)
    ham[:, :4, :4] = ham[:, 4:, 4:] = h
    with (directory / 'source_hr.dat').open('w', encoding='utf-8') as stream:
        stream.write('Spin-independent square s+p model; eV\n8\n5\n1 1 1 1 1\n')
        for r, block in zip(rvec, ham):
            for j in range(8):
                for i in range(8):
                    z = block[i, j]
                    stream.write(f'{r[0]} {r[1]} {r[2]} {i+1} {j+1} {z.real:.16e} {z.imag:.16e}\n')
    model = System_tb(str(directory), hamR='source', projections={'X': [0, 1]}, ispin=2)
    model.verbose = False
    # Structural proof that no spin-dependent term entered through file input.
    np.testing.assert_array_equal(model.ham_R[:, :4, :4], model.ham_R[:, 4:, 4:])
    np.testing.assert_array_equal(model.ham_R[:, :4, 4:], 0)
    np.testing.assert_array_equal(model.ham_R[:, 4:, :4], 0)
    return model


def calculate(model, grid, fermi, kbt=0.05, linewidth=0.05):
    """Return full native tensors and I/E channels; axes Fermi,drive,flow,polarization."""
    options = dict(model=model, kgrid=(grid, grid, 1), fermiList=np.asarray(fermi),
                   kBT=kbt, collisionSpecification=CollisionSpecification(
                       'crta', constantLinewidth=linewidth),
                   finiteTemperatureOccupations=True, fullCovariantDerivative=True)
    result = {}
    for name, cls in [('ohc', OrbitalHallConductance_LG_DM),
                      ('shc', SpinHallConductance_LG_DM)]:
        calc = cls(**options)
        result[name] = calc.sigma.copy()
        for channel, value in calc.sigma_channels.items():
            result[f'{name}_{channel}'] = value.copy()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('tmp/orbital_hall_source'))
    parser.add_argument('--grids', type=int, nargs='+', default=[12, 24, 48,])
    parser.add_argument('--fermi', type=float, nargs='+', default=np.linspace(-3.5, 2.0, 56).tolist())
    parser.add_argument('--kbt', type=float, default=0.05)
    parser.add_argument('--linewidth', type=float, default=0.05)
    args = parser.parse_args(argv)
    if min(args.grids) < 3 or not np.isfinite(args.kbt) or args.kbt <= 0 or not np.isfinite(args.linewidth) or args.linewidth <= 0:
        parser.error('grids must be >=3, and kbt/linewidth finite and positive')
    if not np.all(np.isfinite(args.fermi)):
        parser.error('Fermi energies must be finite')
    grids = sorted(set(args.grids))
    args.output.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        model = build_model(args.output / 'model')
        control = build_model(args.output / 'no_hybridization', tsp=0)
    fermi = np.asarray(args.fermi)
    arrays = dict(fermi_eV=fermi, grids=np.array(grids))
    rows = []
    with threadpool_limits(limits=1):
        for grid in grids:
            result = calculate(model, grid, fermi, args.kbt, args.linewidth)
            arrays.update({f'{key}_n{grid}': value for key, value in result.items()})
            ohc = result['ohc'][:, 0, 1, 2]
            row = dict(grid=grid, max_abs_shc=float(np.max(np.abs(result['shc']))),
                       max_abs_ohc_xy_z=float(np.max(np.abs(ohc))))
            rows.append(row)
            print(json.dumps(row), flush=True)
        baseline = calculate(control, grids[-1], fermi, args.kbt, args.linewidth)
    arrays.update({f'control_{key}': value for key, value in baseline.items()})
    index = int(np.argmax(np.abs(ohc)))
    tolerance = 1e-10
    checks = dict(
        shc_zero_all_components=all(row['max_abs_shc'] < tolerance for row in rows),
        ohc_nonzero=bool(np.max(np.abs(ohc)) > 1e-3),
        no_hybridization_ohc_zero=bool(np.max(np.abs(baseline['ohc'])) < tolerance),
        no_hybridization_shc_zero=bool(np.max(np.abs(baseline['shc'])) < tolerance),
        ohc_antisymmetric=bool(np.max(np.abs(result['ohc'][:, 0, 1, 2]
                                               + result['ohc'][:, 1, 0, 2])) < tolerance))
    summary = dict(
        model='square s+p, spin independent, no SOC or exchange',
        parameters_eV=dict(eps_s=-2, eps_p=0, ts=-0.5, t_sigma=0.5, t_pi=-0.1,
                           tsp=0.7, kBT=args.kbt, linewidth=args.linewidth),
        tensor_axes=['fermi', 'drive', 'flow', 'polarization'],
        units='native DM Hall normalization: (2*pi/cell_area)*BZ_average; magnetic-moment vertices',
        zero_absolute_tolerance=tolerance, checks=checks, mesh_results=rows,
        representative_fermi_eV=float(fermi[index]),
        representative_ohc_native=float(ohc[index]),
        representative_ohc_by_grid=[float(arrays[f'ohc_n{n}'][index, 0, 1, 2]) for n in grids],
        control_max_abs_ohc=float(np.max(np.abs(baseline['ohc']))))
    if len(grids) > 1:
        previous = arrays[f'ohc_n{grids[-2]}'][:, 0, 1, 2]
        summary['last_mesh_max_difference_over_peak'] = float(np.max(np.abs(ohc-previous))/max(np.max(np.abs(ohc)), 1e-30))
    np.savez_compressed(args.output / 'responses.npz', **arrays)
    np.savetxt(args.output / 'hall_scan.csv', np.column_stack((fermi, ohc,
        result['shc'][:, 0, 1, 2], np.max(np.abs(result['shc']), axis=(1, 2, 3)),
        baseline['ohc'][:, 0, 1, 2])), delimiter=',',
        header='fermi_eV,OHC_Ex_Jy_Lz_native,SHC_Ex_Jy_Sz_native,max_abs_SHC_all_components,control_OHC_native', comments='')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(7, 6), sharex=True, constrained_layout=True)
    for n in grids:
        ax[0].plot(fermi, arrays[f'ohc_n{n}'][:, 0, 1, 2], label=f'OHC {n} x {n}')
    ax[0].plot(fermi, baseline['ohc'][:, 0, 1, 2], 'k--', label='OHC, no s-p mixing')
    ax[0].set_ylabel('OHC (native normalization)')
    ax[0].legend()
    ax[1].plot(fermi, np.max(np.abs(result['shc']), axis=(1, 2, 3)))
    ax[1].set_ylabel('max |SHC|, all components\n(native normalization)')
    ax[1].set_xlabel('Fermi energy (eV)')
    for axis in ax:
        axis.grid(alpha=0.25)
    fig.savefig(args.output / 'hall_scan.png', dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    print(f'Outputs: {args.output.resolve()}')
    if not all(checks.values()):
        raise RuntimeError('Source validation failed; inspect summary.json')


if __name__ == '__main__':
    main()
