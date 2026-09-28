"""Spinful s-only staggered honeycomb: BSR OHC versus ACA OHC and SHC.

python examples/orbital_torque_hex/orbital_hall_source.py
"""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from system import System_tb
from calculator import (CollisionSpecification, OrbitalHallConductance_BlochSelfRotate_LG_DM,
                        OrbitalHallConductance_LG_DM, SpinHallConductance_LG_DM)


def build_model(directory, hopping=1., delta=.5, lattice_constant=2.4):
    """BNcol geometry and hoppings, duplicated as (A up,B up,A down,B down).

    H_AB=-t*(1+exp(i k.a1)+exp(-i k.a2)); H_AA=Delta=-H_BB.
    All sites host only s orbitals. Energies eV, lengths Angstrom.
    """
    if not np.all(np.isfinite([hopping, delta, lattice_constant])) or lattice_constant <= 0:
        raise ValueError('Finite parameters and positive lattice constant required')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    a = lattice_constant
    (directory/'POSCAR').write_text(
        f'Spinful s-only honeycomb\n1.0\n{a:.16g} 0 0\n{-a/2:.16g} {a*np.sqrt(3)/2:.16g} 0\n'
        '0 0 15\nA B\n1 1\nDirect\n'
        '0.6666666666666667 0.3333333333333333 0.5\n'
        '0.3333333333333333 0.6666666666666667 0.5\n')
    rvec = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, -1, 0], [0, 1, 0]])
    h = np.zeros((5, 2, 2), complex)
    h[0] = [[delta, -hopping], [-hopping, -delta]]
    h[1, 0, 1] = h[3, 0, 1] = -hopping
    h[2] = h[1].conj().T
    h[4] = h[3].conj().T
    ham = np.zeros((5, 4, 4), complex)
    ham[:, :2, :2] = ham[:, 2:, 2:] = h
    with (directory/'source_hr.dat').open('w') as stream:
        stream.write('Spin-degenerate s-only honeycomb; eV\n4\n5\n1 1 1 1 1\n')
        for r, block in zip(rvec, ham):
            for j in range(4):
                for i in range(4):
                    value = block[i, j]
                    stream.write(f'{r[0]} {r[1]} {r[2]} {i+1} {j+1} {value.real:.16e} {value.imag:.16e}\n')
    model = System_tb(str(directory), hamR='source', projections={'A': [0], 'B': [0]}, ispin=2)
    model.verbose = False
    np.testing.assert_array_equal(model.ham_R[:, :2, :2], model.ham_R[:, 2:, 2:])
    np.testing.assert_array_equal(model.ham_R[:, :2, 2:], 0)
    np.testing.assert_array_equal(model.ham_R[:, 2:, :2], 0)
    np.testing.assert_array_equal(model.orbital, 0)
    return model


def calculate(model, grid, fermi, kbt=.01, linewidth=.02):
    """Native magnetic-moment Hall tensors: Fermi, field, flow, polarization."""
    options = dict(model=model, kgrid=(grid, grid, 1), fermiList=np.asarray(fermi), kBT=kbt,
        collisionSpecification=CollisionSpecification('crta', constantLinewidth=linewidth),
        fullCovariantDerivative=True, finiteTemperatureOccupations=True)
    result = {}
    states = None
    for name, cls in [('bsr', OrbitalHallConductance_BlochSelfRotate_LG_DM),
                      ('aca', OrbitalHallConductance_LG_DM), ('shc', SpinHallConductance_LG_DM)]:
        calc = cls(states=states, **options)
        states = calc.states
        result[name] = calc.sigma.copy()
        result.update({f'{name}_{channel}': value.copy() for channel, value in calc.sigma_channels.items()})
    result['band_edges'] = np.array([states.energies[..., 1].max(), states.energies[..., 2].min()])
    return result


def plot_results(arrays, summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fermi, grids = arrays['fermi_eV'], arrays['grids']
    grid = int(grids[-1])
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True, constrained_layout=True)
    for n in grids:
        axes[0].plot(fermi, arrays[f'bsr_n{n}'][:, 0, 1, 2], label=f'BSR OHC {n} x {n}')
    axes[0].plot(fermi, arrays[f'aca_n{grid}'][:, 0, 1, 2], '--', label='ACA OHC (s orbitals)')
    axes[0].plot(fermi, arrays['atomic_bsr'][:, 0, 1, 2], ':', label='BSR OHC, hopping off')
    for channel in ('I', 'E'):
        axes[1].plot(fermi, arrays[f'bsr_{channel}_n{grid}'][:, 0, 1, 2], label=f'BSR {channel} channel')
    axes[2].plot(fermi, arrays[f'shc_n{grid}'][:, 0, 1, 2], label='SHC: Ex, Jy, Sz')
    axes[2].plot(fermi, np.max(np.abs(arrays[f'shc_n{grid}']), axis=(1, 2, 3)), '--', label='max |SHC| over all components')
    axes[2].set_ylim(-1e-10, 1e-10)
    axes[2].text(.5, .85, 'Numerical zero; raw values retained', transform=axes[2].transAxes, ha='center')
    for ax in axes:
        ax.axvspan(*summary['analytic_gap_edges_eV'], color='grey', alpha=.12)
        ax.axhline(0, color='0.5', lw=.6)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        ax.set_ylabel('Hall response\n(native normalization)')
    axes[0].set_title('s-only honeycomb: Ex-induced Jy of z magnetic moment\nShaded: bulk band gap')
    axes[2].set_xlabel('Fermi energy (eV)')
    fig.savefig(Path(output)/'hall_scan.png', dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('tmp/orbital_torque_hex/source'))
    parser.add_argument('--grids', type=int, nargs='+', default=[48, 96, 192])
    parser.add_argument('--fermi', type=float, nargs='+', default=np.linspace(-3.2, 3.2, 81).tolist())
    for key, default in [('hopping', 1.), ('delta', .5), ('lattice-constant', 2.4), ('kbt', .01), ('linewidth', .02)]:
        parser.add_argument('--'+key, type=float, default=default)
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args(argv)
    if args.plot_only:
        with np.load(args.output/'responses.npz') as arrays:
            plot_results(arrays, json.loads((args.output/'summary.json').read_text()), args.output)
        return
    if min(args.grids) < 3 or min(args.kbt, args.linewidth, args.lattice_constant) <= 0 or not np.all(np.isfinite(
            [args.kbt, args.linewidth, args.lattice_constant, args.hopping, args.delta, *args.fermi])):
        parser.error('Finite parameters, grids >=3 and positive kBT, linewidth, lattice constant required')
    if args.delta == 0 or args.hopping == 0:
        parser.error('Use nonzero delta and hopping for the gapped source; hopping-off control is automatic')
    args.output.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        model = build_model(args.output/'model', args.hopping, args.delta, args.lattice_constant)
        atomic = build_model(args.output/'atomic_control', 0., args.delta, args.lattice_constant)
    for name, system in [('model', model), ('atomic_control', atomic)]:
        np.savez_compressed(args.output/name/'model.npz', ham_R=system.ham_R, iRvec=system.iRvec,
            lattice=system.lattice, rList=system.rList, spin=system.spin, orbital=system.orbital)
    grids = sorted(set(args.grids))
    fermi = np.asarray(args.fermi)
    arrays = dict(fermi_eV=fermi, grids=np.array(grids))
    metrics = []
    with threadpool_limits(limits=1):
        for grid in grids:
            result = calculate(model, grid, fermi, args.kbt, args.linewidth)
            arrays.update({f'{key}_n{grid}': value for key, value in result.items()})
            row = dict(grid=grid, max_abs_shc=float(np.max(np.abs(result['shc']))),
                       max_abs_aca=float(np.max(np.abs(result['aca']))),
                       max_abs_bsr=float(np.max(np.abs(result['bsr']))), band_edges_eV=result['band_edges'].tolist())
            metrics.append(row)
            print(json.dumps(row), flush=True)
        control = calculate(atomic, grids[-1], fermi, args.kbt, args.linewidth)
        # Always evaluate midgap explicitly, even when absent from the requested scan.
        midgap = calculate(model, grids[-1], [0.], args.kbt, args.linewidth)
        narrower = calculate(model, grids[-1], [0.], args.kbt/2, args.linewidth/2)
    arrays.update({f'atomic_{key}': value for key, value in control.items()})
    arrays.update({f'midgap_{key}': value for key, value in midgap.items()})
    arrays.update({f'narrower_midgap_{key}': value for key, value in narrower.items()})
    tol = 1e-10
    checks = dict(aca_moment_identically_zero=bool(np.all(model.orbital == 0)),
        aca_ohc_zero=all(row['max_abs_aca'] < tol for row in metrics),
        shc_zero=all(row['max_abs_shc'] < tol for row in metrics),
        atomic_limit_zero=all(np.max(np.abs(control[key])) < tol for key in ('bsr', 'aca', 'shc')),
        midgap_bsr_finite=bool(abs(midgap['bsr'][0, 0, 1, 2]) > 1e-6),
        narrower_midgap_bsr_finite=bool(abs(narrower['bsr'][0, 0, 1, 2]) > 1e-6),
        bsr_hall_antisymmetric=bool(np.max(np.abs(result['bsr'][:, 0, 1, 2]+result['bsr'][:, 1, 0, 2])) < tol))
    summary = dict(parameters={k: v for k, v in vars(args).items() if k != 'output'}, checks=checks,
        analytic_gap_edges_eV=[-abs(args.delta), abs(args.delta)],
        axes=['fermi', 'drive', 'flow', 'polarization'],
        units='Native DM 2D Hall normalization: 2*pi/cell_area times BZ mean, magnetic-moment vertices; not a quantized conductance unit',
        spin_degeneracy=2, mesh_results=metrics, zero_tolerance=tol,
        midgap_bsr=float(midgap['bsr'][0, 0, 1, 2]),
        midgap_bsr_I=float(midgap['bsr_I'][0, 0, 1, 2]), midgap_bsr_E=float(midgap['bsr_E'][0, 0, 1, 2]),
        narrower_midgap_bsr=float(narrower['bsr'][0, 0, 1, 2]),
        narrower_kbt=args.kbt/2, narrower_linewidth=args.linewidth/2)
    if len(grids) > 1:
        summary['last_mesh_max_bsr_difference'] = float(np.max(np.abs(result['bsr']-arrays[f'bsr_n{grids[-2]}'])))
    np.savez_compressed(args.output/'responses.npz', **arrays)
    np.savetxt(args.output/'hall_scan.csv', np.column_stack([fermi, result['bsr'][:, 0, 1, 2],
        result['bsr_I'][:, 0, 1, 2], result['bsr_E'][:, 0, 1, 2], result['aca'][:, 0, 1, 2],
        result['shc'][:, 0, 1, 2], np.max(np.abs(result['shc']), axis=(1, 2, 3)), control['bsr'][:, 0, 1, 2]]),
        delimiter=',', header='fermi_eV,BSR_OHC,BSR_I,BSR_E,ACA_OHC,SHC,max_abs_SHC,atomic_BSR_OHC', comments='')
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    plot_results(arrays, summary, args.output)
    print(json.dumps(summary, indent=2), flush=True)
    if not all(checks.values()):
        raise RuntimeError('Honeycomb source checks failed; inspect summary.json')


if __name__ == '__main__':
    main()
