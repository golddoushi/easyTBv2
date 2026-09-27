"""Step 3: OHC source N plus atomic-SOC converter C, periodic x and finite y.

python examples/orbital_torque/orbital_nc_ribbon.py --workers 8
Compare full model, N mixing off, C SOC off, disconnected interface, and N-only drive.
Also resolve C-total moments versus the individually driven row at the finest mesh.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.orbital_torque.orbital_ribbon import build_ribbon, calculate_profile, effective_workers
from calculator import CollisionSpecification
from calculator.response_stages import DensityMatrixResponse
from calculator.density_matrix import LengthGaugeFirstOrderDMKernel


def calculate_drive_rows(model, n_width, grid, fermi, kbt=.05, linewidth=.05, workers=1):
    """C-integrated moment / Ex for each individually driven row, axes row, EF, xyz.

    Contract the scalar-CRTA kernel with the C observable first, then trace
    against each P_row v_x P_row. This is the same linear response as separate
    row drives, without a new eigensolve or full density matrix per row.
    """
    if model.overlap or not np.all(model.iRvec[:, 1:] == 0):
        raise ValueError('An orthogonal, x-periodic ribbon is required')
    order = np.argsort(np.asarray(model.posinfo['pos ions (cart.)'])[:, 1])
    if not 0 < n_width < len(order) or grid < 3 or workers < 1:
        raise ValueError('Nonempty N/C, grid >= 3 and workers >= 1 required')
    atom_map = model.iorb2iatomMap
    c_mask = np.isin(atom_map, order[n_width:])
    ids = [np.flatnonzero(atom_map == atom) for atom in order]
    operators = {name: -op*c_mask[:, None]*c_mask[None, :]
                 for name, op in [('orbital', model.orbital), ('spin', model.spin)]}
    count = effective_workers(grid, workers)

    def shard(index):
        prepared = DensityMatrixResponse(model=model, kgrid=(grid//count, 1, 1),
            kOffset=(index/grid, 0., 0.), fermiList=np.asarray(fermi), kBT=kbt,
            collisionSpecification=CollisionSpecification('crta', constantLinewidth=linewidth))
        kernel = LengthGaugeFirstOrderDMKernel(prepared.inputs,
            full_covariant_derivative=True, finite_temperature_occupations=True,
            collision=prepared.collision)
        u = prepared.states.vectors
        ud = u.conj().swapaxes(-1, -2)
        vx = u @ prepared.inputs.velocity_K[..., 0, :, :] @ ud
        interrow = atom_map[:, None] != atom_map[None, :]
        if np.max(np.abs(vx[..., interrow])) > 1e-10:
            raise ValueError('Row-drive additivity requires no inter-row v_x matrix elements')
        result = {name: np.zeros((len(order), len(fermi), 3)) for name in operators}
        for fi, mu in enumerate(fermi):
            state = kernel.fermi_state(fi, mu)
            # rho_mn = weight_mn * v_mn. Move this elementwise weight
            # onto O_nm in Tr(rho O); transpose, never conjugate, the weight.
            weight = state.collision.solve(-1j*state.divided_difference,
                                           prepared.inputs.deltaE, 0.)
            for name, operator in operators.items():
                for component in range(3):
                    observable = ud @ operator[component] @ u
                    dual = u @ (weight.swapaxes(-1, -2)*observable) @ ud
                    for row, local in enumerate(ids):
                        v = vx[..., local[:, None], local]
                        b = dual[..., local[:, None], local]
                        result[name][row, fi, component] = np.einsum('...ij,...ji->...', v, b).real.mean()
        return result

    with threadpool_limits(limits=1):
        if count == 1:
            return shard(0)
        with ThreadPoolExecutor(max_workers=count) as pool:
            parts = list(pool.map(shard, range(count)))
        return {name: sum(part[name] for part in parts)/count for name in operators}


def plot_drive_rows(results, regions, fermi, grid, output):
    """Abscissa is the driven row; each ordinate sums moments over all C rows."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = int(np.sum(regions == 'N'))
    x = np.arange(len(regions))
    styles = {'full': ('o-', 'N mixing + C SOC'),
              'no_mixing': ('s--', 'N mixing off; C SOC on'),
              'no_soc': ('^:', 'C SOC off'),
              'disconnected': ('D-.', 'Interface off'),
              'n_drive': ('v--', 'Everything on; N-only drive')}
    def draw(ax, name, component, fi):
        for case, response in results.items():
            fmt, label = styles[case]
            ax.plot(x, response[name][:, fi, component], fmt, ms=4, label=label)
        ax.axvspan(-.5, n-.5, color='steelblue', alpha=.07)
        ax.axvspan(n-.5, len(x)-.5, color='orange', alpha=.09)
        ax.axvline(n-.5, color='0.35', ls='--', lw=.8)
        ax.axhline(0, color='0.5', lw=.6)
        ax.set_ylabel(f'C-total {name} ({"xyz"[component]})\n'+r'$\delta\mu/E_x$ [$\mu_B$ / (V/$\AA$)]')
        ax.grid(alpha=.2)
        if max(np.max(np.abs(p[name][:, fi, component])) for p in results.values()) < 1e-9:
            ax.set_ylim(-1e-9, 1e-9)
            ax.text(.5, .85, 'Numerical zero', transform=ax.transAxes, ha='center')
    for fi, mu in enumerate(fermi):
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
        for ax, name in zip(axes, ('orbital', 'spin')):
            draw(ax, name, 2, fi)
            ax.legend(fontsize=8, ncol=2)
        axes[0].set_title(f'One row driven at a time; moments summed over C\nN({n}) | C({len(x)-n}), EF={mu:g} eV, Nk={grid}')
        axes[1].set_xlabel('Driven row (N: blue shading; C: orange shading)')
        fig.savefig(Path(output)/f'drive_rows_f{fi}.png', dpi=160)
        plt.close(fig)
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, constrained_layout=True)
        for row, name in enumerate(('orbital', 'spin')):
            for component in range(3):
                draw(axes[row, component], name, component, fi)
                if row == 1:
                    axes[row, component].set_xlabel('Driven row (N left; C right)')
        axes[0, 2].legend(fontsize=8)
        fig.suptitle(f'C-total moments versus driven row: N({n}) | C({len(x)-n}), EF={mu:g} eV, Nk={grid}')
        fig.savefig(Path(output)/f'all_components_drive_rows_f{fi}.png', dpi=160)
        plt.close(fig)


def build_nc_ribbon(directory, n_width=8, c_width=6, tsp=0.7, soc=0.4,
                    converter_hopping=0.5, interface_hopping=0.3, converter_shift=0.):
    """Same full s+p shells throughout; no exchange, spin-independent interface.

    N keeps the source hoppings. C has onsite (-2,0,0,0)+shift and -t_C*I4
    nearest-neighbor hopping, with no s-p mixing. The interface is -t_I*I4.
    SOC is lambda*(L/hbar).sigma/2 on C atoms only. This deliberately simple
    converter has no momentum-dependent orbital texture of its own.
    """
    if min(n_width, c_width) < 1:
        raise ValueError('Each region must contain at least one row')
    if not np.all(np.isfinite([tsp, soc, converter_hopping, interface_hopping, converter_shift])):
        raise ValueError('Hamiltonian parameters must be finite')
    model = build_ribbon(directory, n_width+c_width, tsp=tsp)
    order = np.argsort(np.asarray(model.posinfo['pos ions (cart.)'])[:, 1])
    half = model.norbs // 2
    spatial_atoms = model.iorb2iatomMap[:half]
    ids = [np.flatnonzero(spatial_atoms == atom) for atom in order]
    h = model.ham_R[:, :half, :half].copy()
    for ir, r in enumerate(model.iRvec):
        for a in range(len(order)):
            for b in range(len(order)):
                if a < n_width and b < n_width:
                    continue
                block = np.zeros((4, 4), dtype=complex)
                same_cell = np.all(r == 0)
                onsite = same_cell and a == b
                neighbor = (same_cell and abs(a-b) == 1) or (abs(r[0]) == 1 and a == b)
                if a >= n_width and b >= n_width:
                    if onsite:
                        block = np.diag(np.array([-2., 0., 0., 0.])+converter_shift)
                    elif neighbor:
                        block = -converter_hopping*np.eye(4)
                elif neighbor:
                    block = -interface_hopping*np.eye(4)
                h[ir][np.ix_(ids[a], ids[b])] = block
    model.ham_R[:] = 0
    model.ham_R[:, :half, :half] = h
    model.ham_R[:, half:, half:] = h
    soc_R = np.zeros_like(model.ham_R)
    ir0 = next(i for i, r in enumerate(model.iRvec) if np.all(r == 0))
    for atom in order[n_width:]:
        local = np.flatnonzero(model.iorb2iatomMap == atom)
        angular = model.orbital[:, local][:, :, local]
        pauli = model.spin[:, local][:, :, local]
        soc_R[ir0][np.ix_(local, local)] = soc/2*sum(angular[a] @ pauli[a] for a in range(3))
    model.ham_R += soc_R
    model._refreshHamiltonianDerivedMatrices()
    lookup = {tuple(r): i for i, r in enumerate(model.iRvec)}
    for i, r in enumerate(model.iRvec):
        np.testing.assert_allclose(model.ham_R[i], model.ham_R[lookup[tuple(-r)]].conj().T, atol=1e-14)
    regions = np.array(['N']*n_width + ['C']*c_width)
    return model, regions, soc_R


def plot_profiles(results, regions, fermi, grid, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles = {'full': ('o-', 'N mixing + C SOC'),
              'no_mixing': ('s--', 'N mixing off; C SOC on'),
              'no_soc': ('^:', 'C SOC off; uniform drive'),
              'disconnected': ('D-.', 'Interface off; uniform drive'),
              'n_drive': ('v--', 'Everything on; N-only drive')}
    n_width = int(np.sum(regions == 'N'))
    x = np.arange(len(regions))
    for fi, mu in enumerate(fermi):
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, constrained_layout=True)
        for row, name in enumerate(('orbital', 'spin')):
            for component, axis in enumerate(axes[row]):
                for case, profile in results.items():
                    fmt, label = styles[case]
                    axis.plot(x, profile[name][:, fi, 0, component], fmt, ms=4, label=label)
                axis.axvspan(-.5, n_width-.5, color='steelblue', alpha=.07)
                axis.axvspan(n_width-.5, len(x)-.5, color='orange', alpha=.09)
                axis.axvline(n_width-.5, color='0.35', ls='--', lw=.8)
                axis.axhline(0, color='0.5', lw=.6)
                axis.set_title(f'{name.capitalize()} moment: {"xyz"[component]}')
                axis.set_ylabel(r'$\delta\mu/E_x$ [$\mu_B$ / (V/$\AA$)]')
                axis.grid(alpha=.2)
                axis.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
                if max(np.max(np.abs(p[name][:, fi, 0, component])) for p in results.values()) < 1e-9:
                    axis.set_ylim(-1e-9, 1e-9)
                    axis.text(.5, .9, 'Numerical zero', transform=axis.transAxes, ha='center', fontsize=9)
                if row == 1:
                    axis.set_xlabel('Row (N on left, C on right)')
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f'N({n_width}) | C({len(x)-n_width}), EF={mu:g} eV, Nk={grid}; drive specified in legend')
        fig.savefig(Path(output) / f'all_components_f{fi}.png', dpi=160)
        plt.close(fig)
        fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, constrained_layout=True)
        for axis, name in zip(axes, ('orbital', 'spin')):
            for case, profile in results.items():
                fmt, label = styles[case]
                axis.plot(x, profile[name][:, fi, 0, 2], fmt, label=label)
            axis.axvspan(-.5, n_width-.5, color='steelblue', alpha=.07)
            axis.axvspan(n_width-.5, len(x)-.5, color='orange', alpha=.09)
            axis.axvline(n_width-.5, color='0.35', ls='--', lw=.8)
            axis.axhline(0, color='0.5', lw=.6)
            axis.set_ylabel(f'{name.capitalize()} '+r'$\delta\mu_z/E_x$ [$\mu_B$ / (V/$\AA$)]')
            axis.grid(alpha=.2)
        axes[0].legend(fontsize=8, ncol=2)
        axes[0].set_title(f'N({n_width}) | C({len(x)-n_width}), EF={mu:g} eV, Nk={grid}')
        axes[1].set_xlabel('Atomic row (N: blue shading; C: orange shading)')
        fig.savefig(Path(output) / f'profile_f{fi}.png', dpi=160)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('tmp/orbital_torque/nc_ribbon'))
    parser.add_argument('--n-width', type=int, default=8)
    parser.add_argument('--c-width', type=int, default=6)
    parser.add_argument('--grids', type=int, nargs='+', default=[256, 512, 1024])
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--fermi', type=float, nargs='+', default=[0.8])
    parser.add_argument('--soc', type=float, default=0.4, help='Atomic lambda in eV, H_SOC=lambda*L.S/hbar^2')
    parser.add_argument('--tsp', type=float, default=0.7)
    parser.add_argument('--converter-hopping', type=float, default=0.5)
    parser.add_argument('--interface-hopping', type=float, default=0.3)
    parser.add_argument('--converter-shift', type=float, default=0.)
    parser.add_argument('--kbt', type=float, default=0.05)
    parser.add_argument('--linewidth', type=float, default=0.05)
    parser.add_argument('--plot-only', action='store_true', help='Replot saved results in --output')
    args = parser.parse_args(argv)
    if args.plot_only:
        summary = json.loads((args.output / 'summary.json').read_text(encoding='utf-8'))
        grid = max(summary['parameters']['grids'])
        with np.load(args.output / 'responses.npz') as data:
            results = {case: {name: data[f'{case}_{name}_n{grid}'] for name in ('orbital', 'spin')}
                       for case in summary['metrics']}
            plot_profiles(results, data['regions'], data['fermi_eV'], grid, args.output)
            row_results = {case: {name: data[f'drive_rows_{case}_{name}']
                           for name in ('orbital', 'spin')} for case in summary['metrics']}
            plot_drive_rows(row_results, data['regions'], data['fermi_eV'], grid, args.output)
        return
    if min(args.n_width, args.c_width, args.workers) < 1 or min(args.grids) < 3:
        parser.error('Positive widths/workers and grids >=3 required')
    if not np.all(np.isfinite([args.kbt, args.linewidth, args.soc, args.tsp,
        args.converter_hopping, args.interface_hopping, args.converter_shift, *args.fermi])) or min(args.kbt, args.linewidth) <= 0:
        parser.error('Finite parameters and positive kBT/linewidth required')
    args.output.mkdir(parents=True, exist_ok=True)
    grids = sorted(set(args.grids))
    arrays = dict(fermi_eV=np.asarray(args.fermi), grids=np.array(grids))
    records, finest, metrics = [], {}, {}
    row_drives = {}
    cases = [('full', args.tsp, args.soc, args.interface_hopping),
             ('no_mixing', 0., args.soc, args.interface_hopping),
             ('no_soc', args.tsp, 0., args.interface_hopping),
             ('disconnected', args.tsp, args.soc, 0.),
             ('n_drive', args.tsp, args.soc, args.interface_hopping)]
    for case, mixing, coupling, interface in cases:
        with contextlib.redirect_stdout(io.StringIO()):
            model, regions, soc_R = build_nc_ribbon(args.output / case / 'primitive_source',
                args.n_width, args.c_width, mixing, coupling, args.converter_hopping,
                interface, args.converter_shift)
        order = np.argsort(np.asarray(model.posinfo['pos ions (cart.)'])[:, 1])
        drive_atoms = order[:args.n_width].tolist() if case == 'n_drive' else None
        np.savez_compressed(args.output / case / 'model.npz', ham_R=model.ham_R,
            soc_R=soc_R, iRvec=model.iRvec, lattice=model.lattice, rList=model.rList,
            orbital=model.orbital, spin=model.spin, regions=regions)
        case_metrics = []
        previous = None
        for grid in grids:
            start = perf_counter()
            result = calculate_profile(model, grid, args.fermi, args.kbt, args.linewidth,
                                       workers=args.workers, drive_atoms=drive_atoms)
            arrays.update({f'{case}_{key}_n{grid}': value for key, value in result.items()})
            metric = dict(grid=grid, seconds=perf_counter()-start, workers=effective_workers(grid, args.workers))
            for name in ('orbital', 'spin'):
                response = result[name][:, :, 0, :]
                metric[f'max_abs_{name}_Ex'] = float(np.max(np.abs(response)))
                metric[f'max_abs_{name}_Ex_C'] = float(np.max(np.abs(response[args.n_width:])))
                metric[f'{name}_sum_residual'] = float(np.max(np.abs(result[name].sum(axis=0)-result[f'{name}_total'])))
                if previous is not None:
                    metric[f'{name}_mesh_abs_difference'] = float(np.max(np.abs(response-previous[name][:, :, 0, :])))
            previous = result
            case_metrics.append(metric)
            print(json.dumps(dict(case=case, **metric)), flush=True)
        metrics[case], finest[case] = case_metrics, result
        if case == 'n_drive':
            # The same Hamiltonian as full; C rows are not driven in this protocol.
            row_drives[case] = {name: value.copy() for name, value in row_drives['full'].items()}
            for value in row_drives[case].values():
                value[args.n_width:] = 0.
        else:
            start = perf_counter()
            row_drives[case] = calculate_drive_rows(model, args.n_width, grids[-1], args.fermi,
                args.kbt, args.linewidth, workers=args.workers)
            print(json.dumps(dict(case=case, row_drive_seconds=perf_counter()-start)), flush=True)
        arrays.update({f'drive_rows_{case}_{name}': value for name, value in row_drives[case].items()})
        for layer, region in enumerate(regions):
            for fi, mu in enumerate(args.fermi):
                for component in range(3):
                    records.append([case, layer, region, result['y_A'][layer], mu, 'xyz'[component],
                        *[result[key][layer, fi, 0, component] for key in
                          ('orbital', 'orbital_I', 'orbital_E', 'spin', 'spin_I', 'spin_E')]])
    tolerance = 1e-9
    row_residuals = {f'{case}_{name}': float(np.max(np.abs(
        row_drives[case][name].sum(axis=0)-finest[case][name][args.n_width:, :, 0].sum(axis=0))))
        for case in row_drives for name in ('orbital', 'spin')}
    row_residuals.update({f'N_only_{name}': float(np.max(np.abs(
        row_drives['full'][name][:args.n_width].sum(axis=0)
        -finest['n_drive'][name][args.n_width:, :, 0].sum(axis=0)))) for name in ('orbital', 'spin')})
    checks = dict(
        row_drives_reconstruct_uniform_and_N_only=all(value < tolerance for value in row_residuals.values()),
        disconnected_row_drives_zero_C=all(np.max(np.abs(value)) < tolerance for value in row_drives['disconnected'].values()),
        no_mixing_row_drives_zero_C=all(np.max(np.abs(value)) < tolerance for value in row_drives['no_mixing'].values()),
        no_soc_row_drives_zero_spin=bool(np.max(np.abs(row_drives['no_soc']['spin'])) < tolerance),
        no_soc_zero_spin=bool(np.max(np.abs(finest['no_soc']['spin'][:, :, 0])) < tolerance),
        no_mixing_zero_orbital_Ex=bool(np.max(np.abs(finest['no_mixing']['orbital'][:, :, 0])) < tolerance),
        no_mixing_zero_spin_Ex=bool(np.max(np.abs(finest['no_mixing']['spin'][:, :, 0])) < tolerance),
        local_sums_match_totals=all(row[f'{name}_sum_residual'] < tolerance
            for case in metrics.values() for row in case for name in ('orbital', 'spin')),
        finite_N_orbital=bool(np.max(np.abs(finest['full']['orbital'][:args.n_width, :, 0])) > 1e-5),
        finite_C_spin=bool(np.max(np.abs(finest['full']['spin'][args.n_width:, :, 0])) > 1e-5),
        disconnected_zero_C=all(np.max(np.abs(finest['disconnected'][name][args.n_width:, :, 0])) < tolerance for name in ('orbital', 'spin')),
        disconnected_zero_spin=bool(np.max(np.abs(finest['disconnected']['spin'][:, :, 0])) < tolerance),
        n_drive_finite_C_spin=bool(np.max(np.abs(finest['n_drive']['spin'][args.n_width:, :, 0])) > 1e-5))
    summary = dict(parameters={key: value for key, value in vars(args).items() if key != 'output'},
        drive={case: 'N-only projected vertex P_N v P_N' if case == 'n_drive' else 'Uniform Ex in N and C' for case in metrics},
        units='mu_B/(V/Angstrom) per row per x repeat',
        tensor_axes=['layer', 'fermi', 'field', 'moment'], checks=checks, zero_tolerance=tolerance,
        row_drive_axes=['driven_row', 'fermi', 'moment'], row_drive_grid=grids[-1],
        row_drive_units='mu_B/(V/Angstrom) summed over C per x repeat; same field per driven row',
        row_drive_sum_residuals=row_residuals,
        metrics=metrics, interpretation='Accumulation and parameter controls; not a direct interface-current or continuity measurement.')
    arrays['regions'] = regions
    np.savez_compressed(args.output / 'responses.npz', **arrays)
    import csv
    with (args.output / 'profiles.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['case', 'row', 'region', 'y_A', 'fermi_eV', 'component',
                         'orbital', 'orbital_I', 'orbital_E', 'spin', 'spin_I', 'spin_E'])
        writer.writerows(records)
    with (args.output / 'drive_rows.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['case', 'driven_row', 'drive_region', 'fermi_eV', 'component', 'C_total_orbital', 'C_total_spin'])
        for case, response in row_drives.items():
            for row, region in enumerate(regions):
                for fi, mu in enumerate(args.fermi):
                    for component in range(3):
                        writer.writerow([case, row, region, mu, 'xyz'[component],
                            response['orbital'][row, fi, component], response['spin'][row, fi, component]])
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    plot_profiles(finest, regions, args.fermi, grids[-1], args.output)
    plot_drive_rows(row_drives, regions, args.fermi, grids[-1], args.output)
    print(json.dumps(checks, indent=2))
    if not all(checks.values()):
        raise RuntimeError('NC validation failed; inspect summary.json (custom parameters may have no signal)')


if __name__ == '__main__':
    main()
