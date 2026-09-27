"""Step 4: N source / C SOC converter / F exchange magnet, driven along x.

python examples/orbital_torque/orbital_ncf_ribbon.py --workers 8
"""
import argparse
import contextlib
import csv
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.orbital_torque.orbital_nc_ribbon import build_nc_ribbon
from examples.orbital_torque.orbital_ribbon import effective_workers
from calculator import CollisionSpecification
from calculator.response_stages import DensityMatrixResponse
from calculator.density_matrix import LengthGaugeFirstOrderDMKernel


CASES = {
    'full': ('o-', 'Full N/C/F', {}),
    'no_mixing': ('s--', 'N mixing off', {'tsp': 0.}),
    'no_soc': ('^:', 'C SOC off', {'soc': 0.}),
    'no_exchange': ('v--', 'F exchange off', {'exchange': 0.}),
    #'nc_off': ('D-.', 'N-C link off', {'nc_hopping': 0.}),
    #'cf_off': ('x:', 'C-F link off', {'cf_hopping': 0.}),
}
NAMES = ('orbital', 'spin', 'torque')


def build_ncf_ribbon(directory, n_width=8, c_width=4, f_width=4, tsp=.7,
                     soc=.4, exchange=.6, magnetization=(0., 1., 0.),
                     converter_hopping=.5, converter_shift=0., f_hopping=.5,
                     f_shift=0., nc_hopping=.3, cf_hopping=.3):
    """H_XC=+Delta/2 m.sigma: lower electronic magnetic moment aligns with m.

    F has complete s+p shells, orbital-independent hopping, no SOC or s-p
    mixing. SOC is confined to C; explicit exchange_R is confined to F.
    """
    m = np.asarray(magnetization, dtype=float)
    if m.shape != (3,) or not np.all(np.isfinite(m)) or np.linalg.norm(m) == 0:
        raise ValueError('magnetization must be a finite nonzero three-vector')
    m = m/np.linalg.norm(m)
    if min(n_width, c_width, f_width) < 1:
        raise ValueError('All three region widths must be positive')
    if not np.all(np.isfinite([exchange, f_hopping, f_shift, cf_hopping])):
        raise ValueError('F and interface parameters must be finite')
    model, _, soc_R = build_nc_ribbon(directory, n_width, c_width+f_width,
        tsp, soc, converter_hopping, nc_hopping, converter_shift)
    order = np.argsort(np.asarray(model.posinfo['pos ions (cart.)'])[:, 1])
    ids = [np.flatnonzero(model.iorb2iatomMap == atom) for atom in order]
    start = n_width+c_width
    ir0 = next(i for i, r in enumerate(model.iRvec) if np.all(r == 0))
    # Remove C's SOC and replace the scalar C-like blocks in F.
    for a in range(start, len(order)):
        local = ids[a]
        soc_R[ir0][np.ix_(local, local)] = 0.
        model.ham_R[ir0][np.ix_(local, local)] = np.diag(np.tile([-2.+f_shift, f_shift, f_shift, f_shift], 2))
        for ir, r in enumerate(model.iRvec):
            if abs(r[0]) == 1:
                model.ham_R[ir][np.ix_(local, local)] = -f_hopping*np.eye(8)
        if a > start:
            model.ham_R[ir0][np.ix_(ids[a-1], local)] = -f_hopping*np.eye(8)
            model.ham_R[ir0][np.ix_(local, ids[a-1])] = -f_hopping*np.eye(8)
    left, right = ids[start-1], ids[start]
    model.ham_R[ir0][np.ix_(left, right)] = -cf_hopping*np.eye(8)
    model.ham_R[ir0][np.ix_(right, left)] = -cf_hopping*np.eye(8)
    exchange_R = np.zeros_like(model.ham_R)
    for local in ids[start:]:
        pauli = model.spin[:, local][:, :, local]
        exchange_R[ir0][np.ix_(local, local)] = exchange/2*np.einsum('a,aij->ij', m, pauli)
    model.ham_R += exchange_R
    model.exchange_R = exchange_R
    model._refreshHamiltonianDerivedMatrices()
    regions = np.array(['N']*n_width+['C']*c_width+['F']*f_width)
    return model, regions, soc_R, m


def moment_operators(model):
    """Same energy-like exchange vertex as SpinExchangeTorque: -i[mu_S,H_XC]."""
    ir0 = next(i for i, r in enumerate(model.iRvec) if np.all(r == 0))
    exchange = model.exchange_R[ir0]
    spin = -model.spin
    return dict(orbital=-model.orbital, spin=spin,
                torque=-1j*(spin@exchange-exchange@spin))


def calculate_response(model, regions, grid, fermi, kbt=.05, linewidth=.05,
                       workers=1, row_drives=True):
    """LG scalar CRTA: local Ex profiles and F-total response per driven row.

    Profiles retain I/E channels. All xyz components are kept, including for
    tilted magnets. Row drives use P_i v_x P_i and are additive only when
    v_x is block diagonal between rows, checked explicitly below.
    """
    if grid < 3 or workers < 1 or kbt <= 0 or linewidth <= 0:
        raise ValueError('grid >= 3, workers >= 1, positive kBT/linewidth required')
    if model.overlap or not np.all(model.iRvec[:, 1:] == 0):
        raise ValueError('An orthogonal x-periodic ribbon is required')
    order = np.argsort(np.asarray(model.posinfo['pos ions (cart.)'])[:, 1])
    atom_map = model.iorb2iatomMap
    ids = [np.flatnonzero(atom_map == atom) for atom in order]
    operators = moment_operators(model)
    fmask = np.isin(atom_map, order[np.asarray(regions) == 'F'])
    count = effective_workers(grid, workers)

    def shard(index):
        prepared = DensityMatrixResponse(model=model, kgrid=(grid//count, 1, 1),
            kOffset=(index/grid, 0., 0.), fermiList=np.asarray(fermi), kBT=kbt,
            collisionSpecification=CollisionSpecification('crta', constantLinewidth=linewidth))
        kernel = LengthGaugeFirstOrderDMKernel(prepared.inputs, full_covariant_derivative=True,
            finite_temperature_occupations=True, collision=prepared.collision)
        u = prepared.states.vectors
        ud = u.conj().swapaxes(-1, -2)
        result = {}
        for name in NAMES:
            for channel in ('I', 'E'):
                result[f'{name}_{channel}'] = np.zeros((len(order), len(fermi), 3))
            result[f'{name}_total'] = np.zeros((len(fermi), 3))
            if row_drives:
                result[f'drive_{name}'] = np.zeros((len(order), len(fermi), 3))
        if row_drives:
            vx = u@prepared.inputs.velocity_K[..., 0, :, :]@ud
            interrow = atom_map[:, None] != atom_map[None, :]
            if np.max(np.abs(vx[..., interrow])) > 1e-10:
                raise ValueError('Row-drive additivity requires no inter-row v_x')
        for fi, mu in enumerate(fermi):
            state = kernel.fermi_state(fi, mu)
            block = state.at_frequency(0.).block(0)
            for channel, rho in block.rho1.items():
                density = (u@rho@ud).mean(axis=(0, 1, 2))
                for name, op in operators.items():
                    result[f'{name}_total'][fi] += np.einsum('ij,cji->c', density, op).real
                    for row, local in enumerate(ids):
                        result[f'{name}_{channel}'][row, fi] = np.einsum('ij,cji->c',
                            density[np.ix_(local, local)], op[:, local][:, :, local]).real
            if row_drives:
                weight = state.collision.solve(-1j*state.divided_difference, prepared.inputs.deltaE, 0.)
                for name, op in operators.items():
                    for component in range(3):
                        observable = ud@(op[component]*fmask[:, None]*fmask[None, :])@u
                        dual = u@(weight.swapaxes(-1, -2)*observable)@ud
                        for row, local in enumerate(ids):
                            result[f'drive_{name}'][row, fi, component] = np.einsum('...ij,...ji->...',
                                vx[..., local[:, None], local], dual[..., local[:, None], local]).real.mean()
        for name in NAMES:
            result[name] = result[f'{name}_I']+result[f'{name}_E']
        return result

    with threadpool_limits(limits=1):
        if count == 1:
            return shard(0)
        with ThreadPoolExecutor(max_workers=count) as pool:
            parts = list(pool.map(shard, range(count)))
        return {key: sum(part[key] for part in parts)/count for key in parts[0]}


def plot_results(results, regions, fermi, grid, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = np.arange(len(regions))
    n, c = np.sum(regions == 'N'), np.sum(regions == 'C')
    for fi, mu in enumerate(fermi):
        for drive in (False, True):
            for all_components in (False, True):
                fig, axes = plt.subplots(3, 3 if all_components else 1,
                    figsize=(15, 10) if all_components else (9, 10), squeeze=False,
                    sharex=True, constrained_layout=True)
                for row, name in enumerate(NAMES):
                    components = range(3) if all_components else [0 if name == 'torque' else 2]
                    for col, component in enumerate(components):
                        ax = axes[row, col]
                        key = f'drive_{name}' if drive else name
                        for case, response in results.items():
                            fmt, label, _ = CASES[case]
                            ax.plot(x, response[key][:, fi, component], fmt, ms=3, label=label)
                        for lo, hi, color in ((-.5, n-.5, 'steelblue'), (n-.5, n+c-.5, 'orange'),
                                              (n+c-.5, len(x)-.5, 'green')):
                            ax.axvspan(lo, hi, color=color, alpha=.08)
                        for boundary in (n-.5, n+c-.5):
                            ax.axvline(boundary, color='0.4', ls='--', lw=.7)
                        ax.axhline(0, color='0.5', lw=.6)
                        ax.grid(alpha=.2)
                        unit = r'$\mu_B$ eV/(V/$\AA$)' if name == 'torque' else r'$\mu_B$/(V/$\AA$)'
                        ax.set_ylabel(f'{"F-total " if drive else ""}{name} {"xyz"[component]} / Ex\n[{unit}]')
                        if max(np.max(np.abs(p[key][:, fi, component])) for p in results.values()) < 1e-9:
                            ax.set_ylim(-1e-9, 1e-9)
                            ax.text(.5, .85, 'Numerical zero', transform=ax.transAxes, ha='center')
                        if row == 2:
                            ax.set_xlabel(('Driven' if drive else 'Observed')+' row (N blue / C orange / F green)')
                axes[0, -1].legend(fontsize=8, ncol=1 if all_components else 2)
                title = 'F-total response to each driven row' if drive else 'Uniform Ex: local induced response'
                fig.suptitle(f'{title}; EF={mu:g} eV, Nk={grid}')
                filename = ('all_components_' if all_components else '')+('drive_rows' if drive else 'profile')
                fig.savefig(Path(output)/f'{filename}_f{fi}.png', dpi=160)
                plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('tmp/orbital_torque/ncf_ribbon'))
    for name, default in [('n-width', 8), ('c-width', 4), ('f-width', 4), ('workers', 8)]:
        parser.add_argument('--'+name, type=int, default=default)
    for name, default in [('tsp', .7), ('soc', .4), ('exchange', .6), ('converter-hopping', .5),
                          ('converter-shift', 0.), ('f-hopping', .5), ('f-shift', 0.),
                          ('nc-hopping', .3), ('cf-hopping', .3), ('kbt', .05), ('linewidth', .05)]:
        parser.add_argument('--'+name, type=float, default=default)
    parser.add_argument('--magnetization', type=float, nargs=3, default=[0., 1., 0.], metavar=('MX', 'MY', 'MZ'))
    parser.add_argument('--fermi', type=float, nargs='+', default=[.8])
    parser.add_argument('--grids', type=int, nargs='+', default=[256, 512])
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args(argv)
    if args.plot_only:
        summary = json.loads((args.output/'summary.json').read_text())
        grid = max(summary['parameters']['grids'])
        with np.load(args.output/'responses.npz') as data:
            results = {case: {key: data[f'{case}_{key}_n{grid}'] for name in NAMES
                       for key in (name, f'drive_{name}')} for case in CASES}
            plot_results(results, data['regions'], data['fermi_eV'], grid, args.output)
        return
    if min(args.n_width, args.c_width, args.f_width, args.workers) < 1 or min(args.grids) < 3:
        parser.error('Positive widths/workers and grids >= 3 required')
    numeric = [v for v in vars(args).values() if isinstance(v, (float, int))]
    if not np.all(np.isfinite(numeric+args.fermi+args.magnetization)) or min(args.kbt, args.linewidth) <= 0 or np.linalg.norm(args.magnetization) == 0:
        parser.error('Finite parameters, positive kBT/linewidth and nonzero magnetization required')
    args.output.mkdir(parents=True, exist_ok=True)
    grids = sorted(set(args.grids))
    builder_keys = ('n_width', 'c_width', 'f_width', 'tsp', 'soc', 'exchange', 'magnetization',
                    'converter_hopping', 'converter_shift', 'f_hopping', 'f_shift', 'nc_hopping', 'cf_hopping')
    parameters = {key: getattr(args, key) for key in builder_keys}
    arrays = dict(fermi_eV=np.asarray(args.fermi), grids=np.array(grids))
    results, metrics = {}, {}
    for case, (_, _, changes) in CASES.items():
        with contextlib.redirect_stdout(io.StringIO()):
            model, regions, soc_R, m = build_ncf_ribbon(args.output/case/'primitive', **(parameters|changes))
        np.savez_compressed(args.output/case/'model.npz', ham_R=model.ham_R, soc_R=soc_R,
            exchange_R=model.exchange_R, iRvec=model.iRvec, lattice=model.lattice, rList=model.rList,
            spin=model.spin, orbital=model.orbital, regions=regions, magnetization=m)
        metrics[case] = []
        previous = None
        for grid in grids:
            start = perf_counter()
            response = calculate_response(model, regions, grid, args.fermi, args.kbt, args.linewidth,
                                          args.workers, row_drives=grid == grids[-1])
            arrays.update({f'{case}_{key}_n{grid}': value for key, value in response.items()})
            metric = dict(grid=grid, seconds=perf_counter()-start)
            for name in NAMES:
                metric[f'{name}_max_abs'] = float(np.max(np.abs(response[name])))
                metric[f'{name}_sum_residual'] = float(np.max(np.abs(response[name].sum(axis=0)-response[f'{name}_total'])))
                if previous is not None:
                    metric[f'{name}_mesh_difference'] = float(np.max(np.abs(response[name]-previous[name])))
                if grid == grids[-1]:
                    metric[f'{name}_drive_sum_residual'] = float(np.max(np.abs(
                        response[f'drive_{name}'].sum(axis=0)-response[name][regions == 'F'].sum(axis=0))))
            metrics[case].append(metric)
            previous = response
            print(json.dumps(dict(case=case, **metric)), flush=True)
        results[case] = response
    tol = 1e-9
    zero = lambda case, name: bool(np.max(np.abs(results[case][name])) < tol)
    checks = dict(
        observation_and_drive_sums=all(value < tol for rows in metrics.values() for row in rows
            for key, value in row.items() if key.endswith('residual')),
        no_exchange_zero_torque=zero('no_exchange', 'torque'),
        no_soc_zero_torque=zero('no_soc', 'torque'),
        no_mixing_zero_response=all(zero('no_mixing', name) for name in NAMES),
        #nc_off_zero_F_response=all(np.max(np.abs(results['nc_off'][name][regions == 'F'])) < tol for name in NAMES),
        #cf_off_zero_F_response=all(np.max(np.abs(results['cf_off'][name][regions == 'F'])) < tol for name in NAMES),
        torque_only_in_F=all(np.max(np.abs(result['torque'][regions != 'F'])) < tol for result in results.values()),
        torque_transverse=all(np.max(np.abs(np.einsum('rfa,a->rf', result['torque'], m))) < tol for result in results.values()))
    
    checks = {key: bool(value) for key, value in checks.items()}
    summary = dict(parameters={k: v for k, v in vars(args).items() if k != 'output'},
        magnetization_unit=m.tolist(), tensor_axes=['observed_row', 'fermi', 'moment'],
        drive_axes=['driven_row', 'fermi', 'moment'], drive_target='sum over F', field='Ex',
        moment_units='mu_B/(V/Angstrom) per x repeat', torque_units='mu_B*eV/(V/Angstrom) per x repeat',
        torque_definition='-i[mu_S,H_XC], mu_S=-sigma; matches SpinExchangeTorque; not the LLG magnetic-moment rate',
        checks=checks, zero_tolerance=tol, metrics=metrics,
        F_totals_by_drive_region={case: {region: {name: result[f'drive_{name}'][regions == region].sum(axis=0).tolist()
            for name in NAMES} for region in ('N', 'C', 'F')} for case, result in results.items()},
        full_F_torque=results['full']['torque'][regions == 'F'].sum(axis=0).tolist())
    arrays['regions'] = regions
    np.savez_compressed(args.output/'responses.npz', **arrays)
    for drive in (False, True):
        with (args.output/('drive_rows.csv' if drive else 'profiles.csv')).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            keys = [f'drive_{name}' for name in NAMES] if drive else [key for name in NAMES for key in (name, name+'_I', name+'_E')]
            writer.writerow(['case', 'row', 'region', 'fermi_eV', 'component']+keys)
            for case, response in results.items():
                for row, region in enumerate(regions):
                    for fi, mu in enumerate(args.fermi):
                        for component in range(3):
                            writer.writerow([case, row, region, mu, 'xyz'[component]]+
                                            [response[key][row, fi, component] for key in keys])
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    plot_results(results, regions, args.fermi, grids[-1], args.output)
    print(json.dumps(dict(checks=checks, full_F_torque=summary['full_F_torque']), indent=2))
    if not all(checks.values()):
        raise RuntimeError('NCF consistency checks failed; inspect summary.json')


if __name__ == '__main__':
    main()
