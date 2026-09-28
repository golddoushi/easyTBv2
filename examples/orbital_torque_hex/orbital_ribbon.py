"""Honeycomb ribbon: row-resolved fixed-BSR magnetic-moment response to Ex."""
import argparse
import contextlib
import csv
import gc
import io
import json
from pathlib import Path
import sys
import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.orbital_torque_hex.orbital_hall_source import build_model
from calculator import (CollisionSpecification, OrbitalGalvanicConductivity_BlochSelfRotate_LG_DM,
    OrbitalGalvanicConductivity_BlochSelfRotate_VG_DM, OrbitalGalvanicConductivity_LG_DM,
    SpinGalvanicConductivity_LG_DM)


def build_ribbon(directory, width=8, hopping=1., delta=.5):
    """Periodic a1 along x, open a2 (finite y); 2*width atomic rows."""
    if width < 2:
        raise ValueError('width >= 2 required')
    model = build_model(directory, hopping=hopping, delta=delta)
    model.make_supercell(np.diag([1, width, 1]))
    model.cutEdge(1)
    assert np.all(model.iRvec[:, 1:] == 0)
    return model


def _calculate_mesh(model, grid, fermi, kbt=.01, linewidth=.02, batch_size=4, offset=0.):
    """Use the public calculators directly; symmetrized projected BSR moments.

    m_i={P_i,m_BSR}/2, NOT P_i m_BSR P_i: BSR is spatially nonlocal.
    Return all field/moment components and raw I/E channels, including zeros.
    """
    order = np.argsort(np.asarray(model.posinfo['pos ions (cart.)'])[:, 1])
    opts = dict(model=model, kgrid=(grid, 1, 1), kOffset=(offset, 0., 0.), fermiList=np.asarray(fermi), kBT=kbt,
        collisionSpecification=CollisionSpecification('crta', constantLinewidth=linewidth),
        fullCovariantDerivative=True, finiteTemperatureOccupations=True)
    result = dict(y_A=np.asarray(model.posinfo['pos ions (cart.)'])[order, 1])
    states = None
    for name, cls, projkey in [
        ('bsr', OrbitalGalvanicConductivity_BlochSelfRotate_LG_DM, 'orbitalProjectionList'),
        ('bsr_vg', OrbitalGalvanicConductivity_BlochSelfRotate_VG_DM, 'orbitalProjectionList'),
        ('aca', OrbitalGalvanicConductivity_LG_DM, 'orbitalProjectionList'),
        ('spin', SpinGalvanicConductivity_LG_DM, 'spinProjectionList')]:
        chunks = {}
        for start in range(0, len(order), batch_size):
            projections = [dict(source=[(int(a), (0,))], target=[(int(a), (0,))]) for a in order[start:start+batch_size]]
            calc = cls(states=states, **opts, **{projkey: projections})
            states = calc.states
            chunks.setdefault(name, []).append(calc.projected_sigma[0, 1:].copy())
            result[name+'_total'] = calc.sigma.copy()
            for channel, values in calc.projected_sigma_channels.items():
                chunks.setdefault(name+'_'+channel, []).append(values[0, 1:].copy())
            if name == 'bsr_vg':
                chunks.setdefault('vg_pole_residual', []).append(calc.projected_sum_rule_residual[0, 1:].copy())
                result['vg_pole_total'] = calc.sum_rule_residual.copy()
            # Release batch intermediates before preparing the next one.
            del calc
            gc.collect()
        result.update({key: np.concatenate(parts) for key, parts in chunks.items()})
    half = model.norbs//2
    result['band_edges'] = np.array([states.energies[..., half-1].max(), states.energies[..., half].min()])
    return result


def calculate_profile(model, grid, fermi, kbt=.01, linewidth=.02, batch_size=4, mesh_shift=0.):
    """Bound memory with equal interleaved submeshes (scalar CRTA only).

    The union of q/count + (shard+mesh_shift)/grid is the requested grid.
    mesh_shift is measured in mesh spacings; 0.5 avoids endpoint degeneracies.
    BSR vertices and these dc kernels are k-local; no finite-difference mesh
    derivative enters. Atomic-row observables and channels are averaged.
    """
    if grid < 3 or batch_size < 1:
        raise ValueError('grid >= 3 and batch_size >= 1 required')
    count = next(n for n in range(1, grid+1) if grid % n == 0 and grid//n <= 64)
    accumulated = None
    for shard in range(count):
        part = _calculate_mesh(model, grid//count, fermi, kbt, linewidth, batch_size, (shard+mesh_shift)/grid)
        if accumulated is None:
            accumulated = {key: value.copy() for key, value in part.items()}
        else:
            for key in part:
                if key == 'y_A':
                    continue
                if key == 'band_edges':
                    accumulated[key] = np.array([max(accumulated[key][0], part[key][0]), min(accumulated[key][1], part[key][1])])
                else:
                    accumulated[key] += part[key]
        del part
        gc.collect()
    for key in accumulated:
        if key not in ('y_A', 'band_edges'):
            accumulated[key] /= count
    return accumulated


def plot_profiles(arrays, summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = max(summary['parameters']['grids'])
    x = np.arange(len(arrays[f'y_A_n{n}']))
    for fi, mu in enumerate(arrays['fermi_eV']):
        fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True, constrained_layout=True)
        for grid in summary['parameters']['grids']:
            axes[0].plot(x, arrays[f'bsr_n{grid}'][:, fi, 0, 2], 'o-', ms=3, label=f'BSR LG Nk={grid}')
        axes[0].plot(x, arrays[f'bsr_vg_n{n}'][:, fi, 0, 2], 'x--', label='BSR VG finite part')
        axes[0].plot(x, arrays['narrower_bsr'][:, fi, 0, 2], '-.', label='BSR LG: half kBT and linewidth')
        for ch in ('I', 'E'):
            axes[1].plot(x, arrays[f'bsr_{ch}_n{n}'][:, fi, 0, 2], 'o-', ms=3, label=f'BSR {ch}')
        axes[2].plot(x, arrays[f'aca_n{n}'][:, fi, 0, 2], label='ACA z / Ex')
        axes[2].plot(x, np.max(np.abs(arrays[f'spin_n{n}'][:, fi]), axis=(1, 2)), '--', label='max |spin response|')
        axes[2].set_ylim(-1e-10, 1e-10)
        axes[2].text(.5, .85, 'Numerical zero (raw values)', transform=axes[2].transAxes, ha='center')
        for ax in axes:
            ax.axhline(0, color='0.5', lw=.6)
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
            ax.set_ylabel(r'$\delta\mu/E_x$ [$\mu_B$/(V/$\AA$)]')
        axes[0].set_title(f'Projected ribbon BSR moment: EF={mu:g} eV\n{len(x)} atomic rows; fixed-operator response')
        axes[2].set_xlabel('Observed atomic row (increasing y)')
        fig.savefig(Path(output)/f'profile_f{fi}.png', dpi=160)
        plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('tmp/orbital_torque_hex/ribbon'))
    p.add_argument('--width', type=int, default=8)
    p.add_argument('--grids', type=int, nargs='+', default=[256, 512])
    p.add_argument('--fermi', type=float, nargs='+', default=[0., .8])
    p.add_argument('--kbt', type=float, default=.01)
    p.add_argument('--linewidth', type=float, default=.02)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--mesh-shift', type=float, default=0., help='k offset in mesh spacings; 0.5 avoids endpoint degeneracies')
    p.add_argument('--plot-only', action='store_true')
    args = p.parse_args(argv)
    if args.plot_only:
        with np.load(args.output/'responses.npz') as arrays:
            plot_profiles(arrays, json.loads((args.output/'summary.json').read_text()), args.output)
        return
    if args.width < 2 or min(args.grids) < 3 or args.batch_size < 1 or min(args.kbt, args.linewidth) <= 0 or not np.all(np.isfinite([args.kbt, args.linewidth, args.mesh_shift, *args.fermi])):
        p.error('width >=2, grids >=3, batch size >=1, finite fillings and positive finite kBT/linewidth required')
    args.output.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        model = build_ribbon(args.output/'model', args.width)
    for name, system in [('model', model)]:
        np.savez_compressed(args.output/name/'ribbon_model.npz', ham_R=system.ham_R, iRvec=system.iRvec,
            lattice=system.lattice, rList=system.rList, orbital=system.orbital, spin=system.spin)
    grids = sorted(set(args.grids))
    arrays = dict(fermi_eV=np.array(args.fermi))
    metrics = []
    with threadpool_limits(limits=1):
        for grid in grids:
            response = calculate_profile(model, grid, args.fermi, args.kbt, args.linewidth, args.batch_size, args.mesh_shift)
            arrays.update({f'{key}_n{grid}': value for key, value in response.items()})
            metric = dict(grid=grid, max_bsr_Ex_z=float(np.max(np.abs(response['bsr'][:, :, 0, 2]))),
                max_spin=float(np.max(np.abs(response['spin']))), max_aca=float(np.max(np.abs(response['aca']))),
                lg_vg_difference=float(np.max(np.abs(response['bsr']-response['bsr_vg']))),
                projected_sum_residual=float(np.max(np.abs(response['bsr'].sum(axis=0)-response['bsr_total']))),
                vg_pole_residual_Ex=float(np.max(np.abs(response['vg_pole_residual'][..., 0, :]))),
                band_edges_eV=response['band_edges'].tolist())
            metrics.append(metric)
            print(json.dumps(metric), flush=True)
        narrower = calculate_profile(model, grids[-1], args.fermi, args.kbt/2, args.linewidth/2, args.batch_size, args.mesh_shift)
    arrays.update({f'narrower_{key}': value for key, value in narrower.items()})
    checks = dict(lg_vg_equal=all(m['lg_vg_difference'] < 1e-9 for m in metrics),
        projected_moments_sum=all(m['projected_sum_residual'] < 1e-9 for m in metrics),
        zero_aca_and_spin=all(max(m['max_aca'], m['max_spin']) < 1e-9 for m in metrics))
    summary = dict(parameters={k: v for k, v in vars(args).items() if k != 'output'}, checks=checks,
        metrics=metrics, axes=['observed_row', 'fermi', 'field', 'moment'], units='mu_B/(V/Angstrom) per x repeat',
        projection='symmetrized {P_row,m_BSR}/2; ribbon-derived nonlocal moment',
        vg_scope='BSR covariant contact included; inspect static pole residual under mesh refinement, independently of finite-part LG/VG equality',
        max_bsr_Ex_z_by_fermi=np.max(np.abs(response['bsr'][:, :, 0, 2]), axis=0).tolist(),
        narrower_max_bsr_Ex_z_by_fermi=np.max(np.abs(narrower['bsr'][:, :, 0, 2]), axis=0).tolist())
    if len(grids) > 1:
        summary['last_mesh_Ex_z_difference_by_fermi'] = np.max(np.abs(response['bsr'][:, :, 0, 2]-arrays[f'bsr_n{grids[-2]}'][:, :, 0, 2]), axis=0).tolist()
    np.savez_compressed(args.output/'responses.npz', **arrays)
    with (args.output/'profiles.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['row', 'y_A', 'fermi_eV', 'BSR_LG_Ex_z', 'BSR_I', 'BSR_E', 'BSR_VG', 'ACA', 'spin_z', 'narrower_BSR'])
        for row, y in enumerate(response['y_A']):
            for fi, mu in enumerate(args.fermi):
                writer.writerow([row, y, mu]+[response[key][row, fi, 0, 2] for key in ('bsr', 'bsr_I', 'bsr_E', 'bsr_vg', 'aca', 'spin')]+[narrower['bsr'][row, fi, 0, 2]])
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    plot_profiles(arrays, summary, args.output)
    print(json.dumps(summary, indent=2))
    if not all(checks.values()):
        raise RuntimeError('Ribbon consistency checks failed; inspect summary.json')


if __name__ == '__main__':
    main()
