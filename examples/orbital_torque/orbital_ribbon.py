"""Finite-y, periodic-x source ribbon: layer-resolved dc orbital accumulation.

python examples/orbital_torque/orbital_ribbon.py
Uses the existing local-moment LG density-matrix calculators, including their
intraband and interband channels. No SOC or magnetism is introduced.
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
from examples.orbital_torque.orbital_hall_source import build_model
from calculator import (CollisionSpecification, OrbitalGalvanicConductivity_LG_DM,
                        SpinGalvanicConductivity_LG_DM)


def build_ribbon(directory, width=12, tsp=0.7):
    """Repeat the verified bulk source in y, then delete y-boundary hoppings."""
    if width < 2:
        raise ValueError('width must be at least two atomic rows')
    model = build_model(directory, tsp=tsp)
    model.make_supercell(np.diag([1, width, 1]))
    model.cutEdge(1)
    assert np.all(model.iRvec[:, 1:] == 0)
    half = model.norbs // 2
    np.testing.assert_array_equal(model.ham_R[:, :half, :half], model.ham_R[:, half:, half:])
    np.testing.assert_array_equal(model.ham_R[:, :half, half:], 0)
    np.testing.assert_array_equal(model.ham_R[:, half:, :half], 0)
    lookup = {tuple(r): i for i, r in enumerate(model.iRvec)}
    for i, r in enumerate(model.iRvec):
        np.testing.assert_allclose(model.ham_R[i], model.ham_R[lookup[tuple(-r)]].conj().T, atol=1e-14)
    return model


def calculate_profile(model, grid, fermi, kbt=0.05, linewidth=0.05, batch_size=2):
    """Small batches limit projector memory; all batches reuse the same band states.

    Returned local axes: layer (in increasing y), Fermi, field, moment.
    Units: mu_B/(V/Angstrom) per row per longitudinal repeat (a=3 Angstrom).
    """
    positions = np.asarray(model.posinfo['pos ions (cart.)'])
    order = np.argsort(positions[:, 1])
    options = dict(model=model, kgrid=(grid, 1, 1), fermiList=np.asarray(fermi), kBT=kbt,
                   collisionSpecification=CollisionSpecification('crta', constantLinewidth=linewidth),
                   fullCovariantDerivative=True, finiteTemperatureOccupations=True)
    result = dict(y_A=positions[order, 1])
    states = None
    for name, cls, key in [('orbital', OrbitalGalvanicConductivity_LG_DM, 'orbitalProjectionList'),
                           ('spin', SpinGalvanicConductivity_LG_DM, 'spinProjectionList')]:
        chunks = {}
        for start in range(0, len(order), batch_size):
            projections = []
            for atom in order[start:start+batch_size]:
                # P_i O + O P_i over two: identical source/target projectors.
                selection = [(int(atom), (0, 1))]
                projections.append(dict(source=selection, target=selection))
            calc = cls(states=states, **{key: projections}, **options)
            states = calc.states
            chunks.setdefault(name, []).append(calc.projected_sigma[0, 1:].copy())
            result[f'{name}_total'] = calc.sigma.copy()
            for channel, values in calc.projected_sigma_channels.items():
                chunks.setdefault(f'{name}_{channel}', []).append(values[0, 1:].copy())
            del calc
        result.update({key: np.concatenate(value, axis=0) for key, value in chunks.items()})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('tmp/orbital_torque/ribbon'))
    parser.add_argument('--width', type=int, default=12)
    parser.add_argument('--grids', type=int, nargs='+', default=[64, 128])
    parser.add_argument('--fermi', type=float, nargs='+', default=[0.8])
    parser.add_argument('--kbt', type=float, default=0.05)
    parser.add_argument('--linewidth', type=float, default=0.05)
    args = parser.parse_args(argv)
    if args.width < 2 or min(args.grids) < 3:
        parser.error('width >= 2 and grids >= 3 required')
    if not np.all(np.isfinite([args.kbt, args.linewidth, *args.fermi])) or min(args.kbt, args.linewidth) <= 0:
        parser.error('finite Fermi energies and positive finite kbt/linewidth required')
    args.output.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        model = build_ribbon(args.output / 'primitive_source', args.width)
        control = build_ribbon(args.output / 'primitive_control', args.width, tsp=0)
    # Save the actual cut ribbon, not only the primitive files used to build it.
    np.savez_compressed(args.output / 'ribbon_model.npz', ham_R=model.ham_R,
                        iRvec=model.iRvec, lattice=model.lattice, rList=model.rList,
                        spin=model.spin, orbital=model.orbital)
    grids = sorted(set(args.grids))
    arrays = dict(fermi_eV=np.asarray(args.fermi), grids=np.array(grids))
    rows = []
    with threadpool_limits(limits=1):
        for grid in grids:
            profile = calculate_profile(model, grid, args.fermi, args.kbt, args.linewidth)
            arrays.update({f'{key}_n{grid}': value for key, value in profile.items()})
            chi = profile['orbital'][:, :, 0, 2]
            row = dict(grid=grid, max_abs_orbital_Ex_z=float(np.max(np.abs(chi))),
                       max_abs_spin=float(np.max(np.abs(profile['spin']))),
                       mirror_odd_residual=float(np.max(np.abs(chi + chi[::-1]))),
                       projection_sum_residual=float(np.max(np.abs(profile['orbital'].sum(axis=0)-profile['orbital_total']))))
            rows.append(row)
            print(json.dumps(row), flush=True)
        baseline = calculate_profile(control, grids[-1], args.fermi, args.kbt, args.linewidth)
    arrays.update({f'control_{key}': value for key, value in baseline.items()})
    tolerance = 1e-9
    checks = dict(finite_orbital_response=bool(np.max(np.abs(chi)) > 1e-5),
                  zero_spin_response=all(row['max_abs_spin'] < tolerance for row in rows),
                  mirror_odd_profile=all(row['mirror_odd_residual'] < tolerance for row in rows),
                  projected_sum_equals_total=all(row['projection_sum_residual'] < tolerance for row in rows),
                  zero_net_orbital_Ex_z=bool(np.max(np.abs(profile['orbital_total'][:, 0, 2])) < tolerance),
                  no_mixing_zero_orbital=bool(np.max(np.abs(baseline['orbital'])) < tolerance),
                  no_mixing_zero_spin=bool(np.max(np.abs(baseline['spin'])) < tolerance))
    summary = dict(width=args.width, bands=model.norbs, periodic_direction='x', open_direction='y',
                   fermi_eV=args.fermi, kBT_eV=args.kbt, linewidth_eV=args.linewidth,
                   units='mu_B/(V/Angstrom) per atomic row per longitudinal unit cell',
                   local_tensor_axes=['layer_in_increasing_y', 'fermi', 'field', 'moment'],
                   zero_absolute_tolerance=tolerance, checks=checks, mesh_results=rows,
                   lower_edge_Ex_z=chi[0].tolist(), upper_edge_Ex_z=chi[-1].tolist(),
                   control_max_abs_orbital=float(np.max(np.abs(baseline['orbital']))))
    if len(grids) > 1:
        previous = arrays[f'orbital_n{grids[-2]}'][:, :, 0, 2]
        summary['last_mesh_profile_difference_over_peak'] = float(np.max(np.abs(chi-previous))/max(np.max(np.abs(chi)), 1e-30))
    np.savez_compressed(args.output / 'responses.npz', **arrays)
    records = []
    for fi, fermi in enumerate(args.fermi):
        for layer, y in enumerate(profile['y_A']):
            records.append([layer, y, fermi, chi[layer, fi], profile['orbital_I'][layer, fi, 0, 2],
                            profile['orbital_E'][layer, fi, 0, 2],
                            np.max(np.abs(profile['spin'][layer, fi])), baseline['orbital'][layer, fi, 0, 2]])
    np.savetxt(args.output / 'profile.csv', records, delimiter=',',
               header='layer,y_A,fermi_eV,orbital_Ex_z,orbital_I_Ex_z,orbital_E_Ex_z,max_abs_spin,control_orbital_Ex_z', comments='')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    plot_profile(profile, baseline, args.fermi, grids[-1], args.output)
    print(json.dumps(summary, indent=2))
    if not all(checks.values()):
        raise RuntimeError('Ribbon validation failed; inspect summary.json')


def plot_profile(profile, baseline, fermi_list, grid, output):
    """Show raw control values on the same axes; never round them to zero."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(7, 6), sharex=True, constrained_layout=True)
    width = profile['orbital'].shape[0]
    layers = np.arange(width)
    for fi, fermi in enumerate(fermi_list):
        line, = ax[0].plot(layers, profile['orbital'][:, fi, 0, 2], 'o-', label=f'Mixing on, EF={fermi:g} eV')
        ax[0].plot(layers, baseline['orbital'][:, fi, 0, 2], 's--', color=line.get_color(),
                   markerfacecolor='white', label=f'Mixing off (control), EF={fermi:g} eV')
        ax[1].plot(layers, np.max(np.abs(profile['spin'][:, fi]), axis=(1, 2)), 'o-',
                   color=line.get_color(), label=f'Mixing on, EF={fermi:g} eV')
        ax[1].plot(layers, np.max(np.abs(baseline['spin'][:, fi]), axis=(1, 2)), 's--',
                   color=line.get_color(), markerfacecolor='white', label=f'Mixing off, EF={fermi:g} eV')
    ax[0].set_ylabel(r'$\delta\mu_{L,z}/E_x$ [$\mu_B$ / (V/$\AA$)]')
    ax[0].set_title(f'Orbital accumulation: {width} rows, Nk={grid}')
    ax[0].legend()
    ax[1].set_ylabel(r'Max spin response [$\mu_B$ / (V/$\AA$)]')
    ax[1].set_xlabel('Atomic row (increasing y)')
    ax[1].legend(fontsize=8)
    ax[1].set_title('Spin response: numerical roundoff', fontsize=10)
    for axis in ax:
        axis.grid(alpha=0.25)
    fig.savefig(Path(output) / 'profile.png', dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
