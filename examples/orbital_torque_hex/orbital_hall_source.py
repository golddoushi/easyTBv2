"""Spinful honeycomb/Haldane: BSR V1, V2 and finite-PBC OHC; ACA and SHC.

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
                        OrbitalHallConductance_BlochSelfRotateV2_LG_DM,
                        OrbitalHallConductance_LG_DM, SpinHallConductance_LG_DM)

from calculator.electronic_stages import Orbital_BlochSelfRotate, _OrbitalCurrentVertex
from calculator.Grid import _OrbitalHallConductance_LG_DM
from scipy.constants import hbar, electron_mass, elementary_charge, angstrom


class FinitePBCMoment(Orbital_BlochSelfRotate):
    """Eqs. (20),(21),(1),(2) on a finite periodic honeycomb lattice.

    Translation symmetry block-diagonalizes an Nx*Ny-cell torus into the
    discrete momenta on kgrid. Sum real-space bond velocities explicitly,
    then reconstruct off-diagonal r and form r cross v - v cross r. This
    avoids dense torus matrices without changing their momentum blocks.
    Bond displacements are unwrapped across the periodic seam: a diagonal
    sawtooth position operator on a torus is NOT used. As in the paper,
    diagonal position terms are removed. The inverse-gap prescription is
    shared physics, so agreement tests algebra, not that prescription.
    """
    def build_bloch_moment(self):
        if self.model.overlap or self.skip_low:
            raise ValueError('Finite PBC reference requires complete orthogonal s basis')
        if np.any(self.kOffset):
            raise ValueError('Finite PBC reference requires the unshifted torus mesh')
        fractional = np.stack(np.meshgrid(*[np.arange(n)/n for n in self.kgrid],
                                         indexing='ij'), axis=-1)
        v_ao = np.zeros_like(self.velocity_K)
        centers = self.model.rList
        for r, hopping in zip(self.model.iRvec, self.model.ham_R):
            displacement = (r @ self.model.lattice)[:, None, None] + centers[:, None, :] - centers[:, :, None]
            phase = np.exp(2j*np.pi*np.einsum('...a,a->...', fractional, r))
            v_ao += phase[..., None, None, None]*(1j*displacement*hopping)
        u = self.vectors[..., None, :, :]
        v = u.conj().swapaxes(-1, -2) @ v_ao @ u
        np.testing.assert_allclose(v, self.velocity_K, atol=2e-12)
        gap = self.energies[..., :, None] - self.energies[..., None, :]
        inverse = np.zeros_like(gap)
        np.divide(1., gap, out=inverse, where=np.abs(gap) > self.degen)
        r = -1j*v*inverse[..., None, :, :]
        constant = hbar**2/(2*electron_mass*elementary_charge*angstrom**2)
        moment = np.zeros_like(v)
        for c, (a, b) in enumerate(((1, 2), (2, 0), (0, 1))):
            ra, rb, va, vb = r[..., a, :, :], r[..., b, :, :], v[..., a, :, :], v[..., b, :, :]
            moment[..., c, :, :] = -(ra @ vb-rb @ va-va @ rb+vb @ ra)/(4*constant)
        return moment

    def bloch_self_rotation_contact(self):
        raise NotImplementedError('Finite PBC reference is an LG-only benchmark')


class FinitePBCCurrent(_OrbitalCurrentVertex, FinitePBCMoment):
    pass


# Register this example-only stage with the explicit tensor publication API.
from calculator.stage_fields import ELECTRONIC_STAGE_FIELDS
ELECTRONIC_STAGE_FIELDS["FinitePBCCurrent"] = ELECTRONIC_STAGE_FIELDS["OrbitalVelocity_BlochSelfRotateV2"]


class FinitePBCHall(_OrbitalHallConductance_LG_DM):
    orbital_velocity_stage = FinitePBCCurrent


def build_model(directory, hopping=1., delta=.5, lattice_constant=2.4, rotation=0., t2=0., phase=0.):
    """BNcol geometry and hoppings, duplicated as (A up,B up,A down,B down).

    H_AB=-t*(1+exp(i k.a1)+exp(-i k.a2)); onsite +/-Delta.
    NNN hoppings t2*exp(+/-i*phase) follow opposite A/B triangular loops.
    phase is in radians. NN sign differs from Eq.19 by a sublattice gauge.
    All sites host only s orbitals. Energies eV, lengths Angstrom.
    """
    if not np.all(np.isfinite([hopping, delta, lattice_constant, rotation, t2, phase])) or lattice_constant <= 0:
        raise ValueError('Finite parameters and positive lattice constant required')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    a = lattice_constant
    angle = np.deg2rad(rotation)
    rotate = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    lattice = np.array([[a, 0, 0], [-a/2, a*np.sqrt(3)/2, 0], [0, 0, 15]]) @ rotate.T
    (directory/'POSCAR').write_text(
        'Spinful s-only honeycomb\n1.0\n'+''.join(' '.join(f'{v:.16g}' for v in row)+'\n' for row in lattice)+
        'A B\n1 1\nDirect\n'
        '0.6666666666666667 0.3333333333333333 0.5\n'
        '0.3333333333333333 0.6666666666666667 0.5\n')
    blocks = {(0, 0, 0): np.array([[delta, -hopping], [-hopping, -delta]], complex)}
    for r in ((1, 0, 0), (0, -1, 0)):
        blocks[r] = np.array([[0, -hopping], [0, 0]], complex)
        blocks[tuple(-np.asarray(r))] = blocks[r].conj().T
    # Oriented triangular loops, opposite chirality on A and B. Same phase
    # for both spin copies: Haldane (broken T), NOT Kane-Mele SOC.
    for r in ((1, 0, 0), (0, 1, 0), (-1, -1, 0)):
        term = np.diag([t2*np.exp(1j*phase), t2*np.exp(-1j*phase)])
        for key, value in ((r, term), (tuple(-np.asarray(r)), term.conj().T)):
            blocks[key] = blocks.get(key, np.zeros((2, 2), complex)) + value
    rvec = np.array(list(blocks))
    ham = np.zeros((len(rvec), 4, 4), complex)
    ham[:, :2, :2] = ham[:, 2:, 2:] = np.array(list(blocks.values()))
    with (directory/'source_hr.dat').open('w') as stream:
        stream.write(f'Spin-degenerate Haldane honeycomb; eV\n4\n{len(rvec)}\n'+ ' '.join(['1']*len(rvec))+'\n')
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
                      ('bsr_v2', OrbitalHallConductance_BlochSelfRotateV2_LG_DM),
                      ('finite_pbc', FinitePBCHall),
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
    for name, label in [('bsr_v2', 'BSR V2'), ('finite_pbc', 'PBC: Eq.21')]:
        axes[0].plot(fermi, arrays[f'{name}_n{grid}'][:, 0, 1, 2],
                     'o' if name == 'finite_pbc' else '--', markersize=3,
                     markevery=max(1,len(fermi)//20), fillstyle='none', label=label)
    for channel in ('I', 'E'):
        for name in ('bsr', 'bsr_v2'):
            axes[1].plot(fermi, arrays[f'{name}_{channel}_n{grid}'][:, 0, 1, 2], label=f'{name} {channel}')
    axes[2].plot(fermi, arrays[f'shc_n{grid}'][:, 0, 1, 2], label='SHC: Ex, Jy, Sz')
    axes[2].plot(fermi, np.max(np.abs(arrays[f'shc_n{grid}']), axis=(1, 2, 3)), '--', label='max |SHC| over all components')
    axes[2].set_ylim(-1e-10, 1e-10)
    axes[2].text(.5, .85, 'Numerical zero; raw values retained', transform=axes[2].transAxes, ha='center')
    for ax in axes:
        lo, hi = summary['mesh_results'][-1]['band_edges_eV']
        if lo < hi:
            ax.axvspan(lo, hi, color='grey', alpha=.12)
        ax.axhline(0, color='0.5', lw=.6)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
        ax.set_ylabel('Hall response\n(native normalization)')
    axes[0].set_title('s-only honeycomb: Ex-induced Jy of z magnetic moment\nShaded: bulk band gap')
    axes[2].set_xlabel('Fermi energy (eV)')
    fig.savefig(Path(output)/'hall_scan.png', dpi=160)
    plt.close(fig)


def plot_haldane_controls(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, phase_pi in zip(axes.flat, (0., .15, .25, .5)):
        folder = output/f'haldane_phi_{phase_pi:g}pi'
        with np.load(folder/'responses.npz') as arrays:
            grid = int(arrays['grids'][-1])
            fermi = arrays['fermi_eV']
            for key, label, style in [('bsr','BSR V1','-'), ('bsr_v2','BSR V2','--'),
                                      ('finite_pbc','PBC: Eq.21','o')]:
                ax.plot(fermi, arrays[f'{key}_n{grid}'][:,0,1,2], style,
                        label=label, markersize=3, fillstyle='none',
                        markevery=max(1,len(fermi)//20))
            lo, hi = arrays[f'band_edges_n{grid}']
            if lo < hi:
                ax.axvspan(lo, hi, color='grey', alpha=.12)
        ax.set_title(f'phase = {phase_pi:g} pi; {grid} x {grid} cells')
        ax.set_xlabel('Fermi energy (eV)')
        ax.set_ylabel('OHC (native normalization)')
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle('Haldane controls: M=0.4 eV, t2=0.15 eV; two identical spin copies')
    fig.savefig(output/'haldane_controls.png', dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('tmp/orbital_torque_hex/source'))
    parser.add_argument('--grids', type=int, nargs='+', default=[48, 96, 192])
    parser.add_argument('--fermi', type=float, nargs='+', default=np.linspace(-3.2, 3.2, 81).tolist())
    for key, default in [('hopping', 1.), ('delta', .5), ('lattice-constant', 2.4), ('kbt', .01), ('linewidth', .02)]:
        parser.add_argument('--'+key, type=float, default=default)
    parser.add_argument('--t2', type=float, default=0., help='NNN hopping amplitude in eV')
    parser.add_argument('--phase', type=float, default=0., help='NNN phase in radians')
    parser.add_argument('--haldane-controls', action='store_true', help='Also run M=.4, t2=.15 and phase/pi=0,.15,.25,.5 in subdirectories')
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args(argv)
    if args.plot_only:
        with np.load(args.output/'responses.npz') as arrays:
            plot_results(arrays, json.loads((args.output/'summary.json').read_text()), args.output)
        if args.haldane_controls:
            plot_haldane_controls(args.output)
        return
    if min(args.grids) < 3 or min(args.kbt, args.linewidth, args.lattice_constant) <= 0 or not np.all(np.isfinite(
            [args.kbt, args.linewidth, args.lattice_constant, args.hopping, args.delta, args.t2, args.phase, *args.fermi])):
        parser.error('Finite parameters, grids >=3 and positive kBT, linewidth, lattice constant required')
    if args.hopping == 0:
        parser.error('Use nonzero nearest-neighbor hopping')
    run_case(args)
    if args.haldane_controls:
        import copy
        for phase_pi in (0., .15, .25, .5):
            control = copy.copy(args)
            control.delta, control.t2, control.phase = .4, .15, phase_pi*np.pi
            control.output = args.output/f'haldane_phi_{phase_pi:g}pi'
            run_case(control)
        plot_haldane_controls(args.output)


def run_case(args):
    args.output.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        model = build_model(args.output/'model', args.hopping, args.delta, args.lattice_constant, t2=args.t2, phase=args.phase)
    for name, system in [('model', model)]:
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
                       max_abs_bsr=float(np.max(np.abs(result['bsr']))),
                       max_abs_v2_minus_pbc=float(np.max(np.abs(result['bsr_v2']-result['finite_pbc']))),
                       max_abs_v2_minus_v1=float(np.max(np.abs(result['bsr_v2']-result['bsr']))), band_edges_eV=result['band_edges'].tolist())
            metrics.append(row)
            print(json.dumps(row), flush=True)
        # Always evaluate midgap explicitly, even when absent from the requested scan.
        midgap_energy = float(np.mean(result['band_edges']))
        midgap = calculate(model, grids[-1], [midgap_energy], args.kbt, args.linewidth)
        narrower = calculate(model, grids[-1], [midgap_energy], args.kbt/2, args.linewidth/2)
    arrays.update({f'midgap_{key}': value for key, value in midgap.items()})
    arrays.update({f'narrower_midgap_{key}': value for key, value in narrower.items()})
    tol = 1e-10
    checks = dict(aca_moment_identically_zero=bool(np.all(model.orbital == 0)),
        aca_ohc_zero=all(row['max_abs_aca'] < tol for row in metrics),
        shc_zero=all(row['max_abs_shc'] < tol for row in metrics),
        v2_matches_finite_pbc=all(row['max_abs_v2_minus_pbc'] < tol for row in metrics))
    summary = dict(parameters={k: v for k, v in vars(args).items() if k != 'output'}, checks=checks,
        sampled_gap_edges_eV=result['band_edges'].tolist(), midgap_energy_eV=midgap_energy,
        finite_pbc='NxN periodic cells, translation-block calculation of Eqs.20,21,1,2; unwrapped bond velocities; diagonal r removed',
        axes=['fermi', 'drive', 'flow', 'polarization'],
        units='Native DM 2D Hall normalization: 2*pi/cell_area times BZ mean, magnetic-moment vertices; not a quantized conductance unit',
        spin_degeneracy=2, mesh_results=metrics, zero_tolerance=tol,
        has_sampled_indirect_gap=bool(result['band_edges'][0] < result['band_edges'][1]),
        midgap_bsr_v2=float(midgap['bsr_v2'][0, 0, 1, 2]),
        midgap_bsr=float(midgap['bsr'][0, 0, 1, 2]),
        midgap_bsr_I=float(midgap['bsr_I'][0, 0, 1, 2]), midgap_bsr_E=float(midgap['bsr_E'][0, 0, 1, 2]),
        narrower_midgap_bsr=float(narrower['bsr'][0, 0, 1, 2]),
        narrower_kbt=args.kbt/2, narrower_linewidth=args.linewidth/2)
    if len(grids) > 1:
        for key in ('bsr', 'bsr_v2', 'finite_pbc'):
            summary[f'last_mesh_max_{key}_difference'] = float(np.max(np.abs(result[key]-arrays[f'{key}_n{grids[-2]}'])))
    np.savez_compressed(args.output/'responses.npz', **arrays)
    np.savetxt(args.output/'hall_scan.csv', np.column_stack([fermi, result['bsr'][:, 0, 1, 2],
        result['bsr_v2'][:, 0, 1, 2], result['finite_pbc'][:, 0, 1, 2],
        result['bsr_I'][:, 0, 1, 2], result['bsr_E'][:, 0, 1, 2], result['aca'][:, 0, 1, 2],
        result['shc'][:, 0, 1, 2], np.max(np.abs(result['shc']), axis=(1, 2, 3))]),
        delimiter=',', header='fermi_eV,BSR_OHC,BSR_V2_OHC,finite_PBC_OHC,BSR_I,BSR_E,ACA_OHC,SHC,max_abs_SHC', comments='')
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    plot_results(arrays, summary, args.output)
    print(json.dumps(summary, indent=2), flush=True)
    if not all(checks.values()):
        raise RuntimeError('Honeycomb source checks failed; inspect summary.json')


if __name__ == '__main__':
    main()
