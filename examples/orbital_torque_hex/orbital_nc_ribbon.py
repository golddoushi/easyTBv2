"""Hexagonal BSR source / p-shell SOC converter: profiles and row drives.

python examples/orbital_torque_hex/orbital_nc_ribbon.py --edge armchair
Row drive means {P_row,v_x}/2, not P_row v_x P_row.
"""
import argparse
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from system import System_tb
from examples.orbital_torque_hex.orbital_ribbon import build_ribbon
from calculator import CollisionSpecification
from calculator.electronic_stages import Spin, SpinSOCTorque, SpinFluxTorque, OrbitalSOCTorque, OrbitalFluxTorque
from calculator.response_stages import DensityMatrixResponse
from calculator.density_matrix import LengthGaugeFirstOrderDMKernel


SPIN_OBSERVABLES = ('spin', 'soc_torque', 'flux_torque', 'total_torque')
ORBITAL_OBSERVABLES = ('orbital_soc', 'orbital_h0', 'orbital_crystal', 'orbital_transfer', 'orbital_total')
OBSERVABLES = SPIN_OBSERVABLES + ORBITAL_OBSERVABLES


def observable_vertices(prepared):
    # Flux and total vertices require projection BEFORE the commutator.
    orbital_soc = OrbitalSOCTorque(states=prepared.states)
    orbital_h0 = OrbitalFluxTorque(states=prepared.states)
    return dict(spin=prepared.spin_K, soc_torque=prepared.spinTorque_K,
                flux_torque=prepared.spin_K, total_torque=prepared.spin_K,
                orbital_soc=orbital_soc.orbitalTorque_K, orbital_h0=orbital_soc.orbital_K,
                orbital_crystal=orbital_h0.orbitalTorque_K, orbital_total=orbital_soc.orbital_K)


def rate_hamiltonians(prepared):
    h0 = SpinFluxTorque(states=prepared.states).ham0_K
    full = prepared.states.energies[..., :, None]*np.eye(prepared.states.energies.shape[-1])
    return dict(flux_torque=h0, total_torque=full, orbital_h0=h0, orbital_total=full)


def orbital_balance_residual(response, prefix=''):
    residuals = []
    for suffix in ('', '_I', '_E'):
        residuals.extend((np.max(np.abs(response[prefix+'orbital_total'+suffix]
            -response[prefix+'orbital_h0'+suffix]-response[prefix+'orbital_soc'+suffix])),
            np.max(np.abs(response[prefix+'soc_torque'+suffix]+2*response[prefix+'orbital_soc'+suffix]))))
    return float(max(residuals))


def observable_label(name):
    if name != 'spin':
        angular = 'L' if name.startswith('orbital_') else 'S'
        title = name.replace('orbital_', '').replace('_torque', '').upper()
        return title+rf' $\delta T^{angular}_z/E_x$'+'\n'+r'[$\mu_B$ eV/(V/$\AA$)]'
    return r'SPIN $\delta\mu_z/E_x$'+'\n'+r'[$\mu_B$/(V/$\AA$)]'


def _write_model(directory, lattice, positions, n_atoms, h, rvec, soc):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    c_atoms = len(positions)-n_atoms
    species = ('N C\n'+f'{n_atoms} {c_atoms}\n') if n_atoms else f'C\n{c_atoms}\n'
    (directory/'POSCAR').write_text('BSR N / p-shell C\n1.0\n'+
        ''.join(' '.join(map(str, row))+'\n' for row in lattice)+species+'Cartesian\n'+
        ''.join(' '.join(map(str, row))+'\n' for row in positions))
    size = h.shape[-1]
    ham = np.zeros((len(rvec), 2*size, 2*size), complex)
    ham[:, :size, :size] = ham[:, size:, size:] = h
    with (directory/'junction_hr.dat').open('w') as stream:
        stream.write(f'Hexagonal N and p-shell C\n{2*size}\n{len(rvec)}\n'+' '.join(['1']*len(rvec))+'\n')
        for r, block in zip(rvec, ham):
            for j in range(2*size):
                for i in range(2*size):
                    z = block[i, j]
                    stream.write(f'{r[0]} {r[1]} {r[2]} {i+1} {j+1} {z.real:.16e} {z.imag:.16e}\n')
    model = System_tb(str(directory), hamR='junction', projections={'N': [0], 'C': [1]} if n_atoms else {'C': [1]}, ispin=2)
    model.verbose = False
    zero = model.iRvecIndex[(0, 0, 0)]
    soc_r = np.zeros_like(model.ham_R)
    for atom in range(n_atoms, len(positions)):
        ids = np.flatnonzero(model.iorb2iatomMap == atom)
        angular = model.orbital[:, ids][:, :, ids]
        spin = model.spin[:, ids][:, :, ids]
        soc_r[zero][np.ix_(ids, ids)] += soc/2*sum(angular[a]@spin[a] for a in range(3))
    model.ham_R += soc_r
    model.getSOCR('explicit', soc_R=soc_r)
    model.getH0R()
    model.exchange_R = np.zeros_like(model.ham_R)
    np.testing.assert_allclose(model.ham_R, model.ham0_R+model.soc_R, atol=1e-14)
    model._refreshHamiltonianDerivedMatrices()
    return model


def build_nc_ribbon(directory, edge='zigzag', n_width=4, c_width=3, soc=.4,
                    converter_hopping=.5, interface_hopping=.4, converter_shift=0.):
    """N retains exactly the finite s-only source. C has one p-shell per row.

    C basis is (pz,px,py); all bonds are in the xy plane. The interface uses
    t_sp*(0,dx,dy)/|d| and its Hermitian conjugate. Only the uppermost N line
    is connected to C's first row, with neighboring longitudinal images.
    """
    if c_width < 1 or not np.all(np.isfinite([soc, converter_hopping, interface_hopping, converter_shift])):
        raise ValueError('Positive C width and finite Hamiltonian parameters required')
    directory = Path(directory)
    source = build_ribbon(directory/'source', n_width, edge=edge)
    ns = source.norbs//2
    positions = source.rList[:, :ns].T.copy()
    length = source.lattice[0, 0]
    spacing = 2.4/np.sqrt(3)
    top = positions[:, 1].max()
    top_atom = np.flatnonzero(np.isclose(positions[:, 1], top))[0]
    # Keep the same longitudinal home-cell convention as N; its zigzag
    # supercell basis need not lie inside [0,Lx). Wrapping C alone can discard
    # one of the intended oblique bonds from the R=-1,0,1 interface stencil.
    interface_x = positions[top_atom, 0]+length/2
    cp = np.array([[interface_x, top+(i+1)*spacing, positions[0, 2]] for i in range(c_width)])
    positions = np.concatenate([positions, cp])
    lattice = np.diag([length, positions[:, 1].max()-positions[:, 1].min()+2*spacing, 15.])
    rvec = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0]])
    lookup = {tuple(r): i for i, r in enumerate(rvec)}
    h = np.zeros((3, ns+3*c_width, ns+3*c_width), complex)
    for r, block in zip(source.iRvec, source.ham_R):
        h[lookup[tuple(r)], :ns, :ns] = block[:ns, :ns]
    for c in range(c_width):
        ids = np.arange(ns+3*c, ns+3*c+3)
        h[0][np.ix_(ids, ids)] = converter_shift*np.eye(3)
        h[1][np.ix_(ids, ids)] = h[2][np.ix_(ids, ids)] = -converter_hopping*np.eye(3)
        if c+1 < c_width:
            h[0][np.ix_(ids, ids+3)] = h[0][np.ix_(ids+3, ids)] = -converter_hopping*np.eye(3)
    for atom in np.flatnonzero(np.isclose(positions[:ns, 1], top)):
        for ir, r in enumerate(rvec):
            bond = cp[0]+r@lattice-positions[atom]
            if abs(bond[0]) <= .75*length+1e-10:
                value = interface_hopping*np.array([0., bond[0], bond[1]])/np.linalg.norm(bond)
                h[ir, atom, ns:ns+3] += value
                h[lookup[tuple(-r)], ns:ns+3, atom] += value
    model = _write_model(directory, lattice, positions, ns, h, rvec, soc)
    y, groups = np.unique(np.round(positions[:, 1], 8), return_inverse=True)
    regions = np.array(['N' if np.any((groups == row) & (np.arange(len(positions)) < ns)) else 'C' for row in range(len(y))])
    return model, y, groups, regions


def build_converter_bulk(directory, length=2.4, soc=.4, hopping=.5, shift=0.):
    rvec = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]])
    h = np.array([shift*np.eye(3)]+[-hopping*np.eye(3)]*4, dtype=complex)
    return _write_model(directory, np.diag([length, 2.4/np.sqrt(3), 15.]), np.zeros((1, 3)), 0, h, rvec, soc)


def mesh_execution(grid, workers=1):
    """Fixed, equal-size chunks; worker count changes scheduling, not the mesh."""
    if grid < 4 or workers < 1:
        raise ValueError('grid >=4 and workers >=1 required')
    chunks = next(n for n in range(1, grid+1) if grid % n == 0 and grid//n <= 32)
    return chunks, min(workers, chunks)


def calculate(model, groups, regions, grid, fermi, kbt=.01, linewidth=.02, workers=1, profile_rows=None):
    """Established LG scalar-CRTA kernel; exact dual contraction for row drives.

    Observables use {P_row,m}/2. Inter-row v_x bonds are half-assigned to each
    drive row. Save xyz for mirror checks, but plots show only z. Equal chunks
    bound memory; each k point belongs to the shifted uniform mesh once.
    """
    if grid < 4 or kbt <= 0 or linewidth <= 0:
        raise ValueError('grid >=4 and positive kBT/linewidth required')
    count, effective = mesh_execution(grid, workers)
    if profile_rows is not None:
        profile_rows = np.asarray(profile_rows, dtype=int)
        if len(profile_rows)==0 or len(set(profile_rows.tolist())) != len(profile_rows) or np.any((profile_rows < 0) | (profile_rows >= len(regions))):
            raise ValueError('Profile drive rows must be distinct valid zero-based row indices')
    atom_rows = groups[model.iorb2iatomMap]
    masks = np.array([atom_rows == i for i in range(len(regions))], dtype=float)
    c_mask = np.isin(atom_rows, np.flatnonzero(regions == 'C')).astype(float)
    def evaluate_shard(shard):
        results = {}
        prepared = DensityMatrixResponse(model=model, kgrid=(grid//count, 1, 1),
            kOffset=((shard+.5)/grid, 0., 0.), fermiList=np.asarray(fermi), kBT=kbt,
            collisionSpecification=CollisionSpecification('crta', constantLinewidth=linewidth),
            electronic_stages=(Spin, SpinSOCTorque))
        kernel = LengthGaugeFirstOrderDMKernel(prepared.inputs, full_covariant_derivative=True,
            finite_temperature_occupations=True, collision=prepared.collision)
        u = prepared.states.vectors
        ud = u.conj().swapaxes(-1, -2)
        vx = u@prepared.inputs.velocity_K[..., 0, :, :]@ud
        # Spin stage already supplies magnetic moment (minus Pauli).
        observables = observable_vertices(prepared)
        hamiltonians = rate_hamiltonians(prepared)
        for name, band in observables.items():
            ao = u[..., None, :, :]@band@ud[..., None, :, :]
            c_band = ud[..., None, :, :]@(ao*(c_mask[:, None]+c_mask[None, :])/2)@u[..., None, :, :]
            h = hamiltonians.get(name)
            if h is not None:
                c_band = 1j*(h[..., None, :, :]@c_band-c_band@h[..., None, :, :])
            for key in (name, name+'_I', name+'_E', 'drive_'+name):
                results.setdefault(key, np.zeros((len(regions), len(fermi), 3)))
            for fi, mu in enumerate(fermi):
                state = kernel.fermi_state(fi, mu)
                rho_channels = state.at_frequency(0.).block(0).rho1
                for channel, rho in rho_channels.items():
                    effective_rho = rho if h is None else 1j*(rho@h-h@rho)
                    density = (u@effective_rho@ud)[..., None, :, :]
                    diagonal = np.diagonal((density@ao+ao@density)/2, axis1=-2, axis2=-1)
                    local = np.einsum('ri,...ci->...rc', masks, diagonal).real.mean(axis=(0, 1, 2))/count
                    results[name+'_'+channel][:, fi] += local
                    results[name][:, fi] += local
                weight = state.collision.solve(-1j*state.divided_difference, prepared.inputs.deltaE, 0.)
                dual = u[..., None, :, :]@(weight.swapaxes(-1, -2)[..., None, :, :]*c_band)@ud[..., None, :, :]
                diagonal = np.diagonal((vx[..., None, :, :]@dual+dual@vx[..., None, :, :])/2, axis1=-2, axis2=-1)
                results['drive_'+name][:, fi] += np.einsum('ri,...ci->...rc', masks, diagonal).real.mean(axis=(0, 1, 2))/count
        for suffix in ('', '_I', '_E'):
            results['orbital_transfer'+suffix] = results['orbital_h0'+suffix]-results['orbital_crystal'+suffix]
        results['drive_orbital_transfer'] = results['drive_orbital_h0']-results['drive_orbital_crystal']
        if profile_rows is not None:
            results.update(_row_profile_matrices(prepared, kernel, masks, profile_rows, fermi, count))
        return results

    # Set the process-wide BLAS limit outside worker threads to avoid nested
    # thread oversubscription and racing threadpoolctl context restoration.
    with threadpool_limits(limits=1):
        if effective == 1:
            parts = map(evaluate_shard, range(count))
            return _sum_parts(parts)
        with ThreadPoolExecutor(max_workers=effective) as pool:
            # map preserves shard order, giving the same reduction as serial.
            return _sum_parts(pool.map(evaluate_shard, range(count)))


def _sum_parts(parts):
    total = None
    for part in parts:
        if total is None:
            total = part
        else:
            for key in total:
                total[key] += part[key]
    return total


def _row_profile_matrices(prepared, kernel, masks, drive_rows, fermi, chunks):
    """All observed rows for selected drives; z only, with I/E decomposition.

    Build symmetrized vertices from low-rank row projectors, then contract
    the scalar-CRTA band-pair weight. Process one k point and eight observed
    rows at a time; never allocate a k*drive*observed*band*band tensor.
    """
    ids = [np.flatnonzero(mask) for mask in masks]
    shape = (len(drive_rows), len(masks), len(fermi))
    result = {f'row_profile_{name}{suffix}': np.zeros(shape) for name in OBSERVABLES for suffix in ('', '_I', '_E')}
    weights = []
    for fi, mu in enumerate(fermi):
        state = kernel.fermi_state(fi, mu)
        weights.append(state.collision.solve(-1j*state.divided_difference, prepared.inputs.deltaE, 0.))
    scale = chunks*np.prod(prepared.states.energies.shape[:-1])
    observables = observable_vertices(prepared)
    hamiltonians = rate_hamiltonians(prepared)
    for index in np.ndindex(prepared.states.energies.shape[:-1]):
        u = prepared.states.vectors[index]
        def row_vertices(operator, rows):
            product = u@operator
            vertices = []
            for row in rows:
                local = ids[row]
                value = u[local].conj().T@product[local]
                vertices.append((value+value.conj().T)/2)
            return np.asarray(vertices)
        drives = row_vertices(prepared.inputs.velocity_K[index][0], drive_rows)
        drive_diag = np.diagonal(drives, axis1=-2, axis2=-1)
        for name, band in observables.items():
            for start in range(0, len(masks), 8):
                end = min(start+8, len(masks))
                observed = row_vertices(band[index][2], range(start, end))
                if name in ('total_torque', 'orbital_total'):
                    energies = prepared.states.energies[index]
                    observed = 1j*(energies[:, None]-energies[None, :])*observed
                elif name in hamiltonians:
                    h = hamiltonians[name][index]
                    observed = 1j*(h@observed-observed@h)
                flat_observed = observed.swapaxes(-1, -2).reshape(end-start, -1).T.copy()
                observed_diag = np.diagonal(observed, axis1=-2, axis2=-1)
                for fi, weight in enumerate(weights):
                    w = weight[index]
                    total = ((drives*w).reshape(len(drive_rows), -1)@flat_observed).real/scale
                    intra = ((drive_diag*np.diag(w))@observed_diag.T).real/scale
                    result['row_profile_'+name][:, start:end, fi] += total
                    result['row_profile_'+name+'_I'][:, start:end, fi] += intra
                    result['row_profile_'+name+'_E'][:, start:end, fi] += total-intra
    for suffix in ('', '_I', '_E'):
        result['row_profile_orbital_transfer'+suffix] = result['row_profile_orbital_h0'+suffix]-result['row_profile_orbital_crystal'+suffix]
    return result


def save_row_profiles(arrays, y, regions, fermi, drive_rows, output, metadata):
    """Dedicated artifacts; the original summed response files stay intact."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)/'row_profiles'
    output.mkdir(parents=True, exist_ok=True)
    cut_arrays = {case+'_orbital_cut_current': -np.cumsum(arrays[case+'_row_profile_orbital_transfer'], axis=1)[:, :-1]
                  for case in ('full', 'no_soc', 'disconnected')}
    np.savez_compressed(output/'orbital_cut_currents.npz', **cut_arrays,
                        cut_y_A=(y[:-1]+y[1:])/2, drive_rows=drive_rows, fermi_eV=fermi)
    np.savez_compressed(output/'responses.npz', **arrays, y_A=y, regions=regions,
        fermi_eV=fermi, drive_rows=np.asarray(drive_rows))
    (output/'summary.json').write_text(json.dumps(metadata, indent=2)+'\n')
    with (output/'profiles.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['case','drive_row','drive_y_A','drive_region','observed_row','observed_y_A','observed_region','fermi_eV',*[name+suffix for name in OBSERVABLES for suffix in ('_z','_I','_E')]])
        for case in ('full','no_soc','disconnected'):
            for di, row in enumerate(drive_rows):
                for observed in range(len(y)):
                    for fi, mu in enumerate(fermi):
                        values = [arrays[f'{case}_row_profile_{name}{suffix}'][di, observed, fi] for name in OBSERVABLES for suffix in ('','_I','_E')]
                        writer.writerow([case,row,y[row],regions[row],observed,y[observed],regions[observed],mu]+values)
    boundary = (y[regions == 'N'].max()+y[regions == 'C'].min())/2
    for fi, mu in enumerate(fermi):
        for di, row in enumerate(drive_rows):
            fig, axes = plt.subplots(3, 3, figsize=(18, 13), sharex=True, constrained_layout=True)
            axes = axes.ravel()
            for ax, name in zip(axes, OBSERVABLES):
                for case, style in [('full','-'), ('no_soc','--'), ('disconnected',':')]:
                    ax.plot(y, arrays[f'{case}_row_profile_{name}'][di, :, fi], style, label=case.replace('_',' '))
                ax.axvspan(y.min(), boundary, color='steelblue', alpha=.08)
                ax.axvspan(boundary, y.max(), color='orange', alpha=.1)
                ax.axvline(y[row], color='purple', ls=':', label='Driven row')
                ax.axvline(boundary, color='0.4', ls='--', lw=.7)
                ax.axhline(0, color='0.5', lw=.5)
                ax.set_ylabel(observable_label(name))
                ax.legend(fontsize=8)
                ax.grid(alpha=.2)
            fig.suptitle(f'Drive row {row} ({regions[row]}), y={y[row]:.4g} Angstrom; EF={mu:g} eV; Nk={metadata["grid"]}\n'+r'Vertex $\{P_r,v_x\}/2$; z observable on each observed row')
            axes[-1].set_xlabel('Observed row y (Angstrom); N blue / C orange')
            target = output/f'profile_drive_row{row:03d}_f{fi}.png'
            temporary = target.with_suffix('.pending.png')
            fig.savefig(temporary, dpi=140)
            temporary.replace(target)
            plt.close(fig)
        fig, axes = plt.subplots(3, 3, figsize=(18, 14), constrained_layout=True)
        axes = axes.ravel()
        for ax, name in zip(axes, OBSERVABLES):
            values = arrays[f'full_row_profile_{name}'][:, :, fi]
            maximum = max(float(np.max(np.abs(values))), 1e-15)
            im = ax.imshow(values, origin='lower', aspect='auto', cmap='RdBu_r', vmin=-maximum, vmax=maximum)
            ax.set(xlabel='Observed physical row index', ylabel='Driven physical row index', title=f'{name.upper().replace("_", " ")} z, EF={mu:g} eV')
            ticks = np.unique(np.linspace(0, len(drive_rows)-1, min(9, len(drive_rows)), dtype=int))
            ax.set_yticks(ticks, labels=np.asarray(drive_rows)[ticks])
            ax.axvline(np.flatnonzero(regions == 'C')[0]-.5, color='0.4', ls='--', lw=.8)
            fig.colorbar(im, ax=ax, label=('mu_B*eV/(V/Angstrom)' if name!='spin' else 'mu_B/(V/Angstrom)'))
        fig.savefig(output/f'response_matrix_f{fi}.png', dpi=140)
        plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        maximum = max(max(float(np.max(np.abs(v[:,:,fi]))) for v in cut_arrays.values()), 1e-15)
        for ax, case in zip(axes, ('full', 'no_soc', 'disconnected')):
            im = ax.imshow(cut_arrays[case+'_orbital_cut_current'][:,:,fi], origin='lower', aspect='auto',
                           cmap='RdBu_r', vmin=-maximum, vmax=maximum)
            ticks = np.unique(np.linspace(0, len(drive_rows)-1, min(9,len(drive_rows)), dtype=int))
            ax.set_yticks(ticks, labels=np.asarray(drive_rows)[ticks])
            ax.set(xlabel='Cut after physical row', ylabel='Driven physical row', title=case.replace('_',' '))
            ax.axvline(np.flatnonzero(regions=='C')[0]-.5, color='0.4', ls='--')
        fig.colorbar(im, ax=axes, label='Orbital moment current toward +y [mu_B eV/(V/Angstrom)]')
        fig.savefig(output/f'orbital_cut_current_f{fi}.png', dpi=140)
        plt.close(fig)


def save_orbital_cuts(arrays, output):
    """Symmetrized H0 orbital flux across cuts; strictly internal C cuts are unambiguous here."""
    import matplotlib.pyplot as plt
    output = Path(output)
    y, regions = arrays['y_A'], arrays['regions']
    cuts = (y[:-1]+y[1:])/2
    currents = {case: -np.cumsum(arrays[case+'_orbital_transfer'], axis=0)[:-1]
                for case in ('full', 'no_soc', 'disconnected')}
    np.savez_compressed(output/'orbital_cut_currents.npz', **currents, cut_y_A=cuts,
                        fermi_eV=arrays['fermi_eV'], inside_C=(regions[:-1]=='C') & (regions[1:]=='C'))
    with (output/'orbital_cut_currents.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['case','cut_after_row','cut_y_A','inside_C','fermi_eV','orbital_current_z'])
        for case, values in currents.items():
            for cut in range(len(cuts)):
                for fi, mu in enumerate(arrays['fermi_eV']):
                    writer.writerow([case,cut,cuts[cut],regions[cut]==regions[cut+1]=='C',mu,values[cut,fi,2]])
    for fi, mu in enumerate(arrays['fermi_eV']):
        fig, ax = plt.subplots(figsize=(9,5), constrained_layout=True)
        for (case, values), style in zip(currents.items(), ('o-', 's--', '^:')):
            ax.plot(cuts, values[:,fi,2], style, label=case.replace('_',' '), ms=3)
        boundary = (y[regions=='N'].max()+y[regions=='C'].min())/2
        ax.axvline(boundary, ls='--', color='0.4')
        ax.axhline(0, color='0.5', lw=.5)
        ax.set(xlabel='Cut y (Angstrom); positive current toward +y',
               ylabel='Orbital moment current / Ex [mu_B eV/(V/Angstrom)]',
               title=f'Uniform drive, EF={mu:g} eV; symmetrized H0 orbital current')
        ax.legend(); ax.grid(alpha=.2)
        fig.savefig(output/f'orbital_cut_current_f{fi}.png', dpi=150)
        plt.close(fig)


def add_saved_row_profiles(output, drive_rows=None, workers=1):
    """Recompute missing two-row information from saved model parameters.

    The old uniform and C-total marginals do not determine this matrix.
    Verify rebuilt Hamiltonians before using the saved results as checks.
    """
    output = Path(output)
    summary = json.loads((output/'summary.json').read_text())
    p = summary['parameters']
    grid = max(p['grids'])
    with np.load(output/'responses.npz') as saved:
        y, regions = saved['y_A'].copy(), saved['regions'].copy()
        rows = list(range(len(y))) if drive_rows is None or len(drive_rows)==0 else list(drive_rows)
        if not rows or len(set(rows)) != len(rows) or min(rows)<0 or max(rows)>=len(y):
            raise ValueError('Choose distinct row indices within the saved ribbon')
        arrays, checks, timings = {}, {}, {}
        for case in ('full','no_soc','disconnected'):
            with contextlib.redirect_stdout(io.StringIO()):
                model, fresh_y, groups, fresh_regions = build_nc_ribbon(output/'row_profiles'/'models'/case,
                    p['edge'], p['n_width'], p['c_width'], 0. if case=='no_soc' else p['soc'],
                    p['converter_hopping'], 0. if case=='disconnected' else p['interface_hopping'], p['converter_shift'])
            np.testing.assert_allclose(fresh_y, y, atol=1e-10)
            np.testing.assert_array_equal(fresh_regions, regions)
            with np.load(output/case/'model.npz') as original:
                for key in ('ham_R','iRvec','rList','lattice'):
                    np.testing.assert_allclose(getattr(model,key), original[key], atol=1e-12)
            start = perf_counter()
            response = calculate(model, groups, regions, grid, p['fermi'], p['kbt'], p['linewidth'], workers, rows)
            timings[case] = perf_counter()-start
            checks[case+'_orbital_balance'] = orbital_balance_residual(response, 'row_profile_')
            for name in OBSERVABLES:
                for key in (name, 'drive_'+name):
                    if f'{case}_{key}' in saved:
                        checks[f'{case}_{key}_saved_agreement'] = float(np.max(np.abs(response[key]-saved[f'{case}_{key}'])))
                matrix = response['row_profile_'+name]
                checks[f'{case}_{name}_C_sum'] = float(np.max(np.abs(matrix[:, regions=='C'].sum(axis=1)-response['drive_'+name][rows,:,2])))
                if len(rows)==len(y):
                    checks[f'{case}_{name}_drive_sum'] = float(np.max(np.abs(matrix.sum(axis=0)-response[name][:,:,2])))
            arrays.update({case+'_'+key:value for key,value in response.items() if key.startswith('row_profile_')})
            print(f'Row profiles: {case}, Nk={grid}, {len(rows)} drives, {timings[case]:.2f}s', flush=True)
            # Checkpoint computed cases before the next expensive evaluation.
            np.savez_compressed(output/'row_profiles'/'checkpoint.npz', **arrays)
    metadata = dict(parameters=p, grid=grid, workers=workers, effective_workers=mesh_execution(grid,workers)[1], axes=['selected_drive_row','observed_row','fermi'],
        component='z', units={name: ('mu_B*eV/(V/Angstrom)' if name!='spin' else 'mu_B/(V/Angstrom)') for name in OBSERVABLES}, drive_rows=rows, residuals=checks, timings_seconds=timings,
        projection=summary['row_drive'], convergence='Uses saved finest mesh; reconstruction checks are not mesh-convergence tests')
    save_row_profiles(arrays, y, regions, p['fermi'], rows, output, metadata)
    if max(checks.values())>1e-8:
        raise RuntimeError('Row-profile reconstruction failed; inspect row_profiles/summary.json')


def replot_row_profiles(output):
    folder = Path(output)/'row_profiles'
    with np.load(folder/'responses.npz') as saved:
        data = {key:saved[key].copy() for key in saved.files}
    save_row_profiles({key:value for key,value in data.items() if '_row_profile_' in key},
        data['y_A'], data['regions'], data['fermi_eV'], data['drive_rows'], output,
        json.loads((folder/'summary.json').read_text()))


def plot(arrays, summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    y, regions = arrays['y_A'], arrays['regions']
    boundary = (y[regions == 'N'].max()+y[regions == 'C'].min())/2
    for fi, mu in enumerate(arrays['fermi_eV']):
        for drive in (False, True):
            fig, axes = plt.subplots(3, 3, figsize=(18, 13), sharex=True, constrained_layout=True)
            axes = axes.ravel()
            for ax, name in zip(axes, OBSERVABLES):
                for case, style in [('full', 'o-'), ('no_soc', 's--'), ('disconnected', '^:')]:
                    key = f'{case}_'+('drive_' if drive else '')+name
                    ax.plot(y, arrays[key][:, fi, 2], style, ms=4, label=case.replace('_', ' '))
                ax.axvspan(y.min(), boundary, color='steelblue', alpha=.08)
                ax.axvspan(boundary, y.max(), color='orange', alpha=.1)
                ax.axvline(boundary, color='0.4', ls='--', lw=.8)
                ax.axhline(0, color='0.5', lw=.6)
                ax.set_ylabel(('C-total ' if drive else 'Row ') + observable_label(name))
                ax.legend(fontsize=8)
                ax.grid(alpha=.2)
            fig.suptitle(f"{summary['edge'].capitalize()} N / p-shell C, EF={mu:g} eV\n"+
                ('One symmetrized row drive at a time' if drive else 'Uniform Ex; observables summed over each physical row'))
            axes[-1].set_xlabel(('Driven' if drive else 'Observed')+' row y (Angstrom); N blue / C orange')
            target = Path(output)/f'{"drive_rows" if drive else "profile"}_f{fi}.png'
            temporary = target.with_suffix('.pending.png')
            fig.savefig(temporary, dpi=160)
            temporary.replace(target)
            plt.close(fig)


def export_csv(arrays, output):
    for drive in (False, True):
        path = Path(output)/('drive_rows.csv' if drive else 'profiles.csv')
        with path.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['case', 'driven_row' if drive else 'observed_row', 'y_A', 'region', 'fermi_eV',
                *[(('C_total_' if drive else 'row_')+name+'_z') for name in OBSERVABLES]])
            for case in ('full', 'no_soc', 'disconnected'):
                prefix = case+('_drive_' if drive else '_')
                for row, (y, region) in enumerate(zip(arrays['y_A'], arrays['regions'])):
                    for fi, mu in enumerate(arrays['fermi_eV']):
                        writer.writerow([case, row, y, region, mu, *[arrays[prefix+name][row, fi, 2] for name in OBSERVABLES]])


def plot_geometry(model, groups, regions, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    half = model.norbs//2
    atom_map = model.iorb2iatomMap[:half]
    atom_ids = [np.flatnonzero(atom_map == atom) for atom in range(len(groups))]
    positions = np.array([model.rList[:, ids[0]] for ids in atom_ids])
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for cell in (-1, 0, 1):
        points = positions+cell*model.lattice[0]
        for r, block in zip(model.iRvec, model.ham_R[:, :half, :half]):
            for i, left in enumerate(atom_ids):
                for j, right in enumerate(atom_ids):
                    if i == j and np.all(r == 0):
                        continue
                    if np.max(np.abs(block[np.ix_(left, right)])) > 1e-10:
                        bond = np.array([points[i], points[j]+r@model.lattice])
                        ax.plot(bond[:, 0], bond[:, 1], color='0.65', lw=.8)
        for region, color in [('N', 'steelblue'), ('C', 'darkorange')]:
            ids = regions[groups] == region
            ax.scatter(points[ids, 0], points[ids, 1], color=color, zorder=3, label=region if cell == 0 else None)
    ax.set_aspect('equal')
    ax.set(xlabel='x (Angstrom): periodic / Ex', ylabel='y (Angstrom): open', title='N: s-only honeycomb / C: p-shell rows; actual bonds')
    ax.legend()
    fig.savefig(Path(output)/'geometry.png', dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--edge', choices=['zigzag', 'armchair'], default='zigzag')
    parser.add_argument('--n-width', type=int, default=None, help='default 4 zigzag / 2 armchair transverse cells')
    parser.add_argument('--c-width', type=int, default=3)
    parser.add_argument('--grids', nargs='+', type=int, default=[512, 1024])
    parser.add_argument('--fermi', nargs='+', type=float, default=[0., .8])
    parser.add_argument('--soc', type=float, default=.4)
    parser.add_argument('--interface-hopping', type=float, default=.4)
    parser.add_argument('--converter-hopping', type=float, default=.5)
    parser.add_argument('--converter-shift', type=float, default=0.)
    parser.add_argument('--kbt', type=float, default=.01)
    parser.add_argument('--linewidth', type=float, default=.02)
    parser.add_argument('--workers', type=int, default=1, help='Concurrent k-mesh chunks; each uses one BLAS thread')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--row-profiles', action='store_true', help='Also compute profiles for all or selected drive rows')
    parser.add_argument('--row-profiles-only', action='store_true', help='Add row profiles using saved parameters and finest mesh; preserve original outputs')
    parser.add_argument('--profile-drive-rows', type=int, nargs='*', default=None, help='Zero-based physical drive rows; omit or give no indices for all rows')
    args = parser.parse_args(argv)
    if args.n_width is None:
        args.n_width = 2 if args.edge == 'armchair' else 4
    output = args.output or Path('tmp/orbital_torque_hex')/('nc_'+args.edge)
    if args.workers < 1:
        parser.error('workers >=1 required')
    if args.row_profiles_only:
        if args.plot_only:
            replot_row_profiles(output)
        else:
            add_saved_row_profiles(output, args.profile_drive_rows, args.workers)
        return
    if args.plot_only:
        with np.load(output/'responses.npz') as arrays:
            plot(arrays, json.loads((output/'summary.json').read_text()), output)
            export_csv(arrays, output)
            save_orbital_cuts(arrays, output)
        if args.row_profiles:
            replot_row_profiles(output)
        return
    if args.n_width < 2 or args.c_width < 1 or min(args.grids) < 4 or args.workers < 1 or min(args.kbt, args.linewidth) <= 0 or not np.all(np.isfinite([args.kbt,args.linewidth,args.soc,args.interface_hopping,args.converter_hopping,args.converter_shift,*args.fermi])):
        parser.error('Widths N>=2, C>=1; grids>=4; workers>=1; positive kBT/linewidth and finite parameters required')
    output.mkdir(parents=True, exist_ok=True)
    grids = sorted(set(args.grids))
    arrays = dict(fermi_eV=np.array(args.fermi))
    metrics = {}
    timings = {}
    row_arrays, row_checks = {}, {}
    with threadpool_limits(1):
        for case, soc, interface in [('full', args.soc, args.interface_hopping), ('no_soc', 0., args.interface_hopping), ('disconnected', args.soc, 0.)]:
            with contextlib.redirect_stdout(io.StringIO()):
                model, y, groups, regions = build_nc_ribbon(output/case, args.edge, args.n_width, args.c_width, soc,
                    args.converter_hopping, interface, args.converter_shift)
            np.savez_compressed(output/case/'model.npz', ham_R=model.ham_R, iRvec=model.iRvec, rList=model.rList,
                lattice=model.lattice, spin=model.spin, orbital=model.orbital, soc_R=model.soc_R,
                ham0_R=model.ham0_R, exchange_R=model.exchange_R, atom_rows=groups, regions=regions)
            if case == 'full':
                plot_geometry(model, groups, regions, output)
            for grid in grids:
                start = perf_counter()
                rows = None
                if grid == grids[-1] and (args.row_profiles or args.profile_drive_rows is not None):
                    rows = args.profile_drive_rows or list(range(len(regions)))
                result = calculate(model, groups, regions, grid, args.fermi, args.kbt, args.linewidth, args.workers, rows)
                if rows is not None:
                    row_checks[case+'_orbital_balance'] = orbital_balance_residual(result, 'row_profile_')
                    row_arrays.update({case+'_'+key: value for key, value in result.items() if key.startswith('row_profile_')})
                    row_checks[case+'_rate_sum'] = float(max(np.max(np.abs(
                        result['row_profile_total_torque'+suffix]-result['row_profile_flux_torque'+suffix]
                        -result['row_profile_soc_torque'+suffix])) for suffix in ('', '_I', '_E')))
                    for name in OBSERVABLES:
                        matrix = result['row_profile_'+name]
                        row_checks[case+'_'+name+'_C_sum'] = float(np.max(np.abs(matrix[:, regions=='C'].sum(axis=1)-result['drive_'+name][rows,:,2])))
                        if len(rows)==len(regions):
                            row_checks[case+'_'+name+'_drive_sum'] = float(np.max(np.abs(matrix.sum(axis=0)-result[name][:,:,2])))
                    result = {key:value for key,value in result.items() if not key.startswith('row_profile_')}
                timings[f'{case}_n{grid}'] = perf_counter()-start
                arrays.update({f'{case}_{key}_n{grid}': value for key, value in result.items()})
                print(f'{case} Nk={grid}, workers={mesh_execution(grid,args.workers)[1]}, {timings[f"{case}_n{grid}"]:.2f}s: max C spin z={np.max(np.abs(result["spin"][regions=="C", :, 2])):.8g}', flush=True)
            arrays.update({f'{case}_{key}': value for key, value in result.items()})
            metrics[case] = {name+'_drive_sum_residual': float(np.max(np.abs(result['drive_'+name].sum(axis=0)-result[name][regions == 'C'].sum(axis=0)))) for name in OBSERVABLES}
            metrics[case]['rate_sum_residual'] = float(max(np.max(np.abs(result[prefix+'total_torque'+suffix]-result[prefix+'flux_torque'+suffix]-result[prefix+'soc_torque'+suffix])) for prefix,suffix in [('', ''), ('', '_I'), ('', '_E'), ('drive_', '')]))
            metrics[case]['orbital_soc_exchange_residual'] = float(np.max(np.abs(result['soc_torque']+2*result['orbital_soc'])))
            metrics[case]['orbital_rate_sum_residual'] = float(np.max(np.abs(result['orbital_total']-result['orbital_h0']-result['orbital_soc'])))
            metrics[case]['orbital_rate_sum_residual'] = max(metrics[case]['orbital_rate_sum_residual'], orbital_balance_residual(result))
            metrics[case]['orbital_transfer_global_residual'] = float(np.max(np.abs(result['orbital_transfer'].sum(axis=0))))
            metrics[case]['flux_global_residual'] = float(np.max(np.abs(result['flux_torque'].sum(axis=0))))
            metrics[case]['max_inplane'] = float(max(np.max(np.abs(result[key][..., :2])) for key in result))
            metrics[case]['C_total_z_by_fermi'] = {name: result[name][regions == 'C', :, 2].sum(axis=0).tolist() for name in OBSERVABLES}
            metrics[case]['N_drives_C_total_z_by_fermi'] = {name: result['drive_'+name][regions == 'N', :, 2].sum(axis=0).tolist() for name in OBSERVABLES}
            if len(grids)>1:
                metrics[case]['mesh_max_difference'] = {key: float(np.max(np.abs(result[key]-arrays[f'{case}_{key}_n{grids[-2]}']))) for key in result}
    arrays.update(y_A=y, regions=regions)
    checks = dict(orbital_balance=all(row[key]<1e-8 for row in metrics.values() for key in ('orbital_soc_exchange_residual','orbital_rate_sum_residual','orbital_transfer_global_residual')),
        rate_sum=all(row['rate_sum_residual']<1e-8 for row in metrics.values()),
        flux_global_zero=all(row['flux_global_residual']<1e-8 for row in metrics.values()),
        mirror_inplane_zero=all(row['max_inplane']<1e-8 for row in metrics.values()),
        drive_rows_sum=all(row[name+'_drive_sum_residual']<1e-8 for row in metrics.values() for name in OBSERVABLES),
        soc_torque_N_zero=bool(np.max(np.abs(arrays['full_soc_torque'][regions=='N']))<1e-8),
        no_soc_torque_zero=bool(np.max(np.abs(arrays['no_soc_soc_torque']))<1e-8),
        no_soc_spin_zero=bool(np.max(np.abs(arrays['no_soc_spin']))<1e-8),
        disconnected_C_zero=all(np.max(np.abs(arrays['disconnected_'+name][regions=='C']))<1e-8 and np.max(np.abs(arrays['disconnected_drive_'+name]))<1e-8 for name in OBSERVABLES))
    summary = dict(edge=args.edge, parameters={k: v for k,v in vars(args).items() if k!='output'},
        execution={str(grid): dict(zip(('chunks', 'effective_workers'), mesh_execution(grid,args.workers))) for grid in grids}, timings_seconds=timings,
        metrics=metrics, checks=checks, profile_axes=['observed_row','fermi','moment_xyz'], drive_axes=['driven_row','fermi','moment_xyz'],
        row_drive='symmetrized {P_row,v_x}/2; half of cross-row bonds at each endpoint; NOT strictly confined P_row v_x P_row',
        flux_observable='i[H0,{P_row,mu_S}/2]; positive means spin-moment influx',
        total_observable='i[H,{P_row,mu_S}/2]; Hamiltonian rate excluding collision and drive',
        soc_torque_observable='i[H_SOC,mu_S] = hbar*d(mu_S)/dt from SOC; electronic torque, no reaction sign',
        units={name: ('mu_B*eV/(V/Angstrom)' if name!='spin' else 'mu_B/(V/Angstrom)')+' per longitudinal repeat' for name in OBSERVABLES})
    np.savez_compressed(output/'responses.npz', **arrays)
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    plot(arrays, summary, output)
    export_csv(arrays, output)
    save_orbital_cuts(arrays, output)
    print(json.dumps(summary, indent=2))
    if not all(checks.values()):
        raise RuntimeError('NC validation failed; inspect summary.json')
    if args.row_profiles or args.profile_drive_rows is not None:
        save_row_profiles(row_arrays, y, regions, args.fermi, rows, output,
            dict(parameters=summary['parameters'], grid=grids[-1], workers=args.workers,
                 axes=['selected_drive_row','observed_row','fermi'], component='z',
                 units=summary['units'], drive_rows=rows, residuals=row_checks,
                 projection=summary['row_drive']))
        if max(row_checks.values())>1e-8:
            raise RuntimeError('Row-profile reconstruction failed')


if __name__ == '__main__':
    main()
