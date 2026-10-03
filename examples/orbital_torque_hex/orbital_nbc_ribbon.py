"""N / SOC-free p-shell buffer B / p-shell SOC converter C width comparisons.

python examples/orbital_torque_hex/orbital_nbc_ribbon.py --b-width 3 --widths 4 4 --widths 6 8
Width pairs are N transverse cells and C physical rows; B width is shared.
Native row coordinates align B and both interfaces without interpolation.
"""
import argparse
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
from time import perf_counter
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.orbital_torque_hex.orbital_nc_ribbon import (
    build_nc_ribbon, calculate, OBSERVABLES, PLOT_OBSERVABLES,
    observable_label, observable_units, orbital_balance_residual,
    spin_continuity_residual, orbital_continuity_residual)


def build_nbc_ribbon(directory, edge='zigzag', n_width=4, b_width=3, c_width=4,
                     soc=.01, converter_hopping=.5, interface_hopping=.4,
                     converter_shift=0.):
    """Identical scalar p-shell H0 in B/C; only the onsite SOC distinguishes C."""
    if min(b_width, c_width) < 1:
        raise ValueError('B and C widths must be positive')
    model, y, groups, regions = build_nc_ribbon(directory, edge, n_width,
        b_width+c_width, 0., converter_hopping, interface_hopping, converter_shift)
    p_rows = np.flatnonzero(regions == 'C')
    regions[p_rows[:b_width]] = 'B'
    soc_r = np.zeros_like(model.ham_R)
    zero = model.iRvecIndex[(0, 0, 0)]
    for atom, row in enumerate(groups):
        if regions[row] != 'C':
            continue
        ids = np.flatnonzero(model.iorb2iatomMap == atom)
        orbital = model.orbital[:, ids][:, :, ids]
        spin = model.spin[:, ids][:, :, ids]
        soc_r[zero][np.ix_(ids, ids)] = soc/2*sum(orbital[a]@spin[a] for a in range(3))
    model.ham_R += soc_r
    model.getSOCR('explicit', soc_R=soc_r)
    model.getH0R()
    model.exchange_R = np.zeros_like(model.ham_R)
    model._refreshHamiltonianDerivedMatrices()
    np.testing.assert_allclose(model.ham_R, model.ham0_R+model.soc_R, atol=1e-14)
    return model, y, groups, regions


def alignment(region_lists):
    """Align native row indices about a fixed B; absent outer rows stay empty."""
    counts = [tuple(np.count_nonzero(r == label) for label in ('N','B','C'))
              for r in region_lists]
    if not counts or any(min(c) < 1 for c in counts):
        raise ValueError('Every case must contain N, B, and C rows')
    if len({c[1] for c in counts}) != 1:
        raise ValueError('All comparison cases must have the same B width')
    b = counts[0][1]
    x = np.arange(-max(c[0] for c in counts), b+max(c[2] for c in counts))
    labels = np.where(x < 0, 'N', np.where(x < b, 'B', 'C'))
    weights, missing = [], []
    for regions, (n, _, c) in zip(region_lists, counts):
        native = np.arange(-n, b+c)
        if not np.array_equal(regions, np.repeat(['N','B','C'], [n,b,c])):
            raise ValueError('Physical rows must be ordered N, B, C')
        indices = np.searchsorted(x, native)
        w = np.zeros((len(x), len(native)))
        w[indices, np.arange(len(native))] = 1.
        weights.append(w)
        missing.append(~np.isin(x, native))
    return x, labels, np.array([-.5, b-.5]), weights, missing


def stored_observables(result):
    return OBSERVABLES + (('bsr',) if 'bsr' in result else ())


def diagnostics(result, regions):
    checks = dict(spin_continuity=spin_continuity_residual(result),
        orbital_continuity=orbital_continuity_residual(result),
        orbital_balance=orbital_balance_residual(result),
        drive_spin_continuity=spin_continuity_residual(result, 'drive_', ('',)),
        drive_orbital_continuity=orbital_continuity_residual(result, 'drive_', ('',)),
        soc_outside_C=float(np.max(np.abs(result['soc_torque'][regions!='C']))))
    for name in stored_observables(result):
        checks[name+'_drive_sum'] = float(np.max(np.abs(
            result['drive_'+name].sum(axis=0)-result[name][regions=='C'].sum(axis=0))))
        if 'row_profile_'+name in result:
            matrix = result['row_profile_'+name]
            checks[name+'_matrix_drive_sum'] = float(np.max(np.abs(matrix.sum(axis=0)-result[name][:,:,2])))
            checks[name+'_matrix_C_sum'] = float(np.max(np.abs(matrix[:,regions=='C'].sum(axis=1)-result['drive_'+name][:,:,2])))
    if 'row_profile_spin' in result:
        checks['matrix_spin_continuity'] = spin_continuity_residual(result, 'row_profile_')
        checks['matrix_orbital_continuity'] = orbital_continuity_residual(result, 'row_profile_')
    return checks


def save_physical(output, result, y, regions, fermi):
    """All scientific data remain on the native grid, including cut currents."""
    names = stored_observables(result)
    cuts = -np.cumsum(result['orbital_transfer'], axis=0)[:-1]
    data = dict(result, y_A=y, regions=regions, fermi_eV=fermi,
        orbital_cut_current=cuts, cut_y_A=(y[:-1]+y[1:])/2)
    if 'row_profile_orbital_transfer' in result:
        data['row_orbital_cut_current'] = -np.cumsum(result['row_profile_orbital_transfer'], axis=1)[:,:-1]
    np.savez_compressed(output/'responses.npz', **data)
    for prefix, filename in [('', 'profiles.csv'), ('drive_', 'drive_rows.csv')]:
        with (output/filename).open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['physical_row','y_A','region','fermi_eV',*names])
            for r in range(len(y)):
                for fi, mu in enumerate(fermi):
                    writer.writerow([r,y[r],regions[r],mu,*[result[prefix+k][r,fi,2] for k in names]])
    # Interval balance for the SOC-free buffer, using native boundary cuts.
    first, last = np.flatnonzero(regions=='B')[[0,-1]]
    # Include the N/B crystal contribution on B's first row. Internal B cuts
    # are clean transport cuts, but the left boundary here is the s-p interface.
    balance = cuts[first-1]-cuts[last]+sum(result[key][regions=='B'].sum(axis=0)
        for key in ('orbital_source','orbital_crystal','orbital_soc'))-result['orbital_relaxation'][regions=='B'].sum(axis=0)
    return float(np.max(np.abs(balance)))


def plot_comparisons(output, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cases = summary['cases']
    data = []
    for case in cases:
        with np.load(output/case['name']/'responses.npz') as saved:
            data.append(dict(saved))
    plot_names = PLOT_OBSERVABLES + (('bsr',) if all('bsr' in a for a in data) else ())
    panel_rows = (len(plot_names)+2)//3
    x, regions, boundaries, weights, missing = alignment([a['regions'] for a in data])
    display = dict(x=x, regions=regions, interfaces=boundaries,
                   alignment='native rows; B starts at zero; absent outer rows are NaN')
    for case, w, mask in zip(cases, weights, missing):
        display[case['name']+'_weights'] = w
        display[case['name']+'_missing'] = mask
    def decorate(ax):
        for region, color in [('N','steelblue'),('B','green'),('C','orange')]:
            xx = x[regions==region]
            ax.axvspan(xx[0]-.5, xx[-1]+.5, color=color, alpha=.07)
        for boundary in boundaries:
            ax.axvline(boundary, color='.4', ls='--', lw=.7)
        ax.axhline(0, color='.5', lw=.5)
        ax.grid(alpha=.15)
    row_output = output/'row_profiles'
    has_rows = all('row_profile_spin' in a for a in data)
    if has_rows:
        row_output.mkdir(exist_ok=True)
    for fi, mu in enumerate(summary['parameters']['fermi']):
        for prefix, filename in [('', 'profile'), ('drive_', 'drive_rows')]:
            fig, axes = plt.subplots(panel_rows,3,figsize=(18,4.25*panel_rows),constrained_layout=True)
            for ax, name in zip(axes.flat, plot_names):
                for case, a, w, mask in zip(cases, data, weights, missing):
                    values = w@a[prefix+name][:,fi,2]
                    values[mask] = np.nan
                    display[f'{case["name"]}_{prefix}{name}_f{fi}'] = values
                    line, = ax.plot(x,values,label=case['label'])
                    ax.plot(x[~mask],values[~mask],'o',ms=2,color=line.get_color())
                decorate(ax)
                ax.set_ylabel(('C-total ' if prefix else '')+observable_label(name))
                ax.set_xlabel('Aligned row coordinate (not distance); N / B / C')
            axes.flat[0].legend(fontsize=8)
            fig.suptitle(f'N/B/C width comparisons, EF={mu:g} eV; '+('C-total versus driving row' if prefix else 'uniform Ex')+'\nDots: native physical rows; B and both interfaces aligned')
            for unused in list(axes.flat)[len(plot_names):]:
                unused.set_visible(False)
            fig.savefig(output/f'{filename}_f{fi}.png',dpi=140)
            plt.close(fig)
        fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
        for case,a,mask in zip(cases,data,missing):
            native=x[~mask]
            ax.plot((native[:-1]+native[1:])/2,a['orbital_cut_current'][:,fi,2],'o-',ms=3,label=case['label'])
        decorate(ax)
        ax.set(xlabel='Aligned cut coordinate (not distance)',ylabel='Orbital cut current toward +y ['+observable_units('orbital_transfer')+']',title=f'Native cut currents, EF={mu:g} eV')
        ax.legend(fontsize=8)
        fig.savefig(output/f'orbital_cut_current_f{fi}.png',dpi=140)
        plt.close(fig)
        if not has_rows:
            continue
        matrices = [{name:w@a['row_profile_'+name][:,:,fi]@w.T for name in plot_names}
                    for a,w in zip(data,weights)]
        for matrix, mask in zip(matrices, missing):
            for values in matrix.values():
                values[mask, :] = np.nan
                values[:, mask] = np.nan
        # Consistent color scales across all width cases, for each observable.
        vmax = {name:max(1e-15,*(np.nanmax(abs(m[name])) for m in matrices)) for name in plot_names}
        for case, matrix in zip(cases, matrices):
            fig, axes = plt.subplots(panel_rows,3,figsize=(18,4.5*panel_rows),constrained_layout=True)
            for ax,name in zip(axes.flat,plot_names):
                im=ax.pcolormesh(x,x,matrix[name],shading='nearest',cmap='RdBu_r',vmin=-vmax[name],vmax=vmax[name])
                for boundary in boundaries:
                    ax.axvline(boundary,color='.4',ls='--',lw=.6)
                    ax.axhline(boundary,color='.4',ls='--',lw=.6)
                ax.set(title=name.replace('_',' '),xlabel='Aligned observing coordinate',ylabel='Aligned driving coordinate')
                fig.colorbar(im,ax=ax,label=observable_units(name))
                display[f'{case["name"]}_matrix_{name}_f{fi}']=matrix[name]
            fig.suptitle(case['label']+f', EF={mu:g} eV; native rows; blank areas lie outside this ribbon')
            for unused in list(axes.flat)[len(plot_names):]:
                unused.set_visible(False)
            fig.savefig(row_output/f'{case["name"]}_response_matrix_f{fi}.png',dpi=140)
            plt.close(fig)
        for d in range(len(x)):
            fig,axes=plt.subplots(panel_rows,3,figsize=(18,4.25*panel_rows),constrained_layout=True)
            for ax,name in zip(axes.flat,plot_names):
                for case,matrix,mask in zip(cases,matrices,missing):
                    if mask[d]:
                        continue
                    ax.plot(x,matrix[name][d],label=case['label'])
                decorate(ax)
                ax.axvline(x[d],color='purple',ls=':',lw=.7)
                ax.set_ylabel(observable_label(name))
                ax.set_xlabel('Aligned observing coordinate; N / B / C')
            axes.flat[0].legend(fontsize=7)
            fig.suptitle(f'Aligned drive {d}, {regions[d]}, x={x[d]:.3g}; EF={mu:g} eV\nOnly cases containing this physical driving row are shown')
            for unused in list(axes.flat)[len(plot_names):]:
                unused.set_visible(False)
            fig.savefig(row_output/f'profile_drive_node{d:03d}_f{fi}.png',dpi=140)
            plt.close(fig)
    np.savez_compressed(output/'aligned_display.npz',**display)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--b-width',type=int,default=3,help='Shared B width in physical rows')
    parser.add_argument('--widths',nargs=2,type=int,action='append',metavar=('N','C'),help='Repeat for width cases; N cells, C physical rows')
    parser.add_argument('--edge',choices=('zigzag','armchair'),default='zigzag')
    parser.add_argument('--grids',nargs='+',type=int,default=[512,1024])
    parser.add_argument('--fermi',nargs='+',type=float,default=[0.])
    parser.add_argument('--soc',type=float,default=.01)
    parser.add_argument('--converter-hopping',type=float,default=.5)
    parser.add_argument('--interface-hopping',type=float,default=.4)
    parser.add_argument('--converter-shift',type=float,default=0.)
    parser.add_argument('--kbt',type=float,default=.01)
    parser.add_argument('--linewidth',type=float,default=.02)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--bsr',action='store_true',help='Compute and plot BSR moment accumulation (additional cost)')
    parser.add_argument('--row-profiles',action='store_true')
    parser.add_argument('--plot-only',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    output=args.output or Path('tmp/orbital_torque_hex')/f'nbc_{args.edge}'
    if args.plot_only:
        plot_comparisons(output,json.loads((output/'summary.json').read_text()))
        return
    widths=args.widths or [[4,4],[6,4],[4,8],[6,8]]
    if (args.b_width<1 or any(n<2 or c<1 for n,c in widths)) or min(args.grids)<4 or args.workers<1 or min(args.kbt,args.linewidth)<=0:
        parser.error('Require N>=2, B/C>=1, grids>=4, workers>=1, positive kBT/linewidth')
    if len(set(map(tuple,widths)))!=len(widths) or not np.all(np.isfinite([args.soc,args.converter_hopping,args.interface_hopping,args.converter_shift,args.kbt,args.linewidth,*args.fermi])):
        parser.error('Use distinct N/C width pairs and finite parameters')
    output.mkdir(parents=True,exist_ok=True)
    summary=dict(parameters={k:v for k,v in vars(args).items() if k!='output'},cases=[],
        row_drive='velocity projection {P_row,v_x}/2',
        alignment='Fixed B, native row indices; both interfaces aligned; no interpolation',
        units={k:observable_units(k) for k in OBSERVABLES + (('bsr',) if args.bsr else ())})
    summary['parameters']['widths']=widths
    b=args.b_width
    for n,c in widths:
        name=f'n{n}_b{b}_c{c}'
        directory=output/name
        with contextlib.redirect_stdout(io.StringIO()):
            model,y,groups,regions=build_nbc_ribbon(directory,args.edge,n,b,c,args.soc,args.converter_hopping,args.interface_hopping,args.converter_shift)
        np.savez_compressed(directory/'model.npz',ham_R=model.ham_R,ham0_R=model.ham0_R,soc_R=model.soc_R,
            exchange_R=model.exchange_R,iRvec=model.iRvec,rList=model.rList,lattice=model.lattice,regions=regions,atom_rows=groups)
        case=dict(name=name,label=f'N={n} cells, B={b}, C={c} rows',physical_rows=len(y),checks={},timings_seconds={})
        previous=None
        grids=sorted(set(args.grids))
        for grid in grids:
            start=perf_counter()
            rows=list(range(len(y))) if args.row_profiles and grid==grids[-1] else None
            result=calculate(model,groups,regions,grid,args.fermi,args.kbt,args.linewidth,args.workers,rows,compute_bsr=args.bsr)
            case['timings_seconds'][str(grid)]=perf_counter()-start
            case['checks'][str(grid)]=diagnostics(result,regions)
            np.savez_compressed(directory/f'responses_n{grid}.npz',**result,y_A=y,regions=regions,fermi_eV=args.fermi)
            if previous is not None:
                case['mesh_max_difference']={key:float(np.max(abs(result[key]-value))) for key,value in previous.items()}
            previous={k:v for k,v in result.items() if not k.startswith('row_profile_')}
            print(f'{name} Nk={grid}: {case["timings_seconds"][str(grid)]:.2f}s',flush=True)
        case['buffer_cut_balance_residual']=save_physical(directory,result,y,regions,args.fermi)
        case['regional_totals_z']={region:{key:result[key][regions==region,:,2].sum(axis=0).tolist()
            for key in stored_observables(result)} for region in ('N','B','C')}
        summary['cases'].append(case)
        (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        residual=max(case['buffer_cut_balance_residual'],*(v for checks in case['checks'].values() for v in checks.values()))
        if residual>1e-8:
            raise RuntimeError(f'{name} validation failed: {residual}; inspect summary.json')
    plot_comparisons(output,summary)


if __name__=='__main__':
    main()
