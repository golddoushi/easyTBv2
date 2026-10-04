"""Step 3: OHC source N plus atomic-SOC converter C, periodic x and finite y.

python examples/orbital_torque/orbital_nc_ribbon.py --workers 8
Compare N/C widths with their interface rows aligned.
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
    model.getSOCR('explicit', soc_R=soc_R)
    model.getH0R()
    model.exchange_R=np.zeros_like(model.ham_R)
    model._refreshHamiltonianDerivedMatrices()
    lookup = {tuple(r): i for i, r in enumerate(model.iRvec)}
    for i, r in enumerate(model.iRvec):
        np.testing.assert_allclose(model.ham_R[i], model.ham_R[lookup[tuple(-r)]].conj().T, atol=1e-14)
    regions = np.array(['N']*n_width + ['C']*c_width)
    return model, regions, soc_R


def main(argv=None):
    from examples.orbital_torque_hex.orbital_nc_ribbon import calculate, OBSERVABLES
    from examples.orbital_torque_hex.orbital_nbc_ribbon import diagnostics, plot_comparisons
    import csv
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--widths',nargs=2,type=int,action='append',metavar=('N','C'),help='Repeat width pairs in physical rows')
    parser.add_argument('--output',type=Path,default=Path('tmp/orbital_torque/nc_ribbon'))
    parser.add_argument('--grids',type=int,nargs='+',default=[256,512,1024])
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--fermi',type=float,nargs='+',default=[.8])
    parser.add_argument('--soc',type=float,default=.4)
    parser.add_argument('--tsp',type=float,default=.7)
    parser.add_argument('--converter-hopping',type=float,default=.5)
    parser.add_argument('--interface-hopping',type=float,default=.3)
    parser.add_argument('--converter-shift',type=float,default=0.)
    parser.add_argument('--kbt',type=float,default=.05)
    parser.add_argument('--linewidth',type=float,default=.05)
    parser.add_argument('--row-profiles',action='store_true')
    parser.add_argument('--plot-only',action='store_true')
    args=parser.parse_args(argv);out=args.output
    if args.plot_only:
        plot_comparisons(out,json.loads((out/'summary.json').read_text()));return
    widths=args.widths or [[8,6],[12,6],[8,12]]
    if any(min(pair)<1 for pair in widths) or len(set(map(tuple,widths)))!=len(widths) or min(args.grids)<4 or args.workers<1 or min(args.kbt,args.linewidth)<=0 or not np.all(np.isfinite([args.soc,args.tsp,args.converter_hopping,args.interface_hopping,args.converter_shift,args.kbt,args.linewidth,*args.fermi])):
        parser.error('Require distinct positive widths, grids>=4, workers>=1, finite parameters and positive kBT/linewidth')
    out.mkdir(parents=True,exist_ok=True)
    summary=dict(parameters={k:v for k,v in vars(args).items() if k!='output'},cases=[],
        alignment='Last N row -1, first C row 0; no interpolation',
        row_drive='symmetrized velocity {P_row,v_x}/2')
    summary['parameters']['widths']=widths
    for n,c in widths:
        name=f'n{n}_c{c}';directory=out/name
        with contextlib.redirect_stdout(io.StringIO()):
            model,regions,_=build_nc_ribbon(directory,n,c,args.tsp,args.soc,args.converter_hopping,args.interface_hopping,args.converter_shift)
        pos=np.asarray(model.posinfo['pos ions (cart.)'])
        y,groups=np.unique(np.round(pos[:,1],8),return_inverse=True)
        np.savez_compressed(directory/'model.npz',ham_R=model.ham_R,ham0_R=model.ham0_R,soc_R=model.soc_R,exchange_R=model.exchange_R,iRvec=model.iRvec,rList=model.rList,lattice=model.lattice,atom_rows=groups,regions=regions)
        case=dict(name=name,label=f'N={n}, C={c} rows',checks={},timings_seconds={},mesh_differences={})
        previous=None
        for grid in sorted(set(args.grids)):
            start=perf_counter()
            rows=list(range(len(y))) if args.row_profiles and grid==max(args.grids) else None
            result=calculate(model,groups,regions,grid,args.fermi,args.kbt,args.linewidth,args.workers,rows)
            case['checks'][str(grid)]=diagnostics(result,regions)
            case['timings_seconds'][str(grid)]=perf_counter()-start
            if previous is not None:
                case['mesh_differences'][str(grid)]={k:float(np.max(abs(result[k]-v))) for k,v in previous.items()}
            previous={k:v for k,v in result.items() if not k.startswith('row_profile_')}
            np.savez_compressed(directory/f'responses_n{grid}.npz',**result,y_A=y,regions=regions,fermi_eV=args.fermi)
            print(f'{name} Nk={grid}: {case["timings_seconds"][str(grid)]:.2f}s',flush=True)
            if max(case['checks'][str(grid)].values())>1e-8:raise RuntimeError('Continuity/projection checks failed')
        cuts=-np.cumsum(result['orbital_transfer'],axis=0)[:-1]
        np.savez_compressed(directory/'responses.npz',**result,y_A=y,regions=regions,fermi_eV=args.fermi,orbital_cut_current=cuts,cut_y_A=(y[:-1]+y[1:])/2)
        for prefix,file in [('', 'profiles.csv'),('drive_','drive_rows.csv')]:
            with (directory/file).open('w',newline='') as f:
                writer=csv.writer(f);writer.writerow(['row','region','y_A','fermi_eV',*OBSERVABLES])
                for r in range(len(y)):
                    for fi,mu in enumerate(args.fermi):writer.writerow([r,regions[r],y[r],mu,*[result[prefix+k][r,fi,2] for k in OBSERVABLES]])
        case['regional_totals_z']={region:{key:result[key][regions==region,:,2].sum(axis=0).tolist() for key in OBSERVABLES} for region in ('N','C')}
        summary['cases'].append(case)
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plot_comparisons(out,summary)


if __name__=='__main__':
    main()
