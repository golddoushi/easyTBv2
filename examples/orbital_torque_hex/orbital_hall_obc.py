"""Direct-position orbital moment and dipole-driven Hall response of OBC flakes.

python examples/orbital_torque_hex/orbital_hall_obc.py --sizes 4 6 8 12
"""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys

import numpy as np
from scipy.constants import hbar, electron_mass, elementary_charge, angstrom
from scipy.special import expit
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.orbital_torque_hex.orbital_hall_source import build_model, calculate

CONSTANT = hbar**2/(2*electron_mass*elementary_charge*angstrom**2)


def build_flake(model, size):
    """One spin block of a size x size open parallelogram; no wrapped bonds.

    The source contains identical uncoupled spins. Diagonalizing one block
    saves cost; observables below include both copies explicitly by factor 2.
    """
    if size < 2:
        raise ValueError('OBC size must be >=2')
    np.testing.assert_allclose(model.ham_R[:, :2, :2], model.ham_R[:, 2:, 2:])
    np.testing.assert_array_equal(model.ham_R[:, :2, 2:], 0)
    np.testing.assert_array_equal(model.ham_R[:, 2:, :2], 0)
    cells = np.array([(x,y,0) for x in range(size) for y in range(size)])
    positions = ((cells@model.lattice)[:,None,:]+model.rList[:,:2].T[None,:,:]).reshape(-1,3)
    origin = positions.mean(axis=0)
    positions -= origin
    dim = len(positions)
    ham = np.zeros((dim,dim),complex)
    for ic, cell in enumerate(cells):
        for r, block in zip(model.iRvec, model.ham_R):
            target = cell+r
            if target[2] != 0 or np.any(target[:2]<0) or np.any(target[:2]>=size):
                continue
            jc = int(target[0]*size+target[1])
            ham[2*ic:2*ic+2,2*jc:2*jc+2] += block[:2,:2]
    return ham, positions, cells, origin


def prepare_flake(model, size, edge_layers=1):
    ham, pos, cells, origin = build_flake(model,size)
    energies, u = np.linalg.eigh(ham)
    # All velocities are hbar*v_phys, in eV Angstrom.
    velocity = -1j*(pos.T[:,:,None]-pos.T[:,None,:])*ham[None,:,:]
    x, y = pos[:,0], pos[:,1]
    vx, vy = velocity[:2]
    moment = -(x[:,None]*vy-y[:,None]*vx-vx*y[None,:]+vy*x[None,:])/(4*CONSTANT)
    currents = (moment[None,:,:]@velocity[:2]+velocity[:2]@moment[None,:,:])/2
    # Local partition of the same sample-centered current, not a bulk marker.
    mask_cells = np.all((cells[:,:2]>=edge_layers)&(cells[:,:2]<size-edge_layers),axis=1)
    mask = np.repeat(mask_cells,2).astype(float)
    interior = currents*(mask[None,:,None]+mask[None,None,:])/2
    r_band = u.conj().T@(pos.T[:,:,None]*u[None,:,:])
    v_band = u.conj().T@velocity@u
    m_band = u.conj().T@moment@u
    j_band = u.conj().T@currents@u
    interior_band = u.conj().T@interior@u
    gap = energies[:,None]-energies[None,:]
    commutator_error = float(np.max(abs(v_band-1j*gap*r_band)))
    area_cell = abs(np.linalg.det(model.lattice[:2,:2]))
    return dict(ham=ham, positions=pos, cells=cells, origin=origin, energies=energies,
                vectors=u, position_band=r_band, velocity_band=v_band, moment_band=m_band,
                current_band=j_band, interior_current_band=interior_band,
                area=area_cell*size**2, interior_area=area_cell*mask_cells.sum(),
                commutator_error=commutator_error)


def dipole_density_matrix(energies, position, fermi, kbt, linewidth):
    """Exact finite-system CRTA linear response for H'=E*x (normalized E).

    (epsilon_m-epsilon_n-i*eta) rho1_mn = (f_m-f_n) x_mn.
    No velocity substitution, inverse-gap cutoff, or periodic Drude channel.
    """
    f = expit((fermi-energies)/kbt)
    gap = energies[:,None]-energies[None,:]
    return (f[:,None]-f[None,:])*position/(gap-1j*linewidth)


def response(flake, fermi, kbt, linewidth):
    if kbt <= 0 or linewidth <= 0:
        raise ValueError('Positive kBT and linewidth required')
    e, r = flake['energies'], flake['position_band']
    gap = e[:,None]-e[None,:]
    sigma = np.zeros((len(fermi),2,2))
    inner = np.full_like(sigma,np.nan)
    accumulation = np.zeros((len(fermi),2))
    max_residual = 0.
    for i, ef in enumerate(fermi):
        f = expit((ef-e)/kbt)
        for drive in range(2):
            rho = dipole_density_matrix(e,r[drive],ef,kbt,linewidth)
            source_commutator = r[drive]*(f[None,:]-f[:,None])
            residual = -1j*gap*rho-1j*source_commutator-linewidth*rho
            max_residual = max(max_residual,float(np.max(abs(residual))))
            # Same native 2*pi/area normalization as the bulk example, and
            # explicit factor 2 for the identical spin copies.
            sigma[i,drive] = 4*np.pi/flake['area']*np.einsum('mn,anm->a',rho,flake['current_band']).real
            if flake['interior_area']>0:
                inner[i,drive] = 4*np.pi/flake['interior_area']*np.einsum('mn,anm->a',rho,flake['interior_current_band']).real
            accumulation[i,drive] = 2*np.einsum('mn,nm->',rho,flake['moment_band']).real
    return dict(sigma=sigma, interior_sigma=inner, moment_response=accumulation,
                continuity_residual=max_residual)


def plot_results(arrays, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ef = arrays['fermi_eV']
    fig, axes = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for size in arrays['sizes']:
        moment = arrays[f'moment_diag_n{size}']
        axes[0,0].scatter(arrays[f'energies_n{size}'],moment,s=6,label=f'OBC {size}x{size}')
        for eta in arrays['linewidths_eV']:
            key=f'n{size}_eta{eta:g}'
            label=f'{size}x{size}, eta={eta:g}'
            sig=arrays[f'sigma_{key}']
            axes[0,1].plot(ef,sig[:,0,1],label=label)
            axes[1,0].plot(ef,(sig[:,0,1]-sig[:,1,0])/2,label=label)
            axes[1,1].plot(ef,arrays[f'interior_sigma_{key}'][:,0,1],label=label)
    for eta in arrays['linewidths_eV']:
        for kind, style in [('bsr','--'),('bsr_v2',':')]:
            sig=arrays[f'bulk_{kind}_eta{eta:g}']
            axes[0,1].plot(ef,sig[:,0,1,2],style,color='black',label=f'PBC {kind}, eta={eta:g}')
            axes[1,0].plot(ef,(sig[:,0,1,2]-sig[:,1,0,2])/2,style,color='black',label=f'PBC {kind}, eta={eta:g}')
    axes[0,0].set(title='Finite eigenstate orbital moment about sample center',xlabel='Eigenenergy (eV)',ylabel='m_z (mu_B; one spin)')
    axes[0,1].set_title('Whole-sample response: Ex -> Jy of m_z')
    axes[1,0].set_title('Antisymmetric part: (sigma_xy - sigma_yx)/2')
    axes[1,1].set_title('Interior partition of sample-centered current')
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.axhline(0,color='grey',lw=.5)
        ax.legend(fontsize=7)
    for ax in (axes[0,1],axes[1,0],axes[1,1]):
        ax.set_xlabel('Fermi energy (eV)')
        ax.set_ylabel('Response (native 2*pi/area normalization)')
    fig.savefig(output/'obc_comparison.png',dpi=160)
    plt.close(fig)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('tmp/orbital_torque_hex/obc_source'))
    parser.add_argument('--sizes',type=int,nargs='+',default=[4,6,8,12])
    parser.add_argument('--bulk-grid',type=int,default=96)
    parser.add_argument('--edge-layers',type=int,default=1)
    parser.add_argument('--linewidths',type=float,nargs='+',default=[.02])
    parser.add_argument('--fermi',type=float,nargs='+',default=np.linspace(-3.2,3.2,81).tolist())
    for key, default in [('hopping',1.),('delta',.5),('t2',0.),('phase',0.),('kbt',.01),('lattice-constant',2.4)]:
        parser.add_argument('--'+key,type=float,default=default)
    parser.add_argument('--plot-only',action='store_true')
    args=parser.parse_args(argv)
    if args.plot_only:
        with np.load(args.output/'responses.npz') as arrays:
            plot_results(arrays,args.output)
        return
    numeric=[args.hopping,args.delta,args.t2,args.phase,args.kbt,args.lattice_constant,*args.linewidths,*args.fermi]
    if (not np.all(np.isfinite(numeric)) or min(args.sizes)<2 or args.bulk_grid<3
            or args.edge_layers<0 or min(args.linewidths)<=0 or args.kbt<=0 or args.lattice_constant<=0):
        parser.error('Finite parameters, sizes>=2, bulk grid>=3 and positive kBT/linewidth/lattice constant required')
    args.output.mkdir(parents=True,exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        model=build_model(args.output/'model',hopping=args.hopping,delta=args.delta,
                          lattice_constant=args.lattice_constant,t2=args.t2,phase=args.phase)
    arrays=dict(fermi_eV=np.asarray(args.fermi),sizes=np.array(sorted(set(args.sizes))),
                linewidths_eV=np.array(sorted(set(args.linewidths))))
    metrics=[]
    with threadpool_limits(limits=1):
        for eta in arrays['linewidths_eV']:
            bulk=calculate(model,args.bulk_grid,args.fermi,args.kbt,eta)
            arrays.update({f'bulk_{key}_eta{eta:g}':bulk[key] for key in ('bsr','bsr_v2','shc')})
        for size in arrays['sizes']:
            flake=prepare_flake(model,int(size),args.edge_layers)
            arrays[f'energies_n{size}']=flake['energies']
            arrays[f'moment_diag_n{size}']=np.diag(flake['moment_band']).real
            np.savez_compressed(args.output/f'flake_n{size}.npz',ham=flake['ham'],
                positions=flake['positions'],origin=flake['origin'],energies=flake['energies'],
                vectors=flake['vectors'],moment_band=flake['moment_band'])
            for eta in arrays['linewidths_eV']:
                result=response(flake,args.fermi,args.kbt,eta)
                key=f'n{size}_eta{eta:g}'
                arrays.update({f'{name}_{key}':value for name,value in result.items()})
                metrics.append(dict(size=int(size),linewidth=float(eta),
                    commutator_error=flake['commutator_error'],continuity_residual=result['continuity_residual']))
                np.savetxt(args.output/f'hall_{key}.csv',np.column_stack([args.fermi,
                    result['sigma'][:,0,1],result['sigma'][:,1,0],result['interior_sigma'][:,0,1],
                    result['moment_response'][:,0]]),delimiter=',',
                    header='fermi_eV,sigma_xy,sigma_yx,interior_sigma_xy,total_mz_response_Ex',comments='')
                print(json.dumps(metrics[-1]),flush=True)
    summary=dict(parameters={k:v for k,v in vars(args).items() if k!='output'},metrics=metrics,
        definition='Full sample-centered r-cross-v moment, no band-diagonal position subtraction; finite dipole driving and CRTA',
        spin_copies=2,normalization='2*pi/physical sample area times trace over both spins',
        limitations='Finite-flake DC response is not a lead-measured transport conductance. Compare size and linewidth limits; Chern zero does not guarantee equality with the PBC BSR observable.')
    np.savez_compressed(args.output/'responses.npz',**arrays)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plot_results(arrays,args.output)


if __name__=='__main__':
    main()
