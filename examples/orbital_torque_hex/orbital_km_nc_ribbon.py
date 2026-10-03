"""s-only honeycomb N / intrinsic Kane-Mele C ribbon, periodic x and open y."""
import argparse
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from system import System_tb
from examples.orbital_torque_hex.orbital_hall_source import build_model
from examples.orbital_torque_hex.orbital_nc_ribbon import calculate

NAMES = ('bsr', 'spin', 'soc_local_rate', 'soc_torque', 'flux_torque',
         'total_torque', 'spin_source', 'spin_relaxation')
LABELS = ('BSR moment', 'Spin moment', 'SOC hopping influx',
          'SOC spin conversion (zero for z)', 'H0 spin influx',
          'Full Hamiltonian local rate', 'Direct spin source', 'Spin relaxation')


def build_bulk(directory, hopping=1., soc=.1, delta=0.):
    """Intrinsic SOC i*lambda*nu_ij*sigma_z; nu is a two-bond turn sign.

    Construct NNN paths before cutting the lattice, retaining all bulk bonds.
    SOC parameter is the hopping amplitude, not the Dirac mass 3sqrt(3)lambda.
    """
    directory = Path(directory)
    base = build_model(directory, hopping=hopping, delta=delta)
    positions = base.rList[:, :2].T
    blocks = {tuple(r): h.copy() for r, h in zip(base.iRvec, base.ham_R)}
    links = []
    for r, h in zip(base.iRvec, base.ham_R):
        for i in range(2):
            for j in range(2):
                if i != j and abs(h[i,j]) > 1e-12:
                    links.append((i,j,r.copy()))
    for i,j,r in links:
        for j2,k,q in links:
            if j2 != j:
                continue
            shift = r+q
            if i == k and np.all(shift == 0):
                continue
            d1 = positions[j]+r@base.lattice-positions[i]
            d2 = positions[k]+q@base.lattice-positions[j]
            nu = np.sign(np.cross(d1,d2)[2])
            block = blocks.setdefault(tuple(shift), np.zeros((4,4),complex))
            block[i,k] += 1j*soc*nu
            block[i+2,k+2] -= 1j*soc*nu
    with (directory/'km_hr.dat').open('w') as f:
        f.write(f'Intrinsic Kane-Mele; eV\n4\n{len(blocks)}\n'+ ' '.join(['1']*len(blocks))+'\n')
        for r,h in blocks.items():
            for j in range(4):
                for i in range(4):
                    z=h[i,j]
                    f.write(f'{r[0]} {r[1]} {r[2]} {i+1} {j+1} {z.real:.16e} {z.imag:.16e}\n')
    model=System_tb(str(directory),hamR='km',projections={'A':[0],'B':[0]},ispin=2)
    model.verbose=False
    return model


def build_junction(directory, n_width=6, c_width=8, edge='zigzag', hopping=1.,
                   soc=.1, delta_n=.5, delta_c=0., interface_hopping=1.):
    if min(n_width,c_width)<1 or n_width+c_width<2:
        raise ValueError('Positive N/C widths required')
    # Use a nonzero temporary staggered potential to identify sublattices after
    # supercell construction; replace it by the actual regional potentials.
    model=build_bulk(directory,hopping,soc,1.)
    width=n_width+c_width
    if edge=='armchair':
        # Same rotation convention as the established armchair ribbon builder.
        # Rotate real-space vectors and lattice; rebuild primitive from files
        # with the rotated POSCAR before making its rectangular supercell.
        p=Path(directory)/'POSCAR'
        lines=p.read_text().splitlines()
        rot=np.array([[0.,1.,0.],[-1.,0.,0.],[0.,0.,1.]])
        lattice=model.lattice@rot.T
        lines[2:5]=[' '.join(map(str,row)) for row in lattice]
        p.write_text('\n'.join(lines)+'\n')
        model=System_tb(str(directory),hamR='km',projections={'A':[0],'B':[0]},ispin=2)
        model.verbose=False
    transform=[[1,2,0],[-width,0,0],[0,0,1]] if edge=='armchair' else np.diag([1,width,1])
    model.make_supercell(transform)
    if edge=='armchair':
        model.shiftOrigin(1,shift=.25/width)
    model.cutEdge(1)
    zero=model.iRvecIndex[(0,0,0)]
    model.AA_R_W[:]=0
    for i in range(model.norbs):
        model.AA_R_W[zero,:,i,i]=model.rList[:,i]
    half=model.norbs//2
    positions=model.rList[:,:half].T
    y,orb_rows=np.unique(np.round(positions[:,1],8),return_inverse=True)
    # Both geometries have two physical y rows per transverse cell (armchair
    # rows contain two atoms). Determine counts from the generated geometry.
    if len(y)%width:
        raise ValueError('Unexpected physical-row count')
    n_rows=n_width*(len(y)//width)
    regions=np.where(np.arange(len(y))<n_rows,'N','C')
    in_c=regions[orb_rows]=='C'
    sublattice=np.sign(np.diag(model.ham_R[zero])[:half].real)
    h0=(model.ham_R[:,:half,:half]+model.ham_R[:,half:,half:])/2
    so=(model.ham_R[:,:half,:half]-model.ham_R[:,half:,half:])/2
    so*=in_c[:,None]&in_c[None,:]
    # Sharp SOC boundary: retain NNN SOC only when both endpoints lie in C.
    cross=in_c[:,None]!=in_c[None,:]
    h0[:,cross]*=interface_hopping/hopping
    h0[zero,np.arange(half),np.arange(half)]=sublattice*np.where(in_c,delta_c,delta_n)
    ham=np.zeros_like(model.ham_R); sr=np.zeros_like(ham)
    ham[:,:half,:half]=h0+so; ham[:,half:,half:]=h0-so
    sr[:,:half,:half]=so; sr[:,half:,half:]=-so
    model.ham_R=ham
    model.getSOCR('explicit',soc_R=sr)
    model.getH0R(); model.exchange_R=np.zeros_like(ham)
    model._refreshHamiltonianDerivedMatrices()
    groups=np.empty(len(model.posinfo['pos ions (cart.)']),int)
    for atom in range(len(groups)):
        orbital=np.flatnonzero(model.iorb2iatomMap==atom)[0]
        groups[atom]=orb_rows[orbital%half]
    return model,y,groups,regions


def audit(result):
    checks={}
    for prefix,suffixes in [('',('', '_I','_E')),('drive_',('',)),('row_profile_',('', '_I','_E'))]:
        if prefix+'spin' not in result:
            continue
        for suffix in suffixes:
            def a(name): return result[prefix+name+suffix]
            # Independently contracted full rate, source and relaxation.
            residual=a('total_torque')+a('spin_source')-a('spin_relaxation')
            result[prefix+'spin_continuity'+suffix]=residual
            checks[prefix+'continuity'+suffix]=float(np.max(abs(residual)))
            checks[prefix+'rate_split'+suffix]=float(np.max(abs(a('total_torque')-a('flux_torque')-a('soc_local_rate'))))
            z=(slice(None),)*a('soc_torque').ndim if prefix=='row_profile_' else (...,2)
            checks[prefix+'z_conversion'+suffix]=float(np.max(abs(a('soc_torque')[z])))
    return checks


def plot(output, result, y, regions, fermi):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    boundary=(y[regions=='N'][-1]+y[regions=='C'][0])/2
    def panels(values,title,path,drive=None):
        fig,axes=plt.subplots(4,2,figsize=(12,14),constrained_layout=True)
        for ax,name,label in zip(axes.flat,NAMES[:-1],LABELS[:-1]):
            ax.plot(y,values[name],'o-',ms=3)
            ax.axvline(boundary,color='.4',ls='--'); ax.axhline(0,color='.5',lw=.5)
            if drive is not None: ax.axvline(y[drive],color='purple',ls=':')
            ax.set(title=label,xlabel='Physical y (Angstrom)',ylabel='mu_B/(V/Angstrom)' if name in ('spin','bsr') else 'mu_B eV/(V/Angstrom)')
            if name=='soc_torque': ax.set_ylim(-1e-10,1e-10)
        axes.flat[-1].set_visible(False)
        fig.suptitle(title);fig.savefig(path,dpi=140);plt.close(fig)
    for fi,mu in enumerate(fermi):
        for prefix,file,title in [('', 'profile','Uniform Ex'),('drive_','drive_rows','C total versus driving row')]:
            panels({k:result[prefix+k][:,fi,2] for k in NAMES},f'{title}; EF={mu:g} eV',output/f'{file}_f{fi}.png')
        if 'row_profile_spin' not in result: continue
        folder=output/'row_profiles';folder.mkdir(exist_ok=True)
        fig,axes=plt.subplots(4,2,figsize=(12,14),constrained_layout=True)
        for ax,name,label in zip(axes.flat,NAMES[:-1],LABELS[:-1]):
            a=result['row_profile_'+name][:,:,fi];lim=max(np.max(abs(a)),1e-15)
            if name=='soc_torque': lim=max(lim,1e-10)
            im=ax.pcolormesh(y,y,a,shading='nearest',cmap='RdBu_r',vmin=-lim,vmax=lim)
            ax.axvline(boundary,color='.4',ls='--');ax.axhline(boundary,color='.4',ls='--')
            ax.set(title=label,xlabel='Observing y (Angstrom)',ylabel='Driving y (Angstrom)')
            fig.colorbar(im,ax=ax)
        axes.flat[-1].set_visible(False)
        fig.suptitle(f'Row response matrices; EF={mu:g} eV');fig.savefig(folder/f'response_matrix_f{fi}.png',dpi=140);plt.close(fig)
        for d in range(len(y)):
            panels({k:result['row_profile_'+k][d,:,fi] for k in NAMES},f'Drive row {d} ({regions[d]}); EF={mu:g} eV',folder/f'profile_drive_row{d:03d}_f{fi}.png',d)


def reference_outputs(output, model, args):
    """Finite-ribbon spin-resolved bands and isolated bulk N/C Hall responses."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from examples.orbital_torque_hex.orbital_hall_source import calculate as bulk_response
    bulk={}
    for name,delta,soc in [('N',args.delta_n,0.),('C',args.delta_c,args.soc)]:
        with contextlib.redirect_stdout(io.StringIO()):
            reference=build_bulk(output/('bulk_'+name),args.hopping,soc,delta)
        values=bulk_response(reference,args.bulk_grid,args.fermi,args.kbt,args.linewidth)
        np.savez_compressed(output/('bulk_'+name+'.npz'),**values,fermi_eV=args.fermi)
        bulk[name]={k:values[k][:,0,1,2].tolist() for k in ('bsr','aca','shc')}
    fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    for ax,key in zip(axes,('bsr','aca','shc')):
        for name in ('N','C'):ax.plot(args.fermi,bulk[name][key],'o-',label=name)
        ax.set(title=key.upper()+' bulk Hall',xlabel='EF (eV)',ylabel='Native Hall normalization')
        ax.legend();ax.axhline(0,color='.5',lw=.5)
    fig.savefig(output/'bulk_hall.png',dpi=140);plt.close(fig)
    k=np.linspace(-.5,.5,241);half=model.norbs//2
    energies=[]; centers=[]
    fig,axes=plt.subplots(1,2,figsize=(11,5),constrained_layout=True,sharey=True)
    for spin,ax in enumerate(axes):
        block=model.ham_R[:,spin*half:(spin+1)*half,spin*half:(spin+1)*half]
        h=np.einsum('kr,rij->kij',np.exp(2j*np.pi*k[:,None]*model.iRvec[None,:,0]),block)
        e,u=np.linalg.eigh(h)
        y=model.rList[1,spin*half:(spin+1)*half]
        center=np.einsum('kin,i->kn',abs(u)**2,y)
        energies.append(e);centers.append(center)
        im=ax.scatter(np.repeat(k,half),e.ravel(),c=center.ravel(),s=2,cmap='viridis',vmin=y.min(),vmax=y.max())
        ax.set(title=('Spin up','Spin down')[spin],xlabel='k_x L_x / 2pi',ylabel='Energy (eV)')
        ax.set_ylim(-1.5,1.5);ax.axhline(0,color='.5',lw=.5)
    fig.colorbar(im,ax=list(axes),label='State mean y (Angstrom)')
    fig.savefig(output/'bands.png',dpi=140);plt.close(fig)
    np.savez_compressed(output/'bands.npz',k_fraction=k,energies=np.array(energies),mean_y_A=np.array(centers))
    return bulk


def aligned_rows(regions):
    """Keep all native rows and anchor the first C row at zero."""
    n=np.count_nonzero(regions=='N');c=np.count_nonzero(regions=='C')
    if min(n,c)<1 or not np.array_equal(regions,np.repeat(['N','C'],[n,c])):
        raise ValueError('Expected consecutive N then C physical rows')
    return np.arange(-n,c)


def plot_comparisons(output,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cases=summary['cases'];data=[]
    for case in cases:
        with np.load(output/case['name']/'responses.npz') as a:data.append(dict(a))
    coords=[aligned_rows(a['regions']) for a in data]
    x=np.arange(min(v[0] for v in coords),max(v[-1] for v in coords)+1)
    indices=[np.searchsorted(x,v) for v in coords]
    cache=dict(row_coordinate=x,interface=-.5)
    for case,ix in zip(cases,indices):
        cache[case['name']+'_present']=np.isin(np.arange(len(x)),ix)
    def decorate(ax):
        ax.axvline(-.5,color='.4',ls='--')
        ax.axhline(0,color='.5',lw=.5)
        ax.set_xlim(x[0]-.5,x[-1]+.5)
        ax.set_xlabel('Native row relative to first C row; N < 0, C >= 0')
    def panels(values,title,path,drive=None):
        fig,axes=plt.subplots(4,2,figsize=(13,14),constrained_layout=True)
        for ax,name,label in zip(axes.flat,NAMES[:-1],LABELS[:-1]):
            for case,series in zip(cases,values):
                if series is not None:ax.plot(x,series[name],'o-',ms=2,label=case['label'])
            decorate(ax)
            if drive is not None:ax.axvline(drive,color='purple',ls=':')
            ax.set_title(label)
            ax.set_ylabel('mu_B/(V/Angstrom)' if name in ('bsr','spin') else 'mu_B eV/(V/Angstrom)')
            if name=='soc_torque':ax.set_ylim(-1e-10,1e-10)
        axes.flat[0].legend(fontsize=8);axes.flat[-1].set_visible(False)
        fig.suptitle(title);fig.savefig(path,dpi=140);plt.close(fig)
    for fi,mu in enumerate(data[0]['fermi_eV']):
        for prefix,file,title in [('', 'profile','Uniform Ex'),('drive_','drive_rows','C total versus driving row')]:
            series=[]
            for case,a,ix in zip(cases,data,indices):
                values={}
                for name in NAMES[:-1]:
                    v=np.full(len(x),np.nan);v[ix]=a[prefix+name][:,fi,2];values[name]=v
                    cache[f'{case["name"]}_{prefix}{name}_f{fi}']=v
                series.append(values)
            panels(series,f'{title}; EF={mu:g} eV; interface aligned, no interpolation',output/f'{file}_f{fi}.png')
        if not all('row_profile_spin' in a for a in data):continue
        folder=output/'row_profiles';folder.mkdir(exist_ok=True)
        matrices=[]
        for case,a,ix in zip(cases,data,indices):
            values={}
            for name in NAMES[:-1]:
                v=np.full((len(x),len(x)),np.nan)
                v[np.ix_(ix,ix)]=a['row_profile_'+name][:,:,fi];values[name]=v
                cache[f'{case["name"]}_matrix_{name}_f{fi}']=v
            matrices.append(values)
        limits={name:max(1e-10 if name=='soc_torque' else 1e-15,*(np.nanmax(abs(m[name])) for m in matrices)) for name in NAMES[:-1]}
        for case,matrix in zip(cases,matrices):
            fig,axes=plt.subplots(4,2,figsize=(13,14),constrained_layout=True)
            for ax,name,label in zip(axes.flat,NAMES[:-1],LABELS[:-1]):
                im=ax.pcolormesh(x,x,matrix[name],shading='nearest',cmap='RdBu_r',vmin=-limits[name],vmax=limits[name])
                ax.axvline(-.5,color='.4',ls='--');ax.axhline(-.5,color='.4',ls='--')
                ax.set(title=label,xlabel='Observing row relative to C',ylabel='Driving row relative to C')
                fig.colorbar(im,ax=ax)
            axes.flat[-1].set_visible(False)
            fig.suptitle(case['label']+f'; EF={mu:g} eV; blank = outside ribbon')
            fig.savefig(folder/f'{case["name"]}_response_matrix_f{fi}.png',dpi=140);plt.close(fig)
        for d,row in enumerate(x):
            values=[{name:m[name][d] for name in NAMES[:-1]} if d in ix else None for m,ix in zip(matrices,indices)]
            panels(values,f'Drive row {row} relative to C; EF={mu:g} eV; only existing rows',folder/f'profile_drive_row{row:+04d}_f{fi}.png',row)
    np.savez_compressed(output/'aligned_display.npz',**cache)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--widths',nargs=2,type=int,action='append',metavar=('N','C'),help='Repeat for interface-aligned width comparisons (transverse cells)')
    p.add_argument('--n-width',type=int,default=6);p.add_argument('--c-width',type=int,default=8)
    p.add_argument('--edge',choices=['zigzag','armchair'],default='zigzag')
    p.add_argument('--soc',type=float,default=.1);p.add_argument('--hopping',type=float,default=1.)
    p.add_argument('--delta-n',type=float,default=.5);p.add_argument('--delta-c',type=float,default=0.)
    p.add_argument('--interface-hopping',type=float,default=1.)
    p.add_argument('--grids',nargs='+',type=int,default=[256,512]);p.add_argument('--fermi',nargs='+',type=float,default=[0.])
    p.add_argument('--workers',type=int,default=1);p.add_argument('--kbt',type=float,default=.01);p.add_argument('--linewidth',type=float,default=.02)
    p.add_argument('--bulk-grid',type=int,default=48)
    p.add_argument('--row-profiles',action='store_true');p.add_argument('--plot-only',action='store_true');p.add_argument('--output',type=Path)
    args=p.parse_args();out=args.output or Path('tmp/orbital_torque_hex')/('km_nc_'+args.edge)
    if args.plot_only:
        summary=json.loads((out/'summary.json').read_text())
        if 'cases' in summary:
            plot_comparisons(out,summary);return
        with np.load(out/'responses.npz') as a: data=dict(a)
        plot(out,data,data['y_A'],data['regions'],data['fermi_eV']);return
    if min(args.n_width,args.c_width,args.workers)<1 or min(args.grids)<4 or args.bulk_grid<4 or min(args.hopping,args.kbt,args.linewidth)<=0 or not np.all(np.isfinite([args.soc,args.hopping,args.delta_n,args.delta_c,args.interface_hopping,args.kbt,args.linewidth,*args.fermi])):
        p.error('Require positive widths/workers/hopping/kBT/linewidth, grid >=4 and finite parameters')
    widths=args.widths or [[args.n_width,args.c_width]]
    if any(min(pair)<1 for pair in widths) or len(set(map(tuple,widths)))!=len(widths):
        p.error('Use positive, distinct N/C width pairs')
    if args.widths is None:
        run_case(args,out)
        return
    import copy
    out.mkdir(parents=True,exist_ok=True)
    comparison=dict(parameters={k:v for k,v in vars(args).items() if k!='output'},cases=[],
                    alignment='Native rows: N ends at -1, C starts at 0; interface at -0.5; no interpolation')
    for n,c in widths:
        case_args=copy.copy(args);case_args.n_width=n;case_args.c_width=c
        name=f'n{n}_c{c}'
        print(f'Running {name}',flush=True)
        run_case(case_args,out/name)
        comparison['cases'].append(dict(name=name,label=f'N={n}, C={c} cells'))
        (out/'summary.json').write_text(json.dumps(comparison,indent=2)+'\n')
    plot_comparisons(out,comparison)


def run_case(args,out):
    out.mkdir(parents=True,exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        model,y,groups,regions=build_junction(out/'model',args.n_width,args.c_width,args.edge,args.hopping,args.soc,args.delta_n,args.delta_c,args.interface_hopping)
    np.savez_compressed(out/'model.npz',ham_R=model.ham_R,soc_R=model.soc_R,ham0_R=model.ham0_R,iRvec=model.iRvec,lattice=model.lattice,rList=model.rList,atom_rows=groups,regions=regions)
    summary=dict(parameters={k:v for k,v in vars(args).items() if k!='output'},checks={},mesh_differences={},
        bulk_C_topological=bool(abs(args.delta_c)<3*np.sqrt(3)*abs(args.soc)),
        soc_convention='i lambda nu sigma_z; endpoints in C only; no Rashba',
        projection='symmetrized row moment; row drive {P,vx}/2',
        caveat='BSR is a fixed full-junction observable, not an independently injected source; ACA=0 does not prove BSR conversion')
    summary['bulk_Hall_Ex_Jy_z']=reference_outputs(out,model,args)
    previous=None
    for grid in sorted(set(args.grids)):
        rows=list(range(len(y))) if args.row_profiles and grid==max(args.grids) else None
        result=calculate(model,groups,regions,grid,args.fermi,args.kbt,args.linewidth,args.workers,rows,compute_bsr=True,compute_soc_local=True)
        checks=audit(result)
        for key in NAMES:
            checks[key+'_drive_sum']=float(np.max(abs(result['drive_'+key].sum(axis=0)-result[key][regions=='C'].sum(axis=0))))
            if rows is not None:
                checks[key+'_matrix_drive_sum']=float(np.max(abs(result['row_profile_'+key].sum(axis=0)-result[key][:,:,2])))
                checks[key+'_matrix_C_sum']=float(np.max(abs(result['row_profile_'+key][:,regions=='C'].sum(axis=1)-result['drive_'+key][:,:,2])))
        summary['checks'][str(grid)]=checks
        if previous is not None:
            summary['mesh_differences'][str(grid)]={k:float(np.max(abs(result[k]-v))) for k,v in previous.items()}
        previous={k:v for k,v in result.items() if not k.startswith('row_profile_')}
        np.savez_compressed(out/f'responses_n{grid}.npz',**result,y_A=y,regions=regions,fermi_eV=args.fermi)
        print(f'Nk={grid}; maximum balance/sum residual {max(checks.values()):.3e}',flush=True)
        if max(checks.values())>1e-8: raise RuntimeError('Continuity or projection check failed')
    np.savez_compressed(out/'responses.npz',**result,y_A=y,regions=regions,fermi_eV=args.fermi)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    for prefix,file in [('', 'profiles.csv'),('drive_','drive_rows.csv')]:
        with (out/file).open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['row','region','y_A','fermi_eV',*NAMES])
            for r in range(len(y)):
                for fi,mu in enumerate(args.fermi):writer.writerow([r,regions[r],y[r],mu,*[result[prefix+k][r,fi,2] for k in NAMES]])
    plot(out,result,y,regions,args.fermi)
    return summary


if __name__=='__main__':
    main()
