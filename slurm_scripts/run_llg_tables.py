"""Shard Table S4/S5 ensembles over Slurm nodes and collect Origin tables."""
import argparse
import csv
import math
import os
from pathlib import Path
import subprocess
import sys


def cases(table):
    if table == 'S4':
        return [(a,f,100.) for a in [.001,.003,.006,.010,.012,.020] for f in [.25,.5,.75,1.]]
    return [(a,1.,s) for s in [50.,75.,100.,125.,150.] for a in [.001,.003,.006]]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('table',choices=['S4','S5'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data',default='mcalc/MnBi2Te4')
    p.add_argument('--workers',type=int,default=96)
    p.add_argument('--samples',type=int,default=300)
    p.add_argument('--fine-step-fs',type=float,default=500.)
    p.add_argument('--coarse-step-fs',type=float,default=1000.)
    p.add_argument('--first-center-fs',type=float,default=14400.)
    p.add_argument('--relax-ps',type=float,default=2199.9)
    p.add_argument('--merge',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    jobs=cases(args.table)
    if not args.merge:
        rank=int(os.environ.get('SLURM_PROCID',0))
        ranks=int(os.environ.get('SLURM_NTASKS',1))
        for index,(alpha,fluence,sep) in enumerate(jobs):
            if index % ranks != rank: continue
            out=args.output/f'case_{index:02d}'
            end=args.first_center_fs+29*sep*1000+args.relax_ps*1000
            cmd=[sys.executable,'-u','LLGInput.py','--ensemble',
                 '--data',args.data,'--output',str(out),'--workers',str(args.workers),
                 '--samples',str(args.samples),'--temperatures','11.2','--omegas','0.75',
                 '--temperature-seed-index','1','--seed','20261105',
                 '--matrix-entry','1','--xy-plus-target','1','--cells','625',
                 '--anisotropy-mev','0.2','--alpha',str(alpha),
                 '--field-amplitude',str(.005*math.sqrt(fluence)),
                 '--fwhm-fs','7200','--pulses','30',
                 '--first-center-fs',str(args.first_center_fs),
                 '--separation-fs',str(sep*1000),'--total-time-fs',str(end),
                 '--fine-step-fs',str(args.fine_step_fs),
                 '--coarse-step-fs',str(args.coarse_step_fs),
                 '--alignment-threshold','0.95','--pulse-bins','--save-trajectories','0']
            if (out/'config.json').exists():cmd.append('--resume')
            print(f'{args.table} case {index}: alpha={alpha}, F/Fref={fluence}, separation={sep} ps',flush=True)
            if args.dry_run: print(' '.join(cmd))
            else: subprocess.run(cmd,check=True)
        return
    records=[]
    for index,(alpha,fluence,sep) in enumerate(jobs):
        with (args.output/f'case_{index:02d}'/'summary.csv').open() as f:
            rows=list(csv.DictReader(f))
        if len(rows)!=1 or int(rows[0]['samples'])!=args.samples:
            raise ValueError(f'Incomplete case {index}')
        r=rows[0]
        if r['polarization']!='xy-' or int(r['initial_pole'])!=1 or int(r['target_pole'])!=-1:
            raise ValueError(f'Incorrect writing entry in case {index}')
        records.append(dict(alpha=alpha,F_over_Fref=fluence,separation_ps=sep,
            hemisphere_percent=100*float(r['write_success_fraction']),
            near_pole_percent=100*float(r['aligned_success_fraction']),**r))
    with (args.output/'combined_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    if args.table=='S4':
        x=[.001,.003,.006,.010,.012,.020]; columns=[.25,.5,.75,1.]
        select=lambda a,c:next(r for r in records if r['alpha']==a and r['F_over_Fref']==c)
        label='alpha'; prefix='F'
    else:
        x=[50.,75.,100.,125.,150.];columns=[.001,.003,.006]
        select=lambda a,c:next(r for r in records if r['separation_ps']==a and r['alpha']==c)
        label='separation_ps';prefix='alpha'
    for pretty in [False,True]:
        with (args.output/(args.table+('_table.txt' if pretty else '_Origin.txt'))).open('w',newline='') as f:
            w=csv.writer(f,delimiter='\t')
            w.writerow([label]+([f'{prefix}_{c}' for c in columns] if pretty else
                [f'{prefix}_{c}_{metric}_percent' for c in columns for metric in ['hemisphere','near_pole']]))
            for a in x:
                values=[]
                for c in columns:
                    r=select(a,c);h,n=r['hemisphere_percent'],r['near_pole_percent']
                    values.extend([f'{h:.1f} / {n:.1f}'] if pretty else [f'{h:.6f}',f'{n:.6f}'])
                w.writerow([a]+values)

if __name__=='__main__':main()
