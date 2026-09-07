"""Build a four-band inversion-broken spinful Rice-Mele chain.

The primitive cell contains two atoms with one local s orbital and two spin
states on each atom. Dimerized hopping and staggered onsite energy generate a
gapped, inversion-broken optical response. Rashba SOC and a tilted exchange
field keep spin-dependent response channels nontrivial. Only longitudinal
ionic motion modulates the two nearest-neighbor hoppings through SSH EPC.

Energy and length units are eV and Angstrom. No external material files are
required; the generated POSCAR and hr files are written below --output.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from system import System_tb, SSHModelEPC


def write_hr(path, iRvec, values, title):
    """Write dense Wannier90 hr blocks with unit Wigner-Seitz degeneracies."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    norbs = values.shape[-1]
    with path.open('w', encoding='utf-8') as stream:
        stream.write(f'{title}\n{norbs}\n{len(iRvec)}\n')
        for start in range(0, len(iRvec), 15):
            stream.write(' '.join(['1']*min(15, len(iRvec)-start))+'\n')
        for R, block in zip(iRvec, values):
            for j in range(norbs):
                for i in range(norbs):
                    value = complex(block[i, j])
                    stream.write(f'{R[0]:3d} {R[1]:3d} {R[2]:3d} {i+1:3d} {j+1:3d} '
                                 f'{value.real:.16e} {value.imag:.16e}\n')


def build_model(directory, latticeConstant=2.2, intracellDistance=.9,
                hopping=(1.0, .72), onsiteDifference=.7, nextHopping=(.10, .04),
                rashba=(.16, .11), exchange=.24,
                exchangeDirection=(.35, .25, np.sqrt(.815)), epc=(.45, .35),
                masses=(12., 14.), phononEnergy=.04):
    """Write model files and return a spinful four-band ``System_tb``.

    ``hopping=(intra, inter)`` and ``epc=(intra, inter)`` distinguish the two
    bonds. ``onsiteDifference`` is E_A-E_B. The unequal bonds plus this onsite
    contrast form the inversion-broken Rice-Mele sector.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if not 0 < intracellDistance < latticeConstant:
        raise ValueError('intracellDistance must lie strictly inside the primitive cell')
    lattice = np.diag([latticeConstant, 10., 10.])
    positions = np.array([[0., 0., 0.], [intracellDistance, 0., 0.]])
    with (directory/'POSCAR').open('w', encoding='utf-8') as stream:
        stream.write('Four-band inversion-broken spinful chain\n1.0\n')
        for row in lattice:
            stream.write(' '.join(f'{value:.16e}' for value in row)+'\n')
        stream.write('A B\n1 1\nCartesian\n')
        for row in positions:
            stream.write(' '.join(f'{value:.16e}' for value in row)+'\n')

    iRvec = np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0]])
    lookup = {tuple(R): i for i, R in enumerate(iRvec)}
    spatial = np.zeros((len(iRvec), 2, 2))
    lambdas = np.zeros_like(spatial)
    alphas = np.zeros_like(spatial)
    home = lookup[(0, 0, 0)]
    spatial[home] = np.array([[onsiteDifference/2, -hopping[0]],
                              [-hopping[0], -onsiteDifference/2]])
    lambdas[home, 0, 1] = lambdas[home, 1, 0] = rashba[0]
    alphas[home, 0, 1] = alphas[home, 1, 0] = epc[0]
    left, right = lookup[(-1, 0, 0)], lookup[(1, 0, 0)]
    spatial[left, 0, 1] = spatial[right, 1, 0] = -hopping[1]
    lambdas[left, 0, 1] = lambdas[right, 1, 0] = rashba[1]
    alphas[left, 0, 1] = alphas[right, 1, 0] = epc[1]
    for orbital, amplitude in enumerate(nextHopping):
        spatial[left, orbital, orbital] = spatial[right, orbital, orbital] = -amplitude

    hamiltonian = np.zeros((len(iRvec), 4, 4))
    hamiltonian[:, :2, :2] = hamiltonian[:, 2:, 2:] = spatial
    write_hr(directory/'chain_hr.dat', iRvec, hamiltonian,
             'Spinful inversion-broken Rice-Mele chain; eV')
    write_hr(directory/'rashba_hr.dat', iRvec, lambdas,
             'Scalar Rashba coupling on chain bonds; eV')
    write_hr(directory/'epc_hr.dat', iRvec, alphas,
             'SSH derivative dH/dr on chain bonds; eV/Angstrom')

    model = System_tb(str(directory), hamR='chain', projections={'A': [0], 'B': [0]}, ispin=2)
    model.verbose = False
    direction = np.asarray(exchangeDirection, dtype=float)
    if direction.shape != (3,) or not np.all(np.isfinite(direction)) or np.linalg.norm(direction) == 0:
        raise ValueError('exchangeDirection must be a finite nonzero three-vector')
    model.addExchangeSplitting([(0, (0,)), (1, (0,))],
                               -exchange*direction/np.linalg.norm(direction))
    model.addRashbaSOC()
    model.addModelEPC(SSHModelEPC)
    model.setModelPhonons(masses=masses, phononEnergies=phononEnergy, directions=(0,))
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', default='tmp/chain_four_band')
    parser.add_argument('--k-grid', type=int, default=128,
                        help='Number of points along the one-dimensional Brillouin zone')
    args = parser.parse_args(argv)
    model = build_model(Path(args.output))
    from calculator import Energy, Velocity
    energy = Energy(model=model, kgrid=(args.k_grid, 1, 1))
    velocity = Velocity(model=model, kgrid=(args.k_grid, 1, 1))
    bands = energy.energies.reshape(args.k_grid, model.norbs)
    occupied_max = float(np.max(bands[:, 1]))
    empty_min = float(np.min(bands[:, 2]))
    summary = dict(nbands=model.norbs, atoms_per_cell=2, active_phonon_directions=['x'],
                   fundamental_gap_eV=empty_min-occupied_max, transverse_cell_area_A2=100.,
                   max_abs_velocity_x_eV_A=float(np.max(np.abs(velocity.velocity_K[..., 0, :, :]))))
    output = Path(args.output)
    np.savez(output/'model_results.npz', k_fractional=np.arange(args.k_grid)/args.k_grid,
             energies_eV=bands)
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(f'Inputs and outputs: {output.resolve()}')


if __name__ == '__main__':
    main()
