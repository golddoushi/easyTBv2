"""Fast EPC conductivity benchmark on the four-band spinful chain.

The command-line interface is shared with honeycomb_conductivity.py. The most
important grids are:

  --electronic-k-grid N       primitive-chain k mesh for the main DM results
  --clamped-tdse-k-grid N     smaller k mesh for the ion-clamped TDSE check
  --phonon-supercell S        number of primitive cells in the moving-ion chain
  --supercell-k-grid K        electronic k mesh of that S-cell supercell
  --serta-q-grid Q            phonon q mesh used for SERTA linewidths
  --markov-q-grid Q           phonon q mesh used for the Markov collision

For explicit moving phonons, S and K correspond to a primitive electronic
mesh S*K and phonon mesh S. A useful inexpensive comparison is:

python examples/chain_conductivity.py --phonon-dynamics --epc-markov classical \
    --electronic-k-grid 64 --markov-q-grid 4 --phonon-supercell 4 \
    --supercell-k-grid 1 --omega 0.6 --phonon-trajectories 2
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.chain_four_band import build_model
from examples.honeycomb_conductivity import run_conductivity_benchmark


def main(argv=None):
    return run_conductivity_benchmark(
        model_builder=build_model, periodic_dimensions=1,
        default_output='tmp/chain_conductivity', default_electronic_k_grid=64,
        description=__doc__, argv=argv)


if __name__ == '__main__':
    main()
