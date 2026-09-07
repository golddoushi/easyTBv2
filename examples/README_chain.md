# Four-band chain EPC benchmark

`chain_four_band.py` builds a one-dimensional spinful Rice-Mele model with two
atoms per primitive cell. Each atom carries one local s orbital and two spin
states, giving four bands. Unequal intracell/intercell hopping and staggered
onsite energies break inversion. Rashba hopping and a tilted exchange field
make spin response nontrivial. Longitudinal SSH coupling modulates both bonds.

Build and inspect the model with:

```text
python examples/chain_four_band.py --k-grid 128
```

The full conductivity benchmark shares its implementation and parameter names
with the honeycomb example:

```text
python examples/chain_conductivity.py --electronic-k-grid 64 --omega 0.6 1.3
```

For an inexpensive comparison of explicit moving phonons, the matched Fourier
solver, and the level-4 Markov collision:

```text
python examples/chain_conductivity.py --phonon-dynamics --serta none \
  --epc-markov classical --electronic-k-grid 64 --markov-q-grid 4 \
  --phonon-supercell 4 --supercell-k-grid 1 --omega 0.8 \
  --phonon-trajectories 2 --phonon-phase-points 17
```

## Grid vocabulary

- `electronic-k-grid`: primitive-cell k points per periodic direction for the main DM calculations.
- `clamped-tdse-k-grid`: smaller primitive-cell mesh used only for the ion-clamped TDSE check.
- `phonon-supercell`: real-space primitive-cell repeats used by snapshots and moving-ion calculations.
- `supercell-k-grid`: electronic mesh of the enlarged moving-ion supercell.
- `serta-q-grid`: phonon q mesh used to construct diagonal SERTA linewidths.
- `markov-q-grid`: phonon q mesh used to construct the mesh-global Markov collision operator.

For a one-dimensional explicit-phonon calculation, supercell size `S` and
supercell electronic mesh `K` correspond to primitive meshes `(S*K,1,1)` for
electrons and `(S,1,1)` for phonons. Run `chain_conductivity.py --help` for the
remaining thermal, damping, trajectory, and integration parameters. The old
short grid options remain accepted as hidden migration aliases.

The default chain has a gap near 0.69 eV. It produces finite longitudinal
optical conductivity and finite inversion-odd second-order response. It is a
numerical benchmark, not a parameterization of a particular material. Reported
three-dimensional conductivity units use the model cell's arbitrary
`10 Angstrom x 10 Angstrom` transverse area; relative solver comparisons are
meaningful, while the absolute S/m value is not a physical one-dimensional
conductance until a transverse normalization convention is chosen.
