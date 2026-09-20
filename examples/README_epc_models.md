# Electronic, phonon, and EPC models

`System` represents only the electronic model. `EinsteinPhonon` is independent,
and `SSHModelEPC` holds an explicit electronic/phonon pair and its coupling
coefficients. No phonon or EPC object is attached to `System`.

The packages now follow the physical model boundaries:

- `system/`: electronic Hamiltonian, orbitals, and geometry.
- `phonon/`: the abstract `Phonon`, concrete `EinsteinPhonon`,
  `PhononModeSelection`, and prescribed displacement trajectories.
- `elph/`: the abstract `EPCModel`, concrete `SSHModelEPC`, and
  `SSHTimeOperators(epc_model, kpoints)` for displaced electronic operators.
- `calculator/`: self-energy, collision preparation, and response solvers.

Import phonon and EPC models directly from their packages. The previous
`system` exports and plural `EinsteinPhonons` name have been removed.
The separate `epc/` workflow has not been migrated.

`Phonon.modes(q)` returns energies `(nmodes,)` in eV and dimensionless
polarizations `(nmodes,natoms,3)` in mass-weighted coordinates. Each polarization
has unit Euclidean norm. Masses are in amu; atom order and the reciprocal
lattice for fractional q must agree with the electronic model passed to
`EPCModel`. The caller supplies that correspondence; matching atom counts
does not establish it. For cell L, the phase is `exp(2*pi*i*q.L)`; any
basis-position phase belongs in the polarization.

The base `modeDisplacements(q)` applies `sqrt(hbar/(2*M*Omega))` to each
atom's polarization and returns Angstrom amplitudes, without thermal or
`1/sqrt(Ncells)` factors. It requires positive mode energies; zero-frequency
and unstable modes need an explicit physical prescription. This interface
can represent dispersive collective modes, but no Debye or force-constant
model is implemented yet. Snapshot, covariance, supercell, and trajectory
sampling remain Einstein-specific capabilities used by the current static
and dynamical phonon response solvers.

`EPCModel` defines displacement vertices and performs their band-basis
transformation using supplied electronic eigenvectors. Self-energy uses
`has_second_order` and `secondDisplacementVertices` through this interface;
it does not read SSH coefficients. `SSHTimeOperators` receives explicit
fractional k-points and has no dependency on a calculator owner. The SSH
single-point convenience method can still diagonalize the electronic
Hamiltonian when eigenvectors are omitted; response stages supply them.

```python
from elph import SSHModelEPC
from phonon import EinsteinPhonon
from calculator import Velocity, CollisionSpecification, prepare_collision

# model is an initialized electronic System with two atoms.
phonons = EinsteinPhonon([12., 14.], .04, directions=(0,))
epc = SSHModelEPC.from_files(model, phonons, alphaR="epc")
# Alternatively: epc = SSHModelEPC(model, phonons, alpha_R, beta_R)

vertices = Velocity(model=model, kgrid=(16, 1, 1), kBT=.025852)
spec = CollisionSpecification("epc", qgrid=(4, 1, 1), phononTemperature=300)
prepared = prepare_collision(vertices, spec, epc_model=epc)
```

Full EPC responses accept `model=model, epcModel=epc` for automatic preparation,
or `preparedCollision=prepared` for reuse. `CollisionInputs` retains electronic
array views and the explicit EPC reference. CRTA and externally supplied SERTA
linewidths do not require an EPC model.

`ModelEPCLinewidth(epc, ...)` and `ModelElectronSelfEnergy(epc, kpoints, ...)`
consume the same physical model pair. Fermi levels, temperatures, broadening,
and solver choices remain calculation settings. `epc.matrixElements(k, q)` and
`epc.matrixElementsBatch(kpoints, q, vectors_k, vectors_kq)` evaluate coupling
without adding collision statistics or occupations.

For frozen snapshots or explicit phonon dynamics:

```python
import numpy as np

supercell_epc = epc.supercell(np.diag([4, 1, 1]))
supercell_model = supercell_epc.electronic_model
u = supercell_epc.phonon_model.sampleDisplacements(300, seed=7)
snapshot = supercell_epc.makeDisplacedModel(u)
```

`epc.supercell()` returns a new pair and transforms the coupling with exactly
the electronic orbital/cell mapping. It leaves the primitive models unchanged.
`System.make_supercell()` remains an electronic-only in-place operation;
do not use it to transform an existing EPC pair. Snapshots are electronic
models and use their own CRTA response settings, without the equilibrium EPC.

StaticEPC, Phonon, and TDSE with `phononDynamics=True` receive
`model=supercell_model, epcModel=supercell_epc` and an explicit CRTA background.
They do not pass through the Markov collision stage.

The old `System` EPC/phonon attachment and evaluation methods have been removed.
The chain and honeycomb `build_model()` examples now return `(model, epc)`.
See [collision preparation](README_collision_preparation.md) for reuse and
parallel scheduling.
