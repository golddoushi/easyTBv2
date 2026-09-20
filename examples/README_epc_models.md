# Electronic, phonon, and EPC models

`System` represents only the electronic model. `EinsteinPhonons` is independent,
and `SSHModelEPC` holds an explicit electronic/phonon pair and its coupling
coefficients. No phonon or EPC object is attached to `System`.

```python
from system import EinsteinPhonons, SSHModelEPC
from calculator import Velocity, CollisionSpecification, prepare_collision

# model is an initialized electronic System with two atoms.
phonons = EinsteinPhonons([12., 14.], .04, directions=(0,))
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
