# Collision preparation and response reuse

Collision physics and preparation live in `calculator/collision.py`. Grid
responses use the `CollisionPreparation` stage after their electronic vertices
are ready, then pass the operator explicitly into the density-matrix kernel.
The universal `Calculator` base and plain `Energy`/`Velocity` do not store
`collisionSpecification`, `preparedCollision`, or `epcModel`. Calls using an explicit
`collisionSpecification=CollisionSpecification(...)` prepare automatically.
Mode strings and implicit CRTA from `eta` are not supported.
Every collision-based response requires a specification or a prepared object;
Energy, Velocity, and other calculations without collisions can omit both.
The resulting calculator
exposes `preparedCollision` and `collision`, as well as
`density_matrix_kernel.collision`.

`prepare_collision(electronic, specification, epc_model=epc)` always receives
the EPC dependency explicitly; it does not read it from the electronic object.
`resolve_collision(electronic, specification=None, prepared=None, epc_model=None)`
returns a `PreparedCollision` without modifying the electronic object. Use
either `specification` or `prepared`. Density-matrix kernel constructors require
the keyword argument `collision=operator`; they never prepare one implicitly.

Explicit preparation makes the expensive rates reusable:

```python
import numpy as np
from scipy.constants import Boltzmann, electron_volt
from calculator import (
    Velocity, CollisionSpecification, prepare_collision,
    Opticalconductivity_LG_DM, Opticalconductivity_VG_DM,
)

# model is an electronic System; epc explicitly pairs it with a phonon model.
# For example: epc = SSHModelEPC(model, phonons, alpha_R, beta_R)
electronic = dict(model=model, kgrid=(16, 16, 1),
                  kBT=300 * Boltzmann / electron_volt)
specification = CollisionSpecification(
    "epc", qgrid=(4, 4, 1), phononTemperature=300,
    constantLinewidth=.2, delta=.03,
)
vertices = Velocity(**electronic)
prepared = prepare_collision(vertices, specification, epc_model=epc)

scan = dict(fermiList=np.array([-.7]), omegaList=np.array([.5, .8]))
lg = Opticalconductivity_LG_DM(**electronic, **scan, preparedCollision=prepared)
vg = Opticalconductivity_VG_DM(**electronic, **scan, preparedCollision=prepared)
assert lg.collision is vg.collision
```

You can also use `preparedCollision=previous_calculation.preparedCollision`.
For automatic preparation, pass `epcModel=epc` alongside `model=model` and
`collisionSpecification=specification`. A prepared object already holds its
EPC model, so reuse does not require repeating `epcModel`.
This reuses rates, Fermi-specific stationary states, and cached frequency
factorizations. Each response still constructs its own electronic vertices and
observable contractions. It does not reuse the entire electronic calculation.
First-order, DC, galvanic, and second-order DM calculators share this path;
electronic TDSE supports prepared CRTA/SERTA only.

## Compatibility and lifetime

Reuse assumes an unchanged model, collision specification, and providers.
The response must use the same mesh, offset, temperature and band basis.
Binding checks model identity, mesh metadata and array dimensions; it does not
scan Hamiltonian, EPC, phonon or eigensystem contents for modifications.
Changing those inputs while retaining prepared collisions or cached linewidth
sources is outside the supported lifecycle. Construct new preparation and
sources for a changed model.

When supplying `preparedCollision`, omit `collisionSpecification`: the prepared
object already contains its specification. Supplying both is rejected, even
when they refer to the identical specification object or have equal settings.
Linewidths come from the specification, never from the calculator's `eta`;
reuse uses the specification retained by the prepared object.
Fermi/frequency lists and observable projections may differ. SERTA linewidths
remain those of the source's own reference Fermi level and temperature; a
response scan does not recompute that source at each Fermi level.

Preparation retains read-only views of electronic arrays and a reference to
the electronic model and optional EPC model. Do not mutate them during a response. Reuse is sequential: mutable
stationary-state/factorization caches are not shared between concurrent solves.
Use a separate prepared object for each concurrent calculation.

Custom linewidth providers receive `CollisionInputs`: `model`, `kgrid`,
`kOffset`, `kBT`, `energies`, `vectors`, `phaseFactorR`, `velocity_K`, and optional
`epc_model`. Phonons are accessed through `epc_model.phonon_model`, never through
the electronic model. `ModelEPCLinewidth(epc, ...)` holds its own explicit model
pair and can supply SERTA without also passing `epcModel` to the response.
Providers do not need a signature or revision protocol. `ModelEPCLinewidth`
keeps its mesh-and-settings cache for repeated calculations on an unchanged
model; it does not monitor physical input arrays for changes.

## BZ tasks

`BZGridTask` recognizes `epc`, `full`, and `epc_markov`, including a supplied
prepared EPC collision. It evaluates `BZGrid` as one complete coupled mesh and
uses one task worker even when `ncores > 1`. The q grid must close on that full
mesh. Symmetry selection cannot split this coupled task. Local CRTA/SERTA
calculations retain independent shifted-submesh scheduling.

This fixes missing transitions between independently solved partitions; it
does not implement distributed collision assembly or a distributed sparse solve.
Full-mesh rate storage and response arrays can be substantially larger than an
individual legacy submesh. Rate storage remains O(Nk Nq Nb^2).

Task adapters forward `epcModel` and `preparedCollision`. A prepared object requires one task
with the identical model, grid and offset; sharing it across shifted submeshes
raises an error before workers start. Direct calculator calls are the simplest
way to reuse an explicitly prepared mesh.

## Explicit and static phonons

`Opticalconductivity_VG_DM_StaticEPC`, `Opticalconductivity_VG_DM_Phonon`, and
`Opticalconductivity_TDSE(phononDynamics=True)` use a separate CRTA-background
preparation function. They require an explicit
`epcModel=epc` and `CollisionSpecification('crta', constantLinewidth=...)`, and reject
`preparedCollision` and SERTA/Markov collision specifications.
The static and frequency-domain variants use `PhononPreparation`; TDSE selects
its explicit-phonon branch directly. Phonon solvers and `SSHTimeOperators`
receive the EPC model as an argument instead of looking it up through `owner`.
`SSHTimeOperators` lives in `elph` and takes `(epc_model, kpoints)`; it does
not receive a calculator. The response solver constructs these fractional
k-points from its electronic grid and offset.
Their displacement corrections or explicit phonon dynamics
remain in their own response solvers; adding an EPC collision would double
count scattering.

## Validation

`python -m unittest tests.test_collision_preparation` checks reuse across gauges
and response orders, automatic preparation, mesh/configuration compatibility,
SERTA cache reuse, phonon-path separation, and full-mesh task equivalence.
