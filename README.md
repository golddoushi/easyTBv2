# eazyTBv2

`eazyTBv2` is a research code for electronic transport, nonlinear optical
response, light-induced magnetization and torque, and subsequent macrospin
dynamics.  Its input is a real-space Hamiltonian and, when available, overlap
and position/Berry-connection matrices in a localized basis.

This README is also a handoff document for future developers and coding
agents.  Read the conventions, validation rules, and known limitations before
changing a response formula.  Several implementations deliberately coexist so
that length gauge (LG), velocity gauge (VG), density-matrix (DM), and time-domain
(TDSE) results can be cross-checked.

## Quick Start

The main research drivers are scripts rather than a stable command-line API:

```powershell
python spinHallCondInput.py
python nonlinearCondInput.py
python angleDependentSOTMapping.py --stage all
python LLGInput.py
python fermiLoop.py
```

They contain model-specific settings and blocks that are commonly enabled or
disabled while doing a calculation.  Inspect a driver before running it.
Realistic calculations can be very expensive and can overwrite data in their
configured `mcalc/<model>` directory.

The project currently has no complete package/environment specification.
Typical dependencies include NumPy, SciPy, Matplotlib, threadpoolctl, and the compiled
extensions in the repository root.  See `INSTALL` for the legacy C/BLAS/FFTW
build notes.  Windows `.pyd`/`.dll` files in this checkout are not portable to
Linux and must be rebuilt there.

## Repository Map

| Path | Responsibility |
| --- | --- |
| `system/__init__.py` | Base `System`, real-space operators, symmetry, projections, exchange decomposition and rotation |
| `system/system_tb.py` | Orthogonal tight-binding input with diagonal local-basis position |
| `system/system_w90.py` | Wannier90 input, position matrices, spin-separated collinear input, and patched-VASP SOI input |
| `system/system_omx.py` | OpenMX/LCAO input with a nonorthogonal basis |
| `calculator/Grid.py` | BZ-grid operators and response calculators |
| `calculator/Path.py` | Band-path and surface/path calculators |
| `calculator/density_matrix.py` | Reusable first- and second-order LG/VG density-matrix kernels |
| `calculator/__init__.py` | Public calculator imports |
| `task.py` | Parallel BZ/path task wrappers and `solve*` helper functions |
| `fileio.py` | Tensor, polarization-contracted, and BZ-distribution output |
| `angleDependentSOTMapping.py` | Fixed-number angle scan, checkpointing, symmetry-aware harmonic fitting |
| `SphericalHarmonics.py` | Scalar and vector spherical-harmonic fitting |
| `LLG/MagneticDynamics.py` | Fields, torques, and single-moment LLG integration |
| `LLG/Device.py` | `SOTDevice`, `IFTDevice`, and tangent-torque sampling |
| `LLG/Topology.py` | Fixed points, stability, separatrices, basins, and Poincare-index analysis |
| `LLGInput.py` | Dynamics, torque-map, raw-vs-fit, and phase-portrait examples |
| `vasp-patches/` | VASP 6.1.2 SOI-matrix patch and its format documentation |
| `slurm_scripts/` | Slurm sharding scripts for angle-dependent calculations |
| `mcalc/` | Toy models, realistic models, and generated research data |

## Global Conventions

### Units

- Energy, broadening, photon energy, and Hamiltonian elements: eV.
- Real-space matrix elements: Angstrom.
- Electric field: V/Angstrom.  Numerically, `e * E * r` is in eV.
- Vector potential used by the VG Hamiltonian: `1/Angstrom`, with `E = Omega A`
  in the code's energy convention.
- Time in the LLG layer: fs.
- Magnetic moment: `mu_B` per cell unless a projection says otherwise.
- `eta` is a relaxation energy, normally interpreted as `hbar/tau`, not a
  scattering rate in `1/fs`.
- IFE coefficient: `mu_B/(V/Angstrom)^2` per cell.
- IFT coefficient before division by `hbar`: `mu_B*eV/(V/Angstrom)^2`.
  `fileio.outputIFTcoeffs` divides by `HBAR_EV_FS` and reports a torque rate in
  `mu_B/fs/(V/Angstrom)^2`.

### Tensor Axes

The current nonlinear-response convention is

```text
[..., drive_1, drive_2, measured_component]
```

For example, `chi_IFE[..., a, b, c]` and `chi_IFT[..., a, b, c]` contract with
the optical polarization as

```text
response_c = sum_ab e_a e_b^* chi_ab,c .
```

Do not transpose these axes to match a paper without updating every output,
task wrapper, projection, and test.  Off-diagonal complex tensors should be
compared after a physical polarization contraction, or after explicitly taking
their symmetric and antisymmetric combinations.  A real TDSE drive is not, by
itself, the same object as one raw complex off-diagonal frequency-domain entry.

The common polarization aliases are `x`, `y`, `z`, `xy+`, `xy-`, `xz+`,
`xz-`, `yz+`, and `yz-`.  In `LLG/Device.py`, `xy+` is proportional to
`(1, +i, 0)`.  A manuscript must also state the Fourier convention and light
propagation direction before calling this right- or left-circular polarization.

## System Layer

### Local Bases

- `System_tb`: orthogonal, normalized tight-binding basis.  The position
  operator is diagonal in the local basis.  This is the cleanest setting for
  toy models and local orbital-angular-momentum operators.
- `System_w90`: orthogonal Wannier basis.  The position/Berry-connection matrix
  can contain off-diagonal elements reconstructed from Wannier90 data.
- `System_omx`: normalized but nonorthogonal LCAO basis.  Generalized
  eigenproblems are supported, but the overlap-dependent velocity corrections
  have received less validation than the orthogonal paths.  Treat OMX transport
  as an active validation area.

The central real-space arrays are `ham_R`, `overlap_R` when present, and
`AA_R_W` when a local-basis position/Berry connection is available.  Their
Fourier transforms are evaluated on regular BZ grids with optional phase
shifts.

### Wannier90 and Patched VASP SOI

`System_w90` accepts either a normal spinor Wannier90 dataset or separate
collinear `up`/`dn` datasets (`spinSeparated=True`).  The patch under
`vasp-patches/` writes `wannier90.soi`, which is transformed to the common
Wannier basis and used by `includeSOCFromSOI`.

The collinear plus SOI construction is a frozen-subspace approximation:

```text
H(m) approximately equals H_collinear + V_SOI(m).
```

It worked well for the weak-SOC FeCl2 validation, while PbSn exposed visible
errors for strong SOC even inside the frozen window.  It is not equivalent to
a new self-consistent noncollinear DFT calculation for every direction.

`includeSOCFromSOI(theta, phi, keepRegularSpinAxis=True)` uses a local spin
frame aligned with the requested direction and then restores the regular global
Cartesian spin frame.  The restoration is a representation change, not another
physical rotation.  A separate approximation is to include zero-angle SOI and
then call `rotateExchangeDirection`; that keeps lattice-fixed SOI unchanged and
rigidly rotates only the exchange Hamiltonian.

For serious work, benchmark representative directions against independently
self-consistent full-SOC Wannier90 models when they are available.

### Exchange Hamiltonian

`System.getExchangeR` decomposes

```text
ham_R = nonexchange_R + exchange_R
```

and stores `exchange_R_reference` for later rigid rotations.  Supported base
provenances include:

- `explicit`: onsite orbital-diagonal Pauli blocks.  Useful for toy models and
  an approximate local exchange in realistic models.
- `orientation finite difference`: `[H(m) - H(-m)]/2` in the same local gauge.
- `pauli estimate`: the full-real-space spin-traceless Pauli part.  This is
  origin/cell-assignment covariant, but it cannot distinguish exchange from
  SOC, Zeeman, or another spin-dependent interaction.
- `collinear spin difference`: supplied by spin-separated subclasses where
  meaningful, before SOI is included.

`rotateExchangeDirection(theta, phi)` actively rotates every local and nonlocal
matrix element of `exchange_R_reference` while keeping `nonexchange_R` fixed.
`getRExchange_R` and `getRRExchange_R` build the exchange-weighted real-space
moments needed by exchange-velocity/contact vertices.

The total spin torque `-i[mu S,H]` is an electronic spin-rate observable.  The
exchange torque `-i[mu S,H_XC]` is the reaction torque intended for the
macrospin LLG model.  They are not interchangeable.  For a single-sublattice
macrospin, the whole-cell exchange torque is the most robust quantity.  For a
multi-sublattice LLG model, constructing site-resolved exchange Hamiltonians is
preferable to simply projecting the total torque with `{P,T}/2`.

## Operators

`Energy` solves ordinary or generalized eigenproblems and supplies Fermi-Dirac
occupations and energy derivatives.

`Velocity` builds the Hamiltonian derivatives from real-space moments and adds
the local-basis position correction.  Optional high-order vertices (`w_K` and
related arrays) are required for complete VG second-order responses.  With
`exchangeVelocity=True`, it also builds exchange velocity/contact vertices for
nonlocal `H_XC`.

`Dipole` means the off-diagonal interband dipole:

```text
r_mn = v_mn / (i DeltaE_mn),  m != n.
```

The diagonal position operator of a periodic crystal is the differential part
of `i D_k`; it is not a finite band-space matrix.  `highOrderDipole=True` builds
the generalized derivative `r_deriv_K` used by shift-response formulas and
the current VG/TDSE covariant-relaxation construction. The full second-order
LG DM kernel instead differentiates `f(H)` directly, as described below.
Do not replace this object blindly with the local-basis `position_K`: the latter
is one connection contribution, not the complete observable dipole.

## Density-Matrix Kernels

All new DM work should reuse `calculator/density_matrix.py`.  Calculator classes
provide observable vertices, projections, BZ integration, and output storage;
the kernel module provides the response density matrices.

### First-Order DC Kernel

`LengthGaugeDCFirstOrderDMKernel` separates:

- `I`: intraband displaced population.
- `E`: interband coherence.

It is consumed by `AnormalousHallConductance_LG_DM` and
`SpinHallConductance_LG_DM`.

### Second-Order Optical Kernels

`LengthGaugeSecondOrderDMKernel` and `VelocityGaugeSecondOrderDMKernel` preserve
four channels:

- `II`: second-order intraband response driven from first-order intraband.
- `EI`: second-order interband response driven from first-order intraband.
- `IE`: second-order intraband response driven from first-order interband.
- `EE`: second-order interband response driven from first-order interband.

These kernels are shared by BPVE, IFE, and IFT DM calculators.  Channel-resolved
arrays are intentionally public because they are useful for diagnosing metals,
low-frequency limits, and LG/VG cancellations.

### Important Switches

- `fullCovariantDerivative=False`: historical interband-only behavior.  The
  active second-order channels are `IE` and `EE`.
- `fullCovariantDerivative=True`: includes intraband and mixed channels using
  the full covariant derivative

  ```text
  i D_a rho = i partial_k_a rho + [A_a, rho].
  ```

  It requires `kBT > 0` and finite `eta` so derivatives of the Fermi function
  and the dc population are well defined.
- `finiteTemperatureOccupations=True`: smooths occupations without enabling
  every intraband channel, where the calculator supports that distinction.

### Divided-Difference Full LG

`LengthGaugeSecondOrderDMKernel(..., full_covariant_derivative=True)` computes
the equilibrium derivatives as matrix-function Frechet derivatives:

```text
F = f(H), V_a = D_a H, w[a,b] = D_b V_a
(D_a F)_mn = f[E_m,E_n] V_a,mn
(D_b D_a F)_mn = f[E_m,E_n] w[a,b]_mn
  + sum_l f[E_m,E_l,E_n] (V_b,ml V_a,ln + V_a,ml V_b,ln)
R_s X = X_mn / (E_m - E_n - s Omega - i eta)
rho1_a(s Omega) = R_s(i D_a F)
i D_b rho1_a = R_s(-D_b D_a F - i [V_b,rho1_a])
rho2_ab = R_0(i D_b rho1_a(-Omega) + i D_a rho1_b(+Omega))
```

The divided differences use the analytic limits `f[x,x]=f'(x)` and
`f[x,x,x]=f''(x)/2`, stable particle/hole expressions and a local Taylor
expansion for nearby nodes. There is no added broadening in the geometric
dipole, no masked inverse gap in the total kernel, and no k-neighbor exchange.
Only resonant Liouville denominators contain `eta`. Equilibrium derivatives
are cached per Fermi level, not recomputed for each photon energy; intermediate
band sums never allocate a full k-by-N-by-N-by-N tensor.

This path is shared by `InverseFaradayEffect_LG_DM`,
`InverseFaradayTorque_LG_DM`, and `BulkPhotovoltaicEffect_LG_DM`. They contract
`block.total_second_order` directly for total/BZ outputs rather than recover
the total by subtracting potentially large channel responses. The
`fullCovariantDerivative=False` path is unchanged.

In full LG only, `degen` now controls **channel reporting blocks**, not the
physical total: adjacent eigenvalues separated by at most `degen` belong to
the same block (with a machine-precision floor). Actual eigenvalues are never
flattened. `I` means within-block and `E` means between-block; for isolated
bands these reduce to diagonal and off-diagonal matrices. Differentiating the
block projectors preserves the former four-channel formulas away from
degeneracies, and permits unitary changes of basis within an exactly degenerate
block. Individual channels depend on the block convention. Tiny block gaps
can cause cancellation in channel sums; use the separately evaluated total.

This update does **not** modify the first-order Hall kernels, VG, TDSE, or
observable-specific vertices. In particular, the orbital torque's existing
explicit electric-field vertex still uses `r_K` and may retain cutoff
sensitivity even though the density matrix itself does not. The spin-exchange
torque has no such electric-field vertex. These limits must not be confused
with a claim that all LG/VG approximations or all observables are now cutoff-free.

Run `python test_lg_divided_differences.py` for scalar divided-difference limits,
nondegenerate agreement of all four channels, exact/near degeneracies,
block-unitary covariance, the analytic single-band intraband limit,
independent finite-k-derivative checks with a
nontrivial connection, and toy-model calculator/BZ/projection regressions.

Validation on FeCl2 (33x33x1, mu=-3 eV, kBT=0.026 eV, eta=0.06582 eV,
theta=5 degrees, phi=0) gave identical total spin-exchange IFT for
degen=1e-6, 1e-3 and 0.1 eV at Omega=0, 0.01, 0.95 and 1.5 eV.
This checks cutoff independence, not k-mesh convergence. The older
`test.densityMatrixKernels.py` BPVE snapshot in `tmp/bpve_kernel_baseline.npz`
was already inconsistent with the pre-update code; it was not regenerated.
Direct pre-update/new-code comparisons agree away from degeneracies, and
the separate Hall and VG full-covariant regression tests pass.

The full-covariant metallic response is much more demanding in k space than
the interband-only response.  It contains Fermi-surface derivatives and
cancellations between channels.  A coarse grid can therefore make LG and VG
look very different even when both implementations are correct.  Converge the
total and each channel with `kBT`, `eta`, and BZ grid together.

The kernels have been optimized so repeated Fermi scans contract exact
band-resolved weights instead of rebuilding full band-pair density matrices
where possible.  BPVE/IFE/IFT batch observable directions and skip inactive
channels.  Keep this separation: copying the kernel algebra back into each
final calculator makes both correctness and parallel performance worse.

### Electron-Phonon SERTA

`fileio.ElectronSelfEnergy` reads VASP electron-phonon self energies from the
`self_energy_<calculator>` group in `vaspout.h5`. VASP stores selected
`(spin,k,band)` entries through sparse one-based indices; the reader restores
the dense irreducible-k arrays and defines the on-shell relaxation energy as

```text
Gamma_nk = -2 Im Sigma^R_nk(epsilon_nk) = hbar / tau_nk.
```

The first implementation requires self energies for every DFT band entering
the Wannier disentanglement (`band_start=1`, `band_stop=num_bands`). A full EPI
matrix is not required to recover VASP's full-to-IBZ map. By default,
`System_w90.includeElectronSelfEnergy` matches the full-grid Wannier90 `.eig`
spectra to the IBZ spectra in `vaspout.h5`, then validates the map against
VASP's stored symmetry weights and explicit IBZ representatives. Ambiguous
spectra are rejected. In that case, pass `vaspelph.h5` directly or extract its
`vkpt_k`, `vkpt_kp`, and `indx_fbz2ibz` datasets to a compact `.npz` with the
keys `irreducible_kpoints`, `full_kpoints`, and zero-based
`full_to_irreducible`.

`System_w90.includeElectronSelfEnergy` applies the same checkpoint matrices
used for the Hamiltonian, reports both the raw DFT-band and projected
Wannier-subspace eigenvalue mismatches, and Fourier transforms the result to
the Wannier real-space grid. A projected mismatch above
`eigenvalueTolerance` is rejected; a raw mismatch is warned because agreement
after projection does not make two inconsistent source calculations identical.
The positive-semidefinite square root of the projected linewidth is
interpolated; reconstructing `Gamma=L L^dagger` on the target grid prevents
unphysical negative relaxation rates from Fourier ringing.

The loader also reports squared-norm Wigner-Seitz boundary weights for
`Sigma_R`, `Gamma_R`, and `sqrt(Gamma)_R`. The last is the most direct
localization diagnostic for the implemented collision interpolation and emits
a warning above `0.1%`.

```python
from calculator import CollisionSpecification
from fileio import ElectronSelfEnergy

self_energy = ElectronSelfEnergy(
    "mcalc/Fe/vaspout.h5",
)
model.includeElectronSelfEnergy(self_energy, temperature=300.0)

calculation_args["collisionSpecification"] = CollisionSpecification("serta")
calculation_args["full covariant derivative"] = True
```

`CollisionSpecification("crta", constantLinewidth=...)` reproduces the 
constant-relaxation model. `CollisionSpecification("serta")` supplies band- 
and k-dependent population linewidths and the coherence rate 
`(Gamma_mk+Gamma_nk)/2` to the reusable first-order d.c. kernel. It therefore 
changes both the `I` and `E` channels, although VASP's reported SERTA 
conductivity is an intraband quantity and should be compared with this code's 
`I` channel.

State-dependent SERTA is currently restricted to first-order d.c. A
second-order optical collision integral contains field-dependent scattering
sources and vertex corrections; simply replacing every scalar `eta` by
`Gamma_nk` is not a controlled approximation. The second-order kernels reject
SERTA explicitly and continue to support CRTA through the same collision API.

The interpolated on-shell self energy is not as local as a Hamiltonian. Converge
the source self-energy grid, target BZ grid, temperature, and Fermi level. VASP
uses tetrahedron construction of a transport distribution followed by energy
quadrature, whereas the current DM calculator directly samples the thermally
broadened Fermi surface. Agreement at an identical finite k grid is therefore
not expected to be exact even when conventions and units agree.

## Response Calculators

### Hall Responses

- `AnormalousHallConductance` and `SpinHallConductance`: established resonant
  denominator implementations.
- `AnormalousHallConductance_LG_DM` and `SpinHallConductance_LG_DM`: reusable
  first-order LG-DM implementations with `I`/`E` channels and optional
  k-resolved output.
- `OrbitalHallConductance`, diagonal variants, Berry-curvature path tools, and
  FHS Chern/Z2 calculators are also exposed from `calculator`.

A spin Hall conductivity need not be quantized merely because an FHS spin-Chern
diagnostic is integer: conventional spin current is generally not conserved in
the presence of SOC, and units/normalization differ.  Validate topology and
transport as distinct observables.

### BPVE

Available implementations include `BulkPhotovoltaicEffect_VG`, `VG_DM`,
`LG_DM`, the LG shift/injection decompositions, and TDSE variants.  The DM
classes use the same second-order kernels as IFE/IFT; the observable vertex is
the charge current.  Preserve the contact/diamagnetic current vertices in VG.

### IFE

Available paths include perturbative VG TDSE, finite-field Bloch-acceleration
TDSE, VG/VG-DM, and LG/LG-DM.  The observable is spin and/or orbital magnetic
moment.  `fullCovariantDerivative=True` enables the complete collision/
intraband channel structure in the DM and comparable perturbative TDSE paths.

`InverseFaradayEffect_BlochAcceleration_TDSE` evaluates the finite shifted
Hamiltonian `H(k-A(t))` and is the nonperturbative fixed-Hamiltonian check.  It
does not update the exchange-correlation potential self-consistently when a
strong pulse changes the magnetic moment.  Large-moment enhancement or actual
switching can therefore require rt-TDDFT or an explicitly coupled electronic-
magnetic model.

### IFT

`InverseFaradayTorque_LG_DM`, `InverseFaradayTorque_VG_DM`, and
`InverseFaradayTorque_TDSE` compute exchange spin torque and optional orbital
torque.  VG and TDSE require high-order torque/contact terms; the full VG result
has been checked against TDSE and should not be replaced by the naive
`Tr[T rho2]` term alone.  Nonlocal exchange additionally requires the exchange
velocity/contact vertices.

Orbital torque is a meaningful microscopic angular-momentum transfer, but its
conversion into one macrospin torque is less direct than the exchange spin
torque because SOC, crystal-field anisotropy, and intersite flux participate.
Do not add spin and orbital torques to LLG without stating a coupling model.

### Gauge Comparisons

In a complete basis with consistent relaxation and every contact term, LG and
VG describe the same physics.  In this finite localized basis they need not be
numerically identical.  VG also contains explicit `1/Omega` and `1/Omega^2`
factors, so low-frequency cancellations are sensitive to basis completeness
and contact terms.  LG is often numerically calmer as `Omega -> 0`; that alone
does not prove that an omitted LG intraband term is physical zero.

The preferred validation ladder is:

1. Check Hermiticity, operator identities, and tensor symmetry.
2. Check channel sums and projections against the unprojected result.
3. Compare LG-DM with VG-DM on a converged BZ grid.
4. Compare VG-DM with perturbative TDSE at weak field.
5. Compare perturbative TDSE with Bloch acceleration at small `E0`.

## Projections and BZ Output

`responseOmegaIndexList` and `responseFermiIndexList` select the frequencies
and Fermi energies for which a full k-resolved response is retained.  This is
essential because storing every point of a 3rd-rank tensor for every scan value
can dominate memory.

The standard task result key is `"projected response"`; channel-resolved task
helpers also expose their channel arrays.  Each phase-grid task owns only part
of the BZ, so the main driver must place each returned block using its `iq`
index before writing a distribution.

Important output helpers in `fileio.py` are:

- `outputIFEcoeffs`: contracts IFE with common linear/circular polarizations.
- `outputIFTcoeffs`: contracts IFT and converts Cartesian torque to
  `(tau_r, tau_theta, tau_phi)` for a specified magnetic direction.
- `outputIFEResponseOnBZ`: writes polarization-contracted `(M_x,M_y,M_z)`.
- `outputIFTResponseOnBZ`: writes polarization-contracted spherical torque.
- `outputBZDistributionData`: generic tensor/vector BZ writer.

BZ coordinates are derived from `reciLattice`.  `bzRepeat=(1,1,0)` repeats a
2D response over neighboring reciprocal cells so Gamma appears at the center
of a conventional plot.  The specialized IFE/IFT writers also generate quick
figures for `x`, `y`, `xy+`, and `xy-` when the grid is two-dimensional.

## Parallel Execution

### Brillouin-Zone Tasks

`BZGridTask` decomposes a target BZ grid into FFT subgrids with phase shifts.
Its default `parallelBackend="process"` gives each worker a separate Python
interpreter. `parallelBackend="thread"` selects the previous thread backend.
The same backend handles AHC, SHC, BPVE, IFE, and IFT; no response formulas change.

```python
calc = BZGridTask(model, solveIFTorque_LG_DM, args=settings, BZGrid=(99,99,1),
                  ncores=16, parallelBackend="process", workerThreads=1)
```

The model and common settings are installed once per process. Only phase data
are sent with each job; queued jobs are bounded, and results retain phase order
for reproducible summation. `iq`, response tensors, channels, and BZ records
keep their existing API. Worker failures propagate with the failed phase index.
`ncores` is capped by the number of phase tasks; one task/worker uses the serial
path. The Windows process pool also has the Python limit of 61 workers.

Install `threadpoolctl` with `python -m pip install threadpoolctl`. For parallel
runs, `workerThreads=1` limits loaded BLAS/OpenMP libraries and sets the worker
environment to prevent nested numerical thread teams. These settings are
restored on pool exit. Single-worker runs retain the caller's library settings.

On Linux, a caller with one Python thread uses `fork` to share the model arrays
copy-on-write. Otherwise the default is `spawn`, including on Windows. Set
`processStartMethod="spawn"` explicitly when the parent uses background/native
threads or libraries that are unsafe after fork. Spawn copies the model once
per worker and needs more startup time/RAM. Use an importable task function and
an `if __name__ == "__main__":` entry guard. Treat the model as read-only during
BZ integration, and do not launch overlapping pools from threads in one parent.
The angle-mapping outer process pool should continue to use one BZ worker per
orientation unless nested parallelism is deliberately budgeted.

On the Linux server, the reduced FeCl2 IFT sample (one Fermi point, six photon
energies, 81 phase tasks) took 36.72 s with the old 96-thread pool and 5.69 s
with 16 processes, with identical tensors. A fresh 81-process run averaged
about 68 busy cores but took 10.22 s. More CPU utilization does not necessarily
mean shorter runtime: startup, cache/memory contention, and workload size matter.
Benchmark several worker counts for each material/grid. See `SERVER.md` for
measurements and remote usage, `benchmark_bz_parallel.py` for repeatable timing,
and `test_bz_parallel.py` for serial/thread/fork/spawn numerical checks.

### Angle Scans on Slurm

`angleDependentSOTMapping.py` supports:

```text
--stage prepare    determine shared fixed-number/filling data
--stage compute    compute one deterministic orientation shard
--stage aggregate  collect checkpoints and fit harmonics
--stage all        local end-to-end execution
```

`slurm_scripts/slurm.angleDependentSOTMapping.sh` launches one Slurm task per
node and local multiprocessing inside each node.  Change `#SBATCH --nodes=N`
to select the node count.  `--cpus-per-task` sets the allocated cores per node;
`ORIENTATION_WORKERS` controls the independent orientation processes and
`BZ_WORKERS_PER_ORIENTATION` controls the shared-model BZ threads.  Their
product must not exceed `--cpus-per-task`.  The memory-conservative default is
one orientation process with all allocated cores assigned to its BZ task;
increase `ORIENTATION_WORKERS` only after measuring per-node memory use.  The
legacy `WORKERS_PER_NODE` environment variable remains an alias for
`ORIENTATION_WORKERS`.  The script sets BLAS/OpenMP thread counts to one.

Raw electronic results are saved as `orientation_XXXX.npz`; fitting is a later
stage.  Changing the spherical-harmonic degree should not require recomputing
electronic checkpoints.  If aggregate rejects a checkpoint after only a fit
setting changed, inspect the separation between electronic and fitting
signatures rather than deleting all raw data.

## Angle-Dependent IFT Mapping

The production mapping workflow uses approximately uniform Fibonacci-sphere
directions and a two-pass fixed-electron-number procedure:

1. Solve the electronic spectrum for each magnetic direction and determine the
   chemical potential that preserves the target electron count.
2. Compute IFT at that direction and chemical potential.

The magnetic anisotropy energy is fitted with scalar spherical harmonics.  The
tangent torque is fitted with vector spherical harmonics (VSH).  Crystal
symmetry is enforced during fitting; projected responses use the subgroup that
preserves the selected atoms/projection.  The aggregate output reports fit and
cross-validation errors, including polarization-contracted errors.

`AngleDependentDynamicsMap.npz` is the compact input to `IFTDevice`.  Keep the
raw `orientation_XXXX.npz` files: they are required for refitting, raw torque
maps, outlier diagnosis, and reproducibility.

Near-degenerate bands can produce isolated orientation spikes if the band
gauge or degeneracy handling becomes unstable.  Such a point must be
recomputed directly and converged with k grid, `degen`, `eta`, and temperature;
do not assume an outlined arrow is merely a plotting error.  The raw-vs-VSH
plots in `LLGInput.py` show black first-principles arrows, fitted arrows, and
pointwise residuals.

## Magnetic Dynamics

`SingleMomentLLG` integrates the explicit Gilbert equation for a unit
macrospin.  The implicit Gilbert derivative is algebraically isolated before
integration, and the current integrator is hand-written RK4.  `time_step` may
be a positive constant or a callable `time_step(time,m)`, which permits small
steps during femtosecond pulses and large steps between a pulse train.

`SOTDevice` provides a phenomenological current-driven field-like and
damping-like SOT model.  `IFTDevice` loads `AngleDependentDynamicsMap.npz`,
selects/interpolates `omega0`, contracts a chosen optical polarization, applies
an electric-field envelope, and feeds the resulting exchange torque to LLG.
It supports:

- `external_field=(Hx,Hy,Hz)` in tesla.
- `use_DFT_mae=True/False` to include or omit the fitted electronic MAE.
- `extra_uniaxial_mae` in eV and `extra_uniaxial_mae_direction` to add a manual
  uniaxial energy `K[1-(m dot u)^2]`.

`LLGInput.py` can plot regular and raw tangent torque maps, raw-vs-VSH errors,
and magnetic trajectories. `plot_tangent_torque_map` now defaults to
`torque_mode="total"`: at every orientation it freezes `drive_amplitude`, adds
the active DFT/manual anisotropy, external/additional fields and background
torques, and evaluates the same explicit Gilbert derivative used by the RK4
trajectory. `map_time` selects the time at which any non-drive time-dependent
terms are sampled. Use `torque_mode="drive"` for the historical source-only
map; the raw electronic torque map remains source-only by definition.

Optional phase-portrait analysis uses the same selected constant-drive field,
identifies stable/unstable/saddle fixed points, estimates attraction basins, traces stable
and unstable separatrices, and checks the Poincare-index sum on the sphere.
Use these precise dynamical-systems objects if describing a torque-map
"topology" in a paper. Built-in SOT and fitted IFT/MAE fields are evaluated in
vectorized direction batches during basin integration. Every result records
`stage_times_seconds` for map sampling, fixed-point refinement, separatrix
tracing, basins, and the total analysis. Pass `topology_options={"workers": N}`
to `plot_tangent_torque_map` to use `N` threads for independent fixed-point
refinements. Threading can be slower for Python-bound fitted fields, so compare
the reported timings before increasing it. The search uses all grid-local
minima, the two poles, and `candidate_padding=8` additional low-magnitude
seeds; `maximum_candidates` is only a safety cap. Basin integration is already
batched. `compute_basins=False` is the largest speed-up when only fixed points
and separatrices are needed. `candidate_padding`, `maximum_flow_steps`, and the
map's `theta_count`/`phi_count` control the main accuracy/cost tradeoffs and
should be converged for publication calculations.

The current dynamics is deterministic.  Thermal fluctuations may motivate an
initial tilt, but a stochastic thermal field/noise integrator has not yet been
established as a production feature.  Do not present a chosen initial angle as
a finite-temperature simulation.

## Symmetry

Symmetry is both a physical constraint and a numerical diagnostic.  Check it at
three levels:

1. Real-space Hamiltonian/overlap/connection matrices.
2. k-space operators and magnetic little-group transformations.
3. Final response tensors after polarization contraction.

`System.symmerizeHamAndOverlap` and related tensor symmetrizers act on real-space
objects.  For spin-separated collinear plus SOI models, symmetrize the
spin-independent Hamiltonian and Cartesian orbital-SOI coefficients before
assembling a direction-dependent spinor Hamiltonian.  The SOI Cartesian index
is an axial-vector index, not three unrelated scalar Hamiltonians.

Wannier position-matrix symmetrization is more delicate than Hamiltonian
symmetrization because gauge and orbital-center translations enter.  Earlier
tests found that forcing a TB-diagonal position sometimes restored expected
response constraints.  Therefore, a large `AA_R_W` symmetrization residual is
not cosmetic: validate the connection transformation and the final tensor
before trusting a Wannier LG result.

## Validation Models and Tests

Useful regression targets under `mcalc/` include:

- `2D-random-spin-ife`: low-symmetry off-diagonal LG/VG/TDSE tests.
- `2D-rashba-ife`: metallic and full-covariant-derivative tests.
- `BN`: four-band C3 honeycomb model with exchange and Rashba hopping; useful
  for tensor symmetry and optical off-diagonal response.
- `FeCl2`: spin-separated Wannier plus patched-VASP SOI, exchange rotation,
  angle mapping, symmetry, and LLG validation.
- `PbSn`: strong-SOC stress test for the collinear plus SOI approximation.
- `MnBi2Te4`: realistic IFE/IFT mapping and switching workflow.
- `Pt`: dense 3D Hall-response and parallel-scaling benchmark.
- `Pb`: FHS/spin-Hall comparison case.

The repository uses executable regression scripts rather than one unified test
runner.  Tracked examples include:

```powershell
python test.densityMatrixKernels.py
python test.hallDensityMatrixKernels.py
python test.ifeVGFullCovariant.py
python test.rawTangentTorqueMap.py
python test.torqueTopology.py
python test.mnbi2te4TorqueTopology.py
```

Additional diagnostic scripts may exist locally or under `tmp/`.  Before a
large refactor, run the relevant toy-model comparisons and record numerical
tolerances.  Do not use a realistic material calculation as the first test of
an index or sign change.

## Established Numerical Checks

At the time of this handoff, focused regression work has established:

- Historical and optimized interband Hall contractions agree to machine
  precision in the AHC test.
- The old resonant and LG-DM SHC agree in the clean-limit test up to the
  expected finite-`eta` difference.
- Full-covariant IFE channel sums and their BZ integration agree with direct
  reference contractions.
- Full VG IFT agrees with perturbative TDSE; omitting high-order observable
  vertices does not.
- Optimized BPVE VG reproduces its previous DM result, and LG observable
  batching preserves component traces.
- For bcc Fe at 300 K, the first on-shell SERTA implementation gives diagonal
  `I`-channel conductivities of `2.55e7`, `2.12e7`, and `2.00e7 S/m` on
  `30^3`, `60^3`, and `90^3` target grids. A linear-in-inverse-grid estimate
  is about `1.70-1.75e7 S/m`, compared with VASP's `1.6808e7 S/m`. Treat this
  as encouraging convergence evidence, not a substitute for a denser source
  self-energy grid and an explicit target-grid convergence study.

These checks are guardrails, not proof for every material.  Repeat convergence
and symmetry tests whenever the basis, degeneracy threshold, temperature,
broadening, or exchange provenance changes.

## Known Limitations

1. Nonorthogonal/OpenMX overlap corrections are less thoroughly validated than
   the orthogonal TB/Wannier paths.
2. LG/VG equivalence is limited by finite basis, incomplete high-energy bands,
   relaxation modeling, and the completeness of contact vertices.
3. Full covariant derivatives in metals require dense BZ sampling; coarse-grid
   disagreement is expected but must converge systematically.
4. The collinear plus SOI/WISOI construction is a frozen-potential and
   frozen-Wannier-subspace approximation.
5. Bloch acceleration is nonperturbative in the applied field but still uses a
   fixed Hamiltonian.  It cannot self-consistently reverse the exchange field.
6. Site-resolved exchange torques need a physical site decomposition of
   `H_XC`; anticommutator projection of a total torque is only approximate.
7. Orbital angular momentum in the current TB architecture is primarily local
   circulation.  Its transfer to a collective magnetic order is model
   dependent.
8. Angle-map VSH errors and isolated degeneracy-induced outliers must be
   reported and converged, not hidden by interpolation.
9. The API retains historical spelling such as `AnormalousHallConductance`.
   Rename only with a compatibility plan.
10. The present SERTA path interpolates diagonal on-shell VASP self energies;
    it does not yet evaluate scattering rates from the full EPI matrix on an
    arbitrary target grid, include a scattering-in term, or solve the iterative
    Boltzmann equation.

## Rules for Future Development

1. Preserve the nonlinear tensor order `[drive_1, drive_2, measure]`.
2. Keep LG/VG/TDSE implementations independent enough to remain useful as
   cross-checks; do not label copied VG algebra as LG.
3. Put reusable density-matrix algebra in `calculator/density_matrix.py`, not
   separately in every observable calculator.
4. Preserve `I/E` and `II/EI/IE/EE` channel outputs when adding an observable.
5. Include all gauge-required Hamiltonian and observable contact vertices in
   VG.  A naive `Tr[O rho2]` is generally incomplete when `O` depends on `A`.
6. Treat spin-frame rotations, active magnetization rotations, and crystal
   symmetry rotations as distinct operations.
7. State the exchange provenance in every IFT/LLG calculation.
8. Validate formulas on toy models with symmetry, LG/VG, and TDSE checks before
   launching a large material or angle scan.
9. Keep response-index filtering for k-resolved output to control memory.
10. Benchmark parallel scaling and memory, not only thread count.
11. Never delete raw orientation checkpoints after fitting; they are the
    reproducible first-principles data.
12. When preparing a paper, report k grid, Fermi level or fixed filling, `kBT`,
    `eta`, magnetic-angle sampling, harmonic degree, fit/CV errors, polarization
    convention, pulse envelope/FWHM convention, LLG parameters, and integration
    step policy.

## Suggested Next Reading

For a new coding session, inspect these files in order:

1. `README.md` (this document).
2. The model block currently active in `spinHallCondInput.py` or
   `angleDependentSOTMapping.py`.
3. The target final calculator in `calculator/Grid.py`.
4. Its kernel in `calculator/density_matrix.py`.
5. The corresponding `solve*` helper in `task.py` and output helper in
   `fileio.py`.
6. The closest toy-model regression script.

The central design principle is simple: construct physically meaningful
operators in the system layer, construct reusable response density matrices in
the kernel layer, contract observables in calculator classes, and let task and
output layers handle parallel placement and presentation.  Keeping those
boundaries intact is the fastest way to continue the project without losing
the cross-checks accumulated so far.
