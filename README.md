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
Typical dependencies include NumPy, SciPy, Matplotlib, and the compiled
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
the generalized derivative `r_deriv_K` used by full LG covariant derivatives.
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

`BZGridTask` decomposes a target BZ grid into FFT subgrids with phase shifts and
uses a `ThreadPoolExecutor`.  Threads share the model and avoid serializing its
large arrays, while NumPy/compiled kernels can release the GIL.

Scaling is not guaranteed up to the physical core count.  A 96-thread DM run
can become limited by memory bandwidth, allocator pressure, or large private
calculator arrays even when a simpler legacy SHC kernel reaches 96 cores.  The
optimized DM contractions reduce that pressure, but production runs should
still benchmark several `ncores` values.  Also set BLAS/OpenMP thread counts to
one when the outer BZ task already supplies parallelism, or deliberately use a
smaller outer pool with a measured BLAS team.

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
and magnetic trajectories.  Optional phase-portrait analysis identifies
stable/unstable/saddle fixed points, estimates attraction basins, traces stable
and unstable separatrices, and checks the Poincare-index sum on the sphere.
Use these precise dynamical-systems objects if describing a torque-map
"topology" in a paper.

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
