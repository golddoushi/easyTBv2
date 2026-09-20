# Chain BPVE with EPC

`chain_BPVE_alignment.py` compares the longitudinal rectified response
`xxx`, at output frequency zero and input photon energies `(-omega,+omega)`.
It uses the existing BPVE LG/VG DM calculators, Energy eigensystems, chain
model, phonons and displacement sampler. It does not implement another
electronic eigensolver.

Run from the repository root:

```powershell
python examples/chain_BPVE_alignment.py --grid 512 --qgrid 16 --supercell 16 --pairs 20 --scale 0.25 --phonon-energy 0.02 --temperature 300 --background 0.00658 --delta 0.05 --fermi -0.7 --omega 0.8 --curvature
```

For a cheaper example with a broader background:

```powershell
python examples/chain_BPVE_alignment.py --grid 512 --qgrid 8 --supercell 8 --pairs 40 --scale 0.125 --background 0.026 --omega 0.8 --curvature --output tmp/bpve_broad_weak
```

Positive pump frequencies are required: `--omega 0` is not a DC BPVE
benchmark. Multiple photon energies can be supplied. Other useful options:

- `--pairs 0` skips stochastic sampling; combine with `--curvature` for the
  deterministic leading displacement correction.
- `--skip-vg` omits the primitive-cell VG cross-checks.
- `--fixed-centers` also samples hopping changes with the position vertices
  fixed, to separate changes of optical vertices from hopping disorder.
- `--curvature-step 0.001` refines the finite-displacement derivative used by
  `--curvature`. This is a numerical step, independent of `--scale`.

## Quantities and conventions

The script evaluates CRTA, static SERTA, dynamic SERTA, balanced RTA, full
nonlinear Pauli collision, and snapshots with CRTA background. Each RTA is
evaluated both with historical frozen linewidth derivatives and with the new
covariant derivatives. Both gauges are evaluated unless `--skip-vg` is set.
Main comparison columns always use LG, so the reported derivative effect is
not contaminated by changing gauges.

**The BPVE DM calculators currently return raw band traces, not the SI
conductivity returned by Opticalconductivity.** The script divides supercell
responses by the number of primitive cells. It checks equilibrium folding and
reports everything in the raw calculator convention per primitive cell. No
additional volume factor should be applied to these comparisons. An SI
conversion is not provided here.

`--scale` changes masses so displacement amplitudes and EPC matrix elements
scale together. The reference masses are 12 and 14 amu at 40 meV. When phonon
energy changes, masses are also adjusted to preserve the quantum equal-time
displacement covariance at the chosen temperature. Thus this particular
phonon-energy scan deliberately isolates dynamics rather than changing the
static thermal variance. Actual masses and RMS hopping changes are recorded.

`--background` is included once in every method. `--delta` broadens the energy
conservation condition in EPC rates; it is not an additional response
linewidth and does not enter snapshots. For SERTA, the script explicitly adds
the background to the EPC rates. Balanced RTA receives the diagonal population
damping from the full operator, which already includes the background.

Balanced RTA omits scattering-in and the nonlinear collision Hessian. Its
coherences use the RTA half-sum of population linewidths. The full model uses
its own secular coherence damping and nonlinear population/coherence terms.
Full minus balanced RTA therefore does not isolate scattering-in alone.
Full also projects onto fixed particle number, whereas the RTA target is the
fixed-chemical-potential equilibrium. Snapshots share the same chemical
potential; their equilibrium carrier count is not constrained across samples.

## Linewidth derivative implementation

The new opt-in collision setting is:

```python
CollisionSpecification(
    'serta', linewidthSource=epc_rates + background,
    secondOrderLinewidthDerivative='covariant',
)
```

It requires `fullCovariantDerivative=True`, a complete orthogonal model basis,
and a resolved electronic mesh. The default `'frozen'` preserves previous
calculations. First-order calculators are unaffected by this option.

For the existing half-sum damping, define the band-space matrix
`G = diag(Gamma_n)` and the positive damping map `D[X] = {G,X}/2`.
Differentiating it requires `(D_k D)[X] = {D_k G,X}/2`. The code transforms
`G` to the orbital basis, differentiates it spectrally on the electronic
mesh, includes the orbital position connection, and transforms back. This
includes both linewidth gradients and derivatives of band projectors; simply
differentiating the numbers `Gamma_n(k)` would omit the latter.

The first-order equation is differentiated analytically with this additional
term. Electronic Fermi derivatives remain analytic. A constant identity is
subtracted before the FFT, preserving the exact CRTA limit. The VG response is
assembled through the same gauge transformation as the existing Markov
second-order kernel. These operations change the k derivative, not the rates,
their temperature, or their dependence on driven occupations. RTA still has
no nonlinear collision Hessian.

## Saved results and convergence

The output directory contains:

- `comparison.csv`: complex BPVE and EPC changes for every photon energy and
  method, including snapshot standard errors where available.
- `results.npz`: the CSV quantities plus linewidth arrays, LG/VG differences,
  the CRTA limit check, equilibrium-supercell folding error, individual
  antithetic pair means, and derivative-on minus derivative-off differences.
- `metadata.json`: all parameters, normalization, masses, timing and the full
  operator's stationary-state diagnostics.

EPC changes are measured relative to primitive CRTA for the frequency-domain
methods and undisplaced-supercell CRTA for snapshots. The folding error is
reported separately. One pair is two correlated samples, `u` and `-u`;
uncertainties are calculated across independent pair means. A single pair
cannot supply an error estimate.

The `snapshot_second_order` array is second order in **displacements**, while
every reported BPVE is second order in **electric field**. For independent
longitudinal Einstein atoms it evaluates
`sum_i <u_i^2> (d^2 chi/du_i^2)/2`; translation symmetry reduces the number
of required displacements to the two inequivalent primitive-cell atoms.

Converge electronic k points for the EPC correction, not only the pristine
response. Supercell disorder creates additional transitions which can need a
finer mesh. Separately converge q and delta for a continuum bath, or retain a
specified finite q grid to match a supercell. A primitive electronic grid can
be refined at fixed supercell size. The script permits unequal qgrid and
supercell for convergence scans but marks that mismatch.

Snapshots also change energies, eigenstates and optical vertices. Neither
linewidth-only SERTA nor the present full collision adds all real self-energy,
Debye-Waller or optical vertex corrections. Even in weak coupling their EPC
corrections need not coincide. A small RMS hopping change alone does not
guarantee a perturbative optical correction near a narrow resonance.

## Validation performed

`tests/test_bpve_linewidth_derivative.py` checks:

- CRTA recovery and converged LG/VG agreement;
- the exact identity `D_k G = 0.03 v` for `G = 0.4 I + 0.03 H`, which tests
  band-projector derivatives as well as eigenvalue derivatives;
- the differentiated first-order density matrix against a direct spectral
  derivative of the solved density matrix;
- primitive/supercell folding and the complete benchmark output path.

Together with `tests/test_epc_collision.py`, 19 tests passed.

## Numerical investigation (2026-09-08)

All numbers below are **EPC corrections**, not final response values. They
are raw xxx responses per primitive cell. These are finite-q benchmarks, not
claims of continuum phonon-grid convergence.

### Weak coupling with a broader background

Parameters: `qgrid=supercell=8`, scale 0.125, phonon energy 0.02 eV,
temperature 300 K, background 0.026 eV, delta 0.05 eV, Fermi energy -0.7 eV,
and photon energy 0.8 eV. The pristine response is -0.333071026.
The main sampling run uses k=512 and 40 independent pairs (80 snapshots).
The table's frequency-domain and curvature numbers use the k=2048 check.

| Scheme | Frozen linewidth derivative | Covariant linewidth derivative |
|---|---:|---:|
| Static SERTA | +0.00238893 | +0.02069105 |
| Dynamic SERTA | +0.00241667 | +0.01705234 |
| Balanced RTA | +0.00295712 | -0.00387520 |
| Full collision | — | -0.00535746 |

The snapshot correction is **+0.0106482 +/- 0.0018144** (one standard error,
k=512). The deterministic O(u^2) snapshot correction is **+0.0144597**.
The uncertainty is sampling uncertainty only; it excludes higher displacement
orders and finite-size effects. The residual difference from the curvature
result is about two sampling standard errors, so these samples do not cleanly
separate sampling fluctuations from higher displacement orders.

The deterministic snapshot correction changes from 0.01453107 at k=512 to
0.014459706 at k=1024 and 0.014459704 at k=2048. Thus its k=512 error is much
smaller than the sampling error, and the refined curvature result is stable.
The RTA and full corrections are also stable between k=1024 and k=2048.

Including derivatives changes the static and dynamic SERTA EPC corrections
by factors of approximately 8.66 and 7.06, respectively. Balanced RTA changes
sign. The correction is therefore not generically negligible even at this
small displacement amplitude. The full collision does not agree with the
positive snapshot correction here: it is not automatically a better static
snapshot approximation. The different bath, conservation, and optical-vertex
assumptions described above still matter.

Raw data: `tmp/bpve_broad_weak/`, `tmp/bpve_broad_weak_k1024/`, and
`tmp/bpve_broad_weak_k2048/`. The corresponding scale-0.25 runs are in
`tmp/bpve_snapshots/` (k=256) and `tmp/bpve_broad_k512/` (k=512 curvature).
At the common k=512, the covariant static/dynamic corrections divided by
scale squared change by about 0.6%/0.7% between scale 0.125 and 0.25. This
supports their leading weak-EPC scaling; it does not establish agreement
with the separate snapshot approximation.

### Narrower background and phonon-grid sensitivity

For background 0.00658 eV, scale 0.25, and otherwise the same temperature,
phonon energy, delta, Fermi energy and 0.8-eV photon energy, a q=16 bath gives:

| Scheme | Frozen EPC correction | Covariant EPC correction |
|---|---:|---:|
| Static SERTA | +0.02321573 | +0.02451461 |
| Dynamic SERTA | +0.02591075 | -0.00118079 |
| Balanced RTA | +0.04989319 | -0.18909612 |
| Full collision | — | -0.32731637 |

These numbers are stable between k=1024 and k=2048. At k=1024 and this photon
energy, covariant LG/VG differences are below 4e-10 for the three RTA methods;
the historical frozen paths differ by about 1e-4 to 2.4e-4. Small historical
LG/VG disagreement alone therefore does not bound the error from omitting
the derivative: both gauges can miss a much larger contribution.

The same narrow-background parameters with q=256 give static SERTA
0.02308997 -> 0.02120530 and dynamic SERTA 0.02629843 -> 0.02679538.
The derivative effect is then about 8.2% and 1.9% of the respective frozen
corrections, much smaller than the dynamic effect at q=16. These q scans do
not match a 16-cell snapshot ensemble; they test sensitivity to the bath mesh.
The full correction is still appreciably q dependent between q=128 and 256.

The q=16, k=512, 20-pair snapshot run gives -0.22341 +/- 0.03231. However,
its curvature correction changes from -0.29153 at k=512 to -0.38024 at
k=1024, despite an already stable pristine response. Therefore this narrow
background snapshot result must not be treated as a converged alignment
benchmark. Halving the curvature step at k=512 changes the correction only
from -0.291526 to -0.291500: this discrepancy comes predominantly from the
electronic mesh, not the finite-displacement step.

At 1.5 eV with the narrow background, even the pristine response remains
mesh dependent at k=1024 versus 2048. No converged comparison is claimed at
that frequency.

Raw data: `tmp/bpve_k512/`, `tmp/bpve_k1024/`, `tmp/bpve_k2048/`,
`tmp/bpve_q64/`, `tmp/bpve_q128/`, `tmp/bpve_q256/`,
`tmp/bpve_matched_snapshots/`, `tmp/bpve_snapshot_k1024/`, and
`tmp/bpve_curvature_step/`.
