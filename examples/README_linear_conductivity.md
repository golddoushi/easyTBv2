# Linear conductivity: LG, VG, and real time

The new calculators exported by `calculator` are:

| Calculator | Output `sigma` axes | Method |
|---|---|---|
| `AnormalousHallConductance_VG_DM` | Fermi, drive, current | Analytic dc limit of first-order VG |
| `Opticalconductivity_LG_DM` | Fermi, photon energy, drive, current | First-order LG |
| `Opticalconductivity_VG_DM` | Fermi, photon energy, drive, current | First-order VG with contact current |
| `Opticalconductivity_TDSE` | Fermi, photon energy, drive, current | Real-time VG ensemble propagation |

Here “optical conductivity” means the finite-frequency linear conductivity,
not the second-order dc photocurrent calculated by the BPVE classes. `sigma` is in
S/m; the optical result is complex. `projected_sigma` has two leading
projection axes, with `[0,0]` the total response. The last two axes follow
the existing dc API: electric-field direction first, current direction second.
Use `(sigma[...,0,1]-sigma[...,1,0])/2` for the antisymmetric xy component.
The 2D model uses its three-dimensional cell volume including vacuum; multiply
S/m by the cell height in metres for sheet conductance.

## Example

```python
import numpy as np
from calculator import Opticalconductivity_LG_DM, Opticalconductivity_VG_DM, Opticalconductivity_TDSE

options = dict(model=model, kgrid=(24,24,1), eta=0.2, kBT=0.15,
               fermiList=np.array([-0.7]), omegaList=np.array([0.6,1.3]),
               fullCovariantDerivative=True)
lg = Opticalconductivity_LG_DM(**options)
vg = Opticalconductivity_VG_DM(**options)
# Use a small grid initially; compare TDSE with VG on that same grid.
small = dict(options, kgrid=(4,4,1))
reference = Opticalconductivity_VG_DM(**small)
td = Opticalconductivity_TDSE(E0=1e-4, n_steps_per_cycle=256, **small)
```

Run the standalone benchmark:

```text
python examples/honeycomb_conductivity.py --grid 32 --tdse-grid 4
python examples/honeycomb_conductivity.py --grid 32 --skip-tdse --snapshots 20 --supercell 2
python examples/honeycomb_conductivity.py --grid 24 --serta both --skip-tdse
python examples/honeycomb_conductivity.py --grid 12 --serta static --snapshots 200 --supercell 2 --skip-tdse
```

It writes `tmp/honeycomb_conductivity/summary.json` and `conductivity.npz`.
An example run with the defaults above gave relative maximum errors of
`7.2e-16` for dc LG/VG, `7.8e-7` for optical LG/VG, and `9.7e-5` for
TDSE/VG on the common 4x4 mesh. These are comparisons at the chosen meshes,
temperature, broadening, and time step, not universal accuracy guarantees.

`task.py` also exports `solveAHC_VG_DM`, `solveOpticalconductivity_LG_DM`,
`solveOpticalconductivity_VG_DM`, and `solveOpticalconductivity_TDSE` for
`BZGridTask`. Use the existing `fermi`, `omega`, `collisionSpecification`,
`record fermi ids`, and `record frequency ids` parameter keys. TDSE controls
use their Python argument names. Use `BZSymm=False` and average the phase-grid
results, preserving their complex values.

## Conventions and the dc sign correction

Photon energies `Omega` and linewidths are in eV. The Fourier convention is
`exp(-i*Omega*t)` with time in hbar/eV. The physical electron Hamiltonian is
`H(k+A)`, electric field is `E=-dA/dt`, and charge current is proportional to
`-Tr[rho*v(A)]`. Kernels return **minus the physical first-order density** so
their contraction with `+v` is the charge-current response.

For `Delta_mn=E_m-E_n` and `Df=f[E_m,E_n]*v`, the first-order current-weighted
density is `i*Df/(Delta-Omega-i*Gamma_mn)`. Its diagonal gives the Drude
conductivity `(-f')*v_a*v_b/(Gamma_n-i*Omega)`. The full finite-temperature
kernel uses the stable divided difference, including the `f'` limit at
degeneracy, rather than dropping small energy gaps. I/E denote diagonal and
off-diagonal contributions; only the total is invariant under arbitrary
rotations of exactly degenerate bands.

The older dc LG kernel had `+i*(f_n-f_m)` in its E-channel numerator, while
its I-channel already used the positive-conductivity convention. The E term
gave negative longitudinal absorption. It is corrected to `-i*(f_n-f_m)` in
both the block kernel and the optimized contraction. **Previously computed
first-order DM E-channel tensors, including Hall and shared-kernel spin-Hall
responses, change sign.** The I channel and its VASP SERTA transport validation
are unchanged. The separate historical Berry-curvature calculators are not
modified here. When comparing Hall signs, also account for the drive/current
axis ordering. Existing dc LG still applies its historical small-gap cutoff;
the new full optical kernels retain the finite degenerate limit.

## Why the VG contact and relaxation terms matter

The VG equilibrium must transform with the field, and the observable includes
the diamagnetic contact term. This requirement, including its finite-band
formulation, is discussed by [Passos et al., Sections III–V and Appendix A](https://arxiv.org/html/1712.04924).
Within a consistently coupled finite tight-binding model, a finite number of
bands by itself does not force the gauges to disagree.

In this implementation the physical density per vector potential is

```text
R(Omega) = -([v,f] + i*Collision(Df)) / (Delta-Omega-i*Gamma)
sigma    = (i/Omega) * BZ_average(Tr[R*v_current] + Tr[f*w_current,drive])
```

Here `w[b,a]=D_a v_b` is the existing covariant velocity derivative. The
equilibrium residue `Tr[Df*v+f*w]` integrates to zero over a periodic BZ.
For numerical stability the kernel analytically evaluates
`(R(Omega)-R(0))/Omega`, and the calculator restores the residue separately.
The `C` channel is this `i*residue/Omega` contribution. It is retained at
nonzero frequency, so mesh errors are observable. `sum_rule_residual` stores
the integrated residue in (S/m)*eV. At exactly zero frequency the equilibrium
pole is subtracted analytically, giving the periodic-BZ dc limit; this avoids
an arbitrary tiny probe frequency. Consequently dc LG/VG agreement alone is
not an independent sum-rule convergence test. Check finite-frequency results
and the residual as well, especially near zero frequency.

The DM variants use `CollisionSpecification` for CRTA or frozen secular SERTA.
VG requires the full covariant response, positive electronic temperature,
an orthogonal model, and the complete model basis (`skip_low=False`). Its
contact vertices currently support total current only; LG also supports the
existing current projections. Optional DM `projected_response` records remain
unscaled k-resolved integrands, matching the dc convention.

## Time-domain benchmark and EPC scope

The TDSE variant does not call a response kernel. Each time step propagates
the occupied ensemble with a unitary exponential of
`H0+A*v+A^2*w/2`, together with relaxation toward the independently evaluated
spectral function `f(H(A))`. This is a density ensemble of TDSE trajectories
with relaxation, not isolated pure-state evolution. CRTA gives an exact
frozen-H step; SERTA uses symmetric splitting with frozen equilibrium-band
linewidths. These approximate a midpoint time step with second-order time
accuracy. `E0` is the **peak** field in V/Angstrom; it is not the positive-
frequency amplitude used in the older BPVE TDSE class.

After `transient_lifetimes` relaxation times, the code measures an integer
number of cycles. Opposite fields remove even-order current terms, and
`2*mean(j_odd*exp(i*Omega*t))/E0` gives the complex conductivity. The current
vertex is `v+A*w`; the finite-field Hamiltonian/current Taylor expansion is
intended for convergence to **first order**, not strong-field dynamics.
Refine `n_steps_per_cycle`, reduce `E0`, and increase the discarded transient
to test convergence. `max_phase_step` can increase the requested step count;
inspect `integration_info`. Steady-state extraction requires positive photon
energies and strictly positive relaxation linewidths. It deliberately rejects
dc and undamped cases. `recordCurrent=True` stores time, field, and current
divided by the peak field in S/m for each `(fermi,omega,drive)`.

The optional snapshot example averages displaced SSH supercells at fixed CRTA
eta and reports the equilibrium supercell result plus separate real/imaginary
sampling errors. It is a **static disorder benchmark**, not an implementation
of inelastic EPC transport. Two snapshots are only a smoke test. Converge the
ensemble, supercell, k mesh, and residual broadening before interpreting EPC
changes.

## Toy-EPC SERTA linewidths

`ModelEPCLinewidth` turns the diagonal on-shell imaginary Fan self-energy into
a sampled linewidth source:

```python
from calculator import CollisionSpecification, ModelEPCLinewidth

source = ModelEPCLinewidth(qgrid=(12,12,1), temperature=300,
                           fermi=-0.7, delta=0.02, adiabatic=False)
collision = CollisionSpecification("serta", linewidthSource=source)
sigma = Opticalconductivity_LG_DM(collisionSpecification=collision, **options).sigma
```

It evaluates the calculator's k mesh and caches the result for LG, VG, and TDSE
instances with identical eigenenergies. A shifted `BZGridTask` supplies its
`kOffset`. `adiabatic=False` is standard dynamical SERTA: the linewidth is
`-2 Im Sigma_Fan(epsilon_nk)` with phonon emission/absorption energies and Pauli
factors, as in the [VASP SERTA definition](https://vasp.at/wiki/ELPH_SCATTERING_APPROX).
`adiabatic=True` gives an elastic Born rate for static Gaussian displacements.
We call this **static/adiabatic RTA**, since “SERTA” normally refers to the
dynamical on-shell Fan rate. Its Lorentzian `eta` approximates the elastic
delta function and must be converged together with the q mesh.

The source supplies only Gamma. The transport Hamiltonian, energies,
occupations, and current vertices remain bare. It does not add Re(Fan)+DW or
optical vertex/scattering-in corrections.

## Frozen supercells versus the two rates

Agreement of static Fan+DW **band-energy shifts** with stochastic snapshots
does not imply equal conductivity. That comparison concerns a one-particle
pole shift through second order. Conductivity is also sensitive to eigenstates,
current vertices, and how scattering redistributes momentum.

A frozen calculation evaluates the full independent-electron conductivity of
each `H[u]` and then averages. It contains real spectral changes, modified
eigenvectors/current vertices, elastic state mixing allowed by the supercell,
and all displacement orders in `H[u]`. Its CRTA `eta` is an additional residual
broadening. Static/adiabatic RTA keeps only the elastic Born **out-scattering
lifetime** in a diagonal secular collision operator. It omits scattering-in
and vertex corrections. They therefore are not formally equal.

They may approach one another for weak EPC, large supercells, and slow classical
phonons after matching displacement covariance, k/q resolution, chemical
potential, and broadening, if vertex corrections are negligible. Dynamical
SERTA resolves phonon energy exchange and Pauli blocking. A static snapshot
cannot reproduce these, except approximately when `kBT >> hbar*Omega` and the
electronic probe cannot resolve the phonon dynamics.

The example now makes meshes commensurate: an `S x S` supercell uses
`(grid/S) x (grid/S)`, sampling the same primitive momenta. The undisplaced 2x2
supercell then agrees with the primitive 12x12 tensor to about `5.9e-15`
relative error. At 300 K, 200 frozen 2x2 snapshots differed from pure static
RTA by about 38% under the report normalization. That raw comparison also
replaces the snapshots' 0.2 eV CRTA background by the EPC rate. Adding the same
0.2 eV background to static RTA reduces the discrepancy to about 10.8%. The
largest componentwise sampling standard error was about 1.6% of the response,
and the raw discrepancy reached roughly 51 standard errors. The example stores
both comparisons. This finite benchmark rejects automatic equality while also
showing that unmatched background broadening can dominate the apparent error.

`--tdse-collision static|dynamic` makes the time-domain comparison use model
EPC rates. Small linewidths require very long transients, so this optional path
applies `--tdse-min-linewidth 0.02` eV by default and reports the actual minimum.
Set it to zero for unmodified SERTA and converge the transient. DM rates are
never floored.

A diagram-by-diagram snapshot comparison requires a static-disorder collision
kernel with scattering-in/vertex terms, or an explicit second-order expansion
of ensemble conductivity. The present comparison diagnoses the diagonal
lifetime approximation; it does not establish it as the exact frozen-phonon
limit.
