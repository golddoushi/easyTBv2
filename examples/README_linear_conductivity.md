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
```

It writes `tmp/honeycomb_conductivity/summary.json` and `conductivity.npz`.
An example run with the defaults above gave relative maximum errors of
`7.2e-16` for dc LG/VG, `7.8e-7` for optical LG/VG, and `9.7e-5` for
TDSE/VG on the common 4x4 mesh. These are comparisons at the chosen meshes,
temperature, broadening, and time step, not universal accuracy guarantees.

`task.py` also exports `solveAHC_VG_DM`, `solveOpticalconductivity_LG_DM`,
`solveOpticalconductivity_VG_DM`, and `solveOpticalconductivity_TDSE` for
`BZGridTask`. Use the existing `fermi`, `omega`, `collision operator`,
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

The DM variants use `CollisionOperator` for CRTA or frozen secular SERTA.
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
changes. Turning the toy-model Fan self-energy into a transport collision
operator, including scattering-in/vertex corrections, remains separate work.
