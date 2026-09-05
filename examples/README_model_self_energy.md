# Model electron self-energies and frozen-supercell validation

`calculator.ModelElectronSelfEnergy` computes the leading retarded Fan-Migdal
(FM) and Debye-Waller (DW) contributions using `System.modelEPC` and the
prescribed phonon model. It does not mutate the electronic Hamiltonian, add
linewidths to a DM collision operator, or perform a self-consistent Dyson solve.

## Signed hopping convention

`fileio.readHrDat` stores the file entries directly in `ham_R`.
`Energy.solveEigenEquation` uses an inverse FFT, giving

```text
H(k) = sum_R ham_R(R) exp(+2*pi*i*k.R)
```

There is no implicit minus sign. The historical stored quantity is the signed
Hamiltonian matrix element, not a universally positive hopping magnitude.
The honeycomb example defines a positive parameter t and explicitly writes
`ham_R=-t`. This convention has been retained. EPC uses derivatives of the
stored Hamiltonian, so `alpha=dH/d(projected stretch)` has its literal sign.
The tests compare the direct Fourier sum against the `Energy` FFT.

## Fan-Migdal and Debye-Waller are distinct terms

For a zero-mean displacement with covariance C, isolated-level perturbation
theory gives

```text
<delta epsilon_n> = (1/2) sum_ab C_ab <n|H_,ab|n>
                 + sum_ab C_ab sum_(m != n)
                     <n|H_,a|m><m|H_,b|n> / (epsilon_n-epsilon_m).
```

The first term is DW, from curvature of the **Hamiltonian**. The second is
the static Fan term, from mixing of electronic eigenstates by the linear EPC.
Consequently averaging eigenvalues is different from diagonalizing the average
Hamiltonian: the latter only contains DW to this order.

In the original linear SSH model `H=H0+alpha*s`, with
`s=d_hat^0.(u_j-u_i)`, all second Cartesian displacement derivatives vanish.
DW is therefore exactly zero. Band shifts from static Fan are still present.
The first-order matrix g alone does not determine a nonzero DW contribution.

For a nonzero DW benchmark, a separate quadratic coefficient can be supplied:

```python
model.addModelEPC(directory, alphaR="epc", betaR="quadratic")
```

`quadratic_hr.dat` has the same scalar/spatial-index layout as `epc_hr.dat`,
with real reciprocal coefficients beta in eV/Angstrom^2. It defines

```text
H_ij(u) = H_ij(0) + alpha_ij*s_ij + (1/2)*beta_ij*s_ij^2.
```

Beta is an explicit model parameter. It defaults to zero and survives
supercell construction. Its file must use R points already present in `iRvec`
and existing Hamiltonian bonds, just as the alpha file does; incompatible
files raise `ValueError` without changing the model. The equilibrium bands and first-order g
are unchanged by beta. This expansion is in projected stretch; it does not
assume a dependence on the exact displaced bond length. There is no rigid-ion
reconstruction of beta from alpha.

## Self-energy formulas and normalization

The implementation follows the expressions on the
[VASP theory page](https://vasp.at/wiki/Electron-phonon_interactions_theory).
All electronic energies, phonon energies `w=hbar*Omega`, and eta are in eV.
The electron and phonon temperatures are the same, in kelvin. For an external
energy E the diagonal retarded Fan term is

```text
Sigma_FM[n,k](E) = sum_(q,nu,m) weight_q |g[nu,m,n](k,q)|^2
  * [(N_qnu+1-f_m,k+q)/(E-epsilon_m,k+q-w_qnu+i*eta)
   + (N_qnu+f_m,k+q)/(E-epsilon_m,k+q+w_qnu+i*eta)].
```

The full band matrix uses `g[nu,m,n].conj()*g[nu,m,nprime]` instead of
`abs(g)^2`. Quadrature weights sum to one, so a full uniform mesh has
`weight_q=1/Nq`. There is no extra spin factor or primitive-cell-volume factor.

`adiabatic=True` replaces the bracket with

```text
(2*N_qnu+1)/(E-epsilon_m,k+q+i*eta).
```

Only phonon energies in the denominators are omitted. Actual nonzero phonon
energies still set g's zero-point displacement scale and the Bose populations.
The temperature-zero limit is explicit and avoids dividing by kBT.

The DW term is the band transform of

```text
Sigma_DW(k) = (1/2) sum_(q,nu) weight_q (2*N_qnu+1) H''(k;q,nu;-q,nu).
H''_ij(k;q,nu;-q,nu) = sum_R exp(2*pi*i*k.R) beta_ij(R)
                      * |d_hat_ij(R).(e_j exp(2*pi*i*q.R)-e_i)|^2.
```

Here e includes the zero-point displacement length (Angstrom), as returned by
`model.modelPhonons.modeDisplacements(q)`. `secondDisplacementVertex` returns
H'' without the thermal factor or 1/2. DW is frequency independent and
Hermitian; its diagonal is real. This factor of 1/2 is checked against an
independently averaged quadratic supercell Hamiltonian.

## Usage and output shapes

```python
from calculator import ModelElectronSelfEnergy
from calculator.model_self_energy import uniform_qmesh
from examples.honeycomb_four_band import build_model

model = build_model("tmp/my_model")
kpoints = uniform_qmesh((12, 12, 1))
se = ModelElectronSelfEnergy(
    model, kpoints, qgrid=(12, 12, 1),
    temperature=300, fermi=0.0, delta=0.005, adiabatic=False)

fan = se.fan_migdal                # complex (nk,nb), evaluated on shell
dw = se.debye_waller               # real (nk,nb)
shift = se.energy_shifts           # Re(Fan+DW), eV
bands = se.renormalized_energies    # epsilon + shift; Z=1
width = se.linewidths              # -2*Im(Fan+DW), eV
gap = se.bandGap(occupiedBands=2)   # extrema only on the supplied k mesh
```

To inspect energy dependence, pass a common `energyList=[...]`. Then
`fan_migdal` and `total` have shape `(nk,ne,nb)`; `debye_waller` stays `(nk,nb)`.
`fan_migdal_matrix` and `total_matrix` always have shape `(nk,ne,nb,nb)`.
For on-shell calculations `ne=nb`, with the evaluation energy index separate
from both band indices; `fan_migdal` selects `Sigma_nn(E=epsilon_n)`.
`debye_waller_matrix` has shape `(nk,nb,nb)`.

Explicit `qpoints` and optional `qweights` may replace `qgrid`. All k and q
coordinates are fractional reciprocal coordinates. Use a full mesh unless
the quadrature and EPC transformations justify symmetry reduction.

Electronic states use the existing calculator pipeline. With `qgrid`, the
code constructs `Grid.Energy` once per distinct shifted mesh, using its
`kOffset` FFT-mesh support. All external k points differing by a q-mesh vector
reuse that calculator's states at both k and k+q. Thus the example with a
12x12 k mesh and matching q mesh needs one batched eigensolve of 144 states.
The `Energy` FFT sums R blocks that map to the same index on small grids,
including a 1x1x1 grid. With explicit `qpoints`, `Path.EnergyOnPath` solves the
unique k and k+q points together, identifying reciprocal-lattice duplicates.
The underlying calculators are available as `se.energy_calculators`.

The self-energy loop passes these eigenvectors to `getModelEPC`, preserving
the calculator's band gauge and avoiding repeated solves. A standalone call
to `getModelEPC` without eigenvectors also uses `EnergyOnPath`. The current
EPC model still requires an orthogonal basis; reusing these electronic
solvers does not add overlap-derivative EPC terms.

The retarded convention uses **+i eta**, independent of the conventions used
for some DM response denominators. Eta is a numerical regulator, not an EPC
relaxation time. It must be positive, and its convergence matters. Static
imaginary parts can diverge at finite-grid on-shell poles and must not be
interpreted as quantum transport rates.

The diagonal, on-shell, Z=1 band shifts assume isolated weakly perturbed
levels. At degeneracies use the full self-energy block at a common energy and
diagonalize within the degenerate manifold. A coarse mesh, near-resonant
denominators, or large coupling can invalidate a simple band-shift comparison.

## Reproducible stochastic comparison

```sh
python examples/honeycomb_self_energy.py
python -m unittest discover -s tests -p "test_model*.py" -v
```

The example generates both a linear and a quadratic model. It scales alpha
by s and beta by s^2, using s=0.05 by default to isolate the leading
displacement-order correction. It compares the half-filled band gap of a
3x2 supercell at Gamma with primitive-cell static self-energies on the exactly
matching 3x2 k/q mesh. This is a finite-system consistency benchmark, not a
converged infinite-crystal gap. The bare gap on this mesh is 1.1958807923 eV.

At each temperature, 2000 independent Gaussian displacement configurations
are paired with their negatives (4000 structures). Gap shifts are averaged
over pairs, and standard errors use the independent pair averages. The
Hamiltonians are identical to those produced by `makeDisplacedModel`; the
script avoids rebuilding velocity/position operators needed only for responses.

Default-run results, in **meV**, with Monte Carlo uncertainty of one standard
error:

| Model | T (K) | Static Fan | DW | Static total | Sampled gap shift |
|---|---:|---:|---:|---:|---:|
| Linear | 0 | -0.090587 | 0 | -0.090587 | -0.089742 +/- 0.001523 |
| Linear | 300 | -0.245750 | 0 | -0.245750 | -0.245248 +/- 0.004040 |
| Quadratic | 0 | -0.090587 | -0.009824 | -0.100411 | -0.099474 +/- 0.001592 |
| Quadratic | 300 | -0.245750 | -0.026651 | -0.272401 | -0.271886 +/- 0.004205 |

All four agree within 0.6 standard errors. The script also records dynamical
Fan+DW, but does **not** compare that expression to snapshots as an equality:
phonon energy exchange changes the denominators. Quantum zero-point position
sampling at 0 K still corresponds to the static, not dynamical, comparison.

Results are saved in `tmp/honeycomb_self_energy/comparison.json` and
`self_energy_results.npz`, together with reproducible input files. The tests
add deterministic covariance-exact finite-displacement checks of gap curvature
and DW, primitive/supercell normalization checks, retarded emission/absorption
and Pauli-factor checks, and the historical signed-FFT convention check.

## Why a large beta can disagree on the 6x6 mesh

The command

```sh
python examples/honeycomb_self_energy.py --qgrid 6 6 1 --scale 0.1 --beta 100
```

has an effective quadratic coefficient of `100*0.1^2 = 1 eV/Angstrom^2`.
Its sampled bare gap is 0.2784488268 eV. Both band edges occur at
`k=(1/3,2/3,0)`. With the default equal beta on all three bond orientations
and equal isotropic in-plane Einstein variances, the mean quadratic hopping
is the same on each orientation. Its A-to-B Bloch entry is proportional to

```text
1 + exp(-2*pi*i*k1) + exp(-2*pi*i*k2).
```

This vanishes at that k point. The DW matrix at the band edges, and hence
the leading DW gap shift, is zero within roundoff for this setup. This
is a cancellation in the specific model and observable. Other band energies
have DW shifts up to about 26 meV at 0 K and 70 meV at 300 K.

The reproduced quadratic-model results, in meV, are:

| T (K) | Leading static Fan+DW gap shift | Raw sampled gap shift (one standard error) |
|---|---:|---:|
| 0 | 0.019059 | 0.099593 +/- 0.025467 |
| 300 | 0.051704 | 0.412742 +/- 0.069897 |

The DW matrix agrees with an independent Cartesian Gaussian-covariance
average of the quadratic Hamiltonian to approximately 3e-16 eV. The excess
sampled gap shift comes primarily from higher displacement orders. Writing
`lambda=--scale`, the code defines

```text
H(u;lambda) = H0 + lambda*V1(u) + lambda^2*V2(u).
```

For an isolated band, the leading averaged energy correction is

```text
lambda^2 * [ <V2_nn> + sum_(m != n) <|V1_mn|^2>/(epsilon_n-epsilon_m) ].
```

The first term is DW and the second is static Fan. Diagonalizing each
quadratic Hamiltonian also includes higher-order changes of its eigenvalues.
Even with V1=0, the next term contains
`lambda^4*sum_(m != n) <|V2_mn|^2>/(epsilon_n-epsilon_m)`, proportional to
beta squared. With both couplings there are also mixed and higher-order
linear-coupling terms. A quadratic Hamiltonian does not imply quadratic
eigenvalues. Equal-beta cancellation of the leading DW gap shift makes these
otherwise higher-order terms especially visible relative to the small Fan gap
shift. Increasing beta also increases the sampling noise: pairing u and -u
does not cancel quadratic fluctuations.

The following independent diagnostic checks the covariance contraction and
the displacement-order expansion using common Gaussian samples:

```sh
python examples/honeycomb_dw_diagnostics.py --qgrid 6 6 1 --beta 100
```

For each sample, it evaluates the second-order eigenvalue expansion directly
from the displaced supercell matrices. Subtracting that expansion from the
exact paired gap isolates the higher-order remainder. The known thermal mean
of the subtracted expansion can then be added back to obtain a control-variate
Monte Carlo estimate. This reduces statistical noise without removing the
higher-order discrepancy. At 300 K with 300 common Gaussian pairs, the mean
remainder is 0.313896, 0.018377, and 0.001130 meV for scales 0.1, 0.05, and
0.025. Its approximately fourth-power scaling, and a separate beta-only
expansion, identify the omitted terms.

## A useful DW-dominated check

To obtain a visible leading DW gap shift on the same 6x6 mesh, use unequal
quadratic coefficients on the three bond orientations. For example:

```sh
python examples/honeycomb_dw_diagnostics.py --qgrid 6 6 1 --beta 100 --beta-bond-factors 0 1 0 --scales 0.05 0.025 0.0125
```

The factors refer to the home, (-1,0,0), and (0,-1,0) A-to-B bonds; reciprocal
bonds receive the same coefficient. These are different physical EPC model
parameters, chosen to avoid the equal-bond cancellation. The Hamiltonian and
phonon parameters are otherwise those of the original example. At 300 K:

| Scale | DW gap shift (meV) | Static Fan gap shift (meV) | Leading total (meV) | Control-variate MC total (meV) |
|---|---:|---:|---:|---:|
| 0.05 | 2.592989 | 0.012926 | 2.605915 | 2.681147 +/- 0.002091 |
| 0.025 | 0.648247 | 0.003232 | 0.651479 | 0.656166 +/- 0.000130 |
| 0.0125 | 0.162062 | 0.000808 | 0.162870 | 0.163162 +/- 0.000008 |

DW remains approximately 201 times the Fan gap shift. The higher-order
correction decreases from 2.89% to 0.72% to 0.18% of the leading total as the
scale decreases. This demonstrates a controlled DW-dominated regime; reducing
the expansion parameter does not force DW to become negligible relative to
Fan. The remaining systematic difference can be statistically resolved and
is not treated as zero. This diagnostic requires isolated finite-supercell
band edges; it stops if degenerate perturbation theory is needed.

The diagnostic writes `diagnostics.json` and `samples.npz` to
`tmp/honeycomb_dw_diagnostics` by default. Use `--output` to retain separate
runs. These checks concern the static approximation and a finite matched
supercell; they do not establish dynamical self-energy equivalence or
thermodynamic-limit convergence.
