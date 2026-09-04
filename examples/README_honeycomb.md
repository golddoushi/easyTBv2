# Four-band honeycomb model with SSH EPC

This example uses `System_tb` with two atoms (A, B), one s orbital per atom,
and physical spin. The orbital order is `(A up, B up, A down, B down)`.
It supplies microscopic EPC matrices and frozen structures; it does **not**
yet construct EPC collision operators or linewidths for the DM calculators.
Fan-Migdal/Debye-Waller calculations and a stochastic gap benchmark are now
available separately; see [the self-energy guide](README_model_self_energy.md).
Linear dc/ac LG, VG, and real-time conductivity comparisons are described in
[the conductivity guide](README_linear_conductivity.md), including an optional
frozen-supercell transport average.

From the repository root:

```sh
python examples/honeycomb_four_band.py --responses --grid 12
python -m unittest discover -s tests -p test_model_epc.py -v
```

The example writes its input files, `model_results.npz`, and `summary.json`
to `tmp/honeycomb_four_band`. A different directory can be given with
`--output`. Omit `--responses` to generate only the model, EPC, and one frozen
3x3 supercell snapshot. The input generator is versioned so the ignored
`*_hr.dat` files can always be recreated.

## Hamiltonian and units

All lengths are Angstrom, electronic and phonon energies are eV, ionic masses
are amu, and temperatures are kelvin. Electronic and phonon wavevectors are
fractional reciprocal coordinates, with phases `exp(2*pi*1j*k.R)`.

The equilibrium Hamiltonian contains:

- Nearest-neighbor scalar hopping `H_ij = -t_ij`, with three independently
  adjustable bond amplitudes (default 1.0, 0.9, 1.1 eV).
- Onsite staggered energies `(+Delta, -Delta)` (default Delta = 0.45 eV).
- Exchange `H_XC = -J n.sigma` (default J = 0.35 eV, tilted n).
- Rashba hopping `i lambda (sigma x d_hat).normal` (default lambda = 0.2 eV,
  normal = +z). This is effective intersite SOC; local orbital L remains zero.

The honeycomb nearest-neighbor distance is 1 Angstrom by default. The third
lattice vector has length 10 Angstrom. The existing conductivity calculators
report volumetric values using this cell volume: a physical 2D sheet
conductance is obtained by multiplying conductivity in S/m by the cell height
in meters. The arbitrary height must be retained when comparing results.

## Bond direction and file format

The repository's matrix convention is

```text
H_R[i,j] = <i,cell 0 | H | j,cell R>
d_ij(R) = R @ lattice + model.rList[:,j] - model.rList[:,i]
d_hat_ij(R) = d_ij(R) / |d_ij(R)|
```

Thus the vector points from the row/home orbital to the column/R orbital. It
includes the intracell coordinates, even when R=0. The hat denotes a **unit**
vector; its components are dimensionless. Changing this orientation convention
changes the sign assigned to the Rashba lambda.

`u_i` is the Cartesian displacement from atom i's equilibrium position.
To linear order the bond extension is

```text
stretch_ij(R) = d_hat_ij^0(R) . [u_j(R) - u_i(0)]
t_ij(u) = t_ij^0 - alpha_ij * stretch_ij
H_ij(u) = H_ij^0 + alpha_ij * stretch_ij       # because H_ij=-t_ij
```

`Energy.solveEigenEquation` uses the signed entries in `ham_R` directly:
`H(k)=sum_R ham_R(R)*exp(+2*pi*i*k.R)`. It applies no additional minus sign.
The example's positive parameter t is simply defined by writing `ham_R=-t`.

The EPC file therefore stores `alpha = dH_ij/d|d_ij|`, in eV/Angstrom.
Signed coefficients are allowed. For a convention using positive H_ij as the
decaying hopping, supply the corresponding negative derivative. By default
the model is strictly linear in displacement: it does not use the exact
displaced bond length. An optional `betaR` file adds `beta*stretch^2/2`, with
beta in eV/Angstrom^2, to test a nonzero Debye-Waller contribution. It does not
add the transverse curvature of an exact bond-length law. Rashba strengths,
Rashba bond directions, and exchange are held at their equilibrium values.

Both coupling files use the existing `readFullHrDat` reader:

```text
descriptive title
2
number of R blocks
1 1 ... one unit degeneracy per R block
Rx Ry Rz i j real_strength imaginary_strength
...
```

There are **two spatial indices**, not four spinor indices. Supply all four
matrix entries per R block, including zeros. The imaginary strengths must
vanish. Supply reciprocal bonds: `strength(R)[i,j]=strength(-R)[j,i]`.
Nonzero coefficients on zero-length/onsite bonds are rejected. The code does
not invent Hermitian partners or double-count them.

Use unit degeneracies. `readFullHrDat` now returns the Wigner-Seitz factors as
`Ndegen` but keeps `ham_R` as the raw file entries for compatibility; it does
not divide by those factors implicitly. The model coupling methods request its
unit-degeneracy validation explicitly.
R blocks may be reordered or be a subset of the Hamiltonian's `iRvec`.
Any supplied translation outside that grid raises `ValueError`, including
zero-valued blocks. EPC coefficients must also act on existing Hamiltonian
bonds: at least one spin component of that spatial bond must be nonzero.
Incompatible files leave the Hamiltonian, grid, and attached EPC unchanged.

The two spatial indices above are specific to the four-band example. Models
with several s orbitals on each atom use `norbs/2` spatial indices in these
files. `System.iorb2iatomMap[iorb]` stores the zero-based atom index of each
basis orbital, including spin partners. `System_tb`, `System_omx`, and
`System_w90` populate it during their existing orbital-center initialization
loops; supercell construction remaps it with the orbital indices. SSH EPC
uses this registered map and keeps an equilibrium copy. It does not prescribe
an orbital count per atom. Bond positions come directly from
`rList`, and orbitals assigned to the same atom share that atom's displacement.
Spin partners must have the same atom and center. This remains a scalar bond
model; orbital-dependent angular hopping laws are not implemented.

## Public API

```python
import copy
import numpy as np
from system import System_tb

directory = "tmp/honeycomb_four_band"  # generate inputs with the example first
model = System_tb(directory, hamR="honeycomb", ispin=2,
                  projections={"A": [0], "B": [0]})
model.addExchangeSplitting([(0, (0,)), (1, (0,))], [-0.105, -0.14, -0.3031088913])
model.addRashbaSOC(directory, rashbaR="rashba", normal=(0, 0, 1))
model.addModelEPC(directory, alphaR="epc")
model.setModelPhonons(masses=[12.0, 12.0], phononEnergies=0.02,
                      directions=(0, 1), mode="einstein")

k, q = np.array([0.17, 0.23, 0]), np.array([1/3, 0, 0])
energies, polarizations = model.modelPhonons.modes(q)
g = model.getModelEPC(k, q)                  # shape (4 modes, 4 bands, 4 bands)
g_orbital = model.getModelEPC(k, q, bandBasis=False)

sc = copy.deepcopy(model)
sc.make_supercell(np.diag([3, 3, 1]))
u = sc.sampleModelDisplacements(temperature=300, seed=2026)
snapshot = sc.makeDisplacedModel(u)
```

`addRashbaSOC` adds a Hamiltonian contribution on each call and retains the
accumulated contribution in `model.rashba_R`.
`addModelEPC` replaces the attached EPC specification without changing the
equilibrium hopping. Its optional `betaR` argument uses the same file layout
for quadratic coefficients; omitting it sets beta to zero.
`setModelPhonons` replaces the phonon specification.
Both atom masses and phonon energies may be scalars; phonon energies may
also have shape `(natom,)` or `(natom,len(directions))`.

The supercell builder carries exchange, its rigid-rotation reference, Rashba,
SSH derivatives, and phonon parameters into the enlarged cell. It also
translates the onsite position vertices with the orbital centers. Form the
supercell **before** sampling, so each supercell atom has an independent
displacement. A periodically repeated primitive-cell snapshot contains only
q=0 distortions.

`makeDisplacedModel` returns an independent copy. It updates scalar hopping,
atomic/orbital positions, and Hamiltonian moments; it retains the fixed SOC
and exchange vertices. It does not wrap displaced atoms into the cell or
change their lattice labels. The returned frozen snapshot has no attached
EPC/phonon model. Generate each snapshot from the equilibrium model, avoiding
accumulation of displacements or recentering of the linearized interaction.
Other geometry/grid changes require reattaching EPC if they invalidate its
reference. Currently supported bases are orthogonal point orbitals with one
s orbital per atom and the canonical blocked-spin convention.

## Flat phonons still scatter at finite q

The initial bath is

```text
H_ph = sum_(L,kappa,a) [p_(L,kappa,a)^2/(2 M_kappa)
                        + M_kappa Omega_(kappa,a)^2 u_(L,kappa,a)^2/2].
```

The restoring force is to a fixed reference (a pinned lattice). It is not a
free-standing crystal with acoustic modes. In reciprocal space every active
atom/Cartesian oscillator supplies a flat branch, with real unit polarization
on that atom and direction. Degenerate oscillator combinations can be chosen
as plane waves at any q. Their energies are equal, but their phases and
electronic final momenta are different.

For a displacement `u_(L,kappa)=e_kappa exp(2*pi*i*q.L)`, the orbital vertex is

```text
V_ij(k,q) = sum_R exp(2*pi*i*k.R) alpha_ij(R) d_hat_ij(R)
                   . [e_j exp(2*pi*i*q.R) - e_i].
```

This is implemented by `model.modelEPC.displacementVertex(model,k,q,e)`;
`e` can be any complex `(natom,3)` displacement in Angstrom. Uniform q=0
translation gives zero, while relative A/B q=0 displacement can couple.
At finite q even equal A/B polarization usually couples. A single q therefore
does not determine all EPC scattering, despite the dispersionless frequencies.

`getModelEPC` inserts the zero-point displacement
`sqrt(hbar/(2 M_kappa Omega_nu))*polarization`, and optionally rotates the
vertex using `U(k+q).conj().T @ V @ U(k)`. It returns complex
`g[nu,m,n]` in eV, where m is the final band at k+q and n is the initial band
at k. Its normalization is

```text
H_ep = (1/sqrt(Ncells)) sum_(k,q,nu,m,n)
         g_mnnu(k,q) c_dagger_(m,k+q) c_(n,k) [b_(q,nu)+b_dagger_(-q,nu)].
```

The 1/sqrt(Ncells) factor and thermal Bose populations are not included in g.
Optional `vectors_k` and `vectors_kq` arguments let callers use their own
eigenvectors, including a consistent gauge within degenerate band subspaces.

## Static sampling, Born-Oppenheimer, and the low-temperature limit

Independent Gaussian samples have variance
`hbar/(2 M Omega) * coth(hbar*Omega/(2 kB T))`, with the zero-point limit
handled explicitly at T=0. `quantum=False` uses the classical variance
`kB*T/(M*Omega**2)` instead. These are position distributions of a prescribed
harmonic bath, without electronic feedback on the structural probability.

Born-Oppenheimer separation permits an electronic Hamiltonian parameterized
by instantaneous ionic coordinates. It does not equate quantum inelastic
scattering to an ensemble of permanently static Hamiltonians. Transport
depends on small energy differences near the Fermi surface and long-time
correlations, where even a small phonon energy can matter.

The [VASP overview](https://vasp.at/wiki/Category:Electron-phonon_interactions)
specifically distinguishes static sampling of gap renormalization/spectral
functions from perturbative phonon-limited transport. Its
[theory page](https://vasp.at/wiki/Electron-phonon_interactions_theory)
retains phonon energies in the Fan-Migdal denominators and documents the
adiabatic approximation separately.

At low temperature, distinguish two benchmarks:

- **Static/adiabatic renormalization:** zero-point position sampling can
  produce nonzero band shifts. Its weak-coupling expansion can be compared
  with a consistently static perturbative calculation using the same
  covariance, Hamiltonian, and observable vertices. This linear SSH model
  has no explicit quadratic-displacement (Debye-Waller) vertex.
- **Quantum relaxation:** for an equilibrium metal at T=0, a carrier at the
  Fermi energy cannot absorb a phonon, and emission is Pauli blocked. A
  static random zero-point displacement can nevertheless scatter elastically.
  Thus its disorder-induced linewidth need not equal the quantum EPC
  linewidth. Photoexcited carriers above the emission threshold can still
  emit phonons at T=0; the vanishing-rate statement is not for all energies.

High temperature relative to the relevant phonon energy, weak coupling, and
a quasi-elastic electronic regime are better starting points for comparing
frozen-disorder transport with phonon perturbation theory. Even there, compare
like approximations: an exact static-disorder average contains effects absent
from SERTA. State system size, observation time/broadening, and any bath used
to define photoexcited steady populations. The current example does not
claim that its static snapshots validate quantum transport at low T.

## Verification and response convergence

The tests independently compare complex finite-q EPC matrices against central
finite differences of **actual geometric bond lengths**, using cosine and
sine displacements in a commensurate 3x2 supercell and projecting between
primitive Bloch sectors. They also check Rashba sign/time reversal,
Hermiticity, uniform-translation cancellation, Einstein variances, exchange
rotation, rejection of foreign R blocks, multiple s orbitals per atom,
`rList` bond geometry, and supercell band folding/position vertices.

The response example uses existing CRTA kernels (`eta=0.05 eV`, `kBT=0.025 eV`),
two chemical potentials (-0.8 and 0 eV), and four photon energies (0.5, 1,
1.5, 2 eV). It reports nonzero AHC, BPVE, spin IFE, and exchange IFT, including
helicity-odd IFE/IFT components. These are model smoke checks, not converged
response benchmarks. In particular the coarse 12x12 grid is insufficient.
For the default parameters, the largest LG/VG relative differences over the
two fillings were:

| k mesh | BPVE | Spin IFE | Exchange IFT |
|---|---:|---:|---:|
| 24x24 | 59.0% | 13.0% | 42.3% |
| 48x48 | 39.0% | 5.1% | 7.8% |
| 96x96 | 14.5% | 0.35% | 0.66% |

Each relative difference is `max(abs(LG-VG))/max(abs(LG))` over frequencies
and tensor components at a given filling, then maximized over fillings.
Further convergence is needed before using these spectra as references,
particularly for BPVE. No DM kernel was changed to make the example agree.
