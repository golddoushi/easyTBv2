# Step 2: orbital moment accumulation at source-ribbon edges

```text
python examples/orbital_torque/orbital_ribbon.py
```

Defaults: 12 atomic rows along y, periodic x, Fermi energy 0.8 eV,
`kBT=0.05 eV`, scalar CRTA linewidth `0.05 eV`, and k meshes `(64,1,1)`
and `(128,1,1)`. The Hamiltonian has 8 bands per row. Specify `--width`,
`--grids`, `--fermi`, `--kbt`, `--linewidth`, or `--output` to vary these.
The default `--engine density` reuses one response density for all rows.
`--workers 4` enables parallel k submeshes; the default is one worker.

## Construction and measured response

`build_ribbon` calls the same `build_model` used in step 1, then
`make_supercell(diag(1,width,1))` and `cutEdge(1)`. This preserves all interior
y bonds, removes only inter-supercell y bonds, and leaves x periodic. The
atomic row spacing and x repeat length are both 3 Angstrom. Neither SOC nor
exchange is present. The electric field is uniform and points along x.

For each row i, the response agrees with `OrbitalGalvanicConductivity_LG_DM`:

```text
chi_i = d <mu_L,z(i)> / d E_x
mu_L(i) = {P_i, mu_L}/2,  mu_L/mu_B = -L/hbar
```

`P_i` selects both s and p shells of that atom, including both spins.
Identical source and target projectors implement the local observable.
Since the atomic orbital operator is onsite, this is simply its block on row i.
The original projected calculators remain available with `--engine projected`.
The default path uses their established `LengthGaugeFirstOrderDMKernel`,
transforms each I/E density to the local orbital basis once, averages over k,
and traces all onsite orbital and spin moments. This avoids repeating density
preparation and dense projectors for every pair of rows. No response formula,
spin cancellation, or mirror symmetry is imposed by the optimization.

Both intraband and interband density channels are included. The stored local
tensor axes are `[layer,fermi,field,moment]`, so `[:, :, 0, 2]` is the requested
orbital z-moment response to Ex. Values are in **mu_B/(V/Angstrom) per row per
x unit cell**. There is no division by ribbon width, cell volume, or vacuum.
Multiplication by a small field in V/Angstrom gives the induced moment in mu_B.
These are nonequilibrium linear responses, not equilibrium orbital moments.

Mirror symmetry across the ribbon maps the axial z moment to its negative
while leaving Ex unchanged. Thus `chi_i=-chi_(N-1-i)` and the whole-ribbon
response must vanish even when the individual edge responses are finite.
Spin independence requires zero local spin accumulation as well.

## Validation and scope

The default 12-row calculation at Fermi energy 0.8 eV gives:

| Quantity | Result |
| --- | ---: |
| Lower outermost row, chi_0 | +3.69101145445 |
| Upper outermost row, chi_11 | -3.69101145445 |
| Maximum absolute orbital response (rows 1 and 10) | 4.00640591721 |
| Maximum absolute local spin response, all tensor components | 2.31e-14 |
| Maximum absolute orbital response with tsp=0 | 2.06e-14 |
| Mirror-odd profile residual | 1.04e-14 |

Response entries use mu_B/(V/Angstrom) per row per x repeat. The maximum
profile change from Nk=64 to 128, divided by its peak amplitude, is `3.00e-5`
(0.003%). All built-in checks passed. At the lower outermost row the I and E
channels are approximately 3.439325 and 0.251686, respectively; retaining the
intraband channel is essential for this metallic ribbon. The residual spin
signal is floating-point noise around the symmetry-required zero.

The script checks Hermiticity, identical spin blocks, and the absence of
transverse intercell hoppings. Response checks require nonzero local orbital
accumulation, opposite mirror-related profiles, zero net orbital z response,
zero spin response, and equality of the sum of row projections to the total
orbital response. An otherwise identical ribbon with `tsp=0` must have zero
orbital and spin response. All residuals are measured without rounding the data.

Mesh comparisons quantify the difference of the full Ex-to-mu_L,z profile.
Width and relaxation changes are separate physical/numerical studies; a mesh
comparison at one width does not establish an isolated semi-infinite edge.
Finite ribbons can have oscillatory interior responses. The local atomic
moment used here omits itinerant orbital magnetization contributions.
Accumulation alone is not a measurement of conserved orbital flux: orbital
angular momentum can also exchange with the crystal field and relaxation bath.

Outputs in `tmp/orbital_torque/ribbon/`:

- `ribbon_model.npz`: actual finite-ribbon Hamiltonian, lattice, positions,
  spin and orbital operators. The primitive source/control input files are
  retained separately; the builder reproduces the cut ribbon from those.
- `responses.npz`: full local orbital/spin tensors, I/E channels, total
  responses, positions, all meshes, and the no-hybridization control.
- `profile.csv`: finest-mesh layer distribution, I/E decomposition, spin
  residual, and orbital control, at every selected Fermi energy.
- `profile.png`: orbital moment profile and maximum spin response by layer.
- `summary.json`: numerical settings, convergence metrics, and checks.

The program raises an error if the validation criteria fail, after writing
diagnostic output. A custom filling with no occupied active states can
legitimately fail the finite-accumulation criterion.
