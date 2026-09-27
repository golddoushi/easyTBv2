# Step 3: orbital source N and SOC converter C

```text
python examples/orbital_torque/orbital_nc_ribbon.py --workers 8
```

This computes five cases with controlled parameter changes: the full N-C model;
N s-p mixing off; C SOC off; interface hopping off; and a connected full
Hamiltonian with the drive restricted to N.
Both orbital and spin magnetic-moment profiles are plotted on shared axes,
with N/C shaded and the interface marked. The accelerated solver from step 2
uses the established first-order LG density-matrix kernel, scalar CRTA, and
parallel interleaved k meshes. There is no exchange or magnetic order.

## Hamiltonian and geometry

Periodic x, finite y, a=3 Angstrom, full s+p shells with two spins on every row.
N occupies the first 8 rows and C the next 6 (112 bands by default).

- N: the validated source parameters from step 1. Onsite energies are
  (-2,0,0,0) eV in the (s,pz,px,py) basis. Hoppings are ts=-0.5,
  t_sigma=0.5, t_pi=-0.1 and t_sp=0.7 eV. S-p mixing exists only on bonds
  wholly inside N, including longitudinal N bonds. There is no SOC in N.
- C: the same onsite energies, with optional uniform converter_shift, but
  orbital-independent nearest-neighbor hopping -t_C*I4, t_C=0.5 eV.
  There is no s-p mixing. Its onsite atomic SOC is
  `H_SOC=lambda sum_a (L_a/hbar) sigma_a/2`, with lambda=0.4 eV. The complete
  p shell is retained, including pz. The s orbital has zero SOC.
- Interface: -t_interface*I4, t_interface=0.3 eV, between adjacent N/C rows,
  duplicated for spin. It conserves orbital labels and spin and contains no
  s-p mixing or SOC. Outer y boundaries are open.

The isolated C bulk has no momentum-dependent orbital eigenvectors: its p
dispersion is proportional to the identity, plus onsite atomic SOC. It is
chosen to avoid a separate bulk Hall source in the converter. This is a
deliberately controlled toy converter, not a model of a particular material.
The junction eigenstates still extend across both regions; finite spin
accumulation can extend into N even though its local SOC is zero.

The code saves the actual modified Hamiltonian and a separate soc_R array for
each case. Cached velocity-related Hamiltonian moments are refreshed after
all modifications. SOC is not classified as exchange.

## Driving and observables

The example also computes the C-integrated orbital and spin magnetic moment
response to each individually driven row at the finest requested mesh.
`drive_rows_f0.png` plots z components against the **driven** row for all five
cases shown in the accumulation profiles. `all_components_drive_rows_f0.png`
shows x/y/z components in six subplots. Every point sums over all
C observation rows; it is not a local accumulation profile. Each drive uses
the same field amplitude and the vertex `P_row v_x P_row`, with no division
by the number of C rows. The disconnected control shows C's response without
coupling to N, while retaining its SOC.

For the N-only protocol, N-row contributions equal those of the full model;
C-row contributions are zero because those rows are not driven. These zeros
represent the drive restriction, not a vanishing susceptibility to driving C.

The solver reuses band states and contracts the scalar-CRTA density-matrix
kernel with the C moment operators before evaluating all row vertices.
It avoids a separate full response calculation for every driven row.
Summing all driven rows reconstructs the uniform-drive C total; summing N
rows reconstructs the N-only C total. Both are checked numerically.

C's lack of response in isolation does not imply zero C-driven response in
the coupled junction: N changes the hybridized eigenstates even when only C
is driven. Opposite signs from different driven rows can cancel. An
accumulation concentrated near the interface need not originate from driving
only interface rows. These are finite-ribbon susceptibilities, not an
interface-current measurement or a determination of conversion efficiency.

The first four cases use **uniform Ex in both N and C**, as in step 2.
The fifth, `n_drive`, uses the regional drive vertex `v_x^N=P_N v_x P_N`,
where P_N selects all orbitals and both spins on N atoms. The Hamiltonian,
eigenstates and observable moments remain those of the full connected junction.
Only the driving vertex changes in the established density-matrix kernel;
the response is not obtained by discarding the C portion of a uniform-drive
response. Every case uses the same nominal field amplitude in its driven region.

For this geometry, interface bonds have zero x displacement and there is no
inter-region v_x matrix element. Therefore `v_x^N={P_N,v_x}/2` and
`v_x=v_x^N+v_x^C`. The tests check corresponding response additivity.
This defines a region-selective linear-response drive using the repository's
projected-current convention. It does not solve a self-consistent electrostatic
field or a reservoir-driven dc device; a spatially confined longitudinal
field requires additional boundary/gauge considerations for that interpretation.
Saved responses for other field directions use the same P v P rule; only Ex
is used in this investigation and in the N+C additivity check.

Measured local moments are `mu_L/mu_B=-L/hbar` and
`mu_S/mu_B=-sigma`, with gL=1 and gS=2. Values are derivatives with respect to
Ex, in mu_B/(V/Angstrom) per row per longitudinal unit cell. No division by
width or vacuum volume is made. Full tensors and I/E channels are retained;
the primary figure shows z components and the supplementary figure shows
all moment components. Numerically zero panels are labeled explicitly.

The N-C asymmetry removes the mirror symmetry between opposite edges from
step 2, so neither an antisymmetric profile nor zero total moment is imposed.
The sum of local moments must still equal the independently traced total.

## Controls and their meaning

| Case | N s-p mixing | C SOC | Expected Ex response |
| --- | --- | --- | --- |
| full | 0.7 eV | 0.4 eV | Finite orbital and spin accumulation |
| no_mixing | 0 | 0.4 eV | Both responses vanish for this Hamiltonian |
| no_soc | 0.7 eV | 0 | Orbital accumulation remains; spin vanishes |
| disconnected | 0.7 eV | 0.4 eV | Interface hopping zero; finite orbital response in N, zero response in C |
| n_drive | 0.7 eV | 0.4 eV | Full interface; N-only drive; finite spin response in C |

The first four rows use a uniform drive; the last uses the N-projected vertex.
In `disconnected`, both material functionalities remain enabled; only
interface hopping is removed. All N/C internal bonds and C SOC remain intact.

The no_mixing control switches off N's OHC mechanism, not C's onsite SOC
mixing. The interface and all diagonal hoppings remain unchanged. With this
source mixing removed, H(kx)=H(-kx) in the fixed local basis. The Ex velocity
vertex is odd, and the local-moment vertices are k independent, so their
linear responses cancel between +/-kx. The no_soc control restores exact
spin independence, implying zero spin response.

These profiles and controls demonstrate dependence on both source orbital
mixing and converter SOC in this model. They support the proposed conversion
picture, but do not measure an interface orbital current or establish a
complete angular-momentum balance. That requires bond-current and local
SOC/crystal-field/relaxation torque observables in a subsequent investigation.
The no-mixing control also changes the bands; it is not a perturbation that
changes OHC while holding every transport property fixed.

## Results and convergence

Defaults: EF=0.8 eV, kBT=0.05 eV, CRTA linewidth=0.05 eV, meshes 256,512,1024.
At Nk=1024 the largest absolute response in C, over its rows and moment
components for Ex, is:

| Case | Orbital response | Spin response |
| --- | ---: | ---: |
| full | 5.846540164 | 3.448077196 |
| no_mixing | 3.90e-15 | 3.26e-15 |
| no_soc | 7.691704035 | 4.13e-15 |
| disconnected | 6.83e-15 | 8.50e-15 |
| n_drive | 1.370431388 | 0.575825977 |

In the disconnected case N retains an orbital response with peak magnitude
3.715330778, while C is inactive despite its SOC and the uniform drive.
With only N driven in the connected junction, C still has finite orbital and
spin responses. The N-only 512-to-1024 mesh changes are below 9.5e-10.
Its smaller spin response than the uniformly driven case also shows that
driving C contributes substantially to the connected junction's response;
it should not be assumed negligible solely because isolated C has no Hall
source in this toy Hamiltonian.

Values have the moment-response units above. The maximum full-model orbital
response anywhere is 7.850536987. The 512-to-1024 maximum profile differences
are 1.22e-9 for orbital and 2.00e-9 for spin. The coarser 128-to-256 change
in spin was about 0.0112, so the finer default mesh is intentional.
Each 1024-point response took about 3.5-3.8 seconds with 8 workers on the
16-core Ryzen machine; model building, plotting and I/O are additional.
This is mesh convergence at fixed width, temperature and linewidth, not
width convergence or a clean-limit claim.

All built-in response checks passed. Run the independent checks with:

```text
python tests/test_nc_ribbon.py
```

They check the SOC eigenvalues (-lambda twice, +lambda/2 four times in the p
shell), Hermiticity, time reversal, SOC confinement to C, agreement of all
response arrays with the original projected calculators on a small SOC
ribbon, the hybridization/SOC controls, and vanishing C accumulation when the
interface hopping is zero. The disconnected case is included in the plots.
Additional tests compare the N-only optimized response to the original
projected calculators, check that projecting onto all atoms restores the
uniform drive, check N-only plus C-only response additivity for Ex, and require
zero C response for an N-only drive with a disconnected interface.
The row-drive contraction is checked against separate projected-calculator
runs for every row, all moment components and two fillings, including the
disconnected control and serial/parallel agreement.

## Files and options

`tmp/orbital_torque/nc_ribbon/` contains:

- `profile_f0.png`: z-polarized orbital/spin profiles, all five cases.
- `all_components_f0.png`: x/y/z components; one figure pair per Fermi energy.
- `profiles.csv`: finest-mesh Ex response and I/E channels by case, row,
  region, Fermi energy and moment component.
- `drive_rows_f0.png`: C-total orbital/spin z response versus individually driven
  row, with all five cases.
- `all_components_drive_rows_f0.png`: the same comparison for x/y/z components.
- `drive_rows.csv`: the same signed C totals for each driven row and filling.
- `responses.npz` also contains `drive_rows_{case}_{orbital,spin}` for all five cases;
  axes are driven row, Fermi energy, moment component (Ex drive only).
- `responses.npz`: full local/total tensors and channels for every case/mesh.
  Local tensor axes: layer, Fermi, field, moment.
- `summary.json`: parameters, timing, mesh differences, checks and units.
- `full/model.npz`, `no_mixing/model.npz`, `no_soc/model.npz`,
  `disconnected/model.npz`, `n_drive/model.npz`: actual junction
  Hamiltonians, soc_R, positions, lattice and moment operators. Primitive
  source input files used by the builder are saved separately in each case.

Use `--n-width`, `--c-width`, `--soc`, `--tsp`, `--interface-hopping`,
`--converter-hopping`, `--converter-shift`, `--fermi`, `--grids`, `--workers`,
`--kbt`, `--linewidth`, and `--output` to vary the calculation. Arbitrary
fillings or zero couplings can legitimately fail the finite-signal checks;
diagnostic outputs are saved before the program raises an error.

Replot results generated by this version without recalculating (older archives
must first be regenerated to include the row-drive arrays):

```text
python examples/orbital_torque/orbital_nc_ribbon.py --plot-only
```
