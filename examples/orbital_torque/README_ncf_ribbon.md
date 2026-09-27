# Step 4: N/C/F source, converter and magnet

```text
python examples/orbital_torque/orbital_ncf_ribbon.py --workers 8
python examples/orbital_torque/orbital_ncf_ribbon.py --magnetization 1 2 3
python tests/test_ncf_ribbon.py
```

The ribbon is periodic along x and open along y. Default widths are N=8,
C=4, F=4 atomic rows, with complete spinful s+p shells (128 bands). N is
the validated s-p source; C has atomic SOC and orbital-independent hopping;
F has an onsite exchange field, orbital-independent hopping, no SOC and no
s-p hybridization. Both interfaces conserve spin and orbital labels.

N and C retain the step-3 parameters. F has onsite energies (-2,0,0,0) eV
per spin and nearest-neighbor hopping -0.5 I eV. Both interface hoppings
are -0.3 I eV. Each of the six cases is driven by uniform Ex:

| Case | Change from full model |
| --- | --- |
| full | N mixing 0.7 eV, C SOC 0.4 eV, F exchange splitting 0.6 eV |
| no_mixing | Set N s-p hopping to zero |
| no_soc | Set C SOC to zero |
| no_exchange | Set F exchange to zero |
| nc_off | Break only N-C hopping |
| cf_off | Break only C-F hopping |

Exchange is explicitly stored as `model.exchange_R`, independently of SOC.
The exchange Hamiltonian is `H_XC=+Delta/2 m.sigma` on F. Here m denotes
the direction of the ordered **magnetic moment**: the lower-energy electron
has spin opposite to m, hence magnetic moment along m. The default m=(0,1,0)
is normalized, as is any nonzero vector supplied with `--magnetization`.
This is an imposed exchange field, not a self-consistent ferromagnet or an
LLG magnetization dynamics simulation.

## Observables and torque sign

The calculation uses the established first-order length-gauge density-matrix
kernel with scalar CRTA, including populations and coherences. Saved profiles
are induced changes per Ex, not equilibrium moments. Moment vertices are
`mu_L=-L/hbar` and `mu_S=-sigma` in mu_B. The spin exchange torque is

`T_XC = -i[mu_S,H_XC]`.

This is exactly the repository's `SpinExchangeTorque` convention, with
energy-like units mu_B eV. Its response has units mu_B eV/(V/Angstrom) per
longitudinal repeat. Moment responses have units mu_B/(V/Angstrom).
No cell-volume or width division is applied. For this onsite exchange,
`T_XC = Delta (m cross mu_S)` in F and is zero outside F. For m=+y,
`T_x=Delta mu_S,z`, `T_y=0`, and `T_z=-Delta mu_S,x`.

The vertex equals hbar times the electron magnetic-moment rate due to
exchange. The ordered magnetic-moment rate has the opposite sign:
`d mu_order/dt = -T_XC/hbar`, assuming the matching magnetic-moment
normalization. Equivalently, the angular-momentum torque on the order is
`T_XC/(2 mu_B)` for g=2. These conversions must not be confused with the
plotted energy-like vertex. No total-Hamiltonian torque is substituted for
exchange torque.

The main plots emphasize orbital/spin z and torque x, but exchange can
produce other components. Companion figures and all numerical outputs retain
xyz, which is essential for tilted m. Torque must be transverse to m.

## Local observation and driving location

`profile_f0.png` shows local induced moments and torque versus **observed**
row under uniform Ex. `drive_rows_f0.png` instead plots the sum of each
observable over F versus the **individually driven** row. Both figures
contain all six cases. Their `all_components_` companions have 3x3 subplots.

Row driving uses `P_row v_x P_row`, with the same field amplitude on each
selected row. All eigenstates remain those of the full Hamiltonian for that
case. Inter-row hoppings have no x displacement, so v_x has no inter-row
blocks; the code checks this explicitly. Thus summing every driven row
recovers the F-total uniform-drive response. This is not a general identity
for diagonal projections in arbitrary models. The separately reported local
sum residual checks that observed-row moments sum to the total observable.

`summary.json` also sums the row-drive contributions separately over N, C
and F. N/C drive contributions to the F response are spatially nonlocal;
F driving probes its local-drive contribution in the coupled system.
This distinction refers to driving location, not uniquely to an orbital
current or torque mechanism. The projected field prescription is not a
self-consistent electrostatic device calculation.

In the default model, no mixing, no SOC, no exchange, and either broken
interface suppress exchange torque. Some controls retain orbital/spin
responses elsewhere; switching exchange off need not suppress SOC-induced
spin accumulation. Breaking C-F leaves the N/C subsystem intact. These
controls demonstrate parameter dependence, not a complete angular-momentum
continuity balance; interface currents and collision torques are not yet
measured here.

## Output and validation

Outputs go to `tmp/orbital_torque/ncf_ribbon/`:

- `profile_f0.png`, `all_components_profile_f0.png`: local profiles.
- `drive_rows_f0.png`, `all_components_drive_rows_f0.png`: F-integrated
  response to every driven row. One set per Fermi energy.
- `profiles.csv`: local Ex responses, total and I/E channels.
- `drive_rows.csv`: F-integrated row-drive responses for xyz.
- `responses.npz`: all case/mesh profiles, with axes observed row, Fermi,
  component. `drive_*` arrays at the finest mesh have axes driven row,
  Fermi, component. `*_total` arrays sum observations over the entire ribbon.
- `summary.json`: settings, units, consistency checks, mesh differences,
  F totals by driving region, and total exchange torque in F.
- Each case's `model.npz`: the actual Hamiltonian, separate SOC and exchange
  arrays, geometry, moment operators and magnetization direction.

Defaults are EF=0.8 eV, kBT=linewidth=0.05 eV, grids 256 and 512, 8 workers
with one BLAS thread per worker. Eigensystems are reused for all observables;
row-drive contractions avoid a full density-matrix calculation per row.
Use `--n-width`, `--c-width`, `--f-width`, `--tsp`, `--soc`, `--exchange`,
`--nc-hopping`, `--cf-hopping`, `--converter-hopping`, `--converter-shift`,
`--f-hopping`, `--f-shift`, `--fermi`, `--grids`, `--kbt`, `--linewidth`,
`--workers` and `--output` to vary parameters. `--plot-only` regenerates
figures from saved arrays. Mesh differences are reported rather than used
to claim width or clean-limit convergence. Custom parameters can legitimately
give zero full-model torque.

Tests compare local torque and individual row-drive torque with the established
`SpintorqueGalvanicConductivity_LG_DM`, and moment profiles with the previous
ribbon calculator. They check tilted magnetization, exchange eigenvalues,
SOC/exchange confinement, Hermiticity, torque sign, serial/parallel agreement,
row-drive additivity and all five controls.

## Default numerical result

At Nk=512 the F-integrated torque response is
(-0.1139023872, 0, 0.0353405125) in the energy-like torque units above.
The largest local torque magnitude over components is 0.7957301714; the
maximum profile change from Nk=256 to 512 is 6.49e-6. All five controls
give torque below 6.1e-15, with exchange-off exactly zero. Observation and
drive sum residuals are below 1e-14 at the finest mesh.

The signed x-torque contributions from driving N, C and F are respectively
-0.0358259847, +0.0644707482 and -0.1425471506. Their sum gives the uniform
drive result. These values illustrate that nonlocal injection and driving
the magnet itself both matter in the coupled ribbon; uniform driving alone
does not isolate an orbital injection mechanism.
