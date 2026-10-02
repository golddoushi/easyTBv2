# N/B/C ribbon: separating interface generation and SOC conversion

`orbital_nbc_ribbon.py` extends the hexagonal N/C model with a SOC-free
p-shell buffer B. N is the same s-only zigzag or armchair honeycomb source.
Directional s-p hybridization connects N to B. B and C have identical scalar
onsite energies and orbital-independent p-p hopping; only C has atomic SOC.
There is no exchange field. The B/C boundary therefore changes SOC without
introducing an additional scalar hopping or onsite discontinuity.

The comparison cases vary N and C widths at a shared B width, with all links connected and
finite SOC in C. No disconnected-interface or SOC-off control is included.
Default N/C pairs are `(4,4)`, `(6,4)`, `(4,8)`, `(6,8)`, with B=3.
N width counts honeycomb transverse cells; B and C widths count physical
rows. The number of physical N rows depends on the edge geometry.

## Running

From the repository root:

```text
python examples/orbital_torque_hex/orbital_nbc_ribbon.py --workers 8 --row-profiles
python examples/orbital_torque_hex/orbital_nbc_ribbon.py --edge armchair --b-width 3 --widths 4 4 --widths 6 4 --widths 4 8 --grids 512 1024 --workers 8 --row-profiles
python examples/orbital_torque_hex/orbital_nbc_ribbon.py --plot-only
python tests/test_hex_nbc_ribbon.py
```

Set `--b-width B` once and repeat `--widths N C` for each comparison case. `--row-profiles` enables
the full driving-row/observing-row matrices on the finest requested mesh;
without it, uniform profiles and C-total responses to every row drive are
still computed. `--workers` uses the established NC multiprocessing path.
The default output is `tmp/orbital_torque_hex/nbc_zigzag` or `nbc_armchair`.
Use `--output` to select another directory, including for `--plot-only`.
The default meshes are 512 and 1024; convergence differences are reported,
not interpreted automatically as proof of convergence.

## Observables and physical data

The example reuses the NC length-gauge density-matrix response implementation
without modifying the calculators. Plots show z components of spin and ACA
orbital accumulation, spin SOC/flux/total torques, orbital SOC/H0/crystal/
transfer/total torques, and direct spin/orbital source terms. Relaxation and
independently evaluated continuity residuals remain in the data. BSR is not
computed in this example.

Each `nN_bB_cC/` case contains:

- `model.npz`: the actual Hamiltonian and its H0/SOC/exchange decomposition.
- `responses_n*.npz`: physical results on each k mesh.
- `responses.npz`, `profiles.csv`, `drive_rows.csv`: finest-mesh native results,
  including physical positions and region labels. NPZ also contains channels
  and, when requested, full row matrices and orbital cut currents.

`summary.json` records parameters, timing, mesh differences, regional sums,
SOC-support checks and continuity residuals. Regional sums are taken over
physical rows only. The generated text Hamiltonian from the shared builder
precedes SOC insertion; `model.npz` contains the final SOC-bearing model.

The driving decomposition is still `{P_row,v_x}/2`. A row-specific result
is a contribution under this numerical velocity-vertex partition, not a
self-consistent solution for an electric field spatially confined to that row.
The direct source terms are retained precisely to audit its continuity balance.

## Comparing unequal widths

B is fixed across cases. Its native row indices are 0 through B-1, with
interfaces at -0.5 and B-0.5. N extends left from row -1, and C extends right
from row B. Wider ribbons extend farther outward; their outer boundaries
are intentionally not aligned. No rows are stretched or interpolated.

The coordinate counts physical rows, **not distance in Angstrom**. Dots
mark native rows. Shorter ribbons stop at their own boundaries; absent rows
are NaN in `aligned_display.npz` and blank in matrix figures, never zero
response or an average. The cache stores exact row-selection weights and
missing-row masks. Physical coordinates remain in each case's native data.

- `profile_f*.png`: uniform-drive profiles for every width case.
- `drive_rows_f*.png`: C-total responses versus aligned driving row.
- `orbital_cut_current_f*.png`: cut currents between native neighboring rows.
- With `--row-profiles`, `row_profiles/` contains one response-matrix figure
  per case using common axes and color scales, and one profile comparison
  per driving row. Cases without that driving row are omitted from that
  comparison. All plotted responses come from actual simulated rows.

`--plot-only` requires saved cases with equal B widths; unequal-B datasets
are rejected rather than interpolated. Use native positions for decay lengths
and integrations over distance.

## Interpreting the buffer

In the interior of B, SOC and the crystal torque vanish for this hopping
model. Orbital influx plus the direct source therefore balances orbital
relaxation. The first B row can retain the N/B interface crystal torque.
The buffer-integrated check includes it:

`J_left - J_right + sum_B(crystal + SOC + direct source) - sum_B(relaxation) = 0`.

The cut current uses the established discrete definition
`J[r+1/2] = -sum_{j<=r} orbital_transfer[j]`. A cut on a p-p link within B
or at B/C diagnoses ACA orbital transport. At the s-p boundary the ACA
moment is not conserved, so crystal generation and transfer must be read
together. Since N has only s orbitals, its ACA moment remains identically
zero; these diagnostics alone do not identify transport of BSR moment.

Width dependence tests whether interface-generated ACA response reaches a
spatially separated SOC region. It need not be monotonic in a coherent
finite ribbon. Separate runs can compare different fixed B widths. Retain k-mesh,
linewidth and temperature convergence checks when interpreting that dependence.
