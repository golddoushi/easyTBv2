# Beyond ACA: honeycomb orbital-current and torque investigation

This hierarchy keeps the Bloch-state self-rotation (BSR) moment distinct from
the atom-centered approximation (ACA). An s-only basis has zero ACA moment
but can carry BSR moment through intersite wave-function circulation.

| Stage | Question | Status |
| --- | --- | --- |
| 1 | Does a spin-independent s-only insulator have finite BSR OHC and zero SHC/ACA OHC? | Implemented in [orbital_hall_source.py](orbital_hall_source.py) |
| 2 | What finite-ribbon boundary moment response corresponds to the bulk BSR-current response, in the gap and in the bands? | [orbital_ribbon.py](orbital_ribbon.py), [scope and audit](README_orbital_ribbon.md); symmetrized partition of ribbon BSR moment |
| 3 | Can coupling to an SOC converter yield spin response from this source? | [orbital_nc_ribbon.py](orbital_nc_ribbon.py), [model and projection conventions](README_orbital_nc_ribbon.md); zigzag/armchair sources, p-shell C, BSR/spin/ACA moments and SOC torque under uniform and row drives |
| 4 | Does an attached exchange magnet acquire torque? | Planned; local/remote driving and angular-momentum-transfer checks |

## Run stage 1

```text
python examples/orbital_torque_hex/orbital_hall_source.py
python examples/orbital_torque_hex/orbital_hall_source.py --plot-only
python tests/test_hex_orbital_source.py
```

The generated model matches the honeycomb geometry involving two atoms
per unit cell. Both atoms carry only s orbital with two identical spin 
sectors. It has four bands with basis (A up, B up, A down, B down), not 
SOC-split bands. The default lattice constant is 2.4 Angstrom, nearest-
neighbor hopping is -1 eV and onsite energies are +0.5/-0.5 eV. The analytic 
band energies in either spin sector are

`E_+- = +-sqrt(Delta^2 + t^2 |1+exp(i k.a1)+exp(-i k.a2)|^2)`.

Thus the gap is [-|Delta|,+|Delta|], independent of hopping strength for
nonzero Delta. No SOC, exchange, Rashba term or explicit magnetic flux is
introduced. Spin cancellation is not imposed on the response arrays.

The script uses the existing `OrbitalHallConductance_BlochSelfRotate_LG_DM`,
`OrbitalHallConductance_LG_DM`, and `SpinHallConductance_LG_DM` calculators.
All use the first-order length-gauge density matrix with scalar CRTA and
finite-temperature occupations. It saves full tensors and I/E channels.
The BSR current vertex is the existing `{m_BSR,v}/2`; it is not a full
Wannier local/itinerant-circulation transport theory. ACA uses `{mu_L,v}/2`,
where the onsite `mu_L` is identically zero here.

The spinful result includes both identical spin sectors. Do not assume it
has the same prefactor as an explicitly spinless two-band calculation.
Native Hall normalization is `2*pi/cell_area` times the BZ mean using the
calculator's magnetic-moment and velocity vertices. It is not a universal
quantized conductance unit and should not be compared directly with the
previous examples' local-moment response units.

Defaults: grids 48/96/192 (including the valleys), kBT=0.01 eV,
linewidth=0.02 eV, Fermi scan -3.2 to +3.2 eV. The code also evaluates
midgap explicitly, then halves temperature and linewidth there. Parameters are exposed
as `--hopping`, `--delta`, `--lattice-constant`, `--fermi`, `--grids`, `--kbt`,
`--linewidth`, and `--output`. The source requires nonzero hopping and Delta.
No hopping-off transport control is computed: isolated atoms are not a transport system.

Outputs under `tmp/orbital_torque_hex/source/` include `hall_scan.png` (gap
shaded, BSR versus ACA, I/E decomposition and SHC), `hall_scan.csv`, full
tensor arrays in `responses.npz`, `summary.json` with convergence and
midgap checks, and reproducible Hamiltonian inputs/arrays for the source model.
Tensor axes are Fermi, electric-field direction, current-flow direction,
moment polarization. The plotted component is Ex -> Jy of z moment.

Default results at 192x192: midgap BSR OHC is 0.64753851594 in native
normalization, entirely interband to numerical precision (I=4.94e-38).
ACA OHC and SHC are exactly zero in the saved arrays. With temperature and
linewidth halved, midgap BSR OHC is 0.64766603575. The maximum 96-to-192 mesh difference across the entire scan
is 1.97e-4; the midgap value agrees much more closely. All six source and
existing BSR-calculator tests pass, including energy-reference invariance.

## Why an insulating gap does not force this response to vanish

The response kernel contains occupied-to-empty interband coherences driven
by the electric field. Their occupation difference remains finite when the
chemical potential lies in the gap. No Fermi-surface population change is
needed for this contribution, unlike a Drude response. A nonzero Kubo Hall
coefficient consequently does not demonstrate dissipative longitudinal charge
transport or free carriers in the gap. Finite temperature/linewidth are
numerical/relaxation parameters; the midpoint checks distinguish the signal
from thermally activated population response.

The relevant literature describes gapped graphene's OHE using the BSR
orbital magnetic moment rather than an arbitrary valley label. Opposite
valleys have opposite moments and Berry curvatures, allowing their
moment-weighted Hall responses to add. See Bhowal and Vignale,
[Phys. Rev. B 103, 195309 (2021)](https://arxiv.org/abs/2103.03148).

This model is not a quantum spin Hall insulator: the two spin sectors are
identical rather than oppositely topological, and SHC cancels. More directly,
one can continuously reduce t to zero at fixed Delta without closing the
gap or breaking time reversal. The endpoint is an atomic insulator with
zero BSR response. Finite BSR OHC therefore does not by itself imply a
quantized plateau or protected helical edge channels. This atomic-limit
deformation is also an explicit control in our calculation.

## What remains to establish conversion and torque

The bulk result is a response of a chosen moment-current operator. It alone
does not prove measurable injection across an interface, an edge accumulation,
or torque on an attached magnet. BSR moment is k-dependent and spatially
nonlocal: the onsite ACA operators used in the first hierarchy cannot simply
be reused to measure it in a finite ribbon. Boundary conditions, moment
definition and current/torque balance must be treated consistently.

In an s-only Hilbert space, an onsite atomic L.S term is zero. A future
converter therefore needs additional orbital structure or a justified
effective intersite SOC coupling. Whether such a converter couples to the
source's intersite circulation is the physical question to test, not a
consequence we assume from the bulk OHC. Subsequent stages will compare
in-gap and metallic filling and include independent SOC, interface, exchange
and driving-location controls.
