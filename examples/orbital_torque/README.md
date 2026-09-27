# Orbital-current-induced torque: toy-model hierarchy

This directory groups the example codes and investigation notes for a controlled
study of generation, boundary accumulation, SOC conversion, and exchange torque.
All commands below run from the repository root. Generated results are grouped
under `tmp/orbital_torque/`; model inputs and response data are regenerated there.

| Stage | Model and question | Implementation |
| --- | --- | --- |
| 1 | Periodic spin-independent square s+p source: finite OHC, zero SHC? | [orbital_hall_source.py](orbital_hall_source.py), [notes](README_orbital_hall_source.md) |
| 2 | Finite source ribbon: does a longitudinal field induce opposite edge orbital moments? | [orbital_ribbon.py](orbital_ribbon.py), [notes](README_orbital_ribbon.md) |
| 3 | N-C ribbon: orbital/spin accumulation with mixing, SOC, disconnected-interface and N-only-drive controls | [orbital_nc_ribbon.py](orbital_nc_ribbon.py), [notes](README_nc_ribbon.md) |
| 4 | N/C/F ribbon: local moments, exchange torque and driving-row decomposition with five controls | [orbital_ncf_ribbon.py](orbital_ncf_ribbon.py), [notes](README_ncf_ribbon.md) |

The step-2 ribbon is an intermediate step before the two-zone SOC junction.
It contains no converter or magnetic region. Stages 1 and 2 share
the same model builder and parameters; the ribbon is a transverse supercell of
the source with its transverse boundary hoppings removed.

```text
python examples/orbital_torque/orbital_hall_source.py
python examples/orbital_torque/orbital_ribbon.py
python examples/orbital_torque/orbital_nc_ribbon.py --workers 8
python examples/orbital_torque/orbital_ncf_ribbon.py --workers 8
```

## Common experimental logic

The ribbon supports reusable density evaluation and parallel k meshes.

Use an unpolarized spinful s+p model without SOC as the source. Turn on atomic
SOC only in the converter and explicit exchange only in the magnet when those
stages are introduced. Keep interface hopping spin independent initially.

Record full response tensors, intraband/interband channels, numerical settings,
and control calculations. Exact spin independence gives a strict zero spin
response; numerical residuals should still be retained. Disable s-p hybridization
as a source control, and later vary SOC, exchange, and interface transparency
independently. Converge meshes and ribbon widths at specified temperature and
relaxation before interpreting parameter trends.

Bulk OHC, edge orbital accumulation, transmitted orbital flux, spin conversion,
and exchange torque are distinct observables. Neither bulk OHC nor edge
accumulation alone establishes the full proposed mechanism. Later stages should
add bond/interface currents and local SOC, crystal-field, exchange, and
relaxation contributions to angular-momentum balance.

The bulk Hall output currently uses the calculators' historical moment-current
normalization. Ribbon moment accumulation instead has the galvanic calculator's
units of mu_B/(V/Angstrom) per selected row per longitudinal repeat. Do not
compare their numerical magnitudes as though they had the same units.
