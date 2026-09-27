# Orbital-current-induced torque: toy-model hierarchy

This directory groups the example codes and investigation notes for a controlled
study of generation, boundary accumulation, SOC conversion, and exchange torque.
All commands below run from the repository root. Generated results are grouped
under `tmp/orbital_torque/`; model inputs and response data are regenerated there.

| Stage | Model and question | Implementation |
| --- | --- | --- |
| 1 | Periodic spin-independent square s+p source: finite OHC, zero SHC? | [orbital_hall_source.py](orbital_hall_source.py), [notes](README_orbital_hall_source.md) |
| 2 | Finite source ribbon: does a longitudinal field induce opposite edge orbital moments? | [orbital_ribbon.py](orbital_ribbon.py), [notes](README_orbital_ribbon.md) |
| 3 (planned) | Source plus SOC converter: distinguish transmitted orbital current and local spin conversion | Not implemented |
| 4 (planned) | Source, converter, and exchange-only magnet: measure exchange torque and spin balance | Not implemented |

The ribbon is an intermediate step inserted before the original two-zone SOC
junction. It contains no converter or magnetic region yet. Stages 1 and 2 share
the same model builder and parameters; the ribbon is a transverse supercell of
the source with its transverse boundary hoppings removed.

```text
python examples/orbital_torque/orbital_hall_source.py
python examples/orbital_torque/orbital_ribbon.py
```

## Common experimental logic

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
