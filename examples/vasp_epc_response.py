"""VASP linear EPC → main collision operator → LG/VG optical conductivity.

python examples/vasp_epc_response.py CONFIG.json REPORT.json
The configuration and scientific choices are described in README_vasp_epc.md.
Source files are read only; external preparation remains in epc.cli prepare.
"""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path
import numpy as np
from scipy.constants import Boltzmann, electron_volt
from epc import WannierEPCModel
from epc.runtime import _serializable
from system import System_w90
from elph import VaspWannierEPC
from phonon import PhononBath, PhononModeSelection
from calculator import (compute_states, compute_vertices, prepare_collision,
    CollisionSpecification, EPCLinewidth, Opticalconductivity_LG_DM, Opticalconductivity_VG_DM)


def run(record):
    if (not isinstance(record, dict) or record.get('schema') != 'main-vasp-epc/1'
            or not {'model', 'system', 'response'}.issubset(record)
            or set(record)-{'schema', 'model', 'phonons', 'system', 'response'}):
        raise ValueError('Expected main-vasp-epc/1 with model, system and response')
    model = WannierEPCModel(**record['model'], phonon_options=record.get('phonons'))
    system = System_w90.fromEPCReference(model, **record['system'])
    options = dict(record['response'])
    allowed = {'k_mesh', 'k_offset', 'q_mesh', 'phonon_selection', 'temperature_K',
        'fermi_ev', 'photon_energies_ev', 'gauge', 'collision', 'delta_ev', 'background_ev',
        'transition_storage', 'workspace_bytes', 'k_batch_size', 'degeneracy_policy',
        'degeneracy_tolerance_ev', 'spin_multiplicity'}
    if set(options)-allowed: raise ValueError('Unknown response settings')
    temperature = float(options['temperature_K'])
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('Positive finite temperature_K required')
    gauge, mode = options.get('gauge', 'LG'), options['collision']
    if gauge not in ('LG', 'VG') or mode not in ('serta', 'epc'):
        raise ValueError('gauge must be LG/VG and collision must be serta/epc')
    if mode == 'serta' and options.get('background_ev', 0.) != 0:
        raise ValueError('background_ev is a full-EPC BGK background; SERTA uses its EPC linewidth source')
    selection = options.get('phonon_selection')
    bath = PhononBath(temperature=temperature,
        qgrid=tuple(options.get('q_mesh', options['k_mesh'])),
        selection=None if selection is None else PhononModeSelection(**selection))
    workspace = options.get('workspace_bytes', 512*1024**2)
    epc = VaspWannierEPC(system, model, workspaceBytes=workspace)
    states = compute_states(system, tuple(options['k_mesh']),
                            kOffset=options.get('k_offset', (0., 0., 0.)))
    electronic = compute_vertices(states, highOrderVelocity=gauge == 'VG')
    fermi = np.atleast_1d(np.asarray(options['fermi_ev'], float))
    if fermi.ndim != 1 or not len(fermi) or not np.isfinite(fermi).all():
        raise ValueError('fermi_ev must be a finite scalar or nonempty list')
    omegas = np.atleast_1d(np.asarray(options.get('photon_energies_ev', [0.]), float))
    if omegas.ndim != 1 or not len(omegas) or not np.isfinite(omegas).all():
        raise ValueError('photon_energies_ev must be a finite nonempty list')
    kbt = Boltzmann*temperature/electron_volt
    multiplicity = float(options.get('spin_multiplicity', model.spin_multiplicity))
    if not np.isfinite(multiplicity) or multiplicity <= 0:
        raise ValueError('Positive spin_multiplicity required')
    response_class = Opticalconductivity_LG_DM if gauge == 'LG' else Opticalconductivity_VG_DM
    # SERTA is re-prepared for each mu rather than silently reusing linewidths
    # frozen at a different Fermi level. Full EPC reuses its transition graph.
    full = None
    if mode == 'epc':
        spec = CollisionSpecification('epc', phononBath=bath,
            delta=options.get('delta_ev', .03), constantLinewidth=options.get('background_ev', 0.),
            transitionStorage=options.get('transition_storage', 'auto'), workspaceBytes=workspace,
            kBatchSize=options.get('k_batch_size'))
        full = prepare_collision(electronic, spec, kBT=kbt, epc_model=epc)
    results = []
    for mu in fermi:
        line = None
        if full is None:
            line = EPCLinewidth(epc, float(mu), options.get('delta_ev', .03), phononBath=bath,
                workspaceBytes=workspace, kBatchSize=options.get('k_batch_size'),
                degeneracyPolicy=options.get('degeneracy_policy', 'block_trace'),
                degeneracyTolerance=options.get('degeneracy_tolerance_ev', 1e-8))
            spec = CollisionSpecification('serta', linewidthSource=line)
            prepared = prepare_collision(electronic, spec, kBT=kbt, epc_model=epc)
        else:
            prepared = full
        calc = response_class(electronic=electronic, preparedCollision=prepared, kBT=kbt,
            fermiList=np.array([mu]), omegaList=omegas, fullCovariantDerivative=True)
        results.append(dict(fermi_ev=float(mu), conductivity_s_per_m=multiplicity*calc.sigma[0],
            channels_s_per_m={key: multiplicity*value[0] for key, value in calc.sigma_channels.items()},
            sum_rule_residual=multiplicity*calc.sum_rule_residual[0],
            linewidth_info=None if line is None else line.info,
            collision_storage='local_linewidth' if full is None else full.operator.transitionStorage))
    return _serializable(dict(schema='main-vasp-epc/result/1', configuration=record,
        source_provenance=model.provenance, source_validation=model.validation,
        units='S/m', conductivity_axes='photon_energy,field,current', position_kind=system.epcPositionKind,
        spin_multiplicity=multiplicity, photon_energies_ev=omegas, results=results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path); parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    report = run(json.loads(args.config.read_text(encoding='utf-8')))
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__': main()
