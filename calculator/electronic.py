"""Explicit, reusable results of electronic preparation on a uniform mesh.

The Grid Energy/Velocity classes are stage builders. These results contain
their data, not a response calculator or collision state. An unchanged model
and its prepared arrays form one calculation lifetime.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class ElectronicStates:
    model: object
    kgrid: tuple
    kOffset: np.ndarray
    energies: np.ndarray
    vectors: np.ndarray
    phaseFactorR: np.ndarray
    ibmin: int = 0
    skip_low: bool = False
    ham_K: object = None
    E_K_AO: object = None
    O_K_AO: object = None

    @property
    def kpoints(self):
        from auxiliary import uniform_qmesh
        return uniform_qmesh(self.kgrid) + self.kOffset

    def arrays(self):
        return {name: getattr(self, name) for name in
                ('energies', 'vectors', 'phaseFactorR', 'ibmin', 'ham_K', 'E_K_AO', 'O_K_AO')
                if getattr(self, name) is not None}


@dataclass(frozen=True)
class ElectronicVertices:
    states: ElectronicStates
    velocity_K: np.ndarray
    RHam_K: np.ndarray
    projected_velocity_K_list: tuple = ()
    w_K: object = None
    u_K: object = None
    exchangeVelocity: bool = False
    velocityProjectionList: tuple = ()
    # Additional electronic stages (dipole, spin, orbital, torque) publish
    # named tensors. They never contain response settings or collision state.
    additional: dict = field(default_factory=dict)

    def arrays(self):
        values = self.states.arrays()
        values.update(velocity_K=self.velocity_K, RHam_K=self.RHam_K,
                      projected_velocity_K_list=list(self.projected_velocity_K_list),
                      exchangeVelocity=self.exchangeVelocity)
        if self.w_K is not None:
            values['w_K'] = self.w_K
        if self.u_K is not None:
            values['u_K'] = self.u_K
        values.update(self.additional)
        return values


def states_from_stage(stage):
    """Publish the Energy stage once, preserving the actual band basis."""
    return ElectronicStates(stage.model, tuple(stage.kgrid), stage.kOffset,
        stage.energies, stage.vectors, stage.phaseFactorR, stage.ibmin, stage.skip_low,
        getattr(stage, 'ham_K', None), getattr(stage, 'E_K_AO', None), getattr(stage, 'O_K_AO', None))


def vertices_from_stage(stage, projections):
    exchange = {name: getattr(stage, name) for name in
        ('exchange_K', 'exchange_K_AO', 'RExchange_K', 'RRExchange_K',
         'velocity_xc_bare_K', 'velocity_xc_K', 'w_xc_bare_K', 'w_xc_K')
        if hasattr(stage, name)}
    return ElectronicVertices(stage.states, stage.velocity_K, stage.RHam_K,
        tuple(stage.projected_velocity_K_list), getattr(stage, 'w_K', None),
        getattr(stage, 'u_K', None), stage.exchangeVelocity, tuple(projections), exchange)


def compute_states(model, kgrid=(1, 1, 1), *, kOffset=(0., 0., 0.), skip_low=False):
    """Diagonalize once and return explicit electronic states."""
    from .electronic_stages import Energy
    return Energy(model=model, kgrid=kgrid, kOffset=kOffset, skip_low=skip_low).states


def compute_vertices(states, *, highOrderVelocity=False, exchangeVelocity=False,
                     velocityProjectionList=()):
    """Construct velocity vertices using exactly the supplied eigenvectors."""
    from .electronic_stages import Velocity
    return Velocity(states=states, highOrderVelocity=highOrderVelocity,
        exchangeVelocity=exchangeVelocity, velocityProjectionList=velocityProjectionList).electronic


def resolve_states(states, settings):
    """Resolve mesh metadata at an entry point; never rediagonalize supplied states."""
    if states is None:
        return settings
    if not isinstance(states, ElectronicStates):
        raise TypeError('states must be ElectronicStates')
    settings = dict(settings)
    if settings.get('model', states.model) is not states.model:
        raise ValueError('Electronic states require the same model object')
    for name in ('kgrid', 'kOffset', 'skip_low'):
        if name in settings and not np.array_equal(settings[name], getattr(states, name)):
            raise ValueError(f'Electronic states do not match {name}')
    settings.update(model=states.model, kgrid=states.kgrid,
                    kOffset=states.kOffset, skip_low=states.skip_low)
    return settings
