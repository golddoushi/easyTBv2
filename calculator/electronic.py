"""Explicit, reusable results of electronic preparation on a uniform mesh.

The Grid Energy/Velocity classes are stage builders. These results contain
their data, not a response calculator or collision state. An unchanged model
and its prepared arrays form one calculation lifetime.
"""

from dataclasses import dataclass, fields
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
    ham_K: np.ndarray | None = None
    E_K_AO: np.ndarray | None = None
    O_K_AO: np.ndarray | None = None

    @property
    def kpoints(self):
        from auxiliary import uniform_qmesh
        return uniform_qmesh(self.kgrid) + self.kOffset

@dataclass(frozen=True)
class ElectronicVertices:
    states: ElectronicStates
    velocity_K: np.ndarray
    RHam_K: np.ndarray
    projected_velocity_K_list: tuple = ()
    w_K: np.ndarray | None = None
    u_K: np.ndarray | None = None
    exchangeVelocity: bool = False
    velocityProjectionList: tuple = ()
    RExchange_K: np.ndarray | None = None
    RRExchange_K: np.ndarray | None = None
    exchange_K: np.ndarray | None = None
    exchange_K_AO: np.ndarray | None = None
    exchange_R_intepolated: np.ndarray | None = None
    orbitalTorqueAA_K: np.ndarray | None = None
    orbitalTorqueA_K: np.ndarray | None = None
    orbitalTorqueE_K: np.ndarray | None = None
    orbitalTorqueProjectionList: list | tuple | None = None
    orbitalTorque_K: np.ndarray | None = None
    orbitalVelocity_K: np.ndarray | None = None
    orbital_K: np.ndarray | None = None
    ordered_velocity_derivative_K: np.ndarray | None = None
    connection_K: np.ndarray | None = None
    orbitalA_K: np.ndarray | None = None
    projected_orbitalA_K_list: list | tuple | None = None
    projected_orbitalTorqueAA_K_list: list | tuple | None = None
    projected_orbitalTorqueA_K_list: list | tuple | None = None
    projected_orbitalTorqueE_K_list: list | tuple | None = None
    projected_orbitalTorque_K_list: list | tuple | None = None
    projected_orbitalVelocity_K_list: list | tuple | None = None
    projected_orbital_K_list: list | tuple | None = None
    projected_spinExchangeTorqueAA_K_list: list | tuple | None = None
    projected_spinExchangeTorqueA_K_list: list | tuple | None = None
    projected_spinExchangeTorqueE_K_list: list | tuple | None = None
    projected_spinExchangeTorque_K_list: list | tuple | None = None
    projected_spinTorqueAA_K_list: list | tuple | None = None
    projected_spinTorqueA_K_list: list | tuple | None = None
    projected_spinTorqueE_K_list: list | tuple | None = None
    projected_spinTorque_K_list: list | tuple | None = None
    projected_spinVelocity_K_list: list | tuple | None = None
    projected_spin_K_list: list | tuple | None = None
    r_K: np.ndarray | None = None
    r_deriv_K: np.ndarray | None = None
    spinExchangeTorqueAA_K: np.ndarray | None = None
    spinExchangeTorqueA_K: np.ndarray | None = None
    spinExchangeTorqueE_K: np.ndarray | None = None
    spinExchangeTorque_K: np.ndarray | None = None
    spinTorqueAA_K: np.ndarray | None = None
    spinTorqueA_K: np.ndarray | None = None
    spinTorqueE_K: np.ndarray | None = None
    spinTorqueProjectionList: list | tuple | None = None
    spinTorque_K: np.ndarray | None = None
    spinVelocity_K: np.ndarray | None = None
    soc_K: np.ndarray | None = None
    soc_K_AO: np.ndarray | None = None
    spin_K: np.ndarray | None = None
    velocity_xc_K: np.ndarray | None = None
    velocity_xc_bare_K: np.ndarray | None = None
    w_xc_K: np.ndarray | None = None
    w_xc_bare_K: np.ndarray | None = None

    def require(self, *names):
        """Declare a solver's tensor dependencies before entering its loops."""
        missing = [name for name in names if getattr(self, name) is None]
        if missing:
            raise ValueError('Electronic preparation is missing: ' + ', '.join(missing))


def states_from_stage(stage):
    """Publish the Energy stage once, preserving the actual band basis."""
    return ElectronicStates(stage.model, tuple(stage.kgrid), stage.kOffset,
        stage.energies, stage.vectors, stage.phaseFactorR, stage.ibmin, stage.skip_low,
        getattr(stage, 'ham_K', None), getattr(stage, 'E_K_AO', None), getattr(stage, 'O_K_AO', None))


def vertices_from_stage(stage, projections):
    additional = {name: getattr(stage, name) for name in
        ('ordered_velocity_derivative_K', 'connection_K',
         'exchange_K', 'exchange_K_AO', 'RExchange_K', 'RRExchange_K',
         'velocity_xc_bare_K', 'velocity_xc_K', 'w_xc_bare_K', 'w_xc_K')
        if hasattr(stage, name)}
    return ElectronicVertices(stage.states, stage.velocity_K, stage.RHam_K,
        tuple(stage.projected_velocity_K_list), getattr(stage, 'w_K', None),
        getattr(stage, 'u_K', None), stage.exchangeVelocity, tuple(projections), **additional)


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


class ElectronicTensorView:
    """Declared read-only tensor references shared by solver inputs and Grid."""

    @property
    def energies(self):
        return self.electronic.states.energies

    @property
    def vectors(self):
        return self.electronic.states.vectors

    @property
    def phaseFactorR(self):
        return self.electronic.states.phaseFactorR

    @property
    def ibmin(self):
        return self.electronic.states.ibmin

    @property
    def ham_K(self):
        return self.electronic.states.ham_K

    @property
    def E_K_AO(self):
        return self.electronic.states.E_K_AO

    @property
    def O_K_AO(self):
        return self.electronic.states.O_K_AO

    @property
    def velocity_K(self):
        return self.electronic.velocity_K

    @property
    def RHam_K(self):
        return self.electronic.RHam_K

    @property
    def projected_velocity_K_list(self):
        return list(self.electronic.projected_velocity_K_list)

    @property
    def w_K(self):
        return self.electronic.w_K

    @property
    def u_K(self):
        return self.electronic.u_K

    @property
    def exchangeVelocity(self):
        return self.electronic.exchangeVelocity

    @property
    def RExchange_K(self):
        return self.electronic.RExchange_K

    @property
    def RRExchange_K(self):
        return self.electronic.RRExchange_K

    @property
    def exchange_K(self):
        return self.electronic.exchange_K

    @property
    def exchange_K_AO(self):
        return self.electronic.exchange_K_AO

    @property
    def exchange_R_intepolated(self):
        return self.electronic.exchange_R_intepolated

    @property
    def orbitalTorqueAA_K(self):
        return self.electronic.orbitalTorqueAA_K

    @property
    def orbitalTorqueA_K(self):
        return self.electronic.orbitalTorqueA_K

    @property
    def orbitalTorqueE_K(self):
        return self.electronic.orbitalTorqueE_K

    @property
    def orbitalTorqueProjectionList(self):
        return self.electronic.orbitalTorqueProjectionList

    @property
    def orbitalTorque_K(self):
        return self.electronic.orbitalTorque_K

    @property
    def orbitalVelocity_K(self):
        return self.electronic.orbitalVelocity_K

    @property
    def orbital_K(self):
        return self.electronic.orbital_K

    @property
    def orbitalA_K(self):
        return self.electronic.orbitalA_K

    @property
    def projected_orbitalA_K_list(self):
        return self.electronic.projected_orbitalA_K_list

    @property
    def projected_orbitalTorqueAA_K_list(self):
        return self.electronic.projected_orbitalTorqueAA_K_list

    @property
    def projected_orbitalTorqueA_K_list(self):
        return self.electronic.projected_orbitalTorqueA_K_list

    @property
    def projected_orbitalTorqueE_K_list(self):
        return self.electronic.projected_orbitalTorqueE_K_list

    @property
    def projected_orbitalTorque_K_list(self):
        return self.electronic.projected_orbitalTorque_K_list

    @property
    def projected_orbitalVelocity_K_list(self):
        return self.electronic.projected_orbitalVelocity_K_list
    
    @property
    def projected_orbital_K_list(self):
        return self.electronic.projected_orbital_K_list

    @property
    def projected_spinExchangeTorqueAA_K_list(self):
        return self.electronic.projected_spinExchangeTorqueAA_K_list

    @property
    def projected_spinExchangeTorqueA_K_list(self):
        return self.electronic.projected_spinExchangeTorqueA_K_list

    @property
    def projected_spinExchangeTorqueE_K_list(self):
        return self.electronic.projected_spinExchangeTorqueE_K_list

    @property
    def projected_spinExchangeTorque_K_list(self):
        return self.electronic.projected_spinExchangeTorque_K_list

    @property
    def projected_spinTorqueAA_K_list(self):
        return self.electronic.projected_spinTorqueAA_K_list

    @property
    def projected_spinTorqueA_K_list(self):
        return self.electronic.projected_spinTorqueA_K_list

    @property
    def projected_spinTorqueE_K_list(self):
        return self.electronic.projected_spinTorqueE_K_list

    @property
    def projected_spinTorque_K_list(self):
        return self.electronic.projected_spinTorque_K_list

    @property
    def projected_spinVelocity_K_list(self):
        return self.electronic.projected_spinVelocity_K_list

    @property
    def projected_spin_K_list(self):
        return self.electronic.projected_spin_K_list

    @property
    def r_K(self):
        return self.electronic.r_K

    @property
    def r_deriv_K(self):
        return self.electronic.r_deriv_K

    @property
    def spinExchangeTorqueAA_K(self):
        return self.electronic.spinExchangeTorqueAA_K

    @property
    def spinExchangeTorqueA_K(self):
        return self.electronic.spinExchangeTorqueA_K

    @property
    def spinExchangeTorqueE_K(self):
        return self.electronic.spinExchangeTorqueE_K

    @property
    def spinExchangeTorque_K(self):
        return self.electronic.spinExchangeTorque_K

    @property
    def spinTorqueAA_K(self):
        return self.electronic.spinTorqueAA_K

    @property
    def spinTorqueA_K(self):
        return self.electronic.spinTorqueA_K

    @property
    def spinTorqueE_K(self):
        return self.electronic.spinTorqueE_K

    @property
    def spinTorqueProjectionList(self):
        return self.electronic.spinTorqueProjectionList

    @property
    def spinTorque_K(self):
        return self.electronic.spinTorque_K

    @property
    def spinVelocity_K(self):
        return self.electronic.spinVelocity_K

    @property
    def spin_K(self):
        return self.electronic.spin_K

    @property
    def velocity_xc_K(self):
        return self.electronic.velocity_xc_K

    @property
    def velocity_xc_bare_K(self):
        return self.electronic.velocity_xc_bare_K

    @property
    def w_xc_K(self):
        return self.electronic.w_xc_K

    @property
    def w_xc_bare_K(self):
        return self.electronic.w_xc_bare_K


def restore_states(stage, states):
    """Initialize the mutable electronic stage builder from declared state fields."""
    if states.energies is not None:
        stage.energies = states.energies
    if states.vectors is not None:
        stage.vectors = states.vectors
    if states.phaseFactorR is not None:
        stage.phaseFactorR = states.phaseFactorR
    if states.ibmin is not None:
        stage.ibmin = states.ibmin
    if states.ham_K is not None:
        stage.ham_K = states.ham_K
    if states.E_K_AO is not None:
        stage.E_K_AO = states.E_K_AO
    if states.O_K_AO is not None:
        stage.O_K_AO = states.O_K_AO


def restore_vertices(stage, electronic):
    """Initialize a mutable builder from the declared ElectronicVertices schema.

    Response solvers use read-only ElectronicTensorView properties instead.
    """
    restore_states(stage, electronic.states)
    for item in fields(ElectronicVertices):
        if item.name in ('states', 'velocityProjectionList'):
            continue
        value = getattr(electronic, item.name)
        if value is not None:
            if item.name == 'projected_velocity_K_list':
                value = list(value)
            setattr(stage, item.name, value)
