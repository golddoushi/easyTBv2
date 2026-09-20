"""Composition of electronic preparation, collision preparation and responses."""

from dataclasses import replace
from .electronic import ElectronicVertices, compute_states, resolve_states
from .electronic_algebra import ElectronicAlgebra
from .response_inputs import ResponseInputs
from .collision import resolve_collision, collision_mode, prepare_phonon_background


def response_model(settings):
    electronic = settings.get('electronic')
    if electronic is not None:
        return electronic.states.model
    prepared = settings.get('preparedCollision')
    if prepared is not None:
        return prepared.inputs.model
    states = settings.get('states')
    return states.model if states is not None else settings['model']


def response_requires_full_basis(settings):
    electronic = settings.get('electronic')
    prepared = settings.get('preparedCollision')
    states = (electronic.states if electronic is not None else
              prepared.inputs.states if prepared is not None else settings.get('states'))
    truncated = states.skip_low if states is not None else settings.get('skip_low', False)
    return response_model(settings).overlap or truncated


class ElectronicResponse(ElectronicAlgebra):
    """Response entry point composing explicitly named electronic stages."""
    Iodd = False
    Todd = False

    def __init__(self, *, electronic=None, states=None, preparedCollision=None,
                 electronic_stages=(), velocityProjectionList=(),
                 highOrderVelocity=False, highOrderDipole=False, exchangeVelocity=False,
                 **settings):
        from .electronic_stages import Calculator, Velocity
        if preparedCollision is not None:
            settings.setdefault('kBT', preparedCollision.inputs.kBT)
        if electronic is None and preparedCollision is not None:
            electronic = preparedCollision.inputs.electronic
        if electronic is not None:
            if not isinstance(electronic, ElectronicVertices):
                raise TypeError('electronic must be ElectronicVertices')
            if states is not None and states is not electronic.states:
                raise ValueError('electronic and states must share the same prepared band basis')
            states = electronic.states
        settings = resolve_states(states, settings)
        Calculator.__init__(self, **settings)
        if states is None:
            states = compute_states(self.model, self.kgrid, kOffset=self.kOffset, skip_low=self.skip_low)
        # Compatible vertices are reused; additional requirements use the same
        # states, so extending LG preparation for VG never changes band gauge.
        velocity = Velocity(states=states, electronic=electronic,
            velocityProjectionList=velocityProjectionList,
            highOrderVelocity=highOrderVelocity or highOrderDipole,
            exchangeVelocity=exchangeVelocity)
        electronic = velocity.electronic
        self._electronic_stage_options = dict(settings,
            velocityProjectionList=velocityProjectionList, highOrderVelocity=highOrderVelocity,
            highOrderDipole=highOrderDipole, exchangeVelocity=exchangeVelocity)
        for stage_class in electronic_stages:
            stage = stage_class(states=states, electronic=electronic, **self._electronic_stage_options)
            # Only the fields published by that electronic stage are retained.
            from .stage_fields import ELECTRONIC_STAGE_FIELDS
            tensors = {name:getattr(stage, name) for name in ELECTRONIC_STAGE_FIELDS[stage_class.__name__]
                       if hasattr(stage, name)}
            electronic = replace(electronic, additional={**electronic.additional, **tensors})
        self.electronic = electronic
        self.states = states
        self.inputs = ResponseInputs(electronic, kBT=self.kBT, fermiList=self.fermiList,
            omegaList=self.omegaList, degen=self.degen, eta=self.eta, smr_type=self.smr_type)
        self.__dict__.update(electronic.arrays())

    def accept_first_order_result(self, result):
        self.result = result
        self.density_matrix_kernel = result.kernel
        self.responseFermiIndexList = result.fermi_indices
        self.responseOmegaIndexList = result.omega_indices

    def calcDeltaEInverse(self):
        """Share the input's band-pair arrays with the response entry point."""
        self.inputs.calcDeltaEInverse()
        for name in ('deltaE', 'deltaEInverse', 'deltaEInverse_pure', 'deltaEInverse2'):
            setattr(self, name, getattr(self.inputs, name))

    def prepare_torque_vertices(self, kind, derivative):
        """Publish a torque-contact stage using the prepared electronic arrays."""
        from .electronic_stages import SpinExchangeTorque, OrbitalTorque
        from .stage_fields import ELECTRONIC_STAGE_FIELDS
        stage_class = SpinExchangeTorque if kind == 'spin' else OrbitalTorque
        stage = stage_class(states=self.states, **self._electronic_stage_options)
        stage.__dict__.update(self.electronic.arrays())
        method = ('buildSpinTorque' if kind == 'spin' else 'buildOrbitalTorque')
        method += 'VelocityCommutator' if derivative == 'velocity' else 'DipoleCommutator'
        getattr(stage, method)()
        tensors = {name:getattr(stage, name) for name in ELECTRONIC_STAGE_FIELDS[stage_class.__name__]
                   if hasattr(stage, name)}
        self.electronic = replace(self.electronic, additional={**self.electronic.additional, **tensors})
        self.__dict__.update(tensors)
        self.inputs = ResponseInputs(self.electronic, kBT=self.kBT, fermiList=self.fermiList,
            omegaList=self.omegaList, degen=self.degen, eta=self.eta, smr_type=self.smr_type)
        self.calcDeltaEInverse()


class DensityMatrixResponse(ElectronicResponse):
    def __init__(self, *, collisionSpecification=None, preparedCollision=None, epcModel=None, **settings):
        collision_mode(collisionSpecification, preparedCollision)
        # These are sequential stages, independent of an electronic MRO.
        super().__init__(preparedCollision=preparedCollision, **settings)
        self.calcDeltaEInverse()
        self.preparedCollision = resolve_collision(self.electronic, collisionSpecification,
            prepared=preparedCollision, kBT=self.kBT, epc_model=epcModel)
        self.collision = self.preparedCollision.operator


class PhononResponse(ElectronicResponse):
    def __init__(self, *, collisionSpecification=None, preparedCollision=None, epcModel=None, **settings):
        if preparedCollision is not None:
            raise ValueError('Explicit/static phonon calculations do not accept preparedCollision; supply a CRTA background')
        super().__init__(**settings)
        self.epcModel = epcModel
        self.collision = prepare_phonon_background(self.electronic, collisionSpecification,
            kBT=self.kBT, epc_model=epcModel)
