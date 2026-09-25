"""Numerical mesh requirements available before electronic preparation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ComputationRequirements:
    """A complete mesh is needed for coupled solves or mesh derivatives.

    This says nothing about ownership of already prepared arrays: their exact
    mesh/basis binding is checked separately by the task adapter.
    """

    requires_full_mesh: bool = False

    def combined(self, other):
        return ComputationRequirements(self.requires_full_mesh or other.requires_full_mesh)


def collision_requirements(specification=None, prepared=None):
    from .collision import collision_mode
    collision_mode(specification, prepared)
    return (prepared.specification if prepared is not None else specification).requirements
