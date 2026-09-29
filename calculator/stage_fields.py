"""Fields published by each additional electronic computation stage."""

ELECTRONIC_STAGE_FIELDS = {
    'Dipole': ('r_K', 'r_deriv_K'),
    'Spin': ('projected_spin_K_list', 'spin_K'),
    'Orbital_BlochSelfRotate': ('orbital_K', 'projected_orbital_K_list', 'orbitalA_K', 'projected_orbitalA_K_list', 'ordered_velocity_derivative_K', 'connection_K'),
    'Orbital': ('orbital_K', 'projected_orbital_K_list'),
    'SpinSOCTorque': ('soc_K', 'soc_K_AO', 'spin_K', 'spinTorque_K', 'spinTorqueProjectionList', 'projected_spinTorque_K_list', 'spinTorqueA_K', 'projected_spinTorqueA_K_list'),
    'SpinExchangeTorque': ('exchange_K', 'exchange_K_AO', 'exchange_R_intepolated', 'projected_spinExchangeTorqueAA_K_list', 'projected_spinExchangeTorqueA_K_list', 'projected_spinExchangeTorqueE_K_list', 'projected_spinExchangeTorque_K_list', 'projected_spinTorqueAA_K_list', 'projected_spinTorqueA_K_list', 'projected_spinTorqueE_K_list', 'projected_spinTorque_K_list', 'projected_spin_K_list', 'spinExchangeTorqueAA_K', 'spinExchangeTorqueA_K', 'spinExchangeTorqueE_K', 'spinExchangeTorque_K', 'spinTorqueAA_K', 'spinTorqueA_K', 'spinTorqueE_K', 'spinTorqueProjectionList', 'spinTorque_K', 'spin_K'),
    'OrbitalTorque': ('orbitalTorqueAA_K', 'orbitalTorqueA_K', 'orbitalTorqueE_K', 'orbitalTorqueProjectionList', 'orbitalTorque_K', 'orbital_K', 'projected_orbitalTorqueAA_K_list', 'projected_orbitalTorqueA_K_list', 'projected_orbitalTorqueE_K_list', 'projected_orbitalTorque_K_list', 'projected_orbital_K_list'),
    'SpinVelocity': ('projected_spinVelocity_K_list', 'projected_spin_K_list', 'spinVelocity_K', 'spin_K'),
    'OrbitalVelocity': ('projected_orbitalVelocity_K_list', 'projected_orbital_K_list', 'orbitalVelocity_K', 'orbital_K'),
    'OrbitalVelocity_BlochSelfRotate': ('projected_orbitalVelocity_K_list', 'projected_orbital_K_list', 'orbitalVelocity_K', 'orbital_K'),
}
