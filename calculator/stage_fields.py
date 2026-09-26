"""Fields published by each additional electronic computation stage."""

ELECTRONIC_STAGE_FIELDS = {
    'Dipole': ('r_K', 'r_deriv_K'),
    'Spin': ('projected_spin_K_list', 'spin_K'),
    'Orbital': ('orbital_K', 'projected_orbital_K_list'),
    'SpinExchangeTorque': ('exchange_K', 'exchange_K_AO', 'exchange_R_intepolated', 'projected_spinExchangeTorqueAA_K_list', 'projected_spinExchangeTorqueA_K_list', 'projected_spinExchangeTorqueE_K_list', 'projected_spinExchangeTorque_K_list', 'projected_spinTorqueAA_K_list', 'projected_spinTorqueA_K_list', 'projected_spinTorqueE_K_list', 'projected_spinTorque_K_list', 'projected_spin_K_list', 'spinExchangeTorqueAA_K', 'spinExchangeTorqueA_K', 'spinExchangeTorqueE_K', 'spinExchangeTorque_K', 'spinTorqueAA_K', 'spinTorqueA_K', 'spinTorqueE_K', 'spinTorqueProjectionList', 'spinTorque_K', 'spin_K'),
    'OrbitalTorque': ('orbitalTorqueAA_K', 'orbitalTorqueA_K', 'orbitalTorqueE_K', 'orbitalTorqueProjectionList', 'orbitalTorque_K', 'orbital_K', 'projected_orbitalTorqueAA_K_list', 'projected_orbitalTorqueA_K_list', 'projected_orbitalTorqueE_K_list', 'projected_orbitalTorque_K_list', 'projected_orbital_K_list'),
    'SpinVelocity': ('projected_spinVelocity_K_list', 'projected_spin_K_list', 'spinVelocity_K', 'spin_K'),
    'OrbitalVelocity': ('projected_orbitalVelocity_K_list', 'projected_orbital_K_list', 'orbitalVelocity_K', 'orbital_K'),
}
