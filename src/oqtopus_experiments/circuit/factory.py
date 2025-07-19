#!/usr/bin/env python3
"""
Circuit Factory - Integrated Circuit Factory
Integrates and provides experiment-specific circuit modules
"""

# Import experiment-specific circuit modules
from .chsh_circuits import (
    CHSHCircuitFactory,
    create_bell_state,
    create_chsh_circuit,
    create_custom_bell_measurement,
)
from .common_circuits import (
    CommonCircuitUtils,
    create_ghz_state,
    create_identity_circuit,
    get_circuit_info,
    optimize_circuit,
)
from .parity_circuits import (
    ParityCircuitFactory,
    create_coherence_decay_circuits,
    create_ghz_with_delay_rotation,
    create_parity_scan_circuits,
)
from .rabi_circuits import (
    RabiCircuitFactory,
    create_rabi_circuit,
    create_ramsey_circuit,
    create_t1_circuit,
    create_t2_echo_circuit,
)

# Make all convenience functions available from here as well
__all__ = [
    # CHSH-related
    "CHSHCircuitFactory",
    "create_chsh_circuit",
    "create_bell_state",
    "create_custom_bell_measurement",
    # Parity oscillation-related
    "ParityCircuitFactory",
    "create_ghz_with_delay_rotation",
    "create_parity_scan_circuits",
    "create_coherence_decay_circuits",
    # Rabi-related
    "RabiCircuitFactory",
    "create_rabi_circuit",
    "create_ramsey_circuit",
    "create_t1_circuit",
    "create_t2_echo_circuit",
    # Common utilities
    "CommonCircuitUtils",
    "optimize_circuit",
    "get_circuit_info",
    "create_identity_circuit",
    "create_ghz_state",
]
