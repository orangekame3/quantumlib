#!/usr/bin/env python3
"""
CHSH Circuit Factory - Dedicated Circuit Creation for CHSH Experiments
"""

from typing import Any

# Depends only on Qiskit (OQTOPUS-independent)
try:
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


class CHSHCircuitFactory:
    """
    Dedicated factory for creating CHSH circuits
    Pure circuit creation without dependence on OQTOPUS/quri-parts
    """

    @staticmethod
    def create_chsh_circuit(
        theta_a: float, theta_b: float, phase_phi: float = 0
    ) -> Any:
        """
        Create CHSH circuit (standard CHSH experiment)

        Args:
            theta_a: Alice measurement angle
            theta_b: Bob measurement angle
            phase_phi: Phase parameter (relative phase control of Bell state)

        Returns:
            Qiskit quantum circuit
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for circuit creation")

        # 2 quantum bits + 2 classical bits
        qubits = QuantumRegister(2, "q")
        bits = ClassicalRegister(2, "c")
        qc = QuantumCircuit(qubits, bits)

        # Create Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
        qc.h(0)
        qc.cx(0, 1)

        # Rotation of measurement basis (method from old implementation that works correctly)
        # Alice: Pauli-X measurement after θ_a rotation
        qc.ry(theta_a, 0)

        # Bob: Pauli-X measurement after θ_b + φ rotation (apply phase modulation to measurement angle)
        # This gives the correct dependence S(φ) = S₀ cos(φ)
        qc.ry(theta_b + phase_phi, 1)

        # Measurement (Z-basis measurement = X-basis measurement after rotation)
        qc.measure(0, 0)  # Alice → c[0]
        qc.measure(1, 1)  # Bob → c[1]

        return qc

    @staticmethod
    def create_bell_state_circuit() -> Any:
        """
        Create basic Bell state circuit
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for circuit creation")

        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        return qc

    @staticmethod
    def create_custom_bell_measurement(theta_a: float, theta_b: float) -> Any:
        """
        Custom Bell measurement circuit
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for circuit creation")

        qc = QuantumCircuit(2, 2)

        # Create Bell state
        qc.h(0)
        qc.cx(0, 1)

        # Custom measurement basis
        qc.ry(2 * theta_a, 0)
        qc.ry(2 * theta_b, 1)

        qc.measure(0, 0)
        qc.measure(1, 1)

        return qc


# Convenience functions
def create_chsh_circuit(theta_a: float, theta_b: float, phase_phi: float = 0) -> Any:
    """
    Convenience function for creating CHSH circuits
    """
    return CHSHCircuitFactory.create_chsh_circuit(theta_a, theta_b, phase_phi)


def create_bell_state() -> Any:
    """
    Convenience function for creating Bell states
    """
    return CHSHCircuitFactory.create_bell_state_circuit()


def create_custom_bell_measurement(theta_a: float, theta_b: float) -> Any:
    """
    Convenience function for creating custom Bell measurement circuits
    """
    return CHSHCircuitFactory.create_custom_bell_measurement(theta_a, theta_b)
