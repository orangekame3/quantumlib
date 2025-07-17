#!/usr/bin/env python3
"""
Common Circuit Utilities - Common Circuit Utilities
"""

from typing import Any

# Depends only on Qiskit (OQTOPUS-independent)
try:
    from qiskit import QuantumCircuit, transpile

    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


class CommonCircuitUtils:
    """
    Common circuit utilities class
    """

    @staticmethod
    def optimize_circuit(circuit: Any, optimization_level: int = 1) -> Any:
        """
        Circuit optimization (using Qiskit)
        """
        if not QISKIT_AVAILABLE:
            return circuit

        try:
            return transpile(circuit, optimization_level=optimization_level)
        except ImportError:
            return circuit

    @staticmethod
    def get_circuit_info(circuit: Any) -> dict[str, Any]:
        """
        Get circuit information
        """
        if not QISKIT_AVAILABLE or circuit is None:
            return {}

        return {
            "depth": circuit.depth(),
            "gate_count": len(circuit.data),
            "qubit_count": circuit.num_qubits,
            "classical_bits": circuit.num_clbits,
            "gates": [instr.operation.name for instr in circuit.data],
        }

    @staticmethod
    def create_identity_circuit(num_qubits: int) -> Any:
        """
        Create identity circuit
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for circuit creation")

        qc = QuantumCircuit(num_qubits, num_qubits)
        # Do nothing (identity)
        qc.measure_all()
        return qc

    @staticmethod
    def create_ghz_state(num_qubits: int) -> Any:
        """
        Create GHZ state circuit
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for circuit creation")

        qc = QuantumCircuit(num_qubits, num_qubits)

        # Create GHZ state
        qc.h(0)  # Hadamard on first qubit
        for i in range(1, num_qubits):
            qc.cx(0, i)  # CNOT chain

        qc.measure_all()
        return qc


# Convenience functions
def optimize_circuit(circuit: Any, level: int = 1) -> Any:
    """
    Convenience function for circuit optimization
    """
    return CommonCircuitUtils.optimize_circuit(circuit, level)


def get_circuit_info(circuit: Any) -> dict[str, Any]:
    """
    Convenience function for getting circuit information
    """
    return CommonCircuitUtils.get_circuit_info(circuit)


def create_identity_circuit(num_qubits: int) -> Any:
    """
    Convenience function for creating identity circuits
    """
    return CommonCircuitUtils.create_identity_circuit(num_qubits)


def create_ghz_state(num_qubits: int) -> Any:
    """
    Convenience function for creating GHZ state circuits
    """
    return CommonCircuitUtils.create_ghz_state(num_qubits)
