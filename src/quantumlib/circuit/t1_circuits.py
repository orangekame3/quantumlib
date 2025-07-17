#!/usr/bin/env python3
"""
T1 Circuit Factory - Dedicated Circuit Creation for T1 Experiments
"""

from typing import Any

# Depends only on Qiskit (OQTOPUS-independent)
try:
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from qiskit_aer.noise import NoiseModel, thermal_relaxation_error

    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


class T1CircuitFactory:
    """
    Dedicated factory for creating T1 experiment circuits
    Single qubit T1 decay measurement
    """

    @staticmethod
    def create_t1_circuit(delay_time: float, t1: float = 500, t2: float = 500) -> Any:
        """
        Create T1 decay measurement circuit

        Args:
            delay_time: Delay time [ns]
            t1: T1 relaxation time [ns]
            t2: T2 relaxation time [ns]

        Returns:
            Qiskit quantum circuit
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for circuit creation")

        # 1 quantum bit + 1 classical bit
        qubits = QuantumRegister(1, "q")
        bits = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qubits, bits)

        # Excite to |1⟩ state
        qc.x(0)

        # Wait during delay time (using delay)
        qc.delay(int(delay_time), 0, unit="ns")

        # Z-basis measurement
        qc.measure(0, 0)

        return qc

    @staticmethod
    def create_t1_with_noise_model(
        delay_time: float, t1: float = 500, t2: float = 500
    ) -> tuple[Any, Any]:
        """
        Create T1 decay measurement circuit with noise model

        Args:
            delay_time: Delay time [ns]
            t1: T1 relaxation time [ns]
            t2: T2 relaxation time [ns]

        Returns:
            (Qiskit quantum circuit, noise model)
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for circuit creation")

        # Build noise model
        error = thermal_relaxation_error(t1, t2, delay_time)
        noise_model = NoiseModel()
        noise_model.add_quantum_error(error, "delay", [0])

        # Circuit definition
        qubits = QuantumRegister(1, "q")
        bits = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qubits, bits)

        # Excite to |1⟩ state
        qc.x(0)

        # Wait during delay time (using delay)
        qc.delay(int(delay_time), 0, unit="ns")

        # Z-basis measurement
        qc.measure(0, 0)

        return qc, noise_model

    @staticmethod
    def create_multiple_t1_circuits(
        delay_times: list[float], t1: float = 500, t2: float = 500
    ) -> list[Any]:
        """
        Create T1 circuits for multiple delay times

        Args:
            delay_times: List of delay times [ns]
            t1: T1 relaxation time [ns]
            t2: T2 relaxation time [ns]

        Returns:
            List of Qiskit quantum circuits
        """
        circuits = []
        for delay_time in delay_times:
            circuit = T1CircuitFactory.create_t1_circuit(delay_time, t1, t2)
            circuits.append(circuit)
        return circuits


# Convenience functions
def create_t1_circuit(delay_time: float, t1: float = 500, t2: float = 500) -> Any:
    """
    Convenience function for creating T1 decay measurement circuits
    """
    return T1CircuitFactory.create_t1_circuit(delay_time, t1, t2)


def create_t1_with_noise_model(
    delay_time: float, t1: float = 500, t2: float = 500
) -> tuple[Any, Any]:
    """
    Convenience function for creating T1 decay measurement circuits with noise model
    """
    return T1CircuitFactory.create_t1_with_noise_model(delay_time, t1, t2)


def create_multiple_t1_circuits(
    delay_times: list[float], t1: float = 500, t2: float = 500
) -> list[Any]:
    """
    Convenience function for creating multiple T1 circuits
    """
    return T1CircuitFactory.create_multiple_t1_circuits(delay_times, t1, t2)
