#!/usr/bin/env python3
"""
Ramsey Circuit Factory - Dedicated Circuit Creation for Ramsey Experiments
"""

from typing import Any

import numpy as np

# Depends only on Qiskit
try:
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from qiskit_aer.noise import NoiseModel, thermal_relaxation_error

    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


class RamseyCircuitFactory:
    """
    Dedicated factory for creating Ramsey experiment circuits
    Single qubit Ramsey oscillation measurement
    """

    @staticmethod
    def create_ramsey_circuit(
        delay_time: float, detuning: float = 0.0, t1: float = 500, t2: float = 500
    ) -> Any:
        """
        Create Ramsey oscillation measurement circuit

        Args:
            delay_time: Delay time [ns]
            detuning: Frequency detuning [MHz]
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

        # First π/2 pulse
        qc.rx(np.pi / 2, 0)

        # Wait during delay time (free evolution)
        qc.delay(int(delay_time), 0, unit="ns")

        # Add phase rotation if there is detuning
        if detuning != 0.0:
            # phase = 2π × detuning [MHz] × delay_time [ns] × 1e-3
            phase = 2 * np.pi * detuning * delay_time * 1e-3
            qc.rz(phase, 0)

        # Second π/2 pulse (analysis pulse)
        qc.rx(np.pi / 2, 0)

        # Z-basis measurement
        qc.measure(0, 0)

        return qc

    @staticmethod
    def create_ramsey_with_noise_model(
        delay_time: float, detuning: float = 0.0, t1: float = 500, t2: float = 500
    ) -> tuple[Any, Any]:
        """
        Create Ramsey oscillation measurement circuit with noise model

        Args:
            delay_time: Delay time [ns]
            detuning: Frequency detuning [MHz]
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

        # First π/2 pulse
        qc.ry(np.pi / 2, 0)

        # Wait during delay time
        qc.delay(int(delay_time), 0, unit="ns")

        # Detuning phase
        if detuning != 0.0:
            phase = 2 * np.pi * detuning * delay_time * 1e-3
            qc.rz(phase, 0)

        # Second π/2 pulse
        qc.ry(np.pi / 2, 0)

        # Z-basis measurement
        qc.measure(0, 0)

        return qc, noise_model

    @staticmethod
    def create_multiple_ramsey_circuits(
        delay_times: list[float],
        detuning: float = 0.0,
        t1: float = 500,
        t2: float = 500,
    ) -> list[Any]:
        """
        Create Ramsey circuits for multiple delay times

        Args:
            delay_times: List of delay times [ns]
            detuning: Frequency detuning [MHz]
            t1: T1 relaxation time [ns]
            t2: T2 relaxation time [ns]

        Returns:
            List of Qiskit quantum circuits
        """
        circuits = []
        for delay_time in delay_times:
            circuit = RamseyCircuitFactory.create_ramsey_circuit(
                delay_time, detuning, t1, t2
            )
            circuits.append(circuit)
        return circuits


# Convenience functions
def create_ramsey_circuit(
    delay_time: float, detuning: float = 0.0, t1: float = 500, t2: float = 500
) -> Any:
    """
    Convenience function for creating Ramsey oscillation measurement circuits
    """
    return RamseyCircuitFactory.create_ramsey_circuit(delay_time, detuning, t1, t2)


def create_ramsey_with_noise_model(
    delay_time: float, detuning: float = 0.0, t1: float = 500, t2: float = 500
) -> tuple[Any, Any]:
    """
    Convenience function for creating Ramsey oscillation measurement circuits with noise model
    """
    return RamseyCircuitFactory.create_ramsey_with_noise_model(
        delay_time, detuning, t1, t2
    )


def create_multiple_ramsey_circuits(
    delay_times: list[float], detuning: float = 0.0, t1: float = 500, t2: float = 500
) -> list[Any]:
    """
    Convenience function for creating multiple Ramsey circuits
    """
    return RamseyCircuitFactory.create_multiple_ramsey_circuits(
        delay_times, detuning, t1, t2
    )
