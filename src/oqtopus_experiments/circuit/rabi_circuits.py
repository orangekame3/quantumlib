#!/usr/bin/env python3
"""
Rabi Circuit Factory - Dedicated Circuit Creation for Rabi Experiments
"""

from typing import Any

import numpy as np

# Depends only on Qiskit (OQTOPUS-independent)
try:
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


class RabiCircuitFactory:
    """
    Dedicated factory for creating Rabi oscillation experiment circuits
    Single qubit Rabi oscillation measurement
    """

    @staticmethod
    def create_rabi_circuit(
        drive_amplitude: float, drive_time: float, drive_frequency: float = 0.0
    ) -> Any:
        """
        Create Rabi oscillation circuit

        Args:
            drive_amplitude: Drive amplitude (corresponds to rotation angle)
            drive_time: Drive time (actually used as angle)
            drive_frequency: Drive frequency (used as phase)

        Returns:
            Qiskit quantum circuit
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for circuit creation")

        # 1 quantum bit + 1 classical bit
        qubits = QuantumRegister(1, "q")
        bits = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qubits, bits)

        # Start from |0⟩ state

        # Rabi drive: RX rotation (amplitude × time = rotation angle)
        angle = drive_amplitude * drive_time

        # Consider phase due to frequency
        if drive_frequency != 0.0:
            qc.rz(drive_frequency, 0)  # Z-axis phase rotation

        # Rotation around X-axis (Rabi drive)
        qc.rx(angle, 0)

        # Z-basis measurement
        qc.measure(0, 0)

        return qc

    @staticmethod
    def create_ramsey_circuit(wait_time: float, phase: float = 0.0) -> Any:
        """
        Create Ramsey experiment circuit

        Args:
            wait_time: Wait time (corresponds to phase accumulation)
            phase: Phase of the final π/2 pulse

        Returns:
            Qiskit quantum circuit
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for circuit creation")

        qc = QuantumCircuit(1, 1)

        # First π/2 pulse (create superposition state)
        qc.ry(np.pi / 2, 0)

        # Phase accumulation due to wait time (simulated with Z rotation)
        qc.rz(wait_time, 0)

        # Second π/2 pulse (with phase)
        if phase != 0.0:
            qc.rz(phase, 0)
        qc.ry(np.pi / 2, 0)

        qc.measure(0, 0)

        return qc

    @staticmethod
    def create_t1_circuit(delay_time: float) -> Any:
        """
        Create T1 decay measurement circuit

        Args:
            delay_time: Delay time

        Returns:
            Qiskit quantum circuit
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for circuit creation")

        qc = QuantumCircuit(1, 1)

        # Excite to |1⟩ state
        qc.x(0)

        # Delay (in actual implementation this is waiting, here substituted with identity)
        # In actual hardware, delay causes T1 decay
        for _ in range(int(delay_time * 10)):  # Pseudo delay
            qc.id(0)

        qc.measure(0, 0)

        return qc

    @staticmethod
    def create_t2_echo_circuit(wait_time: float) -> Any:
        """
        Create T2 echo measurement circuit (spin echo)

        Args:
            wait_time: Wait time

        Returns:
            Qiskit quantum circuit
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for circuit creation")

        qc = QuantumCircuit(1, 1)

        # π/2 pulse (create superposition state)
        qc.ry(np.pi / 2, 0)

        # τ/2 wait
        qc.rz(wait_time / 2, 0)

        # π pulse (echo)
        qc.x(0)

        # τ/2 wait
        qc.rz(wait_time / 2, 0)

        # Final π/2 pulse
        qc.ry(np.pi / 2, 0)

        qc.measure(0, 0)

        return qc


# Convenience functions
def create_rabi_circuit(
    drive_amplitude: float, drive_time: float, drive_frequency: float = 0.0
) -> Any:
    """
    Convenience function for creating Rabi oscillation circuits
    """
    return RabiCircuitFactory.create_rabi_circuit(
        drive_amplitude, drive_time, drive_frequency
    )


def create_ramsey_circuit(wait_time: float, phase: float = 0.0) -> Any:
    """
    Convenience function for creating Ramsey experiment circuits
    """
    return RabiCircuitFactory.create_ramsey_circuit(wait_time, phase)


def create_t1_circuit(delay_time: float) -> Any:
    """
    Convenience function for creating T1 decay measurement circuits
    """
    return RabiCircuitFactory.create_t1_circuit(delay_time)


def create_t2_echo_circuit(wait_time: float) -> Any:
    """
    Convenience function for creating T2 echo measurement circuits
    """
    return RabiCircuitFactory.create_t2_echo_circuit(wait_time)
