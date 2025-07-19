#!/usr/bin/env python3
"""
Parity Oscillation Circuit Factory - Circuits for parity oscillation experiments
Specialized circuits for studying GHZ state decoherence through parity measurements
"""

from typing import Any

import numpy as np

try:
    from qiskit import QuantumCircuit

    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


class ParityCircuitFactory:
    """
    Factory class for creating parity oscillation experiment circuits
    """

    @staticmethod
    def create_ghz_with_delay_rotation(
        num_qubits: int,
        delay_us: float = 0.0,
        phi: float = 0.0,
        identity_gate_time_ns: float = 90.0,
    ) -> Any:
        """
        Create GHZ state circuit with delay and Y-rotation analysis

        Circuit structure matches Ozaeta & McMahon (2019):
        1. Generate N-qubit GHZ state |0...0⟩ + |1...1⟩
        2. Apply delay τ using identity gates
        3. Apply rotation U(φ) = exp(-iφσy/2) to each qubit
        4. Measure in computational basis

        Args:
            num_qubits: Number of qubits in GHZ state
            delay_us: Delay time in microseconds
            phi: Rotation phase φ around Y-axis
            identity_gate_time_ns: Time per identity gate in nanoseconds (default: 90ns)

        Returns:
            Quantum circuit for parity oscillation measurement
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for parity oscillation circuits")

        if num_qubits < 1:
            raise ValueError("Number of qubits must be at least 1")

        qc = QuantumCircuit(num_qubits, num_qubits)

        # Step 1: Generate GHZ state
        if num_qubits == 1:
            # For N=1: just |+⟩ state (Hadamard)
            qc.h(0)
        else:
            # For N≥2: |00...0⟩ + |11...1⟩
            qc.h(0)  # Hadamard on first qubit
            for i in range(1, num_qubits):
                qc.cx(0, i)  # CNOT chain from first qubit

        # Step 2: Apply delay using identity gates
        if delay_us > 0:
            num_identity_gates = int(delay_us * 1000 / identity_gate_time_ns)
            for qubit in range(num_qubits):
                for _ in range(num_identity_gates):
                    qc.id(qubit)

        # Step 3: Apply Y-rotation U(φ) to each qubit
        # U(φ) = exp(-iφσy/2) rotation around Y-axis
        if phi != 0:
            for qubit in range(num_qubits):
                qc.ry(phi, qubit)  # Y-rotation by angle phi

        # Step 4: Measurement
        qc.measure_all()

        return qc

    @staticmethod
    def create_parity_scan_circuits(
        num_qubits: int, delay_us: float = 0.0, phase_points: int | None = None
    ) -> list[Any]:
        """
        Create a set of circuits for parity oscillation scan

        Args:
            num_qubits: Number of qubits in GHZ state
            delay_us: Delay time in microseconds
            phase_points: Number of phase points (default: 4N+1 as in paper)

        Returns:
            List of circuits with different phase values
        """
        if phase_points is None:
            phase_points = 4 * num_qubits + 1

        phase_values = np.linspace(0, np.pi, phase_points)
        circuits = []

        for phi in phase_values:
            circuit = ParityCircuitFactory.create_ghz_with_delay_rotation(
                num_qubits, delay_us, phi
            )
            circuits.append(circuit)

        return circuits

    @staticmethod
    def create_coherence_decay_circuits(
        num_qubits_list: list[int],
        delays_us: list[float],
        phase_points: int | None = None,
    ) -> tuple[list[Any], list[dict]]:
        """
        Create complete set of circuits for coherence decay study

        Args:
            num_qubits_list: List of qubit counts to study
            delays_us: List of delay times in microseconds
            phase_points: Number of phase points per scan

        Returns:
            Tuple of (circuits list, metadata list)
        """
        circuits = []
        metadata = []

        for num_qubits in num_qubits_list:
            actual_phase_points = phase_points or (4 * num_qubits + 1)
            phase_values = np.linspace(0, np.pi, actual_phase_points)

            for delay_us in delays_us:
                for phi in phase_values:
                    circuit = ParityCircuitFactory.create_ghz_with_delay_rotation(
                        num_qubits, delay_us, phi
                    )
                    circuits.append(circuit)

                    metadata.append(
                        {
                            "num_qubits": num_qubits,
                            "delay_us": delay_us,
                            "phi": phi,
                            "circuit_index": len(circuits) - 1,
                        }
                    )

        return circuits, metadata


# Convenience functions
def create_ghz_with_delay_rotation(
    num_qubits: int, delay_us: float = 0.0, phi: float = 0.0
) -> Any:
    """
    Convenience function for creating GHZ circuit with delay and rotation
    """
    return ParityCircuitFactory.create_ghz_with_delay_rotation(
        num_qubits, delay_us, phi
    )


def create_parity_scan_circuits(
    num_qubits: int, delay_us: float = 0.0, phase_points: int | None = None
) -> list[Any]:
    """
    Convenience function for creating parity oscillation scan circuits
    """
    return ParityCircuitFactory.create_parity_scan_circuits(
        num_qubits, delay_us, phase_points
    )


def create_coherence_decay_circuits(
    num_qubits_list: list[int], delays_us: list[float], phase_points: int | None = None
) -> tuple[list[Any], list[dict]]:
    """
    Convenience function for creating coherence decay study circuits
    """
    return ParityCircuitFactory.create_coherence_decay_circuits(
        num_qubits_list, delays_us, phase_points
    )
