#!/usr/bin/env python3
"""
CHSH Experiment Class - Simplified implementation for CHSH Bell inequality violation experiments
Inherits from BaseExperiment and provides streamlined implementation for CHSH experiments
"""

from typing import Any

import numpy as np

from ..circuit.chsh_circuits import create_chsh_circuit
from ..core.base_experiment import BaseExperiment


class CHSHExperiment(BaseExperiment):
    """
    CHSH Bell inequality violation experiment class

    Simplified implementation focusing on core functionality:
    - CHSH circuit generation via classmethod
    - 4-measurement result processing
    - Bell inequality analysis
    """

    def __init__(self, experiment_name: str = None, **kwargs):
        # Extract CHSH experiment-specific parameters (not passed to BaseExperiment)
        chsh_specific_params = {
            "phase_points",
            "theta_a",
            "theta_b",
            "points",
            "angles",
        }

        # CLI parameters that should not be passed to BaseExperiment
        cli_only_params = {
            "shots",
            "backend",
            "devices",
            "parallel",
            "no_save",
            "no_plot",
            "show_plot",
            "verbose",
        }

        # Filter out parameters that shouldn't go to BaseExperiment
        base_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in chsh_specific_params and k not in cli_only_params
        }

        super().__init__(experiment_name or "chsh_experiment", **base_kwargs)

        # Store CHSH-specific parameters
        self.chsh_params = {
            k: v for k, v in kwargs.items() if k in chsh_specific_params
        }

    @classmethod
    def create_chsh_circuits(
        cls,
        phase_points: int = 20,
        theta_a: float = 0.0,
        theta_b: float = np.pi / 4,
        angles: list[tuple[float, float]] = None,
        qubit_list: list[int] = None,
        basis_gates: list[str] = None,
        optimization_level: int = 1,
    ) -> tuple[list[Any], dict]:
        """
        Create CHSH experiment circuits using functional approach

        Args:
            phase_points: Number of phase points for scan
            theta_a: Alice measurement angle
            theta_b: Bob measurement angle
            angles: Custom angle list (optional)
            qubit_list: Target qubits
            basis_gates: Transpilation basis gates
            optimization_level: Transpilation optimization level

        Returns:
            Tuple of (circuits_list, metadata_dict)
        """
        if qubit_list is None:
            qubit_list = [0, 1]

        if angles is None:
            angles = [
                (theta_a, theta_b),
                (theta_a, theta_b + np.pi / 2),
                (theta_a + np.pi / 2, theta_b),
                (theta_a + np.pi / 2, theta_b + np.pi / 2),
            ]

        phase_range = np.linspace(0, 2 * np.pi, phase_points)

        circuits = []
        circuit_metadata = []
        measurements = []

        for i, (angle_a, angle_b) in enumerate(angles):
            measurement_label = f"measurement_{i+1}"
            measurements.append((angle_a, angle_b))

            for j, phase in enumerate(phase_range):
                circuit = create_chsh_circuit(
                    theta_a=angle_a,
                    theta_b=angle_b,
                    phase=phase,
                    qubit_list=qubit_list,
                    basis_gates=basis_gates,
                    optimization_level=optimization_level,
                )
                circuits.append(circuit)

                circuit_metadata.append(
                    {
                        "measurement": measurement_label,
                        "phase": phase,
                        "theta_a": angle_a,
                        "theta_b": angle_b,
                        "circuit_index": len(circuits) - 1,
                    }
                )

        metadata = {
            "circuit_metadata": circuit_metadata,
            "phase_range": phase_range,
            "angles": angles,
            "measurements": measurements,
        }

        print(
            f"Created {len(circuits)} CHSH circuits ({len(angles)} measurements × {phase_points} phases)"
        )
        print(
            "CHSH circuit structure: |Φ⁺⟩ → A(θₐ), B(θᵦ) → measure (expected: Bell inequality violation with S-value)"
        )

        return circuits, metadata

    def _process_4_measurement_results(
        self,
        results: dict[str, list[dict]],
        circuit_metadata: list[dict],
        phase_range: np.ndarray,
        measurements: list[tuple],
        device_list: list[str],
    ) -> dict[str, dict]:
        """
        Process 4-measurement CHSH results by device

        Args:
            results: Raw measurement results per device
            circuit_metadata: Circuit metadata from create_chsh_circuits
            phase_range: Phase values array
            measurements: Measurement angle tuples
            device_list: List of devices

        Returns:
            Processed results per device
        """
        processed_results = {}

        for device in device_list:
            if device not in results:
                continue

            device_results = results[device]
            phase_points = len(phase_range)

            # Initialize measurement arrays
            measurement_data = {}
            for i, (theta_a, theta_b) in enumerate(measurements):
                measurement_label = f"measurement_{i+1}"
                measurement_data[measurement_label] = {
                    "theta_a": theta_a,
                    "theta_b": theta_b,
                    "phases": [],
                    "expectation_values": [],
                }

            # Process results by measurement
            for circuit_idx, result in enumerate(device_results):
                metadata = circuit_metadata[circuit_idx]
                measurement_label = metadata["measurement"]
                phase = metadata["phase"]

                # Convert counts if needed
                if "counts" in result:
                    counts = result["counts"]
                    if isinstance(list(counts.keys())[0], int):
                        counts = self._convert_decimal_to_binary_counts_chsh(counts)

                    expectation_value = (
                        self._calculate_expectation_value_oqtopus_compatible(counts)
                    )

                    measurement_data[measurement_label]["phases"].append(phase)
                    measurement_data[measurement_label]["expectation_values"].append(
                        expectation_value
                    )

            # Sort by phase for each measurement
            for measurement_label in measurement_data:
                phases = np.array(measurement_data[measurement_label]["phases"])
                expectation_values = np.array(
                    measurement_data[measurement_label]["expectation_values"]
                )

                sorted_indices = np.argsort(phases)
                measurement_data[measurement_label]["phases"] = phases[sorted_indices]
                measurement_data[measurement_label]["expectation_values"] = (
                    expectation_values[sorted_indices]
                )

            processed_results[device] = measurement_data

        return processed_results

    def _calculate_expectation_value_oqtopus_compatible(self, counts: dict) -> float:
        """
        Calculate expectation value ⟨ZZ⟩ for CHSH compatible with OQTOPUS format

        Args:
            counts: Measurement counts {"00": n1, "01": n2, "10": n3, "11": n4}

        Returns:
            Expectation value ⟨ZZ⟩
        """
        total_shots = sum(counts.values())
        if total_shots == 0:
            return 0.0

        # Calculate ⟨ZZ⟩ = P(00) + P(11) - P(01) - P(10)
        p_00 = counts.get("00", 0) / total_shots
        p_01 = counts.get("01", 0) / total_shots
        p_10 = counts.get("10", 0) / total_shots
        p_11 = counts.get("11", 0) / total_shots

        expectation_value = p_00 + p_11 - p_01 - p_10
        return expectation_value

    def _convert_decimal_to_binary_counts_chsh(
        self, decimal_counts: dict[int, int]
    ) -> dict[str, int]:
        """
        Convert decimal counts to binary string format for CHSH (2-qubit)

        Args:
            decimal_counts: {0: n1, 1: n2, 2: n3, 3: n4}

        Returns:
            Binary counts: {"00": n1, "01": n2, "10": n3, "11": n4}
        """
        binary_counts = {}

        for decimal_state, count in decimal_counts.items():
            # Convert to 2-bit binary string (for 2-qubit CHSH)
            binary_state = format(decimal_state, "02b")
            binary_counts[binary_state] = count

        return binary_counts

    def _create_4_measurement_analysis(
        self,
        phase_range: np.ndarray,
        processed_results: dict[str, dict],
        angles: list[tuple[float, float]],
    ) -> dict[str, Any]:
        """
        Create CHSH Bell inequality analysis from 4-measurement results

        Args:
            phase_range: Phase values array
            processed_results: Processed measurement results per device
            angles: Measurement angle tuples

        Returns:
            CHSH analysis with S-parameter calculation
        """
        analysis = {
            "bell_parameter_S": {},
            "max_violation": {},
            "angles": angles,
            "phase_range": phase_range.tolist(),
        }

        for device, device_data in processed_results.items():
            measurements = list(device_data.keys())

            if len(measurements) != 4:
                print(
                    f"⚠️ Expected 4 measurements for CHSH, got {len(measurements)} for {device}"
                )
                continue

            # Extract expectation values for each measurement
            e_vals = {}
            for measurement in measurements:
                e_vals[measurement] = np.array(
                    device_data[measurement]["expectation_values"]
                )

            # Calculate CHSH S-parameter: S = |E₁ + E₂| + |E₃ - E₄|
            # where E₁, E₂, E₃, E₄ are the 4 measurement expectation values
            s_values = []

            for i in range(len(phase_range)):
                e1 = (
                    e_vals["measurement_1"][i]
                    if i < len(e_vals["measurement_1"])
                    else 0
                )
                e2 = (
                    e_vals["measurement_2"][i]
                    if i < len(e_vals["measurement_2"])
                    else 0
                )
                e3 = (
                    e_vals["measurement_3"][i]
                    if i < len(e_vals["measurement_3"])
                    else 0
                )
                e4 = (
                    e_vals["measurement_4"][i]
                    if i < len(e_vals["measurement_4"])
                    else 0
                )

                s_value = abs(e1 + e2) + abs(e3 - e4)
                s_values.append(s_value)

            s_values = np.array(s_values)
            max_s = np.max(s_values)
            max_phase = phase_range[np.argmax(s_values)]

            analysis["bell_parameter_S"][device] = s_values.tolist()
            analysis["max_violation"][device] = {
                "max_S": float(max_s),
                "max_phase": float(max_phase),
                "classical_limit": 2.0,
                "quantum_limit": 2.828,  # 2√2
                "violation": max_s > 2.0,
                "violation_strength": float(max_s - 2.0) if max_s > 2.0 else 0.0,
            }

            print(
                f"📊 {device}: Max S = {max_s:.3f} (violation: {'✓' if max_s > 2.0 else '✗'})"
            )

        return analysis
