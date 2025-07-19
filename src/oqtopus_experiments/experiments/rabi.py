#!/usr/bin/env python3
"""
Rabi Experiment Class - Simplified Rabi oscillation experiment implementation
Inherits from BaseExperiment and provides streamlined Rabi experiment functionality
"""

from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from scipy.optimize import curve_fit

from ..core.base_experiment import BaseExperiment


class RabiExperiment(BaseExperiment):
    """
    Rabi oscillation experiment class

    Simplified implementation focusing on core functionality:
    - Rabi circuit generation via classmethod
    - Oscillation analysis with amplitude and frequency fitting
    - Drive amplitude calibration
    """

    def __init__(
        self, experiment_name: str = None, disable_mitigation: bool = False, **kwargs
    ):
        # Extract Rabi experiment-specific parameters (not passed to BaseExperiment)
        rabi_specific_params = {
            "amplitude_points",
            "max_amplitude",
            "amplitudes",
            "disable_mitigation",
        }

        # Filter kwargs to pass to BaseExperiment
        base_kwargs = {k: v for k, v in kwargs.items() if k not in rabi_specific_params}

        super().__init__(experiment_name or "rabi_experiment", **base_kwargs)

        # Rabi experiment-specific settings
        self.expected_pi_amplitude = 0.5  # Initial estimate for π pulse amplitude
        self.disable_mitigation = disable_mitigation

    @classmethod
    def create_rabi_circuits(
        cls,
        amplitude_points: int = 20,
        max_amplitude: float = 1.0,
        amplitudes: list[float] = None,
        qubit: int = 0,
        basis_gates: list[str] = None,
        optimization_level: int = 1,
    ) -> tuple[list[Any], dict]:
        """
        Create Rabi oscillation experiment circuits using functional approach

        Args:
            amplitude_points: Number of amplitude points
            max_amplitude: Maximum drive amplitude
            amplitudes: Custom amplitude list (optional)
            qubit: Target qubit for Rabi measurement
            basis_gates: Transpilation basis gates
            optimization_level: Transpilation optimization level

        Returns:
            Tuple of (circuits_list, metadata_dict)
        """
        if amplitudes is None:
            # Generate amplitude values (linear spacing for Rabi oscillations)
            amplitudes = np.linspace(0, max_amplitude, amplitude_points)
        else:
            amplitudes = np.array(amplitudes)
            amplitude_points = len(amplitudes)

        circuits = []

        for amplitude in amplitudes:
            # Create Rabi circuit: |0⟩ → RX(amplitude * π) → measure
            qc = QuantumCircuit(1, 1)

            if amplitude > 0:
                # Apply rotation with specified amplitude
                # amplitude=1.0 corresponds to π pulse
                rotation_angle = amplitude * np.pi
                qc.rx(rotation_angle, 0)

            qc.measure(0, 0)  # Measure final state

            # Transpile if basis gates specified
            if basis_gates is not None:
                qc = transpile(
                    qc,
                    basis_gates=basis_gates,
                    optimization_level=optimization_level,
                )

            circuits.append(qc)

        metadata = {
            "amplitudes": amplitudes,
            "max_amplitude": max_amplitude,
            "amplitude_points": amplitude_points,
            "qubit": qubit,
        }

        print(
            f"Created {len(circuits)} Rabi circuits (amplitude range: {amplitudes[0]:.3f} - {amplitudes[-1]:.3f})"
        )
        print(
            "Rabi circuit structure: |0⟩ → RX(amp·π) → measure (expected: oscillation with amp)"
        )

        return circuits, metadata

    def analyze_results(
        self, results: dict[str, list[dict[str, Any]]], **kwargs
    ) -> dict[str, Any]:
        """
        Analyze Rabi experiment results with oscillation fitting

        Args:
            results: Raw measurement results per device

        Returns:
            Rabi analysis results with fitted oscillation parameters
        """
        if not results:
            return {"error": "No results to analyze"}

        # Get amplitudes from experiment metadata
        amplitudes = np.array(self.experiment_params["amplitudes"])

        analysis = {
            "amplitudes": amplitudes.tolist(),
            "rabi_estimates": {},
            "fit_quality": {},
            "expectation_values": {},
        }

        for device, device_results in results.items():
            if not device_results:
                continue

            # Extract expectation values (probability of measuring |1⟩ for Rabi)
            expectation_values = []

            for result in device_results:
                if "counts" in result:
                    counts = result["counts"]
                    total_shots = sum(counts.values())

                    if total_shots > 0:
                        # Calculate P(|1⟩) = proportion of '1' measurements
                        prob_1 = counts.get("1", counts.get(1, 0)) / total_shots
                        expectation_values.append(prob_1)
                    else:
                        expectation_values.append(0.0)
                else:
                    expectation_values.append(0.0)

            expectation_values = np.array(expectation_values)

            # Fit Rabi oscillation: P(amp) = A * sin²(π * amp * freq + φ) + B
            try:
                # Initial parameter estimates
                initial_amplitude = (
                    np.max(expectation_values) - np.min(expectation_values)
                ) / 2
                initial_frequency = self._estimate_rabi_frequency(
                    amplitudes, expectation_values
                )
                initial_phase = 0.0
                initial_offset = np.mean(expectation_values)

                def rabi_oscillation(amp, amplitude, frequency, phase, offset):
                    return (
                        amplitude * np.sin(np.pi * amp * frequency + phase) ** 2
                        + offset
                    )

                # Perform curve fitting
                popt, pcov = curve_fit(
                    rabi_oscillation,
                    amplitudes,
                    expectation_values,
                    p0=[
                        initial_amplitude,
                        initial_frequency,
                        initial_phase,
                        initial_offset,
                    ],
                    bounds=(
                        [0, 0.1, -2 * np.pi, 0],  # Lower bounds
                        [1, 10, 2 * np.pi, 1],  # Upper bounds
                    ),
                    maxfev=10000,
                )

                fitted_amplitude, fitted_frequency, fitted_phase, fitted_offset = popt

                # Calculate π pulse amplitude (when sin²(π * amp * freq + φ) = 1)
                # This occurs when π * amp * freq + φ = π/2 + n*π
                # For first π pulse: amp_pi = (π/2 - φ) / (π * freq) = (1/2 - φ/π) / freq
                pi_amplitude = (0.5 - fitted_phase / np.pi) / fitted_frequency
                if pi_amplitude < 0:
                    pi_amplitude += 1.0 / fitted_frequency  # Add one period

                # Calculate R-squared for fit quality
                fitted_values = rabi_oscillation(amplitudes, *popt)
                ss_res = np.sum((expectation_values - fitted_values) ** 2)
                ss_tot = np.sum((expectation_values - np.mean(expectation_values)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                # Calculate parameter uncertainties
                param_errors = np.sqrt(np.diag(pcov))

                analysis["rabi_estimates"][device] = {
                    "rabi_frequency": float(fitted_frequency),
                    "pi_amplitude": float(pi_amplitude),
                    "oscillation_amplitude": float(fitted_amplitude),
                    "phase_rad": float(fitted_phase),
                    "offset": float(fitted_offset),
                    "frequency_error": float(param_errors[1]),
                    "fit_parameters": popt.tolist(),
                }

                analysis["fit_quality"][device] = {
                    "r_squared": float(r_squared),
                    "rmse": float(np.sqrt(ss_res / len(expectation_values))),
                }

                print(
                    f"📊 {device}: π-pulse amp = {pi_amplitude:.3f}, freq = {fitted_frequency:.3f}, R² = {r_squared:.3f}"
                )

            except Exception as e:
                print(f"❌ {device}: Rabi fitting failed - {str(e)}")
                analysis["rabi_estimates"][device] = {"error": str(e)}
                analysis["fit_quality"][device] = {"error": str(e)}

            analysis["expectation_values"][device] = expectation_values.tolist()

        return analysis

    def _estimate_rabi_frequency(
        self, amplitudes: np.ndarray, expectation_values: np.ndarray
    ) -> float:
        """
        Estimate Rabi frequency using simple peak counting for initial fitting guess

        Args:
            amplitudes: Drive amplitude values
            expectation_values: Measured probability values

        Returns:
            Estimated Rabi frequency
        """
        try:
            # Count peaks in the data
            # A simple approach: count local maxima
            peaks = 0
            for i in range(1, len(expectation_values) - 1):
                if (
                    expectation_values[i] > expectation_values[i - 1]
                    and expectation_values[i] > expectation_values[i + 1]
                    and expectation_values[i] > np.mean(expectation_values)
                ):
                    peaks += 1

            if peaks > 0:
                # Estimate frequency from peak count and amplitude range
                amplitude_range = amplitudes[-1] - amplitudes[0]
                estimated_freq = peaks / amplitude_range
                return max(0.1, min(10.0, estimated_freq))  # Reasonable bounds
            else:
                return 1.0  # Default frequency

        except Exception:
            return 1.0  # Default frequency if estimation fails
