#!/usr/bin/env python3
"""
T1 Experiment Class - Simplified T1 decay experiment implementation
Inherits from BaseExperiment and provides streamlined T1 experiment functionality
"""

from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from scipy.optimize import curve_fit

from ...core.base_experiment import BaseExperiment


class T1Experiment(BaseExperiment):
    """
    T1 decay experiment class
    
    Simplified implementation focusing on core functionality:
    - T1 circuit generation via classmethod
    - Exponential decay analysis
    - T1 time constant estimation
    """

    def __init__(
        self, experiment_name: str = None, disable_mitigation: bool = False, **kwargs
    ):
        # Extract T1 experiment-specific parameters (not passed to BaseExperiment)
        t1_specific_params = {
            "delay_points",
            "max_delay",
            "delay_times",
            "disable_mitigation",
        }

        # Filter kwargs to pass to BaseExperiment
        base_kwargs = {k: v for k, v in kwargs.items() if k not in t1_specific_params}

        super().__init__(experiment_name or "t1_experiment", **base_kwargs)

        # T1 experiment-specific settings
        self.expected_t1 = 1000  # Initial estimate [ns] for fitting
        self.disable_mitigation = disable_mitigation

    @classmethod
    def create_t1_circuits(
        cls,
        delay_points: int = 20,
        max_delay: float = 50000.0,
        qubit: int = 0,
        basis_gates: list[str] = None,
        optimization_level: int = 1,
    ) -> tuple[list[Any], dict]:
        """
        Create T1 decay experiment circuits using functional approach
        
        Args:
            delay_points: Number of delay time points
            max_delay: Maximum delay time in nanoseconds
            qubit: Target qubit for T1 measurement
            basis_gates: Transpilation basis gates
            optimization_level: Transpilation optimization level
            
        Returns:
            Tuple of (circuits_list, metadata_dict)
        """
        # Generate delay times (logarithmic spacing for better T1 characterization)
        delay_times = np.logspace(
            np.log10(1.0),  # Start from 1 ns
            np.log10(max_delay),
            delay_points
        )

        circuits = []

        for delay in delay_times:
            # Create X-gate preparation + delay + measurement circuit
            qc = QuantumCircuit(1, 1)
            qc.x(0)  # Prepare |1⟩ state
            qc.delay(delay, 0, unit='ns')  # Wait for decay
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
            "delay_times": delay_times,
            "max_delay": max_delay,
            "delay_points": delay_points,
            "qubit": qubit,
        }

        print(f"Created {len(circuits)} T1 circuits (delay range: {delay_times[0]:.1f} - {delay_times[-1]:.1f} ns)")
        print("T1 circuit structure: |0⟩ → X → delay(t) → measure (expected: exponential decay)")

        return circuits, metadata

    def analyze_results(
        self, results: dict[str, list[dict[str, Any]]], **kwargs
    ) -> dict[str, Any]:
        """
        Analyze T1 experiment results with exponential decay fitting
        
        Args:
            results: Raw measurement results per device
            
        Returns:
            T1 analysis results with fitted decay constants
        """
        if not results:
            return {"error": "No results to analyze"}

        # Get delay times from experiment metadata
        delay_times = np.array(self.experiment_params["delay_times"])

        analysis = {
            "delay_times": delay_times.tolist(),
            "t1_estimates": {},
            "fit_quality": {},
            "expectation_values": {},
        }

        for device, device_results in results.items():
            if not device_results:
                continue

            # Extract expectation values (probability of measuring |1⟩)
            expectation_values = []

            for result in device_results:
                if "counts" in result:
                    counts = result["counts"]
                    total_shots = sum(counts.values())

                    if total_shots > 0:
                        # Calculate P(|1⟩) = proportion of '1' measurements
                        prob_1 = counts.get('1', counts.get(1, 0)) / total_shots
                        expectation_values.append(prob_1)
                    else:
                        expectation_values.append(0.0)
                else:
                    expectation_values.append(0.0)

            expectation_values = np.array(expectation_values)

            # Fit exponential decay: P(t) = A * exp(-t/T1) + B
            try:
                # Initial parameter estimates
                initial_amplitude = np.max(expectation_values) - np.min(expectation_values)
                initial_t1 = self.expected_t1
                initial_offset = np.min(expectation_values)

                def exponential_decay(t, amplitude, t1, offset):
                    return amplitude * np.exp(-t / t1) + offset

                # Perform curve fitting
                popt, pcov = curve_fit(
                    exponential_decay,
                    delay_times,
                    expectation_values,
                    p0=[initial_amplitude, initial_t1, initial_offset],
                    bounds=([0, 1, -0.1], [2, 1e6, 1.1]),  # Reasonable bounds
                    maxfev=5000
                )

                fitted_amplitude, fitted_t1, fitted_offset = popt

                # Calculate R-squared for fit quality
                fitted_values = exponential_decay(delay_times, *popt)
                ss_res = np.sum((expectation_values - fitted_values) ** 2)
                ss_tot = np.sum((expectation_values - np.mean(expectation_values)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                # Calculate parameter uncertainties
                param_errors = np.sqrt(np.diag(pcov))

                analysis["t1_estimates"][device] = {
                    "t1_ns": float(fitted_t1),
                    "t1_us": float(fitted_t1 / 1000),
                    "amplitude": float(fitted_amplitude),
                    "offset": float(fitted_offset),
                    "t1_error_ns": float(param_errors[1]),
                    "fit_parameters": popt.tolist(),
                }

                analysis["fit_quality"][device] = {
                    "r_squared": float(r_squared),
                    "rmse": float(np.sqrt(ss_res / len(expectation_values))),
                }

                print(f"📊 {device}: T₁ = {fitted_t1:.1f} ± {param_errors[1]:.1f} ns ({fitted_t1/1000:.2f} μs), R² = {r_squared:.3f}")

            except Exception as e:
                print(f"❌ {device}: T1 fitting failed - {str(e)}")
                analysis["t1_estimates"][device] = {"error": str(e)}
                analysis["fit_quality"][device] = {"error": str(e)}

            analysis["expectation_values"][device] = expectation_values.tolist()

        return analysis
