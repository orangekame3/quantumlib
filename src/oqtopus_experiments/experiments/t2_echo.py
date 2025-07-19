#!/usr/bin/env python3
"""
T2 Echo Experiment Class - Simplified T2 echo experiment implementation (Hahn Echo/CPMG)
Inherits from BaseExperiment and provides streamlined T2 echo experiment functionality
"""

from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from scipy.optimize import curve_fit

from ...core.base_experiment import BaseExperiment


class T2EchoExperiment(BaseExperiment):
    """
    T2 Echo experiment class (Hahn Echo and CPMG sequences)
    
    Simplified implementation focusing on core functionality:
    - T2 echo circuit generation via classmethod
    - Exponential decay analysis with echo refocusing
    - Support for Hahn Echo and CPMG sequences
    """

    def __init__(
        self, experiment_name: str = None, disable_mitigation: bool = False, **kwargs
    ):
        # Extract T2 echo experiment-specific parameters (not passed to BaseExperiment)
        t2_echo_specific_params = {
            "delay_points",
            "max_delay",
            "delay_times",
            "echo_type",
            "num_echoes",
            "disable_mitigation",
        }

        # Filter kwargs to pass to BaseExperiment
        base_kwargs = {k: v for k, v in kwargs.items() if k not in t2_echo_specific_params}

        super().__init__(experiment_name or "t2_echo_experiment", **base_kwargs)

        # T2 echo experiment-specific settings
        self.expected_t2_echo = 2000  # Initial estimate [ns] for fitting
        self.disable_mitigation = disable_mitigation

    @classmethod
    def create_t2_echo_circuits(
        cls,
        delay_points: int = 20,
        max_delay: float = 100000.0,
        echo_type: str = "hahn",
        num_echoes: int = 1,
        qubit: int = 0,
        basis_gates: list[str] = None,
        optimization_level: int = 1,
    ) -> tuple[list[Any], dict]:
        """
        Create T2 echo experiment circuits using functional approach
        
        Args:
            delay_points: Number of delay time points
            max_delay: Maximum total delay time in nanoseconds
            echo_type: Type of echo sequence ("hahn" or "cpmg")
            num_echoes: Number of echo pulses (for CPMG)
            qubit: Target qubit for T2 echo measurement
            basis_gates: Transpilation basis gates
            optimization_level: Transpilation optimization level
            
        Returns:
            Tuple of (circuits_list, metadata_dict)
        """
        # Generate delay times (logarithmic spacing for better T2 characterization)
        delay_times = np.logspace(
            np.log10(1.0),  # Start from 1 ns
            np.log10(max_delay),
            delay_points
        )

        circuits = []

        for total_delay in delay_times:
            qc = QuantumCircuit(1, 1)
            qc.rx(np.pi/2, 0)  # Initial π/2 pulse to create superposition

            if echo_type.lower() == "hahn":
                # Hahn Echo: π/2 - τ/2 - π - τ/2 - π/2
                half_delay = total_delay / 2
                qc.delay(half_delay, 0, unit='ns')
                qc.rx(np.pi, 0)  # π pulse (echo pulse)
                qc.delay(half_delay, 0, unit='ns')

            elif echo_type.lower() == "cpmg":
                # CPMG: π/2 - [τ/(2n) - π - τ/n - π - ... - τ/(2n)] - π/2
                # where n is num_echoes
                inter_pulse_delay = total_delay / (2 * num_echoes)

                # First half delay
                qc.delay(inter_pulse_delay, 0, unit='ns')

                # Echo pulse sequence
                for i in range(num_echoes):
                    qc.rx(np.pi, 0)  # π pulse
                    if i < num_echoes - 1:
                        # Full delay between echoes
                        qc.delay(2 * inter_pulse_delay, 0, unit='ns')
                    else:
                        # Last half delay
                        qc.delay(inter_pulse_delay, 0, unit='ns')

            else:
                raise ValueError(f"Unsupported echo type: {echo_type}")

            qc.rx(np.pi/2, 0)  # Final π/2 pulse for readout
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
            "echo_type": echo_type,
            "num_echoes": num_echoes,
            "qubit": qubit,
        }

        print(f"Created {len(circuits)} T2 Echo circuits ({echo_type.upper()}, n={num_echoes})")
        print(f"Delay range: {delay_times[0]:.1f} - {delay_times[-1]:.1f} ns")
        print("T2 Echo structure: |0⟩ → RX(π/2) → echo_sequence → RX(π/2) → measure")

        return circuits, metadata

    def analyze_results(
        self, results: dict[str, list[dict[str, Any]]], **kwargs
    ) -> dict[str, Any]:
        """
        Analyze T2 echo experiment results with exponential decay fitting
        
        Args:
            results: Raw measurement results per device
            
        Returns:
            T2 echo analysis results with fitted decay constants
        """
        if not results:
            return {"error": "No results to analyze"}

        # Get metadata from experiment parameters
        delay_times = np.array(self.experiment_params["delay_times"])
        echo_type = self.experiment_params.get("echo_type", "hahn")
        num_echoes = self.experiment_params.get("num_echoes", 1)

        analysis = {
            "delay_times": delay_times.tolist(),
            "echo_type": echo_type,
            "num_echoes": num_echoes,
            "t2_echo_estimates": {},
            "fit_quality": {},
            "expectation_values": {},
        }

        for device, device_results in results.items():
            if not device_results:
                continue

            # Extract expectation values (probability of measuring |0⟩ for echo)
            expectation_values = []

            for result in device_results:
                if "counts" in result:
                    counts = result["counts"]
                    total_shots = sum(counts.values())

                    if total_shots > 0:
                        # Calculate P(|0⟩) = proportion of '0' measurements
                        prob_0 = counts.get('0', counts.get(0, 0)) / total_shots
                        expectation_values.append(prob_0)
                    else:
                        expectation_values.append(0.5)
                else:
                    expectation_values.append(0.5)

            expectation_values = np.array(expectation_values)

            # Fit exponential decay: P(t) = A * exp(-t/T2_echo) + B
            try:
                # Initial parameter estimates
                initial_amplitude = np.max(expectation_values) - np.min(expectation_values)
                initial_t2_echo = self.expected_t2_echo
                initial_offset = np.min(expectation_values)

                def exponential_decay(t, amplitude, t2_echo, offset):
                    return amplitude * np.exp(-t / t2_echo) + offset

                # Perform curve fitting
                popt, pcov = curve_fit(
                    exponential_decay,
                    delay_times,
                    expectation_values,
                    p0=[initial_amplitude, initial_t2_echo, initial_offset],
                    bounds=([0, 1, -0.1], [2, 1e6, 1.1]),  # Reasonable bounds
                    maxfev=5000
                )

                fitted_amplitude, fitted_t2_echo, fitted_offset = popt

                # Calculate R-squared for fit quality
                fitted_values = exponential_decay(delay_times, *popt)
                ss_res = np.sum((expectation_values - fitted_values) ** 2)
                ss_tot = np.sum((expectation_values - np.mean(expectation_values)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                # Calculate parameter uncertainties
                param_errors = np.sqrt(np.diag(pcov))

                analysis["t2_echo_estimates"][device] = {
                    "t2_echo_ns": float(fitted_t2_echo),
                    "t2_echo_us": float(fitted_t2_echo / 1000),
                    "amplitude": float(fitted_amplitude),
                    "offset": float(fitted_offset),
                    "t2_echo_error_ns": float(param_errors[1]),
                    "fit_parameters": popt.tolist(),
                    "echo_type": echo_type,
                    "num_echoes": num_echoes,
                }

                analysis["fit_quality"][device] = {
                    "r_squared": float(r_squared),
                    "rmse": float(np.sqrt(ss_res / len(expectation_values))),
                }

                echo_label = f"T₂({echo_type.upper()})"
                if echo_type.lower() == "cpmg":
                    echo_label += f"(n={num_echoes})"

                print(f"📊 {device}: {echo_label} = {fitted_t2_echo:.1f} ± {param_errors[1]:.1f} ns ({fitted_t2_echo/1000:.2f} μs), R² = {r_squared:.3f}")

            except Exception as e:
                print(f"❌ {device}: T2 Echo fitting failed - {str(e)}")
                analysis["t2_echo_estimates"][device] = {"error": str(e)}
                analysis["fit_quality"][device] = {"error": str(e)}

            analysis["expectation_values"][device] = expectation_values.tolist()

        return analysis
