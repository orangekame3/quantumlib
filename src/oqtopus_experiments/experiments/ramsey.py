#!/usr/bin/env python3
"""
Ramsey Experiment Class - Simplified Ramsey oscillation experiment implementation
Inherits from BaseExperiment and provides streamlined Ramsey experiment functionality
"""

from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from scipy.optimize import curve_fit

from ...core.base_experiment import BaseExperiment


class RamseyExperiment(BaseExperiment):
    """
    Ramsey oscillation experiment class
    
    Simplified implementation focusing on core functionality:
    - Ramsey circuit generation via classmethod
    - Oscillation analysis with frequency and T2* fitting
    - Detuning frequency estimation
    """

    def __init__(
        self, experiment_name: str = None, disable_mitigation: bool = False, **kwargs
    ):
        # Extract Ramsey experiment-specific parameters (not passed to BaseExperiment)
        ramsey_specific_params = {
            "delay_points",
            "max_delay",
            "delay_times",
            "detuning",
            "disable_mitigation",
        }

        # Filter kwargs to pass to BaseExperiment
        base_kwargs = {k: v for k, v in kwargs.items() if k not in ramsey_specific_params}

        super().__init__(experiment_name or "ramsey_experiment", **base_kwargs)

        # Ramsey experiment-specific settings
        self.expected_t2_star = 1000  # Initial estimate [ns] for fitting
        self.disable_mitigation = disable_mitigation

    @classmethod
    def create_ramsey_circuits(
        cls,
        delay_points: int = 20,
        max_delay: float = 50000.0,
        detuning: float = 0.0,
        qubit: int = 0,
        basis_gates: list[str] = None,
        optimization_level: int = 1,
    ) -> tuple[list[Any], dict]:
        """
        Create Ramsey oscillation experiment circuits using functional approach
        
        Args:
            delay_points: Number of delay time points
            max_delay: Maximum delay time in nanoseconds
            detuning: Detuning frequency in MHz
            qubit: Target qubit for Ramsey measurement
            basis_gates: Transpilation basis gates
            optimization_level: Transpilation optimization level
            
        Returns:
            Tuple of (circuits_list, metadata_dict)
        """
        # Generate delay times (linear spacing for Ramsey oscillations)
        delay_times = np.linspace(0, max_delay, delay_points)

        circuits = []

        for delay in delay_times:
            # Create Ramsey sequence: X/2 - delay - X/2 - measure
            qc = QuantumCircuit(1, 1)
            qc.rx(np.pi/2, 0)  # First π/2 pulse (create superposition)

            if delay > 0:
                qc.delay(delay, 0, unit='ns')  # Free evolution with detuning

            # Apply detuning phase if specified
            if detuning != 0:
                # Phase accumulation during delay: φ = 2π * f_detuning * t
                detuning_phase = 2 * np.pi * detuning * 1e6 * delay * 1e-9  # MHz * ns
                qc.rz(detuning_phase, 0)

            qc.rx(np.pi/2, 0)  # Second π/2 pulse (analysis pulse)
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
            "detuning": detuning,
            "qubit": qubit,
        }

        print(f"Created {len(circuits)} Ramsey circuits (delay range: {delay_times[0]:.1f} - {delay_times[-1]:.1f} ns)")
        print(f"Ramsey circuit structure: |0⟩ → RX(π/2) → delay(t) → RX(π/2) → measure (detuning: {detuning} MHz)")

        return circuits, metadata

    def analyze_results(
        self, results: dict[str, list[dict[str, Any]]], **kwargs
    ) -> dict[str, Any]:
        """
        Analyze Ramsey experiment results with oscillation fitting
        
        Args:
            results: Raw measurement results per device
            
        Returns:
            Ramsey analysis results with fitted frequencies and T2*
        """
        if not results:
            return {"error": "No results to analyze"}

        # Get delay times from experiment metadata
        delay_times = np.array(self.experiment_params["delay_times"])
        detuning = self.experiment_params.get("detuning", 0.0)

        analysis = {
            "delay_times": delay_times.tolist(),
            "detuning": detuning,
            "ramsey_estimates": {},
            "fit_quality": {},
            "expectation_values": {},
        }

        for device, device_results in results.items():
            if not device_results:
                continue

            # Extract expectation values (probability of measuring |0⟩ for Ramsey)
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

            # Fit Ramsey oscillation: P(t) = A * exp(-t/T2*) * cos(2πft + φ) + B
            try:
                # Initial parameter estimates
                initial_amplitude = (np.max(expectation_values) - np.min(expectation_values)) / 2
                initial_frequency = self._estimate_frequency(delay_times, expectation_values)
                initial_t2_star = self.expected_t2_star
                initial_phase = 0.0
                initial_offset = np.mean(expectation_values)

                def ramsey_oscillation(t, amplitude, frequency, t2_star, phase, offset):
                    # Convert frequency from MHz to Hz, time from ns to s
                    omega = 2 * np.pi * frequency * 1e6  # rad/s
                    t_sec = t * 1e-9  # ns to s
                    return amplitude * np.exp(-t / t2_star) * np.cos(omega * t_sec + phase) + offset

                # Perform curve fitting
                popt, pcov = curve_fit(
                    ramsey_oscillation,
                    delay_times,
                    expectation_values,
                    p0=[initial_amplitude, initial_frequency, initial_t2_star, initial_phase, initial_offset],
                    bounds=(
                        [0, -50, 10, -2*np.pi, 0],  # Lower bounds
                        [1, 50, 1e6, 2*np.pi, 1]   # Upper bounds
                    ),
                    maxfev=10000
                )

                fitted_amplitude, fitted_frequency, fitted_t2_star, fitted_phase, fitted_offset = popt

                # Calculate R-squared for fit quality
                fitted_values = ramsey_oscillation(delay_times, *popt)
                ss_res = np.sum((expectation_values - fitted_values) ** 2)
                ss_tot = np.sum((expectation_values - np.mean(expectation_values)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                # Calculate parameter uncertainties
                param_errors = np.sqrt(np.diag(pcov))

                analysis["ramsey_estimates"][device] = {
                    "frequency_mhz": float(fitted_frequency),
                    "t2_star_ns": float(fitted_t2_star),
                    "t2_star_us": float(fitted_t2_star / 1000),
                    "amplitude": float(fitted_amplitude),
                    "phase_rad": float(fitted_phase),
                    "offset": float(fitted_offset),
                    "frequency_error_mhz": float(param_errors[1]),
                    "t2_star_error_ns": float(param_errors[2]),
                    "fit_parameters": popt.tolist(),
                    "detuning_mhz": detuning,
                }

                analysis["fit_quality"][device] = {
                    "r_squared": float(r_squared),
                    "rmse": float(np.sqrt(ss_res / len(expectation_values))),
                }

                print(f"📊 {device}: f = {fitted_frequency:.3f} ± {param_errors[1]:.3f} MHz, T₂* = {fitted_t2_star:.1f} ± {param_errors[2]:.1f} ns ({fitted_t2_star/1000:.2f} μs), R² = {r_squared:.3f}")

            except Exception as e:
                print(f"❌ {device}: Ramsey fitting failed - {str(e)}")
                analysis["ramsey_estimates"][device] = {"error": str(e)}
                analysis["fit_quality"][device] = {"error": str(e)}

            analysis["expectation_values"][device] = expectation_values.tolist()

        return analysis

    def _estimate_frequency(self, delay_times: np.ndarray, expectation_values: np.ndarray) -> float:
        """
        Estimate oscillation frequency using FFT for initial fitting guess
        
        Args:
            delay_times: Time points in nanoseconds
            expectation_values: Measured probability values
            
        Returns:
            Estimated frequency in MHz
        """
        try:
            # Remove DC component
            signal = expectation_values - np.mean(expectation_values)

            # Calculate sampling rate (convert ns to s)
            dt = (delay_times[1] - delay_times[0]) * 1e-9  # ns to s

            # Perform FFT
            fft = np.fft.fft(signal)
            freqs = np.fft.fftfreq(len(signal), dt)

            # Find peak frequency (positive frequencies only)
            positive_freqs = freqs[:len(freqs)//2]
            positive_fft = np.abs(fft[:len(fft)//2])

            if len(positive_fft) > 1:
                peak_idx = np.argmax(positive_fft[1:]) + 1  # Skip DC component
                estimated_freq = abs(positive_freqs[peak_idx]) / 1e6  # Hz to MHz
                return estimated_freq
            else:
                return 1.0  # Default 1 MHz

        except Exception:
            return 1.0  # Default 1 MHz if estimation fails
