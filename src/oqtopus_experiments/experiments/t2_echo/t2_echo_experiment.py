#!/usr/bin/env python3
"""
T2 Echo Experiment Class - Specialized class for T2 Echo experiments
Inherits from BaseExperiment and provides T2 Echo experiment-specific implementation
"""

import time
from typing import Any

import numpy as np

from ...core.base_experiment import BaseExperiment
from ...core.parallel_execution import ParallelExecutionMixin


class T2EchoExperiment(BaseExperiment, ParallelExecutionMixin):
    """
    T2 Echo experiment class (Hahn Echo/CPMG)

    Specialized features:
    - Automatic T2 Echo circuit generation (Hahn Echo, CPMG)
    - Echo decay fitting
    - Delay time scan experiments
    - T2 time constant estimation
    """

    def __init__(
        self,
        experiment_name: str = None,
        enable_fitting: bool = True,
        echo_type: str = "hahn",
        num_echoes: int = 1,
        **kwargs,
    ):
        # Extract T2 Echo experiment-specific parameters (not passed to BaseExperiment)
        t2_echo_specific_params = {
            "delay_points",
            "max_delay",
            "delay_times",
            "enable_fitting",
            "echo_type",
            "num_echoes",
        }

        # Filter kwargs to pass to BaseExperiment
        base_kwargs = {
            k: v for k, v in kwargs.items() if k not in t2_echo_specific_params
        }

        super().__init__(experiment_name, **base_kwargs)

        # T2 Echo experiment-specific settings
        self.expected_t2 = 10000  # Initial estimate [ns] - T2 is longer than T2*
        self.enable_fitting = enable_fitting  # Fitting enable flag
        self.echo_type = echo_type  # "hahn" or "cpmg"
        self.num_echoes = num_echoes  # Number of echoes (for CPMG)

        # Enable readout mitigation for T2 Echo experiments
        self.mitigation_options = {"ro_error_mitigation": "pseudo_inverse"}
        self.mitigation_info = self.mitigation_options

        if enable_fitting:
            print(
                f"T2 Echo experiment: {echo_type.upper()} echo measurement with fitting enabled (echoes={num_echoes})"
            )
        else:
            print(
                f"T2 Echo experiment: {echo_type.upper()} echo measurement (fitting disabled, echoes={num_echoes})"
            )

    def create_circuits(self, **kwargs) -> list[Any]:
        """
        Create T2 Echo experiment circuits

        Args:
            delay_points: Number of delay time points (default: 51)
            max_delay: Maximum delay time [ns] (default: 500000)
            echo_type: Echo type "hahn" or "cpmg" (default: "hahn")
            num_echoes: Number of echoes (default: 1)
            delay_times: Directly specified delay time list [ns] (optional)

        Returns:
            T2 Echo circuit list
        """
        delay_points = kwargs.get("delay_points", 51)
        max_delay = kwargs.get(
            "max_delay", 500000
        )  # T2 measurement requires longer times
        echo_type = kwargs.get("echo_type", self.echo_type)
        num_echoes = kwargs.get("num_echoes", self.num_echoes)

        # Delay time range
        if "delay_times" in kwargs:
            delay_times = np.array(kwargs["delay_times"])
        else:
            # Default: 51 points on logarithmic scale from 100ns to 500μs
            delay_times = np.logspace(np.log10(100), np.log10(500 * 1000), num=51)
            if delay_points != 51:
                delay_times = np.linspace(100, max_delay, delay_points)

        # Save metadata
        self.experiment_params = {
            "delay_times": delay_times.tolist(),
            "delay_points": len(delay_times),
            "max_delay": max_delay,
            "echo_type": echo_type,
            "num_echoes": num_echoes,
        }

        # Create T2 Echo circuits
        circuits = []
        for delay_time in delay_times:
            circuit = self._create_single_t2_echo_circuit(
                delay_time, echo_type, num_echoes
            )
            circuits.append(circuit)

        print(
            f"T2 Echo circuits: Delay range {len(delay_times)} points from {delay_times[0]:.1f} to {delay_times[-1]:.1f} ns, "
            f"{echo_type.upper()} echo (echoes={num_echoes})"
        )

        return circuits

    def run_t2_echo_experiment_parallel(
        self,
        devices: list[str] = ["qulacs"],
        shots: int = 1024,
        parallel_workers: int = 4,
        verbose_log: bool = False,
    ) -> dict[str, Any]:
        """
        Execute T2 Echo experiment in parallel (following T1/Ramsey pattern)
        """
        print(f"🧪 T2 Echo Experiment: {self.echo_type.upper()} echo")
        print(f"   Echo count: {self.num_echoes}")
        print(f"   Devices: {devices}")
        print(f"   Shots: {shots}")
        print(f"   Workers: {parallel_workers}")

        # 1. Create circuits
        circuits = self.create_circuits()

        # 2. Parallel submission
        job_data = self._submit_t2_echo_circuits_parallel_with_order(
            circuits, devices, shots, parallel_workers
        )

        # 3. Collect results
        raw_results = self._collect_t2_echo_results_parallel_with_order(
            job_data, parallel_workers, verbose_log
        )

        # 4. Analysis
        analysis = self.analyze_results(raw_results)

        return {
            "job_data": job_data,
            "raw_results": raw_results,
            "analysis": analysis,
            "experiment_params": self.experiment_params,
        }

    def _submit_t2_echo_circuits_parallel_with_order(
        self, circuits: list[Any], devices: list[str], shots: int, parallel_workers: int
    ) -> dict[str, list[dict]]:
        """T2 Echo specialized parallel submission (using ParallelExecutionMixin)"""
        print(f"Enhanced T2 Echo parallel submission: {parallel_workers} workers")

        # Use ParallelExecutionMixin for parallel execution
        def submit_single_t2_echo_circuit(device, circuit, shots, circuit_idx):
            """Submit a single T2 Echo circuit"""
            try:
                job_id = self.submit_circuit_to_oqtopus(circuit, shots, device)
                if job_id:
                    return {
                        "job_id": job_id,
                        "device": device,
                        "circuit_idx": circuit_idx,
                        "shots": shots,
                        "submitted": True,
                        "submission_time": time.time(),
                    }
                else:
                    return None
            except Exception as e:
                delay_time = self.experiment_params["delay_times"][circuit_idx]
                print(
                    f"T2 Echo Circuit {circuit_idx} (τ={delay_time:.0f}ns) → {device}: {e}"
                )
                return None

        return self.submit_circuits_parallel_with_order(
            circuits=circuits,
            devices=devices,
            shots=shots,
            parallel_workers=parallel_workers,
            submit_function=submit_single_t2_echo_circuit,
            progress_name="T2 Echo Submission"
        )

    def _collect_t2_echo_results_parallel_with_order(
        self,
        job_data: dict[str, list[dict]],
        parallel_workers: int,
        verbose_log: bool = False,
    ) -> dict[str, list[dict]]:
        """T2 Echo specialized result collection (preserving order)"""
        total_jobs = sum(len(device_jobs) for device_jobs in job_data.values())
        print(
            f"📊 Starting T2 Echo results collection: {total_jobs} jobs from {len(job_data)} devices"
        )

        all_results = {}
        completed_count = 0
        successful_count = 0

        for device, device_job_data in job_data.items():
            device_results = []

            # Sort by circuit_index to preserve order
            sorted_jobs = sorted(device_job_data, key=lambda x: x["circuit_index"])

            for job_info in sorted_jobs:
                if not job_info["success"]:
                    device_results.append(None)  # Fill failed jobs with None
                    continue

                job_id = job_info["job_id"]
                circuit_index = job_info["circuit_index"]
                delay_time = job_info["delay_time"]

                # Poll until job completion
                result = self._poll_job_until_completion(job_id, verbose_log)
                completed_count += 1

                if result and result.get("success", False):
                    successful_count += 1
                    result["circuit_index"] = circuit_index
                    result["delay_time"] = delay_time
                    device_results.append(result)
                    print(
                        f"✅ {device}[{circuit_index}] (τ={delay_time:.0f}ns): {job_id[:8]}... collected ({completed_count}/{total_jobs})"
                    )
                else:
                    device_results.append(None)
                    print(
                        f"⚠️ {device}[{circuit_index}] (τ={delay_time:.0f}ns): {job_id[:8]}... failed"
                    )

                # Progress display
                if completed_count % 10 == 0 or completed_count == total_jobs:
                    success_rate = (successful_count / completed_count) * 100
                    print(
                        f"📈 T2 Echo Collection Progress: {completed_count}/{total_jobs} ({completed_count / total_jobs * 100:.0f}%) - {successful_count} successful"
                    )

            all_results[device] = device_results

            # Success rate per device
            device_successful = len([r for r in device_results if r is not None])
            device_total = len(device_results)
            success_rate = (
                (device_successful / device_total * 100) if device_total > 0 else 0
            )
            print(
                f"✅ {device}: {device_successful}/{device_total} T2 Echo results collected (success rate: {success_rate:.1f}%)"
            )

        overall_success_rate = (
            (successful_count / total_jobs * 100) if total_jobs > 0 else 0
        )
        print(
            f"🎉 T2 Echo Collection Complete: {successful_count}/{total_jobs} successful ({overall_success_rate:.1f}%)"
        )

        return all_results

    def _poll_job_until_completion(
        self, job_id: str, verbose_log: bool = False, max_wait_minutes: int = 30
    ) -> dict[str, Any] | None:
        """Poll until job completion"""
        max_attempts = max_wait_minutes * 12  # 5-second intervals for 30 minutes

        for attempt in range(max_attempts):
            result = self.get_oqtopus_result(
                job_id, timeout_minutes=1, verbose_log=verbose_log
            )

            if result is None:
                time.sleep(5)
                continue

            status = result.get("status", "unknown")

            if status == "succeeded":
                return result
            elif status == "failed":
                return {"success": False, "status": "failed", "job_id": job_id}
            elif status in ["running", "submitted"]:
                if verbose_log and attempt % 6 == 0:  # Log every 30 seconds
                    print(f"⏳ {job_id[:8]}... {status}")
                time.sleep(5)
                continue
            else:
                # Unknown status
                time.sleep(5)
                continue

        # Timeout
        return {"success": False, "status": "timeout", "job_id": job_id}

    def _create_single_t2_echo_circuit(
        self, delay_time: float, echo_type: str = "hahn", num_echoes: int = 1
    ) -> Any:
        """
        Create single T2 Echo circuit

        Args:
            delay_time: Total delay time [ns]
            echo_type: "hahn" or "cpmg"
            num_echoes: Number of echoes
        """
        try:
            from qiskit import QuantumCircuit
        except ImportError:
            raise ImportError(
                "Qiskit is required for T2 Echo circuit creation"
            ) from None

        # 1 qubit + 1 classical bit
        circuit = QuantumCircuit(1, 1)

        if echo_type.lower() == "hahn":
            # Hahn Echo: X/2 - τ/2 - Y - τ/2 - X/2
            circuit.sx(0)  # X/2 pulse

            # τ/2 delay
            half_delay = delay_time / 2
            if half_delay > 0:
                circuit.delay(int(half_delay), 0, unit="ns")

            # π pulse (refocusing)
            circuit.x(0)

            # τ/2 delay
            if half_delay > 0:
                circuit.delay(int(half_delay), 0, unit="ns")

            # Final X/2 pulse
            circuit.sx(0)

        elif echo_type.lower() == "cpmg":
            # CPMG: X/2 - [τ/(2n) - Y - τ/n - Y - τ/(2n)]^n - X/2
            circuit.sx(0)  # Initial X/2 pulse

            # CPMG sequence
            segment_delay = delay_time / (2 * num_echoes)
            middle_delay = delay_time / num_echoes

            for i in range(num_echoes):
                # τ/(2n) delay
                if segment_delay > 0:
                    circuit.delay(int(segment_delay), 0, unit="ns")

                # π pulse
                circuit.x(0)

                # τ/n delay (except for last echo)
                if i < num_echoes - 1:
                    if middle_delay > 0:
                        circuit.delay(int(middle_delay), 0, unit="ns")
                else:
                    # Last segment: τ/(2n)
                    if segment_delay > 0:
                        circuit.delay(int(segment_delay), 0, unit="ns")

            # Final X/2 pulse
            circuit.sx(0)

        else:
            raise ValueError(f"Unknown echo_type: {echo_type}. Use 'hahn' or 'cpmg'")

        # Measurement
        circuit.measure(0, 0)

        return circuit

    def _convert_decimal_to_binary_counts(
        self, counts: dict[str, int]
    ) -> dict[str, int]:
        """
        Convert OQTOPUS decimal counts to binary counts
        """
        binary_counts = {}
        for decimal_str, count in counts.items():
            try:
                decimal_value = int(decimal_str)
                binary_str = format(decimal_value, "01b")  # 1 qubit
                binary_counts[binary_str] = binary_counts.get(binary_str, 0) + count
            except ValueError:
                binary_counts[decimal_str] = count
        return binary_counts

    def _calculate_p0_probability(self, counts: dict[str, int]) -> float:
        """
        Calculate P(0) probability (for T2 Echo)
        In T2 Echo, P(0) decays as P(0) = A * exp(-t/T2) + B
        """
        binary_counts = self._convert_decimal_to_binary_counts(counts)
        total = sum(binary_counts.values())
        if total == 0:
            return 0.5  # Default value
        n_0 = binary_counts.get("0", 0)
        p0 = n_0 / total
        return p0

    def analyze_results(
        self, raw_results: dict[str, list[dict[str, Any]]]
    ) -> dict[str, Any]:
        """
        T2 Echo result analysis (T1 style: compatible with local simulator)
        """
        if not raw_results:
            return {"error": "No results to analyze"}

        delay_times = np.array(self.experiment_params["delay_times"])
        analysis_results = {
            "experiment_info": {
                "delay_points": len(delay_times),
                "expected_t2": self.expected_t2,
                "echo_type": self.experiment_params.get("echo_type", "hahn"),
                "num_echoes": self.experiment_params.get("num_echoes", 1),
            },
            "device_results": {},
        }

        for device, device_results in raw_results.items():
            if not device_results:
                continue

            device_analysis = self._analyze_device_results(device_results, delay_times)
            analysis_results["device_results"][device] = device_analysis

        return analysis_results

    def _analyze_device_results(
        self, device_results: list[dict[str, Any]], delay_times: np.ndarray
    ) -> dict[str, Any]:
        """
        Single device result analysis (T1 style: compatible with local simulator)
        """
        print(f"🔍 Analyzing {len(device_results)} T2 Echo results in order...")

        p0_values = []
        for i, result in enumerate(device_results):
            delay_time = delay_times[i] if i < len(delay_times) else f"unknown[{i}]"

            if result and result.get("success", True):
                counts = result.get("counts", {})

                print(f"🔍 Raw decimal counts: {counts}")

                # Calculate P(0) probability (for T2 Echo)
                p0 = self._calculate_p0_probability(counts)
                p0_values.append(p0)

                binary_counts = self._convert_decimal_to_binary_counts(counts)
                print(f"🔍 Converted binary counts: {binary_counts}")

                # Debug order for first 5 points
                if i < 5:
                    print(
                        f"🔍 Point {i}: τ={delay_time}ns, P(0)={p0:.3f}, counts={dict(counts)}"
                    )
            else:
                p0_values.append(np.nan)
                if i < 5:
                    print(f"🔍 Point {i}: τ={delay_time}ns, FAILED")

        # Summary for order verification
        valid_p0s = np.array([p for p in p0_values if not np.isnan(p)])
        if len(valid_p0s) >= 2:
            trend = "decreasing" if valid_p0s[-1] < valid_p0s[0] else "increasing"
            print(
                f"📈 T2 Echo trend: P(0) {valid_p0s[0]:.3f} → {valid_p0s[-1]:.3f} ({trend})"
            )

        # Calculate statistics
        initial_p0 = float(valid_p0s[0]) if len(valid_p0s) > 0 else 0.5
        final_p0 = float(valid_p0s[-1]) if len(valid_p0s) > 0 else 0.5
        success_rate = len(valid_p0s) / len(p0_values) if p0_values else 0.0
        success_count = len(valid_p0s)
        total_count = len(p0_values)
        decay_amplitude = (
            (np.max(valid_p0s) - np.min(valid_p0s)) if len(valid_p0s) > 1 else 0.0
        )

        device_analysis = {
            "p0_values": p0_values,
            "delay_times": delay_times.tolist(),
            "statistics": {
                "initial_p0": initial_p0,
                "final_p0": final_p0,
                "success_rate": success_rate,
                "successful_jobs": success_count,
                "failed_jobs": total_count - success_count,
                "total_jobs": total_count,
                "decay_amplitude": decay_amplitude,
            },
        }

        # Fitting
        if self.enable_fitting and len(valid_p0s) >= 3:
            try:
                # Fitting with data excluding NaN values
                valid_indices = [i for i, p in enumerate(p0_values) if not np.isnan(p)]
                valid_delay_times = delay_times[valid_indices]

                fitting_result = self._fit_t2_decay(valid_delay_times, valid_p0s)
                device_analysis.update(fitting_result)

                t2_fitted = fitting_result.get("t2_fitted", 0.0)
                r_squared = fitting_result.get("fitting_quality", {}).get(
                    "r_squared", 0.0
                )
                method = fitting_result.get("fitting_quality", {}).get(
                    "method", "unknown"
                )

                print(
                    f"T2 Echo: T2 = {t2_fitted:.1f} ns ({method}, R²={r_squared:.3f}) [with RO mitigation]"
                )
            except Exception as e:
                print(f"T2 Echo: Fitting failed: {e}")
                device_analysis["t2_fitted"] = 0.0
                device_analysis["fitting_quality"] = {
                    "method": "fitting_failed",
                    "r_squared": 0.0,
                    "error": str(e),
                }
        else:
            device_analysis["t2_fitted"] = 0.0
            device_analysis["fitting_quality"] = {
                "method": (
                    "no_fitting" if not self.enable_fitting else "insufficient_data"
                ),
                "r_squared": 0.0,
                "error": "disabled" if not self.enable_fitting else "insufficient_data",
            }

            print(
                f"T2 Echo: Raw data decay amplitude = {decay_amplitude:.3f} [with RO mitigation]"
            )

        return device_analysis

    def _fit_t2_decay(
        self, delay_times: np.ndarray, p0_values: np.ndarray
    ) -> dict[str, Any]:
        """
        T2 Echo decay fitting (exponential decay) based on P(0)
        """
        try:
            from scipy.optimize import curve_fit
        except ImportError:
            raise ImportError("scipy is required for T2 fitting") from None

        def exponential_decay(t, a, t2, c):
            """T2 Echo exponential decay model: P(0) = a * exp(-t/T2) + c"""
            return a * np.exp(-t / t2) + c

        try:
            # Initial parameter settings compliant with Qiskit T2 Hahn
            # Theoretically expected values: P(0) = A * exp(-t/T2) + B
            # A = 0.5 (amplitude), B = 0.5 (baseline), T2 = expected_t2
            a_init = 0.5  # Theoretical amplitude
            c_init = 0.5  # Theoretical baseline (converges to 0.5)
            t2_init = self.expected_t2  # Initial T2 estimate

            # Fine-tuning based on data
            p0_max = np.max(p0_values)
            p0_min = np.min(p0_values)
            if p0_max > 0.1 and p0_min < 0.9:  # When significant data is available
                data_amplitude = p0_max - p0_min
                data_baseline = p0_min
                # Take the average between data and theoretical values
                a_init = (0.5 + data_amplitude) / 2
                c_init = (0.5 + data_baseline) / 2

            # Execute fitting with physically reasonable boundary conditions
            # amp: [0, 1], tau: [100ns, 1ms], base: [0, 1]
            popt, pcov = curve_fit(
                exponential_decay,
                delay_times,
                p0_values,
                p0=[a_init, t2_init, c_init],
                bounds=([0, 100, 0], [1.0, 1000000, 1.0]),  # T2: 100ns ~ 1ms
                maxfev=10000,
            )

            a_fitted, t2_fitted, c_fitted = popt

            # Evaluate fitting quality
            y_pred = exponential_decay(delay_times, *popt)
            ss_res = np.sum((p0_values - y_pred) ** 2)
            ss_tot = np.sum((p0_values - np.mean(p0_values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            return {
                "t2_fitted": t2_fitted,
                "amplitude_fitted": a_fitted,
                "offset_fitted": c_fitted,
                "fitting_quality": {
                    "method": "exponential_decay_t2",
                    "r_squared": r_squared,
                    "fit_parameters": {
                        "amplitude": a_fitted,
                        "t2": t2_fitted,
                        "offset": c_fitted,
                    },
                },
            }

        except Exception as e:
            # Fallback when fitting fails
            return {
                "t2_fitted": 0.0,
                "amplitude_fitted": 0.0,
                "offset_fitted": 0.0,
                "fitting_quality": {
                    "method": "fitting_failed",
                    "r_squared": 0.0,
                    "error": str(e),
                },
            }

    # Implementation of BaseExperiment abstract methods
    def save_experiment_data(
        self, results: dict[str, Any], metadata: dict[str, Any] = None
    ) -> str:
        """Save T2 Echo experiment data"""
        timestamp = int(time.time())

        experiment_data = {
            "experiment_type": "T2_Echo",
            "experiment_timestamp": timestamp,
            "experiment_parameters": self.experiment_params,
            "analysis_results": results,
            "oqtopus_configuration": {
                "transpiler_options": self.transpiler_options,
                "mitigation_options": self.mitigation_options,
                "basis_gates": self.anemone_basis_gates,
            },
            "metadata": metadata or {},
        }

        return self.data_manager.save_data(
            experiment_data, "t2_echo_experiment_results"
        )

    def save_complete_experiment_data(self, results: dict[str, Any]) -> str:
        """Save complete T2 Echo experiment data"""
        # Save main results
        main_path = self.save_experiment_data(results)

        # Save device-specific details
        device_data = {}
        for device, device_result in results.get("device_results", {}).items():
            device_data[device] = {
                "t2_fitted": device_result.get("t2_fitted", 0.0),
                "echo_type": self.experiment_params.get("echo_type", "hahn"),
                "num_echoes": self.experiment_params.get("num_echoes", 1),
                "statistics": device_result.get("statistics", {}),
                "fitting_quality": device_result.get("fitting_quality", {}),
            }

        self.data_manager.save_data(device_data, "device_t2_echo_summary")

        # Save data for plotting
        plot_data = {}
        for device, device_result in results.get("device_results", {}).items():
            plot_data[device] = {
                "delay_times": device_result.get("delay_times", []),
                "p0_values": device_result.get("p0_values", []),
                "t2_fitted": device_result.get("t2_fitted", 0.0),
            }

        self.data_manager.save_data(plot_data, "t2_echo_p0_values_for_plotting")

        # Generate plot
        if hasattr(self, "generate_t2_echo_plot"):
            try:
                self.generate_t2_echo_plot(results, save_plot=True, show_plot=False)
            except Exception as e:
                print(f"Plot generation failed: {e}")

        return main_path

    def generate_t2_echo_plot(
        self, results: dict[str, Any], save_plot: bool = True, show_plot: bool = False
    ) -> str | None:
        """Generate T2 Echo experiment plot with all formatting"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available - skipping plot generation")
            return None

        delay_times = results.get("delay_times", np.linspace(100, 500000, 51))
        device_results = results.get("device_results", {})

        if not device_results:
            print("No device results for plotting")
            return None

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot experimental data for each device
        colors = ["blue", "red", "green", "orange", "purple"]

        for i, (device, device_data) in enumerate(device_results.items()):
            if "p0_values" in device_data:
                p0_values = device_data["p0_values"]
                t2_fitted = device_data.get("t2_fitted", 0.0)
                fitting_quality = device_data.get("fitting_quality", {})
                r_squared = fitting_quality.get("r_squared", 0.0)
                color = colors[i % len(colors)]

                # Plot experimental data
                ax.semilogx(
                    delay_times,
                    p0_values,
                    "o",
                    markersize=6,
                    label=f"{device} data",
                    alpha=0.8,
                    color=color,
                )

                # Plot fitting curve
                if t2_fitted > 0:
                    # Draw fitted exponential decay curve
                    fit_delays = np.logspace(
                        np.log10(min(delay_times)), np.log10(max(delay_times)), 100
                    )

                    # T2 Echo decay: P(t) = A * exp(-t/T2) + offset
                    amplitude = device_data.get(
                        "amplitude_fitted",
                        max(p0_values) - min(p0_values) if p0_values else 0.5,
                    )
                    offset = device_data.get(
                        "offset_fitted", min(p0_values) if p0_values else 0.5
                    )
                    fit_curve = amplitude * np.exp(-fit_delays / t2_fitted) + offset

                    ax.semilogx(
                        fit_delays,
                        fit_curve,
                        "-",
                        linewidth=2,
                        color=color,
                        alpha=0.7,
                        label=f"{device} fit (T2={t2_fitted:.0f}ns, R²={r_squared:.3f})",
                    )

        # Formatting
        echo_type = self.experiment_params.get("echo_type", "hahn")
        num_echoes = self.experiment_params.get("num_echoes", 1)

        ax.set_xlabel("Delay time τ [ns] (log scale)", fontsize=14)
        ax.set_ylabel("P(0)", fontsize=14)
        ax.set_title(
            f"OQTOPUS Experiments T2 Echo Experiment - {echo_type.upper()} (echoes={num_echoes})",
            fontsize=16,
            fontweight="bold",
        )
        ax.grid(True, which="both", ls="--", linewidth=0.5)
        ax.legend(fontsize=12)
        ax.set_ylim(0, 1.1)

        plot_filename = None
        if save_plot:
            # Save plot in experiment results directory
            plt.tight_layout()
            plot_filename = (
                f"t2_echo_plot_{self.experiment_name}_{int(time.time())}.png"
            )

            # Always save to experiment results directory
            if hasattr(self, "data_manager") and hasattr(
                self.data_manager, "session_dir"
            ):
                plot_path = f"{self.data_manager.session_dir}/plots/{plot_filename}"
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                print(f"Plot saved: {plot_path}")
                plot_filename = plot_path  # Return full path
            else:
                # Fallback: save in current directory but warn
                plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
                print(f"⚠️ Plot saved to current directory: {plot_filename}")
                print("   (data_manager not available)")

        # Try to display plot
        if show_plot:
            try:
                plt.show()
            except Exception as e:
                print(f"Could not display plot: {e}")

        plt.close(fig)  # Clean up memory
        return plot_filename

    def display_results(self, results: dict[str, Any], use_rich: bool = True) -> None:
        """Display T2 Echo results"""
        if use_rich:
            try:
                from rich.console import Console
                from rich.table import Table

                console = Console()
                table = Table(
                    title="T2 Echo Results",
                    show_header=True,
                    header_style="bold magenta",
                )

                table.add_column("Device", style="cyan", no_wrap=True)
                table.add_column("T2 Fitted [ns]", justify="right")
                table.add_column("Echo Type", justify="center")
                table.add_column("Echoes", justify="center")
                table.add_column("Decay Amp", justify="right")
                table.add_column("Success Rate", justify="right")
                table.add_column("Clear Signal", justify="center")

                for device, device_result in results.get("device_results", {}).items():
                    t2_fitted = device_result.get("t2_fitted", 0.0)
                    echo_type = self.experiment_params.get("echo_type", "hahn")
                    num_echoes = self.experiment_params.get("num_echoes", 1)
                    decay_amp = device_result.get("statistics", {}).get(
                        "decay_amplitude", 0.0
                    )
                    success_rate = device_result.get("statistics", {}).get(
                        "success_rate", 0.0
                    )
                    successful_jobs = device_result.get("statistics", {}).get(
                        "successful_jobs", 0
                    )
                    total_jobs = device_result.get("statistics", {}).get(
                        "total_jobs", 0
                    )

                    clear_signal = "YES" if decay_amp > 0.1 else "NO"

                    table.add_row(
                        device.upper(),
                        f"{t2_fitted:.1f}" if t2_fitted > 0 else "N/A",
                        echo_type.upper(),
                        f"{num_echoes}",
                        f"{decay_amp:.3f}",
                        f"{success_rate * 100:.1f}%\n({successful_jobs}/{total_jobs})",
                        clear_signal,
                    )

                console.print(table)

                # Display experiment parameters
                echo_type = self.experiment_params.get("echo_type", "hahn")
                num_echoes = self.experiment_params.get("num_echoes", 1)
                max_delay = self.experiment_params.get("max_delay", 0) / 1000  # μs

                console.print(f"\nExpected T2: {self.expected_t2} ns")
                console.print(f"Echo type: {echo_type.upper()} (echoes={num_echoes})")
                console.print(f"Max delay: {max_delay:.0f} μs")
                console.print(
                    f"Parameter fitting: {'enabled' if self.enable_fitting else 'disabled'}"
                )
                console.print("Clear signal threshold: 0.1")

            except ImportError:
                # Rich not available, fallback to simple print
                self._display_results_simple(results)
        else:
            self._display_results_simple(results)

    def _display_results_simple(self, results: dict[str, Any]) -> None:
        """Simple T2 Echo results display"""
        print("\n=== T2 Echo Results ===")
        for device, device_result in results.get("device_results", {}).items():
            t2_fitted = device_result.get("t2_fitted", 0.0)
            echo_type = self.experiment_params.get("echo_type", "hahn")
            num_echoes = self.experiment_params.get("num_echoes", 1)
            decay_amp = device_result.get("statistics", {}).get("decay_amplitude", 0.0)
            success_rate = device_result.get("statistics", {}).get("success_rate", 0.0)

            print(
                f"{device}: T2={t2_fitted:.1f}ns, {echo_type.upper()}(echoes={num_echoes}), "
                f"decay={decay_amp:.3f}, success={success_rate * 100:.1f}%"
            )
