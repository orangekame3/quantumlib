#!/usr/bin/env python3
"""
CHSH Experiment Class - Specialized class for CHSH Bell inequality violation experiments
Inherits from BaseExperiment and provides implementation specialized for CHSH experiments
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np

from ...circuit.chsh_circuits import create_chsh_circuit
from ...core.base_experiment import BaseExperiment
from ...core.parallel_execution import ParallelExecutionMixin


class CHSHExperiment(BaseExperiment, ParallelExecutionMixin):
    """
    CHSH Bell inequality violation experiment class

    Specialized features:
    - Automatic CHSH circuit generation
    - Bell inequality value calculation
    - Phase scan experiments
    - CHSH-specific data saving
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

        # Filter kwargs to pass to BaseExperiment
        base_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in chsh_specific_params
            and k not in cli_only_params
            and k != "experiment_name"
        }
        super().__init__(experiment_name, **base_kwargs)

        # CHSH experiment-specific settings
        self.classical_bound = 2.0
        self.theoretical_max_s = 2 * np.sqrt(2)

        print(
            f"CHSH bounds: Classical ≤ {self.classical_bound}, "
            f"Quantum max ≈ {self.theoretical_max_s:.3f}"
        )

    def create_circuits(self, **kwargs) -> list[Any]:
        """
        Create CHSH experiment circuits (T1/Ramsey standard pattern)
        Generate batch circuits using 4-measurement method

        Args:
            points: Number of phase points (default: 20) ← passed from CLI
            phase_points: Number of phase points (default: 20)
            theta_a: Alice angle (default: 0)
            theta_b: Bob angle (default: π/4)

        Returns:
            List of CHSH circuits (4 measurements × phase_points circuits)
        """
        # Prioritize points parameter from CLI (same pattern as T1/Ramsey)
        phase_points = kwargs.get("points", kwargs.get("phase_points", 20))
        kwargs.get("theta_a", 0)
        kwargs.get("theta_b", np.pi / 4)

        # Phase range
        phase_range = np.linspace(0, 2 * np.pi, phase_points)

        # Standard CHSH measurement angles
        angles = {
            "theta_a0": 0,  # Alice measurement angle 1
            "theta_a1": np.pi / 2,  # Alice measurement angle 2
            "theta_b0": np.pi / 4,  # Bob measurement angle 1
            "theta_b1": -np.pi / 4,  # Bob measurement angle 2
        }

        # 4-measurement combinations
        measurements = [
            (angles["theta_a0"], angles["theta_b0"]),  # ⟨A₀B₀⟩
            (angles["theta_a0"], angles["theta_b1"]),  # ⟨A₀B₁⟩
            (angles["theta_a1"], angles["theta_b0"]),  # ⟨A₁B₀⟩
            (angles["theta_a1"], angles["theta_b1"]),  # ⟨A₁B₁⟩
        ]

        # Save metadata (T1/Ramsey pattern)
        self.experiment_params = {
            "phase_range": phase_range.tolist(),
            "phase_points": len(phase_range),
            "angles": angles,
            "measurements": measurements,
        }

        # Circuit creation: sequentially generate all phase×measurement combinations (T1/Ramsey pattern)
        circuits = []
        for _i, phase_phi in enumerate(phase_range):
            for _j, (theta_a_meas, theta_b_meas) in enumerate(measurements):
                circuit = self._create_single_chsh_circuit(
                    theta_a_meas, theta_b_meas, phase_phi
                )
                circuits.append(circuit)

        # Unified with T1/Ramsey standard log pattern
        print(
            f"CHSH circuits: Phase range {len(phase_range)} points from {phase_range[0]:.3f} to {phase_range[-1]:.3f}, 4 measurements = {len(circuits)} circuits"
        )
        print(
            "CHSH circuit structure: |Φ⁺⟩ → A(θₐ), B(θᵦ) → measure (expected: Bell inequality violation with S-value)"
        )

        return circuits

    def _create_single_chsh_circuit(
        self, theta_a: float, theta_b: float, phase_phi: float
    ):
        """
        Create single CHSH circuit (T1/Ramsey pattern)
        """
        return create_chsh_circuit(theta_a, theta_b, phase_phi)

    def analyze_results(
        self, results: dict[str, list[dict[str, Any]]], **kwargs
    ) -> dict[str, Any]:
        """
        Analyze CHSH experiment results (T1/Ramsey standard pattern)

        Args:
            results: Raw measurement results (passed from BaseExperimentCLI)

        Returns:
            CHSH analysis results
        """
        if not results:
            return {"error": "No results to analyze"}

        # Get necessary information from experiment_params
        phase_range = np.array(self.experiment_params["phase_range"])
        angles = self.experiment_params["angles"]
        measurements = self.experiment_params["measurements"]

        print("   → Processing CHSH 4-measurement results...")

        # Convert results from BaseExperimentCLI to 4-measurement CHSH format
        processed_results = self._analyze_chsh_device_results(
            results, phase_range, measurements
        )

        print("   → Creating CHSH analysis...")
        analysis = self._create_chsh_analysis(phase_range, processed_results, angles)

        return analysis

    def _analyze_chsh_device_results(
        self,
        results: dict[str, list[dict[str, Any]]],
        phase_range: np.ndarray,
        measurements: list[tuple],
    ) -> dict[str, dict]:
        """
        Analyze CHSH results by device (T1/Ramsey pattern)
        """
        all_results = {}
        phase_points = len(phase_range)

        for device, device_results in results.items():
            print(f"   Processing {device} results...")

            device_s_values = []
            device_expectations = []

            for phase_idx in range(phase_points):
                phase_expectations = []

                for meas_idx in range(4):
                    circuit_idx = phase_idx * 4 + meas_idx

                    if (
                        circuit_idx < len(device_results)
                        and device_results[circuit_idx] is not None
                    ):
                        result = device_results[circuit_idx]
                        # Check if result was successful
                        if result and result.get("success", False):
                            counts = result.get("counts", {})
                            if counts:
                                expectation = self._calculate_expectation_value_oqtopus_compatible(
                                    counts
                                )
                                phase_expectations.append(expectation)

                                if phase_idx == 0 and meas_idx < 2:
                                    print(
                                        f"   Debug - Phase {phase_idx}, Meas {meas_idx}: counts={counts}, exp={expectation:.3f}"
                                    )
                            else:
                                phase_expectations.append(0.0)
                        else:
                            phase_expectations.append(0.0)
                    else:
                        phase_expectations.append(0.0)

                # CHSH S value calculation: S = E1 + E2 + E3 - E4
                if len(phase_expectations) == 4:
                    E1, E2, E3, E4 = phase_expectations
                    S = E1 + E2 + E3 - E4
                else:
                    S = 0.0

                device_s_values.append(S)
                device_expectations.append(phase_expectations)

            all_results[device] = {
                "S_values": device_s_values,
                "expectations": device_expectations,
                "measurement_angles": {
                    "theta_a0": 0,
                    "theta_a1": np.pi / 2,
                    "theta_b0": np.pi / 4,
                    "theta_b1": -np.pi / 4,
                },
            }

            # Statistics
            S_array = np.array(device_s_values)
            max_S = np.max(np.abs(S_array))
            violations = int(np.sum(np.abs(S_array) > 2.0))
            print(
                f"   {device}: Max |S| = {max_S:.3f}, Bell violations: {violations}/{phase_points}"
            )

        return all_results

    def _create_chsh_analysis(
        self,
        phase_range: np.ndarray,
        processed_results: dict[str, dict],
        angles: dict[str, float],
    ) -> dict[str, Any]:
        """
        Create CHSH analysis results (T1/Ramsey pattern)
        """
        analysis = {
            "experiment_info": {
                "theta_a0": angles["theta_a0"],
                "theta_a1": angles["theta_a1"],
                "theta_b0": angles["theta_b0"],
                "theta_b1": angles["theta_b1"],
                "phase_points": len(phase_range),
                "classical_bound": 2.0,
                "theoretical_max_s": 2 * np.sqrt(2),
            },
            "theoretical_values": {
                "phase_range": phase_range.tolist(),
                "S_theoretical": (2 * np.sqrt(2) * np.cos(phase_range)).tolist(),
            },
            "device_results": {},
        }

        for device, device_data in processed_results.items():
            S_values = device_data["S_values"]
            S_array = np.array(S_values)
            bell_violations = int(np.sum(np.abs(S_array) > 2.0))
            max_S = float(np.max(np.abs(S_array)))

            analysis["device_results"][device] = {
                "S_values": S_values,
                "expectations": device_data["expectations"],
                "statistics": {
                    "max_S_magnitude": max_S,
                    "bell_violations": bell_violations,
                    "success_rate": 1.0,
                    "mean_S_magnitude": float(np.mean(np.abs(S_array))),
                },
            }

        return analysis

    def _analyze_device_results(
        self, device_results: list[dict[str, Any]], phase_range: np.ndarray
    ) -> dict[str, Any]:
        """
        Single device result analysis
        """
        S_values = []
        expectations = []

        for i, result in enumerate(device_results):
            if result and result["success"]:
                counts = result["counts"]

                # Calculate expectation value
                expectation = self._calculate_expectation_value(counts)
                expectations.append(expectation)

                # Calculate S value (simplified version)
                phi = phase_range[i] if i < len(phase_range) else 0
                S = 2 * np.sqrt(2) * expectation * np.cos(phi)
                S_values.append(S)
            else:
                expectations.append(np.nan)
                S_values.append(np.nan)

        # Calculate statistics
        valid_s = np.array([s for s in S_values if not np.isnan(s)])

        return {
            "S_values": S_values,
            "expectations": expectations,
            "statistics": {
                "max_S_magnitude": (
                    float(np.max(np.abs(valid_s))) if len(valid_s) > 0 else 0
                ),
                "bell_violations": int(np.sum(np.abs(valid_s) > self.classical_bound)),
                "success_rate": len(valid_s) / len(S_values) if S_values else 0,
                "mean_expectation": float(np.nanmean(expectations)),
            },
        }

    def _calculate_expectation_value(self, counts: dict[str, int]) -> float:
        """
        Calculate CHSH expectation value
        """
        total = sum(counts.values())
        if total == 0:
            return 0.0

        # CHSH expectation value: E = (N_00 + N_11 - N_01 - N_10) / N_total
        n_00 = counts.get("00", 0)
        n_11 = counts.get("11", 0)
        n_01 = counts.get("01", 0)
        n_10 = counts.get("10", 0)

        expectation = (n_00 + n_11 - n_01 - n_10) / total
        return expectation

    def _compare_devices(
        self, device_results: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Inter-device comparison analysis
        """
        if len(device_results) < 2:
            return {"note": "Multiple devices required for comparison"}

        comparison = {
            "device_count": len(device_results),
            "bell_violation_comparison": {},
            "max_S_comparison": {},
        }

        for device, analysis in device_results.items():
            stats = analysis["statistics"]
            comparison["bell_violation_comparison"][device] = stats["bell_violations"]
            comparison["max_S_comparison"][device] = stats["max_S_magnitude"]

        return comparison

    def save_experiment_data(
        self, results: dict[str, Any], metadata: dict[str, Any] = None
    ) -> str:
        """
        Save CHSH experiment data
        """
        # CHSH experiment-specific save format
        chsh_data = {
            "experiment_type": "CHSH_Bell_Inequality",
            "experiment_timestamp": time.time(),
            "experiment_parameters": self.experiment_params,
            "analysis_results": results,
            "oqtopus_configuration": {
                "transpiler_options": self.transpiler_options,
                "mitigation_options": self.mitigation_options,
                "basis_gates": self.anemone_basis_gates,
            },
            "metadata": metadata or {},
        }

        # Save main results
        main_file = self.data_manager.save_data(chsh_data, "chsh_experiment_results")

        # Save additional files
        if "device_results" in results:
            # Device-specific summary
            device_summary = {
                device: analysis["statistics"]
                for device, analysis in results["device_results"].items()
            }
            self.data_manager.save_data(device_summary, "device_performance_summary")

            # S values only (for plotting)
            s_values_data = {
                "phase_range": self.experiment_params["phase_range"],
                "theoretical_S": results["theoretical_values"]["S_theoretical"],
                "device_S_values": {
                    device: analysis["S_values"]
                    for device, analysis in results["device_results"].items()
                },
            }
            self.data_manager.save_data(s_values_data, "s_values_for_plotting")

        return main_file

    # Convenience methods specific to CHSH experiments
    def run_phase_scan(
        self,
        devices: list[str] = ["qulacs"],
        phase_points: int = 20,
        theta_a: float = 0,
        theta_b: float = np.pi / 4,
        shots: int = 1024,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Execute phase scan CHSH experiment
        """
        return self.run_experiment(
            devices=devices,
            shots=shots,
            phase_points=phase_points,
            theta_a=theta_a,
            theta_b=theta_b,
            **kwargs,
        )

    def run_chsh_experiment_parallel(
        self,
        devices: list[str] = ["qulacs"],
        shots: int = 1024,
        parallel_workers: int = 4,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Parallel execution of CHSH experiment (preserving phase order, T1/Ramsey standard pattern)
        """
        print(f"🔬 Running CHSH experiment with {parallel_workers} parallel workers")

        # Create circuits
        circuits = self.create_circuits(**kwargs)
        phase_range = self.experiment_params["phase_range"]

        print(
            f"   📊 {len(circuits)} circuits × {len(devices)} devices = {len(circuits) * len(devices)} jobs"
        )

        # Parallel execution (preserving order)
        job_data = self._submit_chsh_circuits_parallel_with_order(
            circuits, devices, shots, parallel_workers
        )

        # Collect results (preserving order)
        raw_results = self._collect_chsh_results_parallel_with_order(
            job_data, parallel_workers
        )

        # Analyze results (with error handling)
        try:
            analysis = self.analyze_results(raw_results)
        except Exception as e:
            print(f"Analysis failed: {e}, creating minimal analysis")
            analysis = {
                "experiment_info": {"phase_points": len(phase_range), "error": str(e)},
                "device_results": {},
            }

        return {
            "phase_range": phase_range,
            "device_results": analysis["device_results"],
            "analysis": analysis,
            "method": "chsh_parallel_quantumlib",
        }

    def run_experiment(
        self,
        devices: list[str] = ["qulacs"],
        shots: int = 1024,
        submit_interval: float = 1.0,
        wait_minutes: int = 30,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Execute CHSH experiment (following base_cli unified flow)
        """
        # Since base_cli calls parallel methods directly, only basic result collection is needed here
        print("⚠️ run_experiment called directly - use CLI framework instead")
        return self.run_chsh_experiment_parallel(
            devices=devices, shots=shots, parallel_workers=4, **kwargs
        )

    def run_4_measurement_chsh(
        self,
        devices: list[str] = ["qulacs"],
        phase_points: int = 20,
        shots: int = 1024,
        parallel_workers: int = 4,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Execute 4-measurement CHSH method (standard quantum Bell inequality method)
        Measure ⟨A₀B₀⟩, ⟨A₀B₁⟩, ⟨A₁B₀⟩, ⟨A₁B₁⟩ individually and calculate S = E₁ + E₂ + E₃ - E₄
        """
        # Standard CHSH measurement angles
        angles = {
            "theta_a0": 0,  # Alice measurement angle 1
            "theta_a1": np.pi / 2,  # Alice measurement angle 2
            "theta_b0": np.pi / 4,  # Bob measurement angle 1
            "theta_b1": -np.pi / 4,  # Bob measurement angle 2
        }

        phase_range = np.linspace(0, 2 * np.pi, phase_points)

        # 4-measurement combinations
        measurements = [
            (angles["theta_a0"], angles["theta_b0"]),  # ⟨A₀B₀⟩
            (angles["theta_a0"], angles["theta_b1"]),  # ⟨A₀B₁⟩
            (angles["theta_a1"], angles["theta_b0"]),  # ⟨A₁B₀⟩
            (angles["theta_a1"], angles["theta_b1"]),  # ⟨A₁B₁⟩
        ]

        # Batch circuit creation: create all phase×measurement combinations at once
        all_circuits = []
        circuit_metadata = []

        for i, phase_phi in enumerate(phase_range):
            for j, (theta_a, theta_b) in enumerate(measurements):
                circuit = create_chsh_circuit(theta_a, theta_b, phase_phi)
                all_circuits.append(circuit)
                circuit_metadata.append(
                    {
                        "phase_index": i,
                        "measurement_index": j,
                        "phase_phi": phase_phi,
                        "theta_a": theta_a,
                        "theta_b": theta_b,
                    }
                )

        print(
            f"Creating batch circuits: {phase_points} phases × 4 measurements = {len(all_circuits)} circuits"
        )

        # Parallel submission and collection of batch circuits
        job_data = self._submit_chsh_circuits_parallel_with_order(
            all_circuits, devices, shots, parallel_workers
        )

        raw_results = self._collect_chsh_results_parallel_with_order(
            job_data, parallel_workers
        )

        # Structure results for CHSH analysis
        processed_results = self._process_4_measurement_results(
            raw_results, circuit_metadata, phase_range, measurements, devices
        )

        # Create standard analysis results
        analysis = self._create_4_measurement_analysis(
            phase_range, processed_results, angles
        )

        # Set experiment_params (for data saving)
        self.experiment_params = {
            "theta_a": 0,
            "theta_b": np.pi / 4,
            "phase_range": phase_range.tolist(),
            "phase_points": len(phase_range),
        }

        return {
            "phase_range": phase_range,
            "device_results": processed_results,
            "analysis": analysis,
            "method": "4_measurement_chsh_quantumlib",
        }

    def run_angle_comparison(
        self,
        devices: list[str] = ["qulacs"],
        angle_pairs: list[tuple] | None = None,
        shots: int = 1024,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Execute angle comparison CHSH experiment
        """
        if angle_pairs is None:
            angle_pairs = [(0, np.pi / 4), (np.pi / 4, 0), (np.pi / 8, np.pi / 8)]

        all_results = []

        for i, (theta_a, theta_b) in enumerate(angle_pairs):
            print(
                f"\n🔬 Angle pair {i + 1}/{len(angle_pairs)}: θ_A={theta_a:.3f}, θ_B={theta_b:.3f}"
            )

            # Experiment at single phase point
            result = self.run_experiment(
                devices=devices,
                shots=shots,
                phase_range=[0],  # φ=0 only
                theta_a=theta_a,
                theta_b=theta_b,
                **kwargs,
            )

            result["angle_pair"] = (theta_a, theta_b)
            all_results.append(result)

        # Integrated results
        return {
            "experiment_type": "CHSH_Angle_Comparison",
            "angle_pairs": angle_pairs,
            "individual_results": all_results,
            "comparison_summary": self._summarize_angle_comparison(all_results),
        }

    def _summarize_angle_comparison(
        self, results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Summary of angle comparison results
        """
        summary = {
            "angle_performance": {},
            "best_angle_pair": None,
            "max_bell_violation": 0,
        }

        for result in results:
            angle_pair = result["angle_pair"]
            angle_key = f"θA={angle_pair[0]:.3f}_θB={angle_pair[1]:.3f}"

            if "device_results" in result["analyzed_results"]:
                for device, analysis in result["analyzed_results"][
                    "device_results"
                ].items():
                    bell_violations = analysis["statistics"]["bell_violations"]
                    max_s = analysis["statistics"]["max_S_magnitude"]

                    if angle_key not in summary["angle_performance"]:
                        summary["angle_performance"][angle_key] = {}

                    summary["angle_performance"][angle_key][device] = {
                        "bell_violations": bell_violations,
                        "max_S_magnitude": max_s,
                    }

                    if max_s > summary["max_bell_violation"]:
                        summary["max_bell_violation"] = max_s
                        summary["best_angle_pair"] = angle_pair

        return summary

    def _submit_chsh_circuits_parallel_with_order(
        self, circuits: list[Any], devices: list[str], shots: int, parallel_workers: int
    ) -> dict[str, list[dict]]:
        """
        Parallel submission of CHSH circuits (using ParallelExecutionMixin)
        """
        print(f"Enhanced CHSH parallel submission: {parallel_workers} workers")

        # Emergency response when experiment_params is not set
        if not hasattr(self, "experiment_params") or not self.experiment_params:
            print("⚠️ experiment_params not set, creating default...")
            phase_points = len(circuits) // 4  # Circuit count ÷ 4 since 4 measurements
            phase_range = np.linspace(0, 2 * np.pi, phase_points)
            self.experiment_params = {
                "phase_range": phase_range.tolist(),
                "phase_points": phase_points,
            }

        if not self.oqtopus_available:
            return self._submit_chsh_circuits_locally_parallel(
                circuits, devices, shots, parallel_workers
            )

        # Use ParallelExecutionMixin for parallel execution
        def submit_single_chsh_circuit(device, circuit, shots, circuit_idx):
            """Submit a single CHSH circuit"""
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
                phase_idx = circuit_idx // 4
                meas_idx = circuit_idx % 4
                print(
                    f"CHSH Circuit {circuit_idx} (phase {phase_idx}, meas {meas_idx}) → {device}: {e}"
                )
                return None

        return self.submit_circuits_parallel_with_order(
            circuits=circuits,
            devices=devices,
            shots=shots,
            parallel_workers=parallel_workers,
            submit_function=submit_single_chsh_circuit,
            progress_name="CHSH Submission",
        )

    def _submit_chsh_circuits_locally_parallel(
        self, circuits: list[Any], devices: list[str], shots: int, parallel_workers: int
    ) -> dict[str, list[dict]]:
        """Submit circuits to local simulator with parallel execution"""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        print(f"CHSH Local parallel execution: {parallel_workers} workers")

        all_job_data = {device: [None] * len(circuits) for device in devices}

        circuit_device_pairs = []
        for circuit_idx, circuit in enumerate(circuits):
            for device in devices:
                circuit_device_pairs.append((circuit_idx, circuit, device))

        def run_single_circuit_locally(args):
            circuit_idx, circuit, device = args
            try:
                result = self.run_circuit_locally(circuit, shots)
                if result:
                    job_id = result["job_id"]
                    if not hasattr(self, "_local_results"):
                        self._local_results = {}
                    self._local_results[job_id] = result
                    return device, job_id, circuit_idx, True
                else:
                    return device, None, circuit_idx, False
            except Exception as e:
                # Logging matching T1/Ramsey pattern
                phase_idx = circuit_idx // 4
                meas_idx = circuit_idx % 4
                print(
                    f"Local CHSH circuit {circuit_idx} (phase {phase_idx}, meas {meas_idx}) → {device}: {e}"
                )
                return device, None, circuit_idx, False

        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = [
                executor.submit(run_single_circuit_locally, args)
                for args in circuit_device_pairs
            ]

            for future in as_completed(futures):
                device, job_id, circuit_idx, success = future.result()
                if success and job_id:
                    all_job_data[device][circuit_idx] = {
                        "job_id": job_id,
                        "circuit_index": circuit_idx,
                        "submitted": True,
                    }
                else:
                    all_job_data[device][circuit_idx] = {
                        "job_id": None,
                        "circuit_index": circuit_idx,
                        "submitted": False,
                    }

        for device in devices:
            successful = sum(
                1 for job in all_job_data[device] if job and job["submitted"]
            )
            print(
                f"✅ {device}: {successful} CHSH circuits completed locally (order preserved)"
            )

        return all_job_data

    def _collect_chsh_results_parallel_with_order(
        self, job_data: dict[str, list[dict]], parallel_workers: int
    ) -> dict[str, list[dict]]:
        """Parallel collection of CHSH results (using ParallelExecutionMixin)"""

        # Use ParallelExecutionMixin for parallel execution
        def collect_single_chsh_result(device, job_info, circuit_idx):
            """Collect a single CHSH result"""
            job_id = job_info.get("job_id")
            shots = job_info.get("shots")

            try:
                result = self.get_job_result(job_id)
                if result and result.get("status") == "completed":
                    counts = result.get("counts")
                    if counts:
                        processed_result = {
                            "success": True,
                            "counts": dict(counts),
                            "status": result.get("status"),
                            "execution_time": result.get("execution_time", 0),
                            "shots": shots or sum(counts.values()) if counts else 0,
                        }

                        phase_idx = circuit_idx // 4
                        meas_idx = circuit_idx % 4
                        phase_phi = (
                            self.experiment_params["phase_range"][phase_idx]
                            if hasattr(self, "experiment_params")
                            and self.experiment_params
                            and phase_idx < len(self.experiment_params["phase_range"])
                            else 0
                        )
                        print(
                            f"✅ {device}[{circuit_idx}] (φ={phase_phi:.3f}, meas{meas_idx}): {job_id[:8]}... collected"
                        )
                        return processed_result
                    else:
                        print(
                            f"⚠️ {device}[{circuit_idx}]: {job_id[:8]}... no measurement data"
                        )
                        return None
                else:
                    status = result.get("status", "unknown") if result else "no_result"
                    print(f"⚠️ {device}[{circuit_idx}]: {job_id[:8]}... {status}")
                    return None
            except Exception as e:
                print(
                    f"❌ {device}[{circuit_idx}]: {job_id[:8]}... error: {str(e)[:50]}"
                )
                return None

        return self.collect_results_parallel_with_order(
            job_data=job_data,
            parallel_workers=parallel_workers,
            collect_function=collect_single_chsh_result,
            progress_name="CHSH Collection",
        )

        # Handle local results
        if hasattr(self, "_local_results"):
            print("Using cached local CHSH simulation results...")
            all_results = {}
            for device, device_job_data in job_data.items():
                device_results = []
                for job_info in device_job_data:
                    if (
                        job_info
                        and job_info["submitted"]
                        and job_info["job_id"] in self._local_results
                    ):
                        result = self._local_results[job_info["job_id"]]
                        device_results.append(result)
                    else:
                        device_results.append(None)
                all_results[device] = device_results
                successful = sum(1 for r in device_results if r is not None)
                print(f"✅ {device}: {successful} CHSH local results collected")
            return all_results

        if not self.oqtopus_available:
            print("OQTOPUS not available for CHSH collection")
            return {
                device: [None] * len(device_job_data)
                for device, device_job_data in job_data.items()
            }

        all_results = {
            device: [None] * len(device_job_data)
            for device, device_job_data in job_data.items()
        }

        job_collection_tasks = []
        for device, device_job_data in job_data.items():
            for circuit_idx, job_info in enumerate(device_job_data):
                if job_info and job_info["submitted"] and job_info["job_id"]:
                    job_collection_tasks.append(
                        (job_info["job_id"], device, circuit_idx)
                    )

        def collect_single_chsh_result(args):
            job_id, device, circuit_idx = args
            try:
                # Use BaseExperiment's get_oqtopus_result directly (no polling)
                result = self.get_oqtopus_result(job_id, timeout_minutes=5)

                # Debug information output (first 3 times only)
                if not hasattr(self, "_chsh_debug_count"):
                    self._chsh_debug_count = 0
                if self._chsh_debug_count < 3:
                    print(
                        f"🔍 CHSH Debug[{circuit_idx}]: Full result structure = {result}"
                    )
                    self._chsh_debug_count += 1

                # Flexible success determination (matching Ramsey and T1 patterns)
                success_conditions = [
                    result and result.get("status") == "succeeded",  # OQTOPUS standard
                    result and result.get("success", False),  # BaseExperiment legacy
                    result and "counts" in result,  # When counts are directly available
                ]

                if any(success_conditions):
                    # Try multiple methods to obtain measurement results
                    counts = None
                    shots = 0

                    # Method 1: When BaseExperiment's get_oqtopus_result returns counts directly
                    if "counts" in result:
                        counts = result["counts"]
                        shots = result.get("shots", 0)
                        print(f"🔍 CHSH[{circuit_idx}]: Direct counts found = {counts}")

                    # Method 2: Get from result structure within job_info
                    if not counts:
                        job_info = result.get("job_info", {})
                        if isinstance(job_info, dict):
                            # Explore OQTOPUS result structure
                            sampling_result = job_info.get("result", {}).get(
                                "sampling", {}
                            )
                            if sampling_result:
                                counts = sampling_result.get("counts", {})
                                print(
                                    f"🔍 CHSH[{circuit_idx}]: job_info.result.sampling counts = {counts}"
                                )

                    # Method 3: When job_info itself is in result format
                    if not counts and "job_info" in result:
                        job_info = result["job_info"]
                        if isinstance(job_info, dict) and "job_info" in job_info:
                            inner_job_info = job_info["job_info"]
                            if isinstance(inner_job_info, dict):
                                result_data = inner_job_info.get("result", {})
                                if "sampling" in result_data:
                                    counts = result_data["sampling"].get("counts", {})
                                    print(
                                        f"🔍 CHSH[{circuit_idx}]: nested job_info counts = {counts}"
                                    )
                                elif "counts" in result_data:
                                    counts = result_data["counts"]
                                    print(
                                        f"🔍 CHSH[{circuit_idx}]: nested result counts = {counts}"
                                    )

                    if counts:
                        # Convert successful data to standard format
                        processed_result = {
                            "success": True,
                            "counts": dict(counts),  # Convert Counter to dictionary
                            "status": result.get("status", "success"),
                            "execution_time": result.get("execution_time", 0),
                            "shots": shots or sum(counts.values()) if counts else 0,
                        }
                        print(
                            f"✅ CHSH[{circuit_idx}]: Processed successfully, counts={dict(counts)}"
                        )
                        return device, processed_result, job_id, circuit_idx, True
                    else:
                        print(
                            f"⚠️ {device}[{circuit_idx}]: {job_id[:8]}... no measurement data in any structure"
                        )
                        # Display part of result structure for debugging
                        if result:
                            print(f"   Available keys: {list(result.keys())}")
                        return device, None, job_id, circuit_idx, False
                else:
                    # When job failed
                    status = result.get("status", "unknown") if result else "no_result"
                    print(
                        f"⚠️ {device}[{circuit_idx}]: {job_id[:8]}... failed ({status})"
                    )
                    if result:
                        print(f"   Available keys: {list(result.keys())}")
                        print(f"   Success flag: {result.get('success', 'missing')}")
                    return device, None, job_id, circuit_idx, False
            except Exception as e:
                print(
                    f"❌ {device}[{circuit_idx}]: {job_id[:8]}... error: {str(e)[:50]}"
                )
                import traceback

                print(f"   Traceback: {traceback.format_exc()}")
                return device, None, job_id, circuit_idx, False

        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = [
                executor.submit(collect_single_chsh_result, args)
                for args in job_collection_tasks
            ]

            completed_jobs = 0
            successful_jobs = 0
            total_jobs = len(futures)
            last_progress_percent = 0

            for future in as_completed(futures):
                device, result, job_id, circuit_idx, success = future.result()
                completed_jobs += 1

                if success and result:
                    successful_jobs += 1
                    all_results[device][circuit_idx] = result
                    phase_idx = circuit_idx // 4
                    meas_idx = circuit_idx % 4
                    phase_phi = (
                        self.experiment_params["phase_range"][phase_idx]
                        if phase_idx < len(self.experiment_params["phase_range"])
                        else 0
                    )
                    print(
                        f"✅ {device}[{circuit_idx}] (φ={phase_phi:.3f}, meas{meas_idx}): {job_id[:8]}... collected ({completed_jobs}/{total_jobs})"
                    )
                else:
                    # Failure cases already logged in individual methods
                    pass

                # Display progress summary every 20%
                progress_percent = (completed_jobs * 100) // total_jobs
                if (
                    progress_percent >= last_progress_percent + 20
                    and progress_percent < 100
                ):
                    print(
                        f"📈 CHSH Collection Progress: {completed_jobs}/{total_jobs} ({progress_percent}%) - {successful_jobs} successful"
                    )
                    last_progress_percent = progress_percent

        # Final results summary
        total_successful = sum(
            1
            for device_results in all_results.values()
            for r in device_results
            if r is not None
        )
        total_attempted = sum(
            1
            for device_jobs in job_data.values()
            for job in device_jobs
            if job and job.get("submitted", False)
        )
        success_rate = (
            (total_successful / total_attempted * 100) if total_attempted > 0 else 0
        )

        print(
            f"🎉 CHSH Collection Complete: {total_successful}/{total_attempted} successful ({success_rate:.1f}%)"
        )

        for device in job_data.keys():
            successful = sum(1 for r in all_results[device] if r is not None)
            total = len(job_data[device])
            failed = total - successful

            if failed > 0:
                device_success_rate = (successful / total * 100) if total > 0 else 0
                print(
                    f"✅ {device}: {successful}/{total} CHSH results collected (success rate: {device_success_rate:.1f}%)"
                )
                print(
                    f"   ⚠️ {failed} jobs failed - analysis will continue with available data"
                )
            else:
                print(
                    f"✅ {device}: {successful}/{total} CHSH results collected (100% success)"
                )

        return all_results

    def _process_4_measurement_results(
        self,
        raw_results: dict[str, list[dict]],
        circuit_metadata: list[dict],
        phase_range,
        measurements: list[tuple],
        devices: list[str],
    ) -> dict[str, dict]:
        """Process 4-measurement CHSH results into structured format"""
        all_results = {}

        for device in devices:
            print(f"\nProcessing results for {device}...")

            device_results = raw_results.get(device, [])
            phase_points = len(phase_range)

            device_s_values = []
            device_expectations = []

            for phase_idx in range(phase_points):
                phase_range[phase_idx]
                phase_expectations = []

                for meas_idx in range(4):
                    circuit_idx = phase_idx * 4 + meas_idx

                    if (
                        circuit_idx < len(device_results)
                        and device_results[circuit_idx] is not None
                    ):
                        result = device_results[circuit_idx]
                        # Check if result was successful
                        if result and result.get("success", False):
                            counts = result.get("counts", {})
                            if counts:
                                expectation = self._calculate_expectation_value_oqtopus_compatible(
                                    counts
                                )
                                phase_expectations.append(expectation)

                                if phase_idx == 0 and meas_idx < 2:
                                    print(
                                        f"Debug - Phase {phase_idx}, Meas {meas_idx}: counts={counts}, exp={expectation:.3f}"
                                    )
                            else:
                                print(
                                    f"⚠️ Empty counts for circuit {circuit_idx} (phase {phase_idx}, meas {meas_idx})"
                                )
                                phase_expectations.append(0.0)
                        else:
                            print(
                                f"⚠️ Failed result for circuit {circuit_idx} (phase {phase_idx}, meas {meas_idx})"
                            )
                            phase_expectations.append(0.0)
                    else:
                        print(
                            f"⚠️ Missing result for circuit {circuit_idx} (phase {phase_idx}, meas {meas_idx})"
                        )
                        phase_expectations.append(0.0)

                # CHSH S value calculation: S = E1 + E2 + E3 - E4
                if len(phase_expectations) == 4:
                    E1, E2, E3, E4 = phase_expectations
                    S = E1 + E2 + E3 - E4
                else:
                    S = 0.0

                device_s_values.append(S)
                device_expectations.append(phase_expectations)

            all_results[device] = {
                "S_values": device_s_values,
                "expectations": device_expectations,
                "measurement_angles": {
                    "theta_a0": 0,
                    "theta_a1": np.pi / 2,
                    "theta_b0": np.pi / 4,
                    "theta_b1": -np.pi / 4,
                },
            }

            # Statistics
            S_array = np.array(device_s_values)
            max_S = np.max(np.abs(S_array))
            violations = int(np.sum(np.abs(S_array) > 2.0))
            print(
                f"{device}: Max |S| = {max_S:.3f}, Bell violations: {violations}/{phase_points}"
            )

        return all_results

    def _calculate_expectation_value_oqtopus_compatible(self, counts: dict) -> float:
        """Calculate CHSH expectation value compatible with OQTOPUS format (enhanced with Ramsey/T1 patterns)"""
        # Convert OQTOPUS decimal counts to binary format (using Ramsey and T1 patterns)
        binary_counts = self._convert_decimal_to_binary_counts_chsh(counts)

        total = sum(binary_counts.values())
        if total == 0:
            return 0.0

        # Display debug information (first time only)
        if not hasattr(self, "_chsh_counts_debug_shown"):
            print(f"🔍 CHSH Raw decimal counts: {dict(counts)}")
            print(f"🔍 CHSH Converted binary counts: {dict(binary_counts)}")
            self._chsh_counts_debug_shown = True

        # Calculate expectation value from 2-qubit measurement results
        n_00 = binary_counts.get("00", 0)
        n_11 = binary_counts.get("11", 0)
        n_01 = binary_counts.get("01", 0)
        n_10 = binary_counts.get("10", 0)

        # CHSH expectation value: E = (N_00 + N_11 - N_01 - N_10) / N_total
        expectation = (n_00 + n_11 - n_01 - n_10) / total
        return expectation

    def _convert_decimal_to_binary_counts_chsh(
        self, decimal_counts: dict[str, int]
    ) -> dict[str, int]:
        """
        Convert OQTOPUS decimal counts to binary format (for CHSH - 2 qubits)

        For 2-qubit case:
        0 -> "00"  (|00⟩ state)
        1 -> "01"  (|01⟩ state)
        2 -> "10"  (|10⟩ state)
        3 -> "11"  (|11⟩ state)
        """
        binary_counts = {}

        for decimal_key, count in decimal_counts.items():
            # Handle both numeric and string keys
            if isinstance(decimal_key, str):
                try:
                    decimal_value = int(decimal_key)
                except ValueError:
                    # Already in binary format
                    binary_counts[decimal_key] = count
                    continue
            else:
                decimal_value = int(decimal_key)

            # Conversion for 2-qubit case
            if decimal_value == 0:
                binary_key = "00"
            elif decimal_value == 1:
                binary_key = "01"
            elif decimal_value == 2:
                binary_key = "10"
            elif decimal_value == 3:
                binary_key = "11"
            else:
                # Skip unexpected values with warning
                print(
                    f"⚠️ Unexpected CHSH count key: {decimal_key} (decimal value: {decimal_value})"
                )
                continue

            # Add to existing key if present
            if binary_key in binary_counts:
                binary_counts[binary_key] += count
            else:
                binary_counts[binary_key] = count

        return binary_counts

    def _create_4_measurement_analysis(
        self, phase_range, all_results: dict[str, dict], angles: dict[str, float]
    ) -> dict[str, Any]:
        """Create comprehensive 4-measurement CHSH analysis"""
        analysis = {
            "experiment_info": {
                "theta_a0": angles["theta_a0"],
                "theta_a1": angles["theta_a1"],
                "theta_b0": angles["theta_b0"],
                "theta_b1": angles["theta_b1"],
                "phase_points": len(phase_range),
                "classical_bound": 2.0,
                "theoretical_max_s": 2 * np.sqrt(2),
            },
            "theoretical_values": {
                "phase_range": phase_range.tolist(),
                "S_theoretical": (2 * np.sqrt(2) * np.cos(phase_range)).tolist(),
            },
            "device_results": {},
        }

        for device, device_data in all_results.items():
            S_values = device_data["S_values"]
            S_array = np.array(S_values)
            bell_violations = int(np.sum(np.abs(S_array) > 2.0))
            max_S = float(np.max(np.abs(S_array)))

            analysis["device_results"][device] = {
                "S_values": S_values,
                "expectations": device_data["expectations"],
                "statistics": {
                    "max_S_magnitude": max_S,
                    "bell_violations": bell_violations,
                    "success_rate": 1.0,
                    "mean_S_magnitude": float(np.mean(np.abs(S_array))),
                },
            }

        return analysis

    def generate_chsh_plot(
        self, results: dict[str, Any], save_plot: bool = True, show_plot: bool = False
    ) -> str | None:
        """Generate CHSH experiment plot with all formatting"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available - skipping plot generation")
            return None

        phase_range = results.get("phase_range", np.linspace(0, 2 * np.pi, 20))
        device_results = results.get("device_results", {})

        if not device_results:
            print("No device results for plotting")
            return None

        theoretical_S = 2 * np.sqrt(2) * np.cos(phase_range)

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot experimental data for each device
        colors = ["blue", "red", "green", "orange", "purple"]

        for i, (device, device_data) in enumerate(device_results.items()):
            if "S_values" in device_data:
                S_values = device_data["S_values"]
                color = colors[i % len(colors)]
                ax.plot(
                    phase_range,
                    S_values,
                    "o-",
                    linewidth=2,
                    markersize=6,
                    label=f"{device} (quantumlib)",
                    alpha=0.8,
                    color=color,
                )

        # Plot theoretical curve
        ax.plot(
            phase_range,
            theoretical_S,
            "k-",
            linewidth=3,
            alpha=0.7,
            label="Theory: 2√2 cos(φ)",
        )

        # Bell inequality bounds
        ax.axhline(
            y=2.0,
            color="red",
            linestyle="--",
            alpha=0.7,
            linewidth=2,
            label="Classical bounds (±2)",
        )
        ax.axhline(y=-2.0, color="red", linestyle="--", alpha=0.7, linewidth=2)

        # Tsirelson bounds
        ax.axhline(
            y=2 * np.sqrt(2),
            color="green",
            linestyle=":",
            alpha=0.7,
            linewidth=2,
            label="Tsirelson bounds (±2√2)",
        )
        ax.axhline(
            y=-2 * np.sqrt(2), color="green", linestyle=":", alpha=0.7, linewidth=2
        )

        # Formatting
        ax.set_xlabel("Phase φ [rad]", fontsize=14)
        ax.set_ylabel("CHSH Parameter S", fontsize=14)
        ax.set_title(
            "OQTOPUS Experiments CHSH: 4-Measurement Bell Inequality Test",
            fontsize=16,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)

        # X-axis labels in π units
        ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        ax.set_xticklabels(["0", "π/2", "π", "3π/2", "2π"])

        plot_filename = None
        if save_plot:
            # Save plot in experiment results directory
            plt.tight_layout()
            plot_filename = f"chsh_plot_{self.experiment_name}_{int(time.time())}.png"

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
            except Exception:
                pass

        plt.close()
        return plot_filename

    def save_complete_experiment_data(self, results: dict[str, Any]) -> str:
        """Save experiment data and generate comprehensive report"""
        # Save main experiment data using existing system
        main_file = self.save_experiment_data(results["analysis"])

        # Generate and save plot
        plot_file = self.generate_chsh_plot(results, save_plot=True, show_plot=False)

        # Create experiment summary
        summary = self._create_experiment_summary(results)
        summary_file = self.data_manager.save_data(summary, "experiment_summary")

        print("📊 Complete experiment data saved:")
        print(f"  • Main results: {main_file}")
        print(f"  • Plot: {plot_file if plot_file else 'Not generated'}")
        print(f"  • Summary: {summary_file}")

        return main_file

    def _create_experiment_summary(self, results: dict[str, Any]) -> dict[str, Any]:
        """Create human-readable experiment summary"""
        device_results = results.get("device_results", {})
        phase_range = results.get("phase_range", [])

        summary = {
            "experiment_overview": {
                "experiment_name": self.experiment_name,
                "timestamp": time.time(),
                "method": results.get("method", "4_measurement_chsh"),
                "phase_points": len(phase_range),
                "devices_tested": list(device_results.keys()),
            },
            "key_results": {},
            "bell_inequality_analysis": {
                "classical_bound": 2.0,
                "quantum_bound": 2 * np.sqrt(2),
                "violations_detected": False,
            },
        }

        # Analyze each device
        total_violations = 0
        max_s_overall = 0

        for device, device_data in device_results.items():
            if "S_values" in device_data:
                S_values = device_data["S_values"]
                S_array = np.array(S_values)
                max_S = float(np.max(np.abs(S_array)))
                violations = int(np.sum(np.abs(S_array) > 2.0))

                summary["key_results"][device] = {
                    "max_S_magnitude": max_S,
                    "bell_violations_count": violations,
                    "bell_violations_percentage": violations / len(S_values) * 100,
                    "quantum_advantage": max_S > 2.0,
                }

                total_violations += violations
                max_s_overall = max(max_s_overall, max_S)

        summary["bell_inequality_analysis"]["violations_detected"] = (
            total_violations > 0
        )
        summary["bell_inequality_analysis"]["max_s_magnitude"] = max_s_overall
        summary["bell_inequality_analysis"]["total_violations"] = total_violations

        return summary

    def display_results(self, results: dict[str, Any], use_rich: bool = True) -> None:
        """Display CHSH experiment results in formatted table"""
        device_results = results.get("device_results", {})

        if not device_results:
            print("No device results found")
            return

        if use_rich:
            try:
                from rich.console import Console
                from rich.table import Table

                console = Console()
                table = Table(
                    title="CHSH Verification Results",
                    show_header=True,
                    header_style="bold blue",
                )
                table.add_column("Device", style="cyan")
                table.add_column("Max |S|", justify="right")
                table.add_column("Bell Violations", justify="right")
                table.add_column("Method", justify="center")
                table.add_column("Quantum Advantage", justify="center")

                method = results.get("method", "quantumlib_chsh")

                for device, device_data in device_results.items():
                    if "S_values" in device_data:
                        S_values = device_data["S_values"]
                        S_array = np.array(S_values)
                        max_S = float(np.max(np.abs(S_array)))
                        violations = int(np.sum(np.abs(S_array) > 2.0))
                        total_points = len(S_values)

                        advantage = "YES" if max_S > 2.0 else "NO"
                        advantage_style = "green" if max_S > 2.0 else "red"

                        table.add_row(
                            device.upper(),
                            f"{max_S:.3f}",
                            f"{violations}/{total_points}",
                            method,
                            advantage,
                            style=advantage_style if max_S > 2.0 else None,
                        )

                console.print(table)
                console.print(f"\nTheoretical maximum: {2 * np.sqrt(2):.3f}")
                console.print("Classical bound: ±2.0")

            except ImportError:
                use_rich = False

        if not use_rich:
            # Fallback to simple text display
            print("\n" + "=" * 60)
            print("CHSH Verification Results")
            print("=" * 60)

            method = results.get("method", "quantumlib_chsh")

            for device, device_data in device_results.items():
                if "S_values" in device_data:
                    S_values = device_data["S_values"]
                    S_array = np.array(S_values)
                    max_S = float(np.max(np.abs(S_array)))
                    violations = int(np.sum(np.abs(S_array) > 2.0))
                    total_points = len(S_values)

                    advantage = "YES" if max_S > 2.0 else "NO"

                    print(f"Device: {device.upper()}")
                    print(f"  Max |S|: {max_S:.3f}")
                    print(f"  Bell Violations: {violations}/{total_points}")
                    print(f"  Method: {method}")
                    print(f"  Quantum Advantage: {advantage}")
                    print()

            print(f"Theoretical maximum: {2 * np.sqrt(2):.3f}")
            print("Classical bound: ±2.0")
            print("=" * 60)

    def run_complete_chsh_experiment(
        self,
        devices: list[str] = ["qulacs"],
        phase_points: int = 20,
        shots: int = 1024,
        parallel_workers: int = 4,
        save_data: bool = True,
        save_plot: bool = True,
        show_plot: bool = False,
        display_results: bool = True,
    ) -> dict[str, Any]:
        """
        Run complete CHSH experiment with all post-processing
        This is the main entry point for CLI usage
        """
        print(f"🔬 Running complete CHSH experiment: {self.experiment_name}")
        print(f"   Devices: {devices}")
        print(f"   Phase points: {phase_points}, Shots: {shots}")
        print(f"   Parallel workers: {parallel_workers}")

        # Use standard BaseExperiment run_experiment method (like Ramsey/T1)
        results = self.run_experiment(
            devices=devices,
            shots=shots,
            parallel_workers=parallel_workers,
            points=phase_points,  # CHSH-specific parameter
        )

        # Save data if requested
        if save_data:
            self.save_complete_experiment_data(results)
        elif save_plot:
            # Just save plot without full data
            self.generate_chsh_plot(results, save_plot=True, show_plot=show_plot)

        # Display results if requested
        if display_results:
            self.display_results(results, use_rich=True)

        return results
