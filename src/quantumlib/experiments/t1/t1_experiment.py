#!/usr/bin/env python3
"""
T1 Experiment Class - T1 decay experiment specialized class
Inherits from BaseExperiment and provides T1 experiment-specific implementation
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np

from ...core.base_experiment import BaseExperiment
from ...core.parallel_execution import ParallelExecutionMixin


class T1Experiment(BaseExperiment, ParallelExecutionMixin):
    """
    T1 decay experiment class

    Specialized features:
    - Automatic T1 decay circuit generation
    - Exponential decay fitting
    - Delay time scan experiments
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

        super().__init__(experiment_name, **base_kwargs)

        # T1 experiment-specific settings (use experimental values only, theoretical values are for reference)
        self.expected_t1 = (
            1000  # Initial estimate [ns] - used only for fitting initial values
        )
        self.t1_theoretical = None  # Not used
        self.t2_theoretical = None  # Not used

        # Enable readout mitigation for T1 experiments (improve single-shot measurement accuracy)
        if disable_mitigation:
            self.mitigation_options = {}  # No mitigation
            print(
                "T1 experiment: Raw measurement data (mitigation disabled for debugging)"
            )
        else:
            self.mitigation_options = {"ro_error_mitigation": "pseudo_inverse"}
            print("T1 experiment: Standard T1 measurement with readout mitigation")
        self.mitigation_info = self.mitigation_options

    def create_circuits(self, **kwargs) -> list[Any]:
        """
        Create T1 experiment circuits

        Args:
            delay_points: Number of delay time points (default: 16)
            max_delay: Maximum delay time [ns] (default: 1000)
            t1: T1 relaxation time [ns] (default: 500)
            t2: T2 relaxation time [ns] (default: 500)
            delay_times: Directly specified delay time list [ns] (optional)

        Returns:
            T1 circuit list
        """
        delay_points = kwargs.get("delay_points", 51)
        max_delay = kwargs.get("max_delay", 100000)
        # t1, t2 parameters are not used (for fitting from measured data)

        # Delay time range
        if "delay_times" in kwargs:
            delay_times = np.array(kwargs["delay_times"])
        else:
            # Default: 51 points on logarithmic scale from 100ns to 100μs
            delay_times = np.logspace(np.log10(100), np.log10(100 * 1000), num=51)
            if delay_points != 51:
                delay_times = np.linspace(1, max_delay, delay_points)

        # Save metadata
        self.experiment_params = {
            "delay_times": delay_times.tolist(),
            "delay_points": len(delay_times),
            "max_delay": max_delay,
        }

        # Create T1 circuits (actual circuits don't need t1, t2 parameters)
        circuits = []
        for delay_time in delay_times:
            circuit = self._create_single_t1_circuit(delay_time)
            circuits.append(circuit)

        print(
            f"T1 circuits: Delay range {len(delay_times)} points from {delay_times[0]:.1f} to {delay_times[-1]:.1f} ns"
        )
        print(
            "T1 circuit structure: |0⟩ → X → delay(τ) → measure (expected: P(1) decreases with time)"
        )

        return circuits

    def run_t1_experiment_parallel(
        self,
        devices: list[str] = ["qulacs"],
        shots: int = 1024,
        parallel_workers: int = 4,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Parallel execution of T1 experiment (preserving delay time order)
        """
        print(f"🔬 Running T1 experiment with {parallel_workers} parallel workers")

        # Create circuits
        circuits = self.create_circuits(**kwargs)
        delay_times = self.experiment_params["delay_times"]

        print(
            f"   📊 {len(circuits)} circuits × {len(devices)} devices = {len(circuits) * len(devices)} jobs"
        )

        # Parallel execution (preserving order)
        job_data = self._submit_t1_circuits_parallel_with_order(
            circuits, devices, shots, parallel_workers
        )

        # Collect results (preserving order)
        raw_results = self._collect_t1_results_parallel_with_order(
            job_data, parallel_workers
        )

        # Analyze results
        analysis = self.analyze_results(raw_results)

        return {
            "delay_times": delay_times,
            "device_results": analysis["device_results"],
            "analysis": analysis,
            "method": "t1_parallel_quantumlib",
        }

    def _submit_t1_circuits_parallel_with_order(
        self, circuits: list[Any], devices: list[str], shots: int, parallel_workers: int
    ) -> dict[str, list[dict]]:
        """
        Parallel submission of T1 circuits (preserving order CHSH-style)
        """
        print(f"Enhanced T1 parallel submission: {parallel_workers} workers")

        if not self.oqtopus_available:
            return self._submit_t1_circuits_locally_parallel(
                circuits, devices, shots, parallel_workers
            )

        # Data structure for preserving order
        all_job_data = {device: [None] * len(circuits) for device in devices}

        # Create circuit and device pairs (preserving delay_time order)
        circuit_device_pairs = []
        for circuit_idx, circuit in enumerate(circuits):
            for device in devices:
                circuit_device_pairs.append((circuit_idx, circuit, device))

        def submit_single_t1_circuit(args):
            circuit_idx, circuit, device = args
            try:
                job_id = self.submit_circuit_to_oqtopus(circuit, shots, device)
                if job_id:
                    return device, job_id, circuit_idx, True
                else:
                    return device, None, circuit_idx, False
            except Exception as e:
                delay_time = self.experiment_params["delay_times"][circuit_idx]
                print(
                    f"T1 Circuit {circuit_idx} (τ={delay_time:.0f}ns) → {device}: {e}"
                )
                return device, None, circuit_idx, False

        # Execute parallel submission
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = [
                executor.submit(submit_single_t1_circuit, args)
                for args in circuit_device_pairs
            ]

            for future in as_completed(futures):
                device, job_id, circuit_idx, success = future.result()
                if success and job_id:
                    all_job_data[device][circuit_idx] = {
                        "job_id": job_id,
                        "circuit_index": circuit_idx,
                        "delay_time": self.experiment_params["delay_times"][
                            circuit_idx
                        ],
                        "submitted": True,
                    }
                    delay_time = self.experiment_params["delay_times"][circuit_idx]
                    print(
                        f"T1 Circuit {circuit_idx + 1} (τ={delay_time:.0f}ns) → {device}: {job_id[:8]}..."
                    )
                else:
                    all_job_data[device][circuit_idx] = {
                        "job_id": None,
                        "circuit_index": circuit_idx,
                        "delay_time": self.experiment_params["delay_times"][
                            circuit_idx
                        ],
                        "submitted": False,
                    }

        for device in devices:
            successful_jobs = sum(
                1
                for job_data in all_job_data[device]
                if job_data and job_data["submitted"]
            )
            print(f"✅ {device}: {successful_jobs} T1 jobs submitted (order preserved)")

        return all_job_data

    def _submit_t1_circuits_locally_parallel(
        self, circuits: list[Any], devices: list[str], shots: int, parallel_workers: int
    ) -> dict[str, list[dict]]:
        """Parallel execution of T1 circuits on local simulator"""
        print(f"T1 Local parallel execution: {parallel_workers} workers")

        all_job_data = {device: [None] * len(circuits) for device in devices}

        circuit_device_pairs = []
        for circuit_idx, circuit in enumerate(circuits):
            for device in devices:
                circuit_device_pairs.append((circuit_idx, circuit, device))

        def run_single_t1_circuit_locally(args):
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
                delay_time = self.experiment_params["delay_times"][circuit_idx]
                print(
                    f"Local T1 circuit {circuit_idx} (τ={delay_time:.0f}ns) → {device}: {e}"
                )
                return device, None, circuit_idx, False

        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = [
                executor.submit(run_single_t1_circuit_locally, args)
                for args in circuit_device_pairs
            ]

            for future in as_completed(futures):
                device, job_id, circuit_idx, success = future.result()
                if success and job_id:
                    all_job_data[device][circuit_idx] = {
                        "job_id": job_id,
                        "circuit_index": circuit_idx,
                        "delay_time": self.experiment_params["delay_times"][
                            circuit_idx
                        ],
                        "submitted": True,
                    }
                else:
                    all_job_data[device][circuit_idx] = {
                        "job_id": None,
                        "circuit_index": circuit_idx,
                        "delay_time": self.experiment_params["delay_times"][
                            circuit_idx
                        ],
                        "submitted": False,
                    }

        for device in devices:
            successful = sum(
                1 for job in all_job_data[device] if job and job["submitted"]
            )
            print(
                f"✅ {device}: {successful} T1 circuits completed locally (order preserved)"
            )

        return all_job_data

    def _collect_t1_results_parallel_with_order(
        self, job_data: dict[str, list[dict]], parallel_workers: int
    ) -> dict[str, list[dict]]:
        """Parallel collection of T1 results (preserving order CHSH-style)"""

        # Calculate total jobs and log collection start
        total_jobs_to_collect = sum(
            1
            for device_jobs in job_data.values()
            for job in device_jobs
            if job and job.get("submitted", False)
        )
        print(
            f"📊 Starting T1 results collection: {total_jobs_to_collect} jobs from {len(job_data)} devices"
        )

        # Handle local results
        if hasattr(self, "_local_results"):
            print("Using cached local T1 simulation results...")
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
                print(f"✅ {device}: {successful} T1 local results collected")
            return all_results

        if not self.oqtopus_available:
            print("OQTOPUS not available for T1 collection")
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

        def collect_single_t1_result(args):
            job_id, device, circuit_idx = args
            try:
                # Poll until job completion
                result = self._poll_job_until_completion(job_id, timeout_minutes=5)
                # Success determination based on OQTOPUS job structure: status == 'succeeded'
                if result and result.get("status") == "succeeded":
                    # Try multiple methods to obtain measurement results
                    counts = None
                    shots = 0

                    # Method 1: When BaseExperiment's get_oqtopus_result directly returns counts
                    if "counts" in result:
                        counts = result["counts"]
                        shots = result.get("shots", 0)

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

                    # Method 3: When job_info itself is in result format
                    if not counts and "job_info" in result:
                        job_info = result["job_info"]
                        if isinstance(job_info, dict) and "job_info" in job_info:
                            inner_job_info = job_info["job_info"]
                            if isinstance(inner_job_info, dict):
                                result_data = inner_job_info.get("result", {})
                                if "sampling" in result_data:
                                    counts = result_data["sampling"].get("counts", {})
                                elif "counts" in result_data:
                                    counts = result_data["counts"]

                    if counts:
                        # Debug: Confirm order and data
                        if not hasattr(self, "_sample_shown"):
                            delay_time = self.experiment_params["delay_times"][
                                circuit_idx
                            ]
                            total_counts = sum(counts.values())
                            p1_raw = (
                                counts.get("1", counts.get(1, 0)) / total_counts
                                if total_counts > 0
                                else 0
                            )
                            print(
                                f"🔍 Sample result [circuit_idx={circuit_idx}] for τ={delay_time:.0f}ns: counts={dict(counts)}, P(1)_raw={p1_raw:.3f}"
                            )
                            self._sample_shown = getattr(self, "_sample_shown", 0) + 1
                            if (
                                self._sample_shown >= 5
                            ):  # Display first 5 results to confirm order
                                self._sample_shown = True

                        # Convert successful data to standard format
                        processed_result = {
                            "success": True,
                            "counts": dict(counts),  # Convert Counter to dictionary
                            "status": result.get("status"),
                            "execution_time": result.get("execution_time", 0),
                            "shots": shots or sum(counts.values()) if counts else 0,
                        }
                        return device, processed_result, job_id, circuit_idx, True
                    else:
                        delay_time = self.experiment_params["delay_times"][circuit_idx]
                        # Debug: Display result structure in more detail
                        print(
                            f"⚠️ {device}[{circuit_idx}] (τ={delay_time:.0f}ns): {job_id[:8]}... no measurement data"
                        )
                        if hasattr(self, "_debug_count") and self._debug_count < 3:
                            print(f"   Debug - Full result: {result}")
                            self._debug_count = getattr(self, "_debug_count", 0) + 1
                        return device, None, job_id, circuit_idx, False
                else:
                    # Case of job failure
                    delay_time = self.experiment_params["delay_times"][circuit_idx]
                    status = result.get("status", "unknown") if result else "no_result"
                    print(
                        f"⚠️ {device}[{circuit_idx}] (τ={delay_time:.0f}ns): {job_id[:8]}... failed ({status})"
                    )
                    return device, None, job_id, circuit_idx, False
            except Exception as e:
                delay_time = self.experiment_params["delay_times"][circuit_idx]
                print(
                    f"❌ {device}[{circuit_idx}] (τ={delay_time:.0f}ns): {job_id[:8]}... error: {str(e)[:50]}"
                )
                return device, None, job_id, circuit_idx, False

        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = [
                executor.submit(collect_single_t1_result, args)
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
                    delay_time = self.experiment_params["delay_times"][circuit_idx]
                    print(
                        f"✅ {device}[{circuit_idx}] (τ={delay_time:.0f}ns): {job_id[:8]}... collected ({completed_jobs}/{total_jobs})"
                    )
                else:
                    # Failure cases are already logged within individual methods
                    pass

                # Display progress summary every 20%
                progress_percent = (completed_jobs * 100) // total_jobs
                if (
                    progress_percent >= last_progress_percent + 20
                    and progress_percent < 100
                ):
                    print(
                        f"📈 T1 Collection Progress: {completed_jobs}/{total_jobs} ({progress_percent}%) - {successful_jobs} successful"
                    )
                    last_progress_percent = progress_percent

        # Final result summary
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
            f"🎉 T1 Collection Complete: {total_successful}/{total_attempted} successful ({success_rate:.1f}%)"
        )

        for device in job_data.keys():
            successful = sum(1 for r in all_results[device] if r is not None)
            total = len(job_data[device])
            print(f"✅ {device}: {successful}/{total} T1 results collected")

        return all_results

    def _poll_job_until_completion(
        self, job_id: str, timeout_minutes: int = 5, poll_interval: float = 2.0
    ):
        """
        Poll until job completion

        Args:
            job_id: Job ID
            timeout_minutes: Timeout time (minutes)
            poll_interval: Polling interval (seconds)

        Returns:
            Result of completed job, or None
        """
        import time

        timeout_seconds = timeout_minutes * 60
        start_time = time.time()
        last_status = None

        while time.time() - start_time < timeout_seconds:
            try:
                print(
                    f"🔍 Polling {job_id[:8]}... (elapsed: {time.time() - start_time:.1f}s)"
                )
                result = self.get_oqtopus_result(
                    job_id, timeout_minutes=1, verbose_log=False
                )  # Get with short timeout

                # Simple status log only
                if not result:
                    continue

                if not result:
                    time.sleep(poll_interval)
                    continue

                status = result.get("status", "unknown")

                # Log only important state changes
                if status != last_status and status in [
                    "succeeded",
                    "failed",
                    "cancelled",
                ]:
                    print(f"🏁 {job_id[:8]}... {status}")
                    last_status = status

                # Check end state (flexible determination based on actual result structure)
                if status in ["succeeded", "failed", "cancelled"]:
                    print(f"🏁 Job {job_id[:8]} completed with status: {status}")
                    return result
                elif status in ["running", "submitted", "pending"]:
                    # Still running - continue
                    time.sleep(poll_interval)
                    continue
                elif result and result.get(
                    "success"
                ):  # Success flag returned by BaseExperiment's get_oqtopus_result
                    print(f"🏁 Job {job_id[:8]} completed successfully (legacy format)")
                    return result
                elif not status:  # Continue if status is not set
                    print(f"⚠️ Job {job_id[:8]} has no status field, continuing...")
                    time.sleep(poll_interval)
                    continue
                else:
                    # Unknown state - wait a bit and retry
                    print(
                        f"❓ Job {job_id[:8]} unknown status: {status}, continuing..."
                    )
                    time.sleep(poll_interval)
                    continue

            except Exception as e:
                # Retry for temporary errors
                print(f"⚠️ Polling error for {job_id[:8]}: {e}")
                time.sleep(poll_interval)
                continue

        # Timeout
        print(f"⏰ Job {job_id[:8]}... timed out after {timeout_minutes} minutes")
        return None

    def _collect_single_t1_result(
        self, device: str, job_id: str, circuit_idx: int
    ) -> tuple:
        """
        Collection of single T1 result
        """
        try:
            result = self.get_oqtopus_result(job_id, wait_minutes=10)
            return device, result, job_id, circuit_idx, True
        except Exception as e:
            delay_time = self.experiment_params["delay_times"][circuit_idx]
            print(
                f"   ❌ {device}[{circuit_idx}] τ={delay_time:.0f}ns: Collection failed - {e}"
            )
            return device, None, job_id, circuit_idx, False

    def run_experiment(
        self,
        devices: list[str] = ["qulacs"],
        shots: int = 1024,
        parallel_workers: int = 4,
        **kwargs,
    ) -> dict[str, Any]:
        """
        T1 experiment execution (parallelized version calling run_t1_experiment_parallel)
        """
        return self.run_t1_experiment_parallel(
            devices=devices, shots=shots, parallel_workers=parallel_workers, **kwargs
        )

    def _create_single_t1_circuit(self, delay_time: float):
        """
        Create single T1 circuit (t1, t2 parameters not required)
        """
        try:
            from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
        except ImportError:
            raise ImportError("Qiskit is required for circuit creation") from None

        # 1 qubit + 1 classical bit
        qubits = QuantumRegister(1, "q")
        bits = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qubits, bits)

        # Excite to |1⟩ state
        qc.x(0)

        # Wait for delay time
        qc.delay(int(delay_time), 0, unit="ns")

        # Z-basis measurement
        qc.measure(0, 0)

        return qc

    def analyze_results(
        self, results: dict[str, list[dict[str, Any]]], **kwargs
    ) -> dict[str, Any]:
        """
        T1 experiment result analysis

        Args:
            results: Raw measurement results

        Returns:
            T1 analysis results
        """
        if not results:
            return {"error": "No results to analyze"}

        delay_times = np.array(self.experiment_params["delay_times"])

        analysis = {
            "experiment_info": {
                "delay_points": len(delay_times),
                "expected_t1": self.expected_t1,
            },
            "device_results": {},
        }

        for device, device_results in results.items():
            if not device_results:
                continue

            device_analysis = self._analyze_device_results(device_results, delay_times)
            analysis["device_results"][device] = device_analysis

            # T1 time constant estimation (using readout mitigation corrected data)
            t1_fitted, fitting_quality = self._estimate_t1_with_quality(
                device_analysis["p1_values"], delay_times
            )

            analysis["device_results"][device]["t1_fitted"] = t1_fitted
            analysis["device_results"][device]["fitting_quality"] = fitting_quality

            quality_str = (
                f"({fitting_quality['method']}, R²={fitting_quality['r_squared']:.3f})"
            )
            print(
                f"{device}: T1 = {t1_fitted:.1f} ns {quality_str} [with RO mitigation]"
            )

            # Check first and last P(1) values
            if device_analysis["p1_values"]:
                p1_initial = device_analysis["p1_values"][0]
                p1_final = device_analysis["p1_values"][-1]
                print(
                    f"   P(1) trend: {p1_initial:.3f} → {p1_final:.3f} ({'decreasing' if p1_final < p1_initial else 'INCREASING - CHECK DATA!'})"
                )

        # Inter-device comparison
        analysis["comparison"] = self._compare_devices(analysis["device_results"])

        return analysis

    def _analyze_device_results(
        self, device_results: list[dict[str, Any]], delay_times: np.ndarray
    ) -> dict[str, Any]:
        """
        Single device result analysis (with order debugging)
        """
        print(f"🔍 Analyzing {len(device_results)} results in order...")

        p1_values = []

        for i, result in enumerate(device_results):
            delay_time = delay_times[i] if i < len(delay_times) else f"unknown[{i}]"

            if result and result["success"]:
                counts = result["counts"]

                # P(1) probability calculation (corrected by readout mitigation)
                p1 = self._calculate_p1_probability(counts)
                p1_values.append(p1)

                # Order debugging with first 5 points
                if i < 5:
                    print(
                        f"🔍 Point {i}: τ={delay_time}ns, P(1)={p1:.3f}, counts={dict(counts)}"
                    )
            else:
                p1_values.append(np.nan)
                if i < 5:
                    print(f"🔍 Point {i}: τ={delay_time}ns, FAILED")

        # Summary for order confirmation
        valid_p1s = np.array([p for p in p1_values if not np.isnan(p)])
        if len(valid_p1s) >= 2:
            trend = "decreasing" if valid_p1s[-1] < valid_p1s[0] else "increasing"
            print(
                f"📈 T1 trend: P(1) {valid_p1s[0]:.3f} → {valid_p1s[-1]:.3f} ({trend})"
            )

        # Statistical calculation

        return {
            "p1_values": p1_values,
            "delay_times": delay_times.tolist(),
            "statistics": {
                "initial_p1": (
                    float(p1_values[0])
                    if len(p1_values) > 0 and not np.isnan(p1_values[0])
                    else 1.0
                ),
                "final_p1": (
                    float(p1_values[-1])
                    if len(p1_values) > 0 and not np.isnan(p1_values[-1])
                    else 0.0
                ),
                "success_rate": len(valid_p1s) / len(p1_values) if p1_values else 0,
                "decay_observed": (
                    float(p1_values[0] - p1_values[-1])
                    if len(p1_values) >= 2
                    and not any(np.isnan([p1_values[0], p1_values[-1]]))
                    else 0.0
                ),
            },
        }

    def _calculate_p1_probability(self, counts: dict[str, int]) -> float:
        """
        P(1) probability calculation (converting OQTOPUS decimal counts to binary)
        """
        # Convert decimal counts from OQTOPUS to binary format
        binary_counts = self._convert_decimal_to_binary_counts(counts)

        total = sum(binary_counts.values())
        if total == 0:
            return 0.0

        # Display debug information (first time only)
        if not hasattr(self, "_counts_debug_shown"):
            print(f"🔍 Raw decimal counts: {dict(counts)}")
            print(f"🔍 Converted binary counts: {dict(binary_counts)}")
            self._counts_debug_shown = True

        # Standard P(1) probability calculation
        n_1 = binary_counts.get("1", 0)
        p1 = n_1 / total
        return p1

    def _convert_decimal_to_binary_counts(
        self, decimal_counts: dict[str, int]
    ) -> dict[str, int]:
        """
        Convert OQTOPUS decimal counts to binary format

        For 1 qubit:
        0 -> "0"  (|0⟩ state)
        1 -> "1"  (|1⟩ state)
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

            # Conversion for 1-qubit case
            if decimal_value == 0:
                binary_key = "0"
            elif decimal_value == 1:
                binary_key = "1"
            else:
                # Skip unexpected values with warning
                print(
                    f"⚠️ Unexpected count key: {decimal_key} (decimal value: {decimal_value})"
                )
                continue

            # Add to existing key if present
            if binary_key in binary_counts:
                binary_counts[binary_key] += count
            else:
                binary_counts[binary_key] = count

        return binary_counts

    def _calculate_z_expectation(self, counts: dict[str, int]) -> float:
        """
        Calculate <Z> expectation value (readout error resistant)
        """
        total = sum(counts.values())
        if total == 0:
            return 0.0

        # Get counts
        if isinstance(list(counts.keys())[0], str):
            n_0 = counts.get("0", 0)
            n_1 = counts.get("1", 0)
        else:
            n_0 = counts.get(0, 0)
            n_1 = counts.get(1, 0)

        # <Z> = P(0) - P(1) = (n_0 - n_1) / total
        z_expectation = (n_0 - n_1) / total
        return z_expectation

    def _estimate_t1(self, p1_values: list[float], delay_times: np.ndarray) -> float:
        """
        T1 time constant estimation (improved exponential decay fitting)
        """
        # Remove NaN and non-positive values
        valid_data = [
            (delay, p1)
            for delay, p1 in zip(delay_times, p1_values, strict=False)
            if not np.isnan(p1) and p1 > 0
        ]

        if len(valid_data) < 3:
            return 0.0

        delays = np.array([d[0] for d in valid_data])
        p1s = np.array([d[1] for d in valid_data])

        try:
            # Try nonlinear fitting: P(t) = A * exp(-t/T1) + offset
            from scipy.optimize import curve_fit

            def exponential_decay(t, A, T1, offset):
                return A * np.exp(-t / T1) + offset

            # Initial estimates
            p0 = [p1s[0], self.expected_t1, 0.0]  # A, T1, offset

            # Execute fitting
            popt, pcov = curve_fit(
                exponential_decay,
                delays,
                p1s,
                p0=p0,
                bounds=([0, 10, -0.1], [2.0, 10000, 0.1]),
                maxfev=2000,
            )

            t1_fitted = popt[1]

            # Check fitting quality
            if pcov is not None and np.all(np.isfinite(pcov)):
                # Calculate standard error from diagonal elements
                param_errors = np.sqrt(np.diag(pcov))
                t1_error = param_errors[1]

                # Accept only when relative error is 50% or less
                if t1_error / t1_fitted < 0.5:
                    return float(t1_fitted)

        except (ImportError, RuntimeError, ValueError, TypeError):
            # Fallback to linear regression when scipy is unavailable or fitting fails
            pass

        try:
            # Fallback: fitting using linear regression
            log_p1s = np.log(p1s)

            # Linear fitting
            coeffs = np.polyfit(delays, log_p1s, 1)
            slope = coeffs[0]

            # T1 = -1/slope
            t1_fitted = -1.0 / slope if slope < 0 else float("inf")

            # 合理的な範囲に制限
            if t1_fitted < 10 or t1_fitted > 10000:
                t1_fitted = self.expected_t1

        except (ValueError, np.linalg.LinAlgError):
            # フィッティングが完全に失敗した場合はデフォルト値
            t1_fitted = self.expected_t1

        return float(t1_fitted)

    def _estimate_t1_with_quality(
        self, p1_values: list[float], delay_times: np.ndarray
    ) -> tuple[float, dict[str, Any]]:
        """
        T1時定数推定と品質評価
        """
        # NaNと非正値を除去
        valid_data = [
            (delay, p1)
            for delay, p1 in zip(delay_times, p1_values, strict=False)
            if not np.isnan(p1) and p1 > 0
        ]

        if len(valid_data) < 3:
            return 0.0, {
                "method": "insufficient_data",
                "r_squared": 0.0,
                "error": "inf",
            }

        delays = np.array([d[0] for d in valid_data])
        p1s = np.array([d[1] for d in valid_data])

        # 非線形フィッティングを試行
        try:
            from scipy.optimize import curve_fit

            def exponential_decay(t, A, T1, offset):
                return A * np.exp(-t / T1) + offset

            # Initial estimates
            p0 = [p1s[0], self.expected_t1, 0.0]

            # Execute fitting
            popt, pcov = curve_fit(
                exponential_decay,
                delays,
                p1s,
                p0=p0,
                bounds=([0, 10, -0.1], [2.0, 10000, 0.1]),
                maxfev=2000,
            )

            t1_fitted = popt[1]

            # 予測値計算とR²算出
            p1_pred = exponential_decay(delays, *popt)
            ss_res = np.sum((p1s - p1_pred) ** 2)
            ss_tot = np.sum((p1s - np.mean(p1s)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            # 標準誤差計算
            param_error = "inf"
            if pcov is not None and np.all(np.isfinite(pcov)):
                param_errors = np.sqrt(np.diag(pcov))
                t1_error = param_errors[1]
                param_error = f"{t1_error:.1f}"

                # 高品質フィッティングの条件
                if t1_error / t1_fitted < 0.5 and r_squared > 0.7:
                    return float(t1_fitted), {
                        "method": "nonlinear",
                        "r_squared": r_squared,
                        "error": param_error,
                        "quality": "high" if r_squared > 0.9 else "medium",
                    }

        except (ImportError, RuntimeError, ValueError, TypeError):
            pass

        # フォールバック: 線形回帰
        try:
            log_p1s = np.log(p1s)
            coeffs = np.polyfit(delays, log_p1s, 1)
            slope, intercept = coeffs[0], coeffs[1]

            t1_fitted = -1.0 / slope if slope < 0 else self.expected_t1

            # 線形回帰のR²計算
            log_p1_pred = slope * delays + intercept
            ss_res = np.sum((log_p1s - log_p1_pred) ** 2)
            ss_tot = np.sum((log_p1s - np.mean(log_p1s)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            if 10 <= t1_fitted <= 10000:
                return float(t1_fitted), {
                    "method": "linear",
                    "r_squared": r_squared,
                    "error": "N/A",
                    "quality": "medium" if r_squared > 0.7 else "low",
                }

        except (ValueError, np.linalg.LinAlgError):
            pass

        # 全ての手法が失敗した場合
        return float(self.expected_t1), {
            "method": "default",
            "r_squared": 0.0,
            "error": "N/A",
            "quality": "poor",
        }

    def _estimate_t1_from_z_expectation(
        self, z_values: list[float], delay_times: np.ndarray
    ) -> tuple[float, dict[str, Any]]:
        """
        <Z>期待値からT1時定数推定（readout error耐性）
        理論: <Z>(t) = -exp(-t/T1) (|1⟩状態から開始)
        """
        # NaNを除去
        valid_data = [
            (delay, z)
            for delay, z in zip(delay_times, z_values, strict=False)
            if not np.isnan(z)
        ]

        if len(valid_data) < 3:
            return 0.0, {
                "method": "insufficient_data",
                "r_squared": 0.0,
                "error": "inf",
            }

        delays = np.array([d[0] for d in valid_data])
        z_vals = np.array([d[1] for d in valid_data])

        # 非線形フィッティングを試行: <Z>(t) = A * exp(-t/T1) + offset
        try:
            from scipy.optimize import curve_fit

            def z_exponential_decay(t, A, T1, offset):
                return A * np.exp(-t / T1) + offset

            # Initial estimates: A≈-1 (|1⟩から開始), T1≈expected, offset≈0
            p0 = [z_vals[0], self.expected_t1, 0.0]

            # Execute fitting
            popt, pcov = curve_fit(
                z_exponential_decay,
                delays,
                z_vals,
                p0=p0,
                bounds=([-2.0, 10, -0.1], [0.0, 50000, 0.1]),
                maxfev=2000,
            )

            t1_fitted = popt[1]

            # 予測値計算とR²算出
            z_pred = z_exponential_decay(delays, *popt)
            ss_res = np.sum((z_vals - z_pred) ** 2)
            ss_tot = np.sum((z_vals - np.mean(z_vals)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

            # 標準誤差計算
            param_error = "inf"
            if pcov is not None and np.all(np.isfinite(pcov)):
                param_errors = np.sqrt(np.diag(pcov))
                t1_error = param_errors[1]
                param_error = f"{t1_error:.1f}"

                # 高品質フィッティングの条件
                if t1_error / t1_fitted < 0.5 and r_squared > 0.7:
                    return float(t1_fitted), {
                        "method": "nonlinear_z",
                        "r_squared": r_squared,
                        "error": param_error,
                        "quality": "high" if r_squared > 0.9 else "medium",
                    }

        except (ImportError, RuntimeError, ValueError, TypeError):
            pass

        # フォールバック: 線形回帰 (log(-<Z>) vs t)
        try:
            # <Z>が負の値のみ使用（|1⟩状態なので）
            negative_z_data = [
                (delay, -z) for delay, z in zip(delays, z_vals, strict=False) if z < 0
            ]

            if len(negative_z_data) >= 3:
                delays_neg = np.array([d[0] for d in negative_z_data])
                neg_z_vals = np.array([d[1] for d in negative_z_data])

                log_neg_z = np.log(neg_z_vals)
                coeffs = np.polyfit(delays_neg, log_neg_z, 1)
                slope, intercept = coeffs[0], coeffs[1]

                t1_fitted = -1.0 / slope if slope < 0 else self.expected_t1

                # 線形回帰のR²計算
                log_pred = slope * delays_neg + intercept
                ss_res = np.sum((log_neg_z - log_pred) ** 2)
                ss_tot = np.sum((log_neg_z - np.mean(log_neg_z)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

                if 10 <= t1_fitted <= 50000:
                    return float(t1_fitted), {
                        "method": "linear_z",
                        "r_squared": r_squared,
                        "error": "N/A",
                        "quality": "medium" if r_squared > 0.7 else "low",
                    }

        except (ValueError, np.linalg.LinAlgError):
            pass

        # 全ての手法が失敗した場合
        return float(self.expected_t1), {
            "method": "default_z",
            "r_squared": 0.0,
            "error": "N/A",
            "quality": "poor",
        }

    def _compare_devices(
        self, device_results: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """
        デバイス間比較分析
        """
        if len(device_results) < 2:
            return {"note": "Multiple devices required for comparison"}

        comparison = {
            "device_count": len(device_results),
            "t1_comparison": {},
            "decay_comparison": {},
        }

        for device, analysis in device_results.items():
            stats = analysis["statistics"]
            comparison["t1_comparison"][device] = analysis.get("t1_fitted", 0.0)
            comparison["decay_comparison"][device] = stats["decay_observed"]

        return comparison

    def save_experiment_data(
        self, results: dict[str, Any], metadata: dict[str, Any] = None
    ) -> str:
        """
        T1実験データ保存
        """
        # T1実験専用の保存形式
        t1_data = {
            "experiment_type": "T1_Decay",
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

        # メイン結果保存
        main_file = self.data_manager.save_data(t1_data, "t1_experiment_results")

        # 追加ファイル保存
        if "device_results" in results:
            # デバイス別サマリー
            device_summary = {
                device: {
                    "t1_fitted": analysis.get("t1_fitted", 0.0),
                    "statistics": analysis["statistics"],
                }
                for device, analysis in results["device_results"].items()
            }
            self.data_manager.save_data(device_summary, "device_t1_summary")

            # P(1)データ（プロット用）
            p1_data = {
                "delay_times": self.experiment_params["delay_times"],
                "device_p1_values": {
                    device: analysis["p1_values"]
                    for device, analysis in results["device_results"].items()
                },
            }
            self.data_manager.save_data(p1_data, "p1_values_for_plotting")

        return main_file

    def generate_t1_plot(
        self, results: dict[str, Any], save_plot: bool = True, show_plot: bool = False
    ) -> str | None:
        """Generate T1 experiment plot with all formatting"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available - skipping plot generation")
            return None

        delay_times = results.get("delay_times", np.linspace(1, 1000, 16))
        device_results = results.get("device_results", {})

        if not device_results:
            print("No device results for plotting")
            return None

        # 理論値は使用しない

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot experimental data for each device
        colors = ["blue", "red", "green", "orange", "purple"]

        for i, (device, device_data) in enumerate(device_results.items()):
            if "p1_values" in device_data:
                p1_values = device_data["p1_values"]
                t1_fitted = device_data.get("t1_fitted", 0.0)
                fitting_quality = device_data.get("fitting_quality", {})
                r_squared = fitting_quality.get("r_squared", 0.0)
                color = colors[i % len(colors)]

                # 実測データプロット
                ax.semilogx(
                    delay_times,
                    p1_values,
                    "o",
                    markersize=6,
                    label=f"{device} data",
                    alpha=0.8,
                    color=color,
                )

                # フィット曲線プロット
                if t1_fitted > 0:
                    # フィットされた指数減衰曲線を描画
                    fit_delays = np.logspace(
                        np.log10(min(delay_times)), np.log10(max(delay_times)), 100
                    )
                    # 簡単な指数減衰: P(t) = P0 * exp(-t/T1)
                    p0_estimate = max(p1_values) if p1_values else 1.0
                    fit_curve = p0_estimate * np.exp(-fit_delays / t1_fitted)
                    ax.semilogx(
                        fit_delays,
                        fit_curve,
                        "-",
                        linewidth=2,
                        color=color,
                        alpha=0.7,
                        label=f"{device} fit (T1={t1_fitted:.0f}ns, R²={r_squared:.3f})",
                    )

        # 理論曲線は削除（実測データとフィットのみ表示）

        # Formatting
        ax.set_xlabel("Delay time τ [ns] (log scale)", fontsize=14)
        ax.set_ylabel("P(1)", fontsize=14)
        ax.set_title(
            "QuantumLib T1 Decay Experiment",
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
            plot_filename = f"t1_plot_{self.experiment_name}_{int(time.time())}.png"

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
        plot_file = self.generate_t1_plot(results, save_plot=True, show_plot=False)

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
        delay_times = results.get("delay_times", [])

        summary = {
            "experiment_overview": {
                "experiment_name": self.experiment_name,
                "timestamp": time.time(),
                "method": results.get("method", "t1_decay"),
                "delay_points": len(delay_times),
                "devices_tested": list(device_results.keys()),
            },
            "key_results": {},
            "t1_analysis": {
                "expected_t1": self.expected_t1,
                "clear_decay_detected": False,
            },
        }

        # Analyze each device
        min_decay_threshold = 0.3  # Minimum decay to consider significant

        for device, device_data in device_results.items():
            if "p1_values" in device_data:
                p1_values = device_data["p1_values"]
                valid_p1s = [p for p in p1_values if not np.isnan(p)]

                if valid_p1s and len(valid_p1s) >= 2:
                    initial_p1 = valid_p1s[0]
                    final_p1 = valid_p1s[-1]
                    decay = initial_p1 - final_p1
                    t1_fitted = device_data.get("t1_fitted", 0.0)

                    summary["key_results"][device] = {
                        "initial_p1": initial_p1,
                        "final_p1": final_p1,
                        "decay_observed": decay,
                        "t1_fitted": t1_fitted,
                        "clear_decay": decay > min_decay_threshold,
                    }

                    if decay > min_decay_threshold:
                        summary["t1_analysis"]["clear_decay_detected"] = True

        return summary

    def display_results(self, results: dict[str, Any], use_rich: bool = True) -> None:
        """Display T1 experiment results in formatted table"""
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
                    title="T1 Decay Results", show_header=True, header_style="bold blue"
                )
                table.add_column("Device", style="cyan")
                table.add_column("T1 Fitted [ns]", justify="right")
                table.add_column("Initial P(1)", justify="right")
                table.add_column("Final P(1)", justify="right")
                table.add_column("Decay", justify="right")
                table.add_column("Clear Decay", justify="center")

                results.get("method", "quantumlib_t1")

                for device, device_data in device_results.items():
                    if "p1_values" in device_data:
                        p1_values = device_data["p1_values"]
                        valid_p1s = [p for p in p1_values if not np.isnan(p)]

                        if valid_p1s and len(valid_p1s) >= 2:
                            initial_p1 = valid_p1s[0]
                            final_p1 = valid_p1s[-1]
                            decay = initial_p1 - final_p1
                            t1_fitted = device_data.get("t1_fitted", 0.0)

                            clear_decay = "YES" if decay > 0.3 else "NO"
                            decay_style = "green" if decay > 0.3 else "yellow"

                            table.add_row(
                                device.upper(),
                                f"{t1_fitted:.1f}",
                                f"{initial_p1:.3f}",
                                f"{final_p1:.3f}",
                                f"{decay:.3f}",
                                clear_decay,
                                style=decay_style if decay > 0.3 else None,
                            )

                console.print(table)
                console.print(f"\nExpected T1: {self.expected_t1} ns")
                console.print("Clear decay threshold: 0.3")

            except ImportError:
                use_rich = False

        if not use_rich:
            # Fallback to simple text display
            print("\n" + "=" * 60)
            print("T1 Decay Results")
            print("=" * 60)

            results.get("method", "quantumlib_t1")

            for device, device_data in device_results.items():
                if "p1_values" in device_data:
                    p1_values = device_data["p1_values"]
                    valid_p1s = [p for p in p1_values if not np.isnan(p)]

                    if valid_p1s and len(valid_p1s) >= 2:
                        initial_p1 = valid_p1s[0]
                        final_p1 = valid_p1s[-1]
                        decay = initial_p1 - final_p1
                        t1_fitted = device_data.get("t1_fitted", 0.0)

                        clear_decay = "YES" if decay > 0.3 else "NO"

                        print(f"Device: {device.upper()}")
                        print(f"  T1 Fitted: {t1_fitted:.1f} ns")
                        print(f"  Initial P(1): {initial_p1:.3f}")
                        print(f"  Final P(1): {final_p1:.3f}")
                        print(f"  Decay: {decay:.3f}")
                        print(f"  Clear Decay: {clear_decay}")
                        print()

            print(f"Expected T1: {self.expected_t1} ns")
            print("Clear decay threshold: 0.3")
            print("=" * 60)

    def run_complete_t1_experiment(
        self,
        devices: list[str] = ["qulacs"],
        delay_points: int = 16,
        max_delay: float = 1000,
        t1: float = 500,
        t2: float = 500,
        shots: int = 1024,
        parallel_workers: int = 4,
        save_data: bool = True,
        save_plot: bool = True,
        show_plot: bool = False,
        display_results: bool = True,
    ) -> dict[str, Any]:
        """
        Run complete T1 experiment with all post-processing
        This is the main entry point for CLI usage
        """
        print(f"🔬 Running complete T1 experiment: {self.experiment_name}")
        print(f"   Devices: {devices}")
        print(f"   Delay points: {delay_points}, Max delay: {max_delay} ns")
        print(f"   T1: {t1} ns, T2: {t2} ns")
        print(f"   Shots: {shots}, Parallel workers: {parallel_workers}")

        # Run the T1 experiment
        results = self.run_experiment(
            devices=devices,
            shots=shots,
            delay_points=delay_points,
            max_delay=max_delay,
            t1=t1,
            t2=t2,
        )

        # Save data if requested
        if save_data:
            self.save_complete_experiment_data(results)
        elif save_plot:
            # Just save plot without full data
            self.generate_t1_plot(results, save_plot=True, show_plot=show_plot)

        # Display results if requested
        if display_results:
            self.display_results(results, use_rich=True)

        return results
