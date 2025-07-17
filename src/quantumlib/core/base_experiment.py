#!/usr/bin/env python3
"""
Base Experiment Class - Base class for experiments
Base class for all quantum experiment classes
"""

import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from .data_manager import SimpleDataManager

# OQTOPUS imports
try:
    from qiskit.qasm3 import dumps
    from quri_parts_oqtopus.backend import OqtopusSamplingBackend

    OQTOPUS_AVAILABLE = True
except ImportError:
    OQTOPUS_AVAILABLE = False


class BaseExperiment(ABC):
    """
    Base class for quantum experiments

    All concrete experiment classes (CHSHExperiment, etc.) inherit from this
    Common functionality: OQTOPUS connection, parallel execution, data management
    """

    def __init__(
        self,
        experiment_name: str | None = None,
        oqtopus_backend: OqtopusSamplingBackend | None = None,
    ):
        """
        Initialize base experiment

        Args:
            experiment_name: Experiment name
            oqtopus_backend: OQTOPUS backend (auto-created if omitted)
        """
        self.experiment_name = (
            experiment_name or f"{self.__class__.__name__.lower()}_{int(time.time())}"
        )
        self.data_manager = SimpleDataManager(self.experiment_name)

        # OQTOPUS backend configuration
        if oqtopus_backend:
            self.oqtopus_backend = oqtopus_backend
            self.oqtopus_available = True
        else:
            self.oqtopus_available = OQTOPUS_AVAILABLE
            if OQTOPUS_AVAILABLE:
                self.oqtopus_backend = OqtopusSamplingBackend()

        # Local simulator configuration
        self.local_simulator = None
        try:
            from qiskit_aer import AerSimulator

            self.local_simulator = AerSimulator()
            self.local_simulator_available = True
        except ImportError:
            self.local_simulator_available = False

        # Default OQTOPUS settings
        self.anemone_basis_gates = ["sx", "x", "rz", "cx"]
        self.transpiler_options = {
            "basis_gates": self.anemone_basis_gates,
            "optimization_level": 1,
        }
        self.mitigation_options = {
            "ro_error_mitigation": "pseudo_inverse",
        }

        # Internal structure for OQTOPUS
        self.transpiler_info = {
            "transpiler_lib": "qiskit",
            "transpiler_options": self.transpiler_options,
        }
        self.mitigation_info = self.mitigation_options

        print(f"{self.__class__.__name__}: {self.experiment_name}")
        print(f"OQTOPUS: {'Available' if self.oqtopus_available else 'Not available'}")
        print(
            f"Local Simulator: {'Available' if self.local_simulator_available else 'Not available'}"
        )

    def submit_circuit_to_oqtopus(
        self, circuit: Any, shots: int, device_id: str
    ) -> str | None:
        """
        Submit single circuit to OQTOPUS
        """
        if not self.oqtopus_available:
            print("OQTOPUS not available")
            return None

        try:
            # Generate QASM3
            qasm_str = dumps(circuit)
            f"circuit_{int(time.time())}"

            # Dynamic configuration update
            self.transpiler_info["transpiler_options"] = self.transpiler_options
            self.mitigation_info = self.mitigation_options

            job = self.oqtopus_backend.sample_qasm(
                qasm_str,
                device_id=device_id,
                shots=shots,
                transpiler_info=self.transpiler_info,
                mitigation_info=self.mitigation_info,
            )

            return job.job_id

        except Exception as e:
            print(f"OQTOPUS submission failed: {e}")
            return None

    def run_circuit_locally(self, circuit: Any, shots: int) -> dict[str, Any] | None:
        """
        Run local simulator
        """
        if not self.local_simulator_available:
            return None

        try:
            import uuid

            from qiskit import transpile

            # Transpile circuit
            compiled_circuit = transpile(circuit, self.local_simulator)

            # Run simulation
            job = self.local_simulator.run(compiled_circuit, shots=shots)
            result = job.result()
            counts = result.get_counts()

            job_id = str(uuid.uuid4())[:8]

            return {
                "job_id": job_id,
                "counts": dict(counts),
                "shots": shots,
                "success": True,
                "simulator": "local",
            }

        except Exception as e:
            print(f"Local simulation failed: {e}")
            return None

    def submit_circuits_parallel(
        self,
        circuits: list[Any],
        devices: list[str] = ["qulacs"],
        shots: int = 1024,
        parallel_workers: int = 4,
    ) -> dict[str, list[str]]:
        """
        Submit multiple circuits in parallel (improved version)
        """
        print(
            f"Submitting {len(circuits)} circuits to {len(devices)} devices using {parallel_workers} workers"
        )

        if not self.oqtopus_available:
            print("OQTOPUS not available, falling back to local simulation...")
            if self.local_simulator_available:
                return self.submit_circuits_locally(circuits, devices, shots)
            else:
                print("Local simulator also not available")
                return {device: [] for device in devices}

        all_job_ids = {device: [] for device in devices}
        submission_tasks = []

        def submit_single_circuit(circuit, device, index):
            try:
                job_id = self.submit_circuit_to_oqtopus(circuit, shots, device)
                if job_id:
                    print(
                        f"Circuit {index + 1}/{len(circuits)} → {device}: {job_id[:8]}..."
                    )
                    return device, job_id
                else:
                    print(f"Circuit {index + 1}/{len(circuits)} → {device}: failed")
                    return device, None
            except Exception as e:
                print(f"❌ Circuit {index + 1} submission error: {e}")
                return device, None

        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            # Store pairs of future and index to maintain order
            future_to_info = {}
            for device in devices:
                for i, circuit in enumerate(circuits):
                    future = executor.submit(submit_single_circuit, circuit, device, i)
                    future_to_info[future] = (device, i)
                    submission_tasks.append(future)

            # Collect results in order
            device_results = {device: [None] * len(circuits) for device in devices}
            for future in as_completed(submission_tasks):
                device, job_id = future.result()
                original_device, original_index = future_to_info[future]
                if job_id:
                    device_results[original_device][original_index] = job_id

            # Set placeholder job_id for failed jobs
            for device in devices:
                final_job_ids = []
                for i, job_id in enumerate(device_results[device]):
                    if job_id is not None:
                        final_job_ids.append(job_id)
                    else:
                        # Generate placeholder job_id for failed cases
                        failed_job_id = f"failed_{device}_{i}_{int(time.time())}"
                        final_job_ids.append(failed_job_id)
                all_job_ids[device] = final_job_ids

        for device, jobs in all_job_ids.items():
            print(f"✅ {device}: {len(jobs)} jobs submitted")

        return all_job_ids

    def submit_circuits_locally(
        self, circuits: list[Any], devices: list[str] = ["qulacs"], shots: int = 1024
    ) -> dict[str, list[str]]:
        """
        Submit circuits for local simulator (also get results immediately)
        """
        print(f"Running {len(circuits)} circuits locally...")

        all_job_ids = {}

        for device in devices:
            device_jobs = []

            for i, circuit in enumerate(circuits):
                result = self.run_circuit_locally(circuit, shots)
                if result:
                    job_id = result["job_id"]
                    device_jobs.append(job_id)

                    # Save results internally (to be retrieved later with collect)
                    if not hasattr(self, "_local_results"):
                        self._local_results = {}
                    self._local_results[job_id] = result

                    print(
                        f"Circuit {i + 1}/{len(circuits)} → {device}: {job_id} (local)"
                    )
                else:
                    # Generate placeholder job_id for failed cases
                    failed_job_id = f"failed_{device}_{i}_{int(time.time())}"
                    device_jobs.append(failed_job_id)

                    # Save failed results locally as well
                    if not hasattr(self, "_local_results"):
                        self._local_results = {}
                    self._local_results[failed_job_id] = {
                        "job_id": failed_job_id,
                        "success": False,
                        "counts": {},
                        "error": "Local simulation failed",
                    }

                    print(f"Circuit {i + 1}/{len(circuits)} → {device}: failed")

            all_job_ids[device] = device_jobs
            print(f"{device}: {len(device_jobs)} circuits completed locally")

        return all_job_ids

    def get_oqtopus_result(
        self, job_id: str, timeout_minutes: int = 30, verbose_log: bool = False
    ) -> dict[str, Any] | None:
        """
        Get OQTOPUS result (with proper job status retrieval)
        """
        # Case of failed job placeholder
        if job_id.startswith("failed_"):
            return {
                "job_id": job_id,
                "success": False,
                "counts": {},
                "error": "Job submission failed",
            }

        # Case where local results are available
        if hasattr(self, "_local_results") and job_id in self._local_results:
            return self._local_results[job_id]

        if not self.oqtopus_available:
            return None

        import time

        max_retries = 5
        retry_delay = 2  # Initial wait time (seconds)

        for attempt in range(max_retries):
            try:
                if verbose_log and attempt > 0:
                    print(f"⏳ Retry {attempt}/{max_retries} for {job_id[:8]}...")
                elif verbose_log:
                    print(f"⏳ Waiting for result: {job_id[:8]}...")

                job = self.oqtopus_backend.retrieve_job(job_id)

                # Use proper job status retrieval method
                try:
                    job_dict = job._job.to_dict()
                    status = job_dict.get("status", "unknown")

                    if verbose_log:
                        print(f"🔍 {job_id[:8]} status: {status}")

                    # Success state
                    if status == "succeeded":
                        try:
                            result = job.result()
                            if result and hasattr(result, "counts"):
                                counts = result.counts
                                return {
                                    "job_id": job_id,
                                    "counts": dict(counts),
                                    "shots": sum(counts.values()),
                                    "status": status,
                                    "success": True,
                                }
                        except Exception as result_error:
                            if verbose_log:
                                print(
                                    f"⚠️ Result extraction failed for {job_id[:8]}: {result_error}"
                                )

                    # Exit immediately if clearly failed
                    elif status in ["failed", "cancelled", "error"]:
                        return {
                            "job_id": job_id,
                            "status": status,
                            "success": False,
                            "error": f"Job {status}",
                        }

                    # Try to get result if in ready state
                    elif status == "ready":
                        try:
                            result = job.result()
                            if result and hasattr(result, "counts"):
                                counts = result.counts
                                return {
                                    "job_id": job_id,
                                    "counts": dict(counts),
                                    "shots": sum(counts.values()),
                                    "status": status,
                                    "success": True,
                                }
                        except Exception as ready_error:
                            if verbose_log:
                                print(
                                    f"⚠️ Ready result extraction failed for {job_id[:8]}: {ready_error}"
                                )

                    # Still processing (submitted, running, queued, etc.)
                    elif status in ["submitted", "running", "queued", "pending"]:
                        if attempt < max_retries - 1:  # Wait if not the last attempt
                            wait_time = retry_delay * (
                                2**attempt
                            )  # Exponential backoff
                            if verbose_log:
                                print(
                                    f"⌛ Job {job_id[:8]} still {status}, waiting {wait_time}s..."
                                )
                            time.sleep(wait_time)
                            continue
                        else:
                            # Still processing even in the last attempt
                            return {
                                "job_id": job_id,
                                "status": status,
                                "success": False,
                                "error": f"Job timeout in {status} state",
                            }

                    # Unknown state
                    else:
                        if attempt < max_retries - 1:
                            wait_time = retry_delay
                            if verbose_log:
                                print(
                                    f"❓ Unknown status {status} for {job_id[:8]}, waiting {wait_time}s..."
                                )
                            time.sleep(wait_time)
                            continue
                        else:
                            return {
                                "job_id": job_id,
                                "status": status,
                                "success": False,
                                "error": f"Unknown job status: {status}",
                            }

                except Exception as status_error:
                    if verbose_log:
                        print(
                            f"⚠️ Status check failed for {job_id[:8]} (attempt {attempt + 1}): {status_error}"
                        )

                    # Fallback: try to get result with legacy method
                    try:
                        result = job.result()
                        if result and hasattr(result, "counts"):
                            counts = result.counts
                            return {
                                "job_id": job_id,
                                "counts": dict(counts),
                                "shots": sum(counts.values()),
                                "success": True,
                            }
                    except Exception:
                        pass

                    # Wait and retry if not the last attempt
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue

            except Exception as e:
                if verbose_log:
                    print(
                        f"❌ Result collection failed for {job_id[:8]} (attempt {attempt + 1}): {e}"
                    )

                # Wait and retry if not the last attempt
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue

        # All attempts failed
        return {
            "job_id": job_id,
            "status": "timeout",
            "success": False,
            "error": f"Failed after {max_retries} attempts",
        }

    def collect_results_parallel(
        self, job_ids: dict[str, list[str]], wait_minutes: int = 30
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Collect results in parallel
        """
        print(f"Collecting results from {len(job_ids)} devices...")

        # Fast processing when local results are available
        if hasattr(self, "_local_results"):
            print("Using local simulation results...")
            all_results = {}
            for device, device_job_ids in job_ids.items():
                device_results = []
                for job_id in device_job_ids:
                    if job_id in self._local_results:
                        result = self._local_results[job_id]
                        device_results.append(result)
                        print(f"{device}: {job_id[:8]}... collected (local)")
                all_results[device] = device_results
                print(f"{device}: {len(device_results)} results collected")
            return all_results

        if not self.oqtopus_available:
            print("OQTOPUS not available")
            return {}

        def collect_from_device(device_data):
            device, device_job_ids = device_data
            device_results = [None] * len(device_job_ids)

            # Collect results with index to maintain order
            for i, job_id in enumerate(device_job_ids):
                result = self.get_oqtopus_result(job_id, wait_minutes, verbose_log=True)
                if result and result.get("success", False):
                    device_results[i] = result
                    print(f"✅ {device}: {job_id[:8]}... collected")
                else:
                    status = result.get("status", "unknown") if result else "no_result"
                    print(f"❌ {device}: {job_id[:8]}... failed (status: {status})")

            # Return None as-is to maintain order
            return device, device_results

        all_results = {}

        # Parallel collection
        with ThreadPoolExecutor(max_workers=len(job_ids)) as executor:
            futures = [
                executor.submit(collect_from_device, item) for item in job_ids.items()
            ]

            for future in as_completed(futures):
                device, results = future.result()
                all_results[device] = results
                print(f"✅ {device}: {len(results)} results collected")

        return all_results

    # Abstract methods: implemented in each experiment class
    @abstractmethod
    def create_circuits(self, **kwargs) -> list[Any]:
        """Experiment-specific circuit creation (implemented in each experiment class)"""
        pass

    @abstractmethod
    def analyze_results(
        self, results: dict[str, list[dict[str, Any]]], **kwargs
    ) -> dict[str, Any]:
        """Experiment-specific result analysis (implemented in each experiment class)"""
        pass

    @abstractmethod
    def save_experiment_data(
        self, results: dict[str, Any], metadata: dict[str, Any] = None
    ) -> str:
        """Experiment-specific data saving (implemented in each experiment class)"""
        pass

    # Common save methods
    def save_job_ids(
        self,
        job_ids: dict[str, list[str]],
        metadata: dict[str, Any] = None,
        filename: str = "job_ids",
    ) -> str:
        """Save job IDs"""
        save_data = {
            "job_ids": job_ids,
            "submitted_at": time.time(),
            "experiment_type": self.__class__.__name__,
            "oqtopus_config": {
                "transpiler_options": self.transpiler_options,
                "mitigation_options": self.mitigation_options,
                "basis_gates": self.anemone_basis_gates,
            },
            "metadata": metadata or {},
        }
        return self.data_manager.save_data(save_data, filename)

    def save_raw_results(
        self,
        results: dict[str, Any],
        metadata: dict[str, Any] = None,
        filename: str = "raw_results",
    ) -> str:
        """Save raw results"""
        save_data = {
            "results": results,
            "saved_at": time.time(),
            "experiment_type": self.__class__.__name__,
            "oqtopus_available": self.oqtopus_available,
            "metadata": metadata or {},
        }
        return self.data_manager.save_data(save_data, filename)

    def save_experiment_summary(self) -> str:
        """Save experiment summary"""
        return self.data_manager.summary()

    # Template method: overall experiment flow
    def run_experiment(
        self,
        devices: list[str] = ["qulacs"],
        shots: int = 1024,
        submit_interval: float = 1.0,
        wait_minutes: int = 30,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Template method for experiment execution
        Can be overridden in each experiment class
        """
        print(f"Running {self.__class__.__name__}")

        # 1. Circuit creation (experiment-specific)
        circuits = self.create_circuits(**kwargs)
        print(f"Created {len(circuits)} circuits")

        # 2. Parallel submission
        job_ids = self.submit_circuits_parallel(
            circuits, devices, shots, submit_interval
        )

        # 3. Result collection
        raw_results = self.collect_results_parallel(job_ids, wait_minutes)

        # 4. Result analysis (experiment-specific)
        analyzed_results = self.analyze_results(raw_results, **kwargs)

        # 5. Data saving (experiment-specific)
        save_path = self.save_experiment_data(analyzed_results)

        print(f"{self.__class__.__name__} completed")
        print(f"Results saved: {save_path}")

        return {
            "job_ids": job_ids,
            "raw_results": raw_results,
            "analyzed_results": analyzed_results,
            "experiment_metadata": {
                "experiment_type": self.__class__.__name__,
                "devices": devices,
                "shots": shots,
                "circuits_count": len(circuits),
            },
        }
