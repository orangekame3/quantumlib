#!/usr/bin/env python3
"""
Quantum Experiment Simple - OQTOPUS-based simple design
Circuit creation is separated, OQTOPUS backend part is visible to users
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from quri_parts_oqtopus.backend import OqtopusSamplingBackend

import numpy as np

from ..circuit.factory import create_chsh_circuit
from ..core.data_manager import SimpleDataManager

# OQTOPUS imports (visible to users)
try:
    import quri_parts_oqtopus.backend  # noqa: F401

    OQTOPUS_AVAILABLE = True
except ImportError:
    OQTOPUS_AVAILABLE = False


class QuantumExperimentSimple:
    """
    Simple quantum experiment class

    Design principles:
    - Circuit creation is separated (using circuit_factory)
    - OQTOPUS backend is visible to users
    - Minimal abstraction
    """

    def __init__(
        self,
        experiment_name: str | None = None,
        oqtopus_backend: "OqtopusSamplingBackend | None" = None,
    ):
        """
        Initialize quantum experiment

        Args:
            experiment_name: Experiment name
            oqtopus_backend: OQTOPUS backend (auto-created if omitted)
        """
        self.experiment_name = experiment_name or f"quantum_exp_{int(time.time())}"
        self.data_manager = SimpleDataManager(self.experiment_name)

        # OQTOPUS backend configuration (visible to users)
        if oqtopus_backend:
            self.oqtopus_backend = oqtopus_backend
            self.oqtopus_available = True
        else:
            self.oqtopus_available = OQTOPUS_AVAILABLE
            if OQTOPUS_AVAILABLE:
                from quri_parts_oqtopus.backend import OqtopusSamplingBackend

                self.oqtopus_backend = OqtopusSamplingBackend()

        # OQTOPUS settings (directly editable by users)
        self.anemone_basis_gates = ["sx", "x", "rz", "cx"]

        # transpiler_options - direct user access
        self.transpiler_options = {
            "basis_gates": self.anemone_basis_gates,
            "optimization_level": 1,
        }

        # mitigation_options - direct user access
        self.mitigation_options = {
            "ro_error_mitigation": "pseudo_inverse",
        }

        # Internal structure for OQTOPUS (backward compatibility)
        self.transpiler_info = {
            "transpiler_lib": "qiskit",
            "transpiler_options": self.transpiler_options,
        }
        self.mitigation_info = self.mitigation_options

        print(f"🧪 QuantumExperiment: {self.experiment_name}")
        print(f"🔧 OQTOPUS: {'✅' if self.oqtopus_available else '❌'}")

    def create_chsh_circuit(
        self, theta_a: float, theta_b: float, phase_phi: float = 0
    ) -> Any:
        """
        Create CHSH circuit (using circuit_factory)
        """
        return create_chsh_circuit(theta_a, theta_b, phase_phi)

    def submit_circuit_to_oqtopus(
        self, circuit: Any, shots: int, device_id: str
    ) -> str | None:
        """
        Submit single circuit to OQTOPUS (implementation visible to users)

        Args:
            circuit: Qiskit circuit
            shots: Number of shots
            device_id: Device ID

        Returns:
            Job ID
        """
        if not self.oqtopus_available:
            print("❌ OQTOPUS not available")
            return None

        try:
            # Use QASM3 as standard
            from qiskit.qasm3 import dumps

            qasm_str = dumps(circuit)

            f"circuit_{int(time.time())}"

            # Dynamically update transpiler_info and mitigation_info
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
            print(f"❌ OQTOPUS submission failed: {e}")
            return None

    def submit_circuits_parallel(
        self,
        circuits: list[Any],
        devices: list[str] = ["qulacs"],
        shots: int = 1024,
        submit_interval: float = 1.0,
    ) -> dict[str, list[str]]:
        """
        Submit multiple circuits in parallel
        """
        print(f"🚀 Submitting {len(circuits)} circuits to {len(devices)} devices")

        if not self.oqtopus_available:
            print("❌ OQTOPUS not available")
            return {device: [] for device in devices}

        all_job_ids = {}

        def submit_to_device(device):
            device_jobs = []
            for i, circuit in enumerate(circuits):
                try:
                    job_id = self.submit_circuit_to_oqtopus(circuit, shots, device)
                    if job_id:
                        device_jobs.append(job_id)
                        print(
                            f"✅ Circuit {i + 1}/{len(circuits)} → {device}: {job_id[:8]}..."
                        )
                    else:
                        print(f"❌ Circuit {i + 1}/{len(circuits)} → {device}: failed")

                    # Reduce server load
                    if submit_interval > 0 and i < len(circuits) - 1:
                        time.sleep(submit_interval)

                except Exception as e:
                    print(f"❌ Circuit {i + 1} submission error: {e}")

            return device, device_jobs

        # Parallel submission
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            futures = [executor.submit(submit_to_device, device) for device in devices]

            for future in as_completed(futures):
                device, job_ids = future.result()
                all_job_ids[device] = job_ids
                print(f"✅ {device}: {len(job_ids)} jobs submitted")

        return all_job_ids

    def get_oqtopus_result(
        self, job_id: str, timeout_minutes: int = 30, verbose_log: bool = False
    ) -> dict[str, Any] | None:
        """
        Get OQTOPUS result (implementation visible to users)

        Args:
            job_id: Job ID
            timeout_minutes: Timeout in minutes
            verbose_log: Enable/disable verbose logging

        Returns:
            Measurement results
        """
        if not self.oqtopus_available:
            return None

        try:
            # Output detailed logs only when enabled
            if verbose_log:
                print(f"⏳ Waiting for result: {job_id[:8]}...")

            # Get OQTOPUS result
            job = self.oqtopus_backend.retrieve_job(job_id)

            # Wait for result (simple implementation)
            import time

            max_wait = timeout_minutes * 60
            wait_time = 0

            while wait_time < max_wait:
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
                    time.sleep(5)
                    wait_time += 5

            print(f"⏳ Timeout waiting for {job_id[:8]}...")
            return None

        except Exception as e:
            print(f"❌ Result collection failed for {job_id}: {e}")
            return None

    def collect_results_parallel(
        self, job_ids: dict[str, list[str]], wait_minutes: int = 30
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Collect results in parallel
        """
        print(f"⏳ Collecting results from {len(job_ids)} devices...")

        if not self.oqtopus_available:
            print("❌ OQTOPUS not available")
            return {}

        def collect_from_device(device_data):
            device, device_job_ids = device_data
            device_results = []

            for job_id in device_job_ids:
                result = self.get_oqtopus_result(job_id, wait_minutes)
                if result:
                    device_results.append(result)
                    print(f"✅ {device}: {job_id[:8]}... collected")
                else:
                    print(f"❌ {device}: {job_id[:8]}... failed")

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

    def run_chsh_experiment(
        self,
        phase_points: int = 20,
        devices: list[str] = ["qulacs"],
        shots: int = 1024,
        submit_interval: float = 2.0,
        wait_minutes: int = 30,
    ) -> dict[str, Any]:
        """
        Run CHSH experiment
        """
        print(f"🎯 CHSH Experiment: {phase_points} points, {shots} shots")

        # Create phase scan circuits (using circuit_factory)
        phase_range = np.linspace(0, 2 * np.pi, phase_points)
        circuits = []

        for phi in phase_range:
            circuit = self.create_chsh_circuit(0, np.pi / 4, phase_phi=phi)
            circuits.append(circuit)

        print(f"🔧 Created {len(circuits)} CHSH circuits")

        # Parallel execution
        job_ids = self.submit_circuits_parallel(
            circuits, devices, shots, submit_interval
        )
        results = self.collect_results_parallel(job_ids, wait_minutes)

        # Calculate theoretical values
        S_theoretical = 2 * np.sqrt(2) * np.cos(phase_range)

        return {
            "job_ids": job_ids,
            "results": results,
            "phase_range": phase_range.tolist(),
            "S_theoretical": S_theoretical.tolist(),
            "experiment_metadata": {
                "phase_points": phase_points,
                "devices": devices,
                "shots": shots,
                "oqtopus_available": self.oqtopus_available,
            },
        }

    def save_job_ids(
        self,
        job_ids: dict[str, list[str]],
        metadata: dict[str, Any] | None = None,
        filename: str = "job_ids",
    ) -> str:
        """Save job IDs"""
        save_data = {
            "job_ids": job_ids,
            "submitted_at": time.time(),
            "oqtopus_config": {
                "transpiler_options": self.transpiler_options,
                "mitigation_options": self.mitigation_options,
                "basis_gates": self.anemone_basis_gates,
            },
            "metadata": metadata or {},
        }
        return self.data_manager.save_data(save_data, filename)

    def save_results(
        self,
        results: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        filename: str = "results",
    ) -> str:
        """Save experiment results"""
        save_data = {
            "results": results,
            "saved_at": time.time(),
            "oqtopus_available": self.oqtopus_available,
            "metadata": metadata or {},
        }
        return self.data_manager.save_data(save_data, filename)

    def save_experiment_summary(self) -> str:
        """Save experiment summary"""
        return self.data_manager.summary()


# Convenience functions (simple version)
def run_chsh_comparison_simple(
    devices: list[str] = ["qulacs"],
    phase_points: int = 20,
    shots: int = 1024,
    submit_interval: float = 2.0,
    experiment_name: str | None = None,
) -> dict[str, Any]:
    """
    Simple execution of CHSH comparison experiment (simple version)
    """
    exp = QuantumExperimentSimple(experiment_name)

    return exp.run_chsh_experiment(
        phase_points=phase_points,
        devices=devices,
        shots=shots,
        submit_interval=submit_interval,
    )
