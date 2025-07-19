"""
Parallel Execution Mixin for OQTOPUS Experiments Experiments

This module provides a unified parallel execution framework to eliminate
code duplication across different experiment types.
"""

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any


class ParallelExecutionMixin:
    """
    Mixin class providing unified parallel execution capabilities for quantum experiments.

    This class consolidates the parallel execution patterns that were duplicated across
    T1, Ramsey, CHSH, and T2 Echo experiments.
    """

    def submit_circuits_parallel_with_order(
        self,
        circuits: list[Any],
        devices: list[str],
        shots: int,
        parallel_workers: int,
        submit_function: Callable,
        progress_name: str = "Submission",
        **kwargs
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Submit circuits in parallel across devices while preserving order.

        Args:
            circuits: List of quantum circuits to submit
            devices: List of device names to submit to
            shots: Number of shots per circuit
            parallel_workers: Number of parallel worker threads
            submit_function: Function to submit a single circuit (device, circuit, shots, idx) -> result
            progress_name: Name for progress reporting
            **kwargs: Additional arguments passed to submit_function

        Returns:
            Dictionary mapping device names to lists of job results
        """
        submission_tasks = []
        for device in devices:
            for circuit_idx, circuit in enumerate(circuits):
                submission_tasks.append((device, circuit, shots, circuit_idx))

        all_jobs = {device: [None] * len(circuits) for device in devices}

        def submit_single_circuit(args):
            device, circuit, shots, circuit_idx = args
            try:
                result = submit_function(device, circuit, shots, circuit_idx, **kwargs)
                return device, circuit_idx, result, True
            except Exception as e:
                print(f"❌ {device}[{circuit_idx}]: Submission error: {str(e)[:50]}")
                return device, circuit_idx, None, False

        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = [
                executor.submit(submit_single_circuit, task)
                for task in submission_tasks
            ]

            completed_jobs = 0
            successful_jobs = 0
            total_jobs = len(futures)
            last_progress_percent = 0

            for future in as_completed(futures):
                device, circuit_idx, result, success = future.result()
                completed_jobs += 1

                if success and result:
                    successful_jobs += 1
                    all_jobs[device][circuit_idx] = result
                    print(f"✅ {device}[{circuit_idx}]: submitted ({completed_jobs}/{total_jobs})")

                # Progress reporting every 20%
                progress_percent = (completed_jobs * 100) // total_jobs
                if (
                    progress_percent >= last_progress_percent + 20
                    and progress_percent < 100
                ):
                    print(
                        f"📈 {progress_name} Progress: {completed_jobs}/{total_jobs} ({progress_percent}%) - {successful_jobs} successful"
                    )
                    last_progress_percent = progress_percent

        # Final summary
        total_successful = sum(
            1 for device_jobs in all_jobs.values()
            for job in device_jobs if job is not None
        )
        success_rate = (total_successful / total_jobs * 100) if total_jobs > 0 else 0

        print(
            f"🎉 {progress_name} Complete: {total_successful}/{total_jobs} successful ({success_rate:.1f}%)"
        )

        return all_jobs

    def collect_results_parallel_with_order(
        self,
        job_data: dict[str, list[dict[str, Any]]],
        parallel_workers: int,
        collect_function: Callable,
        progress_name: str = "Collection",
        **kwargs
    ) -> dict[str, list[dict[str, Any] | None]]:
        """
        Collect results from submitted jobs in parallel while preserving order.

        Args:
            job_data: Dictionary mapping device names to lists of job information
            parallel_workers: Number of parallel worker threads
            collect_function: Function to collect a single result (device, job_info, idx) -> result
            progress_name: Name for progress reporting
            **kwargs: Additional arguments passed to collect_function

        Returns:
            Dictionary mapping device names to lists of collected results
        """
        job_collection_tasks = []
        for device, device_jobs in job_data.items():
            for circuit_idx, job_info in enumerate(device_jobs):
                if job_info and job_info.get("submitted", False):
                    job_collection_tasks.append((device, job_info, circuit_idx))

        all_results = {
            device: [None] * len(device_jobs)
            for device, device_jobs in job_data.items()
        }

        def collect_single_result(args):
            device, job_info, circuit_idx = args
            try:
                result = collect_function(device, job_info, circuit_idx, **kwargs)
                return device, circuit_idx, result, True
            except Exception as e:
                job_id = job_info.get("job_id", "unknown")
                print(f"❌ {device}[{circuit_idx}]: {job_id[:8]}... error: {str(e)[:50]}")
                return device, circuit_idx, None, False

        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = [
                executor.submit(collect_single_result, args)
                for args in job_collection_tasks
            ]

            completed_jobs = 0
            successful_jobs = 0
            total_jobs = len(futures)
            last_progress_percent = 0

            for future in as_completed(futures):
                device, circuit_idx, result, success = future.result()
                completed_jobs += 1

                if success and result:
                    successful_jobs += 1
                    all_results[device][circuit_idx] = result
                    print(f"✅ {device}[{circuit_idx}]: collected ({completed_jobs}/{total_jobs})")

                # Progress reporting every 20%
                progress_percent = (completed_jobs * 100) // total_jobs
                if (
                    progress_percent >= last_progress_percent + 20
                    and progress_percent < 100
                ):
                    print(
                        f"📈 {progress_name} Progress: {completed_jobs}/{total_jobs} ({progress_percent}%) - {successful_jobs} successful"
                    )
                    last_progress_percent = progress_percent

        # Final summary
        total_successful = sum(
            1
            for device_results in all_results.values()
            for r in device_results
            if r is not None
        )
        total_attempted = len(job_collection_tasks)
        success_rate = (
            (total_successful / total_attempted * 100) if total_attempted > 0 else 0
        )

        print(
            f"🎉 {progress_name} Complete: {total_successful}/{total_attempted} successful ({success_rate:.1f}%)"
        )

        return all_results

    def poll_job_until_completion(
        self,
        job_id: str,
        device: str,
        timeout_minutes: int = 30,
        poll_interval_seconds: int = 5,
        backend_instance: Any | None = None
    ) -> dict[str, Any] | None:
        """
        Poll a job until completion with timeout and status reporting.

        Args:
            job_id: The job ID to poll
            device: Device name for logging
            timeout_minutes: Maximum time to wait in minutes
            poll_interval_seconds: Time between status checks in seconds
            backend_instance: Backend instance with get_job_result method

        Returns:
            Job result dictionary or None if failed/timeout
        """
        if not backend_instance:
            print(f"⚠️ No backend instance provided for polling {job_id}")
            return None

        start_time = time.time()
        timeout_seconds = timeout_minutes * 60

        while True:
            try:
                # Get current job status
                result = backend_instance.get_job_result(job_id)

                if result and result.get("status") == "completed":
                    elapsed = time.time() - start_time
                    print(f"✅ {device}: {job_id[:8]}... completed ({elapsed:.1f}s)")
                    return result
                elif result and result.get("status") in ["failed", "cancelled"]:
                    elapsed = time.time() - start_time
                    status = result.get("status", "unknown")
                    print(f"❌ {device}: {job_id[:8]}... {status} ({elapsed:.1f}s)")
                    return result

                # Check timeout
                if time.time() - start_time > timeout_seconds:
                    print(f"⏰ {device}: {job_id[:8]}... timeout after {timeout_minutes}m")
                    return None

                # Wait before next poll
                time.sleep(poll_interval_seconds)

            except Exception as e:
                elapsed = time.time() - start_time
                print(f"❌ {device}: {job_id[:8]}... polling error ({elapsed:.1f}s): {str(e)[:50]}")
                return None

    def execute_parallel_experiment(
        self,
        circuits: list[Any],
        devices: list[str],
        shots: int,
        parallel_workers: int,
        submit_function: Callable,
        collect_function: Callable,
        experiment_name: str = "Experiment",
        **kwargs
    ) -> dict[str, list[dict[str, Any] | None]]:
        """
        Execute a complete parallel experiment: submit circuits and collect results.

        Args:
            circuits: List of quantum circuits
            devices: List of device names
            shots: Number of shots per circuit
            parallel_workers: Number of parallel workers
            submit_function: Function to submit circuits
            collect_function: Function to collect results
            experiment_name: Name for progress reporting
            **kwargs: Additional arguments for submit/collect functions

        Returns:
            Dictionary mapping device names to lists of results
        """
        print(f"🚀 Starting {experiment_name} with {len(circuits)} circuits on {len(devices)} devices")

        # Phase 1: Submit circuits
        job_data = self.submit_circuits_parallel_with_order(
            circuits=circuits,
            devices=devices,
            shots=shots,
            parallel_workers=parallel_workers,
            submit_function=submit_function,
            progress_name=f"{experiment_name} Submission",
            **kwargs
        )

        # Phase 2: Collect results
        results = self.collect_results_parallel_with_order(
            job_data=job_data,
            parallel_workers=parallel_workers,
            collect_function=collect_function,
            progress_name=f"{experiment_name} Collection",
            **kwargs
        )

        return results
