"""
Data Management Utilities

Common utilities for saving experiment data, creating file paths,
and managing experimental metadata.
"""

import json
import os
import time
from pathlib import Path
from typing import Any


def create_experiment_data_structure(
    experiment_type: str,
    results: dict[str, Any],
    experiment_params: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Create a standardized experiment data structure for saving.

    This consolidates the data structure creation used across all experiments.

    Args:
        experiment_type: Type of experiment (e.g., "T1", "Ramsey", "CHSH")
        results: Experiment results
        experiment_params: Experiment parameters
        metadata: Additional metadata

    Returns:
        Standardized data structure for saving
    """
    timestamp = time.time()

    data_structure = {
        "experiment_info": {
            "type": experiment_type,
            "timestamp": timestamp,
            "date_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp)),
            "version": "quantumlib_v0.1.0",
        },
        "experiment_params": experiment_params or {},
        "results": results,
        "metadata": metadata or {},
        "oqtopus_configuration": {
            "backend_used": results.get("backend", "unknown"),
            "devices_used": list(results.get("device_results", {}).keys()),
            "total_circuits": (
                len(
                    results.get("device_results", {}).get(
                        list(results.get("device_results", {}).keys())[0], []
                    )
                )
                if results.get("device_results")
                else 0
            ),
        },
    }

    # Add experiment-specific metadata
    if experiment_type in ["T1", "Ramsey", "T2_Echo"] and experiment_params:
        if "delay_times" in experiment_params:
            data_structure["timing_info"] = {
                "delay_points": len(experiment_params["delay_times"]),
                "min_delay": min(experiment_params["delay_times"]),
                "max_delay": max(experiment_params["delay_times"]),
                "delay_unit": "ns",
            }
    elif experiment_type == "CHSH" and experiment_params:
        if "phase_range" in experiment_params:
            data_structure["phase_info"] = {
                "phase_points": len(experiment_params["phase_range"]),
                "phase_range": [0, 2 * 3.14159],  # 0 to 2π
                "phase_unit": "radians",
            }
    elif experiment_type == "Rabi" and experiment_params:
        if "amplitudes" in experiment_params:
            data_structure["amplitude_info"] = {
                "amplitude_points": len(experiment_params["amplitudes"]),
                "min_amplitude": min(experiment_params["amplitudes"]),
                "max_amplitude": max(experiment_params["amplitudes"]),
                "amplitude_unit": "normalized",
            }

    return data_structure


def save_experiment_data(
    data: dict[str, Any],
    experiment_name: str | None = None,
    output_dir: str = "quantum_experiments",
    include_timestamp: bool = True,
) -> str:
    """
    Save experiment data to JSON file with standardized naming.

    Args:
        data: Data dictionary to save
        experiment_name: Optional custom experiment name
        output_dir: Output directory for saved files
        include_timestamp: Whether to include timestamp in filename

    Returns:
        Path to saved file
    """
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate filename
    experiment_type = data.get("experiment_info", {}).get("type", "unknown")
    timestamp = data.get("experiment_info", {}).get("timestamp", time.time())

    if experiment_name:
        base_name = f"{experiment_type}_{experiment_name}"
    else:
        base_name = experiment_type

    if include_timestamp:
        timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(timestamp))
        filename = f"{base_name}_{timestamp_str}.json"
    else:
        filename = f"{base_name}.json"

    file_path = output_path / filename

    # Save data
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        print(f"💾 Experiment data saved: {file_path}")
        return str(file_path)

    except Exception as e:
        print(f"❌ Error saving experiment data: {e}")
        return ""


def load_experiment_data(file_path: str) -> dict[str, Any]:
    """
    Load experiment data from JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        Loaded experiment data dictionary
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)

        print(f"📁 Experiment data loaded: {file_path}")
        return data

    except Exception as e:
        print(f"❌ Error loading experiment data: {e}")
        return {}


def create_backup_copy(file_path: str, backup_dir: str = "backups") -> str:
    """
    Create a backup copy of an experiment data file.

    Args:
        file_path: Path to original file
        backup_dir: Directory for backup files

    Returns:
        Path to backup file
    """
    import shutil

    original_path = Path(file_path)
    if not original_path.exists():
        print(f"❌ Original file not found: {file_path}")
        return ""

    # Create backup directory
    backup_path = Path(backup_dir)
    backup_path.mkdir(parents=True, exist_ok=True)

    # Generate backup filename with timestamp
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    backup_filename = (
        f"{original_path.stem}_backup_{timestamp_str}{original_path.suffix}"
    )
    backup_file_path = backup_path / backup_filename

    try:
        shutil.copy2(file_path, backup_file_path)
        print(f"🔄 Backup created: {backup_file_path}")
        return str(backup_file_path)

    except Exception as e:
        print(f"❌ Error creating backup: {e}")
        return ""


def generate_experiment_metadata(
    experiment_type: str,
    shots: int,
    devices: list,
    parallel_workers: int = 4,
    additional_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Generate standardized experiment metadata.

    Args:
        experiment_type: Type of experiment
        shots: Number of shots per circuit
        devices: List of devices used
        parallel_workers: Number of parallel workers
        additional_info: Additional metadata to include

    Returns:
        Metadata dictionary
    """
    metadata = {
        "execution_info": {
            "shots_per_circuit": shots,
            "devices": devices,
            "parallel_workers": parallel_workers,
            "total_devices": len(devices),
            "execution_time": time.time(),
        },
        "system_info": {
            "python_version": "3.12+",
            "quantumlib_version": "0.1.0",
            "framework": "OQTOPUS",
        },
        "experiment_config": {
            "type": experiment_type,
            "parallel_execution": True,
            "error_mitigation": False,  # Can be overridden
        },
    }

    if additional_info:
        metadata.update(additional_info)

    return metadata


def cleanup_old_files(
    directory: str,
    max_age_days: int = 30,
    file_pattern: str = "*.json",
    dry_run: bool = True,
) -> list[str]:
    """
    Clean up old experiment files based on age.

    Args:
        directory: Directory to clean
        max_age_days: Maximum age in days
        file_pattern: File pattern to match
        dry_run: If True, only show what would be deleted

    Returns:
        List of files that were (or would be) deleted
    """
    import glob

    directory_path = Path(directory)
    if not directory_path.exists():
        return []

    current_time = time.time()
    max_age_seconds = max_age_days * 24 * 3600
    files_to_delete = []

    # Find files matching pattern
    pattern_path = directory_path / file_pattern
    matching_files = glob.glob(str(pattern_path))

    for file_path in matching_files:
        file_stat = os.stat(file_path)
        file_age = current_time - file_stat.st_mtime

        if file_age > max_age_seconds:
            files_to_delete.append(file_path)

            if not dry_run:
                try:
                    os.remove(file_path)
                    print(f"🗑️ Deleted old file: {file_path}")
                except Exception as e:
                    print(f"❌ Error deleting {file_path}: {e}")
            else:
                print(
                    f"🔍 Would delete: {file_path} (age: {file_age / 86400:.1f} days)"
                )

    if dry_run and files_to_delete:
        print(f"📊 Found {len(files_to_delete)} files older than {max_age_days} days")
        print("   Run with dry_run=False to actually delete them")

    return files_to_delete
