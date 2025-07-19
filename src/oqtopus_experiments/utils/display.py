"""
Display and Formatting Utilities

Common utilities for displaying experiment results and creating summaries
using Rich tables and fallback plain text formatting.
"""

import time
from typing import Any


def display_experiment_results(
    results: dict[str, Any], experiment_type: str, use_rich: bool = True
) -> None:
    """
    Display experiment results using Rich tables with plain text fallback.

    This consolidates the display_results methods that were duplicated across
    all experiment types.

    Args:
        results: Dictionary containing experiment results
        experiment_type: Type of experiment (e.g., "T1", "Ramsey", "CHSH")
        use_rich: Whether to use Rich formatting (fallback to plain text if False)
    """
    if use_rich:
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()

            # Create main results table
            table = Table(
                title=f"{experiment_type} Experiment Results", show_header=True
            )
            table.add_column("Metric", style="bold cyan")
            table.add_column("Value", style="bold green")

            # Add common metrics
            if "analysis" in results:
                analysis = results["analysis"]

                # Add experiment-specific metrics
                if experiment_type == "T1" and "estimated_t1" in analysis:
                    table.add_row("Estimated T1", f"{analysis['estimated_t1']:.3f} ns")
                elif experiment_type == "Ramsey" and "estimated_t2_star" in analysis:
                    table.add_row(
                        "Estimated T2*", f"{analysis['estimated_t2_star']:.3f} ns"
                    )
                    if "estimated_frequency" in analysis:
                        table.add_row(
                            "Detuning Frequency",
                            f"{analysis['estimated_frequency']:.3f} MHz",
                        )
                elif experiment_type == "T2 Echo" and "estimated_t2" in analysis:
                    table.add_row("Estimated T2", f"{analysis['estimated_t2']:.3f} ns")
                elif experiment_type == "CHSH" and "s_value" in analysis:
                    table.add_row("S Value", f"{analysis['s_value']:.3f}")
                    table.add_row("Classical Bound", "2.000")
                    table.add_row("Quantum Maximum", "2.828")
                elif experiment_type == "Rabi" and "rabi_frequency" in analysis:
                    table.add_row(
                        "Rabi Frequency", f"{analysis['rabi_frequency']:.3f} MHz"
                    )

                # Add fit quality if available
                if "fit_quality" in analysis:
                    fit_quality = analysis["fit_quality"]
                    if "r_squared" in fit_quality:
                        table.add_row(
                            "R² (Fit Quality)", f"{fit_quality['r_squared']:.4f}"
                        )

            # Add device information
            if "device_results" in results:
                device_count = len(results["device_results"])
                table.add_row("Devices Used", str(device_count))

                # Show device names
                device_names = list(results["device_results"].keys())
                if device_names:
                    table.add_row("Device Names", ", ".join(device_names))

            # Add timing information
            if "experiment_params" in results:
                params = results["experiment_params"]
                if "delay_times" in params and isinstance(params["delay_times"], list):
                    delay_range = params["delay_times"]
                    table.add_row(
                        "Delay Range",
                        f"{min(delay_range):.1f} - {max(delay_range):.1f} ns",
                    )
                    table.add_row("Delay Points", str(len(delay_range)))

            console.print(table)

            # Create device-specific table if multiple devices
            if "device_results" in results and len(results["device_results"]) > 1:
                device_table = Table(title="Device-Specific Results", show_header=True)
                device_table.add_column("Device", style="bold cyan")
                device_table.add_column("Status", style="bold")
                device_table.add_column("Data Points", style="green")

                for device, device_data in results["device_results"].items():
                    if isinstance(device_data, dict) and "analysis" in device_data:
                        analysis = device_data["analysis"]
                        status = (
                            "✅ Success"
                            if analysis.get("fit_success", False)
                            else "⚠️ Partial"
                        )
                        data_points = len(analysis.get("data_points", []))
                    else:
                        status = "❌ Failed"
                        data_points = 0

                    device_table.add_row(device, status, str(data_points))

                console.print(device_table)

        except ImportError:
            # Fallback to plain text if Rich is not available
            use_rich = False

    if not use_rich:
        # Plain text fallback
        print(f"\n{'=' * 50}")
        print(f"{experiment_type} Experiment Results")
        print(f"{'=' * 50}")

        if "analysis" in results:
            analysis = results["analysis"]

            if experiment_type == "T1" and "estimated_t1" in analysis:
                print(f"Estimated T1: {analysis['estimated_t1']:.3f} ns")
            elif experiment_type == "Ramsey" and "estimated_t2_star" in analysis:
                print(f"Estimated T2*: {analysis['estimated_t2_star']:.3f} ns")
                if "estimated_frequency" in analysis:
                    print(
                        f"Detuning Frequency: {analysis['estimated_frequency']:.3f} MHz"
                    )
            elif experiment_type == "T2 Echo" and "estimated_t2" in analysis:
                print(f"Estimated T2: {analysis['estimated_t2']:.3f} ns")
            elif experiment_type == "CHSH" and "s_value" in analysis:
                print(f"S Value: {analysis['s_value']:.3f}")
                print("Classical Bound: 2.000")
                print("Quantum Maximum: 2.828")
            elif experiment_type == "Rabi" and "rabi_frequency" in analysis:
                print(f"Rabi Frequency: {analysis['rabi_frequency']:.3f} MHz")

            if "fit_quality" in analysis and "r_squared" in analysis["fit_quality"]:
                print(f"R² (Fit Quality): {analysis['fit_quality']['r_squared']:.4f}")

        if "device_results" in results:
            device_count = len(results["device_results"])
            print(f"Devices Used: {device_count}")

            if device_count > 1:
                print("\nDevice-Specific Results:")
                for device, device_data in results["device_results"].items():
                    if isinstance(device_data, dict) and "analysis" in device_data:
                        status = (
                            "Success"
                            if device_data["analysis"].get("fit_success", False)
                            else "Partial"
                        )
                    else:
                        status = "Failed"
                    print(f"  {device}: {status}")

        print(f"{'=' * 50}")


def create_experiment_summary(
    results: dict[str, Any],
    experiment_type: str,
    experiment_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Create a standardized experiment summary.

    This consolidates the summary creation logic used across all experiments.

    Args:
        results: Experiment results dictionary
        experiment_type: Type of experiment
        experiment_params: Experiment parameters

    Returns:
        Standardized summary dictionary
    """
    summary = {
        "experiment_type": experiment_type,
        "timestamp": time.time(),
        "summary_created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_devices": 0,
        "successful_devices": 0,
        "status": "unknown",
    }

    # Add experiment parameters
    if experiment_params:
        summary["experiment_parameters"] = experiment_params.copy()

    # Process device results
    if "device_results" in results:
        device_results = results["device_results"]
        summary["total_devices"] = len(device_results)
        summary["device_names"] = list(device_results.keys())

        successful_count = 0
        for _device, device_data in device_results.items():
            if isinstance(device_data, dict) and "analysis" in device_data:
                if device_data["analysis"].get("fit_success", False):
                    successful_count += 1

        summary["successful_devices"] = successful_count
        summary["success_rate"] = (
            successful_count / len(device_results) if device_results else 0.0
        )

    # Add main analysis results
    if "analysis" in results:
        analysis = results["analysis"]
        summary["main_results"] = {}

        # Experiment-specific main results
        if experiment_type == "T1" and "estimated_t1" in analysis:
            summary["main_results"]["t1_time"] = analysis["estimated_t1"]
        elif experiment_type == "Ramsey":
            if "estimated_t2_star" in analysis:
                summary["main_results"]["t2_star"] = analysis["estimated_t2_star"]
            if "estimated_frequency" in analysis:
                summary["main_results"]["detuning_frequency"] = analysis[
                    "estimated_frequency"
                ]
        elif experiment_type == "T2 Echo" and "estimated_t2" in analysis:
            summary["main_results"]["t2_time"] = analysis["estimated_t2"]
        elif experiment_type == "CHSH" and "s_value" in analysis:
            summary["main_results"]["s_value"] = analysis["s_value"]
            summary["main_results"]["bell_violation"] = analysis["s_value"] > 2.0
        elif experiment_type == "Rabi" and "rabi_frequency" in analysis:
            summary["main_results"]["rabi_frequency"] = analysis["rabi_frequency"]

        # Add fit quality
        if "fit_quality" in analysis:
            summary["fit_quality"] = analysis["fit_quality"]

    # Determine overall status
    if summary["total_devices"] == 0:
        summary["status"] = "no_devices"
    elif summary["successful_devices"] == 0:
        summary["status"] = "all_failed"
    elif summary["successful_devices"] == summary["total_devices"]:
        summary["status"] = "all_success"
    else:
        summary["status"] = "partial_success"

    return summary


def format_time_duration(seconds: float) -> str:
    """
    Format a duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_scientific_notation(value: float, precision: int = 3) -> str:
    """
    Format a number in scientific notation with specified precision.

    Args:
        value: Number to format
        precision: Number of decimal places

    Returns:
        Formatted string
    """
    if abs(value) < 1e-3 or abs(value) >= 1e6:
        return f"{value:.{precision}e}"
    else:
        return f"{value:.{precision}f}"
