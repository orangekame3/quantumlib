"""
Statistical Processing Utilities

Common statistical functions used across different quantum experiments
for probability calculations and parameter estimation.
"""

from typing import Any

import numpy as np


def calculate_probability(counts: dict[str, int], target_state: str) -> float:
    """
    Calculate probability of measuring a specific quantum state.

    This consolidates the p0/p1 probability calculations used across
    T1, Ramsey, T2 Echo, and Rabi experiments.

    Args:
        counts: Dictionary of measurement counts
        target_state: Target state (e.g., "0", "1", "00", "01", etc.)

    Returns:
        Probability of measuring target_state (0.0 to 1.0)
    """
    if not counts:
        return 0.0

    total_shots = sum(counts.values())
    if total_shots == 0:
        return 0.0

    target_count = counts.get(target_state, 0)
    return target_count / total_shots


def calculate_p0_probability(counts: dict[str, int]) -> float:
    """Calculate P(|0⟩) for single-qubit measurements."""
    return calculate_probability(counts, "0")


def calculate_p1_probability(counts: dict[str, int]) -> float:
    """Calculate P(|1⟩) for single-qubit measurements."""
    return calculate_probability(counts, "1")


def calculate_z_expectation(counts: dict[str, int]) -> float:
    """
    Calculate <Z> expectation value for single-qubit measurements.

    Used in T1, Ramsey, and T2 Echo experiments.

    Args:
        counts: Dictionary of measurement counts

    Returns:
        <Z> expectation value: P(0) - P(1)
    """
    if not counts:
        return 0.0

    p0 = calculate_p0_probability(counts)
    p1 = calculate_p1_probability(counts)

    return p0 - p1


def estimate_parameters_with_quality(
    x_data: np.ndarray,
    y_data: np.ndarray,
    fit_function,
    initial_params: list[float],
    param_bounds: tuple[list[float], list[float]] | None = None,
    max_iterations: int = 2000
) -> dict[str, Any]:
    """
    Estimate parameters using curve fitting with quality assessment.

    This consolidates the parameter estimation logic used in Ramsey, T1,
    and T2 Echo experiments.

    Args:
        x_data: Independent variable data
        y_data: Dependent variable data
        fit_function: Function to fit (e.g., exponential_decay, oscillation)
        initial_params: Initial parameter guesses
        param_bounds: Optional parameter bounds (lower, upper)
        max_iterations: Maximum fitting iterations

    Returns:
        Dictionary containing:
        - fitted_params: Optimized parameters
        - param_errors: Parameter uncertainties
        - fit_quality: Quality metrics
        - success: Boolean indicating fit success
    """
    try:
        from scipy.optimize import curve_fit

        # Remove invalid data points
        valid_mask = ~(np.isnan(y_data) | np.isinf(y_data))
        if not np.any(valid_mask):
            return {
                "fitted_params": initial_params,
                "param_errors": [0.0] * len(initial_params),
                "fit_quality": {"r_squared": 0.0, "residual_std": float('inf')},
                "success": False,
                "error": "No valid data points"
            }

        x_valid = x_data[valid_mask]
        y_valid = y_data[valid_mask]

        if len(x_valid) < len(initial_params):
            return {
                "fitted_params": initial_params,
                "param_errors": [0.0] * len(initial_params),
                "fit_quality": {"r_squared": 0.0, "residual_std": float('inf')},
                "success": False,
                "error": f"Insufficient data points: {len(x_valid)} < {len(initial_params)}"
            }

        # Perform curve fitting
        if param_bounds:
            popt, pcov = curve_fit(
                fit_function,
                x_valid,
                y_valid,
                p0=initial_params,
                bounds=param_bounds,
                maxfev=max_iterations
            )
        else:
            popt, pcov = curve_fit(
                fit_function,
                x_valid,
                y_valid,
                p0=initial_params,
                maxfev=max_iterations
            )

        # Calculate parameter uncertainties
        param_errors = np.sqrt(np.diag(pcov)) if pcov is not None else [0.0] * len(popt)

        # Calculate fit quality metrics
        y_fitted = fit_function(x_valid, *popt)
        residuals = y_valid - y_fitted
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_valid - np.mean(y_valid))**2)

        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        residual_std = np.std(residuals)

        return {
            "fitted_params": popt.tolist(),
            "param_errors": param_errors.tolist(),
            "fit_quality": {
                "r_squared": float(r_squared),
                "residual_std": float(residual_std),
                "residuals": residuals.tolist()
            },
            "success": True
        }

    except Exception as e:
        return {
            "fitted_params": initial_params,
            "param_errors": [0.0] * len(initial_params),
            "fit_quality": {"r_squared": 0.0, "residual_std": float('inf')},
            "success": False,
            "error": str(e)
        }


def exponential_decay(t, A, T, offset):
    """Exponential decay function for T1 fitting."""
    return A * np.exp(-t / T) + offset


def damped_oscillation(t, A, T2_star, frequency, phase, offset):
    """Damped oscillation function for Ramsey fitting."""
    return A * np.exp(-t / T2_star) * np.cos(2 * np.pi * frequency * t + phase) + offset


def echo_decay(t, A, T2, offset):
    """Echo decay function for T2 Echo fitting."""
    return A * np.exp(-t / T2) + offset


def rabi_oscillation(t, A, rabi_freq, phase, offset):
    """Rabi oscillation function for Rabi experiment fitting."""
    return A * np.cos(2 * np.pi * rabi_freq * t + phase) + offset


def calculate_fidelity(theoretical_probs: dict[str, float], measured_probs: dict[str, float]) -> float:
    """
    Calculate state fidelity between theoretical and measured probability distributions.

    Args:
        theoretical_probs: Theoretical probability distribution
        measured_probs: Measured probability distribution

    Returns:
        Fidelity value between 0 and 1
    """
    if not theoretical_probs or not measured_probs:
        return 0.0

    # Ensure both distributions have the same states
    all_states = set(theoretical_probs.keys()) | set(measured_probs.keys())

    fidelity = 0.0
    for state in all_states:
        p_theo = theoretical_probs.get(state, 0.0)
        p_meas = measured_probs.get(state, 0.0)
        fidelity += np.sqrt(p_theo * p_meas)

    return float(fidelity)


def calculate_process_fidelity(ideal_results: dict[str, float], actual_results: dict[str, float]) -> float:
    """Calculate process fidelity between ideal and actual experiment results."""
    return calculate_fidelity(ideal_results, actual_results)
