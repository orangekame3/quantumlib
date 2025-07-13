#!/usr/bin/env python3
"""
CHSH Phase Analysis Module

This module provides functions for analyzing CHSH parameter dependence on phase.
Contains phase sweep experiments and visualization functions.
"""

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def chsh_circuit_with_phase(theta_a, theta_b, phase_phi):
    """
    Create CHSH circuit with phase-controlled measurement

    This implementation produces S(φ) = S₀ cos(φ) by modifying Bob's measurement
    angle with the phase parameter. This correctly produces the theoretical
    S(φ) = 2√2 cos(φ) dependence.

    Args:
        theta_a: Alice's measurement angle (radians)
        theta_b: Bob's measurement angle (radians)
        phase_phi: Phase parameter φ (radians)

    Returns:
        QuantumCircuit: Circuit producing S(φ) = S₀ cos(φ)
    """
    qc = QuantumCircuit(2, 2)

    # Create standard maximally entangled Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
    qc.h(0)
    qc.cx(0, 1)

    # Apply measurement rotations with phase modulation
    # The key insight: modify Bob's measurement angle by the phase
    # This creates the correct S(φ) = S₀ cos(φ) dependence
    qc.ry(theta_a, 0)
    qc.ry(theta_b + phase_phi, 1)  # Phase modulation in measurement basis

    # Measurements
    qc.measure(0, 0)
    qc.measure(1, 1)

    return qc


def expectation_value(counts):
    """
    Calculate expectation value ⟨AB⟩ where A,B ∈ {-1,+1}

    Maps: |0⟩ → +1, |1⟩ → -1
    ⟨AB⟩ = P(00) + P(11) - P(01) - P(10)

    Args:
        counts: Dictionary of measurement outcomes

    Returns:
        float: Expectation value between -1 and 1
    """
    total = sum(counts.values())

    prob_00 = counts.get('00', 0) / total
    prob_01 = counts.get('01', 0) / total
    prob_10 = counts.get('10', 0) / total
    prob_11 = counts.get('11', 0) / total

    expectation = prob_00 + prob_11 - prob_01 - prob_10
    return expectation


def run_chsh_single_phase(phase_phi, theta_angles=None, shots=20000):
    """
    Run CHSH experiment for a single phase value φ using actual phase-controlled circuits

    This implementation uses the phase-controlled Bell state circuit to produce
    the physical S(φ) = S₀ cos(φ) dependence through quantum interference.

    Args:
        phase_phi: Phase parameter φ (radians)
        theta_angles: Tuple of (theta_a0, theta_a1, theta_b0, theta_b1) or None for optimal
        shots: Number of measurement shots per combination

    Returns:
        tuple: (S_parameter, expectations_list)
    """
    # Use optimal angles if not provided
    if theta_angles is None:
        theta_a0, theta_a1 = 0, np.pi/2
        theta_b0, theta_b1 = np.pi/4, -np.pi/4
    else:
        theta_a0, theta_a1, theta_b0, theta_b1 = theta_angles

    # Define the 4 measurements for CHSH
    measurements = [
        (theta_a0, theta_b0),  # ⟨A₀B₀⟩
        (theta_a0, theta_b1),  # ⟨A₀B₁⟩
        (theta_a1, theta_b0),  # ⟨A₁B₀⟩
        (theta_a1, theta_b1)   # ⟨A₁B₁⟩
    ]

    simulator = AerSimulator()
    expectations = []

    # Use the phase-controlled circuit for each measurement
    for theta_a, theta_b in measurements:
        qc = chsh_circuit_with_phase(theta_a, theta_b, phase_phi)
        job = simulator.run(transpile(qc, simulator), shots=shots)
        counts = job.result().get_counts()

        exp_val = expectation_value(counts)
        expectations.append(exp_val)

    # Calculate CHSH parameter from the actual phase-controlled measurements
    E1, E2, E3, E4 = expectations
    S = E1 + E2 + E3 - E4

    return S, expectations


def run_phase_sweep(phase_range=None, shots=10000, progress_callback=None):
    """
    Run CHSH experiment across a range of phase values

    Args:
        phase_range: Array of phase values or None for default (0 to 2π, 24 points)
        shots: Number of shots per measurement
        progress_callback: Function to call with progress updates

    Returns:
        tuple: (phase_array, S_values_array, all_expectations)
    """
    if phase_range is None:
        phase_range = np.linspace(0, 2*np.pi, 24)

    S_values = []
    all_expectations = []

    for i, phase_phi in enumerate(phase_range):
        S_phi, expectations = run_chsh_single_phase(phase_phi, shots=shots)
        S_values.append(S_phi)
        all_expectations.append(expectations)

        # Progress callback
        if progress_callback and (i + 1) % max(1, len(phase_range)//10) == 0:
            progress = (i + 1) / len(phase_range) * 100
            progress_callback(progress, phase_phi, S_phi)

    return np.array(phase_range), np.array(S_values), all_expectations


def theoretical_s_vs_phase(phase_range):
    """
    Calculate theoretical CHSH parameter vs phase

    For phase applied as RZ(φ) on Bob's qubit, the theoretical dependence is:
    S(φ) = 2√2 cos(φ) (can be negative)

    Args:
        phase_range: Array of phase values

    Returns:
        np.array: Theoretical S values (can be positive or negative)
    """
    return 2 * np.sqrt(2) * np.cos(phase_range)


def plot_s_vs_phase(phase_range, S_values, save_path=None, show_theory=True):
    """
    Plot CHSH parameter S vs phase φ with theoretical comparison

    Args:
        phase_range: Array of phase values
        S_values: Array of experimental S values
        save_path: Path to save plot (optional)
        show_theory: Whether to show theoretical curve

    Returns:
        tuple: (figure, axes) matplotlib objects
    """
    # Constants
    classical_bound = 2.0
    tsirelson_bound = 2 * np.sqrt(2)

    # Calculate max absolute value for y-axis scaling
    max_abs_s = max(abs(np.min(S_values)), abs(np.max(S_values)))
    if max_abs_s < 3.0:  # Ensure minimum range
        max_abs_s = 3.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Left plot: Full range S vs φ
    ax1.plot(phase_range, S_values, 'bo-', linewidth=2, markersize=6,
             label='Experimental', markerfacecolor='lightblue', markeredgecolor='blue')

    # Theoretical curve
    if show_theory:
        phi_theory = np.linspace(0, 2*np.pi, 200)
        S_theory = theoretical_s_vs_phase(phi_theory)
        ax1.plot(phi_theory, S_theory, 'r-', linewidth=3, alpha=0.8,
                label='Theory: 2√2 cos(φ)')

    # Add bounds (both positive and negative)
    ax1.axhline(y=classical_bound, color='red', linestyle='--', alpha=0.8,
               linewidth=2, label='Classical Bounds (±2)')
    ax1.axhline(y=-classical_bound, color='red', linestyle='--', alpha=0.8, linewidth=2)
    ax1.axhline(y=tsirelson_bound, color='green', linestyle=':', alpha=0.8,
               linewidth=2, label='Tsirelson Bounds (±2√2)')
    ax1.axhline(y=-tsirelson_bound, color='green', linestyle=':', alpha=0.8, linewidth=2)

    # Fill quantum regimes (where |S| > 2)
    ax1.fill_between([0, 2*np.pi], classical_bound, max_abs_s * 1.1,
                    color='lightcoral', alpha=0.2, label='Quantum Regime (+)')
    ax1.fill_between([0, 2*np.pi], -classical_bound, -max_abs_s * 1.1,
                    color='lightcoral', alpha=0.2, label='Quantum Regime (-)')

    ax1.set_xlabel('Phase φ (radians)', fontsize=12)
    ax1.set_ylabel('CHSH Parameter S', fontsize=12)
    ax1.set_title('CHSH Parameter S vs Phase φ', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 2*np.pi)
    # Set y-limits to show both positive and negative values
    ax1.set_ylim(-max_abs_s * 1.1, max_abs_s * 1.1)

    # Set x-axis ticks in terms of π
    ax1.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax1.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])

    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right plot: Detailed view around maximum
    max_idx = np.argmax(S_values)
    zoom_range = 5  # Number of points around maximum
    start_idx = max(0, max_idx - zoom_range//2)
    end_idx = min(len(phase_range), max_idx + zoom_range//2 + 1)

    phase_zoom = phase_range[start_idx:end_idx]
    S_zoom = S_values[start_idx:end_idx]

    ax2.plot(phase_zoom, S_zoom, 'bo-', linewidth=3, markersize=10,
             label='Experimental', markerfacecolor='lightblue', markeredgecolor='blue')

    # Theoretical curve in zoom region
    if show_theory and len(phase_zoom) > 1:
        phi_zoom_theory = np.linspace(phase_zoom[0], phase_zoom[-1], 100)
        S_zoom_theory = theoretical_s_vs_phase(phi_zoom_theory)
        ax2.plot(phi_zoom_theory, S_zoom_theory, 'r-', linewidth=3, alpha=0.8,
                label='Theory')

    # Add bounds
    ax2.axhline(y=classical_bound, color='red', linestyle='--', alpha=0.8, linewidth=2)
    ax2.axhline(y=tsirelson_bound, color='green', linestyle=':', alpha=0.8, linewidth=2)

    # Highlight maximum
    max_phase = phase_range[max_idx]
    max_S = S_values[max_idx]
    ax2.scatter([max_phase], [max_S], color='gold', s=200, marker='*',
               edgecolor='black', linewidth=2, zorder=5,
               label=f'Max: S={max_S:.3f}')

    ax2.set_xlabel('Phase φ (radians)', fontsize=12)
    ax2.set_ylabel('CHSH Parameter S', fontsize=12)
    ax2.set_title('Detailed View Around Maximum', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, (ax1, ax2)


def analyze_phase_dependence(phase_range, S_values, verbose=True):
    """
    Analyze the phase dependence of CHSH parameter

    Args:
        phase_range: Array of phase values
        S_values: Array of S parameter values
        verbose: Whether to print detailed analysis

    Returns:
        dict: Analysis results
    """
    # Find extrema
    max_idx = np.argmax(S_values)
    min_idx = np.argmin(S_values)

    max_phase = phase_range[max_idx]
    min_phase = phase_range[min_idx]
    max_S = S_values[max_idx]
    min_S = S_values[min_idx]

    # Constants
    classical_bound = 2.0
    tsirelson_bound = 2 * np.sqrt(2)

    # Bell inequality violation analysis (check |S| > 2)
    violation_count = np.sum(np.abs(S_values) > classical_bound)
    violation_percentage = (violation_count / len(S_values)) * 100

    # Theoretical comparison
    theoretical_max = tsirelson_bound
    agreement = (max_S / theoretical_max) * 100

    # Find classical-quantum boundary crossings (where |S| crosses 2)
    crossing_phases = []
    for i in range(len(S_values)-1):
        abs_s_i = abs(S_values[i])
        abs_s_next = abs(S_values[i+1])
        if (abs_s_i <= classical_bound and abs_s_next > classical_bound) or \
           (abs_s_i > classical_bound and abs_s_next <= classical_bound):
            crossing_phase = (phase_range[i] + phase_range[i+1]) / 2
            crossing_phases.append(crossing_phase)

    results = {
        'max_S': max_S,
        'min_S': min_S,
        'max_phase': max_phase,
        'min_phase': min_phase,
        'violation_count': violation_count,
        'violation_percentage': violation_percentage,
        'theoretical_agreement': agreement,
        'crossing_phases': crossing_phases,
        'range': max_S - min_S
    }

    if verbose:
        print("=" * 60)
        print("PHASE DEPENDENCE ANALYSIS")
        print("=" * 60)

        print(f"\nExtreme Values:")
        print(f"  Maximum: S = {max_S:.3f} at φ = {max_phase:.3f} rad ({np.degrees(max_phase):.1f}°)")
        print(f"  Minimum: S = {min_S:.3f} at φ = {min_phase:.3f} rad ({np.degrees(min_phase):.1f}°)")
        print(f"  Range: ΔS = {max_S - min_S:.3f}")

        print(f"\nBell Inequality Violation:")
        print(f"  Phases with |S| > 2: {violation_count}/{len(S_values)} ({violation_percentage:.1f}%)")

        print(f"\nTheoretical Comparison:")
        print(f"  Theoretical maximum: 2√2 = {theoretical_max:.3f}")
        print(f"  Experimental maximum: {max_S:.3f}")
        print(f"  Agreement: {agreement:.1f}% of theoretical")

        if crossing_phases:
            print(f"\nClassical-Quantum Boundary Crossings:")
            for i, cp in enumerate(crossing_phases):
                print(f"  Crossing {i+1}: φ ≈ {cp:.3f} rad ({np.degrees(cp):.1f}°)")

        print(f"\nPhysical Insights:")
        print(f"🌊 Phase dependence shows wave-like quantum correlations")
        print(f"📊 Maximum violation at φ = 0 (S = +2√2), φ = π (S = -2√2)")
        print(f"⚖️ Zero violation at φ = π/2, 3π/2 (S = 0)")
        print(f"🔄 Sinusoidal behavior: S = 2√2 cos(φ)")

    return results


def quick_phase_analysis(n_points=24, shots=8000):
    """
    Quick phase analysis with default parameters

    Args:
        n_points: Number of phase points to sample
        shots: Number of shots per measurement

    Returns:
        dict: Complete analysis results including data and plots
    """
    print(f"Running quick phase analysis with {n_points} points...")

    # Progress callback
    def progress_update(percent, phase, s_val):
        if percent % 25 == 0:  # Print every 25%
            print(f"Progress: {percent:.0f}% - φ = {phase:.2f}, S = {s_val:.3f}")

    # Run phase sweep
    phase_range = np.linspace(0, 2*np.pi, n_points)
    phase_data, S_data, expectations_data = run_phase_sweep(
        phase_range, shots=shots, progress_callback=progress_update
    )

    # Analysis
    analysis = analyze_phase_dependence(phase_data, S_data, verbose=True)

    # Plotting
    fig, axes = plot_s_vs_phase(phase_data, S_data)

    return {
        'phase_range': phase_data,
        'S_values': S_data,
        'expectations': expectations_data,
        'analysis': analysis,
        'figure': fig,
        'axes': axes
    }


if __name__ == "__main__":
    # Example usage
    results = quick_phase_analysis(n_points=20, shots=5000)
    plt.show()

    print(f"\n✅ Phase analysis completed!")
    print(f"📊 Maximum S = {results['analysis']['max_S']:.3f}")
    print(f"🎯 Violation percentage: {results['analysis']['violation_percentage']:.1f}%")
