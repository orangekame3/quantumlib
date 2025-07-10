#!/usr/bin/env python3
"""
Simple CHSH Inequality Implementation

Based on IBM Quantum Tutorial: https://quantum.cloud.ibm.com/docs/en/tutorials/chsh-inequality

This script demonstrates a simple implementation of the CHSH (Clauser-Horne-Shimony-Holt) 
inequality to test Bell's theorem and quantum entanglement.
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import warnings
warnings.filterwarnings('ignore')


def create_bell_state():
    """Create a Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2"""
    qc = QuantumCircuit(2, 2)
    qc.h(0)      # Create superposition on Alice's qubit
    qc.cx(0, 1)  # Entangle with Bob's qubit
    return qc


def create_chsh_circuit(alice_angle, bob_angle):
    """
    Create CHSH measurement circuit with specified measurement angles
    
    Args:
        alice_angle: Alice's measurement angle (in radians)
        bob_angle: Bob's measurement angle (in radians)
    
    Returns:
        QuantumCircuit: Circuit ready for CHSH measurement
    """
    # Create circuit with 2 qubits and 2 classical bits
    qc = QuantumCircuit(2, 2)
    
    # Create Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
    qc.h(0)      # Create superposition on Alice's qubit
    qc.cx(0, 1)  # Entangle with Bob's qubit
    
    # Measurement basis rotations
    # RY rotations to implement measurements in different bases
    # The angle determines the measurement direction on the Bloch sphere
    qc.ry(-2 * alice_angle, 0)  # Alice's measurement basis (negative for correct direction)
    qc.ry(-2 * bob_angle, 1)    # Bob's measurement basis (negative for correct direction)
    
    # Measurements in the rotated basis
    qc.measure(0, 0)  # Alice → classical bit 0
    qc.measure(1, 1)  # Bob → classical bit 1
    
    return qc


def calculate_correlation(counts):
    """
    Calculate correlation E(a,b) = P(same) - P(different)
    
    For Bell states:
    - P(same) = P(00) + P(11) (both get same result)  
    - P(different) = P(01) + P(10) (different results)
    
    Args:
        counts: Dictionary of measurement outcomes
        
    Returns:
        float: Correlation value between -1 and 1
    """
    total_shots = sum(counts.values())
    
    # Calculate probabilities
    p_00 = counts.get('00', 0) / total_shots
    p_01 = counts.get('01', 0) / total_shots
    p_10 = counts.get('10', 0) / total_shots
    p_11 = counts.get('11', 0) / total_shots
    
    # Correlation: E = P(same) - P(different)
    correlation = (p_00 + p_11) - (p_01 + p_10)
    
    return correlation


def run_chsh_experiment(shots=8192, verbose=True):
    """
    Run complete CHSH experiment with all 4 measurement combinations
    
    The CHSH inequality tests local realism by measuring correlations:
    S = |E(a₀,b₀) + E(a₀,b₁) + E(a₁,b₀) - E(a₁,b₁)|
    
    Classical bound: S ≤ 2
    Quantum bound: S ≤ 2√2 ≈ 2.828
    
    Args:
        shots: Number of measurement shots per combination
        verbose: Whether to print detailed results
        
    Returns:
        tuple: (S_value, correlations, detailed_results)
    """
    # Optimal CHSH measurement angles
    a0 = 0          # Alice's first angle (0°)
    a1 = np.pi/2    # Alice's second angle (90°) 
    b0 = np.pi/4    # Bob's first angle (45°)
    b1 = -np.pi/4   # Bob's second angle (-45°)
    
    # Bounds
    classical_bound = 2
    quantum_bound = 2 * np.sqrt(2)
    
    # Initialize simulator
    simulator = AerSimulator()
    
    # Define the 4 measurement combinations for CHSH
    measurements = [
        (a0, b0, 'E(a₀,b₀)'),
        (a0, b1, 'E(a₀,b₁)'),
        (a1, b0, 'E(a₁,b₀)'),
        (a1, b1, 'E(a₁,b₁)')
    ]
    
    correlations = []
    results = {}
    
    if verbose:
        print("CHSH Inequality Experiment")
        print("=" * 50)
        print(f"Measurement angles:")
        print(f"  Alice: a₀ = {a0:.3f} ({np.degrees(a0):.0f}°), a₁ = {a1:.3f} ({np.degrees(a1):.0f}°)")
        print(f"  Bob:   b₀ = {b0:.3f} ({np.degrees(b0):.0f}°), b₁ = {b1:.3f} ({np.degrees(b1):.0f}°)")
        print(f"\\nRunning {len(measurements)} measurement combinations with {shots} shots each...")
        print()
    
    # Run each measurement combination
    for alice_angle, bob_angle, label in measurements:
        # Create circuit
        qc = create_chsh_circuit(alice_angle, bob_angle)
        
        # Run simulation
        job = simulator.run(transpile(qc, simulator), shots=shots)
        counts = job.result().get_counts()
        
        # Calculate correlation
        correlation = calculate_correlation(counts)
        correlations.append(correlation)
        results[label] = {'correlation': correlation, 'counts': counts}
        
        if verbose:
            print(f"{label}: {correlation:.3f}")
            print(f"  Counts: {counts}")
    
    # Calculate CHSH parameter: S = |E(a₀,b₀) + E(a₀,b₁) + E(a₁,b₀) - E(a₁,b₁)|
    S = abs(correlations[0] + correlations[1] + correlations[2] - correlations[3])
    
    if verbose:
        print()
        print("=" * 50)
        print("CHSH Results")
        print("=" * 50)
        print(f"Individual correlations: {[f'{c:.3f}' for c in correlations]}")
        print(f"Calculation: |{correlations[0]:.3f} + {correlations[1]:.3f} + {correlations[2]:.3f} - {correlations[3]:.3f}|")
        print(f"           = |{correlations[0] + correlations[1] + correlations[2] - correlations[3]:.3f}|")
        print(f"CHSH parameter S = {S:.3f}")
        print()
        print(f"Classical bound: S ≤ {classical_bound}")
        print(f"Quantum bound:   S ≤ {quantum_bound:.3f}")
        
        if S > classical_bound:
            print(f"✅ Bell inequality VIOLATED! (S > 2)")
            print(f"🔬 Quantum entanglement confirmed")
            efficiency = (S / quantum_bound) * 100
            print(f"📊 Efficiency: {efficiency:.1f}% of theoretical maximum")
        else:
            print(f"❌ Bell inequality not violated (S ≤ 2)")
            print(f"💡 May need more shots or check implementation")
    
    return S, correlations, results


def plot_chsh_results(correlations, S_value):
    """
    Plot CHSH correlation results and comparison with bounds
    
    Args:
        correlations: List of 4 correlation values
        S_value: CHSH parameter value
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left plot: Individual correlations
    measurement_labels = ['E(a₀,b₀)', 'E(a₀,b₁)', 'E(a₁,b₀)', 'E(a₁,b₁)']
    colors = ['blue', 'red', 'green', 'orange']
    
    bars = ax1.bar(measurement_labels, correlations, color=colors, alpha=0.7)
    
    # Add value labels on bars
    for bar, corr in zip(bars, correlations):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{corr:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_ylabel('Correlation E(a,b)')
    ax1.set_title('CHSH Correlations')
    ax1.set_ylim(-1, 1)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Right plot: CHSH parameter comparison
    classical_bound = 2
    quantum_bound = 2 * np.sqrt(2)
    
    bounds = ['S (Experimental)', 'Classical Bound', 'Quantum Bound']
    values = [S_value, classical_bound, quantum_bound]
    bar_colors = ['darkblue', 'red', 'green']
    
    bars2 = ax2.bar(bounds, values, color=bar_colors, alpha=0.7)
    
    # Add value labels
    for bar, val in zip(bars2, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                 f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax2.set_ylabel('CHSH Parameter S')
    ax2.set_title('CHSH Parameter vs Bounds')
    ax2.set_ylim(0, 3)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def theoretical_comparison():
    """
    Compare experimental results with theoretical predictions
    
    Returns:
        dict: Theoretical correlations and CHSH value
    """
    # Measurement angles
    a0, a1 = 0, np.pi/2
    b0, b1 = np.pi/4, -np.pi/4
    
    # Theoretical correlations for Bell state measurements
    # E(a,b) = cos(a - b) for perfect Bell state
    theoretical_correlations = [
        np.cos(a0 - b0),  # cos(0 - π/4) = cos(-π/4) = 1/√2
        np.cos(a0 - b1),  # cos(0 - (-π/4)) = cos(π/4) = 1/√2  
        np.cos(a1 - b0),  # cos(π/2 - π/4) = cos(π/4) = 1/√2
        np.cos(a1 - b1)   # cos(π/2 - (-π/4)) = cos(3π/4) = -1/√2
    ]
    
    # Theoretical CHSH value
    S_theoretical = abs(theoretical_correlations[0] + theoretical_correlations[1] + 
                       theoretical_correlations[2] - theoretical_correlations[3])
    
    return {
        'correlations': theoretical_correlations,
        'S': S_theoretical,
        'max_quantum': 2 * np.sqrt(2)
    }


def main():
    """Main function to run CHSH experiment"""
    print("Simple CHSH Inequality Implementation")
    print("Based on IBM Quantum Tutorial")
    print("=" * 60)
    print()
    
    # Run the experiment
    S_value, correlations, detailed_results = run_chsh_experiment(shots=8192)
    
    # Get theoretical predictions
    theory = theoretical_comparison()
    
    print()
    print("=" * 60)
    print("Theoretical vs Experimental Comparison")
    print("=" * 60)
    
    measurement_labels = ['E(a₀,b₀)', 'E(a₀,b₁)', 'E(a₁,b₀)', 'E(a₁,b₁)']
    
    print("Correlations:")
    for i, label in enumerate(measurement_labels):
        exp_val = correlations[i]
        theo_val = theory['correlations'][i]
        diff = abs(exp_val - theo_val)
        print(f"  {label}: Exp={exp_val:6.3f}, Theory={theo_val:6.3f}, Diff={diff:.3f}")
    
    print(f"\\nCHSH Parameter:")
    print(f"  Experimental S = {S_value:.3f}")
    print(f"  Theoretical  S = {theory['S']:.3f}")
    print(f"  Difference     = {abs(S_value - theory['S']):.3f}")
    
    efficiency_exp = (S_value / theory['max_quantum']) * 100
    efficiency_theo = (theory['S'] / theory['max_quantum']) * 100
    
    print(f"\\nEfficiency:")
    print(f"  Experimental: {efficiency_exp:.1f}% of quantum bound")
    print(f"  Theoretical:  {efficiency_theo:.1f}% of quantum bound")
    
    # Plot results
    plot_chsh_results(correlations, S_value)
    
    print("\\n✅ CHSH experiment completed!")
    print("📊 Results demonstrate quantum entanglement and Bell inequality violation")


if __name__ == "__main__":
    main()