#!/usr/bin/env python3
"""
Correct CHSH Inequality Implementation

This implementation follows the exact approach from IBM Quantum documentation.
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import warnings
warnings.filterwarnings('ignore')


def create_chsh_circuit(theta_alice, theta_bob):
    """
    Create CHSH circuit with measurement angles theta_alice and theta_bob
    
    The key insight: We rotate the qubits BEFORE measurement to effectively
    measure in different bases.
    """
    # Create quantum circuit with 2 qubits and 2 classical bits
    qc = QuantumCircuit(2, 2)
    
    # Prepare Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
    qc.h(0)
    qc.cx(0, 1)
    
    # Rotate qubits to measure in different bases  
    # For CHSH, we need to rotate around Y-axis to change measurement direction
    # The factor of 2 comes from the relationship between rotation angle and measurement basis
    qc.ry(-2 * theta_alice, 0)  # Alice's measurement (negative for correct direction)
    qc.ry(-2 * theta_bob, 1)    # Bob's measurement (negative for correct direction)
    
    # Measure in computational basis (which is now the rotated basis)
    qc.measure(0, 0)  # Alice
    qc.measure(1, 1)  # Bob
    
    return qc


def run_measurement(theta_alice, theta_bob, shots=8192):
    """Run a single measurement with given angles"""
    qc = create_chsh_circuit(theta_alice, theta_bob)
    simulator = AerSimulator()
    
    # Execute the circuit
    job = simulator.run(transpile(qc, simulator), shots=shots)
    result = job.result()
    counts = result.get_counts()
    
    return counts


def calculate_expectation(counts):
    """
    Calculate expectation value E = P(same) - P(different)
    
    For CHSH, we need:
    - P(same) = P(00) + P(11) 
    - P(different) = P(01) + P(10)
    - E = P(same) - P(different)
    """
    total = sum(counts.values())
    
    # Get counts for each outcome
    count_00 = counts.get('00', 0)
    count_01 = counts.get('01', 0) 
    count_10 = counts.get('10', 0)
    count_11 = counts.get('11', 0)
    
    # Calculate probabilities
    p_same = (count_00 + count_11) / total
    p_diff = (count_01 + count_10) / total
    
    # Expectation value
    expectation = p_same - p_diff
    
    return expectation


def run_chsh_experiment():
    """
    Run the complete CHSH experiment
    
    We use the optimal angles:
    - Alice: a₀ = 0, a₁ = π/2
    - Bob: b₀ = π/4, b₁ = -π/4
    """
    
    # Optimal measurement angles for maximum CHSH violation
    a0 = 0
    a1 = np.pi/2
    b0 = np.pi/4  
    b1 = -np.pi/4
    
    print("CHSH Inequality Experiment")
    print("=" * 50)
    print(f"Alice angles: a₀ = {a0:.3f} ({np.degrees(a0):.0f}°), a₁ = {a1:.3f} ({np.degrees(a1):.0f}°)")
    print(f"Bob angles:   b₀ = {b0:.3f} ({np.degrees(b0):.0f}°), b₁ = {b1:.3f} ({np.degrees(b1):.0f}°)")
    print()
    
    # Run the 4 measurements needed for CHSH
    measurements = [
        (a0, b0, "E(a₀,b₀)"),
        (a0, b1, "E(a₀,b₁)"), 
        (a1, b0, "E(a₁,b₀)"),
        (a1, b1, "E(a₁,b₁)")
    ]
    
    expectations = []
    
    print("Running measurements...")
    for alice_angle, bob_angle, label in measurements:
        counts = run_measurement(alice_angle, bob_angle)
        expectation = calculate_expectation(counts)
        expectations.append(expectation)
        
        print(f"{label}: {expectation:.3f}")
        print(f"  Counts: {counts}")
    
    # Calculate CHSH parameter S
    E00, E01, E10, E11 = expectations
    S = abs(E00 + E01 + E10 - E11)
    
    # Bounds
    classical_bound = 2.0
    quantum_bound = 2 * np.sqrt(2)
    
    print()
    print("=" * 50)
    print("CHSH Results")
    print("=" * 50)
    print(f"E(a₀,b₀) = {E00:.3f}")
    print(f"E(a₀,b₁) = {E01:.3f}")
    print(f"E(a₁,b₀) = {E10:.3f}")
    print(f"E(a₁,b₁) = {E11:.3f}")
    print()
    print(f"S = |E(a₀,b₀) + E(a₀,b₁) + E(a₁,b₀) - E(a₁,b₁)|")
    print(f"  = |{E00:.3f} + {E01:.3f} + {E10:.3f} - {E11:.3f}|")
    print(f"  = |{E00 + E01 + E10 - E11:.3f}|")
    print(f"  = {S:.3f}")
    print()
    print(f"Classical bound: S ≤ {classical_bound}")
    print(f"Quantum bound:   S ≤ {quantum_bound:.3f}")
    
    if S > classical_bound:
        violation = (S - classical_bound) / classical_bound * 100
        efficiency = S / quantum_bound * 100
        print(f"\\n✅ BELL INEQUALITY VIOLATED!")
        print(f"🔬 Quantum entanglement confirmed")
        print(f"📊 Violation: {violation:.1f}% above classical bound")
        print(f"⚡ Efficiency: {efficiency:.1f}% of quantum limit")
    else:
        print(f"\\n❌ Bell inequality not violated")
        print(f"💡 S = {S:.3f} ≤ {classical_bound} (classical bound)")
    
    return S, expectations


def theoretical_predictions():
    """Calculate theoretical CHSH values for perfect Bell state"""
    
    # Measurement angles
    a0, a1 = 0, np.pi/2
    b0, b1 = np.pi/4, -np.pi/4
    
    # For Bell state |Φ⁺⟩, the correlation function is:
    # E(θ_a, θ_b) = cos(θ_a - θ_b)
    
    E_theo = [
        np.cos(a0 - b0),  # cos(-π/4) = 1/√2 ≈ 0.707
        np.cos(a0 - b1),  # cos(π/4) = 1/√2 ≈ 0.707
        np.cos(a1 - b0),  # cos(π/4) = 1/√2 ≈ 0.707  
        np.cos(a1 - b1)   # cos(3π/4) = -1/√2 ≈ -0.707
    ]
    
    S_theo = abs(E_theo[0] + E_theo[1] + E_theo[2] - E_theo[3])
    
    print("\\n" + "=" * 50)
    print("Theoretical Predictions (Perfect Bell State)")
    print("=" * 50)
    print(f"E(a₀,b₀) = cos({a0:.3f} - {b0:.3f}) = cos({a0-b0:.3f}) = {E_theo[0]:.3f}")
    print(f"E(a₀,b₁) = cos({a0:.3f} - {b1:.3f}) = cos({a0-b1:.3f}) = {E_theo[1]:.3f}")
    print(f"E(a₁,b₀) = cos({a1:.3f} - {b0:.3f}) = cos({a1-b0:.3f}) = {E_theo[2]:.3f}")
    print(f"E(a₁,b₁) = cos({a1:.3f} - {b1:.3f}) = cos({a1-b1:.3f}) = {E_theo[3]:.3f}")
    print()
    print(f"S_theoretical = {S_theo:.3f} = 2√2")
    print(f"Maximum quantum violation!")
    
    return S_theo, E_theo


def main():
    """Main function"""
    print("Simple CHSH Implementation - IBM Tutorial Method")
    print("=" * 60)
    print()
    
    # Run experiment
    S_exp, E_exp = run_chsh_experiment()
    
    # Show theoretical predictions
    S_theo, E_theo = theoretical_predictions()
    
    # Compare
    print("\\n" + "=" * 50)
    print("Experimental vs Theoretical")
    print("=" * 50)
    
    labels = ["E(a₀,b₀)", "E(a₀,b₁)", "E(a₁,b₀)", "E(a₁,b₁)"]
    for i, label in enumerate(labels):
        diff = abs(E_exp[i] - E_theo[i])
        print(f"{label}: Exp = {E_exp[i]:6.3f}, Theory = {E_theo[i]:6.3f}, Diff = {diff:.3f}")
    
    print(f"\\nCHSH S: Exp = {S_exp:.3f}, Theory = {S_theo:.3f}, Diff = {abs(S_exp - S_theo):.3f}")
    
    print("\\n🎯 Experiment completed!")


if __name__ == "__main__":
    main()