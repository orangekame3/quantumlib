#!/usr/bin/env python3
"""
CHSH Inequality - Final Working Implementation

Based on working examples from IBM Quantum documentation and community.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def chsh_circuit(theta_a, theta_b):
    """
    Create CHSH circuit with measurement angles theta_a and theta_b.
    
    Key insight: We prepare Bell state, then apply basis rotations, then measure.
    The angles theta_a and theta_b define the measurement directions.
    """
    qc = QuantumCircuit(2, 2)
    
    # Create Bell state |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2
    # This is equivalent to |Φ⁺⟩ with a local rotation
    qc.h(0)
    qc.cx(0, 1)
    
    # Apply measurement rotations
    # These define the measurement axes on the Bloch sphere
    qc.ry(theta_a, 0)  # Alice's measurement direction
    qc.ry(theta_b, 1)  # Bob's measurement direction
    
    # Measure
    qc.measure(0, 0)
    qc.measure(1, 1)
    
    return qc


def run_circuit(theta_a, theta_b, shots=10000):
    """Run circuit and return measurement counts"""
    qc = chsh_circuit(theta_a, theta_b)
    
    simulator = AerSimulator()
    job = simulator.run(transpile(qc, simulator), shots=shots)
    return job.result().get_counts()


def expectation_value(counts):
    """
    Calculate expectation value ⟨AB⟩ where A,B ∈ {-1,+1}
    
    We map: |0⟩ → +1, |1⟩ → -1
    So: ⟨AB⟩ = P(00) + P(11) - P(01) - P(10)
    """
    total = sum(counts.values())
    
    prob_00 = counts.get('00', 0) / total
    prob_01 = counts.get('01', 0) / total  
    prob_10 = counts.get('10', 0) / total
    prob_11 = counts.get('11', 0) / total
    
    # ⟨AB⟩ = (+1)(+1)P(00) + (+1)(-1)P(01) + (-1)(+1)P(10) + (-1)(-1)P(11)
    #      = P(00) - P(01) - P(10) + P(11)
    expectation = prob_00 - prob_01 - prob_10 + prob_11
    
    return expectation


def main():
    """Run CHSH experiment"""
    
    print("CHSH Inequality Experiment - Final Implementation")
    print("=" * 60)
    
    # Optimal angles for maximum CHSH violation
    # These are chosen to maximize |S| = |E₁ + E₂ + E₃ - E₄|
    theta_a0 = 0
    theta_a1 = np.pi/2
    theta_b0 = np.pi/4
    theta_b1 = -np.pi/4
    
    print(f"Measurement angles:")
    print(f"  Alice: θ_a0 = {theta_a0:.3f} ({np.degrees(theta_a0):.0f}°)")
    print(f"         θ_a1 = {theta_a1:.3f} ({np.degrees(theta_a1):.0f}°)")
    print(f"  Bob:   θ_b0 = {theta_b0:.3f} ({np.degrees(theta_b0):.0f}°)")
    print(f"         θ_b1 = {theta_b1:.3f} ({np.degrees(theta_b1):.0f}°)")
    print()
    
    # Run the four measurements for CHSH
    measurements = [
        (theta_a0, theta_b0, "⟨A₀B₀⟩"),
        (theta_a0, theta_b1, "⟨A₀B₁⟩"), 
        (theta_a1, theta_b0, "⟨A₁B₀⟩"),
        (theta_a1, theta_b1, "⟨A₁B₁⟩")
    ]
    
    print("Running measurements...")
    expectations = []
    
    for theta_a, theta_b, label in measurements:
        counts = run_circuit(theta_a, theta_b, shots=50000)
        exp_val = expectation_value(counts)
        expectations.append(exp_val)
        
        print(f"{label} = {exp_val:.3f}")
        print(f"  Counts: {dict(sorted(counts.items()))}")
        print()
    
    # Calculate CHSH parameter  
    # S = ⟨A₀B₀⟩ + ⟨A₀B₁⟩ + ⟨A₁B₀⟩ - ⟨A₁B₁⟩ (no absolute value)
    E1, E2, E3, E4 = expectations
    S = E1 + E2 + E3 - E4
    
    # Theoretical values
    classical_bound = 2.0
    tsirelson_bound = 2 * np.sqrt(2)  # ≈ 2.828
    
    print("=" * 60)
    print("CHSH RESULTS")
    print("=" * 60)
    print(f"Expectation values:")
    print(f"  ⟨A₀B₀⟩ = {E1:6.3f}")
    print(f"  ⟨A₀B₁⟩ = {E2:6.3f}")  
    print(f"  ⟨A₁B₀⟩ = {E3:6.3f}")
    print(f"  ⟨A₁B₁⟩ = {E4:6.3f}")
    print()
    print(f"CHSH parameter:")
    print(f"  S = |⟨A₀B₀⟩ + ⟨A₀B₁⟩ + ⟨A₁B₀⟩ - ⟨A₁B₁⟩|")
    print(f"    = |{E1:.3f} + {E2:.3f} + {E3:.3f} - {E4:.3f}|")
    print(f"    = |{E1 + E2 + E3 - E4:.3f}|")
    print(f"    = {S:.3f}")
    print()
    print(f"Bounds:")
    print(f"  Classical bound:  S ≤ {classical_bound:.3f}")
    print(f"  Tsirelson bound:  S ≤ {tsirelson_bound:.3f}")
    print()
    
    if abs(S) > classical_bound:
        violation = ((abs(S) - classical_bound) / classical_bound) * 100
        efficiency = (abs(S) / tsirelson_bound) * 100
        
        print(f"🎉 BELL INEQUALITY VIOLATED!")
        print(f"✅ |S| = {abs(S):.3f} > {classical_bound:.3f} (classical bound)")
        print(f"🎯 S = {S:.3f} (sign indicates correlation direction)")
        print(f"🔬 Quantum entanglement confirmed")
        print(f"📊 Violation: {violation:.1f}% above classical limit")
        print(f"⚡ Efficiency: {efficiency:.1f}% of quantum maximum")
        
        if S > 2.5:
            print(f"🌟 Strong violation! Close to quantum limit.")
        
    else:
        print(f"❌ Bell inequality NOT violated")
        print(f"💡 |S| = {abs(S):.3f} ≤ {classical_bound:.3f}")
        print(f"🎯 S = {S:.3f}")
        print(f"🔧 May need to check implementation or increase shots")
    
    # Theoretical comparison
    print()
    print("=" * 60)
    print("THEORETICAL COMPARISON")
    print("=" * 60)
    
    # For Bell state and optimal angles, theoretical expectations are:
    theo_expectations = [
        1/np.sqrt(2),   # ⟨A₀B₀⟩ = cos(π/4)
        1/np.sqrt(2),   # ⟨A₀B₁⟩ = cos(π/4)  
        1/np.sqrt(2),   # ⟨A₁B₀⟩ = cos(π/4)
        -1/np.sqrt(2)   # ⟨A₁B₁⟩ = cos(3π/4)
    ]
    
    S_theoretical = 2 * np.sqrt(2)
    
    print(f"Theoretical (perfect Bell state):")
    labels = ["⟨A₀B₀⟩", "⟨A₀B₁⟩", "⟨A₁B₀⟩", "⟨A₁B₁⟩"]
    for i, (label, exp_val, theo_val) in enumerate(zip(labels, expectations, theo_expectations)):
        diff = abs(exp_val - theo_val)
        print(f"  {label}: Exp = {exp_val:6.3f}, Theory = {theo_val:6.3f}, |Diff| = {diff:.3f}")
    
    print(f"\\nCHSH parameter:")
    print(f"  Experimental: S = {S:.3f}")
    print(f"  Theoretical:  S = {S_theoretical:.3f}")
    print(f"  Difference:   |S_exp - S_theo| = {abs(S - S_theoretical):.3f}")
    
    accuracy = (S / S_theoretical) * 100
    print(f"  Accuracy: {accuracy:.1f}% of theoretical maximum")


if __name__ == "__main__":
    main()
