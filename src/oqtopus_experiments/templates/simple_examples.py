#!/usr/bin/env python3
"""
Simple OQTOPUS Template - Simple development template based on OQTOPUS
Practical design where OQTOPUS backend is visible to users
"""

import numpy as np

from ..backend.oqtopus import QuantumExperimentSimple
from ..circuit.factory import create_chsh_circuit


def my_experiment_oqtopus():
    """
    Simple experiment based on OQTOPUS
    """
    print("🔬 My OQTOPUS Experiment")
    print("=" * 30)

    # Simple experiment setup
    exp = QuantumExperimentSimple("my_oqtopus_exp")

    # Direct editing of OQTOPUS settings
    exp.transpiler_options["optimization_level"] = 2
    exp.transpiler_options["routing_method"] = "sabre"
    exp.mitigation_options["ro_error_mitigation"] = "least_squares"

    print(f"🔧 Transpiler options: {exp.transpiler_options}")
    print(f"🔧 Mitigation options: {exp.mitigation_options}")

    # Circuit creation (using circuit_factory)
    circuits = []
    params = [(0, np.pi / 4), (np.pi / 4, 0)]

    print("\n🔧 Creating circuits:")
    for theta_a, theta_b in params:
        circuit = create_chsh_circuit(theta_a, theta_b, phase_phi=0)
        circuits.append(circuit)
        print(
            f"  Circuit: θ_A={theta_a:.3f}, θ_B={theta_b:.3f} | "
            f"Depth: {circuit.depth()}"
        )

    # OQTOPUS parallel execution
    devices = ["qulacs"]  # In real environment: ['qulacs', 'anemone']
    print(f"\n🚀 Running on OQTOPUS: {devices}")

    job_ids = exp.submit_circuits_parallel(circuits, devices, shots=500)
    results = exp.collect_results_parallel(job_ids)

    # Save results
    if results:
        exp.save_results(results, {"experiment_type": "basic_oqtopus"})
        print("✅ Results saved")

    print(f"✅ OQTOPUS experiment completed: {len(results)} device results")
    return exp, results


def my_custom_oqtopus_backend():
    """
    Use custom OQTOPUS backend
    """
    print("\n🔧 Custom OQTOPUS Backend")
    print("=" * 30)

    # Create custom OQTOPUS backend (visible to user)
    try:
        from quri_parts_oqtopus.backend import OqtopusSamplingBackend

        # User directly controls OQTOPUS backend
        custom_backend = OqtopusSamplingBackend()

        # Initialize experiment with custom backend
        exp = QuantumExperimentSimple("custom_backend", custom_backend)

        # Custom settings (direct editing)
        exp.anemone_basis_gates = ["sx", "x", "rz", "cx", "ry"]  # Additional gates
        exp.transpiler_options.update(
            {
                "basis_gates": exp.anemone_basis_gates,
                "optimization_level": 3,
                "routing_method": "sabre",
                "layout_method": "dense",
            }
        )
        exp.mitigation_options.update(
            {"ro_error_mitigation": "pseudo_inverse", "calibration_method": "standard"}
        )

        print("✅ Custom OQTOPUS backend configured")

        # Execute experiment
        circuit = create_chsh_circuit(0, np.pi / 4, 0)
        job_ids = exp.submit_circuits_parallel([circuit], ["qulacs"], shots=100)
        results = exp.collect_results_parallel(job_ids)

        return exp, results

    except ImportError:
        print("❌ OQTOPUS not available - using mock")
        exp = QuantumExperimentSimple("mock_backend")
        return exp, {}


def my_phase_scan_oqtopus():
    """
    Phase scan with OQTOPUS
    """
    print("\n🌊 OQTOPUS Phase Scan")
    print("=" * 25)

    exp = QuantumExperimentSimple("phase_scan_oqtopus")

    # Create phase scan circuits
    phases = np.linspace(0, np.pi, 4)  # 4-point scan
    circuits = []

    print("🔧 Creating phase scan circuits:")
    for phase in phases:
        circuit = create_chsh_circuit(0, np.pi / 4, phase_phi=phase)
        circuits.append(circuit)
        expected_s = 2 * np.sqrt(2) * np.cos(phase)
        print(f"  φ={phase:.3f}, Expected S={expected_s:.2f}")

    # OQTOPUS execution
    devices = ["qulacs"]
    job_ids = exp.submit_circuits_parallel(circuits, devices, shots=300)
    results = exp.collect_results_parallel(job_ids)

    # Auto-save (when Bell violation expected)
    expected_violations = sum(1 for p in phases if abs(np.cos(p)) > 1 / np.sqrt(2))
    if results and expected_violations > 0:
        metadata = {
            "phase_scan": True,
            "expected_violations": expected_violations,
            "oqtopus_used": exp.oqtopus_available,
        }
        exp.save_results(results, metadata, "oqtopus_phase_scan")
        print(f"✅ Phase scan saved (violations expected: {expected_violations})")

    return exp, results


def direct_oqtopus_usage():
    """
    Example of directly using OQTOPUS functionality
    """
    print("\n🔗 Direct OQTOPUS Usage")
    print("=" * 25)

    exp = QuantumExperimentSimple("direct_oqtopus")

    # Create circuit
    circuit = create_chsh_circuit(0, np.pi / 4, 0)

    # Direct OQTOPUS submission (visible to user)
    print("🚀 Direct OQTOPUS submission:")
    job_id = exp.submit_circuit_to_oqtopus(circuit, shots=200, device_id="qulacs")

    if job_id:
        print(f"✅ Job submitted: {job_id}")

        # Direct result retrieval
        result = exp.get_oqtopus_result(job_id)
        if result:
            print(f"✅ Result collected: {result.get('success', False)}")

            # Manual save
            exp.save_results({"direct_result": result}, filename="direct_oqtopus")

    return exp


def main():
    """
    Practical examples based on OQTOPUS
    """
    print("🧪 OQTOPUS-Based Quantum Experiments")
    print("=" * 45)

    # Basic OQTOPUS experiment
    exp1, results1 = my_experiment_oqtopus()

    # Custom backend
    exp2, results2 = my_custom_oqtopus_backend()

    # Phase scan
    exp3, results3 = my_phase_scan_oqtopus()

    # Direct OQTOPUS usage
    exp4 = direct_oqtopus_usage()

    print("\n" + "=" * 45)
    print("🎯 All OQTOPUS experiments completed!")
    print("=" * 45)

    # Benefits of OQTOPUS-based design
    print("\n🏗️ OQTOPUS-Based Design Benefits:")
    print("  ✅ Circuit creation is separated (circuit_factory)")
    print("  ✅ OQTOPUS backend is visible and customizable")
    print("  ✅ Direct access to OQTOPUS functions")
    print("  ✅ Simple, practical architecture")
    print("  ✅ No unnecessary abstraction layers")

    return exp1, exp2, exp3, exp4


if __name__ == "__main__":
    direct_oqtopus_usage()
