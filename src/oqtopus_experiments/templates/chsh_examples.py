#!/usr/bin/env python3
"""
CHSH Experiment Template - Inheritance-based experiment class usage examples
Execute experiments using BaseExperiment → CHSHExperiment inheritance pattern
"""

import numpy as np

from ..experiments.chsh.chsh_experiment import CHSHExperiment


def basic_chsh_experiment():
    """
    Basic CHSH experiment
    """
    print("🔬 Basic CHSH Experiment")
    print("=" * 30)

    # Create CHSH experiment class
    exp = CHSHExperiment("basic_chsh")

    # Customize OQTOPUS settings
    exp.transpiler_options.update({"optimization_level": 2, "routing_method": "sabre"})
    exp.mitigation_options.update({"ro_error_mitigation": "pseudo_inverse"})

    # Execute phase scan experiment
    results = exp.run_phase_scan(
        devices=["qulacs"], phase_points=8, theta_a=0, theta_b=np.pi / 4, shots=500
    )

    print("✅ Basic CHSH experiment completed")

    # Check results
    if "device_results" in results["analyzed_results"]:
        for device, analysis in results["analyzed_results"]["device_results"].items():
            stats = analysis["statistics"]
            print(
                f"📊 {device}: {stats['bell_violations']} Bell violations, "
                f"max |S| = {stats['max_S_magnitude']:.3f}"
            )

    return exp, results


def angle_comparison_experiment():
    """
    Angle comparison CHSH experiment
    """
    print("\n📐 Angle Comparison Experiment")
    print("=" * 35)

    exp = CHSHExperiment("angle_comparison")

    # Compare with multiple angle pairs
    angle_pairs = [
        (0, np.pi / 4),  # Standard CHSH
        (np.pi / 4, 0),  # Swapped
        (np.pi / 8, np.pi / 8),  # Symmetric
        (0, np.pi / 8),  # Small angle
        (np.pi / 3, np.pi / 6),  # Non-standard
    ]

    results = exp.run_angle_comparison(
        devices=["qulacs"], angle_pairs=angle_pairs, shots=300
    )

    print("✅ Angle comparison completed")

    # Display best angles
    summary = results["comparison_summary"]
    if summary["best_angle_pair"]:
        best_angles = summary["best_angle_pair"]
        max_violation = summary["max_bell_violation"]
        print(f"🏆 Best angles: θ_A={best_angles[0]:.3f}, θ_B={best_angles[1]:.3f}")
        print(f"🎯 Max |S|: {max_violation:.3f}")

    return exp, results


def custom_phase_range_experiment():
    """
    CHSH experiment with custom phase range
    """
    print("\n🌊 Custom Phase Range Experiment")
    print("=" * 35)

    exp = CHSHExperiment("custom_phase")

    # Custom phase range (focused on range where Bell violation is expected)
    custom_phases = np.array([0, np.pi / 8, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])

    results = exp.run_experiment(
        devices=["qulacs"],
        shots=400,
        phase_range=custom_phases,
        theta_a=0,
        theta_b=np.pi / 4,
    )

    print("✅ Custom phase range experiment completed")

    # Check S values for each phase
    if "device_results" in results["analyzed_results"]:
        for device, analysis in results["analyzed_results"]["device_results"].items():
            s_values = analysis["S_values"]
            print(f"📊 {device} S values:")
            for _i, (phase, s) in enumerate(zip(custom_phases, s_values, strict=False)):
                violation = "🔴" if abs(s) > 2.0 else "⚪"
                print(f"  φ={phase:.3f}: S={s:.3f} {violation}")

    return exp, results


def multi_device_chsh_experiment():
    """
    CHSH experiment with multiple devices
    """
    print("\n🔀 Multi-Device CHSH Experiment")
    print("=" * 35)

    exp = CHSHExperiment("multi_device")

    # Phase scan with multiple devices
    results = exp.run_phase_scan(
        devices=["qulacs"],  # In real environment: ['qulacs', 'anemone']
        phase_points=6,
        theta_a=0,
        theta_b=np.pi / 4,
        shots=200,
    )

    print("✅ Multi-device experiment completed")

    # Device comparison
    if "comparison" in results["analyzed_results"]:
        comparison = results["analyzed_results"]["comparison"]
        print("📊 Device comparison:")

        if "bell_violation_comparison" in comparison:
            for device, violations in comparison["bell_violation_comparison"].items():
                print(f"  {device}: {violations} Bell violations")

    return exp, results


def advanced_chsh_with_custom_settings():
    """
    Advanced CHSH experiment (custom settings)
    """
    print("\n🔬 Advanced CHSH with Custom Settings")
    print("=" * 40)

    exp = CHSHExperiment("advanced_chsh")

    # Advanced OQTOPUS settings
    exp.transpiler_options.update(
        {
            "optimization_level": 3,
            "routing_method": "sabre",
            "layout_method": "dense",
            "approximation_degree": 0.99,
        }
    )

    mitigation_update = {
        "ro_error_mitigation": "least_squares",
        "zne_noise_factors": [1, 2, 3],
        "extrapolation_method": "linear",
    }
    exp.mitigation_options.update(mitigation_update)

    print("🔧 Advanced OQTOPUS settings:")
    print(f"  Optimization: level {exp.transpiler_options['optimization_level']}")
    print(f"  Error mitigation: {exp.mitigation_options['ro_error_mitigation']}")

    # High-precision phase scan
    results = exp.run_phase_scan(
        devices=["qulacs"],
        phase_points=12,
        theta_a=0,
        theta_b=np.pi / 4,
        shots=1000,
        submit_interval=1.5,
    )

    print("✅ Advanced CHSH experiment completed")

    return exp, results


def main():
    """
    Usage examples of CHSHExperiment inheritance class
    """
    print("🧪 CHSH Experiment Class Examples")
    print("=" * 45)

    # Basic CHSH experiment
    exp1, results1 = basic_chsh_experiment()

    # Angle comparison experiment
    exp2, results2 = angle_comparison_experiment()

    # Custom phase range experiment
    exp3, results3 = custom_phase_range_experiment()

    # Multi-device experiment
    exp4, results4 = multi_device_chsh_experiment()

    # Advanced settings experiment
    exp5, results5 = advanced_chsh_with_custom_settings()

    print("\n" + "=" * 45)
    print("🎯 All CHSH experiments completed!")
    print("=" * 45)

    print("\n📋 Inheritance Architecture Benefits:")
    print("  ✅ BaseExperiment: Common OQTOPUS functionality")
    print("  ✅ CHSHExperiment: CHSH-specific analysis & saving")
    print("  ✅ Extensible: Easy to create new experiment types")
    print("  ✅ Clean separation: Each experiment type has its own logic")
    print("  ✅ Reusable: Common patterns abstracted in base class")

    return exp1, exp2, exp3, exp4, exp5


if __name__ == "__main__":
    main()
