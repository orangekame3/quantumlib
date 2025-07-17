#!/usr/bin/env python3
"""
CHSH Research Template - Research experiment template
Customized CHSH experiment script for personal research
"""

import sys
import time

import numpy as np

# Add library paths
sys.path.append("../..")  # quantumlib root
sys.path.append("../../src")  # src directory

from src.quantumlib import CHSHExperiment


class CHSHResearchExperiment:
    """
    Research CHSH experiment class
    Adds research-specific features based on the library's CHSHExperiment
    """

    def __init__(self, research_topic: str = "chsh_research"):
        self.research_topic = research_topic
        self.experiment_log = []

        # Base experiment class
        self.base_exp = CHSHExperiment(f"{research_topic}_{int(time.time())}")

        # Research configuration
        self.setup_research_configuration()

        print(f"🔬 Research Topic: {research_topic}")
        print(f"📝 Experiment ID: {self.base_exp.experiment_name}")

    def setup_research_configuration(self):
        """Advanced research configuration"""
        # High precision settings
        self.base_exp.transpiler_options.update(
            {
                "optimization_level": 3,
                "routing_method": "sabre",
                "layout_method": "dense",
                "approximation_degree": 0.99,
            }
        )

        # Advanced error mitigation
        self.base_exp.mitigation_options.update(
            {
                "ro_error_mitigation": "least_squares",
                "zne_noise_factors": [1, 2, 3],
                "extrapolation_method": "linear",
            }
        )

        print("🔧 Research-grade configuration applied")

    def run_bell_inequality_study(self, devices=["qulacs"], shots=2000):
        """Detailed study of Bell inequality violation"""
        print("\n📊 Bell Inequality Violation Study")
        print("=" * 40)

        # High-density phase scan
        results = self.base_exp.run_phase_scan(
            devices=devices,
            phase_points=50,  # High resolution
            theta_a=0,
            theta_b=np.pi / 4,
            shots=shots,
        )

        self.log_experiment("bell_inequality_study", results)
        return results

    def run_angle_sensitivity_analysis(self, devices=["qulacs"], shots=1500):
        """Angle sensitivity analysis"""
        print("\n📐 Angle Sensitivity Analysis")
        print("=" * 35)

        # Finer angle steps
        theta_a_range = np.linspace(0, np.pi / 2, 8)
        theta_b_range = np.linspace(0, np.pi / 2, 8)

        angle_pairs = []
        for ta in theta_a_range:
            for tb in theta_b_range:
                angle_pairs.append((ta, tb))

        # Run with subset (reduce computational load)
        selected_pairs = angle_pairs[::4]  # Select every 4th pair

        results = self.base_exp.run_angle_comparison(
            devices=devices, angle_pairs=selected_pairs, shots=shots
        )

        self.log_experiment("angle_sensitivity_analysis", results)
        return results

    def run_noise_robustness_test(self, devices=["qulacs"], shots=1000):
        """Noise robustness test (simulator test)"""
        print("\n🔊 Noise Robustness Test")
        print("=" * 30)

        # Test with different shot counts
        shot_counts = [100, 300, 500, 1000, 2000]
        results_by_shots = {}

        for shots_test in shot_counts:
            print(f"🎯 Testing with {shots_test} shots...")

            result = self.base_exp.run_phase_scan(
                devices=devices,
                phase_points=10,
                theta_a=0,
                theta_b=np.pi / 4,
                shots=shots_test,
            )

            results_by_shots[shots_test] = result

            # Simple statistics display
            if "analyzed_results" in result:
                for device, analysis in result["analyzed_results"][
                    "device_results"
                ].items():
                    max_s = analysis["statistics"]["max_S_magnitude"]
                    print(f"   {device}: max|S| = {max_s:.3f}")

        self.log_experiment("noise_robustness_test", results_by_shots)
        return results_by_shots

    def run_theoretical_comparison(self, devices=["qulacs"], shots=1500):
        """Detailed comparison with theoretical values"""
        print("\n📈 Theoretical Comparison Study")
        print("=" * 35)

        # Phase selection based on theoretical predictions
        theoretical_optimal_phases = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]

        results = self.base_exp.run_experiment(
            devices=devices,
            shots=shots,
            phase_range=theoretical_optimal_phases,
            theta_a=0,
            theta_b=np.pi / 4,
        )

        # Comparison analysis between theoretical and experimental values
        if "analyzed_results" in results:
            theoretical_s = results["analyzed_results"]["theoretical_values"][
                "S_theoretical"
            ]

            print("\n📊 Theory vs Experiment:")
            for device, analysis in results["analyzed_results"][
                "device_results"
            ].items():
                experimental_s = analysis["S_values"]

                print(f"\n🔬 Device: {device}")
                for _i, (phase, theo_s, exp_s) in enumerate(
                    zip(
                        theoretical_optimal_phases,
                        theoretical_s,
                        experimental_s,
                        strict=False,
                    )
                ):
                    if not np.isnan(exp_s):
                        diff = abs(exp_s - theo_s)
                        print(
                            f"  φ={phase:.3f}: Theory={theo_s:.3f}, Exp={exp_s:.3f}, Diff={diff:.3f}"
                        )

        self.log_experiment("theoretical_comparison", results)
        return results

    def log_experiment(self, experiment_type: str, results: dict):
        """Experiment log recording"""
        log_entry = {
            "timestamp": time.time(),
            "experiment_type": experiment_type,
            "experiment_id": self.base_exp.experiment_name,
            "results_summary": self.extract_summary(results),
        }
        self.experiment_log.append(log_entry)
        print(f"📝 Logged: {experiment_type}")

    def extract_summary(self, results: dict) -> dict:
        """Extract results summary"""
        summary = {"status": "completed"}

        if isinstance(results, dict) and "analyzed_results" in results:
            analysis = results["analyzed_results"]
            if "device_results" in analysis:
                summary["devices"] = list(analysis["device_results"].keys())
                summary["max_s_values"] = {}

                for device, device_analysis in analysis["device_results"].items():
                    max_s = device_analysis["statistics"]["max_S_magnitude"]
                    summary["max_s_values"][device] = max_s

        return summary

    def generate_research_report(self):
        """Generate research report"""
        print("\n📋 Research Experiment Report")
        print("=" * 40)
        print(f"🔬 Research Topic: {self.research_topic}")
        print(f"📅 Experiments Conducted: {len(self.experiment_log)}")

        for i, entry in enumerate(self.experiment_log, 1):
            print(f"\n{i}. {entry['experiment_type']}")
            print(f"   📊 Devices: {entry['results_summary'].get('devices', 'N/A')}")
            if "max_s_values" in entry["results_summary"]:
                for device, max_s in entry["results_summary"]["max_s_values"].items():
                    print(f"   📈 {device}: max|S| = {max_s:.3f}")

        return self.experiment_log


def run_comprehensive_chsh_research():
    """Run comprehensive CHSH research"""
    print("🧪 Comprehensive CHSH Research Suite")
    print("=" * 50)

    # Create research experiment instance
    research = CHSHResearchExperiment("comprehensive_chsh_study")

    # Run series of research experiments
    devices = ["qulacs"]  # In real environment: ['qulacs', 'anemone']

    try:
        # 1. Detailed Bell inequality study
        research.run_bell_inequality_study(devices, shots=1000)

        # 2. Angle sensitivity analysis
        research.run_angle_sensitivity_analysis(devices, shots=800)

        # 3. Noise robustness test
        research.run_noise_robustness_test(devices, shots=500)

        # 4. Theoretical comparison study
        research.run_theoretical_comparison(devices, shots=1200)

        # 5. Generate research report
        research.generate_research_report()

        print("\n🎉 Comprehensive research completed!")
        print(f"📁 Check results in: {research.base_exp.data_manager.session_dir}")

    except KeyboardInterrupt:
        print("\n⚠️ Research interrupted by user")
        research.generate_research_report()

    return research


def run_quick_verification():
    """Quick verification experiment"""
    print("⚡ Quick CHSH Verification")
    print("=" * 30)

    research = CHSHResearchExperiment("quick_verification")

    # Basic verification only
    results = research.run_bell_inequality_study(["qulacs"], shots=500)

    print("✅ Quick verification completed")
    return research, results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CHSH Research Template")
    parser.add_argument(
        "--mode",
        choices=["comprehensive", "quick"],
        default="quick",
        help="Research mode",
    )
    parser.add_argument(
        "--topic", type=str, default="chsh_research", help="Research topic name"
    )

    args = parser.parse_args()

    if args.mode == "comprehensive":
        research = run_comprehensive_chsh_research()
    else:
        research, results = run_quick_verification()

    print(f"\n📊 Research completed: {args.mode} mode")
