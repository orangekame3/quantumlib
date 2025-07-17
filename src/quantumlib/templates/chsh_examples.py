#!/usr/bin/env python3
"""
CHSH Experiment Template - 継承ベースの実験クラス使用例
BaseExperiment → CHSHExperiment の継承パターンで実験を実行
"""

import numpy as np

from ..experiments.chsh.chsh_experiment import CHSHExperiment


def basic_chsh_experiment():
    """
    基本的なCHSH実験
    """
    print("🔬 Basic CHSH Experiment")
    print("=" * 30)

    # CHSH実験クラス作成
    exp = CHSHExperiment("basic_chsh")

    # OQTOPUS設定カスタマイズ
    exp.transpiler_options.update({"optimization_level": 2, "routing_method": "sabre"})
    exp.mitigation_options.update({"ro_error_mitigation": "pseudo_inverse"})

    # 位相スキャン実験実行
    results = exp.run_phase_scan(
        devices=["qulacs"], phase_points=8, theta_a=0, theta_b=np.pi / 4, shots=500
    )

    print("✅ Basic CHSH experiment completed")

    # 結果確認
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
    角度比較CHSH実験
    """
    print("\n📐 Angle Comparison Experiment")
    print("=" * 35)

    exp = CHSHExperiment("angle_comparison")

    # 複数の角度ペアで比較
    angle_pairs = [
        (0, np.pi / 4),  # 標準CHSH
        (np.pi / 4, 0),  # 入れ替え
        (np.pi / 8, np.pi / 8),  # 対称
        (0, np.pi / 8),  # 小角度
        (np.pi / 3, np.pi / 6),  # 非標準
    ]

    results = exp.run_angle_comparison(
        devices=["qulacs"], angle_pairs=angle_pairs, shots=300
    )

    print("✅ Angle comparison completed")

    # ベスト角度表示
    summary = results["comparison_summary"]
    if summary["best_angle_pair"]:
        best_angles = summary["best_angle_pair"]
        max_violation = summary["max_bell_violation"]
        print(f"🏆 Best angles: θ_A={best_angles[0]:.3f}, θ_B={best_angles[1]:.3f}")
        print(f"🎯 Max |S|: {max_violation:.3f}")

    return exp, results


def custom_phase_range_experiment():
    """
    カスタム位相範囲でのCHSH実験
    """
    print("\n🌊 Custom Phase Range Experiment")
    print("=" * 35)

    exp = CHSHExperiment("custom_phase")

    # カスタム位相範囲（Bell違反が期待される範囲に集中）
    custom_phases = np.array([0, np.pi / 8, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])

    results = exp.run_experiment(
        devices=["qulacs"],
        shots=400,
        phase_range=custom_phases,
        theta_a=0,
        theta_b=np.pi / 4,
    )

    print("✅ Custom phase range experiment completed")

    # 各位相でのS値確認
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
    複数デバイスでのCHSH実験
    """
    print("\n🔀 Multi-Device CHSH Experiment")
    print("=" * 35)

    exp = CHSHExperiment("multi_device")

    # 複数デバイスでの位相スキャン
    results = exp.run_phase_scan(
        devices=["qulacs"],  # 実環境では ['qulacs', 'anemone']
        phase_points=6,
        theta_a=0,
        theta_b=np.pi / 4,
        shots=200,
    )

    print("✅ Multi-device experiment completed")

    # デバイス比較
    if "comparison" in results["analyzed_results"]:
        comparison = results["analyzed_results"]["comparison"]
        print("📊 Device comparison:")

        if "bell_violation_comparison" in comparison:
            for device, violations in comparison["bell_violation_comparison"].items():
                print(f"  {device}: {violations} Bell violations")

    return exp, results


def advanced_chsh_with_custom_settings():
    """
    高度なCHSH実験（カスタム設定）
    """
    print("\n🔬 Advanced CHSH with Custom Settings")
    print("=" * 40)

    exp = CHSHExperiment("advanced_chsh")

    # 高度なOQTOPUS設定
    exp.transpiler_options.update(
        {
            "optimization_level": 3,
            "routing_method": "sabre",
            "layout_method": "dense",
            "approximation_degree": 0.99,
        }
    )

    exp.mitigation_options.update(
        {
            "ro_error_mitigation": "least_squares",
            "zne_noise_factors": [1, 2, 3],
            "extrapolation_method": "linear",
        }
    )

    print("🔧 Advanced OQTOPUS settings:")
    print(f"  Optimization: level {exp.transpiler_options['optimization_level']}")
    print(f"  Error mitigation: {exp.mitigation_options['ro_error_mitigation']}")

    # 高精度位相スキャン
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
    CHSHExperiment継承クラスの使用例
    """
    print("🧪 CHSH Experiment Class Examples")
    print("=" * 45)

    # 基本CHSH実験
    exp1, results1 = basic_chsh_experiment()

    # 角度比較実験
    exp2, results2 = angle_comparison_experiment()

    # カスタム位相範囲実験
    exp3, results3 = custom_phase_range_experiment()

    # 複数デバイス実験
    exp4, results4 = multi_device_chsh_experiment()

    # 高度な設定実験
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
