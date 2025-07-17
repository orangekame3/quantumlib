#!/usr/bin/env python3
"""
Simple OQTOPUS Template - OQTOPUSベースのシンプル開発テンプレート
OQTOPUSバックエンドがユーザーに見える、実用的な設計
"""

import numpy as np

from ..backend.oqtopus import QuantumExperimentSimple
from ..circuit.factory import create_chsh_circuit


def my_experiment_oqtopus():
    """
    OQTOPUSベースのシンプル実験
    """
    print("🔬 My OQTOPUS Experiment")
    print("=" * 30)

    # シンプルな実験セットアップ
    exp = QuantumExperimentSimple("my_oqtopus_exp")

    # OQTOPUS設定を直接編集
    exp.transpiler_options["optimization_level"] = 2
    exp.transpiler_options["routing_method"] = "sabre"
    exp.mitigation_options["ro_error_mitigation"] = "least_squares"

    print(f"🔧 Transpiler options: {exp.transpiler_options}")
    print(f"🔧 Mitigation options: {exp.mitigation_options}")

    # 回路作成（circuit_factory使用）
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

    # OQTOPUS並列実行
    devices = ["qulacs"]  # 実環境では ['qulacs', 'anemone']
    print(f"\n🚀 Running on OQTOPUS: {devices}")

    job_ids = exp.submit_circuits_parallel(circuits, devices, shots=500)
    results = exp.collect_results_parallel(job_ids)

    # 保存
    if results:
        exp.save_results(results, {"experiment_type": "basic_oqtopus"})
        print("✅ Results saved")

    print(f"✅ OQTOPUS experiment completed: {len(results)} device results")
    return exp, results


def my_custom_oqtopus_backend():
    """
    カスタムOQTOPUSバックエンドを使用
    """
    print("\n🔧 Custom OQTOPUS Backend")
    print("=" * 30)

    # カスタムOQTOPUSバックエンド作成（ユーザーに見える）
    try:
        from quri_parts_oqtopus.backend import OqtopusSamplingBackend

        # ユーザーがOQTOPUSバックエンドを直接制御
        custom_backend = OqtopusSamplingBackend()

        # カスタムバックエンドで実験初期化
        exp = QuantumExperimentSimple("custom_backend", custom_backend)

        # カスタム設定（直接編集）
        exp.anemone_basis_gates = ["sx", "x", "rz", "cx", "ry"]  # 追加ゲート
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

        # 実験実行
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
    OQTOPUSでの位相スキャン
    """
    print("\n🌊 OQTOPUS Phase Scan")
    print("=" * 25)

    exp = QuantumExperimentSimple("phase_scan_oqtopus")

    # 位相スキャン回路作成
    phases = np.linspace(0, np.pi, 4)  # 4点スキャン
    circuits = []

    print("🔧 Creating phase scan circuits:")
    for phase in phases:
        circuit = create_chsh_circuit(0, np.pi / 4, phase_phi=phase)
        circuits.append(circuit)
        expected_s = 2 * np.sqrt(2) * np.cos(phase)
        print(f"  φ={phase:.3f}, Expected S={expected_s:.2f}")

    # OQTOPUS実行
    devices = ["qulacs"]
    job_ids = exp.submit_circuits_parallel(circuits, devices, shots=300)
    results = exp.collect_results_parallel(job_ids)

    # 自動保存（Bell違反期待時）
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
    OQTOPUS機能を直接使用する例
    """
    print("\n🔗 Direct OQTOPUS Usage")
    print("=" * 25)

    exp = QuantumExperimentSimple("direct_oqtopus")

    # 回路作成
    circuit = create_chsh_circuit(0, np.pi / 4, 0)

    # 直接OQTOPUS投入（ユーザーが見える）
    print("🚀 Direct OQTOPUS submission:")
    job_id = exp.submit_circuit_to_oqtopus(circuit, shots=200, device_id="qulacs")

    if job_id:
        print(f"✅ Job submitted: {job_id}")

        # 直接結果取得
        result = exp.get_oqtopus_result(job_id)
        if result:
            print(f"✅ Result collected: {result.get('success', False)}")

            # 手動保存
            exp.save_results({"direct_result": result}, filename="direct_oqtopus")

    return exp


def main():
    """
    OQTOPUSベースの実用例
    """
    print("🧪 OQTOPUS-Based Quantum Experiments")
    print("=" * 45)

    # 基本OQTOPUS実験
    exp1, results1 = my_experiment_oqtopus()

    # カスタムバックエンド
    exp2, results2 = my_custom_oqtopus_backend()

    # 位相スキャン
    exp3, results3 = my_phase_scan_oqtopus()

    # 直接OQTOPUS使用
    exp4 = direct_oqtopus_usage()

    print("\n" + "=" * 45)
    print("🎯 All OQTOPUS experiments completed!")
    print("=" * 45)

    # OQTOPUS設計の利点
    print("\n🏗️ OQTOPUS-Based Design Benefits:")
    print("  ✅ Circuit creation is separated (circuit_factory)")
    print("  ✅ OQTOPUS backend is visible and customizable")
    print("  ✅ Direct access to OQTOPUS functions")
    print("  ✅ Simple, practical architecture")
    print("  ✅ No unnecessary abstraction layers")

    return exp1, exp2, exp3, exp4


if __name__ == "__main__":
    direct_oqtopus_usage()
