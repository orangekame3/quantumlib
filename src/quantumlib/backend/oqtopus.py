#!/usr/bin/env python3
"""
Quantum Experiment Simple - OQTOPUSベース・シンプル設計
回路作成は分離、OQTOPUSバックエンド部分はユーザーに見える設計
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np

from ..circuit.factory import create_chsh_circuit
from ..core.data_manager import SimpleDataManager

# OQTOPUS imports (ユーザーに見える)
try:
    from quri_parts_oqtopus.backend import OqtopusSamplingBackend

    OQTOPUS_AVAILABLE = True
except ImportError:
    OQTOPUS_AVAILABLE = False
    OqtopusSamplingBackend = None


class QuantumExperimentSimple:
    """
    シンプル量子実験クラス

    設計方針:
    - 回路作成は分離（circuit_factory使用）
    - OQTOPUSバックエンドはユーザーに見える
    - 必要最小限の抽象化
    """

    def __init__(
        self,
        experiment_name: str = None,
        oqtopus_backend: OqtopusSamplingBackend | None = None,
    ):
        """
        Initialize quantum experiment

        Args:
            experiment_name: 実験名
            oqtopus_backend: OQTOPUSバックエンド（省略時は自動作成）
        """
        self.experiment_name = experiment_name or f"quantum_exp_{int(time.time())}"
        self.data_manager = SimpleDataManager(self.experiment_name)

        # OQTOPUSバックエンド設定（ユーザーに見える）
        if oqtopus_backend:
            self.oqtopus_backend = oqtopus_backend
            self.oqtopus_available = True
        else:
            self.oqtopus_available = OQTOPUS_AVAILABLE
            if OQTOPUS_AVAILABLE:
                self.oqtopus_backend = OqtopusSamplingBackend()
            else:
                self.oqtopus_backend = None

        # OQTOPUS設定（ユーザーが直接編集可能）
        self.anemone_basis_gates = ["sx", "x", "rz", "cx"]

        # transpiler_options - ユーザーが直接アクセス
        self.transpiler_options = {
            "basis_gates": self.anemone_basis_gates,
            "optimization_level": 1,
        }

        # mitigation_options - ユーザーが直接アクセス
        self.mitigation_options = {
            "ro_error_mitigation": "pseudo_inverse",
        }

        # OQTOPUS用の内部構造（後方互換性）
        self.transpiler_info = {
            "transpiler_lib": "qiskit",
            "transpiler_options": self.transpiler_options,
        }
        self.mitigation_info = self.mitigation_options

        print(f"🧪 QuantumExperiment: {self.experiment_name}")
        print(f"🔧 OQTOPUS: {'✅' if self.oqtopus_available else '❌'}")

    def create_chsh_circuit(
        self, theta_a: float, theta_b: float, phase_phi: float = 0
    ) -> Any:
        """
        CHSH回路作成（circuit_factoryを使用）
        """
        return create_chsh_circuit(theta_a, theta_b, phase_phi)

    def submit_circuit_to_oqtopus(
        self, circuit: Any, shots: int, device_id: str
    ) -> str | None:
        """
        単一回路をOQTOPUSに投入（ユーザーに見える実装）

        Args:
            circuit: Qiskit回路
            shots: ショット数
            device_id: デバイスID

        Returns:
            ジョブID
        """
        if not self.oqtopus_available:
            print("❌ OQTOPUS not available")
            return None

        try:
            # QASM3を標準採用
            from qiskit.qasm3 import dumps

            qasm_str = dumps(circuit)

            f"circuit_{int(time.time())}"

            # transpiler_info, mitigation_infoを動的更新
            self.transpiler_info["transpiler_options"] = self.transpiler_options
            self.mitigation_info = self.mitigation_options

            job = self.oqtopus_backend.sample_qasm(
                qasm_str,
                device_id=device_id,
                shots=shots,
                transpiler_info=self.transpiler_info,
                mitigation_info=self.mitigation_info,
            )

            return job.job_id

        except Exception as e:
            print(f"❌ OQTOPUS submission failed: {e}")
            return None

    def submit_circuits_parallel(
        self,
        circuits: list[Any],
        devices: list[str] = ["qulacs"],
        shots: int = 1024,
        submit_interval: float = 1.0,
    ) -> dict[str, list[str]]:
        """
        複数回路を並列投入
        """
        print(f"🚀 Submitting {len(circuits)} circuits to {len(devices)} devices")

        if not self.oqtopus_available:
            print("❌ OQTOPUS not available")
            return {device: [] for device in devices}

        all_job_ids = {}

        def submit_to_device(device):
            device_jobs = []
            for i, circuit in enumerate(circuits):
                try:
                    job_id = self.submit_circuit_to_oqtopus(circuit, shots, device)
                    if job_id:
                        device_jobs.append(job_id)
                        print(
                            f"✅ Circuit {i + 1}/{len(circuits)} → {device}: {job_id[:8]}..."
                        )
                    else:
                        print(f"❌ Circuit {i + 1}/{len(circuits)} → {device}: failed")

                    # サーバー負荷軽減
                    if submit_interval > 0 and i < len(circuits) - 1:
                        time.sleep(submit_interval)

                except Exception as e:
                    print(f"❌ Circuit {i + 1} submission error: {e}")

            return device, device_jobs

        # 並列投入
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            futures = [executor.submit(submit_to_device, device) for device in devices]

            for future in as_completed(futures):
                device, job_ids = future.result()
                all_job_ids[device] = job_ids
                print(f"✅ {device}: {len(job_ids)} jobs submitted")

        return all_job_ids

    def get_oqtopus_result(
        self, job_id: str, timeout_minutes: int = 30, verbose_log: bool = False
    ) -> dict[str, Any] | None:
        """
        OQTOPUS結果取得（ユーザーに見える実装）

        Args:
            job_id: ジョブID
            timeout_minutes: タイムアウト（分）
            verbose_log: 詳細ログ出力の有効/無効

        Returns:
            測定結果
        """
        if not self.oqtopus_available:
            return None

        try:
            # 詳細ログは有効時のみ出力
            if verbose_log:
                print(f"⏳ Waiting for result: {job_id[:8]}...")

            # OQTOPUS結果取得
            job = self.oqtopus_backend.retrieve_job(job_id)

            # 結果待機（簡易実装）
            import time

            max_wait = timeout_minutes * 60
            wait_time = 0

            while wait_time < max_wait:
                try:
                    result = job.result()
                    if result and hasattr(result, "counts"):
                        counts = result.counts
                        return {
                            "job_id": job_id,
                            "counts": dict(counts),
                            "shots": sum(counts.values()),
                            "success": True,
                        }
                except Exception:
                    time.sleep(5)
                    wait_time += 5

            print(f"⏳ Timeout waiting for {job_id[:8]}...")
            return None

        except Exception as e:
            print(f"❌ Result collection failed for {job_id}: {e}")
            return None

    def collect_results_parallel(
        self, job_ids: dict[str, list[str]], wait_minutes: int = 30
    ) -> dict[str, list[dict[str, Any]]]:
        """
        結果を並列収集
        """
        print(f"⏳ Collecting results from {len(job_ids)} devices...")

        if not self.oqtopus_available:
            print("❌ OQTOPUS not available")
            return {}

        def collect_from_device(device_data):
            device, device_job_ids = device_data
            device_results = []

            for job_id in device_job_ids:
                result = self.get_oqtopus_result(job_id, wait_minutes)
                if result:
                    device_results.append(result)
                    print(f"✅ {device}: {job_id[:8]}... collected")
                else:
                    print(f"❌ {device}: {job_id[:8]}... failed")

            return device, device_results

        all_results = {}

        # 並列収集
        with ThreadPoolExecutor(max_workers=len(job_ids)) as executor:
            futures = [
                executor.submit(collect_from_device, item) for item in job_ids.items()
            ]

            for future in as_completed(futures):
                device, results = future.result()
                all_results[device] = results
                print(f"✅ {device}: {len(results)} results collected")

        return all_results

    def run_chsh_experiment(
        self,
        phase_points: int = 20,
        devices: list[str] = ["qulacs"],
        shots: int = 1024,
        submit_interval: float = 2.0,
        wait_minutes: int = 30,
    ) -> dict[str, Any]:
        """
        CHSH実験を実行
        """
        print(f"🎯 CHSH Experiment: {phase_points} points, {shots} shots")

        # 位相スキャン回路作成（circuit_factory使用）
        phase_range = np.linspace(0, 2 * np.pi, phase_points)
        circuits = []

        for phi in phase_range:
            circuit = self.create_chsh_circuit(0, np.pi / 4, phase_phi=phi)
            circuits.append(circuit)

        print(f"🔧 Created {len(circuits)} CHSH circuits")

        # 並列実行
        job_ids = self.submit_circuits_parallel(
            circuits, devices, shots, submit_interval
        )
        results = self.collect_results_parallel(job_ids, wait_minutes)

        # 理論値計算
        S_theoretical = 2 * np.sqrt(2) * np.cos(phase_range)

        return {
            "job_ids": job_ids,
            "results": results,
            "phase_range": phase_range.tolist(),
            "S_theoretical": S_theoretical.tolist(),
            "experiment_metadata": {
                "phase_points": phase_points,
                "devices": devices,
                "shots": shots,
                "oqtopus_available": self.oqtopus_available,
            },
        }

    def save_job_ids(
        self,
        job_ids: dict[str, list[str]],
        metadata: dict[str, Any] = None,
        filename: str = "job_ids",
    ) -> str:
        """ジョブID保存"""
        save_data = {
            "job_ids": job_ids,
            "submitted_at": time.time(),
            "oqtopus_config": {
                "transpiler_options": self.transpiler_options,
                "mitigation_options": self.mitigation_options,
                "basis_gates": self.anemone_basis_gates,
            },
            "metadata": metadata or {},
        }
        return self.data_manager.save_data(save_data, filename)

    def save_results(
        self,
        results: dict[str, Any],
        metadata: dict[str, Any] = None,
        filename: str = "results",
    ) -> str:
        """実験結果保存"""
        save_data = {
            "results": results,
            "saved_at": time.time(),
            "oqtopus_available": self.oqtopus_available,
            "metadata": metadata or {},
        }
        return self.data_manager.save_data(save_data, filename)

    def save_experiment_summary(self) -> str:
        """実験サマリー保存"""
        return self.data_manager.summary()


# 便利関数（シンプル版）
def run_chsh_comparison_simple(
    devices: list[str] = ["qulacs"],
    phase_points: int = 20,
    shots: int = 1024,
    submit_interval: float = 2.0,
    experiment_name: str = None,
) -> dict[str, Any]:
    """
    CHSH比較実験を簡単実行（シンプル版）
    """
    exp = QuantumExperimentSimple(experiment_name)

    return exp.run_chsh_experiment(
        phase_points=phase_points,
        devices=devices,
        shots=shots,
        submit_interval=submit_interval,
    )
