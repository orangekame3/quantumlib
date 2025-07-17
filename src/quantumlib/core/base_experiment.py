#!/usr/bin/env python3
"""
Base Experiment Class - 実験基底クラス
すべての量子実験クラスの基底となるクラス
"""

import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from .data_manager import SimpleDataManager

# OQTOPUS imports
try:
    from qiskit.qasm3 import dumps
    from quri_parts_oqtopus.backend import OqtopusSamplingBackend

    OQTOPUS_AVAILABLE = True
except ImportError:
    OQTOPUS_AVAILABLE = False
    OqtopusSamplingBackend = None


class BaseExperiment(ABC):
    """
    量子実験の基底クラス

    すべての具体的な実験クラス（CHSHExperiment等）がこれを継承
    共通機能：OQTOPUS接続、並列実行、データ管理
    """

    def __init__(
        self,
        experiment_name: str = None,
        oqtopus_backend: Any | None = None,
    ):
        """
        Initialize base experiment

        Args:
            experiment_name: 実験名
            oqtopus_backend: OQTOPUSバックエンド（省略時は自動作成）
        """
        self.experiment_name = (
            experiment_name or f"{self.__class__.__name__.lower()}_{int(time.time())}"
        )
        self.data_manager = SimpleDataManager(self.experiment_name)

        # OQTOPUSバックエンド設定
        if oqtopus_backend:
            self.oqtopus_backend = oqtopus_backend
            self.oqtopus_available = True
        else:
            self.oqtopus_available = OQTOPUS_AVAILABLE
            if OQTOPUS_AVAILABLE:
                self.oqtopus_backend = OqtopusSamplingBackend()
            else:
                self.oqtopus_backend = None

        # ローカルシミュレーター設定
        self.local_simulator = None
        try:
            from qiskit_aer import AerSimulator

            self.local_simulator = AerSimulator()
            self.local_simulator_available = True
        except ImportError:
            self.local_simulator_available = False

        # デフォルトOQTOPUS設定
        self.anemone_basis_gates = ["sx", "x", "rz", "cx"]
        self.transpiler_options = {
            "basis_gates": self.anemone_basis_gates,
            "optimization_level": 1,
        }
        self.mitigation_options = {
            "ro_error_mitigation": "pseudo_inverse",
        }

        # OQTOPUS用の内部構造
        self.transpiler_info = {
            "transpiler_lib": "qiskit",
            "transpiler_options": self.transpiler_options,
        }
        self.mitigation_info = self.mitigation_options

        print(f"{self.__class__.__name__}: {self.experiment_name}")
        print(f"OQTOPUS: {'Available' if self.oqtopus_available else 'Not available'}")
        print(
            f"Local Simulator: {'Available' if self.local_simulator_available else 'Not available'}"
        )

    def submit_circuit_to_oqtopus(
        self, circuit: Any, shots: int, device_id: str
    ) -> str | None:
        """
        単一回路をOQTOPUSに投入
        """
        if not self.oqtopus_available:
            print("OQTOPUS not available")
            return None

        try:
            # QASM3生成
            qasm_str = dumps(circuit)
            f"circuit_{int(time.time())}"

            # 設定動的更新
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
            print(f"OQTOPUS submission failed: {e}")
            return None

    def run_circuit_locally(self, circuit: Any, shots: int) -> dict[str, Any] | None:
        """
        ローカルシミュレーター実行
        """
        if not self.local_simulator_available:
            return None

        try:
            import uuid

            from qiskit import transpile

            # 回路のトランスパイル
            compiled_circuit = transpile(circuit, self.local_simulator)

            # シミュレーション実行
            job = self.local_simulator.run(compiled_circuit, shots=shots)
            result = job.result()
            counts = result.get_counts()

            job_id = str(uuid.uuid4())[:8]

            return {
                "job_id": job_id,
                "counts": dict(counts),
                "shots": shots,
                "success": True,
                "simulator": "local",
            }

        except Exception as e:
            print(f"Local simulation failed: {e}")
            return None

    def submit_circuits_parallel(
        self,
        circuits: list[Any],
        devices: list[str] = ["qulacs"],
        shots: int = 1024,
        parallel_workers: int = 4,
    ) -> dict[str, list[str]]:
        """
        複数回路を並列投入（改善版）
        """
        print(
            f"Submitting {len(circuits)} circuits to {len(devices)} devices using {parallel_workers} workers"
        )

        if not self.oqtopus_available:
            print("OQTOPUS not available, falling back to local simulation...")
            if self.local_simulator_available:
                return self.submit_circuits_locally(circuits, devices, shots)
            else:
                print("Local simulator also not available")
                return {device: [] for device in devices}

        all_job_ids = {device: [] for device in devices}
        submission_tasks = []

        def submit_single_circuit(circuit, device, index):
            try:
                job_id = self.submit_circuit_to_oqtopus(circuit, shots, device)
                if job_id:
                    print(
                        f"Circuit {index + 1}/{len(circuits)} → {device}: {job_id[:8]}..."
                    )
                    return device, job_id
                else:
                    print(f"Circuit {index + 1}/{len(circuits)} → {device}: failed")
                    return device, None
            except Exception as e:
                print(f"❌ Circuit {index + 1} submission error: {e}")
                return device, None

        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            # 順序を保持するために、futureとインデックスのペアを保存
            future_to_info = {}
            for device in devices:
                for i, circuit in enumerate(circuits):
                    future = executor.submit(submit_single_circuit, circuit, device, i)
                    future_to_info[future] = (device, i)
                    submission_tasks.append(future)

            # 結果を順序付きで収集
            device_results = {device: [None] * len(circuits) for device in devices}
            for future in as_completed(submission_tasks):
                device, job_id = future.result()
                original_device, original_index = future_to_info[future]
                if job_id:
                    device_results[original_device][original_index] = job_id

            # 失敗したジョブにはプレースホルダーjob_idを設定
            for device in devices:
                final_job_ids = []
                for i, job_id in enumerate(device_results[device]):
                    if job_id is not None:
                        final_job_ids.append(job_id)
                    else:
                        # 失敗した場合はプレースホルダーjob_idを生成
                        failed_job_id = f"failed_{device}_{i}_{int(time.time())}"
                        final_job_ids.append(failed_job_id)
                all_job_ids[device] = final_job_ids

        for device, jobs in all_job_ids.items():
            print(f"✅ {device}: {len(jobs)} jobs submitted")

        return all_job_ids

    def submit_circuits_locally(
        self, circuits: list[Any], devices: list[str] = ["qulacs"], shots: int = 1024
    ) -> dict[str, list[str]]:
        """
        ローカルシミュレーター用の回路投入（即座に結果も取得）
        """
        print(f"Running {len(circuits)} circuits locally...")

        all_job_ids = {}

        for device in devices:
            device_jobs = []

            for i, circuit in enumerate(circuits):
                result = self.run_circuit_locally(circuit, shots)
                if result:
                    job_id = result["job_id"]
                    device_jobs.append(job_id)

                    # 結果を内部に保存（後でcollectで取得）
                    if not hasattr(self, "_local_results"):
                        self._local_results = {}
                    self._local_results[job_id] = result

                    print(
                        f"Circuit {i + 1}/{len(circuits)} → {device}: {job_id} (local)"
                    )
                else:
                    # 失敗した場合はプレースホルダーjob_idを生成
                    failed_job_id = f"failed_{device}_{i}_{int(time.time())}"
                    device_jobs.append(failed_job_id)

                    # 失敗結果もローカルに保存
                    if not hasattr(self, "_local_results"):
                        self._local_results = {}
                    self._local_results[failed_job_id] = {
                        "job_id": failed_job_id,
                        "success": False,
                        "counts": {},
                        "error": "Local simulation failed",
                    }

                    print(f"Circuit {i + 1}/{len(circuits)} → {device}: failed")

            all_job_ids[device] = device_jobs
            print(f"{device}: {len(device_jobs)} circuits completed locally")

        return all_job_ids

    def get_oqtopus_result(
        self, job_id: str, timeout_minutes: int = 30, verbose_log: bool = False
    ) -> dict[str, Any] | None:
        """
        OQTOPUS結果取得（正しいジョブステータス取得対応）
        """
        # 失敗したジョブのプレースホルダーの場合
        if job_id.startswith("failed_"):
            return {
                "job_id": job_id,
                "success": False,
                "counts": {},
                "error": "Job submission failed",
            }

        # ローカル結果が利用可能な場合
        if hasattr(self, "_local_results") and job_id in self._local_results:
            return self._local_results[job_id]

        if not self.oqtopus_available:
            return None

        import time

        max_retries = 5
        retry_delay = 2  # 初期待機時間（秒）

        for attempt in range(max_retries):
            try:
                if verbose_log and attempt > 0:
                    print(f"⏳ Retry {attempt}/{max_retries} for {job_id[:8]}...")
                elif verbose_log:
                    print(f"⏳ Waiting for result: {job_id[:8]}...")

                job = self.oqtopus_backend.retrieve_job(job_id)

                # 正しいジョブステータス取得方法を使用
                try:
                    job_dict = job._job.to_dict()
                    status = job_dict.get("status", "unknown")

                    if verbose_log:
                        print(f"🔍 {job_id[:8]} status: {status}")

                    # 成功状態の場合
                    if status == "succeeded":
                        try:
                            result = job.result()
                            if result and hasattr(result, "counts"):
                                counts = result.counts
                                return {
                                    "job_id": job_id,
                                    "counts": dict(counts),
                                    "shots": sum(counts.values()),
                                    "status": status,
                                    "success": True,
                                }
                        except Exception as result_error:
                            if verbose_log:
                                print(
                                    f"⚠️ Result extraction failed for {job_id[:8]}: {result_error}"
                                )

                    # 明確に失敗した場合は即座に終了
                    elif status in ["failed", "cancelled", "error"]:
                        return {
                            "job_id": job_id,
                            "status": status,
                            "success": False,
                            "error": f"Job {status}",
                        }

                    # ready状態の場合は結果取得を試行
                    elif status == "ready":
                        try:
                            result = job.result()
                            if result and hasattr(result, "counts"):
                                counts = result.counts
                                return {
                                    "job_id": job_id,
                                    "counts": dict(counts),
                                    "shots": sum(counts.values()),
                                    "status": status,
                                    "success": True,
                                }
                        except Exception as ready_error:
                            if verbose_log:
                                print(
                                    f"⚠️ Ready result extraction failed for {job_id[:8]}: {ready_error}"
                                )

                    # まだ処理中の状態（submitted, running, queued等）の場合
                    elif status in ["submitted", "running", "queued", "pending"]:
                        if attempt < max_retries - 1:  # 最後の試行でなければ待機
                            wait_time = retry_delay * (2**attempt)  # 指数バックオフ
                            if verbose_log:
                                print(
                                    f"⌛ Job {job_id[:8]} still {status}, waiting {wait_time}s..."
                                )
                            time.sleep(wait_time)
                            continue
                        else:
                            # 最後の試行でも処理中の場合
                            return {
                                "job_id": job_id,
                                "status": status,
                                "success": False,
                                "error": f"Job timeout in {status} state",
                            }

                    # 不明な状態の場合
                    else:
                        if attempt < max_retries - 1:
                            wait_time = retry_delay
                            if verbose_log:
                                print(
                                    f"❓ Unknown status {status} for {job_id[:8]}, waiting {wait_time}s..."
                                )
                            time.sleep(wait_time)
                            continue
                        else:
                            return {
                                "job_id": job_id,
                                "status": status,
                                "success": False,
                                "error": f"Unknown job status: {status}",
                            }

                except Exception as status_error:
                    if verbose_log:
                        print(
                            f"⚠️ Status check failed for {job_id[:8]} (attempt {attempt + 1}): {status_error}"
                        )

                    # フォールバック: 旧式メソッドでresult取得を試行
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
                        pass

                    # 最後の試行でなければ待機してリトライ
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue

            except Exception as e:
                if verbose_log:
                    print(
                        f"❌ Result collection failed for {job_id[:8]} (attempt {attempt + 1}): {e}"
                    )

                # 最後の試行でなければ待機してリトライ
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue

        # 全ての試行が失敗した場合
        return {
            "job_id": job_id,
            "status": "timeout",
            "success": False,
            "error": f"Failed after {max_retries} attempts",
        }

    def collect_results_parallel(
        self, job_ids: dict[str, list[str]], wait_minutes: int = 30
    ) -> dict[str, list[dict[str, Any]]]:
        """
        結果を並列収集
        """
        print(f"Collecting results from {len(job_ids)} devices...")

        # ローカル結果が利用可能な場合の高速処理
        if hasattr(self, "_local_results"):
            print("Using local simulation results...")
            all_results = {}
            for device, device_job_ids in job_ids.items():
                device_results = []
                for job_id in device_job_ids:
                    if job_id in self._local_results:
                        result = self._local_results[job_id]
                        device_results.append(result)
                        print(f"{device}: {job_id[:8]}... collected (local)")
                all_results[device] = device_results
                print(f"{device}: {len(device_results)} results collected")
            return all_results

        if not self.oqtopus_available:
            print("OQTOPUS not available")
            return {}

        def collect_from_device(device_data):
            device, device_job_ids = device_data
            device_results = [None] * len(device_job_ids)

            # 順序を保持するために、インデックスと一緒に結果を収集
            for i, job_id in enumerate(device_job_ids):
                result = self.get_oqtopus_result(job_id, wait_minutes, verbose_log=True)
                if result and result.get("success", False):
                    device_results[i] = result
                    print(f"✅ {device}: {job_id[:8]}... collected")
                else:
                    status = result.get("status", "unknown") if result else "no_result"
                    print(f"❌ {device}: {job_id[:8]}... failed (status: {status})")

            # 順序を保持するため、Noneもそのまま返す
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

    # 抽象メソッド：各実験クラスで実装
    @abstractmethod
    def create_circuits(self, **kwargs) -> list[Any]:
        """実験固有の回路作成（各実験クラスで実装）"""
        pass

    @abstractmethod
    def analyze_results(
        self, results: dict[str, list[dict[str, Any]]], **kwargs
    ) -> dict[str, Any]:
        """実験固有の結果解析（各実験クラスで実装）"""
        pass

    @abstractmethod
    def save_experiment_data(
        self, results: dict[str, Any], metadata: dict[str, Any] = None
    ) -> str:
        """実験固有のデータ保存（各実験クラスで実装）"""
        pass

    # 共通保存メソッド
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
            "experiment_type": self.__class__.__name__,
            "oqtopus_config": {
                "transpiler_options": self.transpiler_options,
                "mitigation_options": self.mitigation_options,
                "basis_gates": self.anemone_basis_gates,
            },
            "metadata": metadata or {},
        }
        return self.data_manager.save_data(save_data, filename)

    def save_raw_results(
        self,
        results: dict[str, Any],
        metadata: dict[str, Any] = None,
        filename: str = "raw_results",
    ) -> str:
        """生結果保存"""
        save_data = {
            "results": results,
            "saved_at": time.time(),
            "experiment_type": self.__class__.__name__,
            "oqtopus_available": self.oqtopus_available,
            "metadata": metadata or {},
        }
        return self.data_manager.save_data(save_data, filename)

    def save_experiment_summary(self) -> str:
        """実験サマリー保存"""
        return self.data_manager.summary()

    # テンプレートメソッド：全体的な実験フロー
    def run_experiment(
        self,
        devices: list[str] = ["qulacs"],
        shots: int = 1024,
        submit_interval: float = 1.0,
        wait_minutes: int = 30,
        **kwargs,
    ) -> dict[str, Any]:
        """
        実験実行のテンプレートメソッド
        各実験クラスでオーバーライド可能
        """
        print(f"Running {self.__class__.__name__}")

        # 1. 回路作成（実験固有）
        circuits = self.create_circuits(**kwargs)
        print(f"Created {len(circuits)} circuits")

        # 2. 並列投入
        job_ids = self.submit_circuits_parallel(
            circuits, devices, shots, submit_interval
        )

        # 3. 結果収集
        raw_results = self.collect_results_parallel(job_ids, wait_minutes)

        # 4. 結果解析（実験固有）
        analyzed_results = self.analyze_results(raw_results, **kwargs)

        # 5. データ保存（実験固有）
        save_path = self.save_experiment_data(analyzed_results)

        print(f"{self.__class__.__name__} completed")
        print(f"Results saved: {save_path}")

        return {
            "job_ids": job_ids,
            "raw_results": raw_results,
            "analyzed_results": analyzed_results,
            "experiment_metadata": {
                "experiment_type": self.__class__.__name__,
                "devices": devices,
                "shots": shots,
                "circuits_count": len(circuits),
            },
        }
