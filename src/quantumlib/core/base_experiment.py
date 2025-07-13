#!/usr/bin/env python3
"""
Base Experiment Class - 実験基底クラス
すべての量子実験クラスの基底となるクラス
"""

import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

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

    def __init__(self, experiment_name: str = None,
                 oqtopus_backend: Optional[OqtopusSamplingBackend] = None):
        """
        Initialize base experiment

        Args:
            experiment_name: 実験名
            oqtopus_backend: OQTOPUSバックエンド（省略時は自動作成）
        """
        self.experiment_name = experiment_name or f"{self.__class__.__name__.lower()}_{int(time.time())}"
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
            from qiskit import transpile
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
            "transpiler_options": self.transpiler_options
        }
        self.mitigation_info = self.mitigation_options

        print(f"{self.__class__.__name__}: {self.experiment_name}")
        print(f"OQTOPUS: {'Available' if self.oqtopus_available else 'Not available'}")
        print(f"Local Simulator: {'Available' if self.local_simulator_available else 'Not available'}")

    def submit_circuit_to_oqtopus(self, circuit: Any, shots: int,
                                device_id: str) -> Optional[str]:
        """
        単一回路をOQTOPUSに投入
        """
        if not self.oqtopus_available:
            print("OQTOPUS not available")
            return None

        try:
            # QASM3生成
            qasm_str = dumps(circuit)
            job_label = f"circuit_{int(time.time())}"

            # 設定動的更新
            self.transpiler_info["transpiler_options"] = self.transpiler_options
            self.mitigation_info = self.mitigation_options

            job = self.oqtopus_backend.sample_qasm(
                qasm_str,
                device_id=device_id,
                shots=shots,
                transpiler_info=self.transpiler_info,
                mitigation_info=self.mitigation_info
            )

            return job.job_id

        except Exception as e:
            print(f"OQTOPUS submission failed: {e}")
            return None
    
    def run_circuit_locally(self, circuit: Any, shots: int) -> Optional[Dict[str, Any]]:
        """
        ローカルシミュレーター実行
        """
        if not self.local_simulator_available:
            return None
        
        try:
            from qiskit import transpile
            import uuid
            
            # 回路のトランスパイル
            compiled_circuit = transpile(circuit, self.local_simulator)
            
            # シミュレーション実行
            job = self.local_simulator.run(compiled_circuit, shots=shots)
            result = job.result()
            counts = result.get_counts()
            
            job_id = str(uuid.uuid4())[:8]
            
            return {
                'job_id': job_id,
                'counts': dict(counts),
                'shots': shots,
                'success': True,
                'simulator': 'local'
            }
            
        except Exception as e:
            print(f"Local simulation failed: {e}")
            return None

    def submit_circuits_parallel(self, circuits: List[Any],
                               devices: List[str] = ['qulacs'],
                               shots: int = 1024,
                               submit_interval: float = 1.0) -> Dict[str, List[str]]:
        """
        複数回路を並列投入
        """
        print(f"Submitting {len(circuits)} circuits to {len(devices)} devices")

        # OQTOPUSが利用できない場合、ローカルシミュレーションにフォールバック
        if not self.oqtopus_available:
            print("OQTOPUS not available, trying local simulation...")
            if self.local_simulator_available:
                return self.submit_circuits_locally(circuits, devices, shots)
            else:
                print("Local simulator also not available")
                return {device: [] for device in devices}

        all_job_ids = {}

        def submit_to_device(device):
            device_jobs = []
            for i, circuit in enumerate(circuits):
                try:
                    job_id = self.submit_circuit_to_oqtopus(circuit, shots, device)
                    if job_id:
                        device_jobs.append(job_id)
                        print(f"Circuit {i+1}/{len(circuits)} → {device}: {job_id[:8]}...")
                    else:
                        print(f"Circuit {i+1}/{len(circuits)} → {device}: failed")

                    if submit_interval > 0 and i < len(circuits) - 1:
                        time.sleep(submit_interval)

                except Exception as e:
                    print(f"❌ Circuit {i+1} submission error: {e}")

            return device, device_jobs

        # 並列投入
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            futures = [executor.submit(submit_to_device, device) for device in devices]

            for future in as_completed(futures):
                device, job_ids = future.result()
                all_job_ids[device] = job_ids
                print(f"✅ {device}: {len(job_ids)} jobs submitted")

        return all_job_ids
    
    def submit_circuits_locally(self, circuits: List[Any], 
                               devices: List[str] = ['qulacs'],
                               shots: int = 1024) -> Dict[str, List[str]]:
        """
        ローカルシミュレーター用の回路投入（即座に結果も取得）
        """
        print(f"Running {len(circuits)} circuits locally...")
        
        all_job_ids = {}
        
        for device in devices:
            device_jobs = []
            device_results = []  # ローカル結果を即座に保存
            
            for i, circuit in enumerate(circuits):
                result = self.run_circuit_locally(circuit, shots)
                if result:
                    job_id = result['job_id']
                    device_jobs.append(job_id)
                    
                    # 結果を内部に保存（後でcollectで取得）
                    if not hasattr(self, '_local_results'):
                        self._local_results = {}
                    self._local_results[job_id] = result
                    
                    print(f"Circuit {i+1}/{len(circuits)} → {device}: {job_id} (local)")
                else:
                    print(f"Circuit {i+1}/{len(circuits)} → {device}: failed")
            
            all_job_ids[device] = device_jobs
            print(f"{device}: {len(device_jobs)} circuits completed locally")
        
        return all_job_ids

    def get_oqtopus_result(self, job_id: str, timeout_minutes: int = 30) -> Optional[Dict[str, Any]]:
        """
        OQTOPUS結果取得（ローカル結果もサポート）
        """
        # ローカル結果が利用可能な場合
        if hasattr(self, '_local_results') and job_id in self._local_results:
            return self._local_results[job_id]
        
        if not self.oqtopus_available:
            return None

        try:
            print(f"⏳ Waiting for result: {job_id[:8]}...")

            job = self.oqtopus_backend.retrieve_job(job_id)

            # 結果待機
            max_wait = timeout_minutes * 60
            wait_time = 0

            while wait_time < max_wait:
                try:
                    result = job.result()
                    if result and hasattr(result, 'counts'):
                        counts = result.counts
                        return {
                            'job_id': job_id,
                            'counts': dict(counts),
                            'shots': sum(counts.values()),
                            'success': True
                        }
                except:
                    time.sleep(5)
                    wait_time += 5

            print(f"⏳ Timeout waiting for {job_id[:8]}...")
            return None

        except Exception as e:
            print(f"❌ Result collection failed for {job_id}: {e}")
            return None

    def collect_results_parallel(self, job_ids: Dict[str, List[str]],
                                wait_minutes: int = 30) -> Dict[str, List[Dict[str, Any]]]:
        """
        結果を並列収集
        """
        print(f"Collecting results from {len(job_ids)} devices...")

        # ローカル結果が利用可能な場合の高速処理
        if hasattr(self, '_local_results'):
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
            futures = [executor.submit(collect_from_device, item) for item in job_ids.items()]

            for future in as_completed(futures):
                device, results = future.result()
                all_results[device] = results
                print(f"✅ {device}: {len(results)} results collected")

        return all_results

    # 抽象メソッド：各実験クラスで実装
    @abstractmethod
    def create_circuits(self, **kwargs) -> List[Any]:
        """実験固有の回路作成（各実験クラスで実装）"""
        pass

    @abstractmethod
    def analyze_results(self, results: Dict[str, List[Dict[str, Any]]], **kwargs) -> Dict[str, Any]:
        """実験固有の結果解析（各実験クラスで実装）"""
        pass

    @abstractmethod
    def save_experiment_data(self, results: Dict[str, Any],
                           metadata: Dict[str, Any] = None) -> str:
        """実験固有のデータ保存（各実験クラスで実装）"""
        pass

    # 共通保存メソッド
    def save_job_ids(self, job_ids: Dict[str, List[str]],
                     metadata: Dict[str, Any] = None,
                     filename: str = "job_ids") -> str:
        """ジョブID保存"""
        save_data = {
            'job_ids': job_ids,
            'submitted_at': time.time(),
            'experiment_type': self.__class__.__name__,
            'oqtopus_config': {
                'transpiler_options': self.transpiler_options,
                'mitigation_options': self.mitigation_options,
                'basis_gates': self.anemone_basis_gates
            },
            'metadata': metadata or {}
        }
        return self.data_manager.save_data(save_data, filename)

    def save_raw_results(self, results: Dict[str, Any],
                        metadata: Dict[str, Any] = None,
                        filename: str = "raw_results") -> str:
        """生結果保存"""
        save_data = {
            'results': results,
            'saved_at': time.time(),
            'experiment_type': self.__class__.__name__,
            'oqtopus_available': self.oqtopus_available,
            'metadata': metadata or {}
        }
        return self.data_manager.save_data(save_data, filename)

    def save_experiment_summary(self) -> str:
        """実験サマリー保存"""
        return self.data_manager.summary()

    # テンプレートメソッド：全体的な実験フロー
    def run_experiment(self, devices: List[str] = ['qulacs'],
                      shots: int = 1024,
                      submit_interval: float = 1.0,
                      wait_minutes: int = 30,
                      **kwargs) -> Dict[str, Any]:
        """
        実験実行のテンプレートメソッド
        各実験クラスでオーバーライド可能
        """
        print(f"Running {self.__class__.__name__}")

        # 1. 回路作成（実験固有）
        circuits = self.create_circuits(**kwargs)
        print(f"Created {len(circuits)} circuits")

        # 2. 並列投入
        job_ids = self.submit_circuits_parallel(circuits, devices, shots, submit_interval)

        # 3. 結果収集
        raw_results = self.collect_results_parallel(job_ids, wait_minutes)

        # 4. 結果解析（実験固有）
        analyzed_results = self.analyze_results(raw_results, **kwargs)

        # 5. データ保存（実験固有）
        save_path = self.save_experiment_data(analyzed_results)

        print(f"{self.__class__.__name__} completed")
        print(f"Results saved: {save_path}")

        return {
            'job_ids': job_ids,
            'raw_results': raw_results,
            'analyzed_results': analyzed_results,
            'experiment_metadata': {
                'experiment_type': self.__class__.__name__,
                'devices': devices,
                'shots': shots,
                'circuits_count': len(circuits)
            }
        }
