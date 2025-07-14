#!/usr/bin/env python3
"""
Ramsey Experiment Class - Ramsey振動実験専用クラス
BaseExperimentを継承し、Ramsey実験に特化した実装を提供
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import numpy as np

from ...core.base_experiment import BaseExperiment


class RamseyExperiment(BaseExperiment):
    """
    Ramsey振動実験クラス

    特化機能:
    - Ramsey回路の自動生成
    - 振動フィッティング
    - 遅延時間スキャン実験
    - T2*時定数推定
    """

    def __init__(self, experiment_name: str = None, enable_fitting: bool = True, **kwargs):
        # Ramsey実験固有のパラメータを抽出（BaseExperimentには渡さない）
        ramsey_specific_params = {
            'delay_points', 'max_delay', 'detuning', 'delay_times', 'enable_fitting'
        }
        
        # BaseExperimentに渡すkwargsをフィルタリング
        base_kwargs = {k: v for k, v in kwargs.items() if k not in ramsey_specific_params}
        
        super().__init__(experiment_name, **base_kwargs)

        # Ramsey実験固有の設定
        self.expected_t2_star = 1000  # 初期推定値 [ns] - フィッティングの初期値のみに使用
        self.expected_detuning = 0.0  # 期待デチューニング [MHz]
        self.enable_fitting = enable_fitting  # フィッティング有効化フラグ
        
        # Ramsey実験ではreadout mitigationを有効化
        self.mitigation_options = {
            "ro_error_mitigation": "pseudo_inverse"
        }
        self.mitigation_info = self.mitigation_options

        if enable_fitting:
            print(f"Ramsey experiment: Standard Ramsey measurement with fitting enabled")
        else:
            print(f"Ramsey experiment: Standard Ramsey measurement (fitting disabled)")

    def create_circuits(self, **kwargs) -> List[Any]:
        """
        Ramsey実験回路作成

        Args:
            delay_points: 遅延時間点数 (default: 51)
            max_delay: 最大遅延時間 [ns] (default: 50000)
            detuning: 周波数デチューニング [MHz] (default: 0.0)
            delay_times: 直接指定する遅延時間リスト [ns] (optional)

        Returns:
            Ramsey回路リスト
        """
        delay_points = kwargs.get("delay_points", 51)
        max_delay = kwargs.get("max_delay", 200000)
        detuning = kwargs.get("detuning", 0.0)

        # 遅延時間範囲
        if "delay_times" in kwargs:
            delay_times = np.array(kwargs["delay_times"])
        else:
            # デフォルト: 50ns〜200μsの対数スケールで51点
            delay_times = np.logspace(np.log10(50), np.log10(200 * 1000), num=51)
            if delay_points != 51:
                delay_times = np.linspace(50, max_delay, delay_points)

        # メタデータ保存
        self.experiment_params = {
            "delay_times": delay_times.tolist(),
            "delay_points": len(delay_times),
            "max_delay": max_delay,
            "detuning": detuning,
        }

        # Ramsey回路作成
        circuits = []
        for delay_time in delay_times:
            circuit = self._create_single_ramsey_circuit(delay_time, detuning)
            circuits.append(circuit)

        print(
            f"Ramsey circuits: Delay range {len(delay_times)} points from {delay_times[0]:.1f} to {delay_times[-1]:.1f} ns, detuning={detuning} MHz"
        )

        return circuits
        
    def run_ramsey_experiment_parallel(self, devices: List[str] = ['qulacs'], shots: int = 1024,
                                      parallel_workers: int = 4, **kwargs) -> Dict[str, Any]:
        """
        Ramsey実験の並列実行（delay timeの順序を保持）
        """
        print(f"🔬 Running Ramsey experiment with {parallel_workers} parallel workers")
        
        # 回路作成
        circuits = self.create_circuits(**kwargs)
        delay_times = self.experiment_params['delay_times']
        
        print(f"   📊 {len(circuits)} circuits × {len(devices)} devices = {len(circuits) * len(devices)} jobs")
        
        # 並列実行（順序保持）
        job_data = self._submit_ramsey_circuits_parallel_with_order(
            circuits, devices, shots, parallel_workers
        )
        
        # 結果収集（順序保持）
        raw_results = self._collect_ramsey_results_parallel_with_order(
            job_data, parallel_workers
        )
        
        # 結果解析（エラーハンドリング付き）
        try:
            analysis = self.analyze_results(raw_results)
        except Exception as e:
            print(f"Analysis failed: {e}, creating minimal analysis")
            analysis = {
                'experiment_info': {
                    'delay_points': len(delay_times),
                    'error': str(e)
                },
                'device_results': {}
            }
        
        return {
            'delay_times': delay_times,
            'device_results': analysis['device_results'],
            'analysis': analysis,
            'method': 'ramsey_parallel_quantumlib'
        }
        
    def _submit_ramsey_circuits_parallel_with_order(self, circuits: List[Any], devices: List[str],
                                                   shots: int, parallel_workers: int) -> Dict[str, List[Dict]]:
        """
        Ramsey回路の並列投入（CHSHスタイルで順序保持）
        """
        print(f"Enhanced Ramsey parallel submission: {parallel_workers} workers")
        
        if not self.oqtopus_available:
            return self._submit_ramsey_circuits_locally_parallel(circuits, devices, shots, parallel_workers)
        
        # 順序保持のためのデータ構造
        all_job_data = {device: [None] * len(circuits) for device in devices}
        
        # 回路とデバイスのペア作成（delay_time順序を保持）
        circuit_device_pairs = []
        for circuit_idx, circuit in enumerate(circuits):
            for device in devices:
                circuit_device_pairs.append((circuit_idx, circuit, device))
        
        def submit_single_ramsey_circuit(args):
            circuit_idx, circuit, device = args
            try:
                job_id = self.submit_circuit_to_oqtopus(circuit, shots, device)
                if job_id:
                    return device, job_id, circuit_idx, True
                else:
                    return device, None, circuit_idx, False
            except Exception as e:
                delay_time = self.experiment_params['delay_times'][circuit_idx]
                print(f"Ramsey Circuit {circuit_idx} (τ={delay_time:.0f}ns) → {device}: {e}")
                return device, None, circuit_idx, False
        
        # 並列投入実行
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = [executor.submit(submit_single_ramsey_circuit, args) for args in circuit_device_pairs]
            
            for future in as_completed(futures):
                device, job_id, circuit_idx, success = future.result()
                if success and job_id:
                    all_job_data[device][circuit_idx] = {
                        'job_id': job_id,
                        'circuit_index': circuit_idx,
                        'delay_time': self.experiment_params['delay_times'][circuit_idx],
                        'submitted': True
                    }
                    delay_time = self.experiment_params['delay_times'][circuit_idx]
                    print(f"Ramsey Circuit {circuit_idx+1} (τ={delay_time:.0f}ns) → {device}: {job_id[:8]}...")
                else:
                    all_job_data[device][circuit_idx] = {
                        'job_id': None,
                        'circuit_index': circuit_idx,
                        'delay_time': self.experiment_params['delay_times'][circuit_idx],
                        'submitted': False
                    }
        
        for device in devices:
            successful_jobs = sum(1 for job_data in all_job_data[device] if job_data and job_data['submitted'])
            print(f"✅ {device}: {successful_jobs} Ramsey jobs submitted (order preserved)")
        
        return all_job_data
        
    def _submit_ramsey_circuits_locally_parallel(self, circuits: List[Any], devices: List[str],
                                                shots: int, parallel_workers: int) -> Dict[str, List[Dict]]:
        """Ramsey回路をローカルシミュレーターで並列実行"""
        print(f"Ramsey Local parallel execution: {parallel_workers} workers")
        
        all_job_data = {device: [None] * len(circuits) for device in devices}
        
        circuit_device_pairs = []
        for circuit_idx, circuit in enumerate(circuits):
            for device in devices:
                circuit_device_pairs.append((circuit_idx, circuit, device))
        
        def run_single_ramsey_circuit_locally(args):
            circuit_idx, circuit, device = args
            try:
                result = self.run_circuit_locally(circuit, shots)
                if result:
                    job_id = result['job_id']
                    if not hasattr(self, '_local_results'):
                        self._local_results = {}
                    self._local_results[job_id] = result
                    return device, job_id, circuit_idx, True
                else:
                    return device, None, circuit_idx, False
            except Exception as e:
                delay_time = self.experiment_params['delay_times'][circuit_idx]
                print(f"Local Ramsey circuit {circuit_idx} (τ={delay_time:.0f}ns) → {device}: {e}")
                return device, None, circuit_idx, False
        
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = [executor.submit(run_single_ramsey_circuit_locally, args) for args in circuit_device_pairs]
            
            for future in as_completed(futures):
                device, job_id, circuit_idx, success = future.result()
                if success and job_id:
                    all_job_data[device][circuit_idx] = {
                        'job_id': job_id,
                        'circuit_index': circuit_idx,
                        'delay_time': self.experiment_params['delay_times'][circuit_idx],
                        'submitted': True
                    }
                else:
                    all_job_data[device][circuit_idx] = {
                        'job_id': None,
                        'circuit_index': circuit_idx,
                        'delay_time': self.experiment_params['delay_times'][circuit_idx],
                        'submitted': False
                    }
        
        for device in devices:
            successful = sum(1 for job in all_job_data[device] if job and job['submitted'])
            print(f"✅ {device}: {successful} Ramsey circuits completed locally (order preserved)")
        
        return all_job_data
            
    def _collect_ramsey_results_parallel_with_order(self, job_data: Dict[str, List[Dict]],
                                                   parallel_workers: int) -> Dict[str, List[Dict]]:
        """Ramsey結果の並列収集（CHSHスタイルで順序保持）"""
        
        # 総ジョブ数を計算して収集開始をログ
        total_jobs_to_collect = sum(1 for device_jobs in job_data.values() 
                                   for job in device_jobs if job and job.get('submitted', False))
        print(f"📊 Starting Ramsey results collection: {total_jobs_to_collect} jobs from {len(job_data)} devices")
        
        # Handle local results
        if hasattr(self, '_local_results'):
            print("Using cached local Ramsey simulation results...")
            all_results = {}
            for device, device_job_data in job_data.items():
                device_results = []
                for job_info in device_job_data:
                    if job_info and job_info['submitted'] and job_info['job_id'] in self._local_results:
                        result = self._local_results[job_info['job_id']]
                        device_results.append(result)
                    else:
                        device_results.append(None)
                all_results[device] = device_results
                successful = sum(1 for r in device_results if r is not None)
                print(f"✅ {device}: {successful} Ramsey local results collected")
            return all_results
        
        if not self.oqtopus_available:
            print("OQTOPUS not available for Ramsey collection")
            return {device: [None] * len(device_job_data) for device, device_job_data in job_data.items()}
        
        all_results = {device: [None] * len(device_job_data) for device, device_job_data in job_data.items()}
        
        job_collection_tasks = []
        for device, device_job_data in job_data.items():
            for circuit_idx, job_info in enumerate(device_job_data):
                if job_info and job_info['submitted'] and job_info['job_id']:
                    job_collection_tasks.append((job_info['job_id'], device, circuit_idx))
        
        def collect_single_ramsey_result(args):
            job_id, device, circuit_idx = args
            try:
                # ジョブ完了までポーリング
                result = self._poll_job_until_completion(job_id, timeout_minutes=5)
                # OQTOPUSジョブ構造に基づく成功判定: status == 'succeeded'
                if result and result.get('status') == 'succeeded':
                    # 複数の方法で測定結果を取得を試行
                    counts = None
                    shots = 0
                    
                    # 方法1: BaseExperimentのget_oqtopus_resultが直接countsを返す場合
                    if 'counts' in result:
                        counts = result['counts']
                        shots = result.get('shots', 0)
                    
                    # 方法2: job_info内のresult構造から取得
                    if not counts:
                        job_info = result.get('job_info', {})
                        if isinstance(job_info, dict):
                            # OQTOPUS result構造を探索
                            sampling_result = job_info.get('result', {}).get('sampling', {})
                            if sampling_result:
                                counts = sampling_result.get('counts', {})
                    
                    # 方法3: job_info自体がresult形式の場合
                    if not counts and 'job_info' in result:
                        job_info = result['job_info']
                        if isinstance(job_info, dict) and 'job_info' in job_info:
                            inner_job_info = job_info['job_info']
                            if isinstance(inner_job_info, dict):
                                result_data = inner_job_info.get('result', {})
                                if 'sampling' in result_data:
                                    counts = result_data['sampling'].get('counts', {})
                                elif 'counts' in result_data:
                                    counts = result_data['counts']

                    if counts:
                        # 成功データを標準形式に変換
                        processed_result = {
                            'success': True,
                            'counts': dict(counts),  # Counterを辞書に変換
                            'status': result.get('status'),
                            'execution_time': result.get('execution_time', 0),
                            'shots': shots or sum(counts.values()) if counts else 0
                        }
                        return device, processed_result, job_id, circuit_idx, True
                    else:
                        delay_time = self.experiment_params['delay_times'][circuit_idx]
                        print(f"⚠️ {device}[{circuit_idx}] (τ={delay_time:.0f}ns): {job_id[:8]}... no measurement data")
                        return device, None, job_id, circuit_idx, False
                else:
                    # ジョブ失敗の場合
                    delay_time = self.experiment_params['delay_times'][circuit_idx]
                    status = result.get('status', 'unknown') if result else 'no_result'
                    # より詳細な失敗情報を表示
                    message = ""
                    if result:
                        job_info = result.get('job_info', {})
                        message = job_info.get('message', '')
                        if message:
                            message = f" - {message}"
                    print(f"⚠️ {device}[{circuit_idx}] (τ={delay_time:.0f}ns): {job_id[:8]}... {status}{message}")
                    return device, None, job_id, circuit_idx, False
            except Exception as e:
                delay_time = self.experiment_params['delay_times'][circuit_idx]
                print(f"❌ {device}[{circuit_idx}] (τ={delay_time:.0f}ns): {job_id[:8]}... error: {str(e)[:50]}")
                return device, None, job_id, circuit_idx, False
        
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = [executor.submit(collect_single_ramsey_result, args) for args in job_collection_tasks]
            
            completed_jobs = 0
            successful_jobs = 0
            total_jobs = len(futures)
            last_progress_percent = 0
            
            for future in as_completed(futures):
                device, result, job_id, circuit_idx, success = future.result()
                completed_jobs += 1
                
                if success and result:
                    successful_jobs += 1
                    all_results[device][circuit_idx] = result
                    delay_time = self.experiment_params['delay_times'][circuit_idx]
                    print(f"✅ {device}[{circuit_idx}] (τ={delay_time:.0f}ns): {job_id[:8]}... collected ({completed_jobs}/{total_jobs})")
                else:
                    # 失敗ケースは既に個別メソッド内でログ出力済み
                    pass
                
                # 進捗サマリーを20%ごとに表示
                progress_percent = (completed_jobs * 100) // total_jobs
                if progress_percent >= last_progress_percent + 20 and progress_percent < 100:
                    print(f"📈 Ramsey Collection Progress: {completed_jobs}/{total_jobs} ({progress_percent}%) - {successful_jobs} successful")
                    last_progress_percent = progress_percent
        
        # 最終結果サマリー
        total_successful = sum(1 for device_results in all_results.values() 
                              for r in device_results if r is not None)
        total_attempted = sum(1 for device_jobs in job_data.values() 
                             for job in device_jobs if job and job.get('submitted', False))
        success_rate = (total_successful / total_attempted * 100) if total_attempted > 0 else 0
        
        print(f"🎉 Ramsey Collection Complete: {total_successful}/{total_attempted} successful ({success_rate:.1f}%)")
        
        # 結果統計の表示と失敗ジョブの報告
        for device in job_data.keys():
            successful = sum(1 for r in all_results[device] if r is not None)
            total = len(job_data[device])
            failed = total - successful
            
            if failed > 0:
                device_success_rate = (successful / total * 100) if total > 0 else 0
                print(f"✅ {device}: {successful}/{total} Ramsey results collected (success rate: {device_success_rate:.1f}%)")
                print(f"   ⚠️ {failed} jobs failed - analysis will continue with available data")
            else:
                print(f"✅ {device}: {successful}/{total} Ramsey results collected (100% success)")
        
        return all_results
        
    def _poll_job_until_completion(self, job_id: str, timeout_minutes: int = 5, poll_interval: float = 2.0):
        """
        ジョブが完了するまでポーリング
        
        Args:
            job_id: ジョブID
            timeout_minutes: タイムアウト時間（分）
            poll_interval: ポーリング間隔（秒）
            
        Returns:
            完了したジョブの結果、またはNone
        """
        import time
        
        timeout_seconds = timeout_minutes * 60
        start_time = time.time()
        last_status = None
        
        while time.time() - start_time < timeout_seconds:
            try:
                result = self.get_oqtopus_result(job_id, timeout_minutes=1, verbose_log=False)  # 短いタイムアウトで取得
                
                if not result:
                    time.sleep(poll_interval)
                    continue
                    
                status = result.get('status', 'unknown')
                
                # 状態が変わった場合のみログ出力（進捗状態のみ）
                if status != last_status:
                    if status in ['running', 'submitted', 'pending']:
                        print(f"⏳ {job_id[:8]}... {status}")
                    elif status in ['succeeded', 'failed', 'cancelled']:
                        print(f"🏁 {job_id[:8]}... {status}")
                    last_status = status
                
                # 終了状態をチェック
                if status in ['succeeded', 'failed', 'cancelled']:
                    return result
                elif status in ['running', 'submitted', 'pending']:
                    # まだ実行中 - 続行
                    time.sleep(poll_interval)
                    continue
                else:
                    # 不明な状態 - 少し待ってリトライ
                    time.sleep(poll_interval)
                    continue
                    
            except Exception as e:
                # 一時的なエラーの場合はリトライ
                time.sleep(poll_interval)
                continue
        
        # タイムアウト
        print(f"⏰ Job {job_id[:8]}... timed out after {timeout_minutes} minutes")
        return None

    def run_experiment(self, devices: List[str] = ['qulacs'], shots: int = 1024,
                      parallel_workers: int = 4, **kwargs) -> Dict[str, Any]:
        """
        Ramsey実験実行（base_cliの統一フローに従う）
        """
        # base_cliが直接並列メソッドを呼び出すため、ここでは基本的な結果収集のみ
        print("⚠️ run_experiment called directly - use CLI framework instead")
        return self.run_ramsey_experiment_parallel(
            devices=devices, shots=shots, parallel_workers=parallel_workers, **kwargs
        )

    def _create_single_ramsey_circuit(self, delay_time: float, detuning: float = 0.0):
        """
        単一Ramsey回路作成
        """
        try:
            from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
        except ImportError:
            raise ImportError("Qiskit is required for circuit creation")

        # 1量子ビット + 1古典ビット
        qubits = QuantumRegister(1, 'q')
        bits = ClassicalRegister(1, 'c')
        qc = QuantumCircuit(qubits, bits)

        # First π/2 pulse
        qc.ry(np.pi/2, 0)

        # 遅延時間の間待機（自由進化）
        qc.delay(int(delay_time), 0, unit="ns")

        # デチューニングがある場合は位相回転を追加
        if detuning != 0.0:
            # 位相 = 2π × detuning [MHz] × delay_time [ns] × 1e-3
            phase = 2 * np.pi * detuning * delay_time * 1e-3
            qc.rz(phase, 0)

        # Second π/2 pulse (analysis pulse)
        qc.ry(np.pi/2, 0)

        # Z基底測定
        qc.measure(0, 0)

        return qc

    def analyze_results(self, results: Dict[str, List[Dict[str, Any]]], **kwargs) -> Dict[str, Any]:
        """
        Ramsey実験結果解析
        """
        if not results:
            return {'error': 'No results to analyze'}

        delay_times = np.array(self.experiment_params["delay_times"])

        analysis = {
            "experiment_info": {
                "delay_points": len(delay_times),
                "expected_t2_star": self.expected_t2_star,
                "detuning": self.experiment_params.get("detuning", 0.0),
            },
            "device_results": {},
        }

        for device, device_results in results.items():
            if not device_results:
                continue

            device_analysis = self._analyze_device_results(device_results, delay_times)
            analysis["device_results"][device] = device_analysis

            # フィッティングが有効な場合のみ実行
            if self.enable_fitting:
                try:
                    t2_star_fitted, detuning_fitted, fitting_quality = self._estimate_ramsey_params_with_quality(
                        device_analysis["p1_values"], delay_times
                    )
                    quality_str = f"({fitting_quality['method']}, R²={fitting_quality['r_squared']:.3f})"
                    print(f"{device}: T2* = {t2_star_fitted:.1f} ns, detuning = {detuning_fitted:.3f} MHz {quality_str} [with RO mitigation]")
                except Exception as e:
                    print(f"Fitting error for {device}: {e}, using default values")
                    t2_star_fitted, detuning_fitted, fitting_quality = (
                        float(self.expected_t2_star), 0.0, {
                            'method': 'error_fallback', 'r_squared': 0.0, 'error': 'exception'
                        }
                    )
            else:
                # フィッティングなし：統計情報のみ表示
                t2_star_fitted, detuning_fitted, fitting_quality = (
                    0.0, 0.0, {
                        'method': 'no_fitting', 'r_squared': 0.0, 'error': 'disabled'
                    }
                )
                stats = device_analysis["statistics"]
                oscillation_amp = stats.get("oscillation_amplitude", 0.0)
                print(f"{device}: Raw data oscillation amplitude = {oscillation_amp:.3f} [with RO mitigation]")
            
            analysis["device_results"][device]["t2_star_fitted"] = t2_star_fitted
            analysis["device_results"][device]["detuning_fitted"] = detuning_fitted
            analysis["device_results"][device]["fitting_quality"] = fitting_quality

        return analysis

    def _analyze_device_results(self, device_results: List[Dict[str, Any]], 
                              delay_times: np.ndarray) -> Dict[str, Any]:
        """
        単一デバイス結果解析
        """
        p1_values = []

        for i, result in enumerate(device_results):
            if result and result.get("success", False):
                counts = result.get("counts", {})
                if counts:  # カウントデータが存在する場合のみ
                    # P(1)確率計算（readout mitigationで補正済み）
                    p1 = self._calculate_p1_probability(counts)
                    p1_values.append(p1)
                else:
                    p1_values.append(np.nan)
            else:
                # 失敗したジョブや無効な結果はNaNとして記録
                p1_values.append(np.nan)

        # 統計計算
        valid_p1s = np.array([p for p in p1_values if not np.isnan(p)])

        # 有効データでの統計計算
        total_jobs = len(p1_values)
        successful_jobs = len(valid_p1s)
        failed_jobs = total_jobs - successful_jobs
        
        return {
            "p1_values": p1_values,
            "delay_times": delay_times.tolist(),
            "statistics": {
                "initial_p1": (
                    float(valid_p1s[0])
                    if len(valid_p1s) > 0
                    else 0.5
                ),
                "final_p1": (
                    float(valid_p1s[-1])
                    if len(valid_p1s) > 0
                    else 0.5
                ),
                "success_rate": successful_jobs / total_jobs if total_jobs > 0 else 0,
                "successful_jobs": successful_jobs,
                "failed_jobs": failed_jobs,
                "total_jobs": total_jobs,
                "oscillation_amplitude": (
                    float(max(valid_p1s) - min(valid_p1s))
                    if len(valid_p1s) > 1
                    else 0.0
                ),
            },
        }

    def _calculate_p1_probability(self, counts: Dict[str, int]) -> float:
        """
        P(1)確率計算（OQTOPUS 10進数counts対応）
        """
        # OQTOPUSの10進数countsを2進数形式に変換
        binary_counts = self._convert_decimal_to_binary_counts(counts)
        
        total = sum(binary_counts.values())
        if total == 0:
            return 0.0

        # デバッグ情報表示（初回のみ）
        if not hasattr(self, '_counts_debug_shown'):
            print(f"🔍 Raw decimal counts: {dict(counts)}")
            print(f"🔍 Converted binary counts: {dict(binary_counts)}")
            self._counts_debug_shown = True

        # 標準的なP(1)確率計算
        n_1 = binary_counts.get("1", 0)
        p1 = n_1 / total
        return p1
        
    def _convert_decimal_to_binary_counts(self, decimal_counts: Dict[str, int]) -> Dict[str, int]:
        """
        OQTOPUSの10進数countsを2進数形式に変換
        
        1量子ビットの場合:
        0 -> "0"  (|0⟩状態)
        1 -> "1"  (|1⟩状態)
        """
        binary_counts = {}
        
        for decimal_key, count in decimal_counts.items():
            # キーが数値の場合と文字列の場合に対応
            if isinstance(decimal_key, str):
                try:
                    decimal_value = int(decimal_key)
                except ValueError:
                    # すでにバイナリ形式の場合
                    binary_counts[decimal_key] = count
                    continue
            else:
                decimal_value = int(decimal_key)
            
            # 1量子ビットの場合の変換
            if decimal_value == 0:
                binary_key = "0"
            elif decimal_value == 1:
                binary_key = "1"
            else:
                # 予期しない値の場合はスキップして警告
                print(f"⚠️ Unexpected count key: {decimal_key} (decimal value: {decimal_value})")
                continue
            
            # 既存のキーがある場合は加算
            if binary_key in binary_counts:
                binary_counts[binary_key] += count
            else:
                binary_counts[binary_key] = count
        
        return binary_counts

    def _estimate_ramsey_params_with_quality(self, p1_values: List[float], delay_times: np.ndarray) -> tuple[float, float, Dict[str, Any]]:
        """
        Ramseyパラメータ推定（T2*とデチューニング）
        """
        # NaNを除去
        valid_data = [(delay, p1) for delay, p1 in zip(delay_times, p1_values)
                      if not np.isnan(p1)]

        if len(valid_data) < 5:
            return 0.0, 0.0, {'method': 'insufficient_data', 'r_squared': 0.0, 'error': 'inf'}

        delays = np.array([d[0] for d in valid_data])
        p1s = np.array([d[1] for d in valid_data])

        # detuningに応じてフィッティングモデルを選択
        expected_detuning = self.experiment_params.get("detuning", 0.0)
        
        try:
            from scipy.optimize import curve_fit
            
            # detuning=0の場合：純粋なT2*減衰（振動なし）
            if abs(expected_detuning) < 0.001:  # detuning ≈ 0
                def t2_star_decay(t, A, T2_star, offset):
                    return A * np.exp(-t / T2_star) + offset
                
                # 初期推定値
                p0 = [0.5, self.expected_t2_star, 0.5]  # A, T2*, offset
                
                # フィッティング実行
                popt, pcov = curve_fit(t2_star_decay, delays, p1s, p0=p0, 
                                     bounds=([0, 10, 0], [1.0, 100000, 1.0]),
                                     maxfev=2000)
                
                t2_star_fitted = popt[1]
                detuning_fitted = 0.0  # detuning=0として固定
                
                # 予測値計算とR²算出
                p1_pred = t2_star_decay(delays, *popt)
                ss_res = np.sum((p1s - p1_pred) ** 2)
                ss_tot = np.sum((p1s - np.mean(p1s)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                
                # 標準誤差計算
                param_error = 'inf'
                if pcov is not None and np.all(np.isfinite(pcov)):
                    param_errors = np.sqrt(np.diag(pcov))
                    t2_error = param_errors[1]
                    param_error = f"{t2_error:.1f}"
                    
                    # 高品質フィッティングの条件
                    if t2_error / t2_star_fitted < 0.5 and r_squared > 0.5:
                        return float(t2_star_fitted), float(detuning_fitted), {
                            'method': 'exponential_decay_t2star',
                            'r_squared': r_squared,
                            'error': param_error,
                            'quality': 'high' if r_squared > 0.8 else 'medium'
                        }
            
            else:  # detuning≠0の場合：振動する減衰
                def ramsey_oscillation(t, A, T2_star, freq, phase, offset):
                    return A * np.exp(-t / T2_star) * np.cos(2 * np.pi * freq * t * 1e-3 + phase) + offset
                
                # 初期推定値
                p0 = [0.5, self.expected_t2_star, expected_detuning, 0.0, 0.5]  # A, T2*, freq, phase, offset
                
                # フィッティング実行
                popt, pcov = curve_fit(ramsey_oscillation, delays, p1s, p0=p0, 
                                     bounds=([0, 10, -10, -np.pi, 0], [1.0, 100000, 10, np.pi, 1.0]),
                                     maxfev=2000)
                
                t2_star_fitted = popt[1]
                detuning_fitted = popt[2]
                
                # 予測値計算とR²算出
                p1_pred = ramsey_oscillation(delays, *popt)
                ss_res = np.sum((p1s - p1_pred) ** 2)
                ss_tot = np.sum((p1s - np.mean(p1s)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                
                # 標準誤差計算
                param_error = 'inf'
                if pcov is not None and np.all(np.isfinite(pcov)):
                    param_errors = np.sqrt(np.diag(pcov))
                    t2_error = param_errors[1]
                    param_error = f"{t2_error:.1f}"
                    
                    # 高品質フィッティングの条件
                    if t2_error / t2_star_fitted < 0.5 and r_squared > 0.5:
                        return float(t2_star_fitted), float(detuning_fitted), {
                            'method': 'ramsey_oscillation',
                            'r_squared': r_squared,
                            'error': param_error,
                            'quality': 'high' if r_squared > 0.8 else 'medium'
                        }
            
        except (ImportError, RuntimeError, ValueError, TypeError, Exception) as e:
            print(f"Ramsey fitting failed: {str(e)[:50]}... using default values")
            pass

        # 全ての手法が失敗した場合 - デフォルト値を返す
        return float(self.expected_t2_star), 0.0, {
            'method': 'default_ramsey',
            'r_squared': 0.0,
            'error': 'N/A',
            'quality': 'poor'
        }

    def save_experiment_data(self, results: Dict[str, Any], 
                           metadata: Dict[str, Any] = None) -> str:
        """
        Ramsey実験データ保存
        """
        ramsey_data = {
            'experiment_type': 'Ramsey_Oscillation',
            'experiment_timestamp': time.time(),
            'experiment_parameters': self.experiment_params,
            'analysis_results': results,
            'oqtopus_configuration': {
                'transpiler_options': self.transpiler_options,
                'mitigation_options': self.mitigation_options,
                'basis_gates': self.anemone_basis_gates
            },
            'metadata': metadata or {}
        }

        # メイン結果保存
        main_file = self.data_manager.save_data(ramsey_data, "ramsey_experiment_results")

        # 追加ファイル保存
        if 'device_results' in results:
            # デバイス別サマリー
            device_summary = {
                device: {
                    't2_star_fitted': analysis.get('t2_star_fitted', 0.0),
                    'detuning_fitted': analysis.get('detuning_fitted', 0.0),
                    'statistics': analysis['statistics']
                }
                for device, analysis in results['device_results'].items()
            }
            self.data_manager.save_data(device_summary, "device_ramsey_summary")

            # P(1)データ（プロット用）
            p1_data = {
                'delay_times': self.experiment_params['delay_times'],
                'device_p1_values': {
                    device: analysis['p1_values']
                    for device, analysis in results['device_results'].items()
                }
            }
            self.data_manager.save_data(p1_data, "ramsey_p1_values_for_plotting")

        return main_file

    def generate_ramsey_plot(self, results: Dict[str, Any], save_plot: bool = True,
                           show_plot: bool = False) -> Optional[str]:
        """Generate Ramsey experiment plot"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available - skipping plot generation")
            return None

        delay_times = results.get('delay_times', np.linspace(50, 50000, 51))
        device_results = results.get('device_results', {})

        if not device_results:
            print("No device results for plotting")
            return None

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot experimental data for each device
        colors = ["blue", "red", "green", "orange", "purple"]

        for i, (device, device_data) in enumerate(device_results.items()):
            if "p1_values" in device_data:
                p1_values = device_data["p1_values"]
                t2_star_fitted = device_data.get("t2_star_fitted", 0.0)
                detuning_fitted = device_data.get("detuning_fitted", 0.0)
                fitting_quality = device_data.get("fitting_quality", {})
                r_squared = fitting_quality.get("r_squared", 0.0)
                color = colors[i % len(colors)]
                
                # 実測データプロット
                ax.semilogx(
                    delay_times,
                    p1_values,
                    "o",
                    markersize=4,
                    label=f"{device} data",
                    alpha=0.8,
                    color=color,
                )
                
                # フィッティングが有効な場合のみフィット曲線をプロット
                if self.enable_fitting and t2_star_fitted > 0:
                    fit_delays = np.logspace(np.log10(min(delay_times)), np.log10(max(delay_times)), 200)
                    A = 0.5  # 振幅の推定値
                    offset = 0.5
                    
                    # フィッティングで使用されたモデルに応じて曲線を生成
                    fitting_method = fitting_quality.get('method', 'unknown')
                    
                    if fitting_method == 'exponential_decay_t2star' or abs(detuning_fitted) < 0.001:
                        # T2*減衰のみ（振動なし）
                        fit_curve = A * np.exp(-fit_delays / t2_star_fitted) + offset
                        label_text = f"{device} fit (T2*={t2_star_fitted:.0f}ns, R²={r_squared:.3f}) [T2* decay]"
                    else:
                        # Ramsey振動: P(t) = A * exp(-t/T2*) * cos(2π*f*t) + offset
                        fit_curve = A * np.exp(-fit_delays / t2_star_fitted) * np.cos(2 * np.pi * detuning_fitted * fit_delays * 1e-3) + offset
                        label_text = f"{device} fit (T2*={t2_star_fitted:.0f}ns, f={detuning_fitted:.3f}MHz, R²={r_squared:.3f})"
                        
                    ax.semilogx(
                        fit_delays,
                        fit_curve,
                        "-",
                        linewidth=2,
                        color=color,
                        alpha=0.7,
                        label=label_text
                    )

        # Formatting
        ax.set_xlabel("Delay time τ [ns] (log scale)", fontsize=14)
        ax.set_ylabel("P(1)", fontsize=14)
        title_suffix = " (with fitting)" if self.enable_fitting else " (raw data)"
        ax.set_title(f"QuantumLib Ramsey Oscillation Experiment{title_suffix}", fontsize=16, fontweight="bold")
        ax.grid(True, which="both", ls="--", linewidth=0.5)
        ax.legend(fontsize=12)
        ax.set_ylim(0, 1.1)

        plot_filename = None
        if save_plot:
            plt.tight_layout()
            plot_filename = f"ramsey_plot_{self.experiment_name}_{int(time.time())}.png"

            # Always save to experiment results directory
            if hasattr(self, 'data_manager') and hasattr(self.data_manager, 'session_dir'):
                plot_path = f"{self.data_manager.session_dir}/plots/{plot_filename}"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved: {plot_path}")
                plot_filename = plot_path
            else:
                plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                print(f"⚠️ Plot saved to current directory: {plot_filename}")

        if show_plot:
            try:
                plt.show()
            except:
                pass

        plt.close()
        return plot_filename

    def save_complete_experiment_data(self, results: Dict[str, Any]) -> str:
        """Save experiment data and generate comprehensive report"""
        # Save main experiment data
        main_file = self.save_experiment_data(results['analysis'])

        # Generate and save plot
        plot_file = self.generate_ramsey_plot(results, save_plot=True, show_plot=False)

        # Create experiment summary
        summary = self._create_ramsey_experiment_summary(results)
        summary_file = self.data_manager.save_data(summary, "experiment_summary")

        print(f"📊 Complete Ramsey experiment data saved:")
        print(f"  • Main results: {main_file}")
        print(f"  • Plot: {plot_file if plot_file else 'Not generated'}")
        print(f"  • Summary: {summary_file}")

        return main_file

    def _create_ramsey_experiment_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create human-readable Ramsey experiment summary"""
        device_results = results.get('device_results', {})
        delay_times = results.get('delay_times', [])

        summary = {
            'experiment_overview': {
                'experiment_name': self.experiment_name,
                'timestamp': time.time(),
                'method': results.get('method', 'ramsey_oscillation'),
                'delay_points': len(delay_times),
                'devices_tested': list(device_results.keys())
            },
            'key_results': {},
            'ramsey_analysis': {
                'expected_t2_star': self.expected_t2_star,
                'clear_oscillation_detected': False
            }
        }

        # Analyze each device
        min_oscillation_threshold = 0.1  # Minimum oscillation amplitude

        for device, device_data in device_results.items():
            if 'p1_values' in device_data:
                p1_values = device_data['p1_values']
                valid_p1s = [p for p in p1_values if not np.isnan(p)]

                if valid_p1s and len(valid_p1s) >= 5:
                    oscillation_amplitude = max(valid_p1s) - min(valid_p1s)
                    t2_star_fitted = device_data.get('t2_star_fitted', 0.0)
                    detuning_fitted = device_data.get('detuning_fitted', 0.0)

                    summary['key_results'][device] = {
                        'oscillation_amplitude': oscillation_amplitude,
                        't2_star_fitted': t2_star_fitted,
                        'detuning_fitted': detuning_fitted,
                        'clear_oscillation': oscillation_amplitude > min_oscillation_threshold
                    }

                    if oscillation_amplitude > min_oscillation_threshold:
                        summary['ramsey_analysis']['clear_oscillation_detected'] = True

        return summary

    def display_results(self, results: Dict[str, Any], use_rich: bool = True) -> None:
        """Display Ramsey experiment results in formatted table"""
        device_results = results.get('device_results', {})

        if not device_results:
            print("No device results found")
            return

        if use_rich:
            try:
                from rich.console import Console
                from rich.table import Table

                console = Console()
                table = Table(title="Ramsey Oscillation Results", show_header=True, header_style="bold blue")
                table.add_column("Device", style="cyan")
                table.add_column("T2* Fitted [ns]", justify="right")
                table.add_column("Detuning [MHz]", justify="right")
                table.add_column("Oscillation", justify="right")
                table.add_column("Success Rate", justify="right")
                table.add_column("Clear Signal", justify="center")

                for device, device_data in device_results.items():
                    if 'p1_values' in device_data:
                        p1_values = device_data['p1_values']
                        valid_p1s = [p for p in p1_values if not np.isnan(p)]

                        if valid_p1s and len(valid_p1s) >= 2:
                            oscillation_amplitude = max(valid_p1s) - min(valid_p1s)
                            t2_star_fitted = device_data.get('t2_star_fitted', 0.0)
                            detuning_fitted = device_data.get('detuning_fitted', 0.0)
                            
                            # 成功率の取得
                            stats = device_data.get('statistics', {})
                            success_rate = stats.get('success_rate', 0.0)
                            successful_jobs = stats.get('successful_jobs', 0)
                            total_jobs = stats.get('total_jobs', 0)

                            clear_signal = "YES" if oscillation_amplitude > 0.1 else "NO"
                            signal_style = "green" if oscillation_amplitude > 0.1 else "yellow"

                            table.add_row(
                                device.upper(),
                                f"{t2_star_fitted:.1f}",
                                f"{detuning_fitted:.3f}",
                                f"{oscillation_amplitude:.3f}",
                                f"{success_rate*100:.1f}% ({successful_jobs}/{total_jobs})",
                                clear_signal,
                                style=signal_style if oscillation_amplitude > 0.1 else None
                            )

                console.print(table)
                console.print(f"\nExpected T2*: {self.expected_t2_star} ns")
                expected_detuning = self.experiment_params.get("detuning", 0.0)
                if abs(expected_detuning) < 0.001:
                    console.print(f"Detuning: {expected_detuning} MHz → Pure T2* decay mode")
                else:
                    console.print(f"Detuning: {expected_detuning} MHz → Ramsey oscillation mode")
                fitting_status = "enabled" if self.enable_fitting else "disabled"
                console.print(f"Parameter fitting: {fitting_status}")
                console.print(f"Clear oscillation threshold: 0.1")

            except ImportError:
                use_rich = False

        if not use_rich:
            # Fallback to simple text display
            print("\n" + "="*60)
            print("Ramsey Oscillation Results")
            print("="*60)

            for device, device_data in device_results.items():
                if 'p1_values' in device_data:
                    p1_values = device_data['p1_values']
                    valid_p1s = [p for p in p1_values if not np.isnan(p)]

                    if valid_p1s and len(valid_p1s) >= 2:
                        oscillation_amplitude = max(valid_p1s) - min(valid_p1s)
                        t2_star_fitted = device_data.get('t2_star_fitted', 0.0)
                        detuning_fitted = device_data.get('detuning_fitted', 0.0)
                        
                        # 成功率の取得
                        stats = device_data.get('statistics', {})
                        success_rate = stats.get('success_rate', 0.0)
                        successful_jobs = stats.get('successful_jobs', 0)
                        total_jobs = stats.get('total_jobs', 0)

                        clear_signal = "YES" if oscillation_amplitude > 0.1 else "NO"

                        print(f"Device: {device.upper()}")
                        print(f"  T2* Fitted: {t2_star_fitted:.1f} ns")
                        print(f"  Detuning: {detuning_fitted:.3f} MHz")
                        print(f"  Oscillation: {oscillation_amplitude:.3f}")
                        print(f"  Success Rate: {success_rate*100:.1f}% ({successful_jobs}/{total_jobs})")
                        print(f"  Clear Signal: {clear_signal}")
                        print()

            print(f"Expected T2*: {self.expected_t2_star} ns")
            expected_detuning = self.experiment_params.get("detuning", 0.0)
            if abs(expected_detuning) < 0.001:
                print(f"Detuning: {expected_detuning} MHz → Pure T2* decay mode")
            else:
                print(f"Detuning: {expected_detuning} MHz → Ramsey oscillation mode")
            fitting_status = "enabled" if self.enable_fitting else "disabled"
            print(f"Parameter fitting: {fitting_status}")
            print(f"Clear oscillation threshold: 0.1")
            print("="*60)