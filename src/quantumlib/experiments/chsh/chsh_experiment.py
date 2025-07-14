#!/usr/bin/env python3
"""
CHSH Experiment Class - CHSH Bell不等式違反実験専用クラス
BaseExperimentを継承し、CHSH実験に特化した実装を提供
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np

from ...circuit.chsh_circuits import create_chsh_circuit
from ...core.base_experiment import BaseExperiment


class CHSHExperiment(BaseExperiment):
    """
    CHSH Bell不等式違反実験クラス

    特化機能:
    - CHSH回路の自動生成
    - Bell不等式値計算
    - 位相スキャン実験
    - CHSH特有のデータ保存
    """

    def __init__(self, experiment_name: str = None, **kwargs):
        # CHSH固有のpointsをkwargsから取り除く
        kwargs.pop('points', None)
        super().__init__(experiment_name, **kwargs)

        # CHSH実験固有の設定
        self.classical_bound = 2.0
        self.theoretical_max_s = 2 * np.sqrt(2)

        print(f"CHSH bounds: Classical ≤ {self.classical_bound}, "
              f"Quantum max ≈ {self.theoretical_max_s:.3f}")

    def create_circuits(self, **kwargs) -> List[Any]:
        """
        CHSH実験回路作成

        Args:
            phase_points: 位相点数 (default: 20)
            theta_a: Alice角度 (default: 0)
            theta_b: Bob角度 (default: π/4)
            phase_range: カスタム位相範囲 (optional)

        Returns:
            CHSH回路リスト
        """
        phase_points = kwargs.get('phase_points', 20)
        theta_a = kwargs.get('theta_a', 0)
        theta_b = kwargs.get('theta_b', np.pi/4)

        # カスタム位相範囲または標準範囲
        if 'phase_range' in kwargs:
            phase_range = np.array(kwargs['phase_range'])
        else:
            phase_range = np.linspace(0, 2*np.pi, phase_points)

        # メタデータ保存
        self.experiment_params = {
            'theta_a': theta_a,
            'theta_b': theta_b,
            'phase_range': phase_range.tolist(),
            'phase_points': len(phase_range)
        }

        circuits = []
        for phi in phase_range:
            circuit = create_chsh_circuit(theta_a, theta_b, phase_phi=phi)
            circuits.append(circuit)

        print(f"CHSH circuits: θ_A={theta_a:.3f}, θ_B={theta_b:.3f}")
        print(f"Phase range: {len(phase_range)} points from {phase_range[0]:.3f} to {phase_range[-1]:.3f}")

        return circuits

    def analyze_results(self, results: Dict[str, List[Dict[str, Any]]], **kwargs) -> Dict[str, Any]:
        """
        CHSH実験結果解析

        Args:
            results: 生測定結果

        Returns:
            CHSH解析結果
        """
        if not results:
            return {'error': 'No results to analyze'}

        phase_range = np.array(self.experiment_params['phase_range'])
        theta_a = self.experiment_params['theta_a']
        theta_b = self.experiment_params['theta_b']

        analysis = {
            'experiment_info': {
                'theta_a': theta_a,
                'theta_b': theta_b,
                'phase_points': len(phase_range),
                'classical_bound': self.classical_bound,
                'theoretical_max_s': self.theoretical_max_s
            },
            'theoretical_values': {
                'phase_range': phase_range.tolist(),
                'S_theoretical': (self.theoretical_max_s * np.cos(phase_range)).tolist()
            },
            'device_results': {}
        }

        for device, device_results in results.items():
            if not device_results:
                continue

            device_analysis = self._analyze_device_results(device_results, phase_range)
            analysis['device_results'][device] = device_analysis

            print(f"{device}: {device_analysis['statistics']['bell_violations']} Bell violations detected")

        # 比較分析
        analysis['comparison'] = self._compare_devices(analysis['device_results'])

        return analysis

    def _analyze_device_results(self, device_results: List[Dict[str, Any]],
                              phase_range: np.ndarray) -> Dict[str, Any]:
        """
        単一デバイス結果解析
        """
        S_values = []
        expectations = []

        for i, result in enumerate(device_results):
            if result and result['success']:
                counts = result['counts']

                # 期待値計算
                expectation = self._calculate_expectation_value(counts)
                expectations.append(expectation)

                # S値計算（簡易版）
                phi = phase_range[i] if i < len(phase_range) else 0
                S = 2 * np.sqrt(2) * expectation * np.cos(phi)
                S_values.append(S)
            else:
                expectations.append(np.nan)
                S_values.append(np.nan)

        # 統計計算
        valid_s = np.array([s for s in S_values if not np.isnan(s)])

        return {
            'S_values': S_values,
            'expectations': expectations,
            'statistics': {
                'max_S_magnitude': float(np.max(np.abs(valid_s))) if len(valid_s) > 0 else 0,
                'bell_violations': int(np.sum(np.abs(valid_s) > self.classical_bound)),
                'success_rate': len(valid_s) / len(S_values) if S_values else 0,
                'mean_expectation': float(np.nanmean(expectations))
            }
        }

    def _calculate_expectation_value(self, counts: Dict[str, int]) -> float:
        """
        CHSH期待値計算
        """
        total = sum(counts.values())
        if total == 0:
            return 0.0

        # CHSH期待値: E = (N_00 + N_11 - N_01 - N_10) / N_total
        n_00 = counts.get('00', 0)
        n_11 = counts.get('11', 0)
        n_01 = counts.get('01', 0)
        n_10 = counts.get('10', 0)

        expectation = (n_00 + n_11 - n_01 - n_10) / total
        return expectation

    def _compare_devices(self, device_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        デバイス間比較分析
        """
        if len(device_results) < 2:
            return {'note': 'Multiple devices required for comparison'}

        comparison = {
            'device_count': len(device_results),
            'bell_violation_comparison': {},
            'max_S_comparison': {}
        }

        for device, analysis in device_results.items():
            stats = analysis['statistics']
            comparison['bell_violation_comparison'][device] = stats['bell_violations']
            comparison['max_S_comparison'][device] = stats['max_S_magnitude']

        return comparison

    def save_experiment_data(self, results: Dict[str, Any],
                           metadata: Dict[str, Any] = None) -> str:
        """
        CHSH実験データ保存
        """
        # CHSH実験専用の保存形式
        chsh_data = {
            'experiment_type': 'CHSH_Bell_Inequality',
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
        main_file = self.data_manager.save_data(chsh_data, "chsh_experiment_results")

        # 追加ファイル保存
        if 'device_results' in results:
            # デバイス別サマリー
            device_summary = {
                device: analysis['statistics']
                for device, analysis in results['device_results'].items()
            }
            self.data_manager.save_data(device_summary, "device_performance_summary")

            # S値のみ（プロット用）
            s_values_data = {
                'phase_range': self.experiment_params['phase_range'],
                'theoretical_S': results['theoretical_values']['S_theoretical'],
                'device_S_values': {
                    device: analysis['S_values']
                    for device, analysis in results['device_results'].items()
                }
            }
            self.data_manager.save_data(s_values_data, "s_values_for_plotting")

        return main_file

    # CHSH実験専用の便利メソッド
    def run_phase_scan(self, devices: List[str] = ['qulacs'],
                      phase_points: int = 20,
                      theta_a: float = 0,
                      theta_b: float = np.pi/4,
                      shots: int = 1024,
                      **kwargs) -> Dict[str, Any]:
        """
        位相スキャンCHSH実験実行
        """
        return self.run_experiment(
            devices=devices,
            shots=shots,
            phase_points=phase_points,
            theta_a=theta_a,
            theta_b=theta_b,
            **kwargs
        )

    def run_4_measurement_chsh(self, devices: List[str] = ['qulacs'],
                              phase_points: int = 20,
                              shots: int = 1024,
                              parallel_workers: int = 4,
                              **kwargs) -> Dict[str, Any]:
        """
        4測定CHSH方式の実行（量子ベル不等式標準方式）
        ⟨A₀B₀⟩, ⟨A₀B₁⟩, ⟨A₁B₀⟩, ⟨A₁B₁⟩ を個別測定してS = E₁ + E₂ + E₃ - E₄計算
        """
        # 標準CHSH測定角度
        angles = {
            'theta_a0': 0,           # Alice 測定角度1
            'theta_a1': np.pi/2,     # Alice 測定角度2
            'theta_b0': np.pi/4,     # Bob 測定角度1
            'theta_b1': -np.pi/4     # Bob 測定角度2
        }

        phase_range = np.linspace(0, 2*np.pi, phase_points)

        # 4測定組み合わせ
        measurements = [
            (angles['theta_a0'], angles['theta_b0']),  # ⟨A₀B₀⟩
            (angles['theta_a0'], angles['theta_b1']),  # ⟨A₀B₁⟩
            (angles['theta_a1'], angles['theta_b0']),  # ⟨A₁B₀⟩
            (angles['theta_a1'], angles['theta_b1'])   # ⟨A₁B₁⟩
        ]

        # バッチ回路作成：全位相×全測定の組み合わせを一括作成
        all_circuits = []
        circuit_metadata = []

        for i, phase_phi in enumerate(phase_range):
            for j, (theta_a, theta_b) in enumerate(measurements):
                circuit = create_chsh_circuit(theta_a, theta_b, phase_phi)
                all_circuits.append(circuit)
                circuit_metadata.append({
                    'phase_index': i,
                    'measurement_index': j,
                    'phase_phi': phase_phi,
                    'theta_a': theta_a,
                    'theta_b': theta_b
                })

        print(f"Creating batch circuits: {phase_points} phases × 4 measurements = {len(all_circuits)} circuits")

        # バッチ回路の並列投入と収集
        job_data = self._submit_circuits_parallel_with_order_preservation(
            all_circuits, devices, shots, parallel_workers
        )

        raw_results = self._collect_results_parallel_with_order_preservation(
            job_data, parallel_workers
        )

        # 結果を構造化してCHSH解析
        processed_results = self._process_4_measurement_results(
            raw_results, circuit_metadata, phase_range, measurements, devices
        )

        # 標準解析結果作成
        analysis = self._create_4_measurement_analysis(phase_range, processed_results, angles)

        # experiment_paramsをセット（データ保存用）
        self.experiment_params = {
            'theta_a': 0,
            'theta_b': np.pi/4,
            'phase_range': phase_range.tolist(),
            'phase_points': len(phase_range)
        }

        return {
            'phase_range': phase_range,
            'device_results': processed_results,
            'analysis': analysis,
            'method': '4_measurement_chsh_quantumlib'
        }

    def run_angle_comparison(self, devices: List[str] = ['qulacs'],
                           angle_pairs: List[tuple] = None,
                           shots: int = 1024,
                           **kwargs) -> Dict[str, Any]:
        """
        角度比較CHSH実験実行
        """
        if angle_pairs is None:
            angle_pairs = [(0, np.pi/4), (np.pi/4, 0), (np.pi/8, np.pi/8)]

        all_results = []

        for i, (theta_a, theta_b) in enumerate(angle_pairs):
            print(f"\n🔬 Angle pair {i+1}/{len(angle_pairs)}: θ_A={theta_a:.3f}, θ_B={theta_b:.3f}")

            # 単一位相点での実験
            result = self.run_experiment(
                devices=devices,
                shots=shots,
                phase_range=[0],  # φ=0のみ
                theta_a=theta_a,
                theta_b=theta_b,
                **kwargs
            )

            result['angle_pair'] = (theta_a, theta_b)
            all_results.append(result)

        # 統合結果
        return {
            'experiment_type': 'CHSH_Angle_Comparison',
            'angle_pairs': angle_pairs,
            'individual_results': all_results,
            'comparison_summary': self._summarize_angle_comparison(all_results)
        }

    def _summarize_angle_comparison(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        角度比較結果のサマリー
        """
        summary = {
            'angle_performance': {},
            'best_angle_pair': None,
            'max_bell_violation': 0
        }

        for result in results:
            angle_pair = result['angle_pair']
            angle_key = f"θA={angle_pair[0]:.3f}_θB={angle_pair[1]:.3f}"

            if 'device_results' in result['analyzed_results']:
                for device, analysis in result['analyzed_results']['device_results'].items():
                    bell_violations = analysis['statistics']['bell_violations']
                    max_s = analysis['statistics']['max_S_magnitude']

                    if angle_key not in summary['angle_performance']:
                        summary['angle_performance'][angle_key] = {}

                    summary['angle_performance'][angle_key][device] = {
                        'bell_violations': bell_violations,
                        'max_S_magnitude': max_s
                    }

                    if max_s > summary['max_bell_violation']:
                        summary['max_bell_violation'] = max_s
                        summary['best_angle_pair'] = angle_pair

        return summary

    def _submit_circuits_parallel_with_order_preservation(self, circuits: List[Any],
                                                        devices: List[str], shots: int,
                                                        parallel_workers: int) -> Dict[str, List[Dict]]:
        """
        回路順序を保持する並列ジョブ投入
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        print(f"Enhanced parallel submission: {parallel_workers} workers")

        if not self.oqtopus_available:
            return self._submit_circuits_locally_parallel(circuits, devices, shots, parallel_workers)

        # 結果を順序付きで管理
        all_job_data = {device: [None] * len(circuits) for device in devices}

        # 回路×デバイスの組み合わせ作成
        circuit_device_pairs = []
        for circuit_idx, circuit in enumerate(circuits):
            for device in devices:
                circuit_device_pairs.append((circuit_idx, circuit, device))

        def submit_single_circuit(args):
            circuit_idx, circuit, device = args
            try:
                job_id = self.submit_circuit_to_oqtopus(circuit, shots, device)
                if job_id:
                    return device, job_id, circuit_idx, True
                else:
                    return device, None, circuit_idx, False
            except Exception as e:
                print(f"Circuit {circuit_idx} → {device}: {e}")
                return device, None, circuit_idx, False

        # 並列投入実行
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = [executor.submit(submit_single_circuit, args) for args in circuit_device_pairs]

            for future in as_completed(futures):
                device, job_id, circuit_idx, success = future.result()
                if success and job_id:
                    all_job_data[device][circuit_idx] = {
                        'job_id': job_id,
                        'circuit_index': circuit_idx,
                        'submitted': True
                    }
                    print(f"Circuit {circuit_idx+1} → {device}: {job_id[:8]}...")
                else:
                    all_job_data[device][circuit_idx] = {
                        'job_id': None,
                        'circuit_index': circuit_idx,
                        'submitted': False
                    }

        for device in devices:
            successful_jobs = sum(1 for job_data in all_job_data[device] if job_data and job_data['submitted'])
            print(f"✅ {device}: {successful_jobs} jobs submitted (order preserved)")

        return all_job_data

    def _submit_circuits_locally_parallel(self, circuits: List[Any], devices: List[str],
                                         shots: int, parallel_workers: int) -> Dict[str, List[Dict]]:
        """Submit circuits to local simulator with parallel execution"""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        print(f"Local parallel execution: {parallel_workers} workers")

        all_job_data = {device: [None] * len(circuits) for device in devices}

        circuit_device_pairs = []
        for circuit_idx, circuit in enumerate(circuits):
            for device in devices:
                circuit_device_pairs.append((circuit_idx, circuit, device))

        def run_single_circuit_locally(args):
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
                print(f"Local circuit {circuit_idx} → {device}: {e}")
                return device, None, circuit_idx, False

        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = [executor.submit(run_single_circuit_locally, args) for args in circuit_device_pairs]

            for future in as_completed(futures):
                device, job_id, circuit_idx, success = future.result()
                if success and job_id:
                    all_job_data[device][circuit_idx] = {
                        'job_id': job_id,
                        'circuit_index': circuit_idx,
                        'submitted': True
                    }
                else:
                    all_job_data[device][circuit_idx] = {
                        'job_id': None,
                        'circuit_index': circuit_idx,
                        'submitted': False
                    }

        for device in devices:
            successful = sum(1 for job in all_job_data[device] if job and job['submitted'])
            print(f"✅ {device}: {successful} circuits completed locally (order preserved)")

        return all_job_data

    def _collect_results_parallel_with_order_preservation(self, job_data: Dict[str, List[Dict]],
                                                        parallel_workers: int) -> Dict[str, List[Dict]]:
        """Collect results while preserving order"""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Handle local results
        if hasattr(self, '_local_results'):
            print("Using cached local simulation results...")
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
                print(f"✅ {device}: {successful} local results collected")
            return all_results

        if not self.oqtopus_available:
            print("OQTOPUS not available")
            return {device: [None] * len(device_job_data) for device, device_job_data in job_data.items()}

        all_results = {device: [None] * len(device_job_data) for device, device_job_data in job_data.items()}

        job_collection_tasks = []
        for device, device_job_data in job_data.items():
            for circuit_idx, job_info in enumerate(device_job_data):
                if job_info and job_info['submitted'] and job_info['job_id']:
                    job_collection_tasks.append((job_info['job_id'], device, circuit_idx))

        def collect_single_result(args):
            job_id, device, circuit_idx = args
            try:
                result = self.get_oqtopus_result(job_id, timeout_minutes=5)
                return device, result, job_id, circuit_idx, True if result else False
            except Exception as e:
                print(f"Collection error {job_id[:8]}: {e}")
                return device, None, job_id, circuit_idx, False

        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = [executor.submit(collect_single_result, args) for args in job_collection_tasks]

            for future in as_completed(futures):
                device, result, job_id, circuit_idx, success = future.result()
                if success and result:
                    all_results[device][circuit_idx] = result
                    print(f"✅ {device}[{circuit_idx}]: {job_id[:8]}... collected")

        for device in job_data.keys():
            successful = sum(1 for r in all_results[device] if r is not None)
            total = len(job_data[device])
            print(f"✅ {device}: {successful}/{total} results collected")

        return all_results

    def _process_4_measurement_results(self, raw_results: Dict[str, List[Dict]],
                                     circuit_metadata: List[Dict], phase_range,
                                     measurements: List[tuple], devices: List[str]) -> Dict[str, Dict]:
        """Process 4-measurement CHSH results into structured format"""
        all_results = {}

        for device in devices:
            print(f"\nProcessing results for {device}...")

            device_results = raw_results.get(device, [])
            phase_points = len(phase_range)

            device_s_values = []
            device_expectations = []

            for phase_idx in range(phase_points):
                phase_phi = phase_range[phase_idx]
                phase_expectations = []

                for meas_idx in range(4):
                    circuit_idx = phase_idx * 4 + meas_idx

                    if circuit_idx < len(device_results) and device_results[circuit_idx] is not None:
                        result = device_results[circuit_idx]
                        counts = result['counts']
                        expectation = self._calculate_expectation_value_oqtopus_compatible(counts)
                        phase_expectations.append(expectation)

                        if phase_idx == 0 and meas_idx < 2:
                            print(f"Debug - Phase {phase_idx}, Meas {meas_idx}: counts={counts}, exp={expectation:.3f}")
                    else:
                        phase_expectations.append(0.0)

                # CHSH S value calculation: S = E1 + E2 + E3 - E4
                if len(phase_expectations) == 4:
                    E1, E2, E3, E4 = phase_expectations
                    S = E1 + E2 + E3 - E4
                else:
                    S = 0.0

                device_s_values.append(S)
                device_expectations.append(phase_expectations)

            all_results[device] = {
                'S_values': device_s_values,
                'expectations': device_expectations,
                'measurement_angles': {
                    'theta_a0': 0, 'theta_a1': np.pi/2,
                    'theta_b0': np.pi/4, 'theta_b1': -np.pi/4
                }
            }

            # Statistics
            S_array = np.array(device_s_values)
            max_S = np.max(np.abs(S_array))
            violations = int(np.sum(np.abs(S_array) > 2.0))
            print(f"{device}: Max |S| = {max_S:.3f}, Bell violations: {violations}/{phase_points}")

        return all_results

    def _calculate_expectation_value_oqtopus_compatible(self, counts: dict) -> float:
        """Calculate CHSH expectation value compatible with OQTOPUS format"""
        total = sum(counts.values())
        if total == 0:
            return 0.0

        # String format (standard qiskit)
        if all(isinstance(k, str) for k in counts.keys()):
            n_00 = counts.get('00', 0)
            n_11 = counts.get('11', 0)
            n_01 = counts.get('01', 0)
            n_10 = counts.get('10', 0)

        # Numeric format (OQTOPUS)
        elif all(isinstance(k, int) for k in counts.keys()):
            n_00 = counts.get(0, 0)  # '00' = 0
            n_11 = counts.get(3, 0)  # '11' = 3
            n_01 = counts.get(1, 0)  # '01' = 1
            n_10 = counts.get(2, 0)  # '10' = 2

        else:
            # Mixed format - convert to string format
            converted_counts = {}
            for k, v in counts.items():
                if isinstance(k, int):
                    bit_str = f"{k:02b}"
                    converted_counts[bit_str] = v
                else:
                    converted_counts[k] = v

            n_00 = converted_counts.get('00', 0)
            n_11 = converted_counts.get('11', 0)
            n_01 = converted_counts.get('01', 0)
            n_10 = converted_counts.get('10', 0)

        expectation = (n_00 + n_11 - n_01 - n_10) / total
        return expectation

    def _create_4_measurement_analysis(self, phase_range, all_results: Dict[str, Dict],
                                     angles: Dict[str, float]) -> Dict[str, Any]:
        """Create comprehensive 4-measurement CHSH analysis"""
        analysis = {
            'experiment_info': {
                'theta_a0': angles['theta_a0'], 'theta_a1': angles['theta_a1'],
                'theta_b0': angles['theta_b0'], 'theta_b1': angles['theta_b1'],
                'phase_points': len(phase_range),
                'classical_bound': 2.0,
                'theoretical_max_s': 2 * np.sqrt(2)
            },
            'theoretical_values': {
                'phase_range': phase_range.tolist(),
                'S_theoretical': (2 * np.sqrt(2) * np.cos(phase_range)).tolist()
            },
            'device_results': {}
        }

        for device, device_data in all_results.items():
            S_values = device_data['S_values']
            S_array = np.array(S_values)
            bell_violations = int(np.sum(np.abs(S_array) > 2.0))
            max_S = float(np.max(np.abs(S_array)))

            analysis['device_results'][device] = {
                'S_values': S_values,
                'expectations': device_data['expectations'],
                'statistics': {
                    'max_S_magnitude': max_S,
                    'bell_violations': bell_violations,
                    'success_rate': 1.0,
                    'mean_S_magnitude': float(np.mean(np.abs(S_array)))
                }
            }

        return analysis

    def generate_chsh_plot(self, results: Dict[str, Any], save_plot: bool = True,
                          show_plot: bool = False) -> Optional[str]:
        """Generate CHSH experiment plot with all formatting"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available - skipping plot generation")
            return None

        phase_range = results.get('phase_range', np.linspace(0, 2*np.pi, 20))
        device_results = results.get('device_results', {})

        if not device_results:
            print("No device results for plotting")
            return None

        theoretical_S = 2 * np.sqrt(2) * np.cos(phase_range)

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot experimental data for each device
        colors = ['blue', 'red', 'green', 'orange', 'purple']

        for i, (device, device_data) in enumerate(device_results.items()):
            if 'S_values' in device_data:
                S_values = device_data['S_values']
                color = colors[i % len(colors)]
                ax.plot(phase_range, S_values, 'o-', linewidth=2, markersize=6,
                        label=f'{device} (quantumlib)', alpha=0.8, color=color)

        # Plot theoretical curve
        ax.plot(phase_range, theoretical_S, 'k-', linewidth=3, alpha=0.7,
                label='Theory: 2√2 cos(φ)')

        # Bell inequality bounds
        ax.axhline(y=2.0, color='red', linestyle='--', alpha=0.7, linewidth=2,
                   label='Classical bounds (±2)')
        ax.axhline(y=-2.0, color='red', linestyle='--', alpha=0.7, linewidth=2)

        # Tsirelson bounds
        ax.axhline(y=2*np.sqrt(2), color='green', linestyle=':', alpha=0.7, linewidth=2,
                   label='Tsirelson bounds (±2√2)')
        ax.axhline(y=-2*np.sqrt(2), color='green', linestyle=':', alpha=0.7, linewidth=2)

        # Formatting
        ax.set_xlabel('Phase φ [rad]', fontsize=14)
        ax.set_ylabel('CHSH Parameter S', fontsize=14)
        ax.set_title('QuantumLib CHSH: 4-Measurement Bell Inequality Test',
                     fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)

        # X-axis labels in π units
        ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])

        plot_filename = None
        if save_plot:
            # Save plot in experiment results directory
            plt.tight_layout()
            plot_filename = f"chsh_plot_{self.experiment_name}_{int(time.time())}.png"

            # Always save to experiment results directory
            if hasattr(self, 'data_manager') and hasattr(self.data_manager, 'session_dir'):
                plot_path = f"{self.data_manager.session_dir}/plots/{plot_filename}"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved: {plot_path}")
                plot_filename = plot_path  # Return full path
            else:
                # Fallback: save in current directory but warn
                plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                print(f"⚠️ Plot saved to current directory: {plot_filename}")
                print("   (data_manager not available)")

        # Try to display plot
        if show_plot:
            try:
                plt.show()
            except:
                pass

        plt.close()
        return plot_filename

    def save_complete_experiment_data(self, results: Dict[str, Any]) -> str:
        """Save experiment data and generate comprehensive report"""
        # Save main experiment data using existing system
        main_file = self.save_experiment_data(results['analysis'])

        # Generate and save plot
        plot_file = self.generate_chsh_plot(results, save_plot=True, show_plot=False)

        # Create experiment summary
        summary = self._create_experiment_summary(results)
        summary_file = self.data_manager.save_data(summary, "experiment_summary")

        print(f"📊 Complete experiment data saved:")
        print(f"  • Main results: {main_file}")
        print(f"  • Plot: {plot_file if plot_file else 'Not generated'}")
        print(f"  • Summary: {summary_file}")

        return main_file

    def _create_experiment_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create human-readable experiment summary"""
        device_results = results.get('device_results', {})
        phase_range = results.get('phase_range', [])

        summary = {
            'experiment_overview': {
                'experiment_name': self.experiment_name,
                'timestamp': time.time(),
                'method': results.get('method', '4_measurement_chsh'),
                'phase_points': len(phase_range),
                'devices_tested': list(device_results.keys())
            },
            'key_results': {},
            'bell_inequality_analysis': {
                'classical_bound': 2.0,
                'quantum_bound': 2 * np.sqrt(2),
                'violations_detected': False
            }
        }

        # Analyze each device
        total_violations = 0
        max_s_overall = 0

        for device, device_data in device_results.items():
            if 'S_values' in device_data:
                S_values = device_data['S_values']
                S_array = np.array(S_values)
                max_S = float(np.max(np.abs(S_array)))
                violations = int(np.sum(np.abs(S_array) > 2.0))

                summary['key_results'][device] = {
                    'max_S_magnitude': max_S,
                    'bell_violations_count': violations,
                    'bell_violations_percentage': violations / len(S_values) * 100,
                    'quantum_advantage': max_S > 2.0
                }

                total_violations += violations
                max_s_overall = max(max_s_overall, max_S)

        summary['bell_inequality_analysis']['violations_detected'] = total_violations > 0
        summary['bell_inequality_analysis']['max_s_magnitude'] = max_s_overall
        summary['bell_inequality_analysis']['total_violations'] = total_violations

        return summary

    def display_results(self, results: Dict[str, Any], use_rich: bool = True) -> None:
        """Display CHSH experiment results in formatted table"""
        device_results = results.get('device_results', {})

        if not device_results:
            print("No device results found")
            return

        if use_rich:
            try:
                from rich.console import Console
                from rich.table import Table

                console = Console()
                table = Table(title="CHSH Verification Results", show_header=True, header_style="bold blue")
                table.add_column("Device", style="cyan")
                table.add_column("Max |S|", justify="right")
                table.add_column("Bell Violations", justify="right")
                table.add_column("Method", justify="center")
                table.add_column("Quantum Advantage", justify="center")

                method = results.get('method', 'quantumlib_chsh')

                for device, device_data in device_results.items():
                    if 'S_values' in device_data:
                        S_values = device_data['S_values']
                        S_array = np.array(S_values)
                        max_S = float(np.max(np.abs(S_array)))
                        violations = int(np.sum(np.abs(S_array) > 2.0))
                        total_points = len(S_values)

                        advantage = "YES" if max_S > 2.0 else "NO"
                        advantage_style = "green" if max_S > 2.0 else "red"

                        table.add_row(
                            device.upper(),
                            f"{max_S:.3f}",
                            f"{violations}/{total_points}",
                            method,
                            advantage,
                            style=advantage_style if max_S > 2.0 else None
                        )

                console.print(table)
                console.print(f"\nTheoretical maximum: {2 * np.sqrt(2):.3f}")
                console.print(f"Classical bound: ±2.0")

            except ImportError:
                use_rich = False

        if not use_rich:
            # Fallback to simple text display
            print("\n" + "="*60)
            print("CHSH Verification Results")
            print("="*60)

            method = results.get('method', 'quantumlib_chsh')

            for device, device_data in device_results.items():
                if 'S_values' in device_data:
                    S_values = device_data['S_values']
                    S_array = np.array(S_values)
                    max_S = float(np.max(np.abs(S_array)))
                    violations = int(np.sum(np.abs(S_array) > 2.0))
                    total_points = len(S_values)

                    advantage = "YES" if max_S > 2.0 else "NO"

                    print(f"Device: {device.upper()}")
                    print(f"  Max |S|: {max_S:.3f}")
                    print(f"  Bell Violations: {violations}/{total_points}")
                    print(f"  Method: {method}")
                    print(f"  Quantum Advantage: {advantage}")
                    print()

            print(f"Theoretical maximum: {2 * np.sqrt(2):.3f}")
            print(f"Classical bound: ±2.0")
            print("="*60)

    def run_complete_chsh_experiment(self, devices: List[str] = ['qulacs'],
                                   phase_points: int = 20, shots: int = 1024,
                                   parallel_workers: int = 4,
                                   save_data: bool = True, save_plot: bool = True,
                                   show_plot: bool = False, display_results: bool = True) -> Dict[str, Any]:
        """
        Run complete CHSH experiment with all post-processing
        This is the main entry point for CLI usage
        """
        print(f"🔬 Running complete CHSH experiment: {self.experiment_name}")
        print(f"   Devices: {devices}")
        print(f"   Phase points: {phase_points}, Shots: {shots}")
        print(f"   Parallel workers: {parallel_workers}")

        # Run the 4-measurement CHSH experiment
        results = self.run_4_measurement_chsh(
            devices=devices,
            phase_points=phase_points,
            shots=shots,
            parallel_workers=parallel_workers
        )

        # Save data if requested
        if save_data:
            self.save_complete_experiment_data(results)
        elif save_plot:
            # Just save plot without full data
            self.generate_chsh_plot(results, save_plot=True, show_plot=show_plot)

        # Display results if requested
        if display_results:
            self.display_results(results, use_rich=True)

        return results
