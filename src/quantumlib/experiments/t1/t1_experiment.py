#!/usr/bin/env python3
"""
T1 Experiment Class - T1減衰実験専用クラス
BaseExperimentを継承し、T1実験に特化した実装を提供
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np

from ...circuit.t1_circuits import (
    create_multiple_t1_circuits,
    create_t1_circuit,
    create_t1_with_noise_model,
)
from ...core.base_experiment import BaseExperiment


class T1Experiment(BaseExperiment):
    """
    T1減衰実験クラス

    特化機能:
    - T1減衰回路の自動生成
    - 指数減衰フィッティング
    - 遅延時間スキャン実験
    - T1時定数推定
    """

    def __init__(self, experiment_name: str = None, **kwargs):
        super().__init__(experiment_name, **kwargs)

        # T1実験固有の設定
        self.expected_t1 = 500  # 期待されるT1時間 [ns]
        self.t1_theoretical = 500  # 理論T1値 [ns]
        self.t2_theoretical = 500  # 理論T2値 [ns]

        print(f"T1 experiment: Expected T1 ≈ {self.expected_t1} ns")

    def create_circuits(self, **kwargs) -> List[Any]:
        """
        T1実験回路作成

        Args:
            delay_points: 遅延時間点数 (default: 16)
            max_delay: 最大遅延時間 [ns] (default: 1000)
            t1: T1緩和時間 [ns] (default: 500)
            t2: T2緩和時間 [ns] (default: 500)
            delay_times: 直接指定する遅延時間リスト [ns] (optional)

        Returns:
            T1回路リスト
        """
        delay_points = kwargs.get('delay_points', 16)
        max_delay = kwargs.get('max_delay', 1000)
        t1 = kwargs.get('t1', self.t1_theoretical)
        t2 = kwargs.get('t2', self.t2_theoretical)

        # 遅延時間範囲
        if 'delay_times' in kwargs:
            delay_times = np.array(kwargs['delay_times'])
        else:
            delay_times = np.array([1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
            if delay_points != 16:
                delay_times = np.linspace(1, max_delay, delay_points)

        # メタデータ保存
        self.experiment_params = {
            'delay_times': delay_times.tolist(),
            't1_theoretical': t1,
            't2_theoretical': t2,
            'delay_points': len(delay_times),
            'max_delay': max_delay
        }

        circuits = create_multiple_t1_circuits(delay_times.tolist(), t1, t2)

        print(f"T1 circuits: T1={t1} ns, T2={t2} ns")
        print(f"Delay range: {len(delay_times)} points from {delay_times[0]:.1f} to {delay_times[-1]:.1f} ns")

        return circuits

    def analyze_results(self, results: Dict[str, List[Dict[str, Any]]], **kwargs) -> Dict[str, Any]:
        """
        T1実験結果解析

        Args:
            results: 生測定結果

        Returns:
            T1解析結果
        """
        if not results:
            return {'error': 'No results to analyze'}

        delay_times = np.array(self.experiment_params['delay_times'])
        t1_theoretical = self.experiment_params['t1_theoretical']
        t2_theoretical = self.experiment_params['t2_theoretical']

        analysis = {
            'experiment_info': {
                't1_theoretical': t1_theoretical,
                't2_theoretical': t2_theoretical,
                'delay_points': len(delay_times),
                'expected_t1': self.expected_t1
            },
            'theoretical_values': {
                'delay_times': delay_times.tolist(),
                'p1_theoretical': np.exp(-delay_times / t1_theoretical).tolist()
            },
            'device_results': {}
        }

        for device, device_results in results.items():
            if not device_results:
                continue

            device_analysis = self._analyze_device_results(device_results, delay_times)
            analysis['device_results'][device] = device_analysis

            # T1時定数推定
            t1_fitted = self._estimate_t1(device_analysis['p1_values'], delay_times)
            analysis['device_results'][device]['t1_fitted'] = t1_fitted

            print(f"{device}: Fitted T1 = {t1_fitted:.1f} ns (theoretical: {t1_theoretical} ns)")

        # デバイス間比較
        analysis['comparison'] = self._compare_devices(analysis['device_results'])

        return analysis

    def _analyze_device_results(self, device_results: List[Dict[str, Any]],
                              delay_times: np.ndarray) -> Dict[str, Any]:
        """
        単一デバイス結果解析
        """
        p1_values = []

        for i, result in enumerate(device_results):
            if result and result['success']:
                counts = result['counts']

                # P(1)確率計算
                p1 = self._calculate_p1_probability(counts)
                p1_values.append(p1)
            else:
                p1_values.append(np.nan)

        # 統計計算
        valid_p1s = np.array([p for p in p1_values if not np.isnan(p)])

        return {
            'p1_values': p1_values,
            'delay_times': delay_times.tolist(),
            'statistics': {
                'initial_p1': float(p1_values[0]) if len(p1_values) > 0 and not np.isnan(p1_values[0]) else 1.0,
                'final_p1': float(p1_values[-1]) if len(p1_values) > 0 and not np.isnan(p1_values[-1]) else 0.0,
                'success_rate': len(valid_p1s) / len(p1_values) if p1_values else 0,
                'decay_observed': float(p1_values[0] - p1_values[-1]) if len(p1_values) >= 2 and not any(np.isnan([p1_values[0], p1_values[-1]])) else 0.0
            }
        }

    def _calculate_p1_probability(self, counts: Dict[str, int]) -> float:
        """
        P(1)確率計算
        """
        total = sum(counts.values())
        if total == 0:
            return 0.0

        # |1⟩状態の確率
        if isinstance(list(counts.keys())[0], str):
            # String format
            n_1 = counts.get('1', 0)
        else:
            # Numeric format
            n_1 = counts.get(1, 0)

        p1 = n_1 / total
        return p1

    def _estimate_t1(self, p1_values: List[float], delay_times: np.ndarray) -> float:
        """
        T1時定数推定（指数減衰フィッティング）
        """
        # NaNを除去
        valid_data = [(delay, p1) for delay, p1 in zip(delay_times, p1_values)
                      if not np.isnan(p1) and p1 > 0]

        if len(valid_data) < 3:
            return 0.0

        delays = np.array([d[0] for d in valid_data])
        p1s = np.array([d[1] for d in valid_data])

        try:
            # 線形回帰によるフィッティング: ln(P1) = ln(P0) - t/T1
            log_p1s = np.log(p1s)
            
            # 線形フィッティング
            coeffs = np.polyfit(delays, log_p1s, 1)
            slope = coeffs[0]
            
            # T1 = -1/slope
            t1_fitted = -1.0 / slope if slope < 0 else float('inf')
            
            # 合理的な範囲に制限
            if t1_fitted < 0 or t1_fitted > 10000:
                t1_fitted = self.expected_t1
                
        except (ValueError, np.linalg.LinAlgError):
            # フィッティングが失敗した場合はデフォルト値
            t1_fitted = self.expected_t1

        return float(t1_fitted)

    def _compare_devices(self, device_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        デバイス間比較分析
        """
        if len(device_results) < 2:
            return {'note': 'Multiple devices required for comparison'}

        comparison = {
            'device_count': len(device_results),
            't1_comparison': {},
            'decay_comparison': {}
        }

        for device, analysis in device_results.items():
            stats = analysis['statistics']
            comparison['t1_comparison'][device] = analysis.get('t1_fitted', 0.0)
            comparison['decay_comparison'][device] = stats['decay_observed']

        return comparison

    def save_experiment_data(self, results: Dict[str, Any],
                           metadata: Dict[str, Any] = None) -> str:
        """
        T1実験データ保存
        """
        # T1実験専用の保存形式
        t1_data = {
            'experiment_type': 'T1_Decay',
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
        main_file = self.data_manager.save_data(t1_data, "t1_experiment_results")

        # 追加ファイル保存
        if 'device_results' in results:
            # デバイス別サマリー
            device_summary = {
                device: {
                    't1_fitted': analysis.get('t1_fitted', 0.0),
                    'statistics': analysis['statistics']
                }
                for device, analysis in results['device_results'].items()
            }
            self.data_manager.save_data(device_summary, "device_t1_summary")

            # P(1)データ（プロット用）
            p1_data = {
                'delay_times': self.experiment_params['delay_times'],
                'theoretical_p1': results['theoretical_values']['p1_theoretical'],
                'device_p1_values': {
                    device: analysis['p1_values']
                    for device, analysis in results['device_results'].items()
                }
            }
            self.data_manager.save_data(p1_data, "p1_values_for_plotting")

        return main_file

    def generate_t1_plot(self, results: Dict[str, Any], save_plot: bool = True,
                        show_plot: bool = False) -> Optional[str]:
        """Generate T1 experiment plot with all formatting"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available - skipping plot generation")
            return None

        delay_times = results.get('delay_times', np.linspace(1, 1000, 16))
        device_results = results.get('device_results', {})

        if not device_results:
            print("No device results for plotting")
            return None

        theoretical_p1 = results.get('theoretical_values', {}).get('p1_theoretical',
                                   np.exp(-np.array(delay_times) / self.expected_t1).tolist())

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot experimental data for each device
        colors = ['blue', 'red', 'green', 'orange', 'purple']

        for i, (device, device_data) in enumerate(device_results.items()):
            if 'p1_values' in device_data:
                p1_values = device_data['p1_values']
                t1_fitted = device_data.get('t1_fitted', 0.0)
                color = colors[i % len(colors)]
                ax.semilogx(delay_times, p1_values, 'o-', linewidth=2, markersize=6,
                          label=f'{device} (T1={t1_fitted:.1f}ns)', alpha=0.8, color=color)

        # Plot theoretical curve
        ax.semilogx(delay_times, theoretical_p1, 'k-', linewidth=3, alpha=0.7,
                   label=f'Theory: exp(-t/{self.expected_t1}ns)')

        # Formatting
        ax.set_xlabel('Delay time τ [ns] (log scale)', fontsize=14)
        ax.set_ylabel('P(1)', fontsize=14)
        ax.set_title(f'QuantumLib T1 Decay Experiment (T1 = {self.expected_t1} ns)', fontsize=16, fontweight='bold')
        ax.grid(True, which="both", ls="--", linewidth=0.5)
        ax.legend(fontsize=12)
        ax.set_ylim(0, 1.1)

        plot_filename = None
        if save_plot:
            # Save plot in experiment results directory
            plt.tight_layout()
            plot_filename = f"t1_plot_{self.experiment_name}_{int(time.time())}.png"

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
        plot_file = self.generate_t1_plot(results, save_plot=True, show_plot=False)

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
        delay_times = results.get('delay_times', [])

        summary = {
            'experiment_overview': {
                'experiment_name': self.experiment_name,
                'timestamp': time.time(),
                'method': results.get('method', 't1_decay'),
                'delay_points': len(delay_times),
                'devices_tested': list(device_results.keys())
            },
            'key_results': {},
            't1_analysis': {
                'expected_t1': self.expected_t1,
                'clear_decay_detected': False
            }
        }

        # Analyze each device
        min_decay_threshold = 0.3  # Minimum decay to consider significant

        for device, device_data in device_results.items():
            if 'p1_values' in device_data:
                p1_values = device_data['p1_values']
                valid_p1s = [p for p in p1_values if not np.isnan(p)]

                if valid_p1s and len(valid_p1s) >= 2:
                    initial_p1 = valid_p1s[0]
                    final_p1 = valid_p1s[-1]
                    decay = initial_p1 - final_p1
                    t1_fitted = device_data.get('t1_fitted', 0.0)

                    summary['key_results'][device] = {
                        'initial_p1': initial_p1,
                        'final_p1': final_p1,
                        'decay_observed': decay,
                        't1_fitted': t1_fitted,
                        'clear_decay': decay > min_decay_threshold
                    }

                    if decay > min_decay_threshold:
                        summary['t1_analysis']['clear_decay_detected'] = True

        return summary

    def display_results(self, results: Dict[str, Any], use_rich: bool = True) -> None:
        """Display T1 experiment results in formatted table"""
        device_results = results.get('device_results', {})

        if not device_results:
            print("No device results found")
            return

        if use_rich:
            try:
                from rich.console import Console
                from rich.table import Table

                console = Console()
                table = Table(title="T1 Decay Results", show_header=True, header_style="bold blue")
                table.add_column("Device", style="cyan")
                table.add_column("T1 Fitted [ns]", justify="right")
                table.add_column("Initial P(1)", justify="right")
                table.add_column("Final P(1)", justify="right")
                table.add_column("Decay", justify="right")
                table.add_column("Clear Decay", justify="center")

                method = results.get('method', 'quantumlib_t1')

                for device, device_data in device_results.items():
                    if 'p1_values' in device_data:
                        p1_values = device_data['p1_values']
                        valid_p1s = [p for p in p1_values if not np.isnan(p)]

                        if valid_p1s and len(valid_p1s) >= 2:
                            initial_p1 = valid_p1s[0]
                            final_p1 = valid_p1s[-1]
                            decay = initial_p1 - final_p1
                            t1_fitted = device_data.get('t1_fitted', 0.0)

                            clear_decay = "YES" if decay > 0.3 else "NO"
                            decay_style = "green" if decay > 0.3 else "yellow"

                            table.add_row(
                                device.upper(),
                                f"{t1_fitted:.1f}",
                                f"{initial_p1:.3f}",
                                f"{final_p1:.3f}",
                                f"{decay:.3f}",
                                clear_decay,
                                style=decay_style if decay > 0.3 else None
                            )

                console.print(table)
                console.print(f"\nExpected T1: {self.expected_t1} ns")
                console.print(f"Clear decay threshold: 0.3")

            except ImportError:
                use_rich = False

        if not use_rich:
            # Fallback to simple text display
            print("\n" + "="*60)
            print("T1 Decay Results")
            print("="*60)

            method = results.get('method', 'quantumlib_t1')

            for device, device_data in device_results.items():
                if 'p1_values' in device_data:
                    p1_values = device_data['p1_values']
                    valid_p1s = [p for p in p1_values if not np.isnan(p)]

                    if valid_p1s and len(valid_p1s) >= 2:
                        initial_p1 = valid_p1s[0]
                        final_p1 = valid_p1s[-1]
                        decay = initial_p1 - final_p1
                        t1_fitted = device_data.get('t1_fitted', 0.0)

                        clear_decay = "YES" if decay > 0.3 else "NO"

                        print(f"Device: {device.upper()}")
                        print(f"  T1 Fitted: {t1_fitted:.1f} ns")
                        print(f"  Initial P(1): {initial_p1:.3f}")
                        print(f"  Final P(1): {final_p1:.3f}")
                        print(f"  Decay: {decay:.3f}")
                        print(f"  Clear Decay: {clear_decay}")
                        print()

            print(f"Expected T1: {self.expected_t1} ns")
            print(f"Clear decay threshold: 0.3")
            print("="*60)

    def run_complete_t1_experiment(self, devices: List[str] = ['qulacs'],
                                 delay_points: int = 16, max_delay: float = 1000,
                                 t1: float = 500, t2: float = 500,
                                 shots: int = 1024, parallel_workers: int = 4,
                                 save_data: bool = True, save_plot: bool = True,
                                 show_plot: bool = False, display_results: bool = True) -> Dict[str, Any]:
        """
        Run complete T1 experiment with all post-processing
        This is the main entry point for CLI usage
        """
        print(f"🔬 Running complete T1 experiment: {self.experiment_name}")
        print(f"   Devices: {devices}")
        print(f"   Delay points: {delay_points}, Max delay: {max_delay} ns")
        print(f"   T1: {t1} ns, T2: {t2} ns")
        print(f"   Shots: {shots}, Parallel workers: {parallel_workers}")

        # Run the T1 experiment
        results = self.run_experiment(
            devices=devices,
            shots=shots,
            delay_points=delay_points,
            max_delay=max_delay,
            t1=t1,
            t2=t2
        )

        # Save data if requested
        if save_data:
            self.save_complete_experiment_data(results)
        elif save_plot:
            # Just save plot without full data
            self.generate_t1_plot(results, save_plot=True, show_plot=show_plot)

        # Display results if requested
        if display_results:
            self.display_results(results, use_rich=True)

        return results