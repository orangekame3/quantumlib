#!/usr/bin/env python3
"""
Rabi Experiment Class - Rabi振動実験専用クラス
BaseExperimentを継承し、Rabi実験に特化した実装を提供
"""

import time
from typing import Any, Dict, List, Optional

import numpy as np

from ...circuit.rabi_circuits import (
    create_rabi_circuit,
    create_ramsey_circuit,
    create_t1_circuit,
)
from ...core.base_experiment import BaseExperiment


class RabiExperiment(BaseExperiment):
    """
    Rabi振動実験クラス

    特化機能:
    - Rabi振動回路の自動生成
    - 励起確率計算
    - ドライブ振幅スキャン実験
    - Rabi周波数フィッティング
    """

    def __init__(self, experiment_name: str = None, **kwargs):
        super().__init__(experiment_name, **kwargs)

        # Rabi実験固有の設定
        self.expected_pi_pulse = np.pi  # 期待されるπパルス角度

        print(f"Rabi experiment: Expected π pulse ≈ {self.expected_pi_pulse:.3f} rad")

    def create_circuits(self, **kwargs) -> List[Any]:
        """
        Rabi実験回路作成

        Args:
            amplitude_points: 振幅点数 (default: 20)
            max_amplitude: 最大振幅 (default: 2π)
            drive_time: ドライブ時間 (default: 1.0)
            drive_frequency: ドライブ周波数 (default: 0.0)

        Returns:
            Rabi回路リスト
        """
        amplitude_points = kwargs.get('amplitude_points', 20)
        max_amplitude = kwargs.get('max_amplitude', 2*np.pi)
        drive_time = kwargs.get('drive_time', 1.0)
        drive_frequency = kwargs.get('drive_frequency', 0.0)

        # 振幅範囲
        if 'amplitude_range' in kwargs:
            amplitude_range = np.array(kwargs['amplitude_range'])
        else:
            amplitude_range = np.linspace(0, max_amplitude, amplitude_points)

        # メタデータ保存
        self.experiment_params = {
            'amplitude_range': amplitude_range.tolist(),
            'drive_time': drive_time,
            'drive_frequency': drive_frequency,
            'amplitude_points': len(amplitude_range)
        }

        circuits = []
        for amplitude in amplitude_range:
            circuit = create_rabi_circuit(amplitude, drive_time, drive_frequency)
            circuits.append(circuit)

        print(f"Rabi circuits: drive_time={drive_time:.3f}, frequency={drive_frequency:.3f}")
        print(f"Amplitude range: {len(amplitude_range)} points from {amplitude_range[0]:.3f} to {amplitude_range[-1]:.3f}")

        return circuits

    def analyze_results(self, results: Dict[str, List[Dict[str, Any]]], **kwargs) -> Dict[str, Any]:
        """
        Rabi実験結果解析

        Args:
            results: 生測定結果

        Returns:
            Rabi解析結果
        """
        if not results:
            return {'error': 'No results to analyze'}

        amplitude_range = np.array(self.experiment_params['amplitude_range'])
        drive_time = self.experiment_params['drive_time']
        drive_frequency = self.experiment_params['drive_frequency']

        analysis = {
            'experiment_info': {
                'drive_time': drive_time,
                'drive_frequency': drive_frequency,
                'amplitude_points': len(amplitude_range),
                'expected_pi_pulse': self.expected_pi_pulse
            },
            'theoretical_values': {
                'amplitude_range': amplitude_range.tolist(),
                'excitation_theoretical': (np.sin(amplitude_range * drive_time / 2) ** 2).tolist()
            },
            'device_results': {}
        }

        for device, device_results in results.items():
            if not device_results:
                continue

            device_analysis = self._analyze_device_results(device_results, amplitude_range)
            analysis['device_results'][device] = device_analysis

            # Rabi周波数推定
            rabi_freq = self._estimate_rabi_frequency(device_analysis['excitation_probabilities'], amplitude_range)
            analysis['device_results'][device]['rabi_frequency'] = rabi_freq

            print(f"{device}: Estimated Rabi frequency = {rabi_freq:.3f} rad/amplitude")

        # デバイス間比較
        analysis['comparison'] = self._compare_devices(analysis['device_results'])

        return analysis

    def _analyze_device_results(self, device_results: List[Dict[str, Any]],
                              amplitude_range: np.ndarray) -> Dict[str, Any]:
        """
        単一デバイス結果解析
        """
        excitation_probs = []

        for i, result in enumerate(device_results):
            if result and result['success']:
                counts = result['counts']

                # 励起確率計算
                excitation_prob = self._calculate_excitation_probability(counts)
                excitation_probs.append(excitation_prob)
            else:
                excitation_probs.append(np.nan)

        # 統計計算
        valid_probs = np.array([p for p in excitation_probs if not np.isnan(p)])

        return {
            'excitation_probabilities': excitation_probs,
            'amplitude_range': amplitude_range.tolist(),
            'statistics': {
                'max_excitation': float(np.max(valid_probs)) if len(valid_probs) > 0 else 0,
                'min_excitation': float(np.min(valid_probs)) if len(valid_probs) > 0 else 0,
                'success_rate': len(valid_probs) / len(excitation_probs) if excitation_probs else 0,
                'mean_excitation': float(np.nanmean(excitation_probs))
            }
        }

    def _calculate_excitation_probability(self, counts: Dict[str, int]) -> float:
        """
        励起確率計算
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

        excitation_prob = n_1 / total
        return excitation_prob

    def _estimate_rabi_frequency(self, excitation_probs: List[float],
                               amplitude_range: np.ndarray) -> float:
        """
        Rabi周波数推定（簡単なフィッティング）
        """
        # NaNを除去
        valid_data = [(amp, prob) for amp, prob in zip(amplitude_range, excitation_probs)
                      if not np.isnan(prob)]

        if len(valid_data) < 3:
            return 0.0

        amplitudes = np.array([d[0] for d in valid_data])
        probs = np.array([d[1] for d in valid_data])

        # 最初の最大値を見つける（π/2パルスに対応）
        max_idx = np.argmax(probs)
        if max_idx > 0:
            pi_half_amplitude = amplitudes[max_idx]
            # Rabi周波数 = π/(2 * amplitude_for_pi_half_pulse)
            rabi_freq = np.pi / (2 * pi_half_amplitude)
        else:
            # フォールバック：理論値
            rabi_freq = 1.0

        return rabi_freq

    def _compare_devices(self, device_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        デバイス間比較分析
        """
        if len(device_results) < 2:
            return {'note': 'Multiple devices required for comparison'}

        comparison = {
            'device_count': len(device_results),
            'rabi_frequency_comparison': {},
            'max_excitation_comparison': {}
        }

        for device, analysis in device_results.items():
            stats = analysis['statistics']
            comparison['rabi_frequency_comparison'][device] = analysis.get('rabi_frequency', 0.0)
            comparison['max_excitation_comparison'][device] = stats['max_excitation']

        return comparison

    def save_experiment_data(self, results: Dict[str, Any],
                           metadata: Dict[str, Any] = None) -> str:
        """
        Rabi実験データ保存
        """
        # Rabi実験専用の保存形式
        rabi_data = {
            'experiment_type': 'Rabi_Oscillation',
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
        main_file = self.data_manager.save_data(rabi_data, "rabi_experiment_results")

        # 追加ファイル保存
        if 'device_results' in results:
            # デバイス別サマリー
            device_summary = {
                device: analysis['statistics']
                for device, analysis in results['device_results'].items()
            }
            self.data_manager.save_data(device_summary, "device_performance_summary")

            # 励起確率データ（プロット用）
            excitation_data = {
                'amplitude_range': self.experiment_params['amplitude_range'],
                'theoretical_excitation': results['theoretical_values']['excitation_theoretical'],
                'device_excitation_probs': {
                    device: analysis['excitation_probabilities']
                    for device, analysis in results['device_results'].items()
                }
            }
            self.data_manager.save_data(excitation_data, "excitation_probabilities_for_plotting")

        return main_file

    def generate_rabi_plot(self, results: Dict[str, Any], save_plot: bool = True,
                          show_plot: bool = False) -> Optional[str]:
        """Generate Rabi experiment plot with all formatting"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available - skipping plot generation")
            return None

        amplitude_range = results.get('amplitude_range', np.linspace(0, 2*np.pi, 20))
        device_results = results.get('device_results', {})

        if not device_results:
            print("No device results for plotting")
            return None

        theoretical_excitation = results.get('theoretical_values', {}).get('excitation_theoretical',
                                           (np.sin(amplitude_range / 2) ** 2).tolist())

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot experimental data for each device
        colors = ['blue', 'red', 'green', 'orange', 'purple']

        for i, (device, device_data) in enumerate(device_results.items()):
            if 'excitation_probabilities' in device_data:
                excitation_probs = device_data['excitation_probabilities']
                color = colors[i % len(colors)]
                ax.plot(amplitude_range, excitation_probs, 'o-', linewidth=2, markersize=6,
                        label=f'{device} (quantumlib)', alpha=0.8, color=color)

        # Plot theoretical curve
        ax.plot(amplitude_range, theoretical_excitation, 'k-', linewidth=3, alpha=0.7,
                label='Theory: sin²(Ωt/2)')

        # Formatting
        ax.set_xlabel('Drive Amplitude [rad]', fontsize=14)
        ax.set_ylabel('Excitation Probability', fontsize=14)
        ax.set_title('QuantumLib Rabi Oscillation Experiment', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        ax.set_ylim(0, 1.1)

        # X-axis labels in π units
        max_amp = max(amplitude_range)
        if max_amp >= 2*np.pi:
            ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
            ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])

        plot_filename = None
        if save_plot:
            # Save plot in experiment results directory
            plt.tight_layout()
            plot_filename = f"rabi_plot_{self.experiment_name}_{int(time.time())}.png"

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
        plot_file = self.generate_rabi_plot(results, save_plot=True, show_plot=False)

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
        amplitude_range = results.get('amplitude_range', [])

        summary = {
            'experiment_overview': {
                'experiment_name': self.experiment_name,
                'timestamp': time.time(),
                'method': results.get('method', 'rabi_oscillation'),
                'amplitude_points': len(amplitude_range),
                'devices_tested': list(device_results.keys())
            },
            'key_results': {},
            'rabi_analysis': {
                'expected_pi_pulse': self.expected_pi_pulse,
                'oscillations_detected': False
            }
        }

        # Analyze each device
        max_excitation_overall = 0
        min_excitation_overall = 1

        for device, device_data in device_results.items():
            if 'excitation_probabilities' in device_data:
                excitation_probs = device_data['excitation_probabilities']
                valid_probs = [p for p in excitation_probs if not np.isnan(p)]

                if valid_probs:
                    max_exc = max(valid_probs)
                    min_exc = min(valid_probs)
                    rabi_freq = device_data.get('rabi_frequency', 0.0)

                    summary['key_results'][device] = {
                        'max_excitation': max_exc,
                        'min_excitation': min_exc,
                        'excitation_range': max_exc - min_exc,
                        'rabi_frequency': rabi_freq,
                        'clear_oscillation': (max_exc - min_exc) > 0.5
                    }

                    max_excitation_overall = max(max_excitation_overall, max_exc)
                    min_excitation_overall = min(min_excitation_overall, min_exc)

        summary['rabi_analysis']['oscillations_detected'] = (max_excitation_overall - min_excitation_overall) > 0.5
        summary['rabi_analysis']['max_excitation_observed'] = max_excitation_overall
        summary['rabi_analysis']['min_excitation_observed'] = min_excitation_overall

        return summary

    def display_results(self, results: Dict[str, Any], use_rich: bool = True) -> None:
        """Display Rabi experiment results in formatted table"""
        device_results = results.get('device_results', {})

        if not device_results:
            print("No device results found")
            return

        if use_rich:
            try:
                from rich.console import Console
                from rich.table import Table

                console = Console()
                table = Table(title="Rabi Oscillation Results", show_header=True, header_style="bold blue")
                table.add_column("Device", style="cyan")
                table.add_column("Max Excitation", justify="right")
                table.add_column("Rabi Frequency", justify="right")
                table.add_column("Method", justify="center")
                table.add_column("Clear Oscillation", justify="center")

                method = results.get('method', 'quantumlib_rabi')

                for device, device_data in device_results.items():
                    if 'excitation_probabilities' in device_data:
                        excitation_probs = device_data['excitation_probabilities']
                        valid_probs = [p for p in excitation_probs if not np.isnan(p)]

                        if valid_probs:
                            max_exc = max(valid_probs)
                            min_exc = min(valid_probs)
                            rabi_freq = device_data.get('rabi_frequency', 0.0)

                            oscillation = "YES" if (max_exc - min_exc) > 0.5 else "NO"
                            oscillation_style = "green" if (max_exc - min_exc) > 0.5 else "yellow"

                            table.add_row(
                                device.upper(),
                                f"{max_exc:.3f}",
                                f"{rabi_freq:.3f}",
                                method,
                                oscillation,
                                style=oscillation_style if (max_exc - min_exc) > 0.5 else None
                            )

                console.print(table)
                console.print(f"\nExpected π pulse: {self.expected_pi_pulse:.3f} rad")
                console.print(f"Clear oscillation threshold: 0.5 excitation range")

            except ImportError:
                use_rich = False

        if not use_rich:
            # Fallback to simple text display
            print("\n" + "="*60)
            print("Rabi Oscillation Results")
            print("="*60)

            method = results.get('method', 'quantumlib_rabi')

            for device, device_data in device_results.items():
                if 'excitation_probabilities' in device_data:
                    excitation_probs = device_data['excitation_probabilities']
                    valid_probs = [p for p in excitation_probs if not np.isnan(p)]

                    if valid_probs:
                        max_exc = max(valid_probs)
                        min_exc = min(valid_probs)
                        rabi_freq = device_data.get('rabi_frequency', 0.0)

                        oscillation = "YES" if (max_exc - min_exc) > 0.5 else "NO"

                        print(f"Device: {device.upper()}")
                        print(f"  Max Excitation: {max_exc:.3f}")
                        print(f"  Rabi Frequency: {rabi_freq:.3f}")
                        print(f"  Method: {method}")
                        print(f"  Clear Oscillation: {oscillation}")
                        print()

            print(f"Expected π pulse: {self.expected_pi_pulse:.3f} rad")
            print(f"Clear oscillation threshold: 0.5 excitation range")
            print("="*60)

    def run_complete_rabi_experiment(self, devices: List[str] = ['qulacs'],
                                   amplitude_points: int = 20, max_amplitude: float = 2*np.pi,
                                   shots: int = 1024, parallel_workers: int = 4,
                                   save_data: bool = True, save_plot: bool = True,
                                   show_plot: bool = False, display_results: bool = True) -> Dict[str, Any]:
        """
        Run complete Rabi experiment with all post-processing
        This is the main entry point for CLI usage
        """
        print(f"🔬 Running complete Rabi experiment: {self.experiment_name}")
        print(f"   Devices: {devices}")
        print(f"   Amplitude points: {amplitude_points}, Max amplitude: {max_amplitude:.3f}")
        print(f"   Shots: {shots}, Parallel workers: {parallel_workers}")

        # Run the Rabi experiment
        results = self.run_experiment(
            devices=devices,
            shots=shots,
            amplitude_points=amplitude_points,
            max_amplitude=max_amplitude
        )

        # Save data if requested
        if save_data:
            self.save_complete_experiment_data(results)
        elif save_plot:
            # Just save plot without full data
            self.generate_rabi_plot(results, save_plot=True, show_plot=show_plot)

        # Display results if requested
        if display_results:
            self.display_results(results, use_rich=True)

        return results
