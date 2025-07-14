#!/usr/bin/env python3
"""
T2 Echo CLI - QuantumLib T2 Echo実験（Hahn Echo/CPMG）
"""
from typing import Annotated, Any

import numpy as np
import typer

from quantumlib.cli.base_cli import (
    BaseExperimentCLI,
    CommonBackendOption,
    CommonDevicesOption,
    CommonExperimentNameOption,
    CommonNoPlotOption,
    CommonNoSaveOption,
    CommonParallelOption,
    CommonShotsOption,
    CommonShowPlotOption,
    CommonVerboseOption,
    DeviceType,
    ExperimentBackend,
)
from quantumlib.experiments.t2_echo.t2_echo_experiment import T2EchoExperiment


class T2EchoExperimentCLI(BaseExperimentCLI):
    """
    T2 Echo実験専用CLI（QuantumLib統合フレームワーク使用）
    """

    def __init__(self):
        super().__init__(
            experiment_name="T2Echo", help_text="QuantumLib T2 Echo Experiment (Hahn Echo/CPMG)"
        )

    def get_experiment_class(self):
        """T2 Echo実験クラスを返す"""
        return T2EchoExperiment

    def get_experiment_specific_options(self) -> dict[str, Any]:
        """T2 Echo実験固有のオプション"""
        return {
            "delay_points": 51,
            "max_delay": 500000,  # 500μs
            "echo_type": "hahn",
            "num_echoes": 1,
        }

    def create_experiment_config_display(self, **kwargs) -> str:
        """T2 Echo実験設定表示"""
        devices = kwargs.get("devices", ["qulacs"])
        backend = kwargs.get("backend", "local_simulator")
        shots = kwargs.get("shots", 1000)
        parallel = kwargs.get("parallel", 4)
        delay_points = kwargs.get("delay_points", 51)
        max_delay = kwargs.get("max_delay", 500000)
        echo_type = kwargs.get("echo_type", "hahn")
        num_echoes = kwargs.get("num_echoes", 1)
        return (
            f"QuantumLib T2 Echo Experiment\\n"
            f"Devices: {', '.join(devices)}\\n"
            f"Backend: {backend}\\n"
            f"Shots: {shots:,} per delay | Points: {delay_points}\\n"
            f"Max Delay: {max_delay/1000:.1f} μs | Echo: {echo_type.upper()}\\n"
            f"Echo Count: {num_echoes} | Parallel: {parallel} threads\\n"
            f"Total measurements: {delay_points} delay points"
        )

    def generate_circuits(
        self, experiment_instance: T2EchoExperiment, **kwargs
    ) -> tuple[list[Any], dict]:
        """T2 Echo回路生成"""
        delay_points = kwargs.get("delay_points", 51)
        max_delay = kwargs.get("max_delay", 500000)
        echo_type = kwargs.get("echo_type", "hahn")
        num_echoes = kwargs.get("num_echoes", 1)

        # デフォルトの遅延時間設定
        delay_times = np.logspace(np.log10(100), np.log10(500 * 1000), num=51)

        circuits = experiment_instance.create_circuits(
            delay_points=delay_points,
            max_delay=max_delay,
            echo_type=echo_type,
            num_echoes=num_echoes,
            delay_times=delay_times,
        )

        self.console.print(
            f"   Delay range: {delay_points} points from {delay_times[0]:.1f} to {delay_times[-1]:.1f} ns"
        )
        self.console.print(f"   Echo type: {echo_type.upper()} (echoes={num_echoes})")

        return circuits, {
            "delay_times": delay_times,
            "max_delay": max_delay,
            "delay_points": delay_points,
            "echo_type": echo_type,
            "num_echoes": num_echoes,
        }

    def process_results(
        self,
        experiment_instance: T2EchoExperiment,
        raw_results: dict,
        circuits: list,
        metadata: Any,
        **kwargs,
    ) -> dict:
        """T2 Echo結果処理"""
        delay_times = metadata["delay_times"]
        max_delay = metadata["max_delay"]
        delay_points = metadata["delay_points"]
        echo_type = metadata["echo_type"]
        num_echoes = metadata["num_echoes"]

        self.console.print("   → Analyzing T2 Echo decay...")
        analysis = experiment_instance.analyze_results(raw_results)

        # experiment_params設定（保存用）
        experiment_instance.experiment_params = {
            "delay_times": delay_times.tolist(),
            "max_delay": max_delay,
            "delay_points": delay_points,
            "echo_type": echo_type,
            "num_echoes": num_echoes,
        }

        return {
            "delay_times": delay_times,
            "device_results": analysis["device_results"],
            "analysis": analysis,
            "method": "t2_echo_quantumlib_framework",
        }

    def run(
        self,
        devices: CommonDevicesOption = [DeviceType.qulacs],
        shots: CommonShotsOption = 1000,
        backend: CommonBackendOption = ExperimentBackend.local_simulator,
        parallel: CommonParallelOption = 4,
        experiment_name: CommonExperimentNameOption = None,
        no_save: CommonNoSaveOption = False,
        no_plot: CommonNoPlotOption = False,
        show_plot: CommonShowPlotOption = False,
        verbose: CommonVerboseOption = False,
        delay_points: Annotated[
            int, typer.Option(help="Number of delay time points to scan")
        ] = 51,
        max_delay: Annotated[
            float, typer.Option(help="Maximum delay time [ns]")
        ] = 500000,
        echo_type: Annotated[
            str, typer.Option(help="Echo type: 'hahn' or 'cpmg'")
        ] = "hahn",
        num_echoes: Annotated[
            int, typer.Option(help="Number of echoes (for CPMG)")
        ] = 1,
        enable_fitting: Annotated[
            bool, typer.Option("--enable-fitting", help="Enable T2 parameter fitting")
        ] = True,
    ):
        """
        Run T2 Echo experiment (Hahn Echo/CPMG)
        """
        # フレームワークの共通実行ロジックを呼び出し
        self._execute_experiment(
            devices=[d.value for d in devices],
            shots=shots,
            backend=backend.value,
            parallel=parallel,
            experiment_name=experiment_name,
            no_save=no_save,
            no_plot=no_plot,
            show_plot=show_plot,
            verbose=verbose,
            delay_points=delay_points,  # T2 Echo固有オプション
            max_delay=max_delay,  # T2 Echo固有オプション
            echo_type=echo_type,  # T2 Echo固有オプション
            num_echoes=num_echoes,  # T2 Echo固有オプション
            enable_fitting=enable_fitting,  # フィッティング有効化オプション
        )


# CLIインスタンス作成と実行
def main():
    cli = T2EchoExperimentCLI()
    cli.start()


if __name__ == "__main__":
    main()