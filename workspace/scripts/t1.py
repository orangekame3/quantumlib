#!/usr/bin/env python3
"""
T1 CLI - QuantumLib T1 Decay Experiment
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
from quantumlib.experiments.t1.t1_experiment import T1Experiment


class T1ExperimentCLI(BaseExperimentCLI):
    """
    T1実験専用CLI（QuantumLib統合フレームワーク使用）
    """

    def __init__(self):
        super().__init__(
            experiment_name="T1", help_text="QuantumLib T1 Decay Experiment"
        )

    def get_experiment_class(self):
        """T1実験クラスを返す"""
        return T1Experiment

    def get_experiment_specific_options(self) -> dict[str, Any]:
        """T1実験固有のオプション"""
        return {
            "delay_points": 16,
            "max_delay": 1000,
            "t1": 500,
            "t2": 500,
        }

    def create_experiment_config_display(self, **kwargs) -> str:
        """T1実験設定表示"""
        devices = kwargs.get("devices", ["qulacs"])
        backend = kwargs.get("backend", "local_simulator")
        shots = kwargs.get("shots", 1000)
        parallel = kwargs.get("parallel", 4)
        delay_points = kwargs.get("delay_points", 16)
        max_delay = kwargs.get("max_delay", 1000)
        t1 = kwargs.get("t1", 500)
        t2 = kwargs.get("t2", 500)
        return (
            f"QuantumLib T1 Decay Experiment\\n"
            f"Devices: {', '.join(devices)}\\n"
            f"Backend: {backend}\\n"
            f"Shots: {shots:,} per delay | Points: {delay_points}\\n"
            f"Max Delay: {max_delay} ns | T1: {t1} ns | T2: {t2} ns\\n"
            f"Parallel: {parallel} threads\\n"
            f"Total measurements: {delay_points} delay points"
        )

    def generate_circuits(
        self, experiment_instance: T1Experiment, **kwargs
    ) -> tuple[list[Any], dict]:
        """T1回路生成"""
        delay_points = kwargs.get("delay_points", 16)
        max_delay = kwargs.get("max_delay", 1000)
        t1 = kwargs.get("t1", 500)
        t2 = kwargs.get("t2", 500)

        # デフォルトの遅延時間設定
        delay_times = np.array([1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
        if delay_points != 16:
            delay_times = np.linspace(1, max_delay, delay_points)

        circuits = experiment_instance.create_circuits(
            delay_points=delay_points,
            max_delay=max_delay,
            t1=t1,
            t2=t2,
            delay_times=delay_times,
        )

        self.console.print(
            f"   Delay range: {delay_points} points from {delay_times[0]:.1f} to {delay_times[-1]:.1f} ns"
        )
        self.console.print(f"   T1: {t1} ns, T2: {t2} ns")

        return circuits, {
            "delay_times": delay_times,
            "max_delay": max_delay,
            "delay_points": delay_points,
            "t1": t1,
            "t2": t2,
        }

    def process_results(
        self,
        experiment_instance: T1Experiment,
        raw_results: dict,
        circuits: list,
        metadata: Any,
        **kwargs,
    ) -> dict:
        """T1結果処理"""
        delay_times = metadata["delay_times"]
        max_delay = metadata["max_delay"]
        delay_points = metadata["delay_points"]
        t1 = metadata["t1"]
        t2 = metadata["t2"]

        self.console.print("   → Analyzing T1 decay...")
        analysis = experiment_instance.analyze_results(raw_results)

        # experiment_params設定（保存用）
        experiment_instance.experiment_params = {
            "delay_times": delay_times.tolist(),
            "max_delay": max_delay,
            "delay_points": delay_points,
            "t1_theoretical": t1,
            "t2_theoretical": t2,
        }

        return {
            "delay_times": delay_times,
            "device_results": analysis["device_results"],
            "analysis": analysis,
            "method": "t1_quantumlib_framework",
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
        ] = 16,
        max_delay: Annotated[
            float, typer.Option(help="Maximum delay time [ns]")
        ] = 1000,
        t1: Annotated[
            float, typer.Option(help="T1 relaxation time [ns] for simulation")
        ] = 500,
        t2: Annotated[
            float, typer.Option(help="T2 relaxation time [ns] for simulation")
        ] = 500,
    ):
        """
        Run T1 decay experiment
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
            delay_points=delay_points,  # T1固有オプション
            max_delay=max_delay,  # T1固有オプション
            t1=t1,  # T1固有オプション
            t2=t2,  # T1固有オプション
        )


# CLIインスタンス作成と実行
def main():
    cli = T1ExperimentCLI()
    cli.start()


if __name__ == "__main__":
    main()