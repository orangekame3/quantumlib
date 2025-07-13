#!/usr/bin/env python3
"""
CHSH CLI - QuantumLib CHSH Bell Inequality Experiment
"""
from typing import Annotated, Any

import numpy as np
import typer

from quantumlib.circuit.chsh_circuits import create_chsh_circuit
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
from quantumlib.experiments.chsh.chsh_experiment import CHSHExperiment


class CHSHExperimentCLI(BaseExperimentCLI):
    """
    CHSH実験専用CLI（QuantumLib統合フレームワーク使用）
    """

    def __init__(self):
        super().__init__(
            experiment_name="CHSH",
            help_text="QuantumLib CHSH Bell Inequality Experiment",
        )

    def get_experiment_class(self):
        """CHSH実験クラスを返す"""
        return CHSHExperiment

    def create_experiment_config_display(self, **kwargs) -> str:
        """CHSH実験設定表示"""
        devices = kwargs.get("devices", ["qulacs"])
        backend = kwargs.get("backend", "local_simulator")
        shots = kwargs.get("shots", 1000)
        parallel = kwargs.get("parallel", 4)
        points = kwargs.get("points", 20)
        return (
            f"QuantumLib CHSH Bell Inequality Verification\\n"
            f"Devices: {', '.join(devices)}\\n"
            f"Backend: {backend}\\n"
            f"Shots: {shots:,} per measurement | Points: {points}\\n"
            f"Parallel: {parallel} threads\\n"
            f"Total measurements: {points * 4} (4 per phase point)"
        )

    def generate_circuits(
        self, experiment_instance: CHSHExperiment, **kwargs
    ) -> tuple[list[Any], dict]:
        """CHSH回路生成"""
        points = kwargs.get("points", 20)

        # 4測定CHSH用の回路とメタデータを生成
        phase_range = np.linspace(0, 2 * np.pi, points)
        angles = {
            "theta_a0": 0,
            "theta_a1": np.pi / 2,
            "theta_b0": np.pi / 4,
            "theta_b1": -np.pi / 4,
        }
        measurements = [
            (angles["theta_a0"], angles["theta_b0"]),
            (angles["theta_a0"], angles["theta_b1"]),
            (angles["theta_a1"], angles["theta_b0"]),
            (angles["theta_a1"], angles["theta_b1"]),
        ]

        circuits = []
        circuit_metadata = []
        for i, phase_phi in enumerate(phase_range):
            for j, (theta_a, theta_b) in enumerate(measurements):
                circuit = create_chsh_circuit(theta_a, theta_b, phase_phi)
                circuits.append(circuit)
                circuit_metadata.append(
                    {
                        "phase_index": i,
                        "measurement_index": j,
                        "phase_phi": phase_phi,
                        "theta_a": theta_a,
                        "theta_b": theta_b,
                    }
                )

        self.console.print(f"   Phase range: {points} points from 0 to 2π")
        self.console.print("   Measurements: 4 combinations per phase")

        return circuits, {
            "circuit_metadata": circuit_metadata,
            "phase_range": phase_range,
            "angles": angles,
            "measurements": measurements,
        }

    def process_results(
        self,
        experiment_instance: CHSHExperiment,
        raw_results: dict,
        circuits: list,
        metadata: Any,
        **kwargs,
    ) -> dict:
        """CHSH結果処理"""
        device_list = kwargs.get("device_list", ["qulacs"])

        circuit_metadata = metadata["circuit_metadata"]
        phase_range = metadata["phase_range"]
        angles = metadata["angles"]
        measurements = metadata["measurements"]

        self.console.print("   → Processing measurement results...")
        processed_results = experiment_instance._process_4_measurement_results(
            raw_results, circuit_metadata, phase_range, measurements, device_list
        )

        self.console.print("   → Creating analysis...")
        analysis = experiment_instance._create_4_measurement_analysis(
            phase_range, processed_results, angles
        )

        # experiment_params設定（保存用）
        experiment_instance.experiment_params = {
            "theta_a": 0,
            "theta_b": np.pi / 4,
            "phase_range": phase_range.tolist(),
            "phase_points": len(phase_range),
        }

        return {
            "phase_range": phase_range,
            "device_results": processed_results,
            "analysis": analysis,
            "method": "chsh_quantumlib_framework",
        }

    def get_experiment_specific_options(self) -> dict[str, Any]:
        """CHSH実験固有のオプション"""
        return {
            "points": 20,  # デフォルト位相点数
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
        points: Annotated[
            int, typer.Option(help="Number of phase points to scan")
        ] = 20,
    ):
        """
        Run CHSH Bell inequality experiment
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
            points=points,  # CHSH固有オプション
        )


# CLIインスタンス作成と実行
def main():
    cli = CHSHExperimentCLI()
    cli.start()


if __name__ == "__main__":
    main()
