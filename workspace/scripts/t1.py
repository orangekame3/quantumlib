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
    T1 experiment dedicated CLI (using QuantumLib integrated framework)
    """

    def __init__(self):
        super().__init__(
            experiment_name="T1", help_text="QuantumLib T1 Decay Experiment"
        )

    def get_experiment_class(self):
        """Returns the T1 experiment class"""
        return T1Experiment

    def get_experiment_specific_options(self) -> dict[str, Any]:
        """T1 experiment specific options"""
        return {
            "delay_points": 51,
            "max_delay": 100000,
        }

    def create_experiment_config_display(self, **kwargs) -> str:
        """T1 experiment configuration display"""
        devices = kwargs.get("devices", ["qulacs"])
        backend = kwargs.get("backend", "local_simulator")
        shots = kwargs.get("shots", 1000)
        parallel = kwargs.get("parallel", 4)
        delay_points = kwargs.get("delay_points", 51)
        max_delay = kwargs.get("max_delay", 100000)
        return (
            f"QuantumLib T1 Decay Experiment\\n"
            f"Devices: {', '.join(devices)}\\n"
            f"Backend: {backend}\\n"
            f"Shots: {shots:,} per delay | Points: {delay_points}\\n"
            f"Max Delay: {max_delay / 1000:.1f} μs\\n"
            f"Parallel: {parallel} threads\\n"
            f"Total measurements: {delay_points} delay points"
        )

    def generate_circuits(
        self, experiment_instance: T1Experiment, **kwargs
    ) -> tuple[list[Any], dict]:
        """T1 circuit generation"""
        delay_points = kwargs.get("delay_points", 51)
        max_delay = kwargs.get("max_delay", 100000)

        # Default delay time configuration
        delay_times = np.logspace(np.log10(100), np.log10(100 * 1000), num=51)

        circuits = experiment_instance.create_circuits(
            delay_points=delay_points,
            max_delay=max_delay,
            delay_times=delay_times,
        )

        self.console.print(
            f"   Delay range: {delay_points} points from {delay_times[0]:.1f} to {delay_times[-1]:.1f} ns"
        )

        return circuits, {
            "delay_times": delay_times,
            "max_delay": max_delay,
            "delay_points": delay_points,
        }

    def process_results(
        self,
        experiment_instance: T1Experiment,
        raw_results: dict,
        circuits: list,
        metadata: Any,
        **kwargs,
    ) -> dict:
        """T1 result processing"""
        delay_times = metadata["delay_times"]
        max_delay = metadata["max_delay"]
        delay_points = metadata["delay_points"]

        self.console.print("   → Analyzing T1 decay...")
        analysis = experiment_instance.analyze_results(raw_results)

        # experiment_params configuration (for saving)
        experiment_instance.experiment_params = {
            "delay_times": delay_times.tolist(),
            "max_delay": max_delay,
            "delay_points": delay_points,
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
        ] = 51,
        max_delay: Annotated[
            float, typer.Option(help="Maximum delay time [ns]")
        ] = 100000,
    ):
        """
        Run T1 decay experiment
        """
        # Call framework's common execution logic
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
            delay_points=delay_points,  # T1 specific option
            max_delay=max_delay,  # T1 specific option
        )


# CLI instance creation and execution
def main():
    cli = T1ExperimentCLI()
    cli.start()


if __name__ == "__main__":
    main()
