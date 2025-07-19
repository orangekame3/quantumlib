#!/usr/bin/env python3
"""
Ramsey CLI - QuantumLib Ramsey Oscillation Experiment
"""

from typing import Annotated, Any

import numpy as np
import typer

from oqtopus_experiments.cli.base_cli import (
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
from oqtopus_experiments.experiments.ramsey.ramsey_experiment import RamseyExperiment


class RamseyExperimentCLI(BaseExperimentCLI):
    """
    Ramsey experiment dedicated CLI (using QuantumLib integrated framework)
    """

    def __init__(self):
        super().__init__(
            experiment_name="Ramsey",
            help_text="QuantumLib Ramsey Oscillation Experiment",
        )

    def get_experiment_class(self):
        """Returns the Ramsey experiment class"""
        return RamseyExperiment

    def get_experiment_specific_options(self) -> dict[str, Any]:
        """Ramsey experiment specific options"""
        return {
            "delay_points": 51,
            "max_delay": 200000,
            "detuning": 0.0,
        }

    def create_experiment_config_display(self, **kwargs) -> str:
        """Ramsey experiment configuration display"""
        devices = kwargs.get("devices", ["qulacs"])
        backend = kwargs.get("backend", "local_simulator")
        shots = kwargs.get("shots", 1000)
        parallel = kwargs.get("parallel", 4)
        delay_points = kwargs.get("delay_points", 51)
        max_delay = kwargs.get("max_delay", 200000)
        detuning = kwargs.get("detuning", 0.0)
        return (
            f"QuantumLib Ramsey Oscillation Experiment\\n"
            f"Devices: {', '.join(devices)}\\n"
            f"Backend: {backend}\\n"
            f"Shots: {shots:,} per delay | Points: {delay_points}\\n"
            f"Max Delay: {max_delay / 1000:.1f} μs | Detuning: {detuning} MHz\\n"
            f"Parallel: {parallel} threads\\n"
            f"Total measurements: {delay_points} delay points"
        )

    def generate_circuits(
        self, experiment_instance: RamseyExperiment, **kwargs
    ) -> tuple[list[Any], dict]:
        """Ramsey circuit generation"""
        delay_points = kwargs.get("delay_points", 51)
        max_delay = kwargs.get("max_delay", 200000)
        detuning = kwargs.get("detuning", 0.0)

        # Default delay time configuration
        delay_times = np.logspace(np.log10(50), np.log10(200 * 1000), num=51)

        circuits = experiment_instance.create_circuits(
            delay_points=delay_points,
            max_delay=max_delay,
            detuning=detuning,
            delay_times=delay_times,
        )

        self.console.print(
            f"   Delay range: {delay_points} points from {delay_times[0]:.1f} to {delay_times[-1]:.1f} ns"
        )
        self.console.print(f"   Detuning: {detuning} MHz")

        return circuits, {
            "delay_times": delay_times,
            "max_delay": max_delay,
            "delay_points": delay_points,
            "detuning": detuning,
        }

    def process_results(
        self,
        experiment_instance: RamseyExperiment,
        raw_results: dict,
        circuits: list,
        metadata: Any,
        **kwargs,
    ) -> dict:
        """Ramsey result processing"""
        delay_times = metadata["delay_times"]
        max_delay = metadata["max_delay"]
        delay_points = metadata["delay_points"]
        detuning = metadata["detuning"]

        self.console.print("   → Analyzing Ramsey oscillation...")
        analysis = experiment_instance.analyze_results(raw_results)

        # experiment_params configuration (for saving)
        experiment_instance.experiment_params = {
            "delay_times": delay_times.tolist(),
            "max_delay": max_delay,
            "delay_points": delay_points,
            "detuning": detuning,
        }

        return {
            "delay_times": delay_times,
            "device_results": analysis["device_results"],
            "analysis": analysis,
            "method": "ramsey_quantumlib_framework",
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
        ] = 200000,
        detuning: Annotated[float, typer.Option(help="Frequency detuning [MHz]")] = 0.0,
        enable_fitting: Annotated[
            bool,
            typer.Option(
                "--enable-fitting", help="Enable T2*/detuning parameter fitting"
            ),
        ] = True,
    ):
        """
        Run Ramsey oscillation experiment
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
            delay_points=delay_points,  # Ramsey specific option
            max_delay=max_delay,  # Ramsey specific option
            detuning=detuning,  # Ramsey specific option
            enable_fitting=enable_fitting,  # Fitting enable option
        )

    def main(self):
        """CLI callback - Override in subclass"""
        pass


# CLI instance creation and execution
def main():
    cli = RamseyExperimentCLI()
    cli.start()


if __name__ == "__main__":
    main()
