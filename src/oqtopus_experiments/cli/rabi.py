#!/usr/bin/env python3
"""
Rabi CLI - OQTOPUS Experiments Rabi Oscillation Experiment
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
from oqtopus_experiments.experiments.rabi.rabi_experiment import RabiExperiment


class RabiExperimentCLI(BaseExperimentCLI):
    """
    Rabi experiment dedicated CLI (using OQTOPUS Experiments integrated framework)
    """

    def __init__(self):
        super().__init__(
            experiment_name="Rabi", help_text="OQTOPUS Experiments Rabi Oscillation Experiment"
        )

    def get_experiment_class(self):
        """Returns the Rabi experiment class"""
        return RabiExperiment

    def get_experiment_specific_options(self) -> dict[str, Any]:
        """Rabi experiment specific options"""
        return {
            "points": 20,
            "max_amplitude": 2 * np.pi,
        }

    def create_experiment_config_display(self, **kwargs) -> str:
        """Rabi experiment configuration display"""
        devices = kwargs.get("devices", ["qulacs"])
        backend = kwargs.get("backend", "local_simulator")
        shots = kwargs.get("shots", 1000)
        parallel = kwargs.get("parallel", 4)
        points = kwargs.get("points", 20)
        max_amplitude = kwargs.get("max_amplitude", 2 * np.pi)
        return (
            f"OQTOPUS Experiments Rabi Oscillation Experiment\\n"
            f"Devices: {', '.join(devices)}\\n"
            f"Backend: {backend}\\n"
            f"Shots: {shots:,} per amplitude | Points: {points}\\n"
            f"Max Amplitude: {max_amplitude:.3f} rad\\n"
            f"Parallel: {parallel} threads\\n"
            f"Total measurements: {points} amplitude points"
        )

    def generate_circuits(
        self, experiment_instance: RabiExperiment, **kwargs
    ) -> tuple[list[Any], dict]:
        """Rabi circuit generation"""
        points = kwargs.get("points", 20)
        max_amplitude = kwargs.get("max_amplitude", 2 * np.pi)

        # Generate circuits for Rabi experiment
        amplitude_range = np.linspace(0, max_amplitude, points)

        circuits = experiment_instance.create_circuits(
            amplitude_points=points,
            max_amplitude=max_amplitude,
            drive_time=1.0,
            drive_frequency=0.0,
        )

        self.console.print(
            f"   Amplitude range: {points} points from 0 to {max_amplitude:.3f} rad"
        )
        self.console.print(f"   Expected π pulse at: {np.pi:.3f} rad")

        return circuits, {
            "amplitude_range": amplitude_range,
            "max_amplitude": max_amplitude,
            "points": points,
        }

    def process_results(
        self,
        experiment_instance: RabiExperiment,
        raw_results: dict,
        circuits: list,
        metadata: Any,
        **kwargs,
    ) -> dict:
        """Rabi result processing"""
        amplitude_range = metadata["amplitude_range"]
        max_amplitude = metadata["max_amplitude"]
        points = metadata["points"]

        self.console.print("   → Analyzing Rabi oscillation...")
        analysis = experiment_instance.analyze_results(raw_results)

        # experiment_params configuration (for saving)
        experiment_instance.experiment_params = {
            "amplitude_range": amplitude_range.tolist(),
            "max_amplitude": max_amplitude,
            "amplitude_points": points,
        }

        return {
            "amplitude_range": amplitude_range,
            "device_results": analysis["device_results"],
            "analysis": analysis,
            "method": "rabi_quantumlib_framework",
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
            int, typer.Option(help="Number of amplitude points to scan")
        ] = 20,
        max_amplitude: Annotated[
            float, typer.Option(help="Maximum drive amplitude [rad]")
        ] = 2
        * np.pi,
    ):
        """
        Run Rabi oscillation experiment
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
            points=points,  # Rabi specific option
            max_amplitude=max_amplitude,  # Rabi specific option
        )

    def main(self):
        """CLI entry point implementation"""
        pass


def main():
    """Main entry point for quantumlib-rabi CLI command"""
    cli = RabiExperimentCLI()
    cli.start()


if __name__ == "__main__":
    main()
