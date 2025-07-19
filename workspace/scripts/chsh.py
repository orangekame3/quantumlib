#!/usr/bin/env python3
"""
CHSH CLI - QuantumLib CHSH Bell Inequality Experiment
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
from oqtopus_experiments.experiments.chsh.chsh_experiment import CHSHExperiment


class CHSHExperimentCLI(BaseExperimentCLI):
    """
    CHSH experiment dedicated CLI (using QuantumLib integrated framework)
    """

    def __init__(self):
        super().__init__(
            experiment_name="CHSH",
            help_text="QuantumLib CHSH Bell Inequality Experiment",
        )

    def get_experiment_class(self):
        """Returns the CHSH experiment class"""
        return CHSHExperiment

    def create_experiment_config_display(self, **kwargs) -> str:
        """CHSH experiment configuration display"""
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
        """CHSH circuit generation using modern classmethod approach"""
        points = kwargs.get("points", 20)
        basis_gates = kwargs.get("basis_gates")
        optimization_level = kwargs.get("optimization_level", 1)

        # Use the new classmethod for stateless circuit creation
        circuits, metadata = CHSHExperiment.create_chsh_circuits(
            phase_points=points,
            theta_a=0.0,
            theta_b=np.pi / 4,
            basis_gates=basis_gates,
            optimization_level=optimization_level,
        )

        self.console.print(
            f"   Phase range: {metadata['phase_points']} points from 0 to 2π"
        )
        self.console.print(
            f"   Total circuits: {metadata['total_circuits']} (4 measurements per phase)"
        )
        self.console.print(
            f"   Theta A: {metadata['theta_a']:.3f}, Theta B: {metadata['theta_b']:.3f}"
        )

        return circuits, metadata

    def process_results(
        self,
        experiment_instance: CHSHExperiment,
        raw_results: dict,
        circuits: list,
        metadata: Any,
        **kwargs,
    ) -> dict:
        """CHSH result processing"""
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

        # experiment_params configuration (for saving)
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
        """CHSH experiment specific options"""
        return {
            "points": 20,  # Default phase points
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
            points=points,  # CHSH specific option
        )

    def main(self):
        """CLI callback - Override in subclass"""
        pass


# CLI instance creation and execution
def main():
    cli = CHSHExperimentCLI()
    cli.start()


if __name__ == "__main__":
    main()
