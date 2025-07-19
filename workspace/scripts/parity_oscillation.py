#!/usr/bin/env python3
"""
Parity Oscillation CLI - QuantumLib Parity Oscillation Experiment
Study GHZ state decoherence through parity oscillation measurements
"""

from typing import Annotated, Any

import typer

from oqtopus_experiments.cli.base_cli import (
    BaseExperimentCLI,
    CommonBackendOption,
    CommonDevicesOption,
    CommonExperimentNameOption,
    CommonNoMitigationOption,
    CommonNoPlotOption,
    CommonNoSaveOption,
    CommonParallelOption,
    CommonShotsOption,
    CommonShowPlotOption,
    CommonVerboseOption,
    DeviceType,
    ExperimentBackend,
)
from oqtopus_experiments.experiments.parity_oscillation import ParityOscillationExperiment


class ParityOscillationExperimentCLI(BaseExperimentCLI):
    """
    Parity oscillation experiment dedicated CLI (using QuantumLib integrated framework)
    """

    def __init__(self):
        super().__init__(
            experiment_name="parityoscillation",
            help_text="QuantumLib Parity Oscillation (GHZ Decoherence) Experiment",
        )

    def get_experiment_class(self):
        """Returns the Parity Oscillation experiment class"""
        return ParityOscillationExperiment

    def get_experiment_specific_options(self) -> dict[str, Any]:
        """Parity oscillation experiment specific options"""
        return {
            "num_qubits_list": [1, 2, 3, 4, 5],
            "delays_us": [0, 1, 2, 4, 8, 16],
            "phase_points": None,  # Auto: 4N+1
        }

    def create_experiment_config_display(self, **kwargs) -> str:
        """Parity oscillation experiment configuration display"""
        devices = kwargs.get("devices", ["qulacs"])
        backend = kwargs.get("backend", "local_simulator")
        shots = kwargs.get("shots", 1000)
        parallel = kwargs.get("parallel", 4)
        num_qubits_list = kwargs.get("num_qubits_list", [1, 2, 3, 4, 5])
        delays_us = kwargs.get("delays_us", [0, 1, 2, 4, 8, 16])
        phase_points = kwargs.get("phase_points", None)

        max_qubits = max(num_qubits_list)
        actual_phase_points = phase_points or (4 * max_qubits + 1)
        total_circuits = len(num_qubits_list) * len(delays_us) * actual_phase_points

        return (
            f"QuantumLib Parity Oscillation Experiment\\n"
            f"Devices: {', '.join(devices)}\\n"
            f"Backend: {backend}\\n"
            f"Shots: {shots:,} per circuit\\n"
            f"Qubits: {num_qubits_list}\\n"
            f"Delays: {delays_us} μs\\n"
            f"Phase points: {actual_phase_points} (4N+1 for max N={max_qubits})\\n"
            f"Parallel: {parallel} threads\\n"
            f"Total circuits: {total_circuits}"
        )

    def generate_circuits(
        self, experiment_instance: ParityOscillationExperiment, **kwargs
    ) -> tuple[list[Any], dict]:
        """Parity oscillation circuit generation"""
        num_qubits_list = kwargs.get("num_qubits_list", [1, 2, 3, 4, 5])
        delays_us = kwargs.get("delays_us", [0, 1, 2, 4, 8, 16])
        phase_points = kwargs.get("phase_points", None)

        # Generate circuits for parity oscillation experiment
        circuits = experiment_instance.create_circuits(
            num_qubits_list=num_qubits_list,
            delays_us=delays_us,
            phase_points=phase_points,
            no_delay=kwargs.get("no_delay", False),
        )

        self.console.print(f"   Qubit counts: {num_qubits_list}")
        self.console.print(f"   Delay times: {delays_us} μs")
        if phase_points:
            self.console.print(f"   Phase points: {phase_points} (fixed)")
        else:
            max_qubits = max(num_qubits_list)
            auto_points = 4 * max_qubits + 1
            self.console.print(
                f"   Phase points: {auto_points} (4N+1 for max N={max_qubits})"
            )

        return circuits, {
            "num_qubits_list": num_qubits_list,
            "delays_us": delays_us,
            "phase_points": phase_points,
        }

    def process_results(
        self,
        experiment_instance: ParityOscillationExperiment,
        raw_results: dict,
        circuits: list,
        metadata: Any,
        **kwargs,
    ) -> dict:
        """Parity oscillation result processing"""
        num_qubits_list = metadata["num_qubits_list"]
        delays_us = metadata["delays_us"]
        phase_points = metadata["phase_points"]

        self.console.print("   → Analyzing parity oscillations...")
        analysis = experiment_instance.analyze_results(raw_results)

        # experiment_params configuration (for saving)
        experiment_instance.experiment_params = {
            "num_qubits_list": num_qubits_list,
            "delays_us": delays_us,
            "phase_points": phase_points,
            "total_combinations": len(num_qubits_list) * len(delays_us),
        }

        return {
            "num_qubits_list": num_qubits_list,
            "delays_us": delays_us,
            "device_results": analysis,
            "analysis": analysis,
            "method": "parity_oscillation_quantumlib_framework",
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
        num_qubits: Annotated[
            list[int], typer.Option(help="List of qubit counts to test")
        ] = [1, 2, 3, 4, 5],
        delays: Annotated[
            list[float], typer.Option(help="List of delay times in microseconds")
        ] = [0, 1, 2, 4, 8, 16],
        phase_points: Annotated[
            int | None,
            typer.Option(help="Number of phase points (default: 4N+1 for each N)"),
        ] = None,
        no_delay: Annotated[
            bool,
            typer.Option(help="Skip delay gates for faster execution"),
        ] = False,
        no_mitigation: CommonNoMitigationOption = False,
    ):
        """
        Run parity oscillation experiment for GHZ state decoherence study
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
            num_qubits_list=num_qubits,  # Parity oscillation specific option
            delays_us=delays,  # Parity oscillation specific option
            phase_points=phase_points,  # Parity oscillation specific option
            no_delay=no_delay,  # Parity oscillation specific option
            no_mitigation=no_mitigation,  # Mitigation control option
        )

    def main(self):
        """CLI callback - Override in subclass"""
        pass


# CLI instance creation and execution
def main():
    cli = ParityOscillationExperimentCLI()
    cli.start()


if __name__ == "__main__":
    main()
