#!/usr/bin/env python3
"""
Base CLI Framework - QuantumLib experiment CLI common framework
Base class inherited by all experiment CLIs
"""

import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)


class DeviceType(str, Enum):
    qulacs = "qulacs"
    anemone = "anemone"
    Kawasaki = "Kawasaki"


class ExperimentBackend(str, Enum):
    local_simulator = "local_simulator"
    oqtopus = "oqtopus"


# Common option type definitions
CommonDevicesOption = Annotated[
    list[DeviceType], typer.Option(help="Quantum devices to use")
]
CommonShotsOption = Annotated[int, typer.Option(help="Number of measurement shots")]
CommonBackendOption = Annotated[
    ExperimentBackend, typer.Option(help="Experiment backend to use")
]
CommonParallelOption = Annotated[int, typer.Option(help="Number of parallel threads")]
CommonExperimentNameOption = Annotated[
    str | None, typer.Option(help="Custom experiment name")
]
CommonNoSaveOption = Annotated[bool, typer.Option("--no-save", help="Skip saving data")]
CommonNoPlotOption = Annotated[
    bool, typer.Option("--no-plot", help="Skip generating plot")
]
CommonShowPlotOption = Annotated[
    bool, typer.Option("--show-plot", help="Display plot interactively")
]
CommonVerboseOption = Annotated[
    bool, typer.Option("--verbose", "-v", help="Verbose output")
]


class BaseExperimentCLI(ABC):
    """
    Experiment CLI base class

    Common features:
    - 3-step execution procedure (circuit generation → parallel execution → analysis & save)
    - Backend injection support
    - Progress tracking
    - Rich UI
    """

    def __init__(self, experiment_name: str, help_text: str):
        self.experiment_name = experiment_name
        self.console = Console()
        self.app = typer.Typer(
            name=f"{experiment_name}-cli", help=help_text, rich_markup_mode="rich"
        )

        # Add run command
        self.app.command()(self.run)
        self.app.callback()(self.main)

    @abstractmethod
    def get_experiment_class(self):
        """Return experiment class (implemented in subclasses)"""
        pass

    @abstractmethod
    def generate_circuits(self, experiment_instance: Any, **kwargs) -> tuple:
        """
        Circuit generation (implemented in subclasses)

        Returns:
            (circuits, metadata) tuple
        """
        pass

    @abstractmethod
    def get_experiment_specific_options(self) -> dict[str, Any]:
        """Define experiment-specific options (implemented in subclasses)"""
        pass

    @abstractmethod
    def create_experiment_config_display(self, **kwargs) -> str:
        """Create experiment configuration display text (implemented in subclasses)"""
        pass

    @abstractmethod
    def process_results(
        self,
        experiment_instance: Any,
        raw_results: dict,
        circuits: list,
        metadata: Any,
        **kwargs,
    ) -> dict:
        """Process results (implemented in subclasses)"""
        pass

    def run_parallel_execution(
        self,
        experiment_instance: Any,
        circuits: list,
        devices: list[str],
        shots: int,
        parallel_workers: int,
        backend: ExperimentBackend,
    ) -> dict:
        """
        Step 2: Execute circuits in parallel execution environment
        Common execution logic for all experiments
        """
        self.console.print("\\n🔧 Step 2: Parallel Execution Engine")
        self.console.print(f"   Backend: {backend.value}")
        self.console.print(f"   Circuits: {len(circuits)}")
        self.console.print(f"   Devices: {devices}")
        self.console.print(f"   Workers: {parallel_workers}")

        # Switch execution environment according to backend
        if backend == ExperimentBackend.oqtopus:
            self.console.print("   → Using OQTOPUS backend")
            if not experiment_instance.oqtopus_available:
                self.console.print(
                    "   ⚠️  OQTOPUS not available, falling back to local simulator",
                    style="yellow",
                )
                return self._run_local_execution(
                    experiment_instance, circuits, devices, shots, parallel_workers
                )
            else:
                return self._run_oqtopus_execution(
                    experiment_instance, circuits, devices, shots, parallel_workers
                )

        else:  # local_simulator
            self.console.print("   → Using local simulator backend")
            return self._run_local_execution(
                experiment_instance, circuits, devices, shots, parallel_workers
            )

    def _run_oqtopus_execution(
        self,
        experiment_instance: Any,
        circuits: list,
        devices: list[str],
        shots: int,
        parallel_workers: int,
    ) -> dict:
        """Parallel execution with OQTOPUS backend"""

        # Use dedicated parallel implementation for T1 experiments (without progress bar)
        if hasattr(experiment_instance, "_submit_t1_circuits_parallel_with_order"):
            self.console.print("   → Using T1-specific parallel execution")

            # Avoid stack issues with simple approach without progress bar
            self.console.print("   📊 Submitting T1 circuits...")
            job_data = experiment_instance._submit_t1_circuits_parallel_with_order(
                circuits, devices, shots, parallel_workers
            )
            self.console.print("   ✅ T1 circuits submitted")

            self.console.print("   📊 Collecting T1 results...")
            try:
                raw_results = (
                    experiment_instance._collect_t1_results_parallel_with_order(
                        job_data, parallel_workers
                    )
                )
                self.console.print("   ✅ T1 results collected")
            except Exception as e:
                self.console.print(f"   ❌ T1 collection failed: {e}")
                raise

            return raw_results

        # Use dedicated parallel implementation for Ramsey experiments (without progress bar)
        if hasattr(experiment_instance, "_submit_ramsey_circuits_parallel_with_order"):
            self.console.print("   → Using Ramsey-specific parallel execution")

            # Avoid stack issues with simple approach without progress bar
            self.console.print("   📊 Submitting Ramsey circuits...")
            job_data = experiment_instance._submit_ramsey_circuits_parallel_with_order(
                circuits, devices, shots, parallel_workers
            )
            self.console.print("   ✅ Ramsey circuits submitted")

            self.console.print("   📊 Collecting Ramsey results...")
            try:
                raw_results = (
                    experiment_instance._collect_ramsey_results_parallel_with_order(
                        job_data, parallel_workers
                    )
                )
                self.console.print("   ✅ Ramsey results collected")
            except Exception as e:
                self.console.print(f"   ❌ Ramsey collection failed: {e}")
                raise

            return raw_results

        # Use dedicated parallel implementation for T2 Echo experiments (without progress bar)
        if hasattr(experiment_instance, "_submit_t2_echo_circuits_parallel_with_order"):
            self.console.print("   → Using T2 Echo-specific parallel execution")

            # Avoid stack issues with simple approach without progress bar
            self.console.print("   📊 Submitting T2 Echo circuits...")
            job_data = experiment_instance._submit_t2_echo_circuits_parallel_with_order(
                circuits, devices, shots, parallel_workers
            )
            self.console.print("   ✅ T2 Echo circuits submitted")

            self.console.print("   📊 Collecting T2 Echo results...")
            try:
                raw_results = (
                    experiment_instance._collect_t2_echo_results_parallel_with_order(
                        job_data, parallel_workers
                    )
                )
                self.console.print("   ✅ T2 Echo results collected")
            except Exception as e:
                self.console.print(f"   ❌ T2 Echo collection failed: {e}")
                raise

            return raw_results

        # Use dedicated parallel implementation for CHSH experiments (without progress bar)
        if hasattr(experiment_instance, "_submit_chsh_circuits_parallel_with_order"):
            self.console.print("   → Using CHSH-specific parallel execution")

            # Avoid stack issues with simple approach without progress bar
            self.console.print("   📊 Submitting CHSH circuits...")
            job_data = experiment_instance._submit_chsh_circuits_parallel_with_order(
                circuits, devices, shots, parallel_workers
            )
            self.console.print("   ✅ CHSH circuits submitted")

            self.console.print("   📊 Collecting CHSH results...")
            try:
                raw_results = (
                    experiment_instance._collect_chsh_results_parallel_with_order(
                        job_data, parallel_workers
                    )
                )
                self.console.print("   ✅ CHSH results collected")
            except Exception as e:
                self.console.print(f"   ❌ CHSH collection failed: {e}")
                raise

            return raw_results

        # Normal parallel execution (as usual)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            # Submit phase
            submit_task = progress.add_task(
                "Submitting to OQTOPUS...", total=len(circuits) * len(devices)
            )

            job_data = experiment_instance.submit_circuits_parallel(
                circuits, devices, shots, parallel_workers
            )
            progress.update(submit_task, completed=len(circuits) * len(devices))

            # Collect phase
            collect_task = progress.add_task(
                "Collecting results...", total=len(circuits) * len(devices)
            )

            raw_results = experiment_instance.collect_results_parallel(
                job_data, parallel_workers
            )
            progress.update(collect_task, completed=len(circuits) * len(devices))

        return raw_results

    def _run_local_execution(
        self,
        experiment_instance: Any,
        circuits: list,
        devices: list[str],
        shots: int,
        parallel_workers: int,
    ) -> dict:
        """Parallel execution with local simulator"""

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            # Local execution (submit and run are combined)
            execute_task = progress.add_task(
                "Executing locally...", total=len(circuits) * len(devices)
            )

            raw_results = {}
            for device in devices:
                device_results = []
                for circuit in circuits:
                    result = experiment_instance.run_circuit_locally(circuit, shots)
                    device_results.append(result)
                raw_results[device] = device_results

            progress.update(execute_task, completed=len(circuits) * len(devices))

        return raw_results

    def _execute_experiment(
        self,
        devices: list[str],
        shots: int,
        backend: str,
        parallel: int,
        experiment_name: str | None,
        no_save: bool,
        no_plot: bool,
        show_plot: bool,
        verbose: bool,
        **kwargs,
    ):
        """
        Common experiment execution logic
        """

        # Display configuration
        config_text = self.create_experiment_config_display(
            devices=devices, backend=backend, shots=shots, parallel=parallel, **kwargs
        )

        self.console.print(
            Panel.fit(
                config_text, title="Experiment Configuration", border_style="blue"
            )
        )

        try:
            # === Step 1: Experiment → Circuit list creation ===
            self.console.print(
                f"\\n🔬 Step 1: Circuit Generation by {self.experiment_name}Experiment"
            )

            experiment_class = self.get_experiment_class()
            experiment_instance = experiment_class(
                experiment_name=experiment_name
                or f"{self.experiment_name.lower()}_{int(time.time())}",
                **kwargs,  # Pass experiment-specific initialization parameters
            )

            # Experiment-specific circuit generation
            circuits, circuit_metadata = self.generate_circuits(
                experiment_instance, **kwargs
            )

            self.console.print(f"   Generated: {len(circuits)} circuits")

            # === Step 2: CLI → Execute in parallel execution environment ===
            backend_enum = ExperimentBackend(backend)
            raw_results = self.run_parallel_execution(
                experiment_instance, circuits, devices, shots, parallel, backend_enum
            )

            # === Step 3: Experiment → Result analysis and save ===
            self.console.print(
                f"\\n📊 Step 3: Analysis & Save by {self.experiment_name}Experiment"
            )

            # Experiment-specific result processing
            results = self.process_results(
                experiment_instance,
                raw_results,
                circuits,
                circuit_metadata,
                device_list=devices,
                **kwargs,
            )

            # Save processing
            if not no_save:
                self.console.print("   → Saving experiment data...")
                experiment_instance.save_complete_experiment_data(results)
            elif not no_plot:
                # Execute only plot generation even without data saving
                self.console.print("   → Generating plot...")
                plot_method = getattr(
                    experiment_instance, f"generate_{self.experiment_name.lower()}_plot"
                )
                plot_method(results, save_plot=True, show_plot=show_plot)

            # Display results
            self.console.print("   → Displaying results...")
            experiment_instance.display_results(results, use_rich=True)

        except KeyboardInterrupt:
            self.console.print("Experiment interrupted by user", style="yellow")
            raise typer.Exit(1) from None
        except Exception as e:
            self.console.print(f"Experiment failed: {e}", style="red")
            if verbose:
                self.console.print_exception()
            raise typer.Exit(1) from None

    @abstractmethod
    def main(self):
        """
        CLI callback - can be overridden in subclasses
        """
        pass

    @abstractmethod
    def run(self):
        """Experiment execution command - implemented in subclasses"""
        pass

    def start(self):
        """CLI execution"""
        self.app()
