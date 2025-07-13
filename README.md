# QuantumLib

A modular quantum computing framework optimized for research and experimentation, featuring standardized CLI tools, modular circuit design, and comprehensive experiment management.

## 🚀 Features

- **Modular Experiment Framework**: Standardized base classes for quantum experiments
- **Unified CLI Interface**: Consistent command-line tools with shared options
- **Multi-Backend Support**: Local simulators (Qulacs) and cloud quantum devices (OQTOPUS)
- **Rich Visualization**: Automatic plot generation and interactive displays
- **Data Management**: Structured result storage with comprehensive metadata
- **Parallel Execution**: Efficient multi-threaded circuit execution
- **Jupyter Integration**: Seamless notebook development experience

## 📋 Quick Start

### Installation

```bash
git clone https://github.com/your-username/quantumlib.git
cd quantumlib
uv sync
uv pip install -e .
```

### Run Your First Experiment

```bash
# CHSH Bell Inequality Test
uv run workspace/scripts/chsh.py run --devices qulacs --shots 1000 --points 20

# Rabi Oscillation Experiment
uv run workspace/scripts/rabi.py run --devices qulacs --shots 1000 --points 20
```

## 🛠️ Architecture

### Modular Design

```
quantumlib/
├── cli/              # Common CLI framework
├── circuit/          # Experiment-specific circuit factories
├── experiments/      # Experiment implementations
├── backend/          # Device backend abstractions
└── core/            # Base classes and utilities
```

### Experiment Framework

All experiments inherit from `BaseExperiment` and follow a standardized 3-step pattern:

1. **Circuit Generation**: Create quantum circuits for the experiment
2. **Parallel Execution**: Run circuits across devices with progress tracking
3. **Analysis & Storage**: Process results and save comprehensive data

### CLI Framework

CLIs inherit from `BaseExperimentCLI` providing:

- Shared common options (devices, shots, backend, etc.)
- Rich progress indicators and formatted output
- Automatic data saving and visualization
- Consistent help documentation

## 🔬 Supported Experiments

### CHSH Bell Inequality Verification

Tests quantum non-locality through Bell inequality violations.

```bash
uv run workspace/scripts/chsh.py run \
  --devices qulacs \
  --shots 1000 \
  --points 20 \
  --backend local_simulator
```

**Options:**

- `--points`: Number of phase points to scan (default: 20)

**Output:**

- Bell parameter |S| values vs phase
- Violation count and quantum advantage analysis
- Publication-ready plots

### Rabi Oscillation Analysis

Characterizes qubit drive amplitude vs excitation probability.

```bash
uv run workspace/scripts/rabi.py run \
  --devices qulacs \
  --shots 1000 \
  --points 20 \
  --max-amplitude 6.28
```

**Options:**

- `--points`: Number of amplitude points (default: 20)
- `--max-amplitude`: Maximum drive amplitude in radians (default: 2π)

**Output:**

- Excitation probability vs drive amplitude
- Rabi frequency estimation
- π-pulse amplitude identification

## 🎯 Common CLI Options

All experiment CLIs share these options:

| Option              | Type | Default           | Description                 |
| ------------------- | ---- | ----------------- | --------------------------- |
| `--devices`         | List | `[qulacs]`        | Quantum devices to use      |
| `--shots`           | int  | `1000`            | Number of measurement shots |
| `--backend`         | str  | `local_simulator` | Experiment backend          |
| `--parallel`        | int  | `4`               | Number of parallel threads  |
| `--experiment-name` | str  | `None`            | Custom experiment name      |
| `--no-save`         | flag | `False`           | Skip saving data            |
| `--no-plot`         | flag | `False`           | Skip generating plots       |
| `--show-plot`       | flag | `False`           | Display plots interactively |
| `--verbose`         | flag | `False`           | Verbose output              |

## 📊 Data Management

### Automatic Storage

Each experiment run creates structured output:

```shell
.results/experiment_name_timestamp/
├── data/
│   ├── experiment_results.json      # Raw measurement data
│   ├── device_performance.json      # Device comparison metrics
│   ├── plotting_data.json          # Processed data for plots
│   └── experiment_summary.json     # High-level results
└── plots/
    └── experiment_plot.png          # Publication-ready figures
```

### Metadata Tracking

Every experiment includes comprehensive metadata:

- Execution timestamp and unique ID
- Device configuration and backend used
- Experiment parameters and settings
- Performance metrics and execution time
- Analysis results and key findings

## 🔧 Development

### Adding New Experiments

1. **Create Experiment Class**:

```python
from quantumlib.core.base_experiment import BaseExperiment

class MyExperiment(BaseExperiment):
    def create_circuits(self, **params):
        # Generate quantum circuits
        pass

    def analyze_results(self, results):
        # Process measurement data
        pass
```

2. **Create CLI Interface**:

```python
from quantumlib.cli.base_cli import BaseExperimentCLI

class MyExperimentCLI(BaseExperimentCLI):
    def get_experiment_class(self):
        return MyExperiment

    def run(self,
        devices: CommonDevicesOption = [DeviceType.qulacs],
        shots: CommonShotsOption = 1000,
        # ... other common options
        my_param: Annotated[float, typer.Option()] = 1.0
    ):
        self._execute_experiment(
            devices=[d.value for d in devices],
            shots=shots,
            my_param=my_param,
            # ... other parameters
        )
```

### Circuit Modularity

Circuits are organized by experiment type:

```python
# quantumlib/circuit/my_circuits.py
def create_my_circuit(param1, param2):
    # Circuit creation logic
    return circuit
```

### Testing

```bash
# Run specific experiment tests
uv run pytest tests/test_chsh.py
uv run pytest tests/test_rabi.py

# Full test suite
uv run pytest
```

## 🚀 Advanced Usage

### Custom Backend Configuration

```python
from quantumlib.backend.oqtopus import OQTOPUSBackend

# Configure OQTOPUS for real device execution
backend = OQTOPUSBackend(
    device_name="SC",
    optimization_level=2
)
```

### Batch Experiments

```python
from quantumlib.experiments.chsh.chsh_experiment import CHSHExperiment

# Run parameter sweep
experiment = CHSHExperiment()
for phase_points in [10, 20, 50]:
    results = experiment.run_full_experiment(
        devices=['qulacs'],
        phase_points=phase_points
    )
```

### Jupyter Integration

```python
# In Jupyter notebook
from quantumlib.experiments.rabi.rabi_experiment import RabiExperiment

experiment = RabiExperiment()
results = experiment.run_full_experiment(devices=['qulacs'])
experiment.create_rich_visualization(results)
```

## 📈 Performance Features

- **Parallel Circuit Execution**: Automatic multi-threading for circuit batches
- **Progress Tracking**: Real-time progress bars with ETA
- **Memory Optimization**: Efficient handling of large measurement datasets
- **Caching**: Intelligent caching of expensive computations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes following the modular architecture
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

Built with:

- [Qulacs](https://github.com/qulacs/qulacs) for quantum simulation
- [OQTOPUS](https://github.com/oqtopus-team/oqtopus) for cloud quantum access
- [Typer](https://typer.tiangolo.com/) for CLI framework
- [Rich](https://rich.readthedocs.io/) for beautiful terminal output
