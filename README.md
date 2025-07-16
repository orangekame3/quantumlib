# QuantumLib

A modular quantum computing framework for research and experimentation.

## Installation

```bash
pip install git+https://github.com/orangekame3/quantumlib.git
```

## Quick Start

```bash
# CHSH Bell test
quantumlib-chsh run --devices qulacs --shots 1000

# Rabi oscillations
quantumlib-rabi run --devices qulacs --shots 1000

# Other experiments
quantumlib-ramsey run --devices qulacs --shots 1000
quantumlib-t1 run --devices qulacs --shots 1000
quantumlib-t2-echo run --devices qulacs --shots 1000
```

## Usage

### Common Options

All commands support these options:

- `--devices`: Quantum devices to use (default: qulacs)
- `--shots`: Number of measurement shots (default: 1000)
- `--backend`: Experiment backend (default: local_simulator)
- `--parallel`: Number of parallel threads (default: 4)
- `--no-save`: Skip saving data
- `--no-plot`: Skip generating plots
- `--show-plot`: Display plots interactively
- `--verbose`: Verbose output

### CHSH Bell Inequality Test

```bash
quantumlib-chsh run --devices qulacs --shots 1000 --points 20
```

Options:
- `--points`: Number of phase points to scan (default: 20)

### Rabi Oscillation Experiment

```bash
quantumlib-rabi run --devices qulacs --shots 1000 --points 20 --max-amplitude 6.28
```

Options:
- `--points`: Number of amplitude points (default: 20)
- `--max-amplitude`: Maximum drive amplitude in radians (default: 2π)

### Other Experiments

```bash
# Ramsey interference
quantumlib-ramsey run --devices qulacs --shots 1000

# T1 relaxation time
quantumlib-t1 run --devices qulacs --shots 1000

# T2 coherence time
quantumlib-t2-echo run --devices qulacs --shots 1000
```

### Help

Get detailed help for any command:

```bash
quantumlib-chsh --help
quantumlib-rabi --help
```

## Library Usage

You can also use QuantumLib directly in Python code:

### Basic Example

```python
from quantumlib.experiments.chsh.chsh_experiment import CHSHExperiment
from quantumlib.experiments.rabi.rabi_experiment import RabiExperiment

# CHSH Bell inequality test
chsh = CHSHExperiment()
results = chsh.run_experiment(
    devices=['qulacs'],
    shots=1000,
    phase_points=20
)

# Rabi oscillation experiment
rabi = RabiExperiment()
results = rabi.run_experiment(
    devices=['qulacs'],
    shots=1000,
    amplitude_points=20,
    max_amplitude=6.28
)
```

### Circuit Generation

```python
from quantumlib.circuit.chsh_circuits import create_chsh_circuit
from quantumlib.circuit.rabi_circuits import create_rabi_circuit

# Generate CHSH circuit
circuit = create_chsh_circuit(theta_a=0, theta_b=0.785, phi=1.57)

# Generate Rabi circuit
circuit = create_rabi_circuit(amplitude=3.14)
```

### Custom Experiments

```python
from quantumlib.core.base_experiment import BaseExperiment

class MyExperiment(BaseExperiment):
    def create_circuits(self, **params):
        # Your circuit generation logic
        return circuits

    def analyze_results(self, results):
        # Your analysis logic
        return analysis_data

# Run your custom experiment
experiment = MyExperiment()
results = experiment.run_experiment(devices=['qulacs'])
```

## Features

- CHSH Bell inequality experiments
- Rabi oscillation measurements
- Ramsey interference experiments
- T1/T2 coherence time measurements
- Multiple backend support (Qulacs, OQTOPUS)
- Parallel execution and data visualization

## Development

```bash
git clone https://github.com/orangekame3/quantumlib.git
cd quantumlib
uv sync
uv pip install -e .
```

Run tests:
```bash
uv run pytest
```

## Requirements

- Python 3.12+
- Quantum simulators: Qulacs, Qiskit, Cirq
- Scientific computing: NumPy, SciPy, Matplotlib

## License

MIT License
