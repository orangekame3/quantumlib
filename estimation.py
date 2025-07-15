import numpy as np
from quri_parts.circuit import QuantumCircuit
from quri_parts.core.operator import Operator, pauli_label
from quri_parts_oqtopus.backend import OqtopusEstimationBackend

exps = []
number_of_phases = 21
phases = np.linspace(0, 2 * np.pi, number_of_phases)
for theta in phases:
    # Create a simple quantum circuit
    circuit = QuantumCircuit(2)
    circuit.add_H_gate(0)
    circuit.add_CNOT_gate(0, 1)
    circuit.add_RY_gate(0, theta)

    operator = Operator(
        {
            pauli_label("Z0 Z1"): 1,
            pauli_label("Z0 X1"): -1,
            pauli_label("X0 Z1"): 1,
            pauli_label("X0 X1"): 1,
        }
    )

    mitigation_info = {
        "ro_error_mitigation": "pseudo_inverse",
    }
    transpiler_info = {
        "transpiler_lib": "qiskit",
        "transpiler_options": {
            "optimization_level": 1,
        },
    }
    backend = OqtopusEstimationBackend()

    job = backend.estimate(
        circuit,
        operator=operator,
        device_id="anemone",
        shots=1024,
        mitigation_info=mitigation_info,
        transpiler_info=transpiler_info,
    )
    # print(job)
    result = job.result()
    print(result.exp_value)
    exps.append(result.exp_value)
