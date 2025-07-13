#!/usr/bin/env python3
"""
T1 Demonstration Script - Shows T1 decay with noise simulation
This demonstrates the T1 experiment concept using your provided code as a reference
"""
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.qasm3 import dumps
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error

# T1, T2 設定（500ns）
t1 = 500  # [ns]
t2 = 500
shots = 1000
tau_times = np.array(
    [1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
)

p1_values = []

print(f"Running T1 decay simulation with T1={t1}ns, T2={t2}ns")
print("Delay times (ns):", tau_times)

for tau in tau_times:
    # tauに合わせてノイズモデルを構築
    error = thermal_relaxation_error(t1, t2, tau)
    noise_model = NoiseModel()
    noise_model.add_quantum_error(error, "delay", [0])

    # 回路定義
    qc = QuantumCircuit(1, 1)
    qc.x(0)
    qc.delay(int(tau), 0, unit="ns")
    qc.measure(0, 0)

    # シミュレーション
    backend = AerSimulator(noise_model=noise_model)
    result = backend.run(qc, shots=shots).result()
    counts = result.get_counts()
    p1 = counts.get("1", 0) / shots
    p1_values.append(p1)
    print(f"τ={tau:4.0f}ns: P(1)={p1:.3f}")

# Theoretical decay curve
theoretical_p1 = np.exp(-tau_times / t1)

# プロット（logスケール x軸）
plt.figure(figsize=(10, 6))
plt.semilogx(tau_times, p1_values, "o-", label=f"Simulated P(1) (T1={t1}ns)", markersize=6, linewidth=2)
plt.semilogx(tau_times, theoretical_p1, "r--", label=f"Theory: exp(-t/{t1}ns)", linewidth=2)
plt.xlabel("Delay time τ [ns] (log scale)")
plt.ylabel("P(1)")
plt.title(f"T1 decay simulation (T1 = {t1} ns)")
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.legend()
plt.ylim(0, 1.1)
plt.tight_layout()

# Save plot
plot_file = "t1_decay_demo.png"
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"\nPlot saved as: {plot_file}")

# Display QASM for one circuit example
print(f"\nExample QASM3 circuit for τ={tau_times[8]}ns:")
example_qc = QuantumCircuit(1, 1)
example_qc.x(0)
example_qc.delay(int(tau_times[8]), 0, unit="ns")
example_qc.measure(0, 0)
print(dumps(example_qc))

# Extract delay instruction details like in your code
print("\nDelay instruction details:")
for instruction in example_qc.data:
    if instruction.operation.name == "delay":
        print(f"  Instruction: {instruction}")
        print(f"  Delay duration: {instruction.params[0]} ns")

plt.show()