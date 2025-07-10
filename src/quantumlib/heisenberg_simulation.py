#!/usr/bin/env python3
"""
2-Qubit Heisenberg Model Quantum Simulation

This module implements quantum simulation of the 2-qubit Heisenberg model
using Trotterized time evolution on quantum circuits.

Heisenberg Hamiltonian: H = J(X⊗X + Y⊗Y + Z⊗Z) + h₁Z⊗I + h₂I⊗Z
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, partial_trace, DensityMatrix
import warnings
warnings.filterwarnings('ignore')


class HeisenbergSimulator:
    """
    2-qubit Heisenberg model quantum simulator with Trotter decomposition
    """
    
    def __init__(self, J=1.0, h1=0.0, h2=0.0):
        """
        Initialize Heisenberg simulator
        
        Args:
            J: Exchange coupling strength
            h1: Magnetic field on qubit 1
            h2: Magnetic field on qubit 2
        """
        self.J = J
        self.h1 = h1
        self.h2 = h2
        
    def create_trotter_step(self, dt, order=1):
        """
        Create one Trotter step for time evolution
        
        H = J(XX + YY + ZZ) + h₁Z⊗I + h₂I⊗Z
        
        Trotter decomposition:
        exp(-iHt) ≈ exp(-iH₁t)exp(-iH₂t)exp(-iH₃t)...
        
        Args:
            dt: Time step
            order: Trotter order (1 or 2)
            
        Returns:
            QuantumCircuit: Trotter step circuit
        """
        qc = QuantumCircuit(2)
        
        if order == 1:
            # First order Trotter: exp(-iH₁dt)exp(-iH₂dt)exp(-iH₃dt)...
            self._add_xx_evolution(qc, self.J * dt)
            self._add_yy_evolution(qc, self.J * dt)
            self._add_zz_evolution(qc, self.J * dt)
            self._add_z_field_evolution(qc, self.h1 * dt, self.h2 * dt)
            
        elif order == 2:
            # Second order Trotter (symmetric): exp(-iH₁dt/2)exp(-iH₂dt/2)...exp(-iH₂dt/2)exp(-iH₁dt/2)
            self._add_xx_evolution(qc, self.J * dt / 2)
            self._add_yy_evolution(qc, self.J * dt / 2)
            self._add_zz_evolution(qc, self.J * dt / 2)
            self._add_z_field_evolution(qc, self.h1 * dt / 2, self.h2 * dt / 2)
            
            # Reverse order for symmetric decomposition
            self._add_z_field_evolution(qc, self.h1 * dt / 2, self.h2 * dt / 2)
            self._add_zz_evolution(qc, self.J * dt / 2)
            self._add_yy_evolution(qc, self.J * dt / 2)
            self._add_xx_evolution(qc, self.J * dt / 2)
        
        return qc
    
    def _add_xx_evolution(self, qc, theta):
        """Add exp(-iθ X⊗X) evolution"""
        if abs(theta) < 1e-10:
            return
        
        # XX evolution: exp(-iθ X⊗X)
        # Decomposition using CNOT gates
        qc.ry(np.pi/2, 0)  # X basis rotation
        qc.ry(np.pi/2, 1)
        qc.cx(0, 1)
        qc.rz(2 * theta, 1)  # Phase evolution
        qc.cx(0, 1)
        qc.ry(-np.pi/2, 0)  # Rotate back
        qc.ry(-np.pi/2, 1)
    
    def _add_yy_evolution(self, qc, theta):
        """Add exp(-iθ Y⊗Y) evolution"""
        if abs(theta) < 1e-10:
            return
        
        # YY evolution: exp(-iθ Y⊗Y)
        qc.rx(np.pi/2, 0)   # Y basis rotation
        qc.rx(np.pi/2, 1)
        qc.cx(0, 1)
        qc.rz(2 * theta, 1)  # Phase evolution
        qc.cx(0, 1)
        qc.rx(-np.pi/2, 0)  # Rotate back
        qc.rx(-np.pi/2, 1)
    
    def _add_zz_evolution(self, qc, theta):
        """Add exp(-iθ Z⊗Z) evolution"""
        if abs(theta) < 1e-10:
            return
        
        # ZZ evolution: exp(-iθ Z⊗Z)
        qc.cx(0, 1)
        qc.rz(2 * theta, 1)
        qc.cx(0, 1)
    
    def _add_z_field_evolution(self, qc, theta1, theta2):
        """Add magnetic field evolution exp(-iθ₁Z⊗I - iθ₂I⊗Z)"""
        if abs(theta1) > 1e-10:
            qc.rz(2 * theta1, 0)
        if abs(theta2) > 1e-10:
            qc.rz(2 * theta2, 1)
    
    def time_evolve(self, initial_state, total_time, n_steps, trotter_order=1):
        """
        Perform time evolution using Trotterized quantum simulation
        
        Args:
            initial_state: Initial quantum state (string like '00', '01', etc.)
            total_time: Total evolution time
            n_steps: Number of Trotter steps
            trotter_order: Trotter decomposition order (1 or 2)
            
        Returns:
            tuple: (times, states, circuits)
        """
        dt = total_time / n_steps
        times = np.linspace(0, total_time, n_steps + 1)
        
        # Initialize circuit with initial state
        qc = QuantumCircuit(2)
        self._prepare_initial_state(qc, initial_state)
        
        states = []
        circuits = []
        
        # Initial state
        initial_circuit = qc.copy()
        states.append(self._get_statevector(initial_circuit))
        circuits.append(initial_circuit)
        
        # Time evolution
        for step in range(n_steps):
            # Add one Trotter step
            trotter_step = self.create_trotter_step(dt, trotter_order)
            qc = qc.compose(trotter_step)
            
            # Store state and circuit
            evolved_circuit = qc.copy()
            states.append(self._get_statevector(evolved_circuit))
            circuits.append(evolved_circuit)
        
        return times, states, circuits
    
    def _prepare_initial_state(self, qc, state_str):
        """Prepare initial state from string representation"""
        if state_str == '00':
            pass  # |00⟩ is default
        elif state_str == '01':
            qc.x(1)
        elif state_str == '10':
            qc.x(0)
        elif state_str == '11':
            qc.x(0)
            qc.x(1)
        elif state_str == '+0':  # (|00⟩ + |10⟩)/√2
            qc.h(0)
        elif state_str == '+1':  # (|01⟩ + |11⟩)/√2
            qc.x(1)
            qc.h(0)
        elif state_str == 'bell':  # (|00⟩ + |11⟩)/√2
            qc.h(0)
            qc.cx(0, 1)
        else:
            raise ValueError(f"Unknown initial state: {state_str}")
    
    def _get_statevector(self, qc):
        """Get statevector from circuit"""
        simulator = AerSimulator(method='statevector')
        job = simulator.run(transpile(qc, simulator))
        result = job.result()
        statevector = result.get_statevector()
        return statevector
    
    def calculate_observables(self, states):
        """
        Calculate physical observables from evolved states
        
        Args:
            states: List of Statevector objects
            
        Returns:
            dict: Dictionary of observables vs time
        """
        observables = {
            'magnetization_x': {'qubit1': [], 'qubit2': [], 'total': []},
            'magnetization_y': {'qubit1': [], 'qubit2': [], 'total': []},
            'magnetization_z': {'qubit1': [], 'qubit2': [], 'total': []},
            'correlation_xx': [],
            'correlation_yy': [],
            'correlation_zz': [],
            'entanglement': [],
            'energy': []
        }
        
        # Pauli matrices
        I = np.array([[1, 0], [0, 1]], dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        # Two-qubit operators
        XI = np.kron(X, I)
        IX = np.kron(I, X)
        YI = np.kron(Y, I)
        IY = np.kron(I, Y)
        ZI = np.kron(Z, I)
        IZ = np.kron(I, Z)
        XX = np.kron(X, X)
        YY = np.kron(Y, Y)
        ZZ = np.kron(Z, Z)
        
        # Hamiltonian matrix
        H = self.J * (XX + YY + ZZ) + self.h1 * ZI + self.h2 * IZ
        
        for state in states:
            # Convert to numpy array
            psi = state.data
            
            # Single qubit magnetizations
            mx1 = np.real(np.conj(psi) @ XI @ psi)
            mx2 = np.real(np.conj(psi) @ IX @ psi)
            my1 = np.real(np.conj(psi) @ YI @ psi)
            my2 = np.real(np.conj(psi) @ IY @ psi)
            mz1 = np.real(np.conj(psi) @ ZI @ psi)
            mz2 = np.real(np.conj(psi) @ IZ @ psi)
            
            observables['magnetization_x']['qubit1'].append(mx1)
            observables['magnetization_x']['qubit2'].append(mx2)
            observables['magnetization_x']['total'].append(mx1 + mx2)
            
            observables['magnetization_y']['qubit1'].append(my1)
            observables['magnetization_y']['qubit2'].append(my2)
            observables['magnetization_y']['total'].append(my1 + my2)
            
            observables['magnetization_z']['qubit1'].append(mz1)
            observables['magnetization_z']['qubit2'].append(mz2)
            observables['magnetization_z']['total'].append(mz1 + mz2)
            
            # Spin-spin correlations
            cxx = np.real(np.conj(psi) @ XX @ psi)
            cyy = np.real(np.conj(psi) @ YY @ psi)
            czz = np.real(np.conj(psi) @ ZZ @ psi)
            
            observables['correlation_xx'].append(cxx)
            observables['correlation_yy'].append(cyy)
            observables['correlation_zz'].append(czz)
            
            # Energy expectation value
            energy = np.real(np.conj(psi) @ H @ psi)
            observables['energy'].append(energy)
            
            # Entanglement (von Neumann entropy of reduced density matrix)
            dm = DensityMatrix(state)
            rho1 = partial_trace(dm, [1])  # Trace out qubit 2
            eigenvals = np.linalg.eigvals(rho1.data)
            eigenvals = eigenvals[eigenvals > 1e-12]  # Remove zero eigenvalues
            if len(eigenvals) > 1:
                entropy = -np.sum(eigenvals * np.log2(eigenvals))
            else:
                entropy = 0.0
            observables['entanglement'].append(entropy)
        
        return observables


def plot_heisenberg_evolution(times, observables, title="Heisenberg Model Evolution"):
    """
    Plot the time evolution of Heisenberg model observables
    
    Args:
        times: Time points
        observables: Dictionary of observables
        title: Plot title
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # 1. Magnetizations
    ax = axes[0]
    ax.plot(times, observables['magnetization_x']['total'], 'r-', label='⟨Sₓ_total⟩', linewidth=2)
    ax.plot(times, observables['magnetization_y']['total'], 'g-', label='⟨Sᵧ_total⟩', linewidth=2)
    ax.plot(times, observables['magnetization_z']['total'], 'b-', label='⟨Sᵧ_total⟩', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Total Magnetization')
    ax.set_title('Total Magnetization')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Individual qubit magnetizations (Z)
    ax = axes[1]
    ax.plot(times, observables['magnetization_z']['qubit1'], 'b-', label='⟨Z₁⟩', linewidth=2)
    ax.plot(times, observables['magnetization_z']['qubit2'], 'r--', label='⟨Z₂⟩', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Z Magnetization')
    ax.set_title('Individual Z Magnetizations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Spin-spin correlations
    ax = axes[2]
    ax.plot(times, observables['correlation_xx'], 'r-', label='⟨X₁X₂⟩', linewidth=2)
    ax.plot(times, observables['correlation_yy'], 'g-', label='⟨Y₁Y₂⟩', linewidth=2)
    ax.plot(times, observables['correlation_zz'], 'b-', label='⟨Z₁Z₂⟩', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Spin Correlation')
    ax.set_title('Spin-Spin Correlations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Energy
    ax = axes[3]
    ax.plot(times, observables['energy'], 'k-', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Energy ⟨H⟩')
    ax.set_title('Energy Conservation')
    ax.grid(True, alpha=0.3)
    
    # 5. Entanglement
    ax = axes[4]
    ax.plot(times, observables['entanglement'], 'm-', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Entanglement Entropy')
    ax.set_title('Entanglement Evolution')
    ax.grid(True, alpha=0.3)
    
    # 6. Phase space (example: ⟨X₁⟩ vs ⟨Z₁⟩)
    ax = axes[5]
    ax.plot(observables['magnetization_x']['qubit1'], 
           observables['magnetization_z']['qubit1'], 'b-', linewidth=2, alpha=0.7)
    ax.plot(observables['magnetization_x']['qubit1'][0], 
           observables['magnetization_z']['qubit1'][0], 'go', markersize=8, label='Start')
    ax.plot(observables['magnetization_x']['qubit1'][-1], 
           observables['magnetization_z']['qubit1'][-1], 'ro', markersize=8, label='End')
    ax.set_xlabel('⟨X₁⟩')
    ax.set_ylabel('⟨Z₁⟩')
    ax.set_title('Qubit 1 Phase Space')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def run_heisenberg_demo():
    """Run demonstration of Heisenberg model simulation"""
    
    print("🧲 2-Qubit Heisenberg Model Quantum Simulation")
    print("=" * 60)
    
    # Parameters
    J = 1.0      # Exchange coupling
    h1 = 0.1     # Magnetic field on qubit 1  
    h2 = -0.1    # Magnetic field on qubit 2
    
    total_time = 2 * np.pi
    n_steps = 50
    
    print(f"Parameters:")
    print(f"  Exchange coupling J = {J}")
    print(f"  Magnetic fields: h₁ = {h1}, h₂ = {h2}")
    print(f"  Evolution time: {total_time:.2f}")
    print(f"  Trotter steps: {n_steps}")
    print()
    
    # Initialize simulator
    sim = HeisenbergSimulator(J=J, h1=h1, h2=h2)
    
    # Test different initial states
    initial_states = ['00', '01', '+0', 'bell']
    state_names = ['|00⟩', '|01⟩', '(|00⟩+|10⟩)/√2', 'Bell State']
    
    for state, name in zip(initial_states, state_names):
        print(f"🚀 Evolving from initial state: {name}")
        
        # Perform time evolution
        times, states, circuits = sim.time_evolve(
            initial_state=state,
            total_time=total_time,
            n_steps=n_steps,
            trotter_order=2
        )
        
        # Calculate observables
        observables = sim.calculate_observables(states)
        
        # Plot results
        plot_heisenberg_evolution(times, observables, 
                                 title=f"Heisenberg Evolution: {name}")
        
        # Print some statistics
        energy_drift = np.std(observables['energy'])
        max_entanglement = np.max(observables['entanglement'])
        
        print(f"  Energy conservation: σ(E) = {energy_drift:.6f}")
        print(f"  Maximum entanglement: {max_entanglement:.3f} bits")
        print()
    
    print("✅ Heisenberg simulation demo completed!")
    return sim


if __name__ == "__main__":
    # Run demonstration
    simulator = run_heisenberg_demo()