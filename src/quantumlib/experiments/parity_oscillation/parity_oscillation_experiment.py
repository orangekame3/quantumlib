#!/usr/bin/env python3
"""
Parity Oscillation Experiment Class - Decoherence study of GHZ states
Based on "Decoherence of up to 8-qubit entangled states in a 16-qubit superconducting quantum processor"
by Asier Ozaeta and Peter L McMahon (2019)
"""

import time
from typing import Any

import numpy as np

from ...circuit.common_circuits import create_ghz_state
from ...core.base_experiment import BaseExperiment

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UGate
    
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


class ParityOscillationExperiment(BaseExperiment):
    """
    Parity oscillation experiment for studying GHZ state decoherence
    
    Measures the coherence C(N, τ) of N-qubit GHZ states as a function of:
    - Number of qubits N
    - Delay time τ
    - Rotation phase φ
    
    The coherence is extracted from parity oscillations amplitude.
    """

    def __init__(self, experiment_name: str = None, **kwargs):
        # Extract parity oscillation experiment-specific parameters (not passed to BaseExperiment)
        parity_specific_params = {
            "num_qubits_list", "delays_us", "phase_points"
        }
        
        # Filter kwargs to pass to BaseExperiment
        base_kwargs = {k: v for k, v in kwargs.items() if k not in parity_specific_params}
        super().__init__(experiment_name, **base_kwargs)
        
        # Parity oscillation experiment parameters
        self.default_num_qubits = [1, 2, 3, 4, 5]
        self.default_delays_us = [0, 1, 2, 4, 8, 16]  # delay times in microseconds
        self.default_phase_points = 21  # φ from 0 to π (4N+1 points as in paper)
        
        print(f"Parity Oscillation Experiment initialized")
        print(f"Default qubit counts: {self.default_num_qubits}")
        print(f"Default delays (μs): {self.default_delays_us}")

    def create_ghz_with_delay_and_rotation(
        self, 
        num_qubits: int, 
        delay_us: float = 0.0, 
        phi: float = 0.0
    ) -> Any:
        """
        Create GHZ state circuit with delay and rotation analysis
        
        Circuit structure:
        1. Generate N-qubit GHZ state
        2. Apply delay τ (using identity gates)
        3. Apply rotation U(φ) to each qubit
        4. Measure in computational basis
        
        Args:
            num_qubits: Number of qubits in GHZ state
            delay_us: Delay time in microseconds
            phi: Rotation phase φ
            
        Returns:
            Quantum circuit
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for parity oscillation circuits")
        
        # Create base GHZ state (without measurement)
        qc = QuantumCircuit(num_qubits, num_qubits)
        
        # Step 1: Generate GHZ state |0...0⟩ + |1...1⟩
        qc.h(0)  # Hadamard on first qubit
        for i in range(1, num_qubits):
            qc.cx(0, i)  # CNOT chain
        
        # Step 2: Apply delay using identity gates
        # Each identity gate = 80ns + 10ns buffer = 90ns (as in paper)
        if delay_us > 0:
            num_identity_gates = int(delay_us * 1000 / 90)  # Convert μs to 90ns units
            for qubit in range(num_qubits):
                for _ in range(num_identity_gates):
                    qc.id(qubit)
        
        # Step 3: Apply rotation U(φ) to each qubit
        # U(φ) = exp(-iφσy/2) rotation around Y-axis
        if phi != 0:
            for qubit in range(num_qubits):
                # Using U gate: U(θ, φ, λ) where θ=π/2, φ=-φ-π/2, λ=-φ-π/2
                # This implements the rotation from the paper
                qc.u(np.pi/2, -phi - np.pi/2, -phi - np.pi/2, qubit)
        
        # Step 4: Measurement
        qc.measure_all()
        
        return qc

    def create_circuits(self, **kwargs) -> list[Any]:
        """
        Create parity oscillation experiment circuits
        
        Args:
            num_qubits_list: List of qubit counts to test (default: [1,2,3,4,5])
            delays_us: List of delay times in μs (default: [0,1,2,4,8,16])
            phase_points: Number of phase points from 0 to π (default: 21)
            
        Returns:
            List of quantum circuits
        """
        num_qubits_list = kwargs.get('num_qubits_list', self.default_num_qubits)
        delays_us = kwargs.get('delays_us', self.default_delays_us)
        phase_points = kwargs.get('phase_points', self.default_phase_points)
        
        circuits = []
        circuit_metadata = []
        
        for num_qubits in num_qubits_list:
            # Generate phase points: 4N+1 points from 0 to π (as in paper)
            actual_phase_points = 4 * num_qubits + 1
            phase_values = np.linspace(0, np.pi, actual_phase_points)
            
            for delay_us in delays_us:
                for phi in phase_values:
                    circuit = self.create_ghz_with_delay_and_rotation(
                        num_qubits, delay_us, phi
                    )
                    circuits.append(circuit)
                    
                    # Store metadata for analysis
                    circuit_metadata.append({
                        'num_qubits': num_qubits,
                        'delay_us': delay_us,
                        'phi': phi,
                        'circuit_index': len(circuits) - 1
                    })
        
        # Store metadata for later analysis
        self.circuit_metadata = circuit_metadata
        
        print(f"Generated {len(circuits)} parity oscillation circuits")
        print(f"Qubit counts: {num_qubits_list}")
        print(f"Delays: {delays_us} μs")
        print(f"Phase points per (N,τ): {actual_phase_points} for max N={max(num_qubits_list)}")
        
        return circuits

    def calculate_parity(self, counts: dict[str, int]) -> float:
        """
        Calculate parity P_even - P_odd from measurement counts
        
        Args:
            counts: Measurement counts dictionary
            
        Returns:
            Parity value P_even - P_odd
        """
        total_shots = sum(counts.values())
        if total_shots == 0:
            return 0.0
        
        even_count = 0
        odd_count = 0
        
        for bitstring, count in counts.items():
            # Count number of 1s in bitstring
            num_ones = bitstring.count('1')
            if num_ones % 2 == 0:
                even_count += count
            else:
                odd_count += count
        
        p_even = even_count / total_shots
        p_odd = odd_count / total_shots
        
        return p_even - p_odd

    def fit_sinusoid(self, phase_values: np.ndarray, parity_values: np.ndarray) -> dict:
        """
        Fit sinusoid to parity oscillations: P(φ) = A*sin(Nφ + θ) + offset
        
        Args:
            phase_values: Array of phase values φ
            parity_values: Array of corresponding parity values
            
        Returns:
            Dictionary with fit parameters: amplitude, phase, offset, frequency
        """
        try:
            from scipy.optimize import curve_fit
        except ImportError:
            print("Warning: scipy not available, using simple amplitude estimation")
            # Simple amplitude estimation: max - min
            amplitude = (np.max(parity_values) - np.min(parity_values)) / 2
            return {
                'amplitude': amplitude,
                'phase': 0.0,
                'offset': np.mean(parity_values),
                'frequency': 1.0,
                'r_squared': 0.0,
                'fit_success': False
            }
        
        # Determine frequency from number of qubits
        # For N-qubit GHZ state, frequency should be N
        num_qubits = len(phase_values) // 4 - 1  # Approximate from 4N+1 points
        if num_qubits < 1:
            num_qubits = 1
        
        def sinusoid(phi, amplitude, phase, offset):
            return amplitude * np.sin(num_qubits * phi + phase) + offset
        
        try:
            # Initial guess
            amplitude_guess = (np.max(parity_values) - np.min(parity_values)) / 2
            offset_guess = np.mean(parity_values)
            phase_guess = 0.0
            
            popt, pcov = curve_fit(
                sinusoid, 
                phase_values, 
                parity_values,
                p0=[amplitude_guess, phase_guess, offset_guess],
                maxfev=2000
            )
            
            amplitude, phase, offset = popt
            
            # Calculate R²
            y_pred = sinusoid(phase_values, *popt)
            ss_res = np.sum((parity_values - y_pred) ** 2)
            ss_tot = np.sum((parity_values - np.mean(parity_values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return {
                'amplitude': abs(amplitude),  # Coherence C(N,τ)
                'phase': phase,
                'offset': offset,
                'frequency': num_qubits,
                'r_squared': r_squared,
                'fit_success': True
            }
            
        except Exception as e:
            print(f"Sinusoid fitting failed: {e}")
            amplitude = (np.max(parity_values) - np.min(parity_values)) / 2
            return {
                'amplitude': amplitude,
                'phase': 0.0,
                'offset': np.mean(parity_values),
                'frequency': num_qubits,
                'r_squared': 0.0,
                'fit_success': False
            }

    def analyze_results(
        self, 
        results: dict[str, list[dict[str, Any]]], 
        **kwargs
    ) -> dict[str, Any]:
        """
        Analyze parity oscillation results
        
        Args:
            results: Raw measurement results from quantum devices
            
        Returns:
            Analysis results with coherence data
        """
        if not hasattr(self, 'circuit_metadata'):
            raise ValueError("Circuit metadata not found. Run create_circuits first.")
        
        analysis_results = {}
        
        for device, device_results in results.items():
            print(f"\nAnalyzing {device} results...")
            
            device_analysis = {
                'coherence_data': [],
                'parity_oscillations': [],
                'fit_parameters': []
            }
            
            # Group results by (num_qubits, delay_us)
            grouped_results = {}
            
            for i, result in enumerate(device_results):
                if result is None or not result.get('success', False):
                    continue
                
                metadata = self.circuit_metadata[i]
                key = (metadata['num_qubits'], metadata['delay_us'])
                
                if key not in grouped_results:
                    grouped_results[key] = {
                        'phase_values': [],
                        'parity_values': [],
                        'counts_list': []
                    }
                
                # Calculate parity for this measurement
                counts = result.get('counts', {})
                parity = self.calculate_parity(counts)
                
                grouped_results[key]['phase_values'].append(metadata['phi'])
                grouped_results[key]['parity_values'].append(parity)
                grouped_results[key]['counts_list'].append(counts)
            
            # Analyze each (N, τ) combination
            for (num_qubits, delay_us), data in grouped_results.items():
                if len(data['phase_values']) < 5:  # Need sufficient points
                    continue
                
                # Sort by phase for proper fitting
                sorted_indices = np.argsort(data['phase_values'])
                phase_array = np.array(data['phase_values'])[sorted_indices]
                parity_array = np.array(data['parity_values'])[sorted_indices]
                
                # Fit sinusoid to extract coherence
                fit_result = self.fit_sinusoid(phase_array, parity_array)
                coherence = fit_result['amplitude']
                
                coherence_data = {
                    'num_qubits': num_qubits,
                    'delay_us': delay_us,
                    'coherence': coherence,
                    'fit_r_squared': fit_result['r_squared'],
                    'fit_success': fit_result['fit_success']
                }
                
                oscillation_data = {
                    'num_qubits': num_qubits,
                    'delay_us': delay_us,
                    'phase_values': phase_array.tolist(),
                    'parity_values': parity_array.tolist(),
                    'fit_parameters': fit_result
                }
                
                device_analysis['coherence_data'].append(coherence_data)
                device_analysis['parity_oscillations'].append(oscillation_data)
                
                print(f"N={num_qubits}, τ={delay_us}μs: C={coherence:.3f}, R²={fit_result['r_squared']:.3f}")
            
            analysis_results[device] = device_analysis
        
        return analysis_results

    def save_experiment_data(
        self, 
        results: dict[str, Any], 
        metadata: dict[str, Any] = None
    ) -> str:
        """
        Save parity oscillation experiment data
        
        Args:
            results: Analyzed results
            metadata: Additional metadata
            
        Returns:
            Save path
        """
        save_data = {
            'experiment_type': 'ParityOscillation',
            'results': results,
            'parameters': {
                'num_qubits_list': getattr(self, 'default_num_qubits', []),
                'delays_us': getattr(self, 'default_delays_us', []),
                'phase_points': getattr(self, 'default_phase_points', 21)
            },
            'analysis_timestamp': time.time(),
            'metadata': metadata or {}
        }
        
        return self.data_manager.save_data(save_data, 'parity_oscillation_results')

    def generate_parity_plot(
        self, results: dict[str, Any], save_plot: bool = True, show_plot: bool = False
    ) -> str | None:
        """
        Generate parity oscillation experiment plot following quantumlib standards
        
        Args:
            results: Complete experiment results
            save_plot: Save plot to file
            show_plot: Display plot interactively
            
        Returns:
            Plot file path if saved, None otherwise
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available - skipping plot generation")
            return None

        analysis = results.get("analysis", {})
        if not analysis:
            print("No analysis results for plotting")
            return None

        # Create figure with subplots for different delay times
        device_data = list(analysis.values())[0]  # Get first device data
        oscillation_data = device_data.get("parity_oscillations", [])
        
        if not oscillation_data:
            print("No parity oscillation data for plotting")
            return None

        # Group by delay time
        delay_groups = {}
        for data in oscillation_data:
            delay = data['delay_us']
            if delay not in delay_groups:
                delay_groups[delay] = []
            delay_groups[delay].append(data)

        n_delays = len(delay_groups)
        if n_delays == 0:
            return None

        # Create subplots
        fig, axes = plt.subplots(
            nrows=(n_delays + 1) // 2, 
            ncols=2, 
            figsize=(14, 4 * ((n_delays + 1) // 2))
        )
        
        if n_delays == 1:
            axes = [axes]
        elif n_delays <= 2:
            axes = axes.flatten() if n_delays == 2 else [axes]
        else:
            axes = axes.flatten()

        # Colors for different qubit counts (matching existing style)
        colors = ["blue", "red", "green", "orange", "purple"]

        for i, (delay_us, delay_data) in enumerate(delay_groups.items()):
            ax = axes[i] if n_delays > 1 else axes[0]
            
            for j, data in enumerate(delay_data):
                phase_values = np.array(data['phase_values'])
                parity_values = np.array(data['parity_values'])
                num_qubits = data['num_qubits']
                
                color = colors[j % len(colors)]
                ax.plot(
                    phase_values, 
                    parity_values, 
                    'o-', 
                    color=color, 
                    label=f'N={num_qubits}',
                    linewidth=2,
                    markersize=6,
                    alpha=0.8
                )
                
                # Plot fit if available
                fit_params = data.get('fit_parameters', {})
                if fit_params.get('fit_success', False):
                    phi_fit = np.linspace(0, np.pi, 100)
                    amplitude = fit_params['amplitude']
                    phase = fit_params['phase']
                    offset = fit_params['offset']
                    freq = fit_params['frequency']
                    
                    parity_fit = amplitude * np.sin(freq * phi_fit + phase) + offset
                    ax.plot(phi_fit, parity_fit, '--', color=color, alpha=0.6, linewidth=2)
            
            # Formatting (following existing style)
            ax.set_xlabel('Phase φ [rad]', fontsize=12)
            ax.set_ylabel('Parity (P_even - P_odd)', fontsize=12)
            ax.set_title(f'Parity Oscillations (τ = {delay_us} μs)', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            ax.set_xlim(0, np.pi)
            
            # X-axis labels in π units (following existing style)
            ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
            ax.set_xticklabels(['0', 'π/4', 'π/2', '3π/4', 'π'])

        # Hide unused subplots
        for i in range(n_delays, len(axes)):
            axes[i].set_visible(False)

        # Main title
        fig.suptitle('QuantumLib Parity Oscillation (GHZ Decoherence) Experiment', 
                    fontsize=16, fontweight='bold')

        plot_filename = None
        if save_plot:
            plt.tight_layout()
            plot_filename = f"parity_oscillation_plot_{self.experiment_name}_{int(time.time())}.png"

            # Save to experiment results directory (following existing pattern)
            if hasattr(self, "data_manager") and hasattr(self.data_manager, "session_dir"):
                plot_path = f"{self.data_manager.session_dir}/plots/{plot_filename}"
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                print(f"Plot saved: {plot_path}")
                plot_filename = plot_path  # Return full path
            else:
                # Fallback: save in current directory
                plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
                print(f"⚠️ Plot saved to current directory: {plot_filename}")
                print("   (data_manager not available)")

        # Display plot
        if show_plot:
            try:
                plt.show()
            except Exception:
                pass

        plt.close()
        return plot_filename

    def create_plots(self, analysis_results: dict[str, Any], save_dir: str = None) -> None:
        """
        Create plots for parity oscillation experiment results (legacy method)
        
        Args:
            analysis_results: Results from analyze_results()
            save_dir: Directory to save plots (optional)
        """
        try:
            # Use the unified plot generation method
            self.generate_parity_plot({"analysis": analysis_results}, save_plot=True, show_plot=False)
        except ImportError:
            print("Cannot create plots: matplotlib not available")
        except Exception as e:
            print(f"Plot creation failed: {e}")

    def save_complete_experiment_data(self, results: dict[str, Any]) -> str:
        """
        Save complete experiment data including plots and summary
        Required by the unified CLI framework
        
        Args:
            results: Complete experiment results
            
        Returns:
            Path to main results file
        """
        # Save main experiment data
        main_file = self.save_experiment_data(results["analysis"])
        
        # Generate and save plots using unified method
        plot_file = self.generate_parity_plot(results, save_plot=True, show_plot=False)
        
        # Create experiment summary
        summary = self._create_experiment_summary(results)
        summary_file = self.data_manager.save_data(summary, "experiment_summary")
        
        print("📊 Complete parity oscillation experiment data saved:")
        print(f"  • Main results: {main_file}")
        print(f"  • Plots: {plot_file if plot_file else 'Not generated'}")
        print(f"  • Summary: {summary_file}")
        
        return main_file

    def _create_experiment_summary(self, results: dict[str, Any]) -> dict:
        """
        Create experiment summary for unified framework
        
        Args:
            results: Complete experiment results
            
        Returns:
            Summary dictionary
        """
        analysis = results.get("analysis", {})
        
        # Count successful measurements
        total_measurements = 0
        successful_measurements = 0
        
        for device, device_results in analysis.items():
            coherence_data = device_results.get("coherence_data", [])
            total_measurements += len(coherence_data)
            successful_measurements += sum(1 for c in coherence_data if c.get("fit_success", False))
        
        # Extract key findings
        coherence_summary = {}
        for device, device_results in analysis.items():
            coherence_data = device_results.get("coherence_data", [])
            if coherence_data:
                coherence_summary[device] = {
                    "measurements": len(coherence_data),
                    "successful_fits": sum(1 for c in coherence_data if c.get("fit_success", False)),
                    "qubit_counts": sorted(list(set(c["num_qubits"] for c in coherence_data))),
                    "max_coherence": max(c["coherence"] for c in coherence_data),
                    "min_coherence": min(c["coherence"] for c in coherence_data),
                }
        
        return {
            "experiment_type": "ParityOscillation",
            "timestamp": time.time(),
            "parameters": getattr(self, "experiment_params", {}),
            "total_measurements": total_measurements,
            "successful_measurements": successful_measurements,
            "success_rate": successful_measurements / total_measurements if total_measurements > 0 else 0,
            "devices": list(analysis.keys()),
            "coherence_summary": coherence_summary,
            "key_findings": self._extract_key_findings(analysis),
        }

    def _extract_key_findings(self, analysis: dict[str, Any]) -> dict:
        """
        Extract key scientific findings from the analysis
        
        Args:
            analysis: Analysis results
            
        Returns:
            Key findings dictionary
        """
        findings = {}
        
        for device, device_results in analysis.items():
            coherence_data = device_results.get("coherence_data", [])
            if not coherence_data:
                continue
            
            # Group by qubit count to analyze scaling
            qubit_groups = {}
            for data in coherence_data:
                n = data["num_qubits"]
                if n not in qubit_groups:
                    qubit_groups[n] = []
                qubit_groups[n].append(data)
            
            # Analyze initial coherence scaling
            initial_coherences = {}
            for n, group in qubit_groups.items():
                # Find minimum delay (closest to τ=0)
                min_delay_data = min(group, key=lambda x: x["delay_us"])
                initial_coherences[n] = min_delay_data["coherence"]
            
            if len(initial_coherences) >= 2:
                # Simple linear fit slope estimation
                qubits = list(initial_coherences.keys())
                coherences = list(initial_coherences.values())
                
                if len(qubits) >= 2:
                    import numpy as np
                    slope = np.polyfit(qubits, coherences, 1)[0]
                    
                    findings[device] = {
                        "initial_coherence_scaling": {
                            "slope": slope,
                            "interpretation": "Linear decrease" if slope < -0.05 else "Approximately constant"
                        },
                        "coherence_range": {
                            "max": max(coherences),
                            "min": min(coherences)
                        },
                        "tested_qubit_counts": qubits
                    }
        
        return findings

    def display_results(self, results: dict[str, Any], use_rich: bool = True) -> None:
        """
        Display parity oscillation experiment results in formatted table
        
        Args:
            results: Complete experiment results
            use_rich: Use rich formatting if available
        """
        analysis = results.get("analysis", {})
        
        if not analysis:
            print("No analysis results found")
            return

        if use_rich:
            try:
                from rich.console import Console
                from rich.table import Table

                console = Console()
                table = Table(
                    title="Parity Oscillation (GHZ Decoherence) Results",
                    show_header=True,
                    header_style="bold blue",
                )
                table.add_column("Device", style="cyan")
                table.add_column("N (Qubits)", justify="right")
                table.add_column("τ (μs)", justify="right") 
                table.add_column("C(N,τ)", justify="right")
                table.add_column("R²", justify="right")
                table.add_column("Fit Success", justify="center")

                for device, device_results in analysis.items():
                    coherence_data = device_results.get("coherence_data", [])
                    
                    if not coherence_data:
                        table.add_row(device, "—", "—", "—", "—", "❌")
                        continue
                    
                    # Sort by qubit count, then by delay
                    sorted_data = sorted(coherence_data, key=lambda x: (x["num_qubits"], x["delay_us"]))
                    
                    for i, data in enumerate(sorted_data):
                        device_name = device if i == 0 else ""
                        
                        n_qubits = str(data["num_qubits"])
                        delay = f"{data['delay_us']:.1f}"
                        coherence = f"{data['coherence']:.3f}"
                        r_squared = f"{data['fit_r_squared']:.3f}"
                        fit_success = "✓" if data.get("fit_success", False) else "❌"
                        
                        table.add_row(device_name, n_qubits, delay, coherence, r_squared, fit_success)

                console.print(table)
                
                # Additional summary
                for device, device_results in analysis.items():
                    coherence_data = device_results.get("coherence_data", [])
                    if coherence_data:
                        successful_fits = sum(1 for c in coherence_data if c.get("fit_success", False))
                        total_measurements = len(coherence_data)
                        success_rate = successful_fits / total_measurements * 100
                        
                        console.print(f"\n[bold cyan]{device}[/bold cyan] Summary:")
                        console.print(f"  • Measurements: {total_measurements}")
                        console.print(f"  • Successful fits: {successful_fits} ({success_rate:.1f}%)")
                        
                        # Show coherence range
                        coherences = [c["coherence"] for c in coherence_data if c.get("fit_success", False)]
                        if coherences:
                            console.print(f"  • Coherence range: {min(coherences):.3f} - {max(coherences):.3f}")

            except ImportError:
                use_rich = False

        if not use_rich:
            # Fallback to plain text
            print("=== Parity Oscillation (GHZ Decoherence) Results ===")
            
            for device, device_results in analysis.items():
                print(f"\n{device} Results:")
                print("N\tτ(μs)\tC(N,τ)\tR²\tFit")
                print("-" * 40)
                
                coherence_data = device_results.get("coherence_data", [])
                if not coherence_data:
                    print("No coherence data available")
                    continue
                
                # Sort by qubit count, then by delay
                sorted_data = sorted(coherence_data, key=lambda x: (x["num_qubits"], x["delay_us"]))
                
                for data in sorted_data:
                    fit_symbol = "✓" if data.get("fit_success", False) else "✗"
                    print(
                        f"{data['num_qubits']}\t"
                        f"{data['delay_us']:.1f}\t"
                        f"{data['coherence']:.3f}\t"
                        f"{data['fit_r_squared']:.3f}\t"
                        f"{fit_symbol}"
                    )
                
                # Summary
                successful_fits = sum(1 for c in coherence_data if c.get("fit_success", False))
                total_measurements = len(coherence_data)
                success_rate = successful_fits / total_measurements * 100
                
                print(f"\nSummary:")
                print(f"  Measurements: {total_measurements}")
                print(f"  Successful fits: {successful_fits} ({success_rate:.1f}%)")
                
                coherences = [c["coherence"] for c in coherence_data if c.get("fit_success", False)]
                if coherences:
                    print(f"  Coherence range: {min(coherences):.3f} - {max(coherences):.3f}")