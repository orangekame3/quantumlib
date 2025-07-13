#!/usr/bin/env python3
"""
Research CLI - Advanced CHSH research experiments
研究用の高度なCHSH実験CLI（typerベース）
"""

import sys
from pathlib import Path
from typing import List, Optional, Annotated
from enum import Enum
import time

import typer
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, track
from rich.panel import Panel
from rich.tree import Tree
from rich import print as rprint

# Add library paths
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

from src.quantumlib import CHSHExperiment


class ResearchMode(str, Enum):
    bell_study = "bell-study"
    angle_sensitivity = "angle-sensitivity"
    noise_robustness = "noise-robustness"
    theoretical_comparison = "theoretical-comparison"
    comprehensive = "comprehensive"


class DeviceType(str, Enum):
    qulacs = "qulacs"
    anemone = "anemone"


# Initialize console and app
console = Console()
app = typer.Typer(
    name="research-cli",
    help="🔬 Advanced CHSH research experiments for quantum advantage studies",
    rich_markup_mode="rich"
)


class CHSHResearchCLI:
    """Research CLI wrapper for CHSH experiments"""
    
    def __init__(self, research_topic: str, verbose: bool = False):
        self.research_topic = research_topic
        self.verbose = verbose
        self.experiment_log = []
        
        # Create base experiment
        self.base_exp = CHSHExperiment(f"{research_topic}_{int(time.time())}")
        self.setup_research_configuration()
        
        if verbose:
            console.print(f"🔬 Research initialized: {research_topic}")
            console.print(f"📝 Experiment ID: {self.base_exp.experiment_name}")
    
    def setup_research_configuration(self):
        """Configure for research-grade experiments"""
        self.base_exp.transpiler_options.update({
            "optimization_level": 3,
            "routing_method": "sabre",
            "layout_method": "dense",
            "approximation_degree": 0.99
        })
        
        self.base_exp.mitigation_options.update({
            "ro_error_mitigation": "least_squares",
            "zne_noise_factors": [1, 2, 3],
            "extrapolation_method": "linear"
        })
    
    def run_bell_study(self, devices: List[str], shots: int = 2000, points: int = 50):
        """High-resolution Bell inequality violation study"""
        console.print(Panel.fit(
            f"📊 Bell Inequality Violation Study\n"
            f"📱 Devices: {', '.join(devices)}\n"
            f"🎲 Shots: {shots:,} | 📊 Points: {points}",
            title="🔬 Research Configuration",
            border_style="blue"
        ))
        
        results = self.base_exp.run_phase_scan(
            devices=devices,
            phase_points=points,
            theta_a=0,
            theta_b=np.pi/4,
            shots=shots
        )
        
        self.log_experiment("bell_inequality_study", results)
        return results
    
    def run_angle_sensitivity(self, devices: List[str], shots: int = 1500):
        """Comprehensive angle sensitivity analysis"""
        console.print(Panel.fit(
            "📐 Angle Sensitivity Analysis\n"
            "Systematic study of θₐ and θᵦ parameter space",
            title="🔬 Research Configuration",
            border_style="green"
        ))
        
        # Generate angle grid
        theta_a_range = np.linspace(0, np.pi/2, 6)
        theta_b_range = np.linspace(0, np.pi/2, 6)
        
        angle_pairs = []
        for ta in theta_a_range:
            for tb in theta_b_range:
                angle_pairs.append((ta, tb))
        
        # Sample subset for efficiency
        selected_pairs = angle_pairs[::3]  # Every 3rd pair
        
        if self.verbose:
            console.print(f"🔍 Testing {len(selected_pairs)} angle combinations")
        
        results = self.base_exp.run_angle_comparison(
            devices=devices,
            angle_pairs=selected_pairs,
            shots=shots
        )
        
        self.log_experiment("angle_sensitivity_analysis", results)
        return results
    
    def run_noise_robustness(self, devices: List[str], base_shots: int = 1000):
        """Noise robustness analysis with varying shot counts"""
        console.print(Panel.fit(
            "🔊 Noise Robustness Analysis\n"
            "Statistical robustness across different measurement counts",
            title="🔬 Research Configuration",
            border_style="yellow"
        ))
        
        shot_counts = [100, 300, 500, 1000, 2000, 3000]
        results_by_shots = {}
        
        for shots in track(shot_counts, description="Testing shot counts..."):
            console.print(f"🎯 Running with {shots:,} shots")
            
            result = self.base_exp.run_phase_scan(
                devices=devices,
                phase_points=12,  # Moderate resolution
                theta_a=0,
                theta_b=np.pi/4,
                shots=shots
            )
            
            results_by_shots[shots] = result
            
            # Display quick stats
            if 'analyzed_results' in result:
                for device, analysis in result['analyzed_results']['device_results'].items():
                    max_s = analysis['statistics']['max_S_magnitude']
                    violations = analysis['statistics']['bell_violations']
                    console.print(f"   📊 {device}: |S|={max_s:.3f}, violations={violations}")
        
        self.log_experiment("noise_robustness_test", results_by_shots)
        return results_by_shots
    
    def run_theoretical_comparison(self, devices: List[str], shots: int = 1500):
        """Detailed comparison with theoretical predictions"""
        console.print(Panel.fit(
            "📈 Theoretical Comparison Study\n"
            "Precision comparison with quantum mechanical predictions",
            title="🔬 Research Configuration", 
            border_style="purple"
        ))
        
        # Theoretically interesting phases
        theoretical_phases = [0, np.pi/8, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2]
        
        results = self.base_exp.run_experiment(
            devices=devices,
            shots=shots,
            phase_range=theoretical_phases,
            theta_a=0,
            theta_b=np.pi/4
        )
        
        # Detailed comparison analysis
        if 'analyzed_results' in results:
            self.display_theoretical_comparison(results, theoretical_phases)
        
        self.log_experiment("theoretical_comparison", results)
        return results
    
    def display_theoretical_comparison(self, results: dict, phases: List[float]):
        """Display detailed theoretical vs experimental comparison"""
        analysis = results['analyzed_results']
        theoretical_s = analysis['theoretical_values']['S_theoretical']
        
        table = Table(title="📈 Theory vs Experiment Comparison", show_header=True)
        table.add_column("Phase (rad)", justify="right")
        table.add_column("Phase (deg)", justify="right")
        table.add_column("Theory S", justify="right")
        table.add_column("Device", style="cyan")
        table.add_column("Experimental S", justify="right")
        table.add_column("Difference", justify="right")
        table.add_column("Rel. Error (%)", justify="right")
        
        for device, device_analysis in analysis['device_results'].items():
            experimental_s = device_analysis['S_values']
            
            for i, (phase, theo_s, exp_s) in enumerate(zip(phases, theoretical_s, experimental_s)):
                if not np.isnan(exp_s):
                    diff = abs(exp_s - theo_s)
                    rel_error = abs(diff / theo_s) * 100 if theo_s != 0 else 0
                    
                    style = "green" if rel_error < 10 else "yellow" if rel_error < 25 else "red"
                    
                    table.add_row(
                        f"{phase:.3f}",
                        f"{np.degrees(phase):.1f}",
                        f"{theo_s:.3f}",
                        device.upper(),
                        f"{exp_s:.3f}",
                        f"{diff:.3f}",
                        f"{rel_error:.1f}",
                        style=style
                    )
        
        console.print(table)
    
    def log_experiment(self, experiment_type: str, results):
        """Log experiment for research record"""
        log_entry = {
            'timestamp': time.time(),
            'experiment_type': experiment_type,
            'experiment_id': self.base_exp.experiment_name,
            'results_summary': self.extract_summary(results)
        }
        self.experiment_log.append(log_entry)
        
        if self.verbose:
            console.print(f"📝 Logged: {experiment_type}")
    
    def extract_summary(self, results) -> dict:
        """Extract summary statistics from results"""
        summary = {'status': 'completed'}
        
        if isinstance(results, dict) and 'analyzed_results' in results:
            analysis = results['analyzed_results']
            if 'device_results' in analysis:
                summary['devices'] = list(analysis['device_results'].keys())
                summary['max_s_values'] = {}
                summary['bell_violations'] = {}
                
                for device, device_analysis in analysis['device_results'].items():
                    stats = device_analysis['statistics']
                    summary['max_s_values'][device] = stats['max_S_magnitude']
                    summary['bell_violations'][device] = stats['bell_violations']
        
        return summary
    
    def generate_research_report(self):
        """Generate comprehensive research report"""
        tree = Tree("🔬 Research Experiment Report")
        tree.add(f"📋 Topic: [bold]{self.research_topic}[/bold]")
        tree.add(f"📅 Experiments: {len(self.experiment_log)}")
        
        experiments_node = tree.add("🧪 Experiments Conducted")
        
        for i, entry in enumerate(self.experiment_log, 1):
            exp_node = experiments_node.add(f"{i}. {entry['experiment_type']}")
            
            summary = entry['results_summary']
            if 'devices' in summary:
                devices_node = exp_node.add(f"📱 Devices: {', '.join(summary['devices'])}")
            
            if 'max_s_values' in summary:
                for device, max_s in summary['max_s_values'].items():
                    exp_node.add(f"📊 {device}: max|S| = {max_s:.3f}")
            
            if 'bell_violations' in summary:
                for device, violations in summary['bell_violations'].items():
                    exp_node.add(f"⚡ {device}: {violations} Bell violations")
        
        console.print(tree)
        return self.experiment_log


@app.command()
def study(
    mode: Annotated[ResearchMode, typer.Argument(help="🔬 Research study mode")],
    
    devices: Annotated[
        List[DeviceType], 
        typer.Option(help="📱 Quantum devices to use")
    ] = [DeviceType.qulacs],
    
    shots: Annotated[
        int, 
        typer.Option(help="🎲 Number of measurement shots")
    ] = 2000,
    
    topic: Annotated[
        str,
        typer.Option(help="📝 Research topic name")
    ] = "chsh_research",
    
    verbose: Annotated[
        bool, 
        typer.Option("--verbose", "-v", help="🔍 Verbose output")
    ] = False
):
    """
    🔬 Run advanced CHSH research studies
    
    Conducts specialized research experiments for quantum advantage studies.
    """
    
    # Initialize research CLI
    research = CHSHResearchCLI(topic, verbose)
    device_list = [d.value for d in devices]
    
    try:
        if mode == ResearchMode.bell_study:
            results = research.run_bell_study(device_list, shots, points=50)
            
        elif mode == ResearchMode.angle_sensitivity:
            results = research.run_angle_sensitivity(device_list, shots)
            
        elif mode == ResearchMode.noise_robustness:
            results = research.run_noise_robustness(device_list, shots)
            
        elif mode == ResearchMode.theoretical_comparison:
            results = research.run_theoretical_comparison(device_list, shots)
            
        elif mode == ResearchMode.comprehensive:
            # Run all studies
            console.print("🚀 Running comprehensive research suite...")
            
            research.run_bell_study(device_list, shots//2, points=30)
            research.run_angle_sensitivity(device_list, shots//2)
            research.run_noise_robustness(device_list, shots//3)
            research.run_theoretical_comparison(device_list, shots//2)
        
        # Generate final report
        console.print("\n📋 Generating research report...")
        research.generate_research_report()
        
        console.print(f"\n✅ Research study completed!")
        console.print(f"📁 Results saved in: {research.base_exp.data_manager.session_dir}")
        
    except KeyboardInterrupt:
        console.print("⚠️ Research interrupted by user", style="yellow")
        research.generate_research_report()
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"❌ Research failed: {e}", style="red")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def batch(
    config_file: Annotated[
        Path,
        typer.Argument(help="📄 Batch configuration file (JSON)")
    ],
    
    verbose: Annotated[
        bool, 
        typer.Option("--verbose", "-v", help="🔍 Verbose output")
    ] = False
):
    """
    📦 Run batch experiments from configuration file
    
    Executes multiple research studies based on a JSON configuration file.
    """
    
    import json
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        console.print(f"📦 Loading batch configuration from {config_file}")
        
        # Process batch experiments
        for experiment_config in config.get('experiments', []):
            topic = experiment_config['topic']
            mode = ResearchMode(experiment_config['mode'])
            devices = [DeviceType(d) for d in experiment_config.get('devices', ['qulacs'])]
            shots = experiment_config.get('shots', 1000)
            
            console.print(f"\n🔬 Starting: {topic} ({mode.value})")
            
            research = CHSHResearchCLI(topic, verbose)
            device_list = [d.value for d in devices]
            
            # Run experiment based on mode
            if mode == ResearchMode.bell_study:
                research.run_bell_study(device_list, shots)
            elif mode == ResearchMode.angle_sensitivity:
                research.run_angle_sensitivity(device_list, shots)
            # ... other modes
            
            console.print(f"✅ Completed: {topic}")
        
        console.print("\n🎉 Batch experiments completed!")
        
    except FileNotFoundError:
        console.print(f"❌ Configuration file not found: {config_file}", style="red")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"❌ Batch execution failed: {e}", style="red")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.callback()
def main():
    """
    🔬 Research CLI - Advanced CHSH quantum research experiments
    
    Specialized tools for in-depth quantum advantage research using Bell inequality violations.
    """
    pass


if __name__ == "__main__":
    app()