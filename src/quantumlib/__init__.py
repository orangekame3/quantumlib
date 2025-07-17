#!/usr/bin/env python3
"""
QuantumLib - Quantum computing experiment library
"""

# Core components
# Backend implementations
from .backend import QuantumExperimentSimple

# Circuit utilities
from .circuit import create_chsh_circuit
from .core import BaseExperiment, SimpleDataManager

# Experiments
from .experiments import CHSHExperiment

__version__ = "0.1.0"
__author__ = "quantumlib"
__all__ = [
    "BaseExperiment",
    "CHSHExperiment",
    "QuantumExperimentSimple",
    "create_chsh_circuit",
    "SimpleDataManager",
]
