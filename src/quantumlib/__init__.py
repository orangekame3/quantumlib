#!/usr/bin/env python3
"""
QuantumLib - 量子計算実験ライブラリ
"""

# Core components
from .core import BaseExperiment, SimpleDataManager

# Experiments
from .experiments import CHSHExperiment

# Circuit utilities
from .circuit import create_chsh_circuit

# Backend implementations
from .backend import QuantumExperimentSimple

__version__ = "0.1.0"
__author__ = "quantumlib"
__all__ = [
    "BaseExperiment",
    "CHSHExperiment", 
    "QuantumExperimentSimple",
    "create_chsh_circuit",
    "SimpleDataManager"
]