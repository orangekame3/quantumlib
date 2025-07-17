#!/usr/bin/env python3
"""
Parity Oscillation Experiment CLI Script (Unified Framework)
Study GHZ state decoherence through parity oscillation measurements
Based on Ozaeta & McMahon (2019) paper

This script uses the QuantumLib unified CLI framework.
For the traditional CLI interface, use parity_oscillation_cli.py
"""

import sys
from pathlib import Path

# Add the src directory to the path so we can import quantumlib
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from parity_oscillation_cli import main

if __name__ == "__main__":
    main()
