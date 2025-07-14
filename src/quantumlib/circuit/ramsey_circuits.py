#!/usr/bin/env python3
"""
Ramsey Circuit Factory - Ramsey実験専用回路作成
"""

from typing import Any

import numpy as np

# Qiskitのみに依存
try:
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from qiskit_aer.noise import NoiseModel, thermal_relaxation_error
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    QuantumCircuit = None


class RamseyCircuitFactory:
    """
    Ramsey実験回路作成専用ファクトリー
    単一量子ビットのRamsey振動測定
    """

    @staticmethod
    def create_ramsey_circuit(delay_time: float, detuning: float = 0.0, t1: float = 500, t2: float = 500) -> Any:
        """
        Ramsey振動測定回路を作成

        Args:
            delay_time: 遅延時間 [ns]
            detuning: 周波数デチューニング [MHz]
            t1: T1緩和時間 [ns]
            t2: T2緩和時間 [ns]

        Returns:
            Qiskit量子回路
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for circuit creation")

        # 1量子ビット + 1古典ビット
        qubits = QuantumRegister(1, 'q')
        bits = ClassicalRegister(1, 'c')
        qc = QuantumCircuit(qubits, bits)

        # First π/2 pulse
        qc.ry(np.pi/2, 0)

        # 遅延時間の間待機（自由進化）
        qc.delay(int(delay_time), 0, unit="ns")

        # デチューニングがある場合は位相回転を追加
        if detuning != 0.0:
            # 位相 = 2π × detuning [MHz] × delay_time [ns] × 1e-3
            phase = 2 * np.pi * detuning * delay_time * 1e-3
            qc.rz(phase, 0)

        # Second π/2 pulse (analysis pulse)
        qc.ry(np.pi/2, 0)

        # Z基底測定
        qc.measure(0, 0)

        return qc

    @staticmethod
    def create_ramsey_with_noise_model(delay_time: float, detuning: float = 0.0, 
                                     t1: float = 500, t2: float = 500) -> tuple[Any, Any]:
        """
        ノイズモデル付きRamsey振動測定回路を作成

        Args:
            delay_time: 遅延時間 [ns]
            detuning: 周波数デチューニング [MHz]
            t1: T1緩和時間 [ns]
            t2: T2緩和時間 [ns]

        Returns:
            (Qiskit量子回路, ノイズモデル)
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for circuit creation")

        # ノイズモデル構築
        error = thermal_relaxation_error(t1, t2, delay_time)
        noise_model = NoiseModel()
        noise_model.add_quantum_error(error, "delay", [0])

        # 回路定義
        qubits = QuantumRegister(1, 'q')
        bits = ClassicalRegister(1, 'c')
        qc = QuantumCircuit(qubits, bits)

        # First π/2 pulse
        qc.ry(np.pi/2, 0)

        # 遅延時間の間待機
        qc.delay(int(delay_time), 0, unit="ns")

        # デチューニング位相
        if detuning != 0.0:
            phase = 2 * np.pi * detuning * delay_time * 1e-3
            qc.rz(phase, 0)

        # Second π/2 pulse
        qc.ry(np.pi/2, 0)

        # Z基底測定
        qc.measure(0, 0)

        return qc, noise_model

    @staticmethod
    def create_multiple_ramsey_circuits(delay_times: list[float], detuning: float = 0.0,
                                      t1: float = 500, t2: float = 500) -> list[Any]:
        """
        複数の遅延時間に対するRamsey回路を作成

        Args:
            delay_times: 遅延時間のリスト [ns]
            detuning: 周波数デチューニング [MHz]
            t1: T1緩和時間 [ns]
            t2: T2緩和時間 [ns]

        Returns:
            Qiskit量子回路のリスト
        """
        circuits = []
        for delay_time in delay_times:
            circuit = RamseyCircuitFactory.create_ramsey_circuit(delay_time, detuning, t1, t2)
            circuits.append(circuit)
        return circuits


# 便利関数
def create_ramsey_circuit(delay_time: float, detuning: float = 0.0, 
                         t1: float = 500, t2: float = 500) -> Any:
    """
    Ramsey振動測定回路作成の便利関数
    """
    return RamseyCircuitFactory.create_ramsey_circuit(delay_time, detuning, t1, t2)


def create_ramsey_with_noise_model(delay_time: float, detuning: float = 0.0,
                                  t1: float = 500, t2: float = 500) -> tuple[Any, Any]:
    """
    ノイズモデル付きRamsey振動測定回路作成の便利関数
    """
    return RamseyCircuitFactory.create_ramsey_with_noise_model(delay_time, detuning, t1, t2)


def create_multiple_ramsey_circuits(delay_times: list[float], detuning: float = 0.0,
                                   t1: float = 500, t2: float = 500) -> list[Any]:
    """
    複数Ramsey回路作成の便利関数
    """
    return RamseyCircuitFactory.create_multiple_ramsey_circuits(delay_times, detuning, t1, t2)