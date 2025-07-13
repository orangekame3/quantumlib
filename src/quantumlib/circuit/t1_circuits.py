#!/usr/bin/env python3
"""
T1 Circuit Factory - T1実験専用回路作成
"""

from typing import Any

import numpy as np

# Qiskitのみに依存（OQTOPUS非依存）
try:
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from qiskit_aer.noise import NoiseModel, thermal_relaxation_error
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    QuantumCircuit = None


class T1CircuitFactory:
    """
    T1実験回路作成専用ファクトリー
    単一量子ビットのT1減衰測定
    """

    @staticmethod
    def create_t1_circuit(delay_time: float, t1: float = 500, t2: float = 500) -> Any:
        """
        T1減衰測定回路を作成

        Args:
            delay_time: 遅延時間 [ns]
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

        # |1⟩状態に励起
        qc.x(0)

        # 遅延時間の間待機（delayを使用）
        qc.delay(int(delay_time), 0, unit="ns")

        # Z基底測定
        qc.measure(0, 0)

        return qc

    @staticmethod
    def create_t1_with_noise_model(delay_time: float, t1: float = 500, t2: float = 500) -> tuple[Any, Any]:
        """
        ノイズモデル付きT1減衰測定回路を作成

        Args:
            delay_time: 遅延時間 [ns]
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

        # |1⟩状態に励起
        qc.x(0)

        # 遅延時間の間待機（delayを使用）
        qc.delay(int(delay_time), 0, unit="ns")

        # Z基底測定
        qc.measure(0, 0)

        return qc, noise_model

    @staticmethod
    def create_multiple_t1_circuits(delay_times: list[float], t1: float = 500, t2: float = 500) -> list[Any]:
        """
        複数の遅延時間に対するT1回路を作成

        Args:
            delay_times: 遅延時間のリスト [ns]
            t1: T1緩和時間 [ns]
            t2: T2緩和時間 [ns]

        Returns:
            Qiskit量子回路のリスト
        """
        circuits = []
        for delay_time in delay_times:
            circuit = T1CircuitFactory.create_t1_circuit(delay_time, t1, t2)
            circuits.append(circuit)
        return circuits


# 便利関数
def create_t1_circuit(delay_time: float, t1: float = 500, t2: float = 500) -> Any:
    """
    T1減衰測定回路作成の便利関数
    """
    return T1CircuitFactory.create_t1_circuit(delay_time, t1, t2)


def create_t1_with_noise_model(delay_time: float, t1: float = 500, t2: float = 500) -> tuple[Any, Any]:
    """
    ノイズモデル付きT1減衰測定回路作成の便利関数
    """
    return T1CircuitFactory.create_t1_with_noise_model(delay_time, t1, t2)


def create_multiple_t1_circuits(delay_times: list[float], t1: float = 500, t2: float = 500) -> list[Any]:
    """
    複数T1回路作成の便利関数
    """
    return T1CircuitFactory.create_multiple_t1_circuits(delay_times, t1, t2)