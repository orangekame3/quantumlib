#!/usr/bin/env python3
"""
Rabi Circuit Factory - Rabi実験専用回路作成
"""

from typing import Any

import numpy as np

# Qiskitのみに依存（OQTOPUS非依存）
try:
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False


class RabiCircuitFactory:
    """
    Rabi振動実験回路作成専用ファクトリー
    単一量子ビットのRabi振動測定
    """

    @staticmethod
    def create_rabi_circuit(
        drive_amplitude: float, drive_time: float, drive_frequency: float = 0.0
    ) -> Any:
        """
        Rabi振動回路を作成

        Args:
            drive_amplitude: ドライブ振幅（回転角度に対応）
            drive_time: ドライブ時間（実際は角度として使用）
            drive_frequency: ドライブ周波数（位相として使用）

        Returns:
            Qiskit量子回路
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for circuit creation")

        # 1量子ビット + 1古典ビット
        qubits = QuantumRegister(1, "q")
        bits = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qubits, bits)

        # |0⟩状態から開始

        # Rabi drive: RX回転（振幅×時間 = 回転角度）
        angle = drive_amplitude * drive_time

        # 周波数による位相考慮
        if drive_frequency != 0.0:
            qc.rz(drive_frequency, 0)  # Z軸位相回転

        # X軸周りの回転（Rabiドライブ）
        qc.rx(angle, 0)

        # Z基底測定
        qc.measure(0, 0)

        return qc

    @staticmethod
    def create_ramsey_circuit(wait_time: float, phase: float = 0.0) -> Any:
        """
        Ramsey実験回路を作成

        Args:
            wait_time: 待機時間（位相蓄積に対応）
            phase: 最終π/2パルスの位相

        Returns:
            Qiskit量子回路
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for circuit creation")

        qc = QuantumCircuit(1, 1)

        # 第1のπ/2パルス（重ね合わせ状態作成）
        qc.ry(np.pi / 2, 0)

        # 待機時間による位相蓄積（Z回転で模擬）
        qc.rz(wait_time, 0)

        # 第2のπ/2パルス（位相付き）
        if phase != 0.0:
            qc.rz(phase, 0)
        qc.ry(np.pi / 2, 0)

        qc.measure(0, 0)

        return qc

    @staticmethod
    def create_t1_circuit(delay_time: float) -> Any:
        """
        T1減衰測定回路を作成

        Args:
            delay_time: 遅延時間

        Returns:
            Qiskit量子回路
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for circuit creation")

        qc = QuantumCircuit(1, 1)

        # |1⟩状態に励起
        qc.x(0)

        # 遅延（実際の実装では待機、ここではアイデンティティで代用）
        # 実際のハードウェアでは遅延がT1減衰を引き起こす
        for _ in range(int(delay_time * 10)):  # 擬似的な遅延
            qc.id(0)

        qc.measure(0, 0)

        return qc

    @staticmethod
    def create_t2_echo_circuit(wait_time: float) -> Any:
        """
        T2エコー測定回路を作成（スピンエコー）

        Args:
            wait_time: 待機時間

        Returns:
            Qiskit量子回路
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for circuit creation")

        qc = QuantumCircuit(1, 1)

        # π/2パルス（重ね合わせ状態作成）
        qc.ry(np.pi / 2, 0)

        # τ/2 待機
        qc.rz(wait_time / 2, 0)

        # πパルス（エコー）
        qc.x(0)

        # τ/2 待機
        qc.rz(wait_time / 2, 0)

        # 最終π/2パルス
        qc.ry(np.pi / 2, 0)

        qc.measure(0, 0)

        return qc


# 便利関数
def create_rabi_circuit(
    drive_amplitude: float, drive_time: float, drive_frequency: float = 0.0
) -> Any:
    """
    Rabi振動回路作成の便利関数
    """
    return RabiCircuitFactory.create_rabi_circuit(
        drive_amplitude, drive_time, drive_frequency
    )


def create_ramsey_circuit(wait_time: float, phase: float = 0.0) -> Any:
    """
    Ramsey実験回路作成の便利関数
    """
    return RabiCircuitFactory.create_ramsey_circuit(wait_time, phase)


def create_t1_circuit(delay_time: float) -> Any:
    """
    T1減衰測定回路作成の便利関数
    """
    return RabiCircuitFactory.create_t1_circuit(delay_time)


def create_t2_echo_circuit(wait_time: float) -> Any:
    """
    T2エコー測定回路作成の便利関数
    """
    return RabiCircuitFactory.create_t2_echo_circuit(wait_time)
