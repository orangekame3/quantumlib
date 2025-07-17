#!/usr/bin/env python3
"""
CHSH Circuit Factory - CHSH実験専用回路作成
"""

from typing import Any

# Qiskitのみに依存（OQTOPUS非依存）
try:
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    QuantumCircuit = None


class CHSHCircuitFactory:
    """
    CHSH回路作成専用ファクトリー
    OQTOPUS/quri-partsに依存しない純粋な回路作成
    """

    @staticmethod
    def create_chsh_circuit(
        theta_a: float, theta_b: float, phase_phi: float = 0
    ) -> Any:
        """
        CHSH回路を作成（標準的なCHSH実験）

        Args:
            theta_a: Alice測定角度
            theta_b: Bob測定角度
            phase_phi: 位相パラメータ（Bell状態の相対位相制御）

        Returns:
            Qiskit量子回路
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for circuit creation")

        # 2量子ビット + 2古典ビット
        qubits = QuantumRegister(2, "q")
        bits = ClassicalRegister(2, "c")
        qc = QuantumCircuit(qubits, bits)

        # Bell状態作成 |Φ+⟩ = (|00⟩ + |11⟩)/√2
        qc.h(0)
        qc.cx(0, 1)

        # 測定基底の回転（古い実装の正常動作する方法）
        # Alice: θ_a回転後のPauli-X測定
        qc.ry(theta_a, 0)

        # Bob: θ_b + φ回転後のPauli-X測定（位相変調を測定角度に適用）
        # これによりS(φ) = S₀ cos(φ)の正しい依存性が得られる
        qc.ry(theta_b + phase_phi, 1)

        # 測定（Z基底での測定 = 回転後のX基底測定）
        qc.measure(0, 0)  # Alice → c[0]
        qc.measure(1, 1)  # Bob → c[1]

        return qc

    @staticmethod
    def create_bell_state_circuit() -> Any:
        """
        基本Bell状態回路を作成
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for circuit creation")

        qc = QuantumCircuit(2, 2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        return qc

    @staticmethod
    def create_custom_bell_measurement(theta_a: float, theta_b: float) -> Any:
        """
        カスタムBell測定回路
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for circuit creation")

        qc = QuantumCircuit(2, 2)

        # Bell状態作成
        qc.h(0)
        qc.cx(0, 1)

        # カスタム測定基底
        qc.ry(2 * theta_a, 0)
        qc.ry(2 * theta_b, 1)

        qc.measure(0, 0)
        qc.measure(1, 1)

        return qc


# 便利関数
def create_chsh_circuit(theta_a: float, theta_b: float, phase_phi: float = 0) -> Any:
    """
    CHSH回路作成の便利関数
    """
    return CHSHCircuitFactory.create_chsh_circuit(theta_a, theta_b, phase_phi)


def create_bell_state() -> Any:
    """
    Bell状態作成の便利関数
    """
    return CHSHCircuitFactory.create_bell_state_circuit()


def create_custom_bell_measurement(theta_a: float, theta_b: float) -> Any:
    """
    カスタムBell測定回路作成の便利関数
    """
    return CHSHCircuitFactory.create_custom_bell_measurement(theta_a, theta_b)
