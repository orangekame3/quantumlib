#!/usr/bin/env python3
"""
Common Circuit Utilities - 共通回路ユーティリティ
"""

from typing import Any

# Qiskitのみに依存（OQTOPUS非依存）
try:
    from qiskit import QuantumCircuit, transpile

    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    QuantumCircuit = None
    transpile = None


class CommonCircuitUtils:
    """
    共通回路ユーティリティクラス
    """

    @staticmethod
    def optimize_circuit(circuit: Any, optimization_level: int = 1) -> Any:
        """
        回路最適化（Qiskit使用）
        """
        if not QISKIT_AVAILABLE:
            return circuit

        try:
            return transpile(circuit, optimization_level=optimization_level)
        except ImportError:
            return circuit

    @staticmethod
    def get_circuit_info(circuit: Any) -> dict[str, Any]:
        """
        回路情報を取得
        """
        if not QISKIT_AVAILABLE or circuit is None:
            return {}

        return {
            "depth": circuit.depth(),
            "gate_count": len(circuit.data),
            "qubit_count": circuit.num_qubits,
            "classical_bits": circuit.num_clbits,
            "gates": [instr.operation.name for instr in circuit.data],
        }

    @staticmethod
    def create_identity_circuit(num_qubits: int) -> Any:
        """
        アイデンティティ回路を作成
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for circuit creation")

        qc = QuantumCircuit(num_qubits, num_qubits)
        # 何もしない（アイデンティティ）
        qc.measure_all()
        return qc

    @staticmethod
    def create_ghz_state(num_qubits: int) -> Any:
        """
        GHZ状態回路を作成
        """
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required for circuit creation")

        qc = QuantumCircuit(num_qubits, num_qubits)

        # GHZ状態作成
        qc.h(0)  # 最初の量子ビットをHadamard
        for i in range(1, num_qubits):
            qc.cx(0, i)  # CNOT連鎖

        qc.measure_all()
        return qc


# 便利関数
def optimize_circuit(circuit: Any, level: int = 1) -> Any:
    """
    回路最適化の便利関数
    """
    return CommonCircuitUtils.optimize_circuit(circuit, level)


def get_circuit_info(circuit: Any) -> dict[str, Any]:
    """
    回路情報取得の便利関数
    """
    return CommonCircuitUtils.get_circuit_info(circuit)


def create_identity_circuit(num_qubits: int) -> Any:
    """
    アイデンティティ回路作成の便利関数
    """
    return CommonCircuitUtils.create_identity_circuit(num_qubits)


def create_ghz_state(num_qubits: int) -> Any:
    """
    GHZ状態回路作成の便利関数
    """
    return CommonCircuitUtils.create_ghz_state(num_qubits)
