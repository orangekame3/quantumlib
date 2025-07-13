#!/usr/bin/env python3
"""
Simple Data Manager for QuantumLib Project
シンプルで統一されたデータ保存システム
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List
import matplotlib.pyplot as plt


class SimpleDataManager:
    """
    シンプルなデータ管理システム
    """
    
    def __init__(self, experiment_name: str = None):
        """
        Initialize simple data manager
        
        Args:
            experiment_name: 実験名（省略時は自動生成）
        """
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if experiment_name is None:
            experiment_name = f"exp_{self.timestamp}"
        
        # シンプルなディレクトリ構造 (.results はgitignoreで除外)
        self.session_dir = f".results/{experiment_name}_{self.timestamp}"
        
        # 必要最小限のディレクトリ作成
        os.makedirs(f"{self.session_dir}/plots", exist_ok=True)
        os.makedirs(f"{self.session_dir}/data", exist_ok=True)
        
        self.files = []
        print(f"Results: {self.session_dir}")
    
    def save_plot(self, fig, name: str, formats: List[str] = ["png"]) -> str:
        """
        プロット保存
        
        Args:
            fig: matplotlib figure
            name: ファイル名
            formats: 保存形式
            
        Returns:
            保存パス
        """
        saved_files = []
        for fmt in formats:
            filename = f"{name}_{self.timestamp}.{fmt}"
            path = f"{self.session_dir}/plots/{filename}"
            fig.savefig(path, dpi=300, bbox_inches='tight')
            saved_files.append(path)
            self.files.append(path)
        
        print(f"📊 Saved: {len(saved_files)} plot files")
        return saved_files[0] if saved_files else None
    
    def save_data(self, data: Dict[str, Any], name: str) -> str:
        """
        データ保存（JSON固定）
        
        Args:
            data: 保存データ
            name: ファイル名
            
        Returns:
            保存パス
        """
        filename = f"{name}_{self.timestamp}.json"
        path = f"{self.session_dir}/data/{filename}"
        
        # numpy対応のJSON保存
        json_data = self._convert_for_json(data)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        self.files.append(path)
        print(f"Saved: {filename}")
        return path
    
    def summary(self) -> str:
        """
        セッション要約作成
        
        Returns:
            要約ファイルパス
        """
        summary = {
            "session_dir": self.session_dir,
            "timestamp": self.timestamp,
            "total_files": len(self.files),
            "files": [os.path.basename(f) for f in self.files]
        }
        
        path = f"{self.session_dir}/summary.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"📋 Summary: {path}")
        return path
    
    def _convert_for_json(self, obj):
        """JSON変換用ヘルパー"""
        if isinstance(obj, dict):
            return {k: self._convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif hasattr(obj, 'tolist'):  # numpy array
            return obj.tolist()
        elif hasattr(obj, 'item'):   # numpy scalar
            return obj.item()
        else:
            return obj


def main():
    """Demo"""
    manager = SimpleDataManager("demo")
    
    # プロット保存
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 2])
    manager.save_plot(fig, "test_plot")
    
    # データ保存
    data = {"results": [1, 2, 3], "config": {"shots": 1000}}
    manager.save_data(data, "test_data")
    
    # 要約作成
    manager.summary()
    
    print("✅ Demo completed!")


if __name__ == "__main__":
    main()