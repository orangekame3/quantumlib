#!/usr/bin/env python3
"""
CHSH Research Template - 研究用実験テンプレート
個人の研究用にカスタマイズされたCHSH実験スクリプト
"""

import numpy as np
import time
import sys

# Add library paths
sys.path.append('../..')  # quantumlib root
sys.path.append('../../src')  # src directory

from src.quantumlib import CHSHExperiment, create_chsh_circuit


class CHSHResearchExperiment:
    """
    研究用CHSH実験クラス
    ライブラリのCHSHExperimentをベースに、研究特化機能を追加
    """
    
    def __init__(self, research_topic: str = "chsh_research"):
        self.research_topic = research_topic
        self.experiment_log = []
        
        # ベース実験クラス
        self.base_exp = CHSHExperiment(f"{research_topic}_{int(time.time())}")
        
        # 研究用設定
        self.setup_research_configuration()
        
        print(f"🔬 Research Topic: {research_topic}")
        print(f"📝 Experiment ID: {self.base_exp.experiment_name}")
    
    def setup_research_configuration(self):
        """研究用の高度な設定"""
        # 高精度設定
        self.base_exp.transpiler_options.update({
            "optimization_level": 3,
            "routing_method": "sabre",
            "layout_method": "dense",
            "approximation_degree": 0.99
        })
        
        # 高度なエラー軽減
        self.base_exp.mitigation_options.update({
            "ro_error_mitigation": "least_squares",
            "zne_noise_factors": [1, 2, 3],
            "extrapolation_method": "linear"
        })
        
        print("🔧 Research-grade configuration applied")
    
    def run_bell_inequality_study(self, devices=['qulacs'], shots=2000):
        """Bell不等式違反の詳細研究"""
        print("\n📊 Bell Inequality Violation Study")
        print("=" * 40)
        
        # 高密度位相スキャン
        results = self.base_exp.run_phase_scan(
            devices=devices,
            phase_points=50,  # 高解像度
            theta_a=0,
            theta_b=np.pi/4,
            shots=shots
        )
        
        self.log_experiment("bell_inequality_study", results)
        return results
    
    def run_angle_sensitivity_analysis(self, devices=['qulacs'], shots=1500):
        """角度感度解析"""
        print("\n📐 Angle Sensitivity Analysis")
        print("=" * 35)
        
        # より細かい角度ステップ
        theta_a_range = np.linspace(0, np.pi/2, 8)
        theta_b_range = np.linspace(0, np.pi/2, 8)
        
        angle_pairs = []
        for ta in theta_a_range:
            for tb in theta_b_range:
                angle_pairs.append((ta, tb))
        
        # サブセットで実行（計算量削減）
        selected_pairs = angle_pairs[::4]  # 4個に1個選択
        
        results = self.base_exp.run_angle_comparison(
            devices=devices,
            angle_pairs=selected_pairs,
            shots=shots
        )
        
        self.log_experiment("angle_sensitivity_analysis", results)
        return results
    
    def run_noise_robustness_test(self, devices=['qulacs'], shots=1000):
        """ノイズ耐性テスト（シミュレータでのテスト）"""
        print("\n🔊 Noise Robustness Test")
        print("=" * 30)
        
        # 異なるショット数でのテスト
        shot_counts = [100, 300, 500, 1000, 2000]
        results_by_shots = {}
        
        for shots_test in shot_counts:
            print(f"🎯 Testing with {shots_test} shots...")
            
            result = self.base_exp.run_phase_scan(
                devices=devices,
                phase_points=10,
                theta_a=0,
                theta_b=np.pi/4,
                shots=shots_test
            )
            
            results_by_shots[shots_test] = result
            
            # 簡易統計表示
            if 'analyzed_results' in result:
                for device, analysis in result['analyzed_results']['device_results'].items():
                    max_s = analysis['statistics']['max_S_magnitude']
                    print(f"   {device}: max|S| = {max_s:.3f}")
        
        self.log_experiment("noise_robustness_test", results_by_shots)
        return results_by_shots
    
    def run_theoretical_comparison(self, devices=['qulacs'], shots=1500):
        """理論値との詳細比較"""
        print("\n📈 Theoretical Comparison Study")
        print("=" * 35)
        
        # 理論予測に基づく位相選択
        theoretical_optimal_phases = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
        
        results = self.base_exp.run_experiment(
            devices=devices,
            shots=shots,
            phase_range=theoretical_optimal_phases,
            theta_a=0,
            theta_b=np.pi/4
        )
        
        # 理論値と実験値の比較分析
        if 'analyzed_results' in results:
            theoretical_s = results['analyzed_results']['theoretical_values']['S_theoretical']
            
            print(f"\n📊 Theory vs Experiment:")
            for device, analysis in results['analyzed_results']['device_results'].items():
                experimental_s = analysis['S_values']
                
                print(f"\n🔬 Device: {device}")
                for i, (phase, theo_s, exp_s) in enumerate(zip(theoretical_optimal_phases, theoretical_s, experimental_s)):
                    if not np.isnan(exp_s):
                        diff = abs(exp_s - theo_s)
                        print(f"  φ={phase:.3f}: Theory={theo_s:.3f}, Exp={exp_s:.3f}, Diff={diff:.3f}")
        
        self.log_experiment("theoretical_comparison", results)
        return results
    
    def log_experiment(self, experiment_type: str, results: dict):
        """実験ログ記録"""
        log_entry = {
            'timestamp': time.time(),
            'experiment_type': experiment_type,
            'experiment_id': self.base_exp.experiment_name,
            'results_summary': self.extract_summary(results)
        }
        self.experiment_log.append(log_entry)
        print(f"📝 Logged: {experiment_type}")
    
    def extract_summary(self, results: dict) -> dict:
        """結果サマリー抽出"""
        summary = {'status': 'completed'}
        
        if isinstance(results, dict) and 'analyzed_results' in results:
            analysis = results['analyzed_results']
            if 'device_results' in analysis:
                summary['devices'] = list(analysis['device_results'].keys())
                summary['max_s_values'] = {}
                
                for device, device_analysis in analysis['device_results'].items():
                    max_s = device_analysis['statistics']['max_S_magnitude']
                    summary['max_s_values'][device] = max_s
        
        return summary
    
    def generate_research_report(self):
        """研究レポート生成"""
        print("\n📋 Research Experiment Report")
        print("=" * 40)
        print(f"🔬 Research Topic: {self.research_topic}")
        print(f"📅 Experiments Conducted: {len(self.experiment_log)}")
        
        for i, entry in enumerate(self.experiment_log, 1):
            print(f"\n{i}. {entry['experiment_type']}")
            print(f"   📊 Devices: {entry['results_summary'].get('devices', 'N/A')}")
            if 'max_s_values' in entry['results_summary']:
                for device, max_s in entry['results_summary']['max_s_values'].items():
                    print(f"   📈 {device}: max|S| = {max_s:.3f}")
        
        return self.experiment_log


def run_comprehensive_chsh_research():
    """包括的CHSH研究実行"""
    print("🧪 Comprehensive CHSH Research Suite")
    print("=" * 50)
    
    # 研究実験インスタンス作成
    research = CHSHResearchExperiment("comprehensive_chsh_study")
    
    # 一連の研究実験実行
    devices = ['qulacs']  # 実環境では ['qulacs', 'anemone']
    
    try:
        # 1. Bell不等式詳細研究
        research.run_bell_inequality_study(devices, shots=1000)
        
        # 2. 角度感度解析
        research.run_angle_sensitivity_analysis(devices, shots=800)
        
        # 3. ノイズ耐性テスト
        research.run_noise_robustness_test(devices, shots=500)
        
        # 4. 理論比較研究
        research.run_theoretical_comparison(devices, shots=1200)
        
        # 5. 研究レポート生成
        research.generate_research_report()
        
        print(f"\n🎉 Comprehensive research completed!")
        print(f"📁 Check results in: {research.base_exp.data_manager.session_dir}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Research interrupted by user")
        research.generate_research_report()
    
    return research


def run_quick_verification():
    """クイック検証実験"""
    print("⚡ Quick CHSH Verification")
    print("=" * 30)
    
    research = CHSHResearchExperiment("quick_verification")
    
    # 基本的な検証のみ
    results = research.run_bell_inequality_study(['qulacs'], shots=500)
    
    print(f"✅ Quick verification completed")
    return research, results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='CHSH Research Template')
    parser.add_argument('--mode', choices=['comprehensive', 'quick'], 
                       default='quick',
                       help='Research mode')
    parser.add_argument('--topic', type=str, default='chsh_research',
                       help='Research topic name')
    
    args = parser.parse_args()
    
    if args.mode == 'comprehensive':
        research = run_comprehensive_chsh_research()
    else:
        research, results = run_quick_verification()
    
    print(f"\n📊 Research completed: {args.mode} mode")