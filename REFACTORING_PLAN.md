# QuantumLib リファクタリング計画

## 📋 概要

このドキュメントは、QuantumLibプロジェクトの包括的なリファクタリング計画を記載しています。主な目標は、コード重複の削除、未使用コードの整理、機能統合の促進です。

## 🎯 目標

- **コード重複削除**: 1,500-2,000行の重複コードを60-70%削減
- **保守性向上**: 統一されたアーキテクチャによる開発効率化
- **API改善**: 隠蔽された機能の公開と一貫したインターフェース
- **品質向上**: 未使用コードの削除とコード品質の標準化

## 🚨 重要な問題点

### 1. **重大: CLIエントリーポイントの欠落**
- **問題**: `pyproject.toml`で定義された5つのCLIコマンドが存在しない
- **影響**: パッケージインストール後にCLIが動作しない
- **優先度**: 🔴 最高

### 2. **大量のコード重複**
- **並列実行**: 1,000行以上の重複
- **ユーティリティ**: 500行以上の重複
- **表示ロジック**: 300行以上の重複
- **優先度**: 🔴 高

### 3. **二重実装**
- **CLI**: `workspace/scripts/` と `src/quantumlib/cli/` で重複
- **実験**: 機能レベルでの重複実装
- **優先度**: 🟡 中

## 📊 実装計画

### **🔴 フェーズ1: CLIエントリーポイント修正**
**目標**: パッケージの基本機能を復旧
**期間**: 即座
**影響**: 重大な機能不全の修正

#### タスク
- [ ] 欠落しているCLIモジュールの作成
  - `src/quantumlib/cli/chsh.py`
  - `src/quantumlib/cli/rabi.py` 
  - `src/quantumlib/cli/ramsey.py`
  - `src/quantumlib/cli/t1.py`
  - `src/quantumlib/cli/t2_echo.py`
- [ ] `workspace/scripts/` からの実装移行
- [ ] パッケージビルドとインストールテスト

#### 成果物
```
src/quantumlib/cli/
├── __init__.py
├── base_cli.py (既存)
├── chsh.py (新規)
├── rabi.py (新規) 
├── ramsey.py (新規)
├── t1.py (新規)
└── t2_echo.py (新規)
```

### **🔴 フェーズ2: 並列実行フレームワーク統合**
**目標**: 最大のコード重複を削除
**期間**: 中期
**影響**: 1,000行以上の重複削除

#### タスク
- [ ] `ParallelExecutionMixin` の設計と実装
- [ ] 全実験クラスでの並列実行ロジック統合
- [ ] テストケースの作成と検証
- [ ] 既存機能の動作保証

#### 成果物
```python
# src/quantumlib/core/parallel_execution.py
class ParallelExecutionMixin:
    def submit_circuits_parallel_with_order(self, circuits, devices, shots, parallel_workers)
    def collect_results_parallel_with_order(self, job_data, parallel_workers)
    def poll_job_until_completion(self, job_id, timeout_minutes)
```

#### 対象ファイル
- `src/quantumlib/experiments/chsh/chsh_experiment.py`
- `src/quantumlib/experiments/rabi/rabi_experiment.py`
- `src/quantumlib/experiments/ramsey/ramsey_experiment.py`
- `src/quantumlib/experiments/t1/t1_experiment.py`
- `src/quantumlib/experiments/t2_echo/t2_echo_experiment.py`

### **🟡 フェーズ3: ユーティリティ関数統合**
**目標**: 共通機能の統一化
**期間**: 中期
**影響**: 500行の重複削除

#### タスク
- [ ] OQTOPUS変換ユーティリティの統合
- [ ] 結果表示インフラの統合
- [ ] 共通統計処理の抽出
- [ ] インポートパターンの標準化

#### 成果物
```python
# src/quantumlib/utils/
├── __init__.py
├── oqtopus_utils.py
├── display.py
├── statistics.py
└── dependencies.py
```

### **🟡 フェーズ4: workspace機能の本体統合**
**目標**: 高機能な機能の本体取り込み
**期間**: 長期
**影響**: 機能性の大幅向上

#### タスク
- [ ] 研究用CLIフレームワークの統合
- [ ] 設定システムの本体組み込み
- [ ] 拡張CHSHテンプレートの統合
- [ ] API公開の改善

#### 対象
- `workspace/experiments/research_cli.py` → 本体統合
- `workspace/configs/` → 設定システム統合
- `workspace/experiments/chsh_research_template.py` → テンプレート統合

### **🟢 フェーズ5: 未使用コード削除とAPI改善**
**目標**: コードベースの最終整理
**期間**: 長期
**影響**: 保守性とAPIの改善

#### タスク
- [ ] 未使用モジュールの削除判断
  - `chsh_legacy.py` (478行) の評価
- [ ] 非公開APIの公開化
  - `RamseyExperiment`, `T2EchoExperiment` の公開
  - 回路作成関数の公開
  - テンプレートAPIの公開
- [ ] ドキュメント更新

## 📈 進捗追跡

### 完了基準

#### フェーズ1
- [ ] 全CLIコマンドが正常に動作
- [ ] パッケージがエラーなくインストール可能
- [ ] CI/CDパイプラインが通過

#### フェーズ2  
- [ ] 全実験で並列実行が統一フレームワークで動作
- [ ] 既存のテストが全て通過
- [ ] 新しいmixinのテストカバレッジ ≥ 90%

#### フェーズ3
- [ ] ユーティリティ関数の重複が完全に削除
- [ ] APIドキュメントが更新済み
- [ ] リグレッションテストの実装

#### フェーズ4
- [ ] workspace機能が本体で利用可能
- [ ] 設定システムが統合済み
- [ ] 機能拡張のドキュメント作成

#### フェーズ5
- [ ] 未使用コードが削除または統合済み
- [ ] API参照ドキュメントが完全
- [ ] コードベースが一貫したアーキテクチャに統一

## 🔍 品質管理

### テスト戦略
- **ユニットテスト**: 新規作成する全ての共通機能
- **統合テスト**: 既存機能との互換性確認
- **リグレッションテスト**: 既存の動作保証

### コード品質
- **Linting**: ruff, black による自動フォーマット
- **型チェック**: mypy による型安全性確保
- **カバレッジ**: 新規コードで ≥ 90% のテストカバレッジ

## 📅 予想スケジュール

```
Week 1-2:   フェーズ1 (CLIエントリーポイント修正)
Week 3-6:   フェーズ2 (並列実行フレームワーク統合)  
Week 7-10:  フェーズ3 (ユーティリティ関数統合)
Week 11-16: フェーズ4 (workspace機能統合)
Week 17-20: フェーズ5 (最終整理とAPI改善)
```

## 🚀 期待効果

### 定量的効果
- **コード行数削減**: 1,500-2,000行 (約60-70%削減)
- **ファイル数整理**: 重複ファイルの統合
- **保守工数削減**: 30-40%の保守工数削減見込み

### 定性的効果  
- **開発者体験向上**: 一貫したAPIと統一されたアーキテクチャ
- **機能発見性向上**: 隠蔽された機能の公開
- **コード品質向上**: 標準化されたパターンとベストプラクティス

## 📝 備考

- 各フェーズは独立しており、段階的な実装が可能
- 既存機能の動作を保証しながらの漸進的改善
- 変更による影響範囲を最小限に抑制
- コミュニティへの影響を考慮した後方互換性の維持

---

**最終更新**: 2025-07-18  
**ステータス**: 計画策定完了、実装準備中