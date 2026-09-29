# リポジトリ案内

| 場所 | 内容 |
|---|---|
| `speaker-notes` | 通常のCLI入口。省略時は全編 |
| `scripts/scene_pipeline.py` | 逐次処理・原音声時刻・ログ・失敗・キャンセル |
| `scripts/vad_chunking.py`, `asr_chunking.py` | VAD休止候補・低エネルギーfallback・サンプル保存 |
| `scripts/qwen_asr.py` | 1回ロードしたQwenで全チャンクを認識 |
| `scripts/align_qwen.py`, `compose_alignment.py` | チャンクの時刻候補と原文位置の統合 |
| `scripts/aligned_speaker_turns.py`, `turn_candidates.py` | 話者交代と不明・混在・境界の保留 |
| `scripts/review_aligned_scene.py`, `scene_transcript.py` | 結合と各形式への出力 |
| `scripts/text_anchor_audit.py` | Apple/Qwen差分。正解参照ではない |
| `scripts/cached_pipeline.py` | 全編の完了済み結果の検証・再利用 |
| `scripts/validate_output.py` | 完成出力の構造検査 |
| `scripts/apply_scene_review.py` | 人の確認票を照合し、原文と別に保存 |
| `Sources/{Normalize,AppleTranscribe,SileroVADFrames,FluidDiarize}` | SwiftPMの4つのCLI |
| `scripts/setup.sh`, `setup_cli.py` | 準備計画・明示取得・準備確認 |
| `config/models/`, `config/runtime.json` | 固定モデルの取得元・版・SHA-256、依存条件 |
| `requirements/`, `patches/` | 固定Python依存とFluidAudioの不要TTS除外パッチ |
| `scripts/test_*.py` | モデル不要の回帰検査 |
| `docs/` | 現行の運用・品質・検証記録 |
| `docs/DESIGN_DECISIONS.md` | 目的・ASR選定・分割・話者候補の採用理由 |

## 実行結果

1回の実行ディレクトリには、`provenance.json`（入力・モデル・コード）、`processing.json`（実測工程）、`logs/`、`normalized.wav`（全音声）、`audio.wav`（選択範囲）を保存します。

`vad/`・`vad-chunk-plan.json` に休止候補と分割根拠、`qwen-input/` に分割PCM、`diarization/` に全選択範囲の話者候補、`apple/`・`qwen/` に原ASRを保持します。`alignments/` は各チャンク、`alignment.json` は統合した時刻候補です。完成資料は `result/`。

`result/segments.jsonl` が後段エージェントの本文入口です。原音声の秒数と `artifact_id + id` を出典に使い、`asr_difference_ids` が指す `asr-differences.jsonl` を必要に応じて読みます。詳細は `transcript.json` の原出力参照へ戻れます。

## 再実行の注意

- 出力先は必ず新規。失敗時は `failure.json` の工程・エラーとその工程のログを見る。
- 不足モデルは `setup.sh --verify` で特定し、明示的に準備する。通常処理で自動取得しない。
- キャッシュは完了済みの原結果だけ再利用。途中再開・手動修正結果の自動上書きはしない。
- 詳細版はローカルの絶対パスも持つため、実行ディレクトリを移動すると原出力への参照が切れることがある。利用中の成果物と参照先を一組で残し、不要な試行は削除する。
- `vendor/`・`models/`・`runs/` はGit管理外。クローンだけでは音声・モデル・過去結果は付属しない。

利用可能な対象1話の完成結果は `runs/cli-release-v1/full-v2/`。同階層の `scene-candidates.md` とその根拠JSONはシーン候補の資料です。旧実験の試行データや旧コードの保管先は設けません。
