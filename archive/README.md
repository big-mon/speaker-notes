# 過去の試作・検証資料

2026-09-30のCLI整理前の資料を、ローカルの `archive/local-pre-cli/` に保持しています。このディレクトリはGit管理外です。公開クローンには含まれません。

| ローカル保管先 | 内容 |
|---|---|
| `research/README.md` | 過去の比較実験の索引 |
| `research/*.md` | ASR・話者分離・VAD・境界・アラインメント・GUIの検証記録 |
| `research/*manifest.json`, `*.log`, `*.json` | 当時の条件、取得記録、実測、原出力への参照 |
| `Sources/SpeakerNotesApp/` | SwiftUI試作 |
| `Sources/` のその他旧ターゲット | Nemotron・Parakeetなどの比較用コード |
| `scripts/` | 当時のCLI一式、比較用処理、GUIビルド・旧セットアップ |
| `README.md`, `AGENTS.md`, `docs/`, `Package.swift` | 整理前の案内と構成 |
| `inventory.json` | 元の相対パス・保管パス・サイズ・SHA-256 |
| `relocated-files.json` | 初回整理で現行配置から分離したコード一覧 |

元音声・旧Apple結果は `~/Downloads/guildtalk-verification-20260928/`、実行結果・モデル・ランタイムは従来の `runs/`・`models/`・`vendor/` に残しています。これらは一括削除・移動していません。

保管資料には当時の絶対パスや全文認識文への参照があります。公開PRには含めません。旧資料の相対リンクは当時の配置を前提とするため、実行時には旧スクリプトの固定出力先と参照を必ず確認してください。保管資料を読むために旧セットアップや比較処理を実行する必要はありません。

現行CLIに必要な4モデルのmanifestは `config/models/`、固定ランタイムは `config/runtime.json` に抜き出しました。過去モデルの比較は現在の採用条件ではありません。

現在の判断は、ユーザーによる意味内容重視の比較を受けて **Qwen本文＋Apple照合** です。Cohere・Parakeet・Nemotron ASRの追加検証は停止しています。Community-1は話者候補生成に使いますが、話者分離の完全な品質判定は保留しています。
