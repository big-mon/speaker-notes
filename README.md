# Speaker Notes CLI

日本語ポッドキャストをMac上で文字起こしし、要約・価値あるシーン抽出に渡す**時刻付きの元資料**を生成します。Qwenを本文、Appleを照合用に使用し、話者候補と確認が必要な箇所を残します。要約やシーンの順位付け自体は後段のエージェントが行います。

## 準備

Apple silicon / macOS 26以降 / Swift 6.2以降 / Python 3.12以降が必要です。実測環境はM4 Pro・48GB・macOS 26.6.2・Swift 6.3.3・Python 3.14.7です。他の構成での動作保証ではありません。

```sh
# 取得元・固定版・容量・ライセンスを表示（ダウンロードしない）
./scripts/setup.sh --plan

# 内容を確認してから依存・モデルを取得し、4つのCLIをビルド
./scripts/setup.sh --install

# Apple日本語モデルが未準備の場合だけ、明示的に準備
.build/release/apple-transcribe --prepare-model

# 準備済みモデルのハッシュ・実行環境・Apple日本語モデルを確認
./scripts/setup.sh --verify
```

必要なモデルは約5.94GB、Python依存の展開済みファイルは今回約0.60GBです。Apple管理モデル、Swiftビルド、キャッシュ、入力・出力は別です。詳細は[依存関係](docs/DEPENDENCIES.md)。通常の文字起こしはモデルを自動取得せず、音声を外部へ送信しません。

## 文字起こし

リポジトリ直下で実行します。出力先は新しいディレクトリを指定してください。

```sh
# 省略時は全編
./speaker-notes "/path/to/podcast.mp3" runs/my-episode

# 冒頭3分だけ試す（既定で前後最大15秒の文脈を含む）
./speaker-notes "/path/to/podcast.mp3" runs/my-preview --duration 180

# 中盤の指定範囲を試す
./speaker-notes "/path/to/podcast.mp3" runs/my-middle --start 1290 --duration 120
```

入力はAVFoundationが読み込める音声です。今回の検証対象はMP3です。`--context-seconds 0` なら指定範囲だけ処理します。指定範囲と実処理範囲は別々に記録します。

工程名とアラインメントの処理済みチャンク数を表示し、最後に出力パスをJSONで返します。架空の進捗率は表示しません。`Ctrl-C` で子プロセスも停止し、途中結果とログを残します。終了コードは成功0、入力引数エラー2、処理失敗1、中止130です。既存の出力先は上書きしません。

## 結果をエージェントに渡す

`runs/my-episode/result/` に以下を生成します。

| ファイル | 用途 |
|---|---|
| `source-material.json` | 入力の識別情報、モデル・設定、時間の基準 |
| `segments.jsonl` | **エージェント用の本文**。1行1区間、原音声の秒数、ID、話者候補、未確認事項 |
| `asr-differences.jsonl` | Appleとの違い。正解や自動修正文ではない |
| `transcript.md` / `.txt` | 人が通読する時刻付き本文 |
| `transcript.json` | 原出力参照・アラインメント・全メタデータを含む詳細版 |
| `README.md` | この素材の読み方と利用上の制約 |

後段ではまず `source-material.json` と `segments.jsonl` を読み、質問・主張・理由・具体例を含むシーン候補を選びます。候補には `artifact_id`、区間ID、元音声の秒範囲を付けます。固有名詞・否定・数字・引用を確定する前に、差分と音声を確認してください。話者が不確かな候補は匿名で扱い、文章を推測で補いません。

```sh
# 結合による文字の欠落・二重化、時刻範囲、各出力の整合を検査
python3 scripts/validate_output.py runs/my-episode

# 同一入力・設定の完了済み結果を再利用したい場合（全編専用）
python3 scripts/cached_pipeline.py "/path/to/podcast.mp3" --cache runs/cli-cache
```

キャッシュは入力・モデル・実装・実行環境を識別し、完了結果を検証して再利用します。失敗工程からの途中再開は行いません。異なる条件・変更された結果には新しい試行を作ります。詳しい出力構造は[リポジトリ案内](docs/REPOSITORY_MAP.md)、人による確認は[確認手順](docs/REVIEW_WORKFLOW.md)。

出力はテキストとJSONに限定し、確認用HTMLは生成しません。人の修正は確認票JSONで別保存できます。Qwen本文・Apple照合・Silero VAD・Community-1の固定構成なので、旧 `--transcript-mode`・`--chunk-policy`・`--diarizer`・キャッシュの `--engine` は不要です。使用設定の記録は引き続き出力に含めます。

## 処理と残る制限

`音声 → 16kHzモノラルPCM → Silero VADで分割位置を検出 → 全選択範囲を1回だけ話者分離 → Apple照合用ASR → Qwen本文 → 強制アラインメント → 時刻による結合・出力` を順次実行します。

Qwenの入力は最大180秒で、VADが見つけた短い休止に分割点を寄せます。休止が見つからない場合は低エネルギー箇所で切り、未解決の境界として記録します。音声の無音部分は削除せず、全サンプルを一度ずつ処理します。VADの候補が常に単語間・息継ぎとは限りません。

- Qwenの固有名詞の誤認、相づち・重なりの欠落、時刻ずれは残ります。細かい時刻が使えない行は入力チャンクの範囲を明示します。チャンク境界は意味の区切りとは限らないので前後の行も読んでください。Appleの一致も正解の証明にはなりません。
- Community-1は現在の話者候補生成に使用します。完全な話者識別やNemotronに対する精度優位を保証しません。不明・混在はそのまま残します。
- 話者IDは1回の処理内で最初の登場順にA/B/Cを割り当てます。別々に処理したクリップ間では対応しません。
- 全文処理成功、構造検査、人の聴取確認は別です。出力は未校正の素材で、全文精度を保証しません。

[用途別の品質基準](docs/SCENE_QUALITY.md)と[今回の全編検証](docs/VALIDATION.md)に確認範囲と制限を記載します。

## 開発・資料の配置

```sh
python3 -m unittest discover -s scripts -p 'test_*.py'
```

テストはモデルを取得・実行しません。macOSの制限された環境では、キャンセル検査に必要なプロセス参照権限が必要です。Swift変更時は対象productをビルドしてください。

現行実装は `scripts/` と4つの `Sources/`、固定条件は `config/`・`requirements/`・`patches/` にあります。過去のGUI・モデル比較・調査ログは[保管案内](archive/README.md)に分離しました。音声、モデル、実行結果、ローカルの過去資料はGitへ含めません。
