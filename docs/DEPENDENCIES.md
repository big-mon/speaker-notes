# 準備・依存関係

対象は Apple silicon / macOS 26 以降、Swift 6.2 以降、Python 3.12 以降です（固定したNumPy/SciPyの必要条件）。実測環境は M4 Pro・48 GB・macOS 26.6.2・Swift 6.3.3・Python 3.14.7。別の機種・OS・Python 版での動作や処理時間は未検証です。Xcode Command Line Tools、Git、Python は事前に用意します。

リポジトリ直下で実行します。引数なしでは準備計画を表示するだけです。

```sh
scripts/setup.sh --plan
scripts/setup.sh --install --prepare-apple
scripts/setup.sh --verify
```

`--install` は固定版の FluidAudio を取得し、`.venv-asr` に固定 Python 依存を準備し、下記 4 モデルと CLI の 4 実行物を用意します。GUI・Cohere・Parakeet・Nemotron は準備しません。既存の `.venv-asr` が固定版と一致すれば Python パッケージの再インストールはしません。新しい仮想環境の Python は `PYTHON=/path/to/python3.14 scripts/setup.sh --install` で選べます。Python依存は配布wheelのみを使い、対象Python版に対応するwheelがなければ停止します。未固定のビルド依存を自動取得してソースビルドには進みません。`PYTHON` を明示し、既存環境のベースPythonと異なる場合は `.venv-asr` をクリアして指定Pythonで再作成します。旧仮想環境は保管せず、固定依存を入れ直します。

Apple の日本語モデル取得は `--prepare-apple` を明示した場合のみ行います。既に入っていれば再取得しません。Apple モデルが未準備の `--install` は、それ以外を準備した後の検証で終了コード 1 を返します。後から次のコマンドだけで Apple モデルを準備できます。

```sh
.build/release/apple-transcribe --check-model
.build/release/apple-transcribe --prepare-model
```

`--verify` はモデルファイル全体の SHA-256、Python パッケージ版、FluidAudio の commit・パッチ・未追跡ファイルの不在、CLI の存在、Apple 日本語モデルの準備状態を確認します。推論やモデル取得は行いません。実行物の存在確認は実際の音声処理の成功を保証しないため、準備後は README の短区間実行で確認します。macOS の Speech/Metal 機能がサンドボックス内で利用できない場合は、通常のターミナルか環境の承認手順を通した実行で確認してください。

## モデル

精度を変更しない固定構成です。各ファイルの容量と SHA-256 は [config/models](../config/models/) に保存しています。準備計画は各モデルの目的・取得先・commit・ライセンス・合計容量・未取得ファイル容量を表示します。計画段階の既存ファイルはハッシュ未検証です。

| 用途・モデル | 配布元と固定 revision | 容量（bytes） | ライセンス |
|---|---|---:|---|
| 主ASR・Qwen3-ASR-1.7B BF16 | [mlx-community/Qwen3-ASR-1.7B-bf16](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-bf16/tree/e1f6c266914abc5a46e8756e02580f834a6cf8a7) | 4,080,710,353 | Apache-2.0 |
| 時刻候補・Qwen3-ForcedAligner-0.6B BF16 | [mlx-community/Qwen3-ForcedAligner-0.6B-bf16](https://huggingface.co/mlx-community/Qwen3-ForcedAligner-0.6B-bf16/tree/53c8c0e46733eec430e4b53dd6471d0e5dee45f8) | 1,840,062,950 | Apache-2.0 |
| 切点候補・Silero VAD v6.0.0 / 32 ms Core ML | [FluidInference/silero-vad-coreml](https://huggingface.co/FluidInference/silero-vad-coreml/tree/b419383c55c110e2c9271fa6ee0ea83d03c70d96) | 911,498 | MIT |
| 話者候補・Community-1 Core ML | [FluidInference/speaker-diarization-coreml](https://huggingface.co/FluidInference/speaker-diarization-coreml/tree/df2625ac79a7ac6b65ad868fee6d80f320da4232) | 21,635,985 | CC-BY-4.0 |
| 補助ASR・Apple SpeechTranscriber ja-JP | Apple AssetInventory が管理 | APIでは取得不可 | Apple の利用条件 |

固定モデルは合計 **5,943,320,786 bytes（約5.94 GB）**。Apple のモデルは別途必要で、正確なモデル版と容量はこのAPIから取得できません。Community-1 のモデル配布物に含まれる `LICENSE`・`NOTICE.md`・`provenance.json` も取得・検証します。使用に際して配布元の条件・帰属表記を確認してください。

Community-1の準備・検証・実行は、FluidAudioが実際に読む `models/fluid/speaker-diarization/` に揃えています。`setup.sh --install` は全ファイルの検証後にローカルの `.fluidaudio-revision` を作り、`--verify` もその固定版を確認します。通常の話者分離では `ModelHub.offlineMode` を有効にし、不足・破損時の自動ダウンロードを禁止します。

ダウンロード途中のファイルは検証前に本来の名前へ変更しません。既存ファイルが固定ハッシュと異なる場合は上書きせず停止します。失敗時も元音声や過去の出力を削除しません。エラーに表示された対象を確認し、異なるモデルを別途保管したうえで再準備してください。

## ランタイム

- [FluidAudio](https://github.com/FluidInference/FluidAudio/tree/20d4f0bd46d11d7f50a6eb4f7835cfdbd2b4ba14): commit `20d4f0bd46d11d7f50a6eb4f7835cfdbd2b4ba14`、Apache-2.0。未使用のTTSバイナリ依存を外す [パッチ](../patches/fluid-no-tts-artifact.patch) のみ適用します。話者モデルや推論処理への変更ではありません。既存checkoutの版や変更が違えばリセットせず停止します。
- Qwen は MLX-Audio `0.5.7` / MLX `0.32.2` / transformers `5.17.0` を使用します。日本語アラインメントは nagisa `0.3.0` / DyNet38 `2.2` を追加しています。[requirements/asr.txt](../requirements/asr.txt) に間接依存も含めて固定しています。取得先は PyPI、各パッケージの版・ライセンス・取得元は [config/runtime.json](../config/runtime.json) に記録しています。
- 実測環境の上記主要ライブラリの Python ソースは配布パッケージの `RECORD` ハッシュと一致しました。MLX のローカルパッチは不要です。動的な parser の観測処理は `align_qwen.py` 内で生時刻の保存に使います。
- Python 配布パッケージのインストール済みファイルは合計 **604,446,274 bytes**（この Mac の測定値）。wheel の転送容量ではありません。Python 本体、Swift のビルド出力、ダウンロード一時領域、音声と処理結果の容量は別途必要です。目安としてモデル以外にも数GBの余裕を確保し、空き容量を確認してください。
- 通常の文字起こしは `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` を設定し、Apple も既存モデルのみ使用します。準備でモデルやコードを取得しますが、音声を外部へ送信しません。

既存環境での再利用・ハッシュ検証と実音声での実行を確認しています。別の空のMacで全依存を新規ダウンロードする試験とは区別します。
