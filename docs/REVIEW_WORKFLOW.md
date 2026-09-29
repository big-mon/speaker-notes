# 音声の確認・修正を後段へ渡す

CLIは時刻付きテキストとJSONを生成します。`segments.jsonl` / `transcript.md` で候補を選び、元音声を手元のプレーヤーで開いて、記録された秒数の前後を聴いてください。確認画面やHTMLは生成しません。判定基準は [SCENE_QUALITY.md](SCENE_QUALITY.md)、確認範囲は [VALIDATION.md](VALIDATION.md) に記載しています。

## 確認票を作る

`transcript.json` の `artifact_id` と入力のSHA-256を使い、確認する行の元値をコピーします。次の例は行ID `0` の未記入票を作ります。既存ファイルは上書きしません。対象の行IDに置き換えて実行してください。

```sh
python3 - runs/my-episode/result/transcript.json review.json <<'PY'
import json
import sys
from pathlib import Path

document = json.loads(Path(sys.argv[1]).read_text())
selected = {'0'}
rows = {row['id']: {
    **{key: row[key] for key in ('text', 'speakers', 'start', 'end')},
    'content': 'review_required', 'speaker': 'review_required',
    'boundary': 'review_required', 'note': '', 'corrections': {}
} for row in document['segments'] if row['id'] in selected}
assert set(rows) == selected, '対象の行IDがありません'
review = {
    'schema_version': 1, 'source_artifact_id': document['artifact_id'],
    'source': {'sha256': document['input']['sha256']},
    'reviewer': '', 'reviewed_at': '', 'listened_ranges': [], 'rows': rows
}
with Path(sys.argv[2]).open('x') as stream:
    json.dump(review, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
PY
```

人が聴いた後に次を記入します。未記入の票は適用時に拒否されます。

- `reviewer`: 確認者の呼び名。
- `reviewed_at`: 実際の確認日時。タイムゾーン付きISO 8601（`YYYY-MM-DDTHH:MM:SS+09:00`）。
- `listened_ranges`: 実際に聴いた**元音声上の**秒範囲。`[{"start": 33.9, "end": 124.0}]` の形式。機械が再生・時刻を確認しただけで埋めません。
- 各行の `content` / `speaker` / `boundary`: `review_required`（未確認）、`pass`、`fail`、`not_applicable`。`fail` / `not_applicable` には `note` で理由を記入します。内容・境界は `pass` が必要です。
- `corrections`: 聴取で確認した変更だけを記入。例: `{"text": "聴いて確認した本文", "speakers": ["A"], "start": 40.2, "end": 45.1}`。変更しない項目は省略し、空のままでも構いません。

行のトップレベルにある `text`・`speakers`・`start`・`end` は照合用の元値なので変更しません。話者IDは `speaker_names` にあるものを使います。帰属を必要としない内容なら `speaker` を理由付き `not_applicable` にできます。短い発言でも、否定・訂正・重要な補足なら内容確認が必要です。

判定・修正する行は、元の範囲と修正後の範囲を両方とも全て聴く必要があります。一部分しか聴いていない行全体を通過にしません。ASRの一致も合格根拠にはなりません。会話で受け取った訂正も、報告された対象版・聴取範囲・訂正内容だけを転記し、未報告の判定を代作しません。

## 確認票を反映する

確認票と同じ版の `transcript.json` を指定し、未作成の出力先を使います。

```sh
python3 scripts/apply_scene_review.py \
  runs/my-episode/result/transcript.json \
  review.json \
  runs/my-episode-reviewed
```

| 出力 | 用途 |
|---|---|
| `transcript-reviewed.json` | 未確認行を含む全体。修正前の行、修正内容、確認者、日時を保持 |
| `ready-segments.json` | 明示的に確認した部分集合。匿名用／帰属付き用を区別 |
| `review.json` | 受領した確認票そのもの |
| `application.json` | 元ファイルのSHA-256、対象artifact ID、適用件数 |

対象版・音声hash・行の元値・確認者・日時・聴取範囲を照合し、別版への誤適用、不正な話者ID・時刻、聴取範囲外の判定を拒否します。手動修正後の文字に元の細かな時刻を流用したとは主張しません。

`ready_for_anonymous_summary=true` は内容・境界が通過し、話者が通過または理由付き対象外の行です。`ready_for_attributed_summary=true` はさらに単一話者の帰属が通過した行です。これは人の申告を検証した状態であり、実際に聴いたことをプログラムが証明するものではありません。未確認・不合格の行はこの部分集合に入りません。

一部が通過しても全文を合格にしません。独立クリップのA/Bは連結せず、入力hash・秒範囲・確認記録を出典として保持してください。識別情報のない古い票を検査なしで移し替えません。要約やシーンの順位付け自体は後段で行います。
