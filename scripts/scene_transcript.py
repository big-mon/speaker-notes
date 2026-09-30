"""Export source material for podcast scenes without generating or rewriting prose.

The pipeline supplies aligned Qwen text and an Apple comparison. Diarization
proposes a main speaker, never ownership of every word in overlapping speech.
Review is separate.
"""
import hashlib
import json
import math
from pathlib import Path


def stamp(seconds):
    milliseconds = round(seconds * 1000)
    return f'{milliseconds//60000:02}:{milliseconds//1000%60:02}.{milliseconds%1000:03}'


def scene_artifact_id(document):
    """Identify the immutable review subject, excluding mutable run timing."""
    keys = ('input', 'source_window', 'segments', 'speaker_mapping', 'speaker_names',
            'raw_sources', 'rules', 'review_audio', 'comparison')
    subject = {key: document.get(key) for key in keys}
    data = json.dumps(subject, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode()
    return 'sha256:' + hashlib.sha256(data).hexdigest()


def source_material(document):
    """Compact views of the same artifact; never infer correctness from agreement.

    IDs are copied from the full transcript and scoped to its artifact ID. This
    intentionally does not align IDs between independent processing attempts.
    """
    artifact_id = document['artifact_id']
    rows = document['segments']
    differences = document.get('comparison', {}).get('differences', [])
    row_ids, records, difference_rows = set(), [], [[] for _ in differences]
    optional_fields = ('playback_start', 'playback_end', 'time_kind', 'timing_provenance',
                       'attribution_status', 'source_raw_span', 'apple_raw_span',
                       'source_chunk_extent', 'alignment_part_indices',
                       'quality', 'review_reasons', 'flags', 'human_verified',
                       'ready_for_anonymous_summary', 'ready_for_attributed_summary')
    for row in rows:
        row_id = row['id']
        if not isinstance(row_id, str) or not row_id or row_id in row_ids:
            raise ValueError('Segment IDs must be unique nonempty strings within an artifact')
        row_ids.add(row_id)
        if not isinstance(row['text'], str):
            raise ValueError('Segment text must be a string')
        if not all(type(row[k]) in (int, float) and math.isfinite(row[k]) for k in ('start', 'end')):
            raise ValueError('Segment timestamps must be finite numbers')
        if not 0 <= row['start'] <= row['end']:
            raise ValueError('Segment timestamps must be nonnegative and ordered')
        if row['state'] not in ('single', 'mixed', 'unknown'):
            raise ValueError('Unexpected segment speaker state')
        if any(s not in document['speaker_names'] for s in row['speakers']):
            raise ValueError('Segment references an unknown speaker label')
        difference_ids = row.get('asr_difference_ids', [])
        for index in difference_ids:
            if type(index) is not int or not 0 <= index < len(differences):
                raise ValueError('Segment references an unavailable ASR difference')
            difference_rows[index].append(row_id)
        record = {key: row[key] for key in ('id', 'start', 'end', 'text', 'speakers', 'state')}
        record.update(artifact_id=artifact_id,
                      asr_difference_ids=difference_ids,
                      **{key: row[key] for key in optional_fields if key in row})
        records.append(record)
    comparisons = [{'id': index, 'artifact_id': artifact_id,
                    'segment_ids': difference_rows[index], 'difference': difference}
                   for index, difference in enumerate(differences)]
    provenance = document.get('provenance', {})
    models = {}
    for name, model in provenance.get('models', {}).items():
        recorded = model.get('recorded', {})
        models[name] = {key: recorded[key] for key in
                        ('repo', 'revision', 'license', 'model', 'purpose') if key in recorded}
        models[name]['manifest'] = model.get('manifest')
        models[name]['full_metadata_pointer'] = '/provenance/models/' + name
    metadata = {
        'schema_version': 1, 'artifact_id': artifact_id,
        'purpose': 'Source material for candidate summaries and valuable scenes; no summary generated',
        'input': document.get('input'),
        'source_window': document['source_window'],
        'requested_window': document.get('requested_window'),
        'context_conditions': document.get('context_conditions'),
        'asr': document.get('asr'), 'models': models,
        'settings': provenance.get('settings'),
        'speaker_names': document['speaker_names'],
        'speaker_identity_scope': document.get('speaker_identity_scope'),
        'time_basis': 'segments start/end and playback_start/end are seconds in the original audio',
        'segment_id_scope': 'artifact_id + id; not matched across independent runs',
        'text_rule': 'segments.jsonl text is copied verbatim from transcript.json; no rewrite or deduplication',
        'comparison_rule': 'ASR differences are hypotheses for audio review, not corrections or ground truth',
        'comparison_method': document.get('comparison', {}).get('method'),
        'comparison_time_basis': 'ASR difference native time envelopes are clip-local; not Qwen word times',
        'counts': {'segments': len(records), 'asr_differences': len(comparisons)},
        'files': {'segments': 'segments.jsonl', 'asr_differences': 'asr-differences.jsonl',
                  'full_transcript': 'transcript.json'},
        'full_metadata_pointers': {'raw_sources': '/raw_sources', 'provenance': '/provenance',
                                   'qwen_settings': '/qwen_settings',
                                   'diarization': '/diarization_metadata', 'review': '/review'},
        'quality': {'accuracy_verified': document.get('validation', {}).get('accuracy_verified', False),
                    'ready_for_attributed_summary': document.get('ready_for_attributed_summary', False)},
    }
    return metadata, records, comparisons


SOURCE_MATERIAL_README = '''# 要約・価値あるシーン抽出の元資料

まず `source-material.json` で入力・処理範囲・モデルと設定を確認し、
`segments.jsonl` を先頭から読みます。1行が1区間で、`text` は認識原文を
書き換えずに保持しています。前後の行も読み、主張・理由・具体例・否定を
一緒に扱ってください。この出力自体は要約や完成した引用ではありません。

- シーン候補には `artifact_id` と区間の `id`、開始・終了秒を添えます。
  IDは同じ成果物内で共通です。別の実行や切り出しのID・話者A/Bとは対応しません。
- `start` / `end` は元音声上の秒です。追加の文脈が含まれる場合は
  `requested_window` と `source_window` を区別します。細かな語境界の正解を保証しません。
- `speakers` は主発言の候補です。`unknown` は不明、`mixed` は他の話者の
  発言が混ざる可能性を示します。匿名の候補抽出には使えますが、人物への断定的な
  帰属には音声確認が必要です。短い応答や重なり発話を補って創作しません。
- `review_reasons`・`flags`・`quality` は未確認点です。判定保留を精度不足の
  確率と解釈せず、重要な語句・数字・否定や引用の候補を音声に戻って確認します。
- `asr_difference_ids` は `asr-differences.jsonl` の `id` を参照します。
  AppleとQwenの差分は別の読みで、正解や自動訂正ではありません。
  差分のApple時刻は切り出し内の元テキスト単位の範囲であり、Qwenの語時刻に転用しません。

`transcript.md` / `transcript.txt` は読むための表示、`transcript.json` は
原出力への参照・全モデル設定・時刻判定の根拠を含む詳細版です。
`source-material.json` の `full_metadata_pointers` は詳細版内のJSON Pointerです。
原出力と人による修正を別保存し、全文処理の成功を全文の聴取確認と扱いません。
'''


def readable_exports(document):
    contents = {}
    entries=[]
    for row in document['segments']:
        label=' / '.join(document['speaker_names'][s] for s in row['speakers']) or '話者不明'
        if row['speakers']:
            label+='（主発言の候補）'
        if row['state'] == 'mixed':
            label+='［別話者の発言が混ざる可能性あり］'
        entries.append(f'[{stamp(row["start"])}–{stamp(row["end"])}] {label}\n{row["text"]}')
    text='\n\n'.join(entries)+'\n'
    contents['transcript.txt'] = text
    source_note = document.get('text_source_note', 'Apple原文を保持し、Qwenの表現で置き換えていません。')
    contents['transcript.md'] = '# シーン抽出用の確認素材\n\n未校正。音声の意味・主要話者・境界は要確認。'+source_note+'\n\n'+text
    contents['README.md'] = SOURCE_MATERIAL_README
    return contents


def save(document, output):
    output=Path(output)
    if output.exists():
        raise FileExistsError(output)
    document['artifact_id'] = scene_artifact_id(document)
    metadata, records, comparisons = source_material(document)
    # Validate all JSON before creating a partial bundle. I/O failures still
    # remain visible to the caller and are recorded by the pipeline.
    json_files = {'transcript.json': document, 'source-material.json': metadata}
    contents = {name: json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
                for name, value in json_files.items()}
    for name, values in (('segments.jsonl', records), ('asr-differences.jsonl', comparisons)):
        contents[name] = ''.join(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n' for value in values)
    contents.update(readable_exports(document))
    output.mkdir(parents=True,exist_ok=False)
    for name, content in contents.items():
        (output/name).write_text(content, encoding='utf-8')
