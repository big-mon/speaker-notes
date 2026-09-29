"""Export source material for podcast scenes without generating or rewriting prose.

The default pipeline supplies aligned Qwen text and an Apple comparison. The
legacy assemble() path below retains Apple text. Diarization proposes a main
speaker, never ownership of every word in overlapping speech. Review is separate.
"""
import argparse
import bisect
import hashlib
import json
import math
import unicodedata
from pathlib import Path

from merge import sha256, stamp
from text_anchor_audit import audit, native_result_boundaries, supported_sentence_boundaries
from turn_candidates import build_candidates


def union_seconds(intervals):
    total, end = 0., -math.inf
    for a, b in sorted(intervals):
        total += max(0., b - max(a, end))
        end = max(end, b)
    return total


def nearby_handover(activity, second, tolerance=1.5):
    """Match a text proposal to sustained speaker activity, not to a word time."""
    candidates = [c for c in activity['candidates'] if c['kind'] == 'sustained_activity']
    matches = []
    for left in candidates:
        for right in candidates:
            if (left['speaker'] == right['speaker'] or right['start'] <= left['start'] or
                    right['end'] <= left['end'] or abs(left['end'] - right['start']) > 3):
                continue
            midpoint = (left['end'] + right['start']) / 2
            if abs(midpoint - second) <= tolerance:
                matches.append({'from': left['speaker'], 'to': right['speaker'],
                                'activity_boundary_range': sorted([left['end'], right['start']]),
                                'candidate_ids': [left['id'], right['id']],
                                'tolerance_seconds': tolerance,
                                'acoustically_verified': False})
    return matches


def assemble(apple, qwen_text, diarization, duration, *, source_offset=0., critical_terms=()):
    if not math.isfinite(duration) or duration <= 0 or not math.isfinite(source_offset) or source_offset < 0:
        raise ValueError('Invalid audio extent')
    comparison = audit(apple, qwen_text, critical_terms=critical_terms)
    proposals = supported_sentence_boundaries(apple, qwen_text, include_native_fallback=True)
    activity = build_candidates(diarization['segments'], end=duration)
    flat, cursor = [], 0
    for si, segment in enumerate(apple):
        source_units=segment.get('units',[{'text':segment['text'],'start':None,'end':None}])
        for ui, unit in enumerate(source_units):
            flat.append(dict(unit, raw_start=cursor, raw_end=cursor+len(unit['text']),
                             asr_segment=si, asr_unit=ui))
            cursor += len(unit['text'])
    raw_text = ''.join(u['text'] for u in flat)
    if not raw_text.strip():
        raise ValueError('Apple transcript is empty; secondary text cannot silently replace it')
    assert raw_text == comparison['apple_text']
    timed=[u for u in flat if all(isinstance(u.get(k),(int,float)) and math.isfinite(u[k])
                                 for k in ('start','end'))]
    monotonic=all(r['start'] >= l['start'] and r['end'] >= l['end'] for l,r in zip(timed,timed[1:]))
    ends = [u['raw_end'] for u in flat]
    cuts = {len(flat)}
    cut_provenance = {}
    selected_proposals = []
    deferred_proposals = []
    for proposal in proposals:
        if proposal.get('requires_diarization_handover'):
            handovers = nearby_handover(activity, proposal['candidate_apple_unit_end'])
            # A nearby speaker change cannot determine which side owns an
            # unmatched suffix. Keep both texts and the hypothesis for review,
            # but do not split a sentence through disputed Apple words.
            deferred_proposals.append(dict(proposal, supporting_activity_handovers=handovers,
                                           selection='deferred; diarization_does_not_resolve_unmatched_text'))
            continue
        selected_proposals.append(proposal)
    if not any(p['apple_last_matched_raw_span'][1] < len(raw_text) for p in selected_proposals):
        selected_proposals = native_result_boundaries(apple)
    for proposal in selected_proposals:
        # The matched character may be inside a multi-character native unit.
        # Cut after that whole unit; never invent an intra-unit time.
        index = bisect.bisect_left(ends, proposal['apple_last_matched_raw_span'][1])+1
        while index < len(flat) and flat[index]['text'] and all(
                c.isspace() or unicodedata.category(c).startswith('P') for c in flat[index]['text']):
            index += 1
        if 0 < index <= len(flat):
            cuts.add(index)
            cut_provenance[index] = proposal
    # Unsupported text stays in a larger block. Mixed major speakers in that
    # block become unknown, not tiny word fragments.
    previous, rows = 0, []
    ids = {}
    for event in sorted(diarization['segments'], key=lambda s: (s['start'], s['end'])):
        ids.setdefault(str(event['speaker']), chr(65+len(ids)))
    for stop in sorted(cuts):
        units = flat[previous:stop]
        if not units:
            continue
        valid = all(isinstance(u.get(k), (int, float)) and math.isfinite(u[k])
                    for u in units for k in ('start', 'end'))
        # A selected native unit may straddle a review window. Preserve its
        # original extent and whole text; clip only the displayed/playback
        # envelope. A wholly outside or reversed unit remains invalid.
        valid = valid and monotonic and all(u['start'] <= u['end'] and
                    ((u['end'] > 0 and u['start'] < duration) or
                     (u['start'] == u['end'] and 0 <= u['start'] <= duration)) for u in units)
        valid = valid and all(r['start'] >= l['start'] and r['end'] >= l['end']
                              for l,r in zip(units,units[1:]))
        native_envelope = None
        if valid:
            a, b = min(u['start'] for u in units), max(u['end'] for u in units)
            native_envelope = [a, b]
            valid = a < b
        edge_clipped = valid and (a < 0 or b > duration)
        if not valid:
            a, b = 0., duration
        else:
            a, b = max(0., a), min(duration, b)
        support = {}
        for candidate in activity['candidates']:
            x, y = max(a, candidate['start']), min(b, candidate['end'])
            if y > x:
                support.setdefault(candidate['speaker'], []).append((x,y))
        seconds = {s: union_seconds(spans) for s, spans in support.items()}
        ranking = sorted(seconds, key=seconds.get, reverse=True)
        main = ranking[0] if ranking else None
        coverage = seconds.get(main, 0.) / max(b-a, 1e-9)
        competing = [s for s in ranking[1:] if seconds[s] >= 2.5 or
                     (seconds[s] >= 1 and seconds[s]/(b-a) >= .2)]
        assigned = valid and b-a >= 2.5 and main is not None and coverage >= .65 and not competing
        overlap_events = [e for e in activity['raw_events'] if e['start'] < b and e['end'] > a
                          and (not assigned or e['speaker'] != main)]
        differences = [i for i,d in enumerate(comparison['differences'])
                       if (d['apple_raw_span'][0] < units[-1]['raw_end']
                           or (stop == len(flat) and d['apple_raw_span'][0] == len(raw_text)))
                       and d['apple_raw_span'][1] >= units[0]['raw_start']]
        reasons = ['human_content_speaker_and_boundary_review_pending']
        if not assigned:
            reasons.append('major_speaker_unresolved')
        if b-a < 2.5:
            reasons.append('brief_text_not_inherited_from_surrounding_main_speaker')
        if overlap_events:
            reasons.append('other_speech_not_separated_into_text')
        if differences:
            reasons.append('asr_texts_differ')
        if any('potential_substantive_omission' in comparison['differences'][i].get('review_signals', [])
               for i in differences):
            reasons.append('long_secondary_only_passage; possible_omission_requires_audio_review')
        if not valid:
            reasons.append('native_timing_unavailable; playback_is_whole_input_extent')
        if edge_clipped:
            reasons.append('whole_native_unit_crosses_window_edge; displayed_envelope_clipped_not_text')
        point_units = [u for u in units if u.get('start') is not None and u.get('start') == u.get('end')]
        sparse_units = [u for u in units if isinstance(u.get('start'), (int, float)) and
                        isinstance(u.get('end'), (int, float)) and len(u['text'].strip()) <= 2 and
                        u['end'] - u['start'] > 2.5]
        if point_units:
            reasons.append('native_zero_duration_units; point_times_not_word_intervals')
        if sparse_units:
            reasons.append('long_native_time_for_very_short_text; possible_recognition_gap_requires_audio_review')
        if b-a > 45:
            reasons.append('long_block_without_supported_sentence_boundary')
        boundary_proposals = [cut_provenance[i] for i in (previous, stop) if i in cut_provenance]
        if any(p.get('requires_diarization_handover') for p in boundary_proposals):
            reasons.append('text_boundary_has_bounded_asr_difference; inspect_adjacent_rows_together')
        if any(p.get('boundary_support') == 'native_asr_result_boundary' for p in boundary_proposals):
            reasons.append('native_result_group_boundary; not_verified_sentence_or_speaker_boundary')
        if stop == len(flat):
            reasons.append('input_end_may_cut_continuing_context')
        rows.append({
            'id': str(len(rows)), 'start': a+source_offset, 'end': b+source_offset,
            'text': ''.join(u['text'] for u in units),
            'speakers': [ids[main]] if assigned else [],
            'state': ('mixed' if overlap_events else 'single') if assigned else 'unknown',
            'attribution_status': 'main_speaker_candidate_not_word_ownership' if assigned else 'unknown',
            'playback_start': max(source_offset, a+source_offset-2),
            'playback_end': min(source_offset+duration, b+source_offset+2),
            'time_kind': ('native_apple_unit_envelope_clipped_to_window' if edge_clipped else
                          'native_apple_unit_envelope') if valid else 'input_extent',
            'native_unit_envelope_clip_local': native_envelope,
            'unit_start': previous, 'unit_end': stop,
            'apple_raw_span': [units[0]['raw_start'],units[-1]['raw_end']],
            'source_unit_references': [[u['asr_segment'],u['asr_unit']] for u in units],
            'candidate_support_seconds': {ids[s]: n for s,n in seconds.items()},
            'overlap_event_ids': [e['id'] for e in overlap_events],
            'asr_difference_ids': differences,
            'end_proposal': cut_provenance.get(stop),
            'boundary_proposals': boundary_proposals,
            'timing_review_signals': {
                'point_unit_references': [[u['asr_segment'], u['asr_unit']] for u in point_units],
                'sparse_unit_references': [[u['asr_segment'], u['asr_unit']] for u in sparse_units],
                'sparse_rule': 'at most 2 stripped characters over more than 2.5 seconds; diagnostic only',
            },
            'review_reasons': reasons,
            'quality': {k: 'review_required' for k in ('content','speaker','boundary')},
            'ready_for_attributed_summary': False,
        })
        previous = stop
    assert ''.join(r['text'] for r in rows) == raw_text
    return {'schema_version':1,
            'purpose':'reviewable input for later summary and scene extraction; no summary generated',
            'asr':{'engine':'Apple SpeechAnalyzer / SpeechTranscriber','text_modified':False,
                   'secondary':'Qwen3-ASR-1.7B BF16; text audit and proposed sentence cuts only'},
            'speaker_names':{s:'話者'+s for s in ids.values()}, 'speaker_mapping':ids,
            'speaker_identity_scope':'one diarization invocation; IDs not matched across independent clips',
            'source_window':{'start':source_offset,'end':source_offset+duration},
            'segments':rows, 'units':flat, 'diarization':diarization['segments'],
            'raw_time_origin':'units, diarization, activity and comparison are clip-local; segment/playback times use source clock',
            'activity':activity, 'comparison':comparison,
            'deferred_boundary_proposals': deferred_proposals,
            'rules':{'minimum_main_candidate_coverage':.65,'competing_major_seconds':2.5,
                     'competing_major_fraction':.2,'competing_fraction_minimum_seconds':1.,
                     'competition_rule':'absolute_seconds OR (minimum_seconds AND fraction)',
                     'numbers_are_confidence':False,
                     'bounded_text_difference_handover_tolerance_seconds':1.5,
                     'bounded_apple_text_gap_automatically_selected':False,
                     'minimum_text_extent_for_main_candidate_seconds':2.5,
                     'main_speaker_is_not_word_ownership':True},
            'review':{'human_verified':False,'windows':[],'notes':[]},
            'validation':{'asr_text_preserved':True,'unit_count':len(flat),
                          'row_count':len(rows),'accuracy_verified':False},
            'ready_for_attributed_summary':False}


def scene_artifact_id(document):
    """Identify the immutable review subject, excluding mutable run timing."""
    keys = ('input', 'source_window', 'segments', 'speaker_mapping', 'speaker_names',
            'raw_sources', 'rules', 'review_audio')
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
    contents['review.html'] = REVIEW_HTML.replace('__DATA__',json.dumps(document,ensure_ascii=False,allow_nan=False).replace('<','\\u003c'))
    contents['README.md'] = SOURCE_MATERIAL_README
    output.mkdir(parents=True,exist_ok=False)
    for name, content in contents.items():
        (output/name).write_text(content, encoding='utf-8')


REVIEW_HTML='''<!doctype html><html lang="ja"><meta charset="utf-8"><title>音声処理の品質確認</title>
<style>body{font:16px/1.8 system-ui;max-width:1000px;margin:30px auto;padding:0 20px;background:#f5f6f8;color:#223}article{background:white;padding:20px;margin:16px 0;border-radius:10px}button,select,input{font:inherit}button{cursor:pointer;margin:4px;padding:6px}audio{width:100%}.note{color:#805400}.text{white-space:pre-wrap}.status{font-size:14px;color:#566}summary{cursor:pointer}textarea{width:98%;min-height:70px;font:inherit}</style>
<h1>意味・主な話者・再生位置を確認</h1>
<p class="note">機械出力の確認素材です。話者は主発言の候補で、短い応答・重なりの本文は未分離です。全行が要確認から始まります。</p>
<p id="text-source"></p>
<p>元の音声に切替（任意）: <input id="file" type="file" accept="audio/*"></p><audio id="audio" controls></audio>
<p id="scope"></p><button id="key">最初の確認: 45〜105秒を再生</button>
<p>確認者（必須・呼び名は自由）: <input id="reviewer" placeholder="例: 自分"></p>
<p>実際に聴いた範囲（必須・元音声の秒、1行に開始–終了）。再生した範囲は自動記録されません:</p>
<textarea id="listened" placeholder="例: 45-105"></textarea>
<p>意味・話者・境界は行全体を聴いてから判定します。一部だけ聴いた行は未確認のままにしてください。対象外・誤りにはメモを添えてください。</p>
<button id="show-json">確認票JSONを表示</button><button id="download">JSONをダウンロード</button>
<p id="export-status" role="status" aria-live="polite"></p>
<section id="json-fallback" hidden>
<p>ダウンロードできない場合も、下のJSON全体をコピーしてファイルやチャットに貼り付けられます。入力を変更したら、もう一度表示・コピーしてください。</p>
<textarea id="review-json" readonly rows="12" aria-label="確認票JSON"></textarea>
<button id="copy-json">JSONをコピー</button><button id="select-json">JSONをすべて選択</button>
<p>自動コピーできない場合は「JSONをすべて選択」の後に ⌘C（Windowsでは Ctrl+C）でコピーしてください。</p>
</section>
<main id="rows"></main><script>
const data=__DATA__, $=x=>document.getElementById(x), audio=$('audio');let url, stop, audioOffset=0;
$('text-source').textContent=data.text_source_note||'Apple原文を保持し、Qwenの表現で置き換えていません。';
const review={schema_version:1,source_artifact_id:data.artifact_id,source:data.input,source_window:data.source_window,raw_sources:data.raw_sources,rules:data.rules,processing:data.processing,audio:data.review_audio||null,reviewer:'',listened_ranges:[],reviewed_at:null,rows:{}};
$('scope').textContent='処理範囲: '+data.source_window.start.toFixed(2)+'〜'+data.source_window.end.toFixed(2)+'秒。行の時刻は元音声上の位置。プレーヤーは選択中の音声内の時刻です。';
if(data.requested_window&&data.context_conditions?.applied)$('scope').textContent+=' 指定範囲 '+data.requested_window.start.toFixed(2)+'〜'+data.requested_window.end.toFixed(2)+'秒の前後も、文脈を保つために含めています。';
if(data.review_audio){audio.src=data.review_audio.src;audioOffset=data.review_audio.source_offset}
$('file').onchange=async()=>{const f=$('file').files[0];if(!f)return;
 if(f.size!==data.input.bytes){alert('対象の元音声とファイルサイズが異なります');return}
 const digest=await crypto.subtle.digest('SHA-256',await f.arrayBuffer());const hash=Array.from(new Uint8Array(digest)).map(b=>b.toString(16).padStart(2,'0')).join('');
 if(hash!==data.input.sha256){alert('対象の元音声と内容が一致しません');return}
 if(url)URL.revokeObjectURL(url);url=URL.createObjectURL(f);audioOffset=0;audio.src=url;review.audio={source_offset:0,sha256:hash,bytes:f.size};};
function play(a,b){if(!audio.src){alert('元の音声ファイルを選択してください');return}audio.currentTime=Math.max(0,a-audioOffset);stop=b-audioOffset;audio.play()}
audio.ontimeupdate=()=>{if(stop!==undefined&&audio.currentTime>=stop)audio.pause()};
$('key').onclick=()=>play(Math.max(data.source_window.start,45),Math.min(data.source_window.end,105));
if(data.source_window.start>=105||data.source_window.end<=45)$('key').hidden=true;
for(const row of data.segments){
 const el=document.createElement('article'), heading=document.createElement('h2');heading.textContent=row.start.toFixed(2)+'–'+row.end.toFixed(2)+'秒 '+(row.speakers.map(s=>data.speaker_names[s]).join(' / ')||'話者不明')+'（候補）';el.append(heading);
 const p=document.createElement('p');p.className='text';p.textContent=row.text;el.append(p);
 if(row.state==='mixed'){const warning=document.createElement('p');warning.className='note';warning.textContent='別話者の発言が混ざる可能性があります。このラベルは全ての言葉の話者を保証しません。';el.append(warning)}
 const b=document.createElement('button');b.textContent='前後2秒を含めて再生';b.onclick=()=>play(row.playback_start,row.playback_end);el.append(b);
 const detail=document.createElement('details'), summary=document.createElement('summary');summary.textContent='ASR間の差・未確認の理由';detail.append(summary);
 const status=document.createElement('p');status.className='status';status.textContent=row.review_reasons.join(' / ');detail.append(status);
 for(const id of row.asr_difference_ids){const d=data.comparison.differences[id],p=document.createElement('p');p.textContent='Apple: '+(d.apple_text||'∅')+' / Qwen: '+(d.qwen_text||'∅');detail.append(p)}el.append(detail);
 review.rows[row.id]={start:row.start,end:row.end,text:row.text,speakers:row.speakers,content:'review_required',speaker:'review_required',boundary:'review_required',note:''};
 for(const [field,label] of [['content','意味の保持'],['speaker','主な話者'],['boundary','再生位置']]){const l=document.createElement('label');l.textContent=label+' ';const s=document.createElement('select');for(const [v,t]of [['review_required','未確認'],['pass','確認できた'],['fail','誤りがある'],['not_applicable','対象外']])s.add(new Option(t,v));s.onchange=()=>review.rows[row.id][field]=s.value;l.append(s);el.append(l)}
 const note=document.createElement('textarea');note.placeholder='気になる箇所、正しい話者や本文、確認した音声範囲';note.oninput=()=>review.rows[row.id].note=note.value;el.append(note);$('rows').append(el);
 const correction=document.createElement('details'), ctitle=document.createElement('summary');ctitle.textContent='聴いて確認した修正（任意・原出力は保持）';correction.append(ctitle);
 const correctedText=document.createElement('textarea');correctedText.placeholder='修正後の本文（空欄なら変更しません）';correction.append(correctedText);
 const correctedSpeaker=document.createElement('select');correctedSpeaker.add(new Option('話者を変更しない','unchanged'));correctedSpeaker.add(new Option('話者不明','unknown'));for(const [id,name]of Object.entries(data.speaker_names))correctedSpeaker.add(new Option(name,id));correction.append(correctedSpeaker);
 const correctedStart=document.createElement('input'),correctedEnd=document.createElement('input');for(const x of [correctedStart,correctedEnd]){x.type='number';x.step='0.01';correction.append(x)}correctedStart.placeholder='修正後の開始秒';correctedEnd.placeholder='修正後の終了秒';
 function recordCorrection(){const c={};if(correctedText.value)c.text=correctedText.value;if(correctedSpeaker.value!=='unchanged')c.speakers=correctedSpeaker.value==='unknown'?[]:[correctedSpeaker.value];if(correctedStart.value)c.start=Number(correctedStart.value);if(correctedEnd.value)c.end=Number(correctedEnd.value);review.rows[row.id].corrections=c}
 for(const x of [correctedText,correctedSpeaker,correctedStart,correctedEnd])x.addEventListener('input',recordCorrection);el.append(correction);
}
// Review JSON export: always expose the generated text before trying browser APIs.
function exportStatus(message,error=false){const el=$('export-status');el.textContent=message;el.className=error?'note':'status';el.setAttribute('role',error?'alert':'status')}
function showReviewJSON(){
 $('json-fallback').hidden=true;$('review-json').value='';
 const reviewer=$('reviewer').value.trim();
 if(!reviewer){exportStatus('入力不足: 確認者の呼び名を記入してください（例: 自分）。ダウンロードはまだ要求していません。',true);$('reviewer').focus();return null}
 const lines=$('listened').value.trim().split(/\\n/).filter(line=>line.trim()),ranges=[];
 if(!lines.length){exportStatus('入力不足: 実際に聴いた範囲を記入してください（例: 0-180）。再生しただけでは記録されません。',true);$('listened').focus();return null}
 for(const [i,line]of lines.entries()){
  const m=line.trim().match(/^([0-9]+(?:\\.[0-9]+)?)\\s*[-–〜]\\s*([0-9]+(?:\\.[0-9]+)?)$/),a=m?Number(m[1]):NaN,b=m?Number(m[2]):NaN;
  if(!m||!Number.isFinite(a)||!Number.isFinite(b)||a>=b){exportStatus('入力形式のエラー: 聴いた範囲の'+(i+1)+'行目を 0-180 のように開始–終了で記入してください。',true);$('listened').focus();return null}
  if(a<data.source_window.start||b>data.source_window.end){exportStatus('入力範囲のエラー: 聴いた範囲はこの確認素材の '+data.source_window.start+'〜'+data.source_window.end+' 秒以内で記入してください。',true);$('listened').focus();return null}
  ranges.push({start:a,end:b});
 }
 try{review.reviewer=reviewer;review.listened_ranges=ranges;review.reviewed_at=new Date().toISOString();const text=JSON.stringify(review,null,2);$('review-json').value=text;$('json-fallback').hidden=false;exportStatus('確認票JSONを下に表示しました。まだファイルには保存していません。');return text}
 catch(error){exportStatus('JSONの生成に失敗しました: '+String(error.message||error),true);return null}
}
function selectReviewJSON(){$('review-json').focus();$('review-json').select()}
$('show-json').onclick=showReviewJSON;
$('select-json').onclick=()=>{if(showReviewJSON()!==null){selectReviewJSON();exportStatus('JSON全体を選択しました。⌘C（Windowsでは Ctrl+C）でコピーしてください。')}};
$('copy-json').onclick=async()=>{
 const text=showReviewJSON();if(text===null)return;
 try{if(!navigator.clipboard?.writeText)throw new Error('この環境では自動コピーを利用できません');await navigator.clipboard.writeText(text);exportStatus('確認票JSONをクリップボードにコピーしました。ファイル保存は行っていません。')}
 catch(error){selectReviewJSON();exportStatus('自動コピーできませんでした。下のJSONを選択しています。⌘C（Windowsでは Ctrl+C）でコピーしてください。詳細: '+String(error.message||error),true)}
};
$('download').onclick=()=>{
 const text=showReviewJSON();if(text===null)return;let u,a;
 try{a=document.createElement('a');if(!('download' in a)||typeof URL.createObjectURL!=='function')throw new Error('この環境ではダウンロード機能を利用できません');u=URL.createObjectURL(new Blob([text],{type:'application/json;charset=utf-8'}));a.href=u;a.download='scene-quality-review.json';a.hidden=true;document.body.append(a);a.click();exportStatus('ダウンロードを要求しました。保存の完了はこの画面から確認できません。ファイルが見当たらない場合は、下のJSONをコピーしてください。')}
 catch(error){exportStatus('ダウンロードを開始できませんでした。入力済みの確認票JSONは下に表示しています。コピーして取り出してください。詳細: '+String(error.message||error),true)}
 finally{if(a)a.remove();if(u)setTimeout(()=>URL.revokeObjectURL(u),60000)}
};
</script></html>'''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('apple','qwen','diarization','input','output'):
        p.add_argument(name,type=Path)
    p.add_argument('--duration',type=float,required=True)
    p.add_argument('--source-offset',type=float,default=0)
    p.add_argument('--critical-term',action='append',default=[])
    args=p.parse_args()
    document=assemble(json.loads(args.apple.read_text()),args.qwen.read_text(),
                      json.loads(args.diarization.read_text()),args.duration,
                      source_offset=args.source_offset,critical_terms=args.critical_term)
    document['input']={'path':str(args.input.resolve()),'sha256':sha256(args.input),'bytes':args.input.stat().st_size}
    document['raw_sources']={name:{'path':str(path.resolve()),'sha256':sha256(path)}
                             for name,path in [('apple',args.apple),('qwen',args.qwen),('diarization',args.diarization)]}
    save(document,args.output)


if __name__=='__main__':
    main()
