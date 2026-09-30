#!/usr/bin/env python3
"""Read-only structural validation of a completed Qwen scene-pipeline run.

Checks preservation and internal consistency, not recognition accuracy, missing
spoken words, speaker correctness or acoustic timestamp accuracy. No inference.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

from scene_pipeline import read_float_wav
from scene_transcript import scene_artifact_id, source_material, readable_exports


def read_json(path):
    def reject(value):
        raise ValueError(f'Nonfinite JSON number: {value}')
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=reject)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_matches(path, recorded):
    path = Path(path)
    require(Path(recorded['path']).resolve() == path.resolve(), f'Recorded file path mismatch: {path}')
    require(path.stat().st_size == recorded['bytes'], f'File size mismatch: {path}')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    require(digest == recorded['sha256'], f'File SHA256 mismatch: {path}')


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def quality_indicators(document, manifest):
    """Expose potentially unhelpful ranges without treating them as corruption."""
    rows = document['segments']
    chunks = manifest['qwen_chunks']
    rate = manifest['sample_rate']
    chunk_durations = [(c['end_sample'] - c['start_sample']) / rate for c in chunks]
    longest_chunk = max(chunk_durations)
    longest_row = max((r['end'] - r['start'] for r in rows), default=0)
    window = document['source_window']
    fallback = [r['id'] for r in rows if 'broad_playback_timing_fallback' in r.get('flags', [])]
    whole_window = [r['id'] for r in rows if r['start'] == window['start'] and r['end'] == window['end']]
    multiple_chunks = [r['id'] for r in rows if len(set(r.get('alignment_part_indices', []))) > 1]
    overlong = [r['id'] for r in rows if r['end'] - r['start'] > longest_chunk + 1e-7]
    vad_fallbacks = [c['index'] for c in chunks if c.get('boundary_kind') == 'forced_low_energy_no_vad_pause']
    indicators = {
        'counts_are_not_accuracy': True,
        'time_kind_counts': dict(Counter(r.get('time_kind', 'unspecified') for r in rows)),
        'timing_provenance_counts': dict(Counter(r.get('timing_provenance', r.get('time_kind', 'unspecified')) for r in rows)),
        'speaker_state_counts': {state: sum(r['state'] == state for r in rows) for state in ('single', 'mixed', 'unknown')},
        'max_row_duration_seconds': longest_row,
        'max_row_duration_ids': [r['id'] for r in rows if r['end'] - r['start'] == longest_row],
        'max_row_text_characters': max((len(r['text']) for r in rows), default=0),
        'broad_timing_fallback': {'count': len(fallback), 'row_ids': fallback},
        'whole_source_window_timing': {'count': len(whole_window), 'row_ids': whole_window},
        'rows_covering_multiple_chunks': {'count': len(multiple_chunks), 'row_ids': multiple_chunks},
        'rows_longer_than_longest_input_chunk': {'threshold_seconds': longest_chunk, 'count': len(overlong), 'row_ids': overlong},
        'chunks': {'min_duration_seconds': min(chunk_durations), 'max_duration_seconds': longest_chunk,
                   'boundary_kind_counts': dict(Counter(c.get('boundary_kind', 'unspecified') for c in chunks)),
                   'vad_fallback_boundaries': {'count': len(vad_fallbacks), 'chunk_indices': vad_fallbacks}},
    }
    warnings = []
    if fallback:
        warnings.append(f'{len(fallback)} rows use fallback timing; inspect their timing provenance before selecting scenes')
    if overlong:
        warnings.append(f'{len(overlong)} rows have ranges longer than the longest ASR chunk; audio lookup may be unhelpful')
    if multiple_chunks:
        warnings.append(f'{len(multiple_chunks)} rows contain text from multiple ASR chunks; inspect boundaries and context')
    if vad_fallbacks:
        warnings.append(f'{len(vad_fallbacks)} input boundaries had no qualifying VAD pause; speech may be cut')
    return indicators, warnings


def validate(run):
    """Return a report even on failure; stop at the first inconsistent stage."""
    run = Path(run).resolve()
    report = {'schema_version': 1, 'run': str(run), 'status': 'failed',
              'scope': 'structural preservation only; not acoustic accuracy or full listening',
              'accuracy_verified': False, 'checks': [], 'warnings': [], 'errors': []}
    check = 'completed_run'
    try:
        processing = read_json(run / 'processing.json')
        require(processing['status'] == 'complete', 'Run is not complete')
        require(processing['stages'] and all(s['status'] == 'complete' for s in processing['stages']),
                'Not all recorded stages completed')
        document = read_json(run / 'result/transcript.json')
        provenance = read_json(run / 'provenance.json')
        require(provenance['settings']['transcript_mode'] == 'qwen-aligned',
                'This validator supports the Qwen-aligned production route')
        require(document['input'] == provenance['input'], 'Input provenance differs')
        require(document['artifact_id'] == scene_artifact_id(document), 'Transcript artifact ID mismatch')
        report['artifact_id'] = document['artifact_id']
        report['checks'].append(check)

        check = 'source_input_identity'
        original = Path(provenance['input']['path'])
        if original.is_file():
            file_matches(original, provenance['input'])
            report['checks'].append(check)
            report['source_input_sha256'] = 'matched'
        else:
            report['source_input_sha256'] = 'not_checked_input_unavailable'
            report['warnings'].append('Original input is unavailable; its identity was not rechecked')

        check = 'raw_source_fingerprints'
        raw_files = {'apple': 'apple/transcript-segments.json', 'diarization': 'diarization/pass-1.json',
                     'qwen': 'qwen-comparison.json', 'audio_manifest': 'audio-manifest.json',
                     'alignment': 'alignment.json'}
        require(set(document['raw_sources']) == set(raw_files), 'Unexpected or missing raw-source references')
        for name, relative in raw_files.items():
            file_matches(run / relative, document['raw_sources'][name])
        report['checks'].append(check)

        check = 'selected_audio_preservation'
        manifest = read_json(run / 'audio-manifest.json')
        for name, relative in (('normalized', 'normalized.wav'), ('clip', 'audio.wav')):
            file_matches(run / relative, manifest[name])
        rate, pcm = read_float_wav(run / 'audio.wav')
        original_rate, normalized = read_float_wav(run / 'normalized.wav')
        window = manifest['window']
        a, b = window['start_sample'], window['end_sample']
        require(type(a) is int and type(b) is int and 0 <= a < b <= len(normalized) // 4,
                'Selected sample range is invalid')
        require(rate == original_rate == manifest['sample_rate'], 'Sample rates differ')
        require(pcm == normalized[a * 4:b * 4], 'Selected audio differs from the normalized source range')
        source_start, source_end = a / rate, b / rate
        require(window['source_offset'] == source_start and window['duration'] == (b - a) / rate,
                'Window times differ from sample positions')
        require(document['source_window'] == {'start': source_start, 'end': source_end},
                'Transcript window differs from selected audio')
        require(abs(manifest['source_duration'] - len(normalized) / (4 * rate)) < 1e-7,
                'Normalized source duration mismatch')
        del normalized
        report['checks'].append(check)

        check = 'vad_evidence_and_chunk_plan'
        plan_path, vad_path = run/'vad-chunk-plan.json', run/'vad/frames.json'
        file_matches(plan_path, manifest['vad_chunk_plan'])
        policy = manifest['qwen_chunk_policy']
        file_matches(vad_path, policy['raw_vad'])
        vad, plan = read_json(vad_path), read_json(plan_path)
        require(Path(vad['file']).resolve() == (run/'audio.wav').resolve()
                and vad['sha256'] == manifest['clip']['sha256'], 'VAD refers to different audio')
        require(vad['sample_rate'] == rate and vad['sample_count'] == len(pcm)//4,
                'VAD sample extent differs from selected audio')
        from vad_chunking import _pause_candidates
        _pause_candidates(vad, len(pcm)//4, rate)
        require(plan['policy'] == policy, 'VAD plan policy differs from manifest')
        # Only these fields are added after the plan is saved; all cut decisions
        # must match, not merely the start/end sample positions.
        planned = [{k: v for k, v in part.items() if k not in ('file', 'source_start', 'source_end')}
                   for part in manifest['qwen_chunks']]
        require(plan['parts'] == planned, 'VAD plan cuts differ from actual chunks')
        report['checks'].append(check)

        check = 'qwen_chunk_coverage_and_raw_text'
        qwen = read_json(run / 'qwen-comparison.json')
        parts = manifest['qwen_chunks']
        require(parts and len(parts) == len(qwen['chunks']), 'Qwen input/output chunk counts differ')
        require(qwen['incomplete'] is False, 'Qwen reported an incomplete result')
        cursor, texts = 0, []
        for index, (part, chunk) in enumerate(zip(parts, qwen['chunks'])):
            require(part['index'] == chunk['index'] == index, 'Chunk indices are not consecutive')
            require(type(part['start_sample']) is int and type(part['end_sample']) is int,
                    'Chunk sample positions must be integers')
            require(part['start_sample'] == cursor < part['end_sample'] <= len(pcm) // 4,
                    'Chunk coverage has a gap, overlap or invalid range')
            require(all(chunk.get(k) == value for k, value in part.items()),
                    'Qwen output chunk metadata differs from its input')
            require(part['source_start'] == source_start + part['start_sample'] / rate and
                    part['source_end'] == source_start + part['end_sample'] / rate,
                    'Chunk source timestamps differ from sample positions')
            audio_path = run / 'qwen-input' / f'{index:03d}.wav'
            file_matches(audio_path, part['file'])
            chunk_rate, samples = read_float_wav(audio_path)
            require(chunk_rate == rate and samples == pcm[cursor * 4:part['end_sample'] * 4],
                    'Chunk PCM differs from the selected source range')
            raw_path = run / 'qwen' / f'{index:03d}.json'
            file_matches(raw_path, chunk['raw_output'])
            raw = read_json(raw_path)
            require(Path(raw['file']).resolve() == Path(part['file']['path']).resolve(),
                    'Raw Qwen result identifies a different input chunk')
            require(isinstance(raw['raw']['text'], str) and raw['raw']['text'] == chunk['text'],
                    'Qwen raw text differs from the composed chunk text')
            require(not raw.get('token_limit_reached') and not chunk['token_limit_reached'],
                    'Qwen hit its token limit')
            cursor = part['end_sample']
            texts.append(chunk['text'])
        require(cursor == len(pcm) // 4, 'Qwen chunks do not cover the complete selected audio')
        text = '\n'.join(texts)
        require((run / 'qwen-comparison.txt').read_text(encoding='utf-8') == text + '\n',
                'Combined Qwen text file differs')
        del pcm
        report['checks'].append(check)

        check = 'alignment_and_segment_text_preservation'
        alignment = read_json(run / 'alignment.json')
        require(alignment.get('kind') == 'derived_composite_of_independent_forced_alignments',
                'Expected production composite alignment with child evidence')
        from compose_alignment import compose_alignments
        children = alignment['raw_children']
        require(len(children) == len(parts), 'Child alignment count differs from ASR chunks')
        for index, child in enumerate(children):
            require(child['start'] == parts[index]['start_sample']/rate
                    and child['end'] == parts[index]['end_sample']/rate,
                    'Child alignment extent differs from its ASR input chunk')
            require(child['alignment']['mapping']['original_text'] == qwen['chunks'][index]['text'],
                    'Child alignment text differs from its ASR chunk')
            path = run / 'alignments' / f'{index:03d}' / 'alignment.json'
            file_matches(path, child['source'])
            require(read_json(path) == child['alignment'], 'Child alignment differs from embedded evidence')
        expected = compose_alignments(children, source_start, window['duration'])
        require(all(alignment[key] == expected[key] for key in
                    ('mapping', 'items', 'composition_parts', 'composition_rules')),
                'Composed alignment differs from its child evidence')
        require(alignment['mapping']['original_text'] == text, 'Alignment changed the Qwen text')
        require(alignment['source_offset_seconds'] == source_start and
                alignment['duration_seconds'] == window['duration'], 'Alignment audio extent differs')
        rows = document['segments']
        require(''.join(row['text'] for row in rows) == text, 'Transcript rows changed the Qwen text')
        cursor = 0
        for row in rows:
            span = row['source_raw_span']
            require(span == [cursor, cursor + len(row['text'])], 'Row text spans are not contiguous')
            cursor = span[1]
        report['checks'].append(check)

        check = 'timestamps_ids_and_references'
        diarization = read_json(run / 'diarization/pass-1.json')
        reference_bounds = {'forced_token_references': len(alignment['items']),
                            'alignment_part_indices': len(alignment['composition_parts']),
                            'raw_diarization_event_ids': len(diarization['segments']),
                            'raw_other_speaker_event_ids': len(diarization['segments'])}
        for row in rows:
            for left, right in (('start', 'end'), ('playback_start', 'playback_end')):
                x, y = row[left], row[right]
                require(finite(x) and finite(y) and source_start <= x <= y <= source_end,
                        f'Invalid original-audio timestamp range in row {row["id"]}')
            for name, count in reference_bounds.items():
                require(all(type(i) is int and 0 <= i < count for i in row[name]),
                        f'Invalid {name} in row {row["id"]}')
        metadata, compact, differences = source_material(document)
        report['checks'].append(check)

        check = 'derived_transcript_from_raw_evidence'
        from review_aligned_scene import make_document
        expected_audio = dict(manifest['clip'], src='../audio.wav', source_offset=source_start)
        rebuilt = make_document(alignment, diarization, read_json(run/'apple/transcript-segments.json'),
                                provenance['input'], document['raw_sources'], expected_audio)
        # make_document owns these fields. Only processing is replaced later by
        # the pipeline's measured stages; compare every other derived field.
        for key, expected in rebuilt.items():
            if key != 'processing':
                require(document.get(key) == expected, f'Derived transcript differs from raw evidence: {key}')
        retained = {'provenance': provenance, 'processing': processing, 'qwen_chunks': qwen,
                    'qwen_settings': read_json(run/'qwen/settings.json'),
                    'diarization_metadata': {k: v for k, v in diarization.items() if k != 'segments'}}
        retained.update({key: manifest[key] for key in
                         ('requested_window', 'actual_window', 'context_conditions')})
        for key, expected in retained.items():
            require(document.get(key) == expected, f'Exported metadata differs from retained evidence: {key}')
        report['checks'].append(check)

        check = 'compact_exports_and_readable_text'
        for relative, expected in (('segments.jsonl', compact), ('asr-differences.jsonl', differences)):
            actual = [json.loads(line) for line in (run / 'result' / relative).read_text(encoding='utf-8').splitlines()]
            require(actual == expected, f'Compact export differs from full transcript: {relative}')
        require(read_json(run / 'result/source-material.json') == metadata,
                'Source material metadata differs from full transcript')
        for name, expected in readable_exports(document).items():
            require((run / 'result' / name).read_text(encoding='utf-8') == expected, f'Readable export differs from full transcript: {name}')
        report['checks'].append(check)
        indicators, warnings = quality_indicators(document, manifest)
        report['quality_indicators'] = indicators
        report['warnings'].extend(warnings)
        report.update(status='passed', counts={'chunks': len(parts), 'segments': len(rows),
                                               'text_characters': len(text), 'selected_samples': b - a},
                      source_window=document['source_window'])
    except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
        report['errors'].append({'check': check, 'type': type(error).__name__, 'message': str(error)})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--report', type=Path, help='Optional new JSON file; existing files are never overwritten')
    args = parser.parse_args()
    report = validate(args.run)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.report:
        try:
            with args.report.open('x', encoding='utf-8') as stream:
                stream.write(rendered)
        except OSError as error:
            report['status'] = 'failed'
            report['errors'].append({'check': 'report_output', 'type': type(error).__name__, 'message': str(error)})
            rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    print(rendered, end='')
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())
