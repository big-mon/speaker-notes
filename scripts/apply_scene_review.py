"""Validate a human review and export separately, preserving model outputs.

Only explicitly reviewed rows can enter ready-segments.json. A review records
a person's assertions; this CLI cannot establish that audio was actually heard.
"""
import argparse
import copy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

FIELDS = ('content', 'speaker', 'boundary')
STATUSES = {'pass', 'fail', 'review_required', 'not_applicable'}
ORIGINAL_FIELDS = ('start', 'end', 'text', 'speakers')


def _time_range(value, window, name):
    if not isinstance(value, dict):
        raise ValueError(f'{name} must be an object')
    a, b = value.get('start'), value.get('end')
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
               for x in (a, b)) or not window['start'] <= a < b <= window['end']:
        raise ValueError(f'{name} must have a valid source-clock range inside source_window')
    return a, b


def _covered(start, end, ranges):
    cursor = start
    for a, b in sorted(ranges):
        if a > cursor + 1e-9:
            break
        cursor = max(cursor, b)
        if cursor >= end - 1e-9:
            return True
    return False


def _speakers(value, known):
    if not isinstance(value, list) or any(not isinstance(x, str) or x not in known for x in value):
        raise ValueError('speakers must contain only known speaker IDs')
    if len(set(value)) != len(value):
        raise ValueError('Duplicate speaker IDs are invalid')
    return value


def apply_review(document, review):
    from scene_transcript import scene_artifact_id
    if not isinstance(document.get('artifact_id'), str) or document['artifact_id'] != scene_artifact_id(document):
        raise ValueError('Source transcript artifact_id does not match its contents')
    if review.get('schema_version') != 1:
        raise ValueError('Expected review schema_version 1')
    if review.get('source_artifact_id') != document['artifact_id']:
        raise ValueError('Review source_artifact_id does not match the source transcript')
    source_hash = document.get('input', {}).get('sha256')
    if not isinstance(source_hash, str) or len(source_hash) != 64 or review.get('source', {}).get('sha256') != source_hash:
        raise ValueError('Review input hash does not match the source transcript')
    reviewer = review.get('reviewer')
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError('A nonempty reviewer is required')
    reviewed_at = review.get('reviewed_at')
    try:
        stamp = datetime.fromisoformat(reviewed_at.replace('Z', '+00:00'))
    except (AttributeError, TypeError, ValueError):
        raise ValueError('reviewed_at must be an ISO8601 date with a timezone') from None
    if stamp.utcoffset() is None:
        raise ValueError('reviewed_at must include a timezone')
    window = document['source_window']
    if not isinstance(review.get('listened_ranges'), list) or not review['listened_ranges']:
        raise ValueError('At least one explicitly declared listened range is required')
    ranges = [_time_range(r, window, 'listened range') for r in review['listened_ranges']]
    rows = review.get('rows')
    if not isinstance(rows, dict):
        raise ValueError('Review rows must be an object keyed by source row ID')
    originals = {r['id']: r for r in document['segments']}
    if len(originals) != len(document['segments']) or any(not isinstance(k, str) for k in originals):
        raise ValueError('Source row IDs must be unique strings')
    if set(rows) - set(originals):
        raise ValueError('Review contains unknown row IDs')
    known = set(document['speaker_names'])
    result = copy.deepcopy(document)
    record = {'source_artifact_id': document['artifact_id'], 'reviewer': reviewer.strip(),
              'reviewed_at': reviewed_at, 'listened_ranges': copy.deepcopy(review['listened_ranges']),
              'audio': copy.deepcopy(review.get('audio')),
              'verification_kind': 'explicit human review record; listening not instrumentally verified',
              'rows_with_decisions': [], 'corrected_row_ids': []}
    changed_text = False
    for row in result['segments']:
        original = originals[row['id']]
        _time_range(original, window, f'source row {row["id"]}')
        _speakers(original['speakers'], known)
        row['quality'] = {field: 'review_required' for field in FIELDS}
        row['ready_for_attributed_summary'] = False
        row['ready_for_anonymous_summary'] = False
        row['human_review'] = None
        if row['id'] not in rows:
            continue
        received = rows[row['id']]
        if not isinstance(received, dict):
            raise ValueError(f'Review row {row["id"]} must be an object')
        for field in ORIGINAL_FIELDS:
            if field not in received or received[field] != original[field]:
                raise ValueError(f'Review row {row["id"]} original {field} does not match')
        statuses = {field: received.get(field) for field in FIELDS}
        if any(not isinstance(value, str) or value not in STATUSES for value in statuses.values()):
            raise ValueError(f'Review row {row["id"]} contains an invalid or missing status')
        note = received.get('note', '')
        if not isinstance(note, str):
            raise ValueError('Review note must be text')
        if any(value in {'fail', 'not_applicable'} for value in statuses.values()) and not note.strip():
            raise ValueError('A reason in note is required for fail or not_applicable')
        corrections = received.get('corrections', {})
        if not isinstance(corrections, dict) or set(corrections) - set(ORIGINAL_FIELDS):
            raise ValueError('corrections may contain only text, speakers, start and end')
        corrected = {field: copy.deepcopy(corrections.get(field, original[field])) for field in ORIGINAL_FIELDS}
        if not isinstance(corrected['text'], str):
            raise ValueError('Corrected text must be a string')
        a, b = _time_range(corrected, window, f'corrected row {row["id"]}')
        _speakers(corrected['speakers'], known)
        decided = any(value != 'review_required' for value in statuses.values())
        if decided or corrections:
            if not _covered(original['start'], original['end'], ranges) or not _covered(a, b, ranges):
                raise ValueError(f'Row {row["id"]} extends outside the declared listened ranges; partial listening cannot approve the whole row')
        if statuses['speaker'] == 'pass' and not corrected['speakers']:
            raise ValueError('speaker pass requires at least one known speaker; use reasoned not_applicable for an anonymous summary')
        if statuses['content'] == 'pass' and not corrected['text'].strip():
            raise ValueError('content pass cannot approve an empty transcript')
        row.update(corrected)
        row['quality'] = statuses
        row['human_review'] = {'reviewer': reviewer.strip(), 'reviewed_at': reviewed_at,
                               'note': note, 'source_artifact_id': document['artifact_id'],
                               'listened_ranges': copy.deepcopy(review['listened_ranges'])}
        row['model_segment'] = {field: copy.deepcopy(original[field]) for field in ORIGINAL_FIELDS}
        row['manual_corrections'] = copy.deepcopy(corrections)
        row['playback_start'] = max(window['start'], a-2)
        row['playback_end'] = min(window['end'], b+2)
        if corrections:
            record['corrected_row_ids'].append(row['id'])
            row['text_and_time_provenance'] = 'manual correction; source units refer to unchanged model text, not corrected-character alignment'
            if 'start' in corrections or 'end' in corrections:
                row['time_kind'] = 'human_corrected_source_range'
        if decided:
            record['rows_with_decisions'].append(row['id'])
        changed_text |= row['text'] != original['text']
        content_ready = statuses['content'] == 'pass' and statuses['boundary'] == 'pass'
        row['ready_for_anonymous_summary'] = content_ready and (
            statuses['speaker'] == 'pass' or statuses['speaker'] == 'not_applicable')
        row['ready_for_attributed_summary'] = content_ready and statuses['speaker'] == 'pass' and len(row['speakers']) == 1
        if statuses['speaker'] == 'pass':
            mixed = (row.get('overlap_event_ids') or row.get('raw_overlap_intervals')
                     or original.get('state') == 'mixed')
            row['state'] = ('mixed' if mixed else 'single') if len(row['speakers']) == 1 else 'overlap'
            row['attribution_status'] = 'human_reviewed' if len(row['speakers']) == 1 else 'human_reviewed_multiple_speakers_not_individual_word_ownership'
        elif statuses['speaker'] == 'not_applicable':
            row['candidate_speakers'] = row['speakers']
            row['speakers'] = []
            row['state'] = 'unknown'
            row['attribution_status'] = 'not_applicable_to_anonymous_summary'

    result['manual_review'] = record
    result['ready_for_attributed_summary'] = bool(result['segments']) and all(r['ready_for_attributed_summary'] for r in result['segments'])
    result['ready_for_anonymous_summary'] = bool(result['segments']) and all(r['ready_for_anonymous_summary'] for r in result['segments'])
    result['validation'] = dict(result.get('validation', {}),
                                asr_text_preserved=not changed_text,
                                model_raw_outputs_preserved=True,
                                manual_review_applied=True,
                                full_audio_accuracy_verified=False)
    result['asr'] = dict(result.get('asr', {}), text_modified=changed_text,
                         raw_units_and_comparison_unchanged=True)
    result['artifact_id'] = scene_artifact_id(result)
    ready = {'schema_version': 1, 'source_artifact_id': document['artifact_id'],
             'reviewed_artifact_id': result['artifact_id'], 'input': copy.deepcopy(result['input']),
             'source_window': copy.deepcopy(window), 'speaker_names': copy.deepcopy(result['speaker_names']),
             'speaker_mapping': copy.deepcopy(result.get('speaker_mapping', {})),
             'speaker_identity_scope': result.get('speaker_identity_scope', 'source artifact only; no cross-clip speaker matching'),
             'raw_sources': copy.deepcopy(result.get('raw_sources', {})),
             'review_audio': copy.deepcopy(result.get('review_audio')),
             'reviewed_transcript_file': 'transcript-reviewed.json',
             'review': copy.deepcopy(record),
             'segments': [copy.deepcopy(r) for r in result['segments'] if r['ready_for_anonymous_summary']],
             'scope': 'only explicitly passed rows; not proof of whole-episode accuracy',
             'whole_document_ready_for_attributed_summary': result['ready_for_attributed_summary'],
             'whole_document_ready_for_anonymous_summary': result['ready_for_anonymous_summary']}
    return result, ready


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('transcript', type=Path)
    parser.add_argument('review', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    original_bytes, review_bytes = args.transcript.read_bytes(), args.review.read_bytes()
    document, ready = apply_review(json.loads(original_bytes), json.loads(review_bytes))
    args.output.mkdir(parents=True, exist_ok=False)
    for filename, value in [('transcript-reviewed.json', document), ('ready-segments.json', ready)]:
        (args.output/filename).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    (args.output/'review.json').write_bytes(review_bytes)
    (args.output/'application.json').write_text(json.dumps({
        'source_transcript': {'path': str(args.transcript.resolve()), 'sha256': hashlib.sha256(original_bytes).hexdigest()},
        'source_review': {'path': str(args.review.resolve()), 'sha256': hashlib.sha256(review_bytes).hexdigest()},
        'source_artifact_id': json.loads(original_bytes)['artifact_id'],
        'reviewed_artifact_id': document['artifact_id'], 'source_files_modified': False,
        'ready_anonymous_rows': len(ready['segments']),
        'ready_attributed_rows': sum(r['ready_for_attributed_summary'] for r in ready['segments'])
    }, ensure_ascii=False, indent=2)+'\n')
    print(f'Saved separate reviewed output: {args.output}; {len(ready["segments"])} rows ready for anonymous summary')


if __name__ == '__main__':
    main()
