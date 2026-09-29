"""Experimental sentence/major-turn candidates from forced alignment.

No inference, text correction, acoustic verification, or summary approval occurs.
Input diarization times must refer to the same clip as alignment.json. All text
offsets are Python Unicode code points, end exclusive; output times are source
seconds. Raw model outputs remain embedded separately and are never modified.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

from compose_alignment import compose_alignments
from turn_candidates import build_candidates, _uncovered_seconds


ANCHOR_SECONDS = 2.5
REPAIR_GATE_FRACTION = .10
TERMINALS = '。？！?!.'
CLOSERS = '」』）)]}”’"'


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def sentence_spans(text):
    """Partition text without trimming, including punctuation and whitespace."""
    cuts, i = [0], 0
    while i < len(text):
        decimal = (text[i] == '.' and i > 0 and i + 1 < len(text)
                   and text[i - 1].isdigit() and text[i + 1].isdigit())
        if text[i] in TERMINALS and not decimal:
            i += 1
            while i < len(text) and (text[i] in TERMINALS + CLOSERS or text[i].isspace()):
                i += 1
            cuts.append(i)
        else:
            i += 1
    if cuts[-1] != len(text):
        cuts.append(len(text))
    return [[a, b] for a, b in zip(cuts, cuts[1:]) if a < b]


def _token_bounds(item):
    spans = item['original_character_spans']
    return (spans[0][0], spans[-1][1]) if spans else None


def _raw_valid(item, duration):
    pair = item.get('raw_clip_seconds')
    return (isinstance(pair, list) and len(pair) == 2 and all(_finite(v) for v in pair)
            and 0 <= pair[0] < pair[1] <= duration)


def _eligible(item, duration):
    return (item.get('eligible_for_boundary_candidate') is True
            and not item.get('library_repaired') and not item.get('flags')
            and bool(item['original_character_spans']) and _raw_valid(item, duration))


def _validate_alignment(alignment):
    text = alignment['mapping']['original_text']
    duration, offset = alignment['duration_seconds'], alignment['source_offset_seconds']
    if not isinstance(text, str) or not text or not _finite(duration) or duration <= 0:
        raise ValueError('Expected nonempty original text and finite positive clip duration')
    if not _finite(offset) or offset < 0:
        raise ValueError('Expected finite nonnegative source offset')
    items = alignment['items']
    previous = 0
    for index, item in enumerate(items):
        if item['index'] != index or not isinstance(item.get('library_repaired'), bool):
            raise ValueError('Expected ordered token indices and explicit repair flags')
        for a, b in item['original_character_spans']:
            if not isinstance(a, int) or not isinstance(b, int) or not 0 <= a < b <= len(text) or a < previous:
                raise ValueError('Original token character spans must be ordered and nonoverlapping')
            previous = b
    return text, duration, offset, items


def _validated_composite(alignment):
    """Only a reproducible composition may replace the single-clip repair gate."""
    if alignment.get('kind') != 'derived_composite_of_independent_forced_alignments':
        return False
    try:
        expected = compose_alignments(alignment['raw_children'], alignment['source_offset_seconds'],
                                      alignment['duration_seconds'])
        if alignment['mapping']['original_text'] != expected['mapping']['original_text'] or any(
                alignment[key] != expected[key] for key in ('items', 'composition_parts', 'composition_rules')):
            raise ValueError('Derived text, items or composition metadata differs from raw children')
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f'Invalid composite alignment contract: {error}') from error
    return True


def _chunk_bounded_spans(text, alignment, composite):
    """Partition text at real ASR input seams as well as sentence punctuation.

    A child chunk's source extent is known even when its word alignment fails.
    The separator newline belongs to the preceding child; no text is invented,
    omitted, or treated as evidence of a sentence or speaker boundary.
    """
    natural = sentence_spans(text)
    if not composite:
        return natural, set(), [{'character_start': 0, 'character_end': len(text),
                                 'start': 0., 'end': alignment['duration_seconds'],
                                 'alignment_part_index': None}]
    parts = alignment['composition_parts']
    regions = [dict(character_start=part['original_text_span'][0],
                    character_end=parts[index + 1]['original_text_span'][0] if index + 1 < len(parts) else len(text),
                    start=part['start'], end=part['end'],
                    alignment_part_index=part['alignment_part_index'])
               for index, part in enumerate(parts)]
    natural_ends = {end for _, end in natural}
    chunk_cuts = {region['character_start'] for region in regions[1:]}
    cuts = sorted({0, len(text)} | natural_ends | chunk_cuts)
    return [[a, b] for a, b in zip(cuts, cuts[1:]) if a < b], chunk_cuts - natural_ends, regions


def _sentence_boundaries(text, spans, items, duration, offset, chunk_splits=()):
    boundaries = []
    for _, cut in spans[:-1]:
        left = next((t for t in reversed(items) if _token_bounds(t) and _token_bounds(t)[1] <= cut), None)
        right = next((t for t in items if _token_bounds(t) and _token_bounds(t)[0] >= cut), None)
        reasons = ['source_chunk_boundary_not_sentence_boundary'] if cut in chunk_splits else []
        if left is None or right is None:
            reasons.append('missing_adjacent_mapped_token')
        else:
            # Do not skip a bad final token, or cross an unmapped lexical gap.
            if right['index'] != left['index'] + 1:
                reasons.append('adjacent_tokens_not_consecutive')
            between = text[_token_bounds(left)[1]:_token_bounds(right)[0]]
            if any(c.isalnum() for c in between):
                reasons.append('unmapped_lexical_characters_at_boundary')
            if not _eligible(left, duration) or not _eligible(right, duration):
                reasons.append('adjacent_token_timing_ineligible')
            if _raw_valid(left, duration) and _raw_valid(right, duration):
                gap = right['raw_clip_seconds'][0] - left['raw_clip_seconds'][1]
                if gap < 0 or gap > 2.0:
                    reasons.append('adjacent_token_gap_or_overlap_unsafe')
        safe = not reasons
        boundaries.append({
            'id': len(boundaries), 'raw_character_offset': cut,
            'eligible': safe, 'flags': reasons,
            'left_token_index': left['index'] if left else None,
            'right_token_index': right['index'] if right else None,
            'left_raw_clip_seconds': left.get('raw_clip_seconds') if left else None,
            'right_raw_clip_seconds': right.get('raw_clip_seconds') if right else None,
            'clip_seconds': left['raw_clip_seconds'][1] if safe else None,
            'source_seconds': offset + left['raw_clip_seconds'][1] if safe else None,
            'time_rule': 'left token raw end; punctuation has no model timestamp; no interpolation',
            'boundary_origin': 'ASR input chunk seam' if cut in chunk_splits else 'sentence punctuation',
            'human_verified': False,
        })
    return boundaries


def _major_turns(candidates):
    turns = []
    for candidate in candidates:
        if candidate['kind'] != 'sustained_activity':
            continue
        if turns and turns[-1]['speaker'] == candidate['speaker']:
            turns[-1]['end'] = max(turns[-1]['end'], candidate['end'])
            turns[-1]['candidate_ids'].append(candidate['id'])
            turns[-1]['source_event_ids'].extend(candidate['source_event_ids'])
        else:
            turns.append({'id': len(turns), 'start': candidate['start'], 'end': candidate['end'],
                          'speaker': candidate['speaker'], 'candidate_ids': [candidate['id']],
                          'source_event_ids': list(candidate['source_event_ids'])})
    return turns


def _transitions(turns, boundaries, offset, snap_seconds, gated):
    transitions = []
    for previous, turn in zip(turns, turns[1:]):
        near = [b for b in boundaries if b['eligible'] and abs(b['clip_seconds'] - turn['start']) <= snap_seconds]
        near.sort(key=lambda b: (abs(b['clip_seconds'] - turn['start']), b['raw_character_offset']))
        best = near[0] if near and not gated else None
        transitions.append({
            'id': len(transitions), 'from_speaker': previous['speaker'], 'to_speaker': turn['speaker'],
            'from_turn_id': previous['id'], 'to_turn_id': turn['id'],
            'original_event_transition': {'clip_seconds': turn['start'], 'source_seconds': offset + turn['start'],
                                          'source_event_ids': list(turn['source_event_ids']),
                                          'reference': 'new sustained speaker start, never overlap midpoint'},
            'boundary_id': best['id'] if best else None,
            'raw_character_offset': best['raw_character_offset'] if best else None,
            'snapped_clip_seconds': best['clip_seconds'] if best else None,
            'snapped_source_seconds': best['source_seconds'] if best else None,
            'snap_delta_seconds': best['clip_seconds'] - turn['start'] if best else None,
            'state': 'sentence_boundary_candidate' if best else 'unresolved',
            'flags': [] if best else ['input_repair_gate' if gated else 'no_eligible_sentence_boundary_near_start'],
            'human_verified': False,
        })
    # Two speaker changes cannot share one sentence cut or reverse text order.
    selected = [t for t in transitions if t['boundary_id'] is not None]
    conflicts = set()
    for left, right in zip(selected, selected[1:]):
        if left['raw_character_offset'] >= right['raw_character_offset']:
            conflicts.update([left['id'], right['id']])
    for transition in transitions:
        if transition['id'] in conflicts:
            transition['state'] = 'unresolved'
            transition['flags'].append('competing_or_nonmonotonic_sentence_boundary')
    return transitions


def build_aligned_turns(alignment, diarization, snap_seconds=2.0):
    """Return tentative sentence turns; preserves both raw documents unchanged."""
    if not _finite(snap_seconds) or snap_seconds <= 0:
        raise ValueError('snap_seconds must be finite and positive')
    text, duration, offset, items = _validate_alignment(alignment)
    activity = build_candidates(diarization['segments'], end=duration, anchor_seconds=ANCHOR_SECONDS)
    turns = _major_turns(activity['candidates'])
    composite = _validated_composite(alignment)
    spans, chunk_splits, regions = _chunk_bounded_spans(text, alignment, composite)
    boundaries = _sentence_boundaries(text, spans, items, duration, offset, chunk_splits)
    repaired = sum(t['library_repaired'] for t in items)
    fraction = repaired / len(items) if items else 0.
    gated = not items or (not composite and fraction > REPAIR_GATE_FRACTION)
    transitions = _transitions(turns, boundaries, offset, snap_seconds, gated)
    sustained_event_ids = {event_id for turn in turns for event_id in turn['source_event_ids']}
    mapped_characters = {i for token in items for a, b in token['original_character_spans'] for i in range(a, b)}
    segments = []
    for a, b in spans:
        region = next(region for region in regions if region['character_start'] <= a < region['character_end'])
        tokens = [t for t in items if any(x < b and y > a for x, y in t['original_character_spans'])]
        usable = [t for t in tokens if _raw_valid(t, duration)
                  and region['start'] <= t['raw_clip_seconds'][0] < t['raw_clip_seconds'][1] <= region['end']]
        flags = ['not_human_verified', 'asr_text_accuracy_not_established']
        unmapped_lexical = any(text[i].isalnum() and i not in mapped_characters for i in range(a, b))
        if not unmapped_lexical and usable and usable[0] is tokens[0] and usable[-1] is tokens[-1] and usable[0]['raw_clip_seconds'][0] < usable[-1]['raw_clip_seconds'][1]:
            start, end = usable[0]['raw_clip_seconds'][0], usable[-1]['raw_clip_seconds'][1]
            timing = 'first and last token raw times; internal anomalies retained in references'
            time_kind = 'forced_alignment_token_envelope'
        else:
            start, end = region['start'], region['end']
            timing = 'source ASR input chunk extent; text has no safe timed envelope' if composite else 'whole clip fallback; sentence has no safe timed envelope'
            time_kind = 'source_chunk_envelope_fallback' if composite else 'source_clip_envelope_fallback'
            flags.append('broad_playback_timing_fallback')
        if any(t.get('flags') or t.get('library_repaired') for t in tokens):
            flags.append('contains_flagged_alignment_tokens')
        child_repair_gate = any('alignment_chunk_repair_gate' in t.get('flags', []) for t in tokens)
        part_indices = sorted({t['alignment_part_index'] for t in tokens if 'alignment_part_index' in t})
        crosses_alignment_seam = len(part_indices) > 1
        split_at_chunk = a in chunk_splits or b in chunk_splits
        if child_repair_gate:
            flags.append('alignment_chunk_repair_gate')
        if crosses_alignment_seam:
            flags.append('alignment_chunk_seam_inside_sentence')
        if split_at_chunk:
            flags.append('sentence_split_at_alignment_chunk_boundary')
        if unmapped_lexical:
            flags.append('unmapped_lexical_text_timing_unresolved')
        main = turns[0]['speaker'] if turns else None
        unsafe = unmapped_lexical or child_repair_gate or crosses_alignment_seam or split_at_chunk
        for transition in transitions:
            if transition['state'] == 'sentence_boundary_candidate':
                cut = transition['raw_character_offset']
                if a >= cut:
                    main = transition['to_speaker']
                elif b > cut:
                    unsafe = True
            else:
                when = transition['original_event_transition']['clip_seconds']
                if start >= when:
                    main = transition['to_speaker']
                elif end > when:
                    unsafe = True
        active = [e for e in activity['raw_events'] if e['start'] < end and e['end'] > start]
        other = [e for e in active if e['speaker'] != main]
        if other:
            flags.append('other_speaker_activity_ownership_unresolved')
        own_intervals = [(e['start'], e['end']) for e in active if e['speaker'] == main]
        # A brief sequential correction can fall inside a long unpunctuated
        # sentence. Keeping its raw event alone is not enough: do not attribute
        # that sentence to the surrounding speaker. Sustained turn transitions
        # still use the explicit sentence snap evidence above.
        if any(e['id'] not in sustained_event_ids and
               _uncovered_seconds(max(start, e['start']), min(end, e['end']), own_intervals) > 1e-9
               for e in other):
            flags.append('unseparated_brief_other_speaker_activity')
            unsafe = True
        overlap = []
        for i, left in enumerate(active):
            for right in active[i + 1:]:
                lo, hi = max(start, left['start'], right['start']), min(end, left['end'], right['end'])
                if left['speaker'] != right['speaker'] and lo < hi:
                    overlap.append({'start': offset + lo, 'end': offset + hi,
                                    'speakers': [left['speaker'], right['speaker']],
                                    'source_event_ids': [left['id'], right['id']]})
        if overlap:
            flags.append('raw_diarization_overlap_ownership_unverified')
        if unsafe:
            flags.append('unresolved_major_transition_inside_sentence')
        if not any(e['speaker'] == main for e in active):
            flags.append('no_raw_activity_support_for_candidate_main')
            unsafe = True
        short = end - start < ANCHOR_SECONDS
        if short:
            flags.append('brief_sentence_ownership_unresolved')
        if gated:
            flags.append('input_repair_gate')
        if 'broad_playback_timing_fallback' in flags:
            unsafe = True
        speakers = [main] if main is not None and not (gated or unsafe or short) else []
        segments.append({
            'id': f'sentence-{len(segments):04d}', 'start': offset + start, 'end': offset + end,
            'text': text[a:b], 'source_raw_span': [a, b],
            'forced_token_references': [t['index'] for t in tokens],
            'alignment_part_indices': part_indices,
            'speakers': speakers, 'candidate_main_speaker': main,
            'state': 'candidate_main_not_verified' if speakers else 'unknown_requires_review',
            'flags': flags, 'raw_overlap_intervals': overlap,
            'raw_diarization_event_ids': [e['id'] for e in active],
            'raw_other_speaker_event_ids': [e['id'] for e in other],
            'playback': {'start': offset + start, 'end': offset + end},
            'time_kind': time_kind, 'timing_provenance': timing,
            'source_chunk_extent': {'start': offset + region['start'], 'end': offset + region['end'],
                                    'alignment_part_index': region['alignment_part_index']},
            'human_verified': False, 'ready_for_summary': False,
        })
    if ''.join(s['text'] for s in segments) != text:
        raise ValueError('Original text was not preserved')
    return {
        'schema_version': 1, 'kind': 'experimental_forced_alignment_major_turn_candidates',
        'original_text': text, 'source_window': {'start': offset, 'end': offset + duration},
        'time_basis': {'segments': 'original audio seconds', 'raw_diarization': 'clip-local seconds',
                       'source_offset_seconds': offset},
        'speaker_identity_scope': 'one input diarization invocation; raw IDs never combined across clips',
        'rules': {'anchor_seconds': ANCHOR_SECONDS, 'snap_seconds': snap_seconds,
                  'maximum_adjacent_token_gap_seconds': 2., 'repair_gate_fraction': REPAIR_GATE_FRACTION,
                  'repair_gate_test': 'strictly greater than threshold; engineering diagnostic, not calibrated accuracy',
                  'brief_events': 'all retained; duration alone never establishes backchannel or importance',
                  'major_turn_join': 'neighboring same-speaker sustained candidates; intervening brief events retained',
                  'composite_alignment': 'child repair gates cannot be diluted; source chunk seams bound text rows, never prove sentence or speaker boundaries',
                  'timing_fallback': 'validated source ASR chunk extent, never the full composite recording; unknown speaker; no invented word times',
                  'speaker_assignment': 'tentative main speech only; short sentences remain unattributed'},
        'diagnostics': {'tokens': len(items), 'repaired_tokens': repaired, 'repaired_token_fraction': fraction,
                        'flagged_tokens': sum(bool(t.get('flags')) for t in items),
                        'repair_gate_scope': 'validated_child_alignments' if composite else 'whole_single_clip',
                        'automatic_assignment_blocked': gated, 'original_text_preserved': True,
                        'counts_are_not_accuracy': True},
        'segments': segments, 'major_turns': turns, 'transitions': transitions,
        'sentence_boundaries': boundaries, 'activity_candidates': activity,
        'raw_alignment': copy.deepcopy(alignment), 'raw_diarization': copy.deepcopy(diarization),
        'human_verified': False, 'ready_for_summary': False,
    }


def _fingerprint(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def save_candidates(document, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'candidates.json').write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    lines = ['# 強制アラインメントによる話者候補（実験）', '',
             '本文・話者・時刻は未確認。要約への受け渡しは未承認。短い応答と重なりは要聴取。', '']
    for row in document['segments']:
        speaker = ', '.join(row['speakers']) or '話者不明'
        candidate = row['candidate_main_speaker'] or 'なし'
        lines.extend([f"[{row['start']:.3f}–{row['end']:.3f}] {speaker}（主話者候補: {candidate}）",
                      row['text'], '', f"状態: {row['state']} / {', '.join(row['flags'])}", ''])
    (output / 'candidates.md').write_text('\n'.join(lines), encoding='utf-8')
    (output / 'text.txt').write_text(document['original_text'], encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('alignment', type=Path, help='align_qwen.py alignment.json')
    parser.add_argument('diarization', type=Path, help='Same-clip JSON with segments[].start/end/speaker')
    parser.add_argument('output', type=Path, help='New directory only')
    parser.add_argument('--snap-seconds', type=float, default=2.)
    args = parser.parse_args()
    document = build_aligned_turns(json.loads(args.alignment.read_text()),
                                   json.loads(args.diarization.read_text()), args.snap_seconds)
    document['raw_sources'] = {'alignment': _fingerprint(args.alignment),
                               'diarization': _fingerprint(args.diarization),
                               'implementation': _fingerprint(__file__),
                               'candidate_implementation': _fingerprint(Path(__file__).with_name('turn_candidates.py')),
                               'composition_implementation': _fingerprint(Path(__file__).with_name('compose_alignment.py'))}
    manifest = args.alignment.with_name('manifest.json')
    if manifest.is_file():
        document['raw_sources']['alignment_manifest'] = _fingerprint(manifest)
        document['alignment_manifest'] = json.loads(manifest.read_text())
    save_candidates(document, args.output)
    print(args.output)


if __name__ == '__main__':
    main()
