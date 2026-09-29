"""Independent lexical attribution pilot; no text edits or acoustic inference.

NLTokenizer supplies textual boundaries, not verified acoustic words. Apple
native runs are never split. All durations below are geometric measurements,
not confidence; even brief simultaneous speaker activity remains explicit.
"""
import math


MIN_COVERAGE = .65
MAX_INTERNAL_GAP = .15
COLLAR_SECONDS = .2
COLLAR_MARGIN = .15


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _union(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def _seconds(intervals):
    return sum(b - a for a, b in intervals)


def _diarization(diarization):
    grouped = {}
    for segment in diarization['segments']:
        a, b, speaker = segment['start'], segment['end'], segment['speaker']
        if not (_finite(a) and _finite(b) and 0 <= a < b):
            raise ValueError('Diarization must contain finite positive intervals')
        if not isinstance(speaker, (str, int)) or isinstance(speaker, bool) or str(speaker) == '':
            raise ValueError('Diarization speaker IDs must be nonempty strings or integers')
        grouped.setdefault(str(speaker), []).append([a, b])
    grouped = {s: _union(intervals) for s, intervals in grouped.items()}
    points = sorted({t for spans in grouped.values() for span in spans for t in span})
    epochs = []
    for a, b in zip(points, points[1:]):
        active = sorted(s for s, spans in grouped.items() if any(x < b and y > a for x, y in spans))
        if active:
            epochs.append({'start': a, 'end': b, 'speakers': active})
    changes = set()
    for left, right in zip(epochs, epochs[1:]):
        if left['speakers'] != right['speakers'] and len(set(left['speakers'] + right['speakers'])) >= 2:
            # Preserve both actual edges across gaps; never invent a midpoint.
            changes.update((left['end'], right['start']))
    return grouped, epochs, sorted(changes)


def _assign(units, grouped, epochs, changes, collar):
    reasons = []
    pairs = [[u.get('start'), u.get('end')] for u in units]
    valid = all(_finite(a) and _finite(b) and 0 <= a < b for a, b in pairs)
    if not valid:
        reasons.append('missing_invalid_or_zero_duration_native_timing')
    usable = [[a, b] for a, b in pairs if _finite(a) and _finite(b) and 0 <= a < b]
    if valid and any(c < a or d < b for (a, b), (c, d) in zip(pairs, pairs[1:])):
        reasons.append('nonmonotonic_native_timing')
    timeline = _union(usable)
    gaps = [[left[1], right[0]] for left, right in zip(timeline, timeline[1:])]
    if any(b - a > MAX_INTERNAL_GAP + 1e-9 for a, b in gaps):
        reasons.append('native_timing_gap_exceeds_limit')
    duration = _seconds(timeline)
    scores = {}
    for speaker, spans in grouped.items():
        intersections = _union([[max(a, x), min(b, y)] for a, b in timeline for x, y in spans
                                if max(a, x) < min(b, y)])
        if intersections:
            seconds = _seconds(intersections)
            scores[speaker] = {'seconds': seconds, 'fraction': seconds / duration}
    overlap = []
    for epoch in epochs:
        if len(epoch['speakers']) < 2:
            continue
        for a, b in timeline:
            x, y = max(a, epoch['start']), min(b, epoch['end'])
            if x < y:
                overlap.append({'start': x, 'end': y, 'speakers': list(epoch['speakers'])})
    overlap_speakers = sorted({s for item in overlap for s in item['speakers']})
    ranking = sorted(scores, key=lambda s: (-scores[s]['fraction'], s))
    nearby = [t for t in changes if any(a - COLLAR_SECONDS <= t <= b + COLLAR_SECONDS for a, b in timeline)]
    ambiguous = (len(ranking) >= 2 and
                 scores[ranking[0]]['fraction'] - scores[ranking[1]]['fraction'] <= COLLAR_MARGIN + 1e-9)
    timing_safe = not reasons
    speakers, state = [], 'unknown'
    if timing_safe and overlap:
        speakers, state = overlap_speakers, 'overlap'
    elif timing_safe:
        speakers = [s for s in ranking if scores[s]['fraction'] >= MIN_COVERAGE]
        if len(speakers) == 1:
            state = 'single'
        else:
            speakers = []
            reasons.append('insufficient_or_competing_speaker_coverage')
    if overlap:
        reasons.append('simultaneous_speaker_activity_text_unseparated')
    if collar and ambiguous and nearby:
        speakers, state = [], 'unknown'
        reasons.append('ambiguous_coverage_near_raw_speaker_change')
    return {'start': timeline[0][0] if timeline else None, 'end': timeline[-1][1] if timeline else None,
            'timing_safe': timing_safe, 'native_time_union': timeline, 'native_time_seconds': duration,
            'internal_gaps': gaps, 'speaker_scores': scores, 'speakers': speakers, 'state': state,
            'overlap_speakers': overlap_speakers, 'overlap_intervals': overlap,
            'overlap_seconds': _seconds(_union([[x['start'], x['end']] for x in overlap])),
            'nearby_raw_speaker_change_times': nearby, 'collar_ambiguous': ambiguous,
            'reasons': reasons, 'human_verified': False}


def build_buckets(asr, lexical, diarization, collar=False):
    """Return exact text buckets using common native/lexical endpoints.

    ``lexical`` is NLTokenizer's list of piece lists, one per Apple result.
    Only piece.text is consumed: Swift grapheme offsets are intentionally ignored.
    Native indices are global; source_unit_references additionally retain [result,
    unit] indices. All times use the unmodified input clock.
    """
    if not isinstance(collar, bool):
        raise ValueError('collar must be a boolean')
    if not isinstance(asr, list) or not isinstance(lexical, list) or len(asr) != len(lexical):
        raise ValueError('Expected matching Apple and lexical segment lists')
    grouped, epochs, changes = _diarization(diarization)
    buckets, originals, global_char, global_unit = [], [], 0, 0
    for si, (segment, pieces) in enumerate(zip(asr, lexical)):
        text, units = segment['text'], segment['units']
        if not isinstance(text, str) or not isinstance(units, list) or not isinstance(pieces, list):
            raise ValueError('Expected text, native units and lexical pieces')
        if any(not isinstance(u.get('text'), str) for u in units) or any(not isinstance(p.get('text'), str) for p in pieces):
            raise ValueError('Every native unit and lexical piece requires text')
        if ''.join(u['text'] for u in units) != text or ''.join(p['text'] for p in pieces) != text:
            raise ValueError('Native units and lexical pieces must exactly preserve segment text')
        originals.append(text)
        native_ends, cursor = {}, 0
        for ui, unit in enumerate(units):
            cursor += len(unit['text'])
            native_ends[cursor] = ui + 1
        lexical_ends, cursor = set(), 0
        for piece in pieces:
            cursor += len(piece['text'])
            lexical_ends.add(cursor)
        cuts = sorted(set(native_ends) & lexical_ends - {0})
        previous_char, previous_unit = 0, 0
        for cut in cuts:
            stop = native_ends[cut]
            selected = units[previous_unit:stop]
            bucket = {'id': len(buckets), 'text': text[previous_char:cut],
                      'character_span': [global_char + previous_char, global_char + cut],
                      'asr_segment': si, 'native_unit_indices': list(range(global_unit + previous_unit, global_unit + stop)),
                      'source_unit_references': [[si, ui] for ui in range(previous_unit, stop)],
                      'raw_timing': [{'native_unit_index': global_unit + ui,
                                     'start': units[ui].get('start'), 'end': units[ui].get('end')}
                                    for ui in range(previous_unit, stop)]}
            bucket.update(_assign(selected, grouped, epochs, changes, collar))
            buckets.append(bucket)
            previous_char, previous_unit = cut, stop
        global_char += len(text)
        global_unit += len(units)
    original = ''.join(originals)
    assert ''.join(b['text'] for b in buckets) == original
    return {'schema_version': 1, 'original_text': original, 'buckets': buckets,
            'raw_speaker_change_times': changes,
            'settings': {'minimum_coverage': MIN_COVERAGE, 'maximum_internal_gap_seconds': MAX_INTERNAL_GAP,
                         'collar_enabled': collar, 'collar_seconds': COLLAR_SECONDS,
                         'collar_maximum_coverage_margin': COLLAR_MARGIN,
                         'character_offsets': 'Python Unicode code points, end exclusive; derived from piece.text',
                         'partition': 'common native-unit and lexical-piece endpoints within each Apple result',
                         'speaker_scores_are_confidence': False, 'duration_denominator': 'union of valid native intervals',
                         'overlap_rule': 'any positive simultaneous activity of distinct raw speaker IDs; text unseparated',
                         'invalid_timing_rule': 'unknown; measurements from valid subset are diagnostic only',
                         'short_bucket_absorption': False, 'neighbor_speaker_inheritance': False,
                         'lexical_boundaries_are_verified_acoustic_words': False},
            'validation': {'text_preserved': True, 'native_units_not_split': True, 'accuracy_verified': False}}
