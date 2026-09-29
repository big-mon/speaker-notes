"""Conservative ASR crop candidates from nonexclusive diarization events.

These are activity-based hypotheses, not text attribution or acoustic ground
truth. No event is removed. Short, fully overlapped speech stays unresolved;
its duration is never interpreted as proof that it is an unimportant backchannel.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def _overlap(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def _uncovered_seconds(start, end, intervals):
    cursor, covered = start, 0.0
    for a, b in sorted(intervals):
        a, b = max(start, a), min(end, b)
        if b > max(a, cursor):
            covered += b - max(a, cursor)
            cursor = b
    return max(0.0, end - start - covered)


def _union(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def build_candidates(segments, start=0.0, end=None, anchor_seconds=2.5,
                     merge_gap_seconds=0.35):
    """Return raw events plus potentially overlapping crops in original time.

    A sustained event is a crop anchor. A shorter event is also a crop when it
    has any single-speaker support; if entirely overlapped it remains an
    unresolved event. Same-speaker spans bridge only a short silence containing
    no other speaker activity. IDs are copied, not independently renumbered.
    """
    if not math.isfinite(start) or start < 0:
        raise ValueError('start must be a finite nonnegative second')
    if end is not None and (not math.isfinite(end) or end <= start):
        raise ValueError('end must be finite and greater than start')
    if not math.isfinite(anchor_seconds) or anchor_seconds <= 0:
        raise ValueError('anchor_seconds must be finite and positive')
    if not math.isfinite(merge_gap_seconds) or merge_gap_seconds < 0:
        raise ValueError('merge_gap_seconds must be finite and nonnegative')

    events = []
    for i, raw in enumerate(segments):
        a, b = float(raw['start']), float(raw['end'])
        if not math.isfinite(a) or not math.isfinite(b) or not 0 <= a < b:
            raise ValueError(f'invalid source event {i}')
        if raw.get('speaker') is None:
            raise ValueError(f'missing speaker for source event {i}')
        if b <= start or (end is not None and a >= end):
            continue
        events.append({'id': i, 'start': max(a, start),
                       'end': min(b, end) if end is not None else b,
                       'speaker': str(raw['speaker']),
                       'source_start': a, 'source_end': b,
                       'source_event': dict(raw)})
    events.sort(key=lambda x: (x['start'], x['end'], x['id']))
    if not events:
        return _document([], [], [], start, end, anchor_seconds, merge_gap_seconds)

    # Coalesce only same-speaker activity across actual silence. A short response
    # inside a gap prevents coalescing even when its duration is very small.
    groups = []
    for speaker in sorted({e['speaker'] for e in events}):
        previous = None
        for event in (e for e in events if e['speaker'] == speaker):
            can_join = previous is not None and (
                event['start'] - previous['end'] <= merge_gap_seconds)
            if can_join and event['start'] > previous['end']:
                can_join = not any(
                    e['speaker'] != speaker and
                    _overlap((previous['end'], event['start']),
                             (e['start'], e['end'])) > 0 for e in events)
            if can_join:
                previous['end'] = max(previous['end'], event['end'])
                previous['source_event_ids'].append(event['id'])
                previous['source_start'] = min(previous['source_start'], event['source_start'])
                previous['source_end'] = max(previous['source_end'], event['source_end'])
            else:
                previous = {k: event[k] for k in
                            ('start', 'end', 'speaker', 'source_start', 'source_end')}
                previous['source_event_ids'] = [event['id']]
                groups.append(previous)

    candidates, unresolved = [], []
    for group in sorted(groups, key=lambda x: (x['start'], x['end'], x['speaker'])):
        own = [e for e in events if e['id'] in group['source_event_ids']]
        other = [e for e in events if e['speaker'] != group['speaker'] and
                 _overlap((group['start'], group['end']), (e['start'], e['end'])) > 0]
        single_seconds = sum(_uncovered_seconds(a, b, [(e['start'], e['end']) for e in other])
                             for a, b in _union([(e['start'], e['end']) for e in own]))
        # Using source extent prevents a window edge from turning a long utterance
        # into a supposedly brief event. Silent gaps do not count as speech support.
        source_speech_seconds = sum(b - a for a, b in _union(
            [(e['source_start'], e['source_end']) for e in own]))
        sustained = source_speech_seconds >= anchor_seconds
        if not sustained and single_seconds <= 1e-9:
            unresolved.extend(group['source_event_ids'])
            continue

        overlaps = [{'start': max(group['start'], e['start']),
                     'end': min(group['end'], e['end']),
                     'speaker': e['speaker'], 'source_event_id': e['id']}
                    for e in other]
        boundary_ranges = []
        for side in ('start', 'end'):
            t = group[side]
            crossing = [e for e in other if e['start'] < t < e['end']]
            if crossing:
                boundary_ranges.append({
                    'side': side,
                    'start': min(max(group['start'], e['start']) for e in crossing),
                    'end': max(min(group['end'], e['end']) for e in crossing),
                    'other_speakers': sorted({e['speaker'] for e in crossing}),
                    'reason': 'transition lies inside simultaneous speaker activity'})
        clipped = group['source_start'] < start or (
            end is not None and group['source_end'] > end)
        reasons = []
        if overlaps:
            reasons.append('other speech is unresolved; may contain meaningful interruption')
        if boundary_ranges:
            reasons.append('speaker transition has a range, not a verified word boundary')
        if clipped:
            reasons.append('window clips a source event; context may continue outside crop')
        candidates.append(dict(group, id=len(candidates),
                               kind='sustained_activity' if sustained else 'brief_exposed_activity',
                               source_speech_seconds=source_speech_seconds,
                               single_speaker_support_seconds=single_seconds,
                               overlaps=overlaps, uncertain_boundaries=boundary_ranges,
                               clipped_by_window=clipped, review_reasons=reasons,
                               text_attribution_verified=False))
    return _document(events, candidates, unresolved, start, end,
                     anchor_seconds, merge_gap_seconds)


def _document(events, candidates, unresolved, start, end, anchor_seconds, gap):
    return {'schema_version': 1,
            'purpose': 'ASR crop candidates; not speaker-labeled text or acoustic ground truth',
            'window': {'start': start, 'end': end},
            'parameters': {'anchor_seconds': anchor_seconds, 'merge_gap_seconds': gap,
                           'thresholds_are_confidence': False},
            'raw_events': events, 'candidates': candidates,
            'unresolved_fully_overlapped_event_ids': sorted(unresolved),
            'raw_events_preserved': True, 'human_verified': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('diarization', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--start', type=float, default=0)
    parser.add_argument('--end', type=float)
    parser.add_argument('--anchor-seconds', type=float, default=2.5)
    parser.add_argument('--merge-gap-seconds', type=float, default=0.35)
    args = parser.parse_args()
    data = args.diarization.read_bytes()
    source = json.loads(data)
    result = build_candidates(source['segments'], args.start, args.end,
                              args.anchor_seconds, args.merge_gap_seconds)
    result['source'] = {'path': str(args.diarization.resolve()),
                        'sha256': hashlib.sha256(data).hexdigest(),
                        'metadata': {k: v for k, v in source.items() if k != 'segments'}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
