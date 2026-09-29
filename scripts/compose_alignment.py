"""Compose independent alignment documents without joining audio or speakers.

Child text is copied exactly, with one explicit newline between children. Times
remain predictions from separate inferences; seam tokens cannot anchor a cut.
"""
import copy
import math


TOLERANCE_SECONDS = 1e-6
REPAIR_GATE_FRACTION = .10


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _same(left, right):
    return _finite(left) and _finite(right) and abs(left - right) <= TOLERANCE_SECONDS


def _pair(value):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError('Expected a timestamp pair')
    for v in value:
        if _finite(v):
            continue
        try:
            nonfinite = not isinstance(v, bool) and not math.isfinite(float(v))
        except (TypeError, ValueError, OverflowError):
            nonfinite = False
        if not nonfinite:
            raise ValueError('Timestamp must be numeric or explicitly nonfinite')
    return value


def _rebase_time(item, key, start, child_offset):
    clip = _pair(item[key + '_clip_seconds'])
    source = item[key + '_source_seconds']
    if source is None:
        if all(_finite(v) for v in clip):
            raise ValueError('Finite clip times are missing full-source times')
    else:
        _pair(source)
        for c, s in zip(clip, source):
            if (_finite(c) and not _same(s, child_offset + c)) or (not _finite(c) and _finite(s)):
                raise ValueError('Full-source time does not match child offset plus clip time')
    item['original_part_' + key + '_clip_seconds'] = copy.deepcopy(clip)
    item[key + '_clip_seconds'] = [v + start if _finite(v) else v for v in clip]
    # Full-source values, including None/nonfinite evidence, are never shifted.


def compose_alignments(parts, source_offset, duration):
    """Return a derived document accepted by the experimental turn builder.

    Parts are ordered dictionaries with clip-local start/end, alignment, and an
    optional source fingerprint. Validation tolerance only accommodates floating
    point arithmetic; no missing interval or timestamp is filled/interpolated.
    """
    if not parts or not _finite(source_offset) or source_offset < 0 or not _finite(duration) or duration <= 0:
        raise ValueError('Expected parts, nonnegative source offset and positive duration')
    texts, items, metadata, children = [], [], [], []
    expected_start, character_offset = 0., 0
    for part_index, part in enumerate(parts):
        start, end, alignment = part['start'], part['end'], part['alignment']
        if not _finite(start) or not _finite(end) or start < 0 or end <= start or not _same(start, expected_start):
            raise ValueError('Parts must have contiguous positive extents starting at zero')
        if not _same(alignment['duration_seconds'], end - start):
            raise ValueError('Child duration differs from its part extent')
        if not _same(alignment['source_offset_seconds'], source_offset + start):
            raise ValueError('Child source offset differs from global offset plus part start')
        text, child_items = alignment['mapping']['original_text'], alignment['items']
        if not isinstance(text, str) or not isinstance(child_items, list):
            raise ValueError('Expected child original text and token list')
        if any(not isinstance(t.get('library_repaired'), bool) for t in child_items):
            raise ValueError('Each child token needs an explicit library repair flag')
        repaired = sum(t['library_repaired'] for t in child_items)
        repair_fraction = repaired / len(child_items) if child_items else 0.
        gated = repair_fraction > REPAIR_GATE_FRACTION
        token_start, previous_character = len(items), 0
        for original_index, original in enumerate(child_items):
            if original['index'] != original_index:
                raise ValueError('Child token indices must be contiguous and ordered')
            item = copy.deepcopy(original)
            spans = item['original_character_spans']
            for a, b in spans:
                if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(text) or a < previous_character:
                    raise ValueError('Child character spans must be ordered, nonoverlapping and within text')
                previous_character = b
            item.update(index=len(items), alignment_part_index=part_index,
                        original_token_index=original_index,
                        original_part_character_spans=copy.deepcopy(spans),
                        original_character_spans=[[a + character_offset, b + character_offset] for a, b in spans])
            for key in ('raw', 'repaired'):
                _rebase_time(item, key, start, alignment['source_offset_seconds'])
            flags = item.get('flags')
            if not isinstance(flags, list) or any(not isinstance(f, str) for f in flags):
                raise ValueError('Expected explicit token flags')
            seam = (part_index > 0 and original_index == 0) or (
                part_index < len(parts) - 1 and original_index == len(child_items) - 1)
            for applies, flag in ((seam, 'alignment_chunk_edge'), (gated, 'alignment_chunk_repair_gate')):
                if applies:
                    if flag not in flags:
                        flags.append(flag)
                    item['eligible_for_boundary_candidate'] = False
            items.append(item)
        metadata.append({'alignment_part_index': part_index, 'start': start, 'end': end,
                         'source_offset_seconds': alignment['source_offset_seconds'],
                         'original_text_span': [character_offset, character_offset + len(text)],
                         'token_index_span': [token_start, len(items)],
                         'tokens': len(child_items), 'repaired_tokens': repaired,
                         'repaired_token_fraction': repair_fraction, 'repair_gate': gated,
                         'source': copy.deepcopy(part.get('source'))})
        children.append(copy.deepcopy(part))
        texts.append(text)
        character_offset += len(text) + 1
        expected_start = end
    if not _same(expected_start, duration):
        raise ValueError('Last part must end at the full composition duration')
    return {
        'schema_version': 1, 'kind': 'derived_composite_of_independent_forced_alignments',
        'source_offset_seconds': source_offset, 'duration_seconds': duration,
        'mapping': {'original_text': '\n'.join(texts)}, 'items': items,
        'composition_parts': metadata, 'raw_children': children,
        'composition_rules': {'separator': '\n', 'separator_is_transcribed_content': False,
                              'separator_has_timestamp': False, 'single_model_inference': False,
                              'offset_unit': 'Unicode code points; end exclusive',
                              'extent_tolerance_seconds': TOLERANCE_SECONDS,
                              'child_repair_gate_fraction': REPAIR_GATE_FRACTION,
                              'repair_gate_test': 'strictly greater; heuristic diagnostic, not accuracy',
                              'audio_or_diarization_modified': False},
        'confidence_provided': False, 'human_review_required': True,
    }
