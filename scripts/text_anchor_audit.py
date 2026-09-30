"""Compare two ASR texts without turning text matches into acoustic alignment.

Apple audioTimeRange values are retained only as envelopes of the original
source units. Qwen insertions/replacements receive no invented timestamps.
This module never selects a winning transcript or edits either transcript.
"""
import argparse
import difflib
import json
import math
import pathlib
import re
import unicodedata


def _normalized(text):
    """NFKC clusters and raw offsets; retain punctuation that can change a number."""
    characters, offsets = [], []
    index = 0
    while index < len(text):
        end = index + 1
        while end < len(text) and (
            unicodedata.combining(text[end]) or text[end] in "\uff9e\uff9f"
        ):
            end += 1
        for character in unicodedata.normalize("NFKC", text[index:end]):
            numeric_neighbor = ((index > 0 and text[index - 1].isnumeric()) or
                                (end < len(text) and text[end].isnumeric()))
            numeric_punctuation = character in ".,:-/+%" and numeric_neighbor
            if not character.isspace() and (numeric_punctuation or
                                           not unicodedata.category(character).startswith("P")):
                characters.append(character)
                offsets.append((index, end))
        index = end
    return "".join(characters), offsets


def _raw_span(offsets, start, end, text_length):
    if start == end:
        boundary = offsets[start][0] if start < len(offsets) else text_length
        return [boundary, boundary]
    return [offsets[start][0], offsets[end - 1][1]]


def _valid_time(unit):
    start, end = unit.get("start"), unit.get("end")
    return all(type(value) in (int, float) and math.isfinite(value)
               for value in (start, end)) and 0 <= start < end


def _apple_text(segments, clip_start, clip_end):
    text, units, untimed_dropped = [], [], []
    dropped_count = 0
    cursor = 0
    for segment_index, segment in enumerate(segments):
        if "units" in segment:
            original_units = segment["units"]
            if "text" in segment and "".join(u["text"] for u in original_units) != segment["text"]:
                raise ValueError("Apple units must preserve the original segment text")
        else:
            # A long segment is not silently promoted to fine timing.
            original_units = [{"text": segment["text"], "start": None, "end": None}]
        for unit_index, original in enumerate(original_units):
            unit = dict(original, segment_index=segment_index, unit_index=unit_index)
            if _valid_time(unit):
                midpoint = (unit["start"] + unit["end"]) / 2
                if ((clip_start is not None and midpoint < clip_start) or
                        (clip_end is not None and midpoint >= clip_end)):
                    dropped_count += 1
                    continue
            elif clip_start is not None or clip_end is not None:
                # Inclusion of an untimed segment cannot be established by a clip filter.
                dropped_count += 1
                untimed_dropped.append([segment_index, unit_index])
                continue
            unit["raw_span"] = [cursor, cursor + len(unit["text"])]
            units.append(unit)
            text.append(unit["text"])
            cursor += len(unit["text"])
    return "".join(text), units, {"count": dropped_count, "untimed_unit_references": untimed_dropped}


def _unit_envelope(units, raw_span):
    selected = [u for u in units if u["raw_span"][0] < raw_span[1]
                and u["raw_span"][1] > raw_span[0]]
    references = [[u["segment_index"], u["unit_index"]] for u in selected]
    if not selected or not all(_valid_time(u) for u in selected):
        return None, references, "missing_or_invalid_source_time"
    if any(right["start"] < left["start"] or right["end"] < left["end"]
           for left, right in zip(selected, selected[1:])):
        return None, references, "non_monotonic_source_times"
    return [selected[0]["start"], selected[-1]["end"]], references, "native_unit_envelope"


def _occurrences(text, needle):
    return [match.start() for match in re.finditer("(?=" + re.escape(needle) + ")", text)]


def _term_spans(text, terms):
    return [(start, start + len(term), term) for term in terms if term
            for start in _occurrences(text, term)]


def _intersects(start, end, term_start, term_end):
    # An inserted token inside a term can change the term even though its source span is empty.
    return (term_start < start < term_end if start == end
            else start < term_end and end > term_start)


def audit(apple_segments, qwen_text, *, clip_start=None, clip_end=None,
          critical_terms=(), minimum_anchor_chars=4):
    """Return exact text anchors, differences and unanchored ranges for review.

    clip_start/end are in the Apple source clock. Timed units are selected by
    midpoint, so a unit straddling adjacent clips is not silently duplicated.
    Without a clip filter all text, including untimed units, is preserved.
    critical_terms are caller-supplied names/terms, never inferred corrections.
    """
    if minimum_anchor_chars < 1:
        raise ValueError("minimum_anchor_chars must be positive")
    if any(value is not None and (not math.isfinite(value) or value < 0)
           for value in (clip_start, clip_end)):
        raise ValueError("Clip bounds must be nonnegative finite seconds")
    if clip_start is not None and clip_end is not None and clip_end <= clip_start:
        raise ValueError("clip_end must follow clip_start")
    apple_text, units, excluded = _apple_text(apple_segments, clip_start, clip_end)
    apple, apple_offsets = _normalized(apple_text)
    qwen, qwen_offsets = _normalized(qwen_text)
    matcher = difflib.SequenceMatcher(None, apple, qwen, autojunk=False)
    anchors, rejected, differences = [], [], []
    masks = {"apple": [False] * len(apple), "qwen": [False] * len(qwen)}
    critical = [_normalized(term)[0] for term in critical_terms]
    negation = ["ない", "なく", "なかった", "ません", "ぬ", "ず", "違う", "違い", "不要", "不可能", "以外"]
    term_sets = {
        "critical_term_difference": (_term_spans(apple, critical), _term_spans(qwen, critical)),
        "negation_difference_candidate": (_term_spans(apple, negation), _term_spans(qwen, negation)),
        "number_or_numeric_character_difference": tuple(
            [(match.start(), match.end(), match.group())
             for match in re.finditer(r"[-+]?\d+(?:[.,:/-]\d+)*(?:%)?", text)]
            for text in (apple, qwen)
        ),
    }
    for operation, a0, a1, q0, q1 in matcher.get_opcodes():
        apple_span = _raw_span(apple_offsets, a0, a1, len(apple_text))
        qwen_span = _raw_span(qwen_offsets, q0, q1, len(qwen_text))
        row = {
            "apple_raw_span": apple_span, "qwen_raw_span": qwen_span,
            "apple_text": apple_text[slice(*apple_span)],
            "qwen_text": qwen_text[slice(*qwen_span)],
        }
        if operation != "equal":
            signals = []
            signal_evidence = {}
            for label, (apple_terms, qwen_terms) in term_sets.items():
                if (any(_intersects(a0, a1, s, e) for s, e, _ in apple_terms) or
                        any(_intersects(q0, q1, s, e) for s, e, _ in qwen_terms)):
                    signals.append(label)
            if ("number_or_numeric_character_difference" not in signals and
                    any(character.isnumeric() for character in apple[a0:a1] + qwen[q0:q1])):
                signals.append("number_or_numeric_character_difference")
            if operation in ("insert", "replace") and q1 - q0 > 20 and a1 - a0 <= 5:
                signals.append("potential_substantive_omission")
                signal_evidence["potential_substantive_omission"] = {
                    "apple_normalized_character_count": a1 - a0,
                    "qwen_normalized_character_count": q1 - q0,
                    "criterion": "Qwen differing span >20 normalized characters and Apple span <=5",
                    "criterion_is_heuristic": True,
                    "error_confirmed": False,
                    "qwen_correctness_verified": False,
                    "requires_audio_review": True,
                }
            time_range, references, timing = _unit_envelope(units, apple_span)
            differences.append(dict(row, operation=operation, review_signals=signals,
                                    apple_time_range=time_range, apple_unit_references=references,
                                    timing_status=timing,
                                    review_signal_evidence=signal_evidence,
                                    apple_normalized_span=[a0, a1], qwen_normalized_span=[q0, q1],
                                    qwen_time_range=None))
            continue
        reason = None
        if a1 - a0 < minimum_anchor_chars:
            reason = "short_exact_match_without_sufficient_context"
        elif len(_occurrences(apple, apple[a0:a1])) != 1 or len(_occurrences(qwen, qwen[q0:q1])) != 1:
            reason = "repeated_exact_phrase_ambiguous"
        if reason:
            rejected.append(dict(row, reason=reason, apple_time_range=None))
            continue
        time_range, references, timing = _unit_envelope(units, apple_span)
        anchors.append(dict(row, id=len(anchors), apple_time_range=time_range,
                            apple_unit_references=references, timing_status=timing,
                            apple_normalized_span=[a0, a1], qwen_normalized_span=[q0, q1]))
        masks["apple"][a0:a1] = [True] * (a1 - a0)
        masks["qwen"][q0:q1] = [True] * (q1 - q0)

    unanchored = {}
    for side, offsets, raw_text in (("apple", apple_offsets, apple_text), ("qwen", qwen_offsets, qwen_text)):
        gaps, index = [], 0
        while index < len(masks[side]):
            if masks[side][index]:
                index += 1
                continue
            end = index + 1
            while end < len(masks[side]) and not masks[side][end]:
                end += 1
            span = _raw_span(offsets, index, end, len(raw_text))
            gaps.append({"raw_span": span, "text": raw_text[slice(*span)], "time_range": None})
            index = end
        unanchored[side] = gaps
    for difference in differences:
        before = [a["id"] for a in anchors if a["qwen_normalized_span"][1] <= difference["qwen_normalized_span"][0]]
        after = [a["id"] for a in anchors if a["qwen_normalized_span"][0] >= difference["qwen_normalized_span"][1]]
        difference["neighboring_anchor_ids"] = {"before": before[-1] if before else None,
                                                 "after": after[0] if after else None}
    return {
        "schema_version": 1,
        "method": "monotonic unique exact text anchors; not acoustic forced alignment",
        "normalization": "NFKC clusters; whitespace and punctuation except numeric separators/signs ignored; no phonetic corrections",
        "minimum_anchor_chars": minimum_anchor_chars,
        "clip": {"start": clip_start, "end": clip_end, "apple_unit_selection": "midpoint"},
        "apple_text": apple_text, "qwen_text": qwen_text,
        "excluded_apple_units": excluded,
        "anchors": anchors, "rejected_anchors": rejected,
        "differences": differences, "unanchored": unanchored,
        "critical_terms": list(critical_terms),
        "limitations": [
            "Text agreement is not acoustic correctness or speaker correctness.",
            "Anchor ranges are whole native Apple source-unit envelopes, not Qwen word times.",
            "Missing or differing text receives no extrapolated/interpolated timestamp.",
            "Review signals are heuristic candidates, not a completeness or accuracy score.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("apple_json", help="Apple transcript-segments.json")
    parser.add_argument("qwen_text", help="UTF-8 text for the same audio clip")
    parser.add_argument("output", help="New JSON file; existing files are not overwritten")
    parser.add_argument("--clip-start", type=float)
    parser.add_argument("--clip-end", type=float)
    parser.add_argument("--critical-term", action="append", default=[])
    args = parser.parse_args()
    document = audit(json.loads(pathlib.Path(args.apple_json).read_text()),
                     pathlib.Path(args.qwen_text).read_text(), clip_start=args.clip_start,
                     clip_end=args.clip_end, critical_terms=args.critical_term)
    with pathlib.Path(args.output).open("x") as output:
        json.dump(document, output, ensure_ascii=False, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
