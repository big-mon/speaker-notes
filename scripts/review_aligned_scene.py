"""Make a reviewable Qwen/forced-alignment experiment from immutable outputs.

No inference, text correction, or human quality decisions are performed here.
The native Apple transcript is retained only as the comparison reading.
"""
import argparse
import json
import os
from pathlib import Path

from align_qwen import fingerprint
from aligned_speaker_turns import build_aligned_turns
from scene_transcript import save
from text_anchor_audit import audit


def build_review(alignment_dir, diarization_path, apple_path, qwen_path, original, output):
    from scene_pipeline import read_float_wav
    alignment = json.loads((alignment_dir / 'alignment.json').read_text())
    manifest = json.loads((alignment_dir / 'manifest.json').read_text())
    if manifest['status'] != 'completed':
        raise ValueError('Alignment did not complete')
    audio = Path(manifest['inputs']['audio']['path'])
    if fingerprint(audio)['sha256'] != manifest['inputs']['audio']['sha256']:
        raise ValueError('Alignment audio changed')
    qwen = json.loads(qwen_path.read_text())
    if qwen.get('token_limit_reached'):
        raise ValueError('Qwen reached its token limit')
    if qwen['raw']['text'] != alignment['mapping']['original_text']:
        raise ValueError('Alignment text differs from the original Qwen result')
    if read_float_wav(audio) != read_float_wav(qwen['file']):
        raise ValueError('Qwen and alignment input PCM differ')
    apple = json.loads(apple_path.read_text())
    diarization = json.loads(diarization_path.read_text())
    sources = {name: fingerprint(path) for name, path in {
        'qwen': qwen_path, 'apple': apple_path, 'diarization': diarization_path,
        'alignment': alignment_dir/'alignment.json', 'alignment_manifest': alignment_dir/'manifest.json',
        'assembler': Path(__file__).with_name('aligned_speaker_turns.py'),
        'review_builder': Path(__file__)}.items()}
    review_audio = {**fingerprint(audio), 'src': os.path.relpath(audio, output),
                    'source_offset': alignment['source_offset_seconds']}
    return make_document(alignment, diarization, apple, fingerprint(original), sources, review_audio)


def difference_overlaps(span, start, end, text_length):
    left, right = span
    if left == right:
        # Apple-only text belongs to the following row; a terminal deletion
        # belongs to the last row. Never link both sides of a boundary.
        return start <= left < end or (left == text_length == end)
    return left < end and right > start


def make_document(alignment, diarization, apple, original_metadata, raw_sources, review_audio):
    """Render one global diarization with aligned Qwen text, including composites."""
    candidates = build_aligned_turns(alignment, diarization)
    original_text = alignment['mapping']['original_text']
    comparison = audit(apple, original_text)
    mapping = {}
    for event in sorted(diarization['segments'], key=lambda e: (e['start'], e['end'])):
        mapping.setdefault(str(event['speaker']), chr(65 + len(mapping)))
    offset, duration = alignment['source_offset_seconds'], alignment['duration_seconds']
    rows = []
    for raw in candidates['segments']:
        a, b = raw['source_raw_span']
        differences = [i for i, d in enumerate(comparison['differences'])
                       if difference_overlaps(d['qwen_raw_span'], a, b, len(original_text))]
        speakers = [mapping[str(s)] for s in raw['speakers']]
        mixed = bool(raw['raw_overlap_intervals']) or 'other_speaker_activity_ownership_unresolved' in raw['flags']
        state = ('mixed' if mixed else 'single') if speakers else 'unknown'
        rows.append(dict(raw, id=str(len(rows)), speakers=speakers, state=state,
                         playback_start=max(offset, raw['start'] - 2),
                         playback_end=min(offset + duration, raw['end'] + 2),
                         asr_difference_ids=differences,
                         review_reasons=['human_content_speaker_and_boundary_review_pending'] + raw['flags'] +
                            (['asr_texts_differ; aligned_text_is_not_a_reference'] if differences else []),
                         quality={k: 'review_required' for k in ('content', 'speaker', 'boundary')},
                         ready_for_attributed_summary=False))
    document = {
        'schema_version': 1, 'input': original_metadata,
        'source_window': {'start': offset, 'end': offset + duration},
        'asr': {'engine': 'Qwen3-ASR-1.7B BF16', 'text_modified': False,
                'secondary': 'Apple SpeechTranscriber; comparison only',
                'alignment': 'Qwen3-ForcedAligner-0.6B BF16'},
        'text_source_note': 'Qwen原文を保持しています。Appleは比較表示のみ。名前の誤認や短い応答の話者は未修正です。',
        'speaker_mapping': mapping, 'speaker_names': {v: '話者' + v for v in mapping.values()},
        'speaker_identity_scope': 'one supplied diarization invocation; first appearance IDs; not human verified',
        'segments': rows, 'comparison': comparison,
        'raw_sources': raw_sources,
        'review_audio': review_audio,
        'rules': candidates.get('rules', candidates.get('parameters', {})),
        'processing': {'mode': 'existing outputs only; forced-aligned Qwen text with tentative speaker turns'},
        'alignment_candidates': candidates,
        'review': {'human_verified': False, 'windows': [], 'notes': []},
        'ready_for_attributed_summary': False,
        'validation': {'asr_text_preserved': ''.join(r['text'] for r in rows) == original_text,
                       'accuracy_verified': False},
    }
    if not document['validation']['asr_text_preserved']:
        raise ValueError('Assembly did not preserve Qwen text')
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('alignment_dir', 'diarization', 'apple', 'qwen', 'original', 'output'):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    document = build_review(args.alignment_dir, args.diarization, args.apple, args.qwen,
                            args.original, args.output.resolve())
    save(document, args.output)
    print(args.output)


if __name__ == '__main__':
    main()
