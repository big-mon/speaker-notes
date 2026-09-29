"""VAD-pause-first ASR partitioning without trimming or acoustic truth claims.

This is a recorded local proposal, not an upstream model default. VAD low-score
runs are candidate pauses only. When none is usable, the existing Qwen
energy cut remains an explicitly unresolved fallback.
"""
import array
import math
import sys

from asr_chunking import partition_for_asr


PAUSE_THRESHOLD = .35
MIN_PAUSE_SECONDS = .16


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _pause_candidates(vad, samples, rate):
    if not isinstance(vad, dict) or type(vad.get('sample_rate')) is not int or vad['sample_rate'] != rate:
        raise ValueError('VAD sample_rate must match PCM sample rate')
    if type(vad.get('sample_count')) is not int or vad['sample_count'] != samples:
        raise ValueError('VAD sample_count must exactly match PCM length')
    frames = vad.get('frames')
    if not isinstance(frames, list) or not frames:
        raise ValueError('VAD requires nonempty contiguous frames')
    cursor, frame_width, runs = 0, None, []
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError('Each VAD frame must be a record')
        start, end, score = (frame.get(k) for k in ('start_sample', 'end_sample', 'probability'))
        if (type(start) is not int or type(end) is not int or start != cursor or
                not 0 <= start < end <= samples):
            raise ValueError('VAD frames must cover PCM exactly once in order, with no gaps or overlaps')
        width = end - start
        if frame_width is None:
            frame_width = width
        if width != frame_width and (index != len(frames) - 1 or width > frame_width):
            raise ValueError('VAD frame widths must be fixed, except for a shorter final frame')
        if not _finite(score) or not 0 <= score <= 1:
            raise ValueError('VAD probability must be finite and between zero and one')
        if score < PAUSE_THRESHOLD:
            if runs and runs[-1]['end_sample'] == start:
                runs[-1]['end_sample'] = end
                runs[-1]['frame_indices'].append(index)
                runs[-1]['frame_probabilities'].append(score)
            else:
                runs.append({'start_sample': start, 'end_sample': end,
                             'frame_indices': [index], 'frame_probabilities': [score]})
        cursor = end
    if cursor != samples:
        raise ValueError('VAD frames do not cover the complete PCM input')
    candidates, edges = [], []
    for run in runs:
        a, b = run['start_sample'], run['end_sample']
        if b - a < math.ceil(MIN_PAUSE_SECONDS * rate):
            continue
        item = dict(run, start=a / rate, end=b / rate, duration=(b - a) / rate,
                    midpoint_sample=(a + b) // 2, midpoint=((a + b) // 2) / rate,
                    verified_silence=False, verified_speech_end=False)
        if a == 0 or b == samples:
            edges.append(item)
        else:
            candidates.append(dict(item, id=len(candidates)))
    return candidates, edges, frame_width


def partition_at_vad_pauses(pcm, vad, rate=16000, max_seconds=None):
    """Return ``parts`` / ``policy`` compatible with ``partition_for_asr``.

    The default maximum is 180 s for Qwen. Prefer the latest candidate midpoint
    in [half maximum, maximum] relative to the current start. If unavailable,
    consider earlier midpoints after min(5 s, maximum/4). A remainder fitting the
    maximum is kept whole. Every PCM sample is retained exactly once.
    """
    if type(rate) is not int or rate <= 0:
        raise ValueError('rate must be a positive integer')
    if not isinstance(pcm, (bytes, bytearray)) or not pcm or len(pcm) % 4:
        raise ValueError('Expected nonempty little-endian float32 mono PCM bytes')
    default = 180.
    requested = default if max_seconds is None else max_seconds
    # Reuse the energy fallback's supported rate / maximum validation and
    # parameter metadata, without unnecessarily partitioning the whole input.
    energy_policy = partition_for_asr(pcm[:4], rate, requested)['policy']
    maximum = energy_policy['maximum_samples']
    values = array.array('f')
    if values.itemsize != 4:
        raise RuntimeError('This platform does not provide 32-bit array floats')
    values.frombytes(pcm)
    if sys.byteorder != 'little':
        values.byteswap()
    if any(not math.isfinite(value) for value in values):
        raise ValueError('PCM contains a nonfinite sample')
    samples = len(values)
    del values
    candidates, edges, frame_width = _pause_candidates(vad, samples, rate)
    minimum_early = math.ceil(min(5 * rate, maximum / 4))
    parts, cursor, fallback_count, early_count = [], 0, 0, 0
    while cursor < samples:
        if samples - cursor <= maximum:
            end, details = samples, {'boundary_kind': 'input_end', 'unresolved': False,
                                    'verified_speech_end': False}
        else:
            lo, hi = cursor + (maximum + 1) // 2, cursor + maximum
            preferred = [c for c in candidates if lo <= c['midpoint_sample'] <= hi]
            early = [c for c in candidates if cursor + minimum_early <= c['midpoint_sample'] < lo]
            selected = max(preferred or early, key=lambda c: c['midpoint_sample'], default=None)
            if selected:
                end = selected['midpoint_sample']
                is_early = not preferred
                early_count += is_early
                details = {'boundary_kind': 'vad_pause_early' if is_early else 'vad_pause_preferred',
                           'selected_pause_candidate': dict(selected), 'selected_early': is_early,
                           'preferred_search_start_sample': lo, 'preferred_search_end_sample': hi,
                           'early_search_start_sample': cursor + minimum_early,
                           'unresolved': False, 'verified_speech_end': False}
            else:
                # The first energy cut only examines this bounded prefix of the
                # remainder. The extra sample forces a cut rather than input_end;
                # its first cut equals an energy run over the full remainder.
                prefix = pcm[cursor * 4:(cursor + maximum + 1) * 4]
                fallback = partition_for_asr(prefix, rate, requested)['parts'][0]
                end = cursor + fallback['end_sample']
                details = {k: v for k, v in fallback.items()
                           if k not in ('index', 'start_sample', 'end_sample', 'start', 'end', 'samples', 'duration', 'boundary_kind')}
                details = {k: v + cursor if k.endswith('_sample') else v for k, v in details.items()}
                details.update(boundary_kind='forced_low_energy_no_vad_pause', unresolved=True,
                               verified_speech_end=False, energy_boundary_kind=fallback['boundary_kind'],
                               fallback_input_start_sample=cursor, fallback_input_samples=len(prefix) // 4,
                               preferred_search_start_sample=lo, preferred_search_end_sample=hi,
                               early_search_start_sample=cursor + minimum_early)
                fallback_count += 1
            if not cursor < end <= min(samples, cursor + maximum):
                raise RuntimeError('Cut violated progress or strict maximum')
        parts.append({'index': len(parts), 'start_sample': cursor, 'end_sample': end,
                      'start': cursor / rate, 'end': end / rate,
                      'samples': end - cursor, 'duration': (end - cursor) / rate, **details})
        cursor = end
    fallback_configuration = {k: v for k, v in energy_policy.items() if k != 'source_sample_count'}
    return {'parts': parts, 'policy': {
        'version': 1, 'name': 'vad_pause_first_with_visible_energy_fallback', 'engine': 'qwen',
        'sample_rate': rate, 'source_sample_count': samples, 'vad_frame_samples': frame_width,
        'default_maximum_seconds': default, 'maximum_override_seconds': max_seconds,
        'maximum_seconds': maximum / rate, 'maximum_samples': maximum,
        'pause_probability_threshold': PAUSE_THRESHOLD, 'pause_comparison': 'strictly less than threshold',
        'minimum_pause_seconds': MIN_PAUSE_SECONDS, 'minimum_pause_samples': math.ceil(MIN_PAUSE_SECONDS * rate),
        'preferred_span': 'current start + [maximum/2, maximum]; latest midpoint',
        'earlier_span_minimum_samples': minimum_early, 'earlier_span_minimum_seconds': minimum_early / rate,
        'earlier_span': 'consider only if preferred span has no pause; latest midpoint',
        'pause_candidates': candidates, 'excluded_edge_low_runs': edges,
        'fallback_count': fallback_count, 'early_pause_count': early_count,
        'energy_fallback_configuration': fallback_configuration,
        'energy_fallback_input': 'bounded prefix of remainder, maximum + one sample; same first cut as full remainder',
        'coverage': 'all source samples exactly once; contiguous, no overlap, gaps, trimming or padding',
        'vad_used_for_cuts': True, 'verified_speech_end': False, 'confidence_provided': False,
        'probability_meaning': 'raw VAD model speech scores, not boundary correctness confidence',
        'unresolved_meaning': 'no qualifying VAD pause for this cut; false does not mean acoustic verification',
        'text_processing': 'none', 'time_granularity': 'source input extents; not speech or word alignment',
        'local_adaptation': 'recorded pause thresholds and Qwen 180 s cap; not an upstream universal best practice',
    }}


def main(argv=None):
    """Create reproducible ASR inputs; this CLI does not perform inference."""
    import argparse
    import json
    from pathlib import Path

    from scene_pipeline import fingerprint, read_float_wav, write_float_wav

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audio', type=Path)
    parser.add_argument('vad', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--max-seconds', type=float)
    args = parser.parse_args(argv)
    audio, vad_path, output = args.audio.resolve(), args.vad.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    record = json.loads(vad_path.read_text())
    source = fingerprint(audio)
    if (not isinstance(record.get('file'), str) or Path(record['file']).resolve() != audio or
            record.get('sha256') != source['sha256']):
        raise ValueError('VAD file path and whole-WAV SHA-256 must match the input audio exactly')
    rate, data = read_float_wav(audio)
    plan = partition_at_vad_pauses(data, record, rate, args.max_seconds)
    plan.update(source=source, vad=fingerprint(vad_path),
                validation={'source_audio_identity_verified': True, 'sample_exact_contiguous_coverage': True,
                            'no_asr_inference': True, 'no_human_accuracy_verification': True})
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / 'inputs'
    inputs.mkdir()
    paths = []
    for part in plan['parts']:
        path = inputs / f'{part["index"]:03d}.wav'
        write_float_wav(path, data[part['start_sample'] * 4:part['end_sample'] * 4], rate)
        part['file'] = fingerprint(path)
        paths.append(path)
    (output / 'inputs.txt').write_text(''.join(str(path) + '\n' for path in paths))
    (output / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'parts': len(plan['parts']), 'fallback_count': plan['policy']['fallback_count'],
                      'plan': str(output / 'plan.json')}, ensure_ascii=False))
    return plan


if __name__ == '__main__':
    main()
