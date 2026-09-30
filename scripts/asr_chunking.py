"""Qwen low-energy cuts, with exact source-sample coverage.

The returned intervals describe existing little-endian float32 mono PCM. This
module neither rewrites audio nor handles text, VAD, ASR, or speaker identities.
Low energy is a cut candidate, not proof of silence or a completed utterance.
"""
import array
import math
import sys


SOURCE = {
    'revision': '7c6daf77a2421100f5fb066495372c00129d39ff',
    'url': 'https://github.com/QwenLM/Qwen3-ASR/blob/7c6daf77a2421100f5fb066495372c00129d39ff/qwen_asr/inference/utils.py',
    'function': 'split_audio_into_chunks',
    'license': 'Apache-2.0',
}


def _qwen_cut(values, left, right, width, target):
    """Minimum absolute-amplitude sum, then minimum sample within that window."""
    if right - left <= width:
        return target, {'boundary_kind': 'target_search_too_short'}
    magnitudes = [abs(value) for value in values[left:right]]
    energy = best_energy = math.fsum(magnitudes[:width])
    best = 0
    for first in range(1, len(magnitudes) - width + 1):
        energy += magnitudes[first + width - 1] - magnitudes[first - 1]
        if energy < best_energy:
            best, best_energy = first, energy
    inner = min(range(best, best + width), key=magnitudes.__getitem__)
    return left + inner, {
        'boundary_kind': 'qwen_minimum_absolute_sum_then_minimum_sample',
        'quiet_window_start_sample': left + best,
        'quiet_window_end_sample': left + best + width,
        'boundary_window_absolute_sum': best_energy,
        'boundary_sample_absolute_amplitude': magnitudes[inner],
    }


def partition_for_asr(pcm, rate=16000, max_seconds=None):
    """Return policy and ordered disjoint sample intervals for Qwen.

    Default strict maximum 180 s, target 175 s, search 170–180 s.
    A maximum override is recorded for controlled comparisons. Every sample, including short tails and silence,
    is retained once. Times are input extents, not predicted speech timestamps.
    """
    if type(rate) is not int or rate <= 0:
        raise ValueError('rate must be a positive integer')
    if not isinstance(pcm, (bytes, bytearray)) or not pcm or len(pcm) % 4:
        raise ValueError('Expected nonempty little-endian float32 mono PCM bytes')
    default = 180.
    requested_maximum = default if max_seconds is None else max_seconds
    if (isinstance(requested_maximum, bool) or
            not isinstance(requested_maximum, (int, float)) or
            not math.isfinite(requested_maximum)):
        raise ValueError('max_seconds must be a finite number')
    minimum = 10
    if requested_maximum <= minimum:
        raise ValueError(f'Qwen max_seconds must exceed its {minimum}-second search span')
    sample_limit = requested_maximum * rate
    if not math.isfinite(sample_limit):
        raise ValueError('max_seconds is too large at this sample rate')
    maximum = math.floor(sample_limit)
    search = 5 * rate
    if maximum <= minimum * rate:
        raise ValueError('max_seconds is too small after conversion to samples')
    values = array.array('f')
    if values.itemsize != 4:
        raise RuntimeError('This platform does not provide 32-bit array floats')
    values.frombytes(pcm)
    if sys.byteorder != 'little':
        values.byteswap()
    if any(not math.isfinite(value) for value in values):
        raise ValueError('PCM contains a nonfinite sample')

    width = max(4, int(.1 * rate))
    target = maximum - search
    adaptations = [
        'Return source sample intervals; never concatenate nonadjacent audio or alter PCM.',
        'Reject nonfinite samples across the entire input, including outside boundary searches.',
        'Keep short tails at their actual length; no dropped samples or added zero padding.',
        'Use Python float64 energy arithmetic, not NumPy/PyTorch float32; near-tied cuts can differ.',
        'Convert requested strict maximum to whole samples by rounding down.',
        'Use strict maximum 180 s by default for the local alignment route; official ordinary ASR target is 1200 s and forced-alignment target is 180 s.',
        'Set search target to strict maximum minus 5 s, so target ±5 s stays inside the maximum.',
        'If the remainder already fits the strict maximum, retain it as one final interval.',
    ]
    if max_seconds is not None:
        adaptations.append('Explicit maximum override for a controlled comparison; not a newly recommended model limit.')
    policy = {
        'version': 1, 'engine': 'qwen', 'source': dict(SOURCE),
        'local_adaptations': adaptations,
        'sample_rate': rate, 'source_sample_count': len(values),
        'default_maximum_seconds': default,
        'maximum_override_seconds': max_seconds,
        'maximum_seconds': maximum / rate, 'maximum_samples': maximum,
        'target_seconds': target / rate,
        'search_before_target_seconds': 5.,
        'search_after_target_seconds': 5.,
        'energy_window_requested_seconds': .1,
        'energy_window_samples': width, 'energy_window_seconds': width / rate,
        'window_step_samples': 1,
        'tie_break': 'first minimum window, then first minimum sample for Qwen',
        'coverage': 'all source samples exactly once; contiguous, no overlap, gaps or padding',
        'vad_used_for_cuts': False, 'verified_speech_end': False,
        'text_processing': 'none', 'confidence_provided': False,
        'time_granularity': 'source input extents; not speech or word alignment',
    }
    parts, cursor = [], 0
    while cursor < len(values):
        if len(values) - cursor <= maximum:
            end, details = len(values), {'boundary_kind': 'input_end'}
        else:
            left, right = cursor + target - search, cursor + maximum
            end, details = _qwen_cut(values, left, right, width, cursor + target)
            details.update(search_start_sample=left, search_end_sample=right,
                           target_sample=cursor + target)
        if not cursor < end <= min(len(values), cursor + maximum):
            raise RuntimeError('Cut violated progress or strict maximum')
        parts.append({'index': len(parts), 'start_sample': cursor, 'end_sample': end,
                      'start': cursor / rate, 'end': end / rate,
                      'samples': end - cursor, 'duration': (end - cursor) / rate,
                      **details})
        cursor = end
    return {'policy': policy, 'parts': parts}
