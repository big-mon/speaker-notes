"""Offline Qwen forced-alignment probe; original text and raw times stay separate.

Requires a prepared local model and nagisa in the selected Python environment.
MLX/numpy/scipy imports occur only in run(), so audit helpers need no MLX.
Character offsets are Python Unicode code-point offsets, end exclusive.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import difflib
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback
import unicodedata


def kept_character(character):
    # Match mlx-audio 0.5.7 ForceAlignProcessor.clean_token, without NFKC.
    return character == "'" or unicodedata.category(character)[0] in 'LN'


def character_spans(indices):
    spans = []
    for index in indices:
        if spans and spans[-1][1] == index:
            spans[-1][1] += 1
        else:
            spans.append([index, index + 1])
    return spans


def map_tokens(text, words):
    """Map exact tokenizer characters, retaining every other original character.

    When tokenization changes lexical characters, only whole tokens inside a
    unique exact matching block receive offsets. Repeated partial matches and
    transformed tokens remain unmapped; no spelling/phonetic repair is applied.
    This is a text coverage check, not an acoustic alignment quality measure.
    """
    if not isinstance(text, str) or any(not isinstance(w, str) or not w for w in words):
        raise ValueError('Expected original text and nonempty tokenizer strings')
    positions = [i for i, char in enumerate(text) if kept_character(char)]
    lexical = ''.join(text[i] for i in positions)
    joined = ''.join(words)
    matcher = difflib.SequenceMatcher(None, lexical, joined, autojunk=False)
    exact = lexical == joined
    blocks = [block for block in matcher.get_matching_blocks() if block.size]
    differences = []
    for operation, a, b, c, d in matcher.get_opcodes():
        if operation != 'equal':
            differences.append({
                'operation': operation,
                'original_lexical_span': [a, b],
                'original_character_spans': character_spans(positions[a:b]),
                'original_lexical_text': lexical[a:b],
                'tokenizer_character_span': [c, d],
                'tokenizer_text': joined[c:d],
                'requires_review': True,
            })
    tokens, owners, cursor = [], {}, 0
    for index, word in enumerate(words):
        end = cursor + len(word)
        block = next((b for b in blocks if b.b <= cursor and end <= b.b + b.size), None)
        raw_indices = []
        reason = 'tokenizer_text_changed_or_unmatched'
        if block is not None:
            context = joined[block.b:block.b + block.size]
            if exact or (lexical.count(context) == 1 and joined.count(context) == 1):
                first = block.a + cursor - block.b
                raw_indices = positions[first:first + len(word)]
                reason = 'exact_characters_after_processor_filter'
            else:
                reason = 'repeated_partial_match_ambiguous'
        spans = character_spans(raw_indices)
        envelope = [raw_indices[0], raw_indices[-1] + 1] if raw_indices else None
        for raw_index in raw_indices:
            if raw_index in owners:
                raise ValueError('Tokenizer character mapping overlaps')
            owners[raw_index] = index
        tokens.append({
            'index': index, 'text': word, 'tokenizer_character_span': [cursor, end],
            'mapping_status': reason, 'original_character_spans': spans,
            'original_raw_span': envelope,
            'original_raw_text': text[envelope[0]:envelope[1]] if envelope else None,
            'mapped': bool(raw_indices),
        })
        cursor = end
    pieces = []
    for index, char in enumerate(text):
        owner = owners.get(index)
        if owner is not None:
            kind = 'mapped_lexical_characters'
        elif kept_character(char):
            kind = 'unmapped_lexical_characters'
        elif char.isspace():
            kind = 'untimed_whitespace'
        elif unicodedata.category(char).startswith('P'):
            kind = 'untimed_punctuation'
        else:
            kind = 'untimed_other_removed_character'
        if pieces and (pieces[-1]['kind'], pieces[-1]['token_index']) == (kind, owner):
            pieces[-1]['raw_span'][1] = index + 1
            pieces[-1]['text'] += char
        else:
            pieces.append({'raw_span': [index, index + 1], 'text': char,
                           'kind': kind, 'token_index': owner})
    preserved = ''.join(p['text'] for p in pieces) == text
    if not preserved:
        raise ValueError('Original text partition is not lossless')
    return {
        'offset_unit': 'Unicode code points; end exclusive',
        'filter': "Unicode L/N or ASCII apostrophe; no normalization or phonetic substitution",
        'original_text': text, 'tokenizer_text': joined,
        'tokenizer_matches_filtered_original': exact,
        'tokens': tokens, 'original_text_partition': pieces,
        'tokenizer_differences': differences,
        'coverage': {
            'original_characters': len(text), 'original_lexical_characters': len(lexical),
            'mapped_lexical_characters': len(owners),
            'unmapped_lexical_characters': len(lexical) - len(owners),
            'removed_characters_retained_untimed': len(text) - len(lexical),
            'original_text_preserved': preserved,
            'all_tokens_mapped': all(t['mapped'] for t in tokens),
            'counts_are_not_accuracy_or_confidence': True,
        },
    }


@contextmanager
def capture_parser(processor, captures):
    """Capture classifier times before the installed library repairs them."""
    original = processor.parse_timestamp

    def capture(words, timestamps):
        record = {'words': list(words), 'raw_timestamp_ms': [float(v) for v in timestamps],
                  'repaired_items_ms': None}
        captures.append(record)  # Keep raw values even when the parser fails.
        result = original(words, timestamps)
        record['repaired_items_ms'] = [
            {'text': str(item['text']), 'start_time': float(item['start_time']),
             'end_time': float(item['end_time'])} for item in result]
        return result

    processor.parse_timestamp = capture
    try:
        yield
    finally:
        processor.parse_timestamp = original


def annotate_timestamps(capture, duration, source_offset=0.):
    if not math.isfinite(duration) or duration <= 0 or not math.isfinite(source_offset) or source_offset < 0:
        raise ValueError('Invalid clip duration or source offset')
    words, raw, repaired = capture['words'], capture['raw_timestamp_ms'], capture['repaired_items_ms']
    if len(raw) != 2 * len(words) or repaired is None or len(repaired) != len(words):
        raise ValueError('Expected exactly two raw timestamps and one repaired item per token')
    if any(item['text'] != word for item, word in zip(repaired, words)):
        raise ValueError('Parser changed token sequence')
    rows, previous = [], {'raw': None, 'repaired': None}
    for index, word in enumerate(words):
        raw_pair = [float(v) / 1000 for v in raw[2*index:2*index+2]]
        fixed_pair = [float(repaired[index][k]) / 1000 for k in ('start_time', 'end_time')]
        flags = []
        changed = raw_pair != fixed_pair
        if changed:
            flags.append('timestamp_repaired_by_library')
        for name, pair in (('raw', raw_pair), ('repaired', fixed_pair)):
            a, b = pair
            if not all(math.isfinite(v) for v in pair):
                flags.append(name + '_nonfinite_timestamp')
            else:
                if a < 0 or b > duration or b < 0 or a > duration:
                    flags.append(name + '_outside_audio_extent')
                if a > b:
                    flags.append(name + '_reversed_interval')
                elif a == b:
                    flags.append(name + '_zero_length')
                prev = previous[name]
                if prev is not None and all(math.isfinite(v) for v in prev):
                    if a < prev[0] or b < prev[1]:
                        flags.append(name + '_nonmonotonic_sequence')
                    if a < prev[1]:
                        flags.append(name + '_overlap_previous_token')
            previous[name] = pair
        def shifted(pair):
            return [source_offset + v for v in pair] if all(math.isfinite(v) for v in pair) else None
        rows.append({
            'index': index, 'text': word,
            'raw_clip_seconds': raw_pair, 'repaired_clip_seconds': fixed_pair,
            'raw_source_seconds': shifted(raw_pair),
            'repaired_source_seconds': shifted(fixed_pair),
            'library_repaired': changed, 'flags': flags,
            'unrepaired_timing_checks_pass': not flags,
            'acoustically_verified': False,
        })
    return rows


def fingerprint(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest}


def model_fingerprint(directory):
    files = []
    for path in sorted(directory.rglob('*')):
        if path.is_file():
            files.append(dict(fingerprint(path), relative_path=path.relative_to(directory).as_posix()))
    canonical = [{k: f[k] for k in ('relative_path', 'bytes', 'sha256')} for f in files]
    digest = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {'path': str(directory), 'files': files, 'sha256': digest,
            'sha256_definition': 'SHA256 of UTF-8 canonical JSON list of relative_path, bytes, sha256',
            'bytes': sum(f['bytes'] for f in files)}


def json_safe(value):
    # Nonfinite model output is evidence; valid JSON retains it as an explicit string.
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_json(path, value):
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def process_memory():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    system = platform.system()
    return {'ru_maxrss_native': peak,
            'native_unit': 'bytes' if system == 'Darwin' else 'KiB' if system == 'Linux' else 'platform_native',
            'peak_rss_bytes': peak if system == 'Darwin' else peak * 1024 if system == 'Linux' else None,
            'scope': 'whole Python process lifetime high-water mark, including preparation and model loading; not stage-only RSS'}


def run(args):
    started = time.perf_counter()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'schema_version': 1, 'status': 'preparing',
                'created_at': datetime.now(timezone.utc).isoformat(),
                'source_offset_seconds': args.source_offset, 'language': 'Japanese',
                'audio_or_text_sent_externally': False,
                'text_correctness_and_speaker_identity_verified': False,
                'nonfinite_json_encoding': 'nonfinite numeric values are strings, never fabricated finite values'}
    captures = []
    try:
        if not math.isfinite(args.source_offset) or args.source_offset < 0:
            raise ValueError('--source-offset must be finite and nonnegative')
        model_path, audio_path, text_path = (Path(p).resolve() for p in (args.model, args.audio, args.text))
        if not model_path.is_dir() or not (model_path / 'config.json').is_file():
            raise ValueError('Expected a prepared local model directory with config.json')
        if not any(model_path.glob('*.safetensors')) and not any(model_path.glob('*.npz')):
            raise ValueError('Local model weights are missing; this script never downloads them')
        text_bytes = text_path.read_bytes()
        text = text_bytes.decode('utf-8')
        if not text.strip():
            raise ValueError('Transcript is empty')
        (output / 'text.txt').write_bytes(text_bytes)
        manifest['inputs'] = {'model': model_fingerprint(model_path),
                              'audio': fingerprint(audio_path), 'text': fingerprint(text_path)}
        manifest['runtime'] = {'python': sys.version, 'platform': platform.platform(), 'packages': {}}
        for name in ('mlx-audio', 'mlx', 'numpy', 'scipy', 'transformers', 'nagisa'):
            manifest['runtime']['packages'][name] = metadata.version(name)
        manifest['implementation'] = fingerprint(Path(__file__))
        # Set before importing any MLX/HF module; missing local assets must fail.
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import numpy as np
        from scipy.io import wavfile
        import mlx.core as mx
        from mlx_audio.stt.utils import load_model
        rate, audio = wavfile.read(audio_path)
        if rate != 16000 or audio.ndim != 1 or not len(audio):
            raise ValueError('Expected a nonempty 16 kHz mono WAV; no implicit resampling or downmixing')
        original_dtype = str(audio.dtype)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.
        elif audio.dtype != np.float32:
            raise ValueError('Expected Float32 or int16 WAV; unsupported precision is not silently converted')
        if not np.isfinite(audio).all() or np.max(np.abs(audio)) > 1:
            raise ValueError('Expected finite normalized audio samples in [-1, 1]')
        duration = len(audio) / rate
        manifest['audio_preprocessing'] = {
            'sample_rate': rate, 'channels': 1, 'samples': len(audio), 'duration_seconds': duration,
            'input_dtype': original_dtype, 'model_input_dtype': str(audio.dtype),
            'conversion': 'int16 / 32768 to Float32' if original_dtype == 'int16' else 'none',
            'resampled': False, 'silence_removed': False, 'clip_time_origin': 0.,
            'source_time_rule': 'source_offset_seconds + model clip seconds; applied once',
        }
        manifest['preparation_seconds'] = time.perf_counter() - started
        manifest['status'] = 'loading_model'
        write_json(output / 'manifest.json', manifest)
        mx.reset_peak_memory()
        begin = time.perf_counter()
        model = load_model(model_path, strict=True)
        mx.synchronize()
        manifest['model_load_seconds'] = time.perf_counter() - begin
        manifest['model_load_mlx_peak_memory_bytes'] = mx.get_peak_memory()
        manifest['model_load_process_memory'] = process_memory()
        if getattr(model.config, 'model_type', None) != 'qwen3_forced_aligner':
            raise ValueError('Loaded model is not qwen3_forced_aligner')
        step = float(model.config.timestamp_segment_time)
        classes = int(model.config.classify_num)
        manifest['settings'] = {'model_type': model.config.model_type, 'strict_weights': True,
                                'timestamp_segment_time_ms': step, 'timestamp_classes': classes,
                                'timestamp_class_max_seconds': (classes - 1) * step / 1000,
                                'class_range_is_not_supported_audio_duration': True,
                                'tokenizer': 'installed nagisa via ForceAlignProcessor',
                                'library_timestamp_repair': 'installed parse_timestamp/fix_timestamp; raw preserved separately',
                                'internal_audio_chunking': False}
        if duration > (classes - 1) * step / 1000:
            raise ValueError('Audio exceeds the timestamp class range; supply a shorter matched audio/text clip')
        processor = model.aligner_processor
        source_module = sys.modules[type(processor).__module__]
        manifest['aligner_implementation'] = fingerprint(Path(source_module.__file__))
        manifest['status'] = 'aligning'
        write_json(output / 'manifest.json', manifest)
        mx.reset_peak_memory()
        begin = time.perf_counter()
        with capture_parser(processor, captures):
            result = model.generate(audio=audio, text=text, language='Japanese')
            mx.synchronize()
        manifest['inference_seconds'] = time.perf_counter() - begin
        manifest['inference_mlx_peak_memory_bytes'] = mx.get_peak_memory()
        manifest['mlx_memory_scope'] = 'MLX allocator peak reset before each stage; inference peak includes resident model weights; not total process or system RAM'
        write_json(output / 'raw-alignment.json', {'captures': captures, 'public_items': [
            {'text': item.text, 'start_time': item.start_time, 'end_time': item.end_time}
            for item in result.items]})
        if len(captures) != 1:
            raise ValueError('Expected exactly one parse_timestamp call for this clip')
        mapping = map_tokens(text, captures[0]['words'])
        times = annotate_timestamps(captures[0], duration, args.source_offset)
        for row, token in zip(times, mapping['tokens']):
            row['original_character_spans'] = token['original_character_spans']
            row['mapping_status'] = token['mapping_status']
            row['eligible_for_boundary_candidate'] = row['unrepaired_timing_checks_pass'] and token['mapped']
            if not token['mapped']:
                row['flags'].append('original_text_mapping_unresolved')
        alignment = {'schema_version': 1, 'source_offset_seconds': args.source_offset,
                     'duration_seconds': duration, 'mapping': mapping, 'items': times,
                     'timing_kind': 'forced alignment of supplied text; raw and library-repaired predictions retained',
                     'confidence_provided': False, 'human_review_required': True,
                     'punctuation_and_whitespace_have_no_model_timestamps': True}
        write_json(output / 'alignment.json', alignment)
        write_json(output / 'validation.json', {
            'text_bytes_preserved': (output / 'text.txt').read_bytes() == text_bytes,
            'coverage': mapping['coverage'], 'tokens': len(times),
            'repaired_tokens': sum(row['library_repaired'] for row in times),
            'flagged_tokens': sum(bool(row['flags']) for row in times),
            'boundary_candidate_tokens': sum(row['eligible_for_boundary_candidate'] for row in times),
            'accuracy_or_audio_content_coverage_established': False,
        })
        manifest['status'] = 'completed'
    except (Exception, KeyboardInterrupt) as error:
        manifest['failed_stage'] = manifest['status']
        manifest['status'] = 'cancelled' if isinstance(error, KeyboardInterrupt) else 'failed'
        manifest['error'] = {'type': type(error).__name__, 'message': str(error)}
        (output / 'error.log').write_text(traceback.format_exc(), encoding='utf-8')
        if captures:
            write_json(output / 'raw-alignment.json', {'captures': captures})
        raise
    finally:
        manifest['elapsed_seconds'] = time.perf_counter() - started
        manifest['process_memory'] = process_memory()
        write_json(output / 'manifest.json', manifest)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Prepared local model directory; never a Hub ID')
    parser.add_argument('--audio', required=True, help='16 kHz mono Float32 or int16 WAV')
    parser.add_argument('--text', required=True, help='UTF-8 text file; copied unchanged')
    parser.add_argument('--source-offset', type=float, default=0., help='Clip start in original audio, seconds')
    parser.add_argument('--output', required=True, help='New output directory; existing paths are rejected')
    args = parser.parse_args()
    try:
        print(run(args))
    except (Exception, KeyboardInterrupt) as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
