"""Sequential local audio -> diarization -> Apple/Qwen -> scene review material.

Defaults to the full recording. --duration selects a preview with up to 15 seconds of context.
Context is a preview heuristic, not a guarantee against cut speech or omissions.
--context-seconds 0 reproduces the former crop; --full processes the whole file.
No downloads, automatic text replacement, or claim of human-verified accuracy.
The pipeline uses Qwen text and per-chunk local forced
alignment, with Apple as a comparison source. One diarization invocation covers
the whole selected recording.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import shutil
import tempfile
import struct
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
RATE = 16000
MAX_CONTEXT_SECONDS = 15.


@contextmanager
def verified_input_snapshot(source, expected, directory):
    """Normalize a private copy, not a source path that can change mid-read."""
    with tempfile.TemporaryDirectory(prefix='.normalization-', dir=directory) as temporary:
        snapshot = Path(temporary) / ('input' + source.suffix)
        shutil.copyfile(source, snapshot)
        actual = fingerprint(snapshot)
        if any(actual[key] != expected[key] for key in ('bytes', 'sha256')):
            raise ValueError('Input changed before normalization; snapshot differs from input identity')
        snapshot.chmod(0o400)
        yield snapshot


def fingerprint(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest}


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def read_float_wav(path):
    """Read IEEE float32 mono WAV; reject unexpected or truncated formats."""
    with Path(path).open('rb') as stream:
        header = stream.read(12)
        if len(header) != 12 or header[:4] != b'RIFF' or header[8:] != b'WAVE':
            raise ValueError('Expected RIFF/WAVE input')
        riff_end = struct.unpack('<I', header[4:8])[0] + 8
        if riff_end > Path(path).stat().st_size:
            raise ValueError('Truncated RIFF input')
        fmt, pcm = None, None
        while stream.tell() + 8 <= riff_end:
            name, size = struct.unpack('<4sI', stream.read(8))
            if stream.tell() + size > riff_end:
                raise ValueError('Truncated WAV chunk')
            if name == b'fmt ':
                fmt = stream.read(size)
            elif name == b'data':
                if pcm is not None:
                    raise ValueError('Multiple WAV data chunks are not supported')
                pcm = stream.read(size)
            else:
                stream.seek(size, 1)
            if size % 2:
                stream.seek(1, 1)
        if fmt is None or len(fmt) < 16 or pcm is None:
            raise ValueError('WAV fmt/data chunk is missing')
        code, channels, rate, byte_rate, align, bits = struct.unpack('<HHIIHH', fmt[:16])
        if (code, channels, byte_rate, align, bits) != (3, 1, rate * 4, 4, 32):
            raise ValueError('Expected IEEE float32 mono PCM WAV')
        if rate <= 0 or len(pcm) % 4 or not pcm:
            raise ValueError('Invalid WAV sample count or sample rate')
        return rate, pcm


def write_float_wav(path, pcm, rate=RATE):
    if rate <= 0 or not pcm or len(pcm) % 4 or len(pcm) + 36 > 0xffffffff:
        raise ValueError('Invalid PCM data for RIFF float32 WAV')
    header = (b'RIFF' + struct.pack('<I', 36 + len(pcm)) + b'WAVEfmt ' +
              struct.pack('<IHHIIHH', 16, 3, 1, rate, rate * 4, 4, 32) +
              b'data' + struct.pack('<I', len(pcm)))
    with Path(path).open('xb') as stream:
        stream.write(header)
        stream.write(pcm)


def clip_samples(pcm, rate, start=0., duration=180., full=False, context_seconds=0.):
    """Select target plus context without changing source-clock sample positions."""
    if rate <= 0 or not pcm or len(pcm) % 4:
        raise ValueError('Invalid PCM input or sample rate')
    if not math.isfinite(start) or start < 0:
        raise ValueError('start must be finite and nonnegative')
    if not math.isfinite(context_seconds) or not 0 <= context_seconds <= MAX_CONTEXT_SECONDS:
        raise ValueError('context_seconds must be finite and between 0 and 15')
    if full and start != 0:
        raise ValueError('--full requires --start 0')
    if not full and (not math.isfinite(duration) or duration <= 0):
        raise ValueError('duration must be finite and positive')
    count = len(pcm) // 4
    target_start = round(start * rate)
    target_end = count if full else min(count, target_start + round(duration * rate))
    if not 0 <= target_start < target_end <= count:
        raise ValueError('Requested clip contains no input samples')
    context = 0 if full else round(context_seconds * rate)
    a, b = max(0, target_start-context), min(count, target_end+context)
    requested = {'start': start, 'end': count/rate if full else start+duration,
                 'duration': count/rate if full else duration}
    actual = {'start': a/rate, 'end': b/rate, 'duration': (b-a)/rate,
              'start_sample': a, 'end_sample': b}
    conditions = {
        'requested_context_seconds': context_seconds,
        'maximum_context_seconds': MAX_CONTEXT_SECONDS,
        'mode': 'not_applied_to_full_recording' if full else 'preview_context_heuristic',
        'applied': a != target_start or b != target_end,
        'before_seconds': (target_start-a)/rate, 'after_seconds': (b-target_end)/rate,
        'sample_aligned_target': {'start': target_start/rate, 'end': target_end/rate,
                                  'start_sample': target_start, 'end_sample': target_end},
        'clamped_to_source_start': not full and target_start-context < 0,
        'clamped_to_source_end': not full and target_end+context > count,
        'requested_target_exceeds_source_end': requested['end'] > count/rate,
        'no_samples_removed_inside_selection': True,
        'endpoints_may_cut_continuing_speech': True,
        'context_guarantees_complete_recognition': False,
        'speaker_identity_scope': 'one actual-window diarization invocation; no cross-clip matching'}
    return pcm[a * 4:b * 4], {'start_sample': a, 'end_sample': b,
                              'source_offset': a/rate, 'duration': (b-a)/rate,
                              'requested_window': requested, 'actual_window': actual,
                              'context_conditions': conditions}


def verify_manifest(path, base):
    """Verify prepared assets; never repair or download a missing model."""
    path, base = Path(path), Path(base).resolve()
    manifest = json.loads(path.read_text())
    items = manifest['files']
    checked = []
    for item in items:
        asset = (base / item['path']).resolve()
        if not asset.is_relative_to(base):
            raise ValueError('Model manifest path escapes its model directory')
        actual = fingerprint(asset)
        if actual['bytes'] != item['bytes'] or actual['sha256'] != item['sha256']:
            raise ValueError(f'Prepared model does not match recorded manifest: {asset}')
        checked.append(actual)
    return {'manifest': fingerprint(path), 'recorded': manifest,
            'verified_assets': checked, 'verification': 'actual bytes and SHA256 matched'}


def run_command(command, log):
    """Keep one child process at a time and terminate its group on cancellation."""
    command = [str(x) for x in command]
    with Path(log).open('x') as stream:
        stream.write(json.dumps({'command': command}, ensure_ascii=False) + '\n')
        stream.flush()
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait()
            if code:
                raise RuntimeError(f'Command exited {code}; see {log}')
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()


def run(args):
    from scene_transcript import save
    started = time.perf_counter()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/'logs').mkdir()
    processing = {'started_at': datetime.now(timezone.utc).isoformat(), 'status': 'running',
                  'execution': 'sequential', 'stages': [], 'peak_memory': None,
                  'peak_memory_note': 'Pipeline peak RSS was not measured',
                  'measurement_scope': 'wall clock including provenance checks and initial artifact export; excludes final timing metadata write'}
    current_stage = 'initialization'

    @contextmanager
    def stage(name):
        nonlocal current_stage
        current_stage = name
        row = {'name': name, 'status': 'running'}
        processing['stages'].append(row)
        begin = time.perf_counter()
        print(name, flush=True)
        try:
            yield row
            row['status'] = 'complete'
        except BaseException:
            row['status'] = 'failed'
            raise
        finally:
            row['wall_seconds'] = time.perf_counter()-begin
            processing['elapsed_seconds'] = time.perf_counter()-started
            write_json(output/'processing.json', processing)

    try:
        with stage('provenance_and_preflight'):
            if not args.input.is_file():
                raise FileNotFoundError(args.input)
            binaries = {'normalize': ROOT/'.build/release/normalize',
                        'apple': ROOT/'.build/release/apple-transcribe',
                        'diarization': ROOT/'.build/release/fluid-diarize',
                        'qwen_python': ROOT/'.venv-asr/bin/python'}
            binaries['vad'] = ROOT/'.build/release/silero-vad-frames'
            for binary in binaries.values():
                if not binary.is_file() or not os.access(binary, os.X_OK):
                    raise FileNotFoundError(f'Prepared executable unavailable: {binary}')
            models = {'qwen': verify_manifest(ROOT/'config/models/qwen17-manifest.json', ROOT/'models/qwen17')}
            models['vad'] = verify_manifest(ROOT/'config/models/silero-vad32-manifest.json', ROOT/'models/silero-vad32')
            models['alignment'] = verify_manifest(ROOT/'config/models/qwen-aligner-manifest.json', ROOT/'models/qwen-aligner')
            models['diarization'] = verify_manifest(ROOT/'config/models/fluid-model-manifest.json', ROOT/'models/fluid/speaker-diarization')
            code = ['scripts/scene_pipeline.py', 'scripts/scene_transcript.py',
                    'scripts/text_anchor_audit.py', 'scripts/turn_candidates.py',
                    'scripts/qwen_asr.py', 'scripts/asr_chunking.py',
                    'Sources/Normalize/main.swift', 'Sources/AppleTranscribe/main.swift',
                    'Sources/FluidDiarize/main.swift']
            code += ['scripts/vad_chunking.py', 'Sources/SileroVADFrames/main.swift']
            code += ['scripts/align_qwen.py', 'scripts/compose_alignment.py',
                     'scripts/aligned_speaker_turns.py', 'scripts/review_aligned_scene.py']
            provenance = {'input': fingerprint(args.input),
                          'environment': {'macOS': platform.mac_ver()[0], 'machine': platform.machine(),
                                          'pipeline_python': sys.version, 'executable': sys.executable},
                          'binaries': {k: fingerprint(v) for k, v in binaries.items()},
                          'code': {p: fingerprint(ROOT/p) for p in code}, 'models': models,
                          'configured_versions': json.loads((ROOT/'config/runtime.json').read_text()),
                          'settings': {'diarizer': 'fluid', 'passes': 1, 'transcript_mode': 'qwen-aligned',
                                       'apple_locale': 'ja-JP', 'normalization': 'AVAudioConverter maximum quality; 16kHz mono float32; no trimming',
                                       'requested_start': args.start, 'requested_duration': None if args.full else args.duration,
                                       'requested_context_seconds': args.context_seconds,
                                       'full_recording_requested': args.full, 'qwen_max_input_seconds': 180,
                                       'qwen_hotwords': False,
                                       'qwen_chunk_policy': 'vad'}}
            write_json(output/'provenance.json', provenance)
        with stage('normalization') as row:
            with verified_input_snapshot(args.input, provenance['input'], output) as snapshot:
                command = [binaries['normalize'], snapshot, output/'normalized.wav']
                row['command'] = [str(x) for x in command]
                row['input_policy'] = 'private read-only copy verified against input SHA256; removed after normalization'
                run_command(command, output/'logs/normalization.log')
        with stage('clip'):
            rate, full_pcm = read_float_wav(output/'normalized.wav')
            if rate != RATE:
                raise ValueError('Normalizer did not produce 16kHz PCM')
            pcm, window = clip_samples(full_pcm, rate, args.start, args.duration, args.full,
                                       context_seconds=args.context_seconds)
            selection = {key: window[key] for key in ('requested_window', 'actual_window', 'context_conditions')}
            provenance.update(selection)
            write_json(output/'provenance.json', provenance)
            source_duration = len(full_pcm)/4/rate
            del full_pcm
            write_float_wav(output/'audio.wav', pcm)
        with stage('vad_pause_detection') as row:
            command = [binaries['vad'], output/'audio.wav',
                       ROOT/'models/silero-vad32/silero-vad-unified-v6.0.0.mlmodelc', output/'vad']
            row['command'] = [str(x) for x in command]
            run_command(command, output/'logs/vad.log')
        with stage('qwen_partition'):
            from vad_chunking import partition_at_vad_pauses
            vad_path = output/'vad/frames.json'
            vad = json.loads(vad_path.read_text())
            audio_identity = fingerprint(output/'audio.wav')
            if (Path(vad['file']).resolve() != Path(audio_identity['path']) or
                    vad['sha256'] != audio_identity['sha256']):
                raise ValueError('VAD frames do not correspond to the selected audio')
            plan = partition_at_vad_pauses(pcm, vad, rate)
            parts, chunk_policy = plan['parts'], plan['policy']
            chunk_policy['raw_vad'] = fingerprint(vad_path)
            write_json(output/'vad-chunk-plan.json', plan)
            (output/'qwen-input').mkdir()
            for part in parts:
                path = output/'qwen-input'/f'{part["index"]:03d}.wav'
                write_float_wav(path, pcm[part['start_sample']*4:part['end_sample']*4])
                part.update(file=fingerprint(path), source_start=window['source_offset']+part['start'],
                            source_end=window['source_offset']+part['end'])
            del pcm
            manifest = {'source_duration': source_duration, 'sample_rate': rate,
                        'normalized': fingerprint(output/'normalized.wav'),
                        'clip': fingerprint(output/'audio.wav'), 'window': window,
                        **selection,
                        'qwen_chunks': parts, 'qwen_chunk_policy': chunk_policy,
                        'contiguous_sample_coverage': True,
                        'low_energy_boundaries_are_verified_silence': False}
            write_json(output/'audio-manifest.json', manifest)
            (output/'qwen-input/files.txt').write_text(''.join(p['file']['path']+'\n' for p in parts))
        with stage('diarization') as row:
            command = [binaries['diarization'], output/'audio.wav', ROOT/'models/fluid', output/'diarization', '1']
            row['command'] = [str(x) for x in command]
            run_command(command, output/'logs/diarization.log')
        with stage('apple_asr') as row:
            command = [binaries['apple'], output/'audio.wav', output/'apple']
            row['command'] = [str(x) for x in command]
            run_command(command, output/'logs/apple.log')
        with stage('qwen_asr') as row:
            command = [binaries['qwen_python'], ROOT/'scripts/qwen_asr.py', output/'qwen-input/files.txt', output/'qwen']
            row['command'] = [str(x) for x in command]
            run_command(command, output/'logs/qwen.log')
        with stage('qwen_output_validation'):
            chunks = []
            for part in parts:
                path = output/'qwen'/f'{part["index"]:03d}.json'
                raw = json.loads(path.read_text())
                if Path(raw['file']).resolve() != Path(part['file']['path']):
                    raise ValueError('Qwen result does not correspond to its input chunk')
                text = raw['raw'].get('text')
                if not isinstance(text, str):
                    raise ValueError(f'Qwen output has no text string: {path}')
                chunks.append(dict(part, text=text, raw_output=fingerprint(path),
                                   token_limit_reached=bool(raw.get('token_limit_reached'))))
            comparison = {'chunks': chunks, 'time_granularity': 'input chunks; not text alignment',
                          'incomplete': any(c['token_limit_reached'] for c in chunks)}
            write_json(output/'qwen-comparison.json', comparison)
            qwen_text = '\n'.join(c['text'] for c in chunks)
            (output/'qwen-comparison.txt').write_text(qwen_text+'\n')
            if comparison['incomplete']:
                raise RuntimeError('Qwen token_limit_reached: output is incomplete; raw results retained, no completed review artifact produced')

        from compose_alignment import compose_alignments
        with stage('forced_alignment') as row:
            (output/'alignment-input').mkdir()
            (output/'alignments').mkdir()
            row.update(total_chunks=len(chunks), completed_chunks=0, commands=[])
            aligned_parts = []
            for chunk in chunks:
                index = chunk['index']
                text_path = output/'alignment-input'/f'{index:03d}.txt'
                text_path.write_text(chunk['text'], encoding='utf-8')
                part_output = output/'alignments'/f'{index:03d}'
                command = [binaries['qwen_python'], ROOT/'scripts/align_qwen.py',
                           '--model', ROOT/'models/qwen-aligner', '--audio', chunk['file']['path'],
                           '--text', text_path, '--source-offset', str(chunk['source_start']),
                           '--output', part_output]
                row['commands'].append([str(x) for x in command])
                run_command(command, output/'logs'/f'alignment-{index:03d}.log')
                aligned = json.loads((part_output/'alignment.json').read_text())
                align_manifest = json.loads((part_output/'manifest.json').read_text())
                if (align_manifest['status'] != 'completed' or
                        aligned['mapping']['original_text'] != chunk['text'] or
                        align_manifest['inputs']['audio']['sha256'] != chunk['file']['sha256']):
                    raise ValueError('Alignment does not match its ASR text and exact input audio')
                aligned_parts.append({'start': chunk['start'], 'end': chunk['end'], 'alignment': aligned,
                                      'source': fingerprint(part_output/'alignment.json')})
                row['completed_chunks'] += 1
                write_json(output/'processing.json', processing)
                print(f'alignment_chunk {row["completed_chunks"]}/{len(chunks)}', flush=True)
            alignment = compose_alignments(aligned_parts, window['source_offset'], window['duration'])
            if alignment['mapping']['original_text'] != qwen_text:
                raise ValueError('Composed alignment changed the Qwen chunk text')
            write_json(output/'alignment.json', alignment)

        with stage('assembly'):
            apple_path, diar_path = output/'apple/transcript-segments.json', output/'diarization/pass-1.json'
            diar = json.loads(diar_path.read_text())
            apple = json.loads(apple_path.read_text())
            sources = {name: fingerprint(path) for name, path in
                       [('apple', apple_path), ('diarization', diar_path),
                        ('qwen', output/'qwen-comparison.json'), ('audio_manifest', output/'audio-manifest.json')]}
            review_audio = dict(manifest['clip'], src='../audio.wav', source_offset=window['source_offset'])
            from review_aligned_scene import make_document
            sources['alignment'] = fingerprint(output/'alignment.json')
            document = make_document(alignment, diar, apple, provenance['input'], sources, review_audio)
            document.update(input=provenance['input'], processing=processing,
                            provenance=provenance, qwen_chunks=comparison,
                            **selection,
                            diarization_metadata={k: v for k, v in diar.items() if k != 'segments'},
                            qwen_settings=json.loads((output/'qwen/settings.json').read_text()),
                            review_audio=review_audio, raw_sources=sources)
        with stage('export'):
            save(document, output/'result')
        processing.update(status='complete', total_seconds=time.perf_counter()-started,
                          source_audio_seconds=source_duration, processed_audio_seconds=window['duration'],
                          requested_audio_seconds=window['requested_window']['duration'])
        processing['within_processed_audio_duration'] = processing['total_seconds'] <= window['duration']
        write_json(output/'processing.json', processing)
        # Only update timing metadata in files created by this invocation.
        write_json(output/'result/transcript.json', document)
        print(json.dumps({'status': 'processing_complete_quality_unverified',
                          'outputs': {kind: str((output/'result'/f'transcript.{kind}').resolve())
                                      for kind in ('txt', 'md', 'json')} |
                                     {name: str((output/'result'/name).resolve()) for name in
                                      ('source-material.json', 'segments.jsonl', 'asr-differences.jsonl', 'README.md')},
                          'total_seconds': processing['total_seconds']}, ensure_ascii=False), flush=True)
    except BaseException as error:
        processing.update(status='cancelled' if isinstance(error, KeyboardInterrupt) else 'failed',
                          total_seconds=time.perf_counter()-started)
        write_json(output/'processing.json', processing)
        write_json(output/'failure.json', {'stage': current_stage, 'type': type(error).__name__,
                                          'error': str(error), 'traceback': traceback.format_exc(),
                                          'raw_results_retained': True})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--start', type=float, default=0, help='target start in original audio seconds (default: 0)')
    extent = parser.add_mutually_exclusive_group()
    extent.add_argument('--duration', type=float, help='preview duration before adding context; omitted: full recording')
    extent.add_argument('--full', action='store_true', help='process the whole file; context is not applied')
    parser.add_argument('--context-seconds', type=float, default=15,
                        help='preview context on each side, 0–15 seconds (default: 15); 0 reproduces old cropping; heuristic only')
    args = parser.parse_args(argv)
    if args.duration is None:
        args.full = True
        args.duration = 180  # unused for full recordings
    if not math.isfinite(args.start) or args.start < 0 or not math.isfinite(args.duration) or args.duration <= 0:
        parser.error('start must be nonnegative; duration must be positive; both finite')
    if args.full and args.start != 0:
        parser.error('--full requires --start 0')
    if not math.isfinite(args.context_seconds) or not 0 <= args.context_seconds <= MAX_CONTEXT_SECONDS:
        parser.error('--context-seconds must be finite and between 0 and 15')
    def cancel(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, cancel)
    try:
        run(args)
    except KeyboardInterrupt:
        print('Cancelled; child processes stopped and logs retained', file=sys.stderr)
        return 130
    except Exception as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
