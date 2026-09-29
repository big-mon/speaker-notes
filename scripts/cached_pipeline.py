#!/usr/bin/env python3
"""Complete-result cache for the Qwen/Apple CLI.

Only complete original results are reused. Attempts keep stable absolute paths;
neither partial stages nor edited transcripts are resumed or overwritten.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import uuid

from scene_pipeline import fingerprint, verify_manifest

ROOT = Path(__file__).resolve().parents[1]
EXPORT_FILES = ('transcript.json', 'transcript.txt', 'transcript.md', 'review.html',
                'source-material.json', 'segments.jsonl', 'asr-differences.jsonl', 'README.md')
STAGES = {
    'provenance_and_preflight': 'モデルと入力の確認',
    'normalization': '音声の形式を整えています',
    'clip': '音声全体を準備しています',
    'vad_pause_detection': '発話の切れ目を探しています',
    'qwen_partition': '発話の切れ目で音声を分割しています',
    'diarization': '話者を識別しています',
    'apple_asr': 'Appleで比較用の文字起こしをしています',
    'qwen_asr': 'Qwenで本文を文字起こししています',
    'qwen_output_validation': '文字起こしの出力を確認しています',
    'forced_alignment': '本文と音声の時刻を対応付けています',
    'assembly': '話者と時刻を本文に対応付けています',
    'export': '結果を保存しています',
}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        write_json(temporary, value)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def make_cache_key(source, settings):
    identity = fingerprint(source)
    document = {'input_content': {name: identity[name] for name in ('sha256', 'bytes')},
                # Result playback uses input.path; a copied/moved source needs its own result.
                'input_path': identity['path'],
                'settings': settings}
    canonical = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest(), identity


def prepared_settings(engine):
    if engine not in ('fluid',):
        raise ValueError('話者識別方式が不明です')
    binaries = ['.build/release/normalize', '.build/release/apple-transcribe',
                '.build/release/silero-vad-frames', '.venv-asr/bin/python',
                '.build/release/fluid-diarize']
    scripts = ['cached_pipeline.py', 'scene_pipeline.py', 'scene_transcript.py',
               'text_anchor_audit.py', 'turn_candidates.py', 'merge.py', 'qwen_asr.py',
               'lexical_speaker_buckets.py',
               'asr_chunking.py', 'vad_chunking.py', 'align_qwen.py',
               'compose_alignment.py', 'aligned_speaker_turns.py', 'review_aligned_scene.py']
    sources = ['Sources/Normalize/main.swift', 'Sources/AppleTranscribe/main.swift',
               'Sources/SileroVADFrames/main.swift',
               'Sources/FluidDiarize/main.swift']
    manifests = {
        'qwen17': (ROOT/'config/models/qwen17-manifest.json', ROOT/'models/qwen17'),
        'alignment': (ROOT/'config/models/qwen-aligner-manifest.json', ROOT/'models/qwen-aligner'),
        'vad': (ROOT/'config/models/silero-vad32-manifest.json', ROOT/'models/silero-vad32'),
        'diarization': (ROOT/f'config/models/{engine}-model-manifest.json',
                        ROOT/'models/fluid/speaker-diarization-coreml'),
    }
    for relative in binaries:
        path = ROOT/relative
        if not path.is_file() or not os.access(path, os.X_OK):
            raise FileNotFoundError(f'準備済みの実行ファイルがありません: {path}')
    # A valid old result must not conceal damaged or replaced prepared models.
    # Match the pipeline's preflight even when no new inference is needed.
    models = {name: verify_manifest(path, base) for name, (path, base) in manifests.items()}
    sites = sorted((ROOT/'.venv-asr/lib').glob('python*/site-packages'))
    if len(sites) != 1:
        raise RuntimeError('Qwen用Python環境を確認できません')
    versions = sorted((dist.metadata['Name'], dist.version) for dist in metadata.distributions(path=[str(sites[0])]))
    runtime_files = [sites[0]/'mlx_audio/stt/utils.py', sites[0]/'mlx_audio/utils.py']
    for family in ('qwen3_asr', 'qwen3_forced_aligner'):
        runtime_files += sorted((sites[0]/'mlx_audio/stt/models'/family).glob('*.py'))
    return {
        'schema_version': 2, 'pipeline': 'scene-qwen-main-apple-comparison',
        'arguments': {'full': True, 'transcript_mode': 'qwen-aligned',
                      'chunk_policy': 'vad', 'diarizer': engine, 'context_seconds': 15},
        'environment': {'macOS': platform.mac_ver()[0], 'machine': platform.machine(),
                        'wrapper_python': sys.version, 'asr_packages': versions},
        'binaries': {p: fingerprint(ROOT/p) for p in binaries},
        'wrapper_interpreter': fingerprint(sys.executable),
        'code': {p: fingerprint(ROOT/p) for p in ['scripts/'+p for p in scripts] + sources + ['config/runtime.json']},
        'runtime_code': {str(p.relative_to(ROOT)): fingerprint(p) for p in runtime_files},
        'model_manifests': {name: fingerprint(path) for name, (path, _) in manifests.items()},
        'model_assets': {name: model['verified_assets'] for name, model in models.items()},
        'model_preflight': 'actual bytes and SHA256 match every manifest before cache reuse or inference',
        'edits_policy': 'return original result only; never read, overwrite or delete edited artifacts',
    }


@contextmanager
def job_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'job.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('同じ音声・設定の処理がすでに実行中です') from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def valid_cached_result(directory, key):
    """Reject stale, altered, partial and path-escaping completion pointers."""
    directory = Path(directory).resolve()
    try:
        pointer = json.loads((directory/'completed.json').read_text())
        attempt_name = pointer['attempt']
        if (pointer['cache_key'] != key or not isinstance(attempt_name, str) or
                not attempt_name or Path(attempt_name).name != attempt_name):
            return None
        attempt = (directory/'attempts'/attempt_name).resolve()
        if not attempt.is_relative_to(directory/'attempts'):
            return None
        result = attempt/'scene/result/transcript.json'
        processing = attempt/'scene/processing.json'
        wrapper = attempt/'wrapper.json'
        for name, path in [('result', result), ('processing', processing), ('wrapper', wrapper)]:
            if not path.resolve().is_relative_to(attempt):
                return None
            actual = fingerprint(path)
            if any(actual[field] != pointer[name][field] for field in ('sha256', 'bytes')):
                return None
        if set(pointer['exports']) != set(EXPORT_FILES):
            return None
        for name in EXPORT_FILES:
            path = result.parent/name
            if not path.resolve().is_relative_to(attempt):
                return None
            actual = fingerprint(path)
            if any(actual[field] != pointer['exports'][name][field] for field in ('sha256', 'bytes')):
                return None
        if json.loads(processing.read_text()).get('status') != 'complete':
            return None
        if json.loads(wrapper.read_text()).get('status') != 'complete':
            return None
        document = json.loads(result.read_text())
        if document.get('processing', {}).get('status') != 'complete':
            return None
        return result
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def descendant_groups(pid):
    """Scene stages start their own sessions, so include their groups too."""
    groups = {pid}
    try:
        listing = subprocess.run(['/bin/ps', '-axo', 'pid=,ppid=,pgid='], capture_output=True,
                                 text=True, timeout=2, check=True)
        rows = [tuple(map(int, line.split())) for line in listing.stdout.splitlines() if line.strip()]
        descendants = {pid}
        changed = True
        while changed:
            added = {child for child, parent, _ in rows if parent in descendants} - descendants
            descendants.update(added)
            changed = bool(added)
        groups.update(group for child, _, group in rows if child in descendants)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass  # The scene process also propagates SIGTERM to its current stage.
    return groups - {os.getpgrp()}


def stop_process(process):
    groups = descendant_groups(process.pid)
    for group in groups:
        try:
            os.killpg(group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=6)
    except subprocess.TimeoutExpired:
        pass
    # Kill any remaining stage group even if the scene parent already exited.
    for group in groups:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=2)


def stream_scene(command, log):
    """Forward actual stage changes while keeping every child output in a log."""
    with Path(log).open('x', encoding='utf-8') as output:
        process = subprocess.Popen(list(map(str, command)), cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, start_new_session=True)
        try:
            for line in process.stdout:
                output.write(line)
                output.flush()
                name = line.strip()
                if name in STAGES:
                    print('工程: ' + STAGES[name], flush=True)
                elif name.startswith('alignment_chunk '):
                    print('工程: 本文と音声の時刻を対応付けています ' + name.split(' ', 1)[1], flush=True)
            if process.wait():
                raise RuntimeError(f'音声処理に失敗しました。詳細: {log}')
        except BaseException:
            stop_process(process)
            raise
        finally:
            if process.poll() is None:
                stop_process(process)
            process.stdout.close()


def run(source, engine='fluid', cache=None):
    source = Path(source).resolve()
    if not source.is_file():
        raise FileNotFoundError(f'音声ファイルがありません: {source}')
    print('工程: 入力と処理設定を確認しています', flush=True)
    settings = prepared_settings(engine)
    key, source_identity = make_cache_key(source, settings)
    directory = (Path(cache) if cache is not None else ROOT/'runs/cli-cache').resolve()/key
    with job_lock(directory):
        cached = valid_cached_result(directory, key)
        if cached is not None:
            print('再利用: 完了済みの文字起こし', flush=True)
            return cached
        attempt = directory/'attempts'/uuid.uuid4().hex
        attempt.mkdir(parents=True, exist_ok=False)
        scene = attempt/'scene'  # The scene CLI creates this; never rename it.
        record = {'status': 'running', 'cache_key': key, 'input': source_identity,
                  'settings': settings, 'started_at': datetime.now(timezone.utc).isoformat(),
                  'attempt_path': str(attempt), 'stage_resume': False}
        write_json(attempt/'wrapper.json', record)
        command = [sys.executable, ROOT/'scripts/scene_pipeline.py', source, scene,
                   '--full', '--transcript-mode', 'qwen-aligned', '--chunk-policy', 'vad',
                   '--diarizer', engine]
        write_json(attempt/'command.json', list(map(str, command)))
        started = time.perf_counter()
        try:
            stream_scene(command, attempt/'scene.log')
            processing = scene/'processing.json'
            result = scene/'result/transcript.json'
            if (json.loads(processing.read_text()).get('status') != 'complete' or
                    json.loads(result.read_text()).get('processing', {}).get('status') != 'complete'):
                raise RuntimeError('処理が完了していないため、結果を再利用対象にしません')
            record.update(status='complete', wall_seconds=time.perf_counter()-started)
            write_json(attempt/'wrapper.json', record)
            pointer = {'cache_key': key, 'attempt': attempt.name,
                       'result': fingerprint(result), 'processing': fingerprint(processing),
                       'wrapper': fingerprint(attempt/'wrapper.json'),
                       'exports': {name: fingerprint(result.parent/name) for name in EXPORT_FILES}}
            atomic_json(directory/'completed.json', pointer)
            if valid_cached_result(directory, key) != result:
                raise RuntimeError('完成結果の保存検証に失敗しました')
            return result
        except BaseException as error:
            record.update(status='cancelled' if isinstance(error, KeyboardInterrupt) else 'failed',
                          wall_seconds=time.perf_counter()-started, error=f'{type(error).__name__}: {error}')
            write_json(attempt/'wrapper.json', record)
            raise


def cancel(signum, frame):
    raise KeyboardInterrupt(f'Received signal {signum}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--engine', choices=['fluid'], default='fluid')
    parser.add_argument('--cache', type=Path, default=ROOT/'runs/cli-cache')
    args = parser.parse_args()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, cancel)
    try:
        result = run(args.input, args.engine, args.cache)
        print('RESULT ' + str(result), flush=True)
        return 0
    except KeyboardInterrupt:
        print('中止しました。途中の結果とログは保持しています。', file=sys.stderr)
        return 130
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
