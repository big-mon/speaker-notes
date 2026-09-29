"""Completed CLI cache checks without loading speech or VAD models."""
import contextlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import cached_pipeline as app


def complete_scene(command, log):
    scene = Path(command[3])
    (scene/'result').mkdir(parents=True)
    app.write_json(scene/'processing.json', {'status': 'complete'})
    app.write_json(scene/'result/transcript.json', {
        'artifact_id': 'unit-test-fixture', 'processing': {'status': 'complete'},
        'input': app.fingerprint(command[2]),
        'segments': [{'text': 'これは未確認のテスト本文です。', 'speaker': None}],
    })
    for name in app.EXPORT_FILES:
        if name != 'transcript.json':
            (scene/'result'/name).write_text('test export: ' + name, encoding='utf-8')
    Path(log).write_text('test fixture; no model invocation\n')


class CachedPipelineTests(unittest.TestCase):
    def test_prepared_models_and_all_runtime_code_participate_in_identity(self):
        """Fixture assets stand in for weights; no real model or MLX is loaded."""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for relative in ('.build/release/normalize', '.build/release/apple-transcribe',
                             '.build/release/silero-vad-frames', '.build/release/fluid-diarize',
                             '.venv-asr/bin/python'):
                path = root/relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'fixture executable')
                path.chmod(0o700)
            # Materialize only source paths, using lightweight content so the
            # test checks dependency coverage without depending on source text.
            for path in app.ROOT.joinpath('scripts').glob('*.py'):
                target = root/path.relative_to(app.ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text('fixture code')
            for path in app.ROOT.joinpath('Sources').glob('*/main.swift'):
                target = root/path.relative_to(app.ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text('fixture source')
            runtime = root/'config/runtime.json'
            runtime.parent.mkdir(parents=True, exist_ok=True)
            runtime.write_text('{}')
            sites = root/'.venv-asr/lib/python-test/site-packages'
            sites.mkdir(parents=True)
            for relative in ('mlx_audio/stt/utils.py', 'mlx_audio/utils.py',
                             'mlx_audio/stt/models/qwen3_asr/model.py',
                             'mlx_audio/stt/models/qwen3_forced_aligner/model.py'):
                path = sites/relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('fixture runtime')
            for name, directory in [('qwen17', 'qwen17'), ('qwen-aligner', 'qwen-aligner'),
                                    ('silero-vad32', 'silero-vad32'),
                                    ('fluid-model', 'fluid/speaker-diarization-coreml')]:
                asset = root/'models'/directory/'weights.bin'
                asset.parent.mkdir(parents=True, exist_ok=True)
                asset.write_bytes(b'original fixture weights')
                manifest = root/'config/models'/f'{name}-manifest.json'
                manifest.parent.mkdir(parents=True, exist_ok=True)
                app.write_json(manifest, {'files': [{**app.fingerprint(asset), 'path': 'weights.bin'}]})
            source = root/'audio.wav'
            source.write_bytes(b'fixture audio')
            with mock.patch.object(app, 'ROOT', root), \
                    mock.patch.object(app.metadata, 'distributions', return_value=[]):
                initial = app.prepared_settings('fluid')
                key = app.make_cache_key(source, initial)[0]
                self.assertIn('scripts/lexical_speaker_buckets.py', initial['code'])
                self.assertIn('config/runtime.json', initial['code'])
                runtime.write_text('{"changed": true}')
                self.assertNotEqual(key, app.make_cache_key(source, app.prepared_settings('fluid'))[0])
                (root/'models/qwen17/weights.bin').write_bytes(b'replaced fixture weights')
                with self.assertRaisesRegex(ValueError, 'does not match recorded manifest'):
                    app.prepared_settings('fluid')

    def test_cache_identity_uses_resolved_path_content_and_complete_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            first, same = Path(folder)/'first.wav', Path(folder)/'renamed.wav'
            first.write_bytes(b'pcm-content')
            same.write_bytes(first.read_bytes())
            settings = {'mode': 'qwen-aligned', 'code': {'scene': 'v1'}, 'model': {'qwen': 'hash-a'}}
            original = app.make_cache_key(first, settings)[0]
            self.assertNotEqual(original, app.make_cache_key(same, settings)[0])
            alias = Path(folder)/'alias.wav'
            alias.symlink_to(first)
            self.assertEqual(original, app.make_cache_key(alias, settings)[0])
            for changed in ({**settings, 'mode': 'apple'},
                            {**settings, 'code': {'scene': 'v2'}},
                            {**settings, 'model': {'qwen': 'hash-b'}}):
                self.assertNotEqual(original, app.make_cache_key(first, changed)[0])
            first.write_bytes(b'other-audio')
            self.assertNotEqual(original, app.make_cache_key(first, settings)[0])

    def test_moved_audio_does_not_reuse_result_with_obsolete_playback_path(self):
        with tempfile.TemporaryDirectory() as folder:
            source, moved, cache = Path(folder)/'input.wav', Path(folder)/'moved.wav', Path(folder)/'cache'
            source.write_bytes(b'audio')
            with mock.patch.object(app, 'prepared_settings', return_value={'fixed': 'settings'}), \
                    mock.patch.object(app, 'stream_scene', side_effect=complete_scene) as execute, contextlib.redirect_stdout(io.StringIO()):
                original = app.run(source, cache=cache)
                source.rename(moved)
                replacement = app.run(moved, cache=cache)
                self.assertNotEqual(replacement, original)
                self.assertEqual(execute.call_count, 2)
                self.assertEqual(json.loads(replacement.read_text())['input']['path'], str(moved.resolve()))
                self.assertEqual(json.loads(original.read_text())['input']['path'], str(source.resolve()))
                self.assertEqual(app.run(moved, cache=cache), replacement)
                self.assertEqual(execute.call_count, 2)

    def test_complete_cache_preserves_edits_and_uses_original_only(self):
        with tempfile.TemporaryDirectory() as folder:
            source, cache = Path(folder)/'input.wav', Path(folder)/'cache'
            source.write_bytes(b'audio')
            output = io.StringIO()
            with mock.patch.object(app, 'prepared_settings', return_value={'fixed': 'settings'}), \
                    mock.patch.object(app, 'stream_scene', side_effect=complete_scene) as execute, contextlib.redirect_stdout(output):
                result = app.run(source, cache=cache)
                edited = result.with_name('transcript-edited.json')
                edited.write_text('manual changes stay untouched')
                self.assertEqual(app.run(source, cache=cache), result)
                self.assertEqual(execute.call_count, 1)
                self.assertIn('再利用:', output.getvalue())
                self.assertEqual(edited.read_text(), 'manual changes stay untouched')
                self.assertEqual(result.name, 'transcript.json')
                # Changed original invalidates reuse but never gets overwritten.
                result.write_text('changed original')
                replacement = app.run(source, cache=cache)
                self.assertNotEqual(replacement, result)
                self.assertEqual(execute.call_count, 2)
                self.assertEqual(result.read_text(), 'changed original')
                self.assertEqual(edited.read_text(), 'manual changes stay untouched')

    def test_failed_attempt_retained_and_next_run_starts_fresh(self):
        with tempfile.TemporaryDirectory() as folder:
            source, cache = Path(folder)/'input.wav', Path(folder)/'cache'
            source.write_bytes(b'audio')
            def failed(command, log):
                scene = Path(command[3])
                scene.mkdir()
                app.write_json(scene/'processing.json', {'status': 'failed'})
                Path(log).write_text('failure evidence')
                raise RuntimeError('test stage failed')
            with mock.patch.object(app, 'prepared_settings', return_value={'fixed': 'settings'}), contextlib.redirect_stdout(io.StringIO()):
                with mock.patch.object(app, 'stream_scene', side_effect=failed), self.assertRaises(RuntimeError):
                    app.run(source, cache=cache)
                key_directory = next(cache.iterdir())
                failed_attempt = next((key_directory/'attempts').iterdir())
                self.assertEqual(json.loads((failed_attempt/'wrapper.json').read_text())['status'], 'failed')
                self.assertFalse((key_directory/'completed.json').exists())
                with mock.patch.object(app, 'stream_scene', side_effect=complete_scene):
                    result = app.run(source, cache=cache)
                self.assertNotEqual(result.parents[2], failed_attempt)
                self.assertEqual((failed_attempt/'scene.log').read_text(), 'failure evidence')
                self.assertEqual(len(list((key_directory/'attempts').iterdir())), 2)

    def test_every_export_must_remain_present_and_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            source, cache = Path(folder)/'input.wav', Path(folder)/'cache'
            source.write_bytes(b'audio')
            with mock.patch.object(app, 'prepared_settings', return_value={'fixed': 'settings'}), \
                    mock.patch.object(app, 'stream_scene', side_effect=complete_scene), contextlib.redirect_stdout(io.StringIO()):
                result = app.run(source, cache=cache)
            directory = next(cache.iterdir())
            pointer = json.loads((directory/'completed.json').read_text())
            self.assertEqual(set(pointer['exports']), {
                'transcript.json', 'transcript.txt', 'transcript.md', 'review.html',
                'source-material.json', 'segments.jsonl', 'asr-differences.jsonl', 'README.md'})
            for name in app.EXPORT_FILES:
                with self.subTest(name=name):
                    export = result.parent/name
                    original = export.read_bytes()
                    export.write_bytes(b'changed export')
                    self.assertIsNone(app.valid_cached_result(directory, pointer['cache_key']))
                    export.unlink()
                    self.assertIsNone(app.valid_cached_result(directory, pointer['cache_key']))
                    export.write_bytes(original)
                    self.assertEqual(app.valid_cached_result(directory, pointer['cache_key']), result)
            del pointer['exports']
            app.write_json(directory/'completed.json', pointer)
            self.assertIsNone(app.valid_cached_result(directory, pointer['cache_key']))

    def test_missing_material_export_never_creates_a_completion_pointer(self):
        with tempfile.TemporaryDirectory() as folder:
            source, cache = Path(folder)/'input.wav', Path(folder)/'cache'
            source.write_bytes(b'audio')
            def incomplete_exports(command, log):
                complete_scene(command, log)
                (Path(command[3])/'result/segments.jsonl').unlink()
            with mock.patch.object(app, 'prepared_settings', return_value={'fixed': 'settings'}), \
                    mock.patch.object(app, 'stream_scene', side_effect=incomplete_exports), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(FileNotFoundError):
                    app.run(source, cache=cache)
            directory = next(cache.iterdir())
            self.assertFalse((directory/'completed.json').exists())
            wrapper = next(directory.glob('attempts/*/wrapper.json'))
            self.assertEqual(json.loads(wrapper.read_text())['status'], 'failed')

    def test_incomplete_processing_and_pointer_escape_are_never_cached(self):
        with tempfile.TemporaryDirectory() as folder:
            source, cache = Path(folder)/'input.wav', Path(folder)/'cache'
            source.write_bytes(b'audio')
            with mock.patch.object(app, 'prepared_settings', return_value={'fixed': 'settings'}), \
                    mock.patch.object(app, 'stream_scene', side_effect=complete_scene), contextlib.redirect_stdout(io.StringIO()):
                result = app.run(source, cache=cache)
            directory = next(cache.iterdir())
            pointer_path = directory/'completed.json'
            pointer = json.loads(pointer_path.read_text())
            processing = result.parents[1]/'processing.json'
            app.write_json(processing, {'status': 'cancelled'})
            pointer['processing'] = app.fingerprint(processing)
            app.write_json(pointer_path, pointer)
            self.assertIsNone(app.valid_cached_result(directory, pointer['cache_key']))
            pointer['attempt'] = '../../outside'
            app.write_json(pointer_path, pointer)
            self.assertIsNone(app.valid_cached_result(directory, pointer['cache_key']))

    def test_lock_blocks_duplicate_then_releases_without_stale_lock_problem(self):
        with tempfile.TemporaryDirectory() as folder:
            with app.job_lock(folder):
                with self.assertRaisesRegex(RuntimeError, 'すでに実行中'):
                    with app.job_lock(folder):
                        self.fail('Same-key work must not start concurrently')
            with app.job_lock(folder):
                pass

    @unittest.skipUnless(hasattr(os, 'killpg'), 'POSIX process-group cancellation test')
    def test_cancel_stops_separately_grouped_stage_and_never_reports_result(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            stage_ready, pids = folder/'stage.ready', folder/'pids.json'
            stage = ('import signal,time; from pathlib import Path; '
                     'signal.signal(signal.SIGTERM,signal.SIG_IGN); '
                     f'Path({str(stage_ready)!r}).write_text("ready"); time.sleep(60)')
            scene = (
                'import os,sys,subprocess,time,json; from pathlib import Path; '
                f'stage=subprocess.Popen([sys.executable,"-c",{stage!r}],start_new_session=True)\n'
                f'while not Path({str(stage_ready)!r}).exists(): time.sleep(.01)\n'
                f'Path({str(pids)!r}).write_text(json.dumps([os.getpid(),stage.pid])); '
                'print("normalization",flush=True); time.sleep(60)'
            )
            supervisor = (
                f'import sys,signal;sys.path.insert(0,{str(Path(__file__).parent)!r}); '
                'from cached_pipeline import stream_scene,cancel; '
                'signal.signal(signal.SIGTERM,cancel)\n'
                f'try: stream_scene([sys.executable,"-c",{scene!r}],{str(folder/"scene.log")!r}); print("RESULT forbidden")\n'
                'except KeyboardInterrupt: sys.exit(130)\n'
            )
            process = subprocess.Popen([sys.executable, '-c', supervisor], stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True)
            children = []
            try:
                deadline = time.monotonic()+5
                while not pids.exists() and time.monotonic() < deadline and process.poll() is None:
                    time.sleep(.02)
                self.assertTrue(pids.exists(), 'Fake scene/stage did not start')
                children = json.loads(pids.read_text())
                process.send_signal(signal.SIGTERM)
                stdout, stderr = process.communicate(timeout=12)
                self.assertEqual(process.returncode, 130, stderr)
                self.assertNotIn('RESULT ', stdout)
                self.assertTrue((folder/'scene.log').exists())
                for pid in children:
                    deadline = time.monotonic()+3
                    while time.monotonic() < deadline:
                        status = subprocess.run(['/bin/ps', '-p', str(pid), '-o', 'stat='], capture_output=True, text=True)
                        if not status.stdout.strip() or status.stdout.strip().startswith('Z'):
                            break
                        time.sleep(.02)
                    else:
                        self.fail(f'Cancelled stage process {pid} remained active')
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate(timeout=5)
                for pid in children:
                    try:
                        os.killpg(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass


if __name__ == '__main__':
    unittest.main()
