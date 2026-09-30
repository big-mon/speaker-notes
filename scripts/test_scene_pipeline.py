"""Offline structural tests; no model initialization or downloads."""
import hashlib
import json
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from scene_pipeline import (clip_samples, fingerprint,
                            read_float_wav, verify_manifest, write_float_wav)


def pcm(values):
    return struct.pack('<' + 'f' * len(values), *values)


class ScenePipelineTests(unittest.TestCase):
    def test_snapshot_binds_normalizer_to_verified_bytes_despite_restored_source(self):
        from scene_pipeline import verified_input_snapshot
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'input.mp3'
            source.write_bytes(b'keyed audio')
            identity = fingerprint(source)
            with verified_input_snapshot(source, identity, directory) as snapshot:
                source.write_bytes(b'transient different audio')
                consumed = snapshot.read_bytes()
                source.write_bytes(b'keyed audio')
                self.assertEqual(consumed, b'keyed audio')
                self.assertEqual(fingerprint(source), identity)
                self.assertEqual(snapshot.stat().st_mode & 0o222, 0)
            self.assertFalse(snapshot.exists())
            source.write_bytes(b'different at copy time')
            with self.assertRaisesRegex(ValueError, 'snapshot differs'):
                with verified_input_snapshot(source, identity, directory):
                    self.fail('Must not start normalizer')
            self.assertEqual(list(Path(directory).glob('.normalization-*')), [])

    def test_loader_inventory_rejects_extra_files_symlinks_and_bad_fluid_marker(self):
        from setup_cli import fetch_model
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base = root/'model'
            base.mkdir()
            asset = base/'weights.bin'
            asset.write_bytes(b'weights')
            manifest = {'repo': 'example/model', 'directory': 'model', 'revision': 'a'*40,
                        'files': [{**fingerprint(asset), 'path': 'weights.bin'}]}
            path = root/'manifest.json'
            path.write_text(json.dumps(manifest))
            verify_manifest(path, base)
            for name in ('tokenizer.json', 'special_tokens_map.json', 'nested/extra.json'):
                extra = base/name
                extra.parent.mkdir(exist_ok=True)
                extra.write_text('{}')
                with self.assertRaisesRegex(ValueError, 'Unmanifested'):
                    verify_manifest(path, base)
                with self.assertRaisesRegex(ValueError, 'Unmanifested'):
                    fetch_model(manifest, root)
                extra.unlink()
            alias = base/'linked'
            alias.symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'symlink'):
                verify_manifest(path, base)
            alias.unlink()
            manifest['repo'] = 'FluidInference/speaker-diarization-coreml'
            path.write_text(json.dumps(manifest))
            marker = base/'.fluidaudio-revision'
            marker.write_text('a'*40)
            verify_manifest(path, base)
            marker.write_text('b'*40)
            with self.assertRaisesRegex(ValueError, 'revision differs'):
                verify_manifest(path, base)

    def test_public_entry_defaults_to_full_and_preview_is_explicit(self):
        import scene_pipeline
        with mock.patch.object(scene_pipeline, 'run') as run, mock.patch.object(scene_pipeline.signal, 'signal'):
            scene_pipeline.main(['input.mp3', 'output'])
            self.assertTrue(run.call_args.args[0].full)
            scene_pipeline.main(['input.mp3', 'output', '--start', '42', '--duration', '60'])
            args = run.call_args.args[0]
            self.assertFalse(args.full)
            self.assertEqual((args.start, args.duration, args.context_seconds), (42, 60, 15))

    def test_float_roundtrip_with_metadata_chunk_and_overwrite_refusal(self):
        values = pcm([.1, -.7, 0., .4])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'sample.wav'
            write_float_wav(path, values, 10)
            self.assertEqual(read_float_wav(path), (10, values))
            with self.assertRaises(FileExistsError):
                write_float_wav(path, values, 10)
            data = path.read_bytes()
            junk = b'JUNK' + struct.pack('<I', 3) + b'abc\0'
            rewritten = data[:4] + struct.pack('<I', len(data)-8+len(junk)) + data[8:12] + junk + data[12:]
            path.write_bytes(rewritten)
            self.assertEqual(read_float_wav(path), (10, values))

    def test_truncated_or_wrong_format_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'sample.wav'
            write_float_wav(path, pcm([0., 1.]), 10)
            data = path.read_bytes()
            path.write_bytes(data[:-1])
            with self.assertRaisesRegex(ValueError, 'Truncated'):
                read_float_wav(path)
            wrong = bytearray(data)
            wrong[20:22] = struct.pack('<H', 1)
            path.write_bytes(wrong)
            with self.assertRaisesRegex(ValueError, 'float32'):
                read_float_wav(path)

    def test_clip_keeps_exact_samples_and_source_clock(self):
        source = pcm([i/100 for i in range(100)])
        clip, window = clip_samples(source, 10, start=1.2, duration=2.3)
        self.assertEqual(clip, source[12*4:35*4])
        self.assertEqual(window['source_offset'], 1.2)
        self.assertEqual(window['duration'], 2.3)
        with self.assertRaises(ValueError):
            clip_samples(source, 10, start=10)
        with self.assertRaises(ValueError):
            clip_samples(source, 10, start=1, full=True)
        self.assertEqual(clip_samples(source, 10, full=True)[0], source)

    def test_preview_context_at_beginning_keeps_requested_and_actual_distinct(self):
        source = pcm([i/1000 for i in range(1000)])
        clip, window = clip_samples(source, 10, start=0, duration=20, context_seconds=15)
        self.assertEqual(clip, source[:350*4])
        self.assertEqual(window['requested_window'], {'start': 0, 'end': 20, 'duration': 20})
        self.assertEqual((window['actual_window']['start'], window['actual_window']['end']), (0, 35))
        self.assertEqual(window['source_offset'], 0)
        self.assertEqual(window['duration'], 35)
        self.assertEqual(window['context_conditions']['before_seconds'], 0)
        self.assertEqual(window['context_conditions']['after_seconds'], 15)
        self.assertTrue(window['context_conditions']['clamped_to_source_start'])

    def test_preview_middle_context_is_one_contiguous_source_selection(self):
        source = pcm([i/1000 for i in range(1000)])
        clip, window = clip_samples(source, 10, start=40, duration=20, context_seconds=15)
        self.assertEqual(clip, source[250*4:750*4])
        self.assertEqual((window['requested_window']['start'], window['requested_window']['end']), (40, 60))
        self.assertEqual((window['actual_window']['start'], window['actual_window']['end']), (25, 75))
        self.assertEqual(window['source_offset'], 25)
        self.assertEqual(window['duration'], 50)
        self.assertEqual(window['context_conditions']['before_seconds'], 15)
        self.assertEqual(window['context_conditions']['after_seconds'], 15)
        self.assertTrue(window['context_conditions']['no_samples_removed_inside_selection'])
        self.assertTrue(window['context_conditions']['endpoints_may_cut_continuing_speech'])

    def test_preview_end_context_and_requested_overrun_are_clamped(self):
        source = pcm([i/1000 for i in range(1000)])
        clip, window = clip_samples(source, 10, start=90, duration=20, context_seconds=15)
        self.assertEqual(clip, source[750*4:])
        self.assertEqual(window['requested_window']['end'], 110)
        self.assertEqual(window['actual_window']['end'], 100)
        self.assertEqual(window['context_conditions']['before_seconds'], 15)
        self.assertEqual(window['context_conditions']['after_seconds'], 0)
        self.assertTrue(window['context_conditions']['clamped_to_source_end'])
        self.assertTrue(window['context_conditions']['requested_target_exceeds_source_end'])

    def test_full_recording_never_adds_preview_context(self):
        source = pcm([.25] * 1000)
        clip, window = clip_samples(source, 10, full=True, context_seconds=15)
        self.assertEqual(clip, source)
        self.assertEqual((window['actual_window']['start'], window['actual_window']['end']), (0, 100))
        self.assertEqual(window['requested_window']['duration'], 100)
        self.assertFalse(window['context_conditions']['applied'])
        self.assertEqual(window['context_conditions']['mode'], 'not_applied_to_full_recording')
        self.assertEqual(window['context_conditions']['before_seconds'], 0)
        self.assertEqual(window['context_conditions']['after_seconds'], 0)

    def test_zero_context_reproduces_the_previous_sample_exact_crop(self):
        source = pcm([i/1000 for i in range(1000)])
        clip, window = clip_samples(source, 10, start=40.2, duration=20.3, context_seconds=0)
        self.assertEqual(clip, source[402*4:605*4])
        self.assertEqual(clip, clip_samples(source, 10, start=40.2, duration=20.3)[0])
        self.assertEqual((window['source_offset'], window['duration']), (40.2, 20.3))
        self.assertFalse(window['context_conditions']['applied'])

    def test_invalid_context_is_rejected_before_any_model_work(self):
        source = pcm([.25] * 100)
        for value in (-1, 15.1, float('nan'), float('inf'), -float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                clip_samples(source, 10, context_seconds=value)
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'must-not-be-created'
            for value in ('-1', '15.1', 'nan', 'inf'):
                process = subprocess.run([sys.executable, str(Path(__file__).with_name('scene_pipeline.py')),
                                          'nonexistent-input.wav', str(output), '--context-seconds', value],
                                         capture_output=True, text=True)
                self.assertEqual(process.returncode, 2)
                self.assertIn('--context-seconds must', process.stderr)
                self.assertFalse(output.exists())


    def test_prepared_model_provenance_is_verified_not_downloaded(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base = root/'model'
            base.mkdir()
            asset = base/'weights.bin'
            asset.write_bytes(b'tiny fixture')
            entry = {'path': asset.name, 'bytes': 12,
                     'sha256': hashlib.sha256(asset.read_bytes()).hexdigest()}
            manifest = root/'manifest.json'
            manifest.write_text(json.dumps({'files': [entry]}))
            result = verify_manifest(manifest, base)
            self.assertEqual(result['verified_assets'][0], fingerprint(asset))
            asset.write_bytes(b'changed data')
            with self.assertRaisesRegex(ValueError, 'does not match'):
                verify_manifest(manifest, base)
            manifest.write_text(json.dumps({'files': [dict(entry, path='../outside')]}))
            with self.assertRaisesRegex(ValueError, 'relative path|escapes'):
                verify_manifest(manifest, base)

    @unittest.skipUnless(hasattr(os, 'killpg'), 'POSIX cancellation test')
    def test_cancel_stops_real_child_and_keeps_log(self):
        with tempfile.TemporaryDirectory() as folder:
            pid_path, log = Path(folder)/'child.pid', Path(folder)/'process.log'
            child = f'import os,time; from pathlib import Path; Path({str(pid_path)!r}).write_text(str(os.getpid())); time.sleep(60)'
            supervisor = (
                f'import sys,signal; sys.path.insert(0,{str(Path(__file__).parent)!r}); '
                'from scene_pipeline import run_command\n'
                'def cancel(a,b): raise KeyboardInterrupt()\n'
                'signal.signal(signal.SIGTERM,cancel)\n'
                f'try: run_command([sys.executable,"-c",{child!r}],{str(log)!r})\n'
                'except KeyboardInterrupt: sys.exit(130)\n')
            process = subprocess.Popen([sys.executable, '-c', supervisor])
            try:
                deadline = time.monotonic()+3
                while not pid_path.exists() and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue(pid_path.exists(), 'child did not start')
                child_pid = int(pid_path.read_text())
                process.send_signal(signal.SIGTERM)
                self.assertEqual(process.wait(timeout=8), 130)
                with self.assertRaises(ProcessLookupError):
                    os.kill(child_pid, 0)
                self.assertIn('command', log.read_text())
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()


if __name__ == '__main__':
    unittest.main()
