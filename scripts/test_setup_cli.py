import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import setup_cli


class SetupTests(unittest.TestCase):
    def test_selected_python_recreates_mismatched_environment(self):
        import sys
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            python = root/'.venv-asr/bin/python'
            python.parent.mkdir(parents=True)
            python.write_bytes(b'fixture executable')
            for actual, expected_calls in [('/different/python', 2), (sys._base_executable, 1)]:
                with self.subTest(actual=actual), patch.dict('os.environ', {'PYTHON': sys.executable}), \
                        patch.object(setup_cli, 'command', return_value=Mock(stdout=actual)) as command:
                    setup_cli.prepare_interpreter(python, root)
                    self.assertEqual(command.call_count, expected_calls)
                    if expected_calls == 2:
                        self.assertEqual(command.call_args.args[0],
                                         [sys.executable, '-m', 'venv', '--clear', root/'.venv-asr'])

    def test_verify_rejects_invalid_successful_apple_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            apple = root/'.build/release/apple-transcribe'
            apple.parent.mkdir(parents=True)
            apple.touch()
            for body, expected in [('not json', False), ('[]', False), ('null', False),
                                   ('{"status":"missing","installed":false}', False),
                                   ('{"status":"ready","installed":true}', True)]:
                with self.subTest(body=body), \
                        patch.object(setup_cli, 'manifests', return_value=[]), \
                        patch.object(setup_cli, 'runtime_config', return_value={'swift_products': []}), \
                        patch.object(setup_cli, 'runtime_versions', return_value={}), \
                        patch.object(setup_cli, 'verify_vendor', return_value={}), \
                        patch.object(setup_cli.subprocess, 'run', return_value=Mock(returncode=0, stdout=body, stderr='')):
                    self.assertEqual(setup_cli.verify(root)['ready'], expected)

    def test_untracked_swift_source_is_rejected_without_modifying_vendor(self):
        import subprocess
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            vendor = root/'vendor/FluidAudio'
            vendor.mkdir(parents=True)
            def git(*args):
                return subprocess.run(['git', '-C', str(vendor), *args], check=True, capture_output=True, text=True).stdout
            git('init', '-q')
            git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                'commit', '--allow-empty', '-qm', 'fixture')
            patch_file = root/'fixture.patch'
            patch_file.write_text('')
            config = {'fluid_audio': {'revision': git('rev-parse', 'HEAD').strip(),
                                     'patch': 'fixture.patch', 'patch_sha256': setup_cli.digest(patch_file)}}
            with patch.object(setup_cli, 'runtime_config', return_value=config):
                setup_cli.verify_vendor(root)
                extra = vendor/'Extra.swift'
                extra.write_text('// untracked compiled source')
                with self.assertRaisesRegex(ValueError, 'changes differ'):
                    setup_cli.verify_vendor(root)
                self.assertTrue(extra.exists())

    def manifest(self, data=b'model'):
        return {'repo': 'example/model', 'revision': 'a' * 40,
                'directory': 'models/example',
                'files': [{'path': 'nested/weights.bin', 'bytes': len(data),
                           'sha256': hashlib.sha256(data).hexdigest()}]}

    def test_existing_valid_model_is_reused_without_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.manifest()
            destination = root / manifest['directory'] / manifest['files'][0]['path']
            destination.parent.mkdir(parents=True)
            destination.write_bytes(b'model')
            opener = Mock(side_effect=AssertionError('Network must not run'))
            setup_cli.fetch_model(manifest, root, opener)
            opener.assert_not_called()
            self.assertEqual(destination.read_bytes(), b'model')

    def test_invalid_existing_model_is_preserved_without_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.manifest()
            destination = root / manifest['directory'] / manifest['files'][0]['path']
            destination.parent.mkdir(parents=True)
            destination.write_bytes(b'other')
            opener = Mock()
            with self.assertRaisesRegex(ValueError, 'preserved'):
                setup_cli.fetch_model(manifest, root, opener)
            opener.assert_not_called()
            self.assertEqual(destination.read_bytes(), b'other')

    def test_bad_download_is_not_promoted_to_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.manifest()
            with self.assertRaises(ValueError):
                setup_cli.fetch_model(manifest, root, lambda *a, **kw: io.BytesIO(b'wrong'))
            self.assertFalse((root / manifest['directory'] / manifest['files'][0]['path']).exists())
            self.assertEqual(list(root.rglob('*.download')), [])

    def test_valid_download_uses_fixed_revision_and_verified_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self.manifest()
            opener = Mock(return_value=io.BytesIO(b'model'))
            setup_cli.fetch_model(manifest, root, opener)
            self.assertEqual(opener.call_args.args[0],
                             'https://huggingface.co/example/model/resolve/' + 'a' * 40 + '/nested/weights.bin')
            self.assertEqual((root / manifest['directory'] / manifest['files'][0]['path']).read_bytes(), b'model')

    def test_manifest_paths_cannot_escape_model_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for invalid in ('../elsewhere', '/absolute', ''):
                with self.assertRaises(ValueError):
                    setup_cli.within(root, invalid)
            (root / 'escape').symlink_to(root.parent)
            with self.assertRaises(ValueError):
                setup_cli.within(root, 'escape/file')

    def test_fluid_cache_is_prepared_from_verified_assets_without_runtime_downloads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = dict(self.manifest(), repo='FluidInference/speaker-diarization-coreml',
                            directory='models/fluid/speaker-diarization')
            setup_cli.fetch_model(manifest, root, lambda *a, **kw: io.BytesIO(b'model'))
            directory = root / manifest['directory']
            marker = directory / '.fluidaudio-revision'
            self.assertEqual(marker.read_text().strip(), manifest['revision'])
            setup_cli.verify_cache_revision(manifest, directory)
            marker.unlink()
            with self.assertRaises(FileNotFoundError):
                setup_cli.verify_cache_revision(manifest, directory)
            self.assertFalse(marker.exists(), 'Verification must not repair missing metadata')
            opener = Mock(side_effect=AssertionError('Verified weights must not be fetched again'))
            setup_cli.fetch_model(manifest, root, opener)
            opener.assert_not_called()
            setup_cli.verify_cache_revision(manifest, directory)

    def test_fluid_marker_cannot_bless_bad_assets_or_overwrite_another_revision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = dict(self.manifest(), repo='FluidInference/speaker-diarization-coreml')
            directory = root / manifest['directory']
            marker = directory / '.fluidaudio-revision'
            with self.assertRaises(ValueError):
                setup_cli.fetch_model(manifest, root, lambda *a, **kw: io.BytesIO(b'wrong'))
            self.assertFalse(marker.exists())
            setup_cli.fetch_model(manifest, root, lambda *a, **kw: io.BytesIO(b'model'))
            marker.write_text('b' * 40 + '\n')
            opener = Mock(side_effect=AssertionError('No network'))
            with self.assertRaisesRegex(ValueError, 'cache revision differs'):
                setup_cli.fetch_model(manifest, root, opener)
            opener.assert_not_called()
            self.assertEqual(marker.read_text(), 'b' * 40 + '\n')

    def test_default_plan_has_only_supported_models_and_no_network(self):
        with patch('urllib.request.urlopen', side_effect=AssertionError('No network')):
            result = setup_cli.plan()
        self.assertEqual(result['total_model_bytes'], 5943320786)
        self.assertEqual(len(result['models']), 4)
        self.assertEqual({item['directory'] for item in result['models']},
                         {'models/qwen17', 'models/qwen-aligner', 'models/silero-vad32',
                          'models/fluid/speaker-diarization'})


if __name__ == '__main__':
    unittest.main()
