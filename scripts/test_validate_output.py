"""Validate synthetic runs without downloading models or recognizing audio."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from scene_pipeline import fingerprint, write_float_wav
from scene_transcript import save, scene_artifact_id
from validate_output import validate


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


def make_run(root, gap=False, altered_chunk=False, broad_fallback=False):
    for name in ('qwen-input', 'qwen', 'apple', 'diarization'):
        (root / name).mkdir()
    original = root / 'original.bin'
    original.write_bytes(b'synthetic source input')
    pcm = struct.pack('<640f', *(i / 640 for i in range(640)))
    write_float_wav(root / 'normalized.wav', pcm)
    selected = pcm[160 * 4:480 * 4]
    write_float_wav(root / 'audio.wav', selected)
    parts, chunks = [], []
    for i, text in enumerate(('主張。', '理由。')):
        start, end = i * 160 + (1 if gap and i == 1 else 0), (i + 1) * 160
        audio_path = root / 'qwen-input' / f'{i:03d}.wav'
        samples = selected[start * 4:end * 4]
        if altered_chunk and i == 1:
            samples = b'\x00' * len(samples)
        write_float_wav(audio_path, samples)
        part = {'index': i, 'start_sample': start, 'end_sample': end,
                'source_start': .01 + start / 16000, 'source_end': .01 + end / 16000,
                'boundary_kind': ('forced_low_energy_no_vad_pause' if broad_fallback else 'vad_pause_preferred') if i == 0 else 'input_end',
                'file': fingerprint(audio_path)}
        parts.append(part)
        raw_path = root / 'qwen' / f'{i:03d}.json'
        write_json(raw_path, {'file': str(audio_path), 'raw': {'text': text}, 'token_limit_reached': False})
        chunks.append(dict(part, text=text, raw_output=fingerprint(raw_path), token_limit_reached=False))
    manifest = {'sample_rate': 16000, 'source_duration': .04,
                'normalized': fingerprint(root / 'normalized.wav'),
                'clip': fingerprint(root / 'audio.wav'),
                'window': {'start_sample': 160, 'end_sample': 480, 'source_offset': .01, 'duration': .02},
                'qwen_chunks': parts}
    write_json(root / 'audio-manifest.json', manifest)
    write_json(root / 'qwen-comparison.json', {'incomplete': False, 'chunks': chunks})
    combined = '主張。\n理由。'
    (root / 'qwen-comparison.txt').write_text(combined + '\n')
    write_json(root / 'alignment.json', {'mapping': {'original_text': combined},
                                        'source_offset_seconds': .01, 'duration_seconds': .02,
                                        'items': [{}, {}], 'composition_parts': [{}, {}]})
    write_json(root / 'apple/transcript-segments.json', [])
    write_json(root / 'diarization/pass-1.json', {'segments': [{'start': 0, 'end': .02, 'speaker': 'raw-1'}]})
    source_files = {'apple': 'apple/transcript-segments.json', 'diarization': 'diarization/pass-1.json',
                    'qwen': 'qwen-comparison.json', 'audio_manifest': 'audio-manifest.json', 'alignment': 'alignment.json'}
    source = fingerprint(original)
    provenance = {'input': source, 'settings': {'transcript_mode': 'qwen-aligned'}}
    write_json(root / 'provenance.json', provenance)
    write_json(root / 'processing.json', {'status': 'complete', 'stages': [{'status': 'complete'}]})
    rows = []
    for i, (start, end, text, span) in enumerate(((.011, .019, '主張。\n', [0, 4]), (.02, .029, '理由。', [4, 7]))):
        rows.append({'id': str(i), 'start': start, 'end': end, 'text': text, 'source_raw_span': span,
                     'speakers': [], 'state': 'unknown', 'playback_start': .01, 'playback_end': .03,
                     'forced_token_references': [i], 'alignment_part_indices': [i],
                     'raw_diarization_event_ids': [0], 'raw_other_speaker_event_ids': [], 'asr_difference_ids': []})
    if broad_fallback:
        rows[0].update(start=.01, end=.03, timing_provenance='whole clip fallback; sentence has no safe timed envelope',
                       flags=['broad_playback_timing_fallback'], alignment_part_indices=[0, 1],
                       state='mixed', speakers=['A'])
    document = {'source_window': {'start': .01, 'end': .03}, 'input': source,
                'provenance': provenance, 'speaker_names': {'A': '話者A'}, 'segments': rows,
                'asr': {'engine': 'synthetic Qwen-shaped fixture'},
                'comparison': {'differences': []},
                'raw_sources': {k: fingerprint(root / path) for k, path in source_files.items()}}
    save(document, root / 'result')
    return document


class ValidateOutputTests(unittest.TestCase):
    def test_entire_textless_run_keeps_audio_coverage_and_empty_exports(self):
        import shutil
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            document = make_run(root)
            manifest = json.loads((root/'audio-manifest.json').read_text())
            part = manifest['qwen_chunks'][0]
            part.update(end_sample=320, source_end=.03, boundary_kind='input_end')
            audio = root/'qwen-input/000.wav'
            audio.write_bytes((root/'audio.wav').read_bytes())
            part['file'] = fingerprint(audio)
            manifest['qwen_chunks'] = [part]
            write_json(root/'audio-manifest.json', manifest)
            raw = root/'qwen/000.json'
            write_json(raw, {'file': str(audio), 'raw': {'text': ''}})
            write_json(root/'qwen-comparison.json', {'incomplete': False, 'chunks': [dict(
                part, text='', raw_output=fingerprint(raw), token_limit_reached=False)]})
            (root/'qwen-comparison.txt').write_text('\n')
            write_json(root/'alignment.json', {'mapping': {'original_text': ''},
                       'source_offset_seconds': .01, 'duration_seconds': .02,
                       'items': [], 'composition_parts': [{}]})
            document['segments'] = []
            for name, entry in document['raw_sources'].items():
                document['raw_sources'][name] = fingerprint(entry['path'])
            shutil.rmtree(root/'result')
            save(document, root/'result')
            report = validate(root)
            self.assertEqual(report['status'], 'passed', report)
            self.assertEqual(report['counts']['selected_samples'], 320)
            self.assertEqual(report['counts']['text_characters'], 0)
            self.assertEqual((root/'result/segments.jsonl').read_text(), '')

    def test_nonempty_but_stale_readable_exports_are_rejected(self):
        for name in ('transcript.txt', 'transcript.md', 'README.md'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                make_run(root)
                (root/'result'/name).write_text('Nonempty stale text')
                report = validate(root)
                self.assertEqual(report['status'], 'failed')
                self.assertEqual(report['errors'][0]['check'], 'compact_exports_and_readable_text')

    def test_two_chunk_nonzero_window_passes_without_quality_claim_or_writes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            make_run(root)
            before = {p.relative_to(root): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
            report = validate(root)
            self.assertEqual(report['status'], 'passed', report)
            self.assertEqual(report['counts'], {'chunks': 2, 'segments': 2, 'text_characters': 7, 'selected_samples': 320})
            self.assertEqual(report['source_input_sha256'], 'matched')
            self.assertFalse(report['accuracy_verified'])
            after = {p.relative_to(root): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
            self.assertEqual(before, after)

    def test_recorded_gap_is_rejected_even_with_matching_fingerprints(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            make_run(root, gap=True)
            report = validate(root)
            self.assertEqual(report['status'], 'failed')
            self.assertIn('gap', report['errors'][0]['message'])

    def test_broad_ranges_are_reported_as_quality_indicators_not_structural_errors(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            make_run(root, broad_fallback=True)
            report = validate(root)
            self.assertEqual(report['status'], 'passed', report)
            self.assertEqual(report['errors'], [])
            self.assertFalse(report['accuracy_verified'])
            indicators = report['quality_indicators']
            self.assertTrue(indicators['counts_are_not_accuracy'])
            self.assertEqual(indicators['speaker_state_counts'], {'single': 0, 'mixed': 1, 'unknown': 1})
            self.assertEqual(indicators['timing_provenance_counts']['whole clip fallback; sentence has no safe timed envelope'], 1)
            self.assertAlmostEqual(indicators['max_row_duration_seconds'], .02)
            self.assertEqual(indicators['max_row_duration_ids'], ['0'])
            for field in ('broad_timing_fallback', 'whole_source_window_timing',
                          'rows_covering_multiple_chunks', 'rows_longer_than_longest_input_chunk'):
                self.assertEqual(indicators[field]['count'], 1)
                self.assertEqual(indicators[field]['row_ids'], ['0'])
            self.assertEqual(indicators['chunks']['min_duration_seconds'], .01)
            self.assertEqual(indicators['chunks']['max_duration_seconds'], .01)
            self.assertEqual(indicators['chunks']['vad_fallback_boundaries'], {'count': 1, 'chunk_indices': [0]})
            self.assertEqual(len(report['warnings']), 4)

    def test_chunk_pcm_mutation_is_rejected_even_with_matching_fingerprints(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            make_run(root, altered_chunk=True)
            report = validate(root)
            self.assertEqual(report['status'], 'failed')
            self.assertIn('Chunk PCM differs', report['errors'][0]['message'])

    def test_original_input_missing_is_explicit_but_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            make_run(root)
            (root / 'original.bin').write_bytes(b'changed source input')
            self.assertEqual(validate(root)['errors'][0]['check'], 'source_input_identity')
            (root / 'original.bin').unlink()
            report = validate(root)
            self.assertEqual(report['status'], 'passed', report)
            self.assertEqual(report['source_input_sha256'], 'not_checked_input_unavailable')
            self.assertTrue(report['warnings'])

    def test_changed_compact_text_and_empty_markdown_fail(self):
        for name, content in (('segments.jsonl', '{}\n'), ('transcript.md', '\n')):
            with self.subTest(file=name), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                make_run(root)
                (root / 'result' / name).write_text(content)
                report = validate(root)
                self.assertEqual(report['status'], 'failed')
                self.assertEqual(report['errors'][0]['check'], 'compact_exports_and_readable_text')

    def test_full_text_times_and_references_are_independently_checked(self):
        cases = (
            ('text', '欠落した文。', 'alignment_and_segment_text_preservation'),
            ('end', .04, 'timestamps_ids_and_references'),
            ('forced_token_references', [2], 'timestamps_ids_and_references'),
            ('id', '1', 'timestamps_ids_and_references'),
        )
        for key, value, check in cases:
            with self.subTest(field=key), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                document = make_run(root)
                document['segments'][0][key] = value
                document['artifact_id'] = scene_artifact_id(document)
                write_json(root / 'result/transcript.json', document)
                report = validate(root)
                self.assertEqual(report['status'], 'failed')
                self.assertEqual(report['errors'][0]['check'], check)

    def test_incomplete_run_and_malformed_json_fail_as_machine_reports(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            make_run(root)
            for text in ('{"status":"running"}', '{bad-json', '{"status":NaN}'):
                (root / 'processing.json').write_text(text)
                report = validate(root)
                self.assertEqual(report['status'], 'failed')
                self.assertEqual(report['errors'][0]['check'], 'completed_run')

    def test_cli_reports_to_new_file_only(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            make_run(root)
            output = root / 'validation.json'
            command = [sys.executable, str(Path(__file__).with_name('validate_output.py')), str(root), '--report', str(output)]
            first = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            before = output.read_bytes()
            self.assertEqual(json.loads(first.stdout)['status'], 'passed')
            second = subprocess.run(command, text=True, capture_output=True)
            self.assertEqual(second.returncode, 1)
            self.assertEqual(json.loads(second.stdout)['errors'][-1]['check'], 'report_output')
            self.assertEqual(before, output.read_bytes())


if __name__ == '__main__':
    unittest.main()
