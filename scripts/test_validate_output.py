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


def add_vad_evidence(root, manifest):
    vad = root/'vad/frames.json'
    write_json(vad, {'file': str(root/'audio.wav'), 'sha256': manifest['clip']['sha256'],
                    'sample_rate': 16000, 'sample_count': 320,
                    'frames': [{'start_sample': 0, 'end_sample': 320, 'probability': .9}]})
    policy = {'raw_vad': fingerprint(vad)}
    plan = root/'vad-chunk-plan.json'
    write_json(plan, {'policy': policy, 'parts': [
        {k: v for k, v in p.items() if k not in ('file', 'source_start', 'source_end')}
        for p in manifest['qwen_chunks']]})
    manifest.update(qwen_chunk_policy=policy, vad_chunk_plan=fingerprint(plan))


def rebuild_document(root):
    from review_aligned_scene import make_document
    files = {'apple': 'apple/transcript-segments.json', 'diarization': 'diarization/pass-1.json',
             'qwen': 'qwen-comparison.json', 'audio_manifest': 'audio-manifest.json', 'alignment': 'alignment.json'}
    read = lambda name: json.loads((root/name).read_text())
    manifest, provenance = read('audio-manifest.json'), read('provenance.json')
    document = make_document(read('alignment.json'), read('diarization/pass-1.json'),
                             read('apple/transcript-segments.json'), provenance['input'],
                             {k: fingerprint(root/v) for k, v in files.items()},
                             dict(manifest['clip'], src='../audio.wav', source_offset=.01))
    document.update(provenance=provenance, processing=read('processing.json'),
                    qwen_chunks=read('qwen-comparison.json'), qwen_settings=read('qwen/settings.json'),
                    diarization_metadata={k: v for k, v in read('diarization/pass-1.json').items() if k != 'segments'},
                    **{k: manifest[k] for k in ('requested_window', 'actual_window', 'context_conditions')})
    return document


def make_run(root, gap=False, altered_chunk=False, broad_fallback=False):
    for name in ('qwen-input', 'qwen', 'apple', 'diarization', 'vad'):
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
    manifest.update(requested_window={'start': .01, 'end': .03},
                    actual_window={'start': .01, 'end': .03}, context_conditions={})
    write_json(root/'qwen/settings.json', {'fixture': True})
    add_vad_evidence(root, manifest)
    write_json(root / 'audio-manifest.json', manifest)
    write_json(root / 'qwen-comparison.json', {'incomplete': False, 'chunks': chunks})
    combined = '主張。\n理由。'
    (root / 'qwen-comparison.txt').write_text(combined + '\n')
    from compose_alignment import compose_alignments
    from test_aligned_speaker_turns import alignment
    children = []
    for i, text in enumerate(('主張。', '理由。')):
        child = alignment(text, [text[:-1]], [(.001, .009)], duration=.01, offset=.01+i*.01)
        if broad_fallback and i == 0:
            child['items'][0]['raw_clip_seconds'] = [0, 0]
            child['items'][0]['raw_source_seconds'] = [.01, .01]
        path = root/'alignments'/f'{i:03d}'/'alignment.json'
        path.parent.mkdir(parents=True)
        write_json(path, child)
        children.append({'start': i*.01, 'end': (i+1)*.01,
                         'alignment': child, 'source': fingerprint(path)})
    write_json(root/'alignment.json', compose_alignments(children, .01, .02))
    write_json(root / 'apple/transcript-segments.json', [])
    write_json(root / 'diarization/pass-1.json', {'segments': [{'start': 0, 'end': .02, 'speaker': 'raw-1'}]})
    source = fingerprint(original)
    provenance = {'input': source, 'settings': {'transcript_mode': 'qwen-aligned'}}
    write_json(root / 'provenance.json', provenance)
    write_json(root / 'processing.json', {'status': 'complete', 'stages': [{'status': 'complete'}]})
    document = rebuild_document(root)
    save(document, root / 'result')
    return document


class ValidateOutputTests(unittest.TestCase):
    def test_regenerated_exports_cannot_bless_changed_derived_evidence(self):
        import shutil
        for field in ('speaker', 'speaker_mapping', 'speaker_names', 'time', 'comparison', 'quality', 'candidates', 'playback', 'processing', 'qwen_settings', 'qwen_chunks', 'actual_window'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                document = make_run(root)
                if field == 'speaker':
                    document['segments'][0].update(speakers=['A'], state='single')
                elif field == 'speaker_mapping':
                    document['speaker_mapping'] = {'different-raw-id': 'A'}
                elif field == 'speaker_names':
                    document['speaker_names']['A'] = 'Edited identity'
                elif field == 'time':
                    document['segments'][0]['start'] += .0001
                elif field == 'comparison':
                    document['comparison']['differences'][0]['apple_text'] = 'Changed comparison evidence'
                elif field == 'quality':
                    document['segments'][0]['quality']['content'] = 'pass'
                elif field == 'candidates':
                    document['alignment_candidates']['diagnostics']['human_verified'] = True
                elif field == 'playback':
                    document['review_audio']['source_offset'] = 0
                else:
                    document[field] = {'stale': True}
                shutil.rmtree(root/'result')
                save(document, root/'result')
                report = validate(root)
                self.assertEqual(report['status'], 'failed', report)
                self.assertEqual(report['errors'][0]['check'], 'derived_transcript_from_raw_evidence')

    def test_all_recorded_file_paths_are_checked_even_for_identical_content(self):
        import shutil
        for name in ('apple', 'diarization', 'qwen', 'audio_manifest', 'alignment'):
            for missing in (True, False):
                with self.subTest(source=name, missing=missing), tempfile.TemporaryDirectory() as folder:
                    root = Path(folder)
                    document = make_run(root)
                    other = root/'another.json'
                    if not missing:
                        other.write_bytes(Path(document['raw_sources'][name]['path']).read_bytes())
                    document['raw_sources'][name]['path'] = str(other)
                    shutil.rmtree(root/'result')
                    save(document, root/'result')
                    report = validate(root)
                    self.assertEqual(report['status'], 'failed', report)
                    self.assertIn('Recorded file path mismatch', report['errors'][0]['message'])
        # The same primitive covers PCM, ASR chunks, VAD and child alignments.
        from validate_output import file_matches
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'audio.wav'
            path.write_bytes(b'pcm')
            identity = fingerprint(path)
            identity['path'] = str(path.with_name('missing.wav'))
            with self.assertRaisesRegex(ValueError, 'Recorded file path mismatch'):
                file_matches(path, identity)

    def test_vad_raw_plan_and_manifest_must_agree(self):
        for mode in ('raw_deleted', 'raw_changed', 'plan_deleted', 'plan_changed', 'policy', 'cut', 'wrong_audio'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                document = make_run(root)
                raw, plan_path = root/'vad/frames.json', root/'vad-chunk-plan.json'
                if mode.endswith('deleted'):
                    (raw if mode.startswith('raw') else plan_path).unlink()
                elif mode.endswith('changed'):
                    (raw if mode.startswith('raw') else plan_path).write_text('{}')
                else:
                    manifest = json.loads((root/'audio-manifest.json').read_text())
                    plan = json.loads(plan_path.read_text())
                    if mode == 'policy':
                        plan['policy']['extra'] = 'contradiction'
                    elif mode == 'cut':
                        plan['parts'][0]['end_sample'] += 1
                    else:
                        vad = json.loads(raw.read_text())
                        vad['sha256'] = '0'*64
                        write_json(raw, vad)
                        plan['policy']['raw_vad'] = fingerprint(raw)
                        manifest['qwen_chunk_policy'] = plan['policy']
                    write_json(plan_path, plan)
                    manifest['vad_chunk_plan'] = fingerprint(plan_path)
                    write_json(root/'audio-manifest.json', manifest)
                    document['raw_sources']['audio_manifest'] = fingerprint(root/'audio-manifest.json')
                    document['artifact_id'] = scene_artifact_id(document)
                    write_json(root/'result/transcript.json', document)
                report = validate(root)
                self.assertEqual(report['status'], 'failed', report)
                self.assertEqual(report['errors'][0]['check'], 'vad_evidence_and_chunk_plan')

    def test_child_alignment_evidence_cannot_be_missing_changed_or_contradictory(self):
        for mode in ('deleted', 'changed', 'metadata_conflict', 'missing_reference', 'missing_contract'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                document = make_run(root)
                child = root/'alignments/001/alignment.json'
                if mode == 'deleted':
                    child.unlink()
                elif mode == 'changed':
                    child.write_text('{}')
                else:
                    # Updating outer hashes cannot bless a contradictory child reference.
                    composed = json.loads((root/'alignment.json').read_text())
                    if mode == 'metadata_conflict':
                        composed['composition_parts'][1]['source']['sha256'] = 'a'*64
                    elif mode == 'missing_reference':
                        composed['raw_children'][1]['source']['path'] = str(root/'missing.json')
                    else:
                        composed.pop('kind')
                    write_json(root/'alignment.json', composed)
                    document['raw_sources']['alignment'] = fingerprint(root/'alignment.json')
                    document['artifact_id'] = scene_artifact_id(document)
                    write_json(root/'result/transcript.json', document)
                report = validate(root)
                self.assertEqual(report['status'], 'failed', report)
                self.assertEqual(report['errors'][0]['check'], 'alignment_and_segment_text_preservation')

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
            add_vad_evidence(root, manifest)
            write_json(root/'audio-manifest.json', manifest)
            raw = root/'qwen/000.json'
            write_json(raw, {'file': str(audio), 'raw': {'text': ''}})
            write_json(root/'qwen-comparison.json', {'incomplete': False, 'chunks': [dict(
                part, text='', raw_output=fingerprint(raw), token_limit_reached=False)]})
            (root/'qwen-comparison.txt').write_text('\n')
            from compose_alignment import compose_alignments
            from test_aligned_speaker_turns import alignment
            child = alignment('', [], [], duration=.02, offset=.01)
            child_path = root/'alignments/000/alignment.json'
            write_json(child_path, child)
            write_json(root/'alignment.json', compose_alignments([
                {'start': 0., 'end': .02, 'alignment': child, 'source': fingerprint(child_path)}], .01, .02))
            document = rebuild_document(root)
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
            self.assertEqual(indicators['speaker_state_counts']['unknown'], 2)
            self.assertEqual(indicators['broad_timing_fallback']['row_ids'], ['0'])
            self.assertAlmostEqual(indicators['max_row_duration_seconds'], .01)
            self.assertEqual(indicators['rows_covering_multiple_chunks']['count'], 0)
            self.assertTrue(report['warnings'])

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
