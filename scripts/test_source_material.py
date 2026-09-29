"""Source-bundle integrity tests; these do not measure acoustic correctness."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scene_transcript import save, scene_artifact_id, source_material


def fixture():
    """Synthetic Qwen-shaped rows, including uncertainty and a comparison tail."""
    document = {
        'schema_version': 1,
        'input': {'path': '/synthetic/input.wav', 'sha256': 'a' * 64, 'bytes': 123},
        'source_window': {'start': 1200, 'end': 1230},
        'requested_window': {'start': 1205, 'end': 1225},
        'context_conditions': {'applied': True},
        'asr': {'engine': 'Qwen3-ASR-1.7B BF16', 'text_modified': False,
                'secondary': 'Apple SpeechTranscriber; comparison only'},
        'speaker_names': {'A': '話者A'}, 'speaker_mapping': {'raw-1': 'A'},
        'speaker_identity_scope': 'one supplied diarization invocation',
        'raw_sources': {'qwen': {'sha256': 'b' * 64}},
        'provenance': {'settings': {'transcript_mode': 'qwen-aligned'}, 'models': {
            'qwen': {'recorded': {'repo': 'example/model', 'revision': 'c' * 40,
                                 'license': 'Apache-2.0', 'files': [{'path': 'large-model'}]},
                     'manifest': {'sha256': 'd' * 64}}}},
        'comparison': {'method': 'text anchors, not acoustic alignment', 'differences': [
            {'apple_text': '原則できません', 'qwen_text': '原則できます',
             'qwen_raw_span': [0, 7], 'apple_native_unit_envelope': [1.2, 5.4]},
            {'apple_text': '別の補足', 'qwen_text': '', 'qwen_raw_span': [16, 16]}]},
        'segments': [
            {'id': '0', 'start': 1201.2, 'end': 1205.4,
             'text': '原則できます。\n「例」は100％。', 'speakers': ['A'], 'state': 'mixed',
             'source_raw_span': [0, 18], 'playback_start': 1200, 'playback_end': 1207.4,
             'timing_provenance': 'forced token envelope',
             'asr_difference_ids': [0], 'flags': ['raw_diarization_overlap_ownership_unverified'],
             'review_reasons': ['human_content_speaker_and_boundary_review_pending'],
             'quality': {'content': 'review_required', 'speaker': 'review_required', 'boundary': 'review_required'},
             'human_verified': False, 'ready_for_attributed_summary': False},
            {'id': '1', 'start': 1205.4, 'end': 1206,
             'text': '違います。', 'speakers': [], 'state': 'unknown',
             'asr_difference_ids': [], 'flags': ['brief_sentence_ownership_unresolved'],
             'ready_for_attributed_summary': False}],
        'validation': {'asr_text_preserved': True, 'accuracy_verified': False},
        'ready_for_attributed_summary': False,
    }
    document['artifact_id'] = scene_artifact_id(document)
    return document


class SourceMaterialTests(unittest.TestCase):
    def test_compact_views_preserve_text_time_ids_uncertainty_without_mutation(self):
        document = fixture()
        before = copy.deepcopy(document)
        metadata, rows, differences = source_material(document)
        self.assertEqual(document, before)
        self.assertEqual(''.join(row['text'] for row in rows),
                         ''.join(row['text'] for row in document['segments']))
        for original, row in zip(document['segments'], rows):
            for field in ('id', 'start', 'end', 'text', 'speakers', 'state', 'flags',
                          'asr_difference_ids', 'ready_for_attributed_summary'):
                self.assertEqual(row[field], original[field])
            self.assertEqual(row['artifact_id'], document['artifact_id'])
        self.assertEqual(rows[0]['quality'], document['segments'][0]['quality'])
        self.assertFalse(rows[0]['human_verified'])
        self.assertEqual(metadata['source_window']['start'], 1200)
        self.assertNotEqual(metadata['source_window'], metadata['requested_window'])
        self.assertFalse(metadata['quality']['accuracy_verified'])
        self.assertEqual(metadata['models']['qwen']['revision'], 'c' * 40)
        self.assertNotIn('files', metadata['models']['qwen'])
        self.assertEqual(differences[0]['difference'], document['comparison']['differences'][0])
        self.assertEqual(differences[0]['segment_ids'], ['0'])
        self.assertEqual(differences[1]['segment_ids'], [])
        self.assertEqual(len(differences), 2, 'Unassigned Apple-only text must remain available')

    def test_serialized_bundle_survives_unicode_newlines_and_keeps_primary_text_separate(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'result'
            document = fixture()
            save(document, path)
            full = json.loads((path / 'transcript.json').read_text())
            rows = [json.loads(line) for line in (path / 'segments.jsonl').read_text().splitlines()]
            differences = [json.loads(line) for line in (path / 'asr-differences.jsonl').read_text().splitlines()]
            self.assertFalse((path / 'review.html').exists())
            self.assertEqual(len(rows), 2, 'Embedded newlines must not split JSONL records')
            self.assertEqual([r['text'] for r in rows], [r['text'] for r in full['segments']])
            self.assertNotIn('原則できません', (path / 'segments.jsonl').read_text())
            self.assertEqual(differences[0]['difference']['apple_text'], '原則できません')
            self.assertEqual({r['artifact_id'] for r in rows}, {full['artifact_id']})
            self.assertEqual(full['artifact_id'], scene_artifact_id(full))
            self.assertIn('全文の聴取確認', (path / 'README.md').read_text())
            self.assertIn('別話者の発言が混ざる可能性', (path / 'transcript.txt').read_text())
            self.assertIn('話者不明', (path / 'transcript.txt').read_text())

    def test_existing_bundle_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'result'
            save(fixture(), path)
            before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir()}
            with self.assertRaises(FileExistsError):
                save(fixture(), path)
            self.assertEqual(before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir()})

    def test_invalid_ids_times_references_or_nonfinite_metadata_leave_no_partial_bundle(self):
        cases = [
            ('duplicate_id', lambda d: d['segments'][1].update(id='0')),
            ('nonstring_id', lambda d: d['segments'][1].update(id=1)),
            ('reversed_time', lambda d: d['segments'][0].update(end=1200)),
            ('negative_time', lambda d: d['segments'][0].update(start=-1)),
            ('boolean_time', lambda d: d['segments'][0].update(start=True)),
            ('bad_reference', lambda d: d['segments'][0].update(asr_difference_ids=[5])),
            ('boolean_reference', lambda d: d['segments'][0].update(asr_difference_ids=[True])),
            ('unknown_speaker', lambda d: d['segments'][0].update(speakers=['Z'])),
            ('nonfinite_metadata', lambda d: d.update(processing={'elapsed': float('nan')})),
        ]
        for name, change in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                document = fixture()
                change(document)
                output = Path(folder) / 'result'
                with self.assertRaises(ValueError):
                    save(document, output)
                self.assertFalse(output.exists())

    def test_ids_are_scoped_to_artifact_and_do_not_depend_on_run_timing(self):
        document = fixture()
        artifact = document['artifact_id']
        document['processing'] = {'total_seconds': 999}
        self.assertEqual(scene_artifact_id(document), artifact)
        document['segments'][0]['text'] = '音声で確認した訂正文。'
        self.assertNotEqual(scene_artifact_id(document), artifact)


if __name__ == '__main__':
    unittest.main()
