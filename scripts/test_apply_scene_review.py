"""Human-review provenance and scope checks; not an acoustic quality evaluation."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from apply_scene_review import apply_review
from scene_transcript import scene_artifact_id


def source():
    document = {
        'schema_version': 1, 'input': {'path': 'fixture.wav', 'sha256': 'a'*64, 'bytes': 123},
        'source_window': {'start': 100., 'end': 130.},
        'speaker_names': {'A': '話者A', 'B': '話者B'}, 'speaker_mapping': {'one': 'A', 'two': 'B'},
        'raw_sources': {'apple': {'sha256': 'b'*64}}, 'rules': {'numbers_are_confidence': False},
        'review_audio': {'sha256': 'c'*64, 'source_offset': 100.},
        'asr': {'text_modified': False}, 'units': [{'text': '原出力を保持'}],
        'comparison': {'apple_text': '原出力を保持', 'qwen_text': '比較原出力'},
        'validation': {'asr_text_preserved': True, 'accuracy_verified': False},
        'ready_for_attributed_summary': False,
        'segments': [
            {'id': '0', 'start': 100., 'end': 110., 'text': '重要な主張です。', 'speakers': ['A'], 'state': 'single'},
            {'id': '1', 'start': 120., 'end': 130., 'text': 'まだ確認していない理由です。', 'speakers': ['B'], 'state': 'single'}]}
    document['artifact_id'] = scene_artifact_id(document)
    return document


def review_for(document):
    review = {'schema_version': 1, 'source_artifact_id': document['artifact_id'],
              'source': copy.deepcopy(document['input']), 'reviewer': 'test reviewer (synthetic)',
              'reviewed_at': '2026-09-29T12:00:00+09:00',
              'listened_ranges': [{'start': 100., 'end': 110.}], 'rows': {}}
    for row in document['segments']:
        review['rows'][row['id']] = {key: copy.deepcopy(row[key]) for key in ('start', 'end', 'text', 'speakers')}
        review['rows'][row['id']].update(content='review_required', speaker='review_required',
                                        boundary='review_required', note='')
    review['rows']['0'].update(content='pass', speaker='pass', boundary='pass')
    return review


class ApplyReviewTests(unittest.TestCase):
    def test_changed_comparison_invalidates_existing_review(self):
        document = source()
        document['comparison']['differences'] = [{'apple_text': '否定', 'qwen_text': '肯定'}]
        document['artifact_id'] = scene_artifact_id(document)
        review = review_for(document)
        document['comparison']['differences'][0]['apple_text'] = '別の根拠'
        with self.assertRaises(ValueError):
            apply_review(document, review)
        document['artifact_id'] = scene_artifact_id(document)
        with self.assertRaises(ValueError):
            apply_review(document, review)

    def test_speaker_pass_preserves_production_overlap_and_mixed_state(self):
        document = source()
        row = document['segments'][0]
        row['state'] = 'mixed'
        row['raw_overlap_intervals'] = [{'start': 101., 'end': 102., 'speakers': ['one', 'two']}]
        document['artifact_id'] = scene_artifact_id(document)
        result, ready = apply_review(document, review_for(document))
        self.assertEqual(result['segments'][0]['state'], 'mixed')
        self.assertEqual(result['segments'][0]['raw_overlap_intervals'], row['raw_overlap_intervals'])
        self.assertEqual(ready['segments'][0]['state'], 'mixed')

    def test_partial_review_never_approves_whole_transcript(self):
        document = source()
        before = copy.deepcopy(document)
        result, ready = apply_review(document, review_for(document))
        self.assertEqual(document, before)
        self.assertEqual([r['id'] for r in ready['segments']], ['0'])
        self.assertTrue(ready['segments'][0]['ready_for_attributed_summary'])
        self.assertFalse(result['ready_for_attributed_summary'])
        self.assertFalse(result['ready_for_anonymous_summary'])
        self.assertEqual(set(result['segments'][1]['quality'].values()), {'review_required'})
        self.assertEqual(result['artifact_id'], scene_artifact_id(result))
        self.assertNotEqual(result['artifact_id'], document['artifact_id'])

    def test_artifact_tampering_or_wrong_input_hash_is_rejected(self):
        for kind in ('source_body', 'source_id', 'review_id', 'input_hash'):
            document = source()
            review = review_for(document)
            if kind == 'source_body':
                document['segments'][0]['text'] = '書き換わった原文'
            elif kind == 'source_id':
                document['artifact_id'] = 'bad'
            elif kind == 'review_id':
                review['source_artifact_id'] = 'bad'
            else:
                review['source']['sha256'] = 'd'*64
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                apply_review(document, review)

    def test_full_row_must_be_inside_listened_ranges(self):
        document = source()
        review = review_for(document)
        review['listened_ranges'] = [{'start': 105, 'end': 110}]
        with self.assertRaisesRegex(ValueError, 'partial listening'):
            apply_review(document, review)
        review['listened_ranges'] = [{'start': 100, 'end': 105}, {'start': 105, 'end': 110}]
        self.assertEqual(len(apply_review(document, review)[1]['segments']), 1)
        review['listened_ranges'][1]['start'] = 105.1
        with self.assertRaisesRegex(ValueError, 'partial listening'):
            apply_review(document, review)

    def test_corrected_range_also_requires_listening_and_valid_speaker(self):
        for correction in ({'start': 99}, {'end': 100}, {'start': 111, 'end': 110},
                           {'speakers': ['C']}, {'speakers': ['A', 'A']},
                           {'end': 112}, {'start': float('nan')}, {'start': True}):
            document = source()
            review = review_for(document)
            review['rows']['0']['corrections'] = correction
            with self.subTest(correction=correction), self.assertRaises(ValueError):
                apply_review(document, review)

    def test_row_id_and_original_fields_must_match(self):
        for field in ('id', 'start', 'end', 'text', 'speakers'):
            document = source()
            review = review_for(document)
            if field == 'id':
                review['rows']['unknown'] = review['rows'].pop('0')
            else:
                review['rows']['0'][field] = {'start': 101, 'end': 109, 'text': '違う原文', 'speakers': ['B']}[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                apply_review(document, review)

    def test_reasoned_speaker_na_allows_only_anonymous_summary(self):
        document = source()
        review = review_for(document)
        review['rows']['0'].update(speaker='not_applicable', note='話者別の引用に用いず、匿名の内容要約に限定する。')
        result, ready = apply_review(document, review)
        row = ready['segments'][0]
        self.assertTrue(row['ready_for_anonymous_summary'])
        self.assertFalse(row['ready_for_attributed_summary'])
        self.assertEqual(row['speakers'], [])
        self.assertEqual(row['candidate_speakers'], ['A'])
        review['rows']['0']['note'] = ''
        with self.assertRaisesRegex(ValueError, 'reason'):
            apply_review(document, review)

    def test_corrections_never_modify_model_units_or_original_rows(self):
        document = source()
        before = copy.deepcopy(document)
        review = review_for(document)
        review['listened_ranges'] = [{'start': 100, 'end': 112}]
        review['rows']['0']['corrections'] = {'text': '音声で確認した修正文です。', 'speakers': ['B'], 'start': 101, 'end': 111}
        result, ready = apply_review(document, review)
        self.assertEqual(document, before)
        self.assertEqual(result['units'], document['units'])
        self.assertEqual(result['comparison'], document['comparison'])
        self.assertEqual(result['segments'][0]['model_segment']['text'], document['segments'][0]['text'])
        self.assertEqual(result['segments'][0]['speakers'], ['B'])
        self.assertEqual(result['segments'][0]['time_kind'], 'human_corrected_source_range')
        self.assertFalse(result['validation']['asr_text_preserved'])
        self.assertTrue(result['validation']['model_raw_outputs_preserved'])

    def test_fail_pending_or_na_content_never_becomes_ready(self):
        for field, status in [('content', 'fail'), ('boundary', 'review_required'),
                              ('content', 'not_applicable'), ('speaker', 'fail')]:
            document = source()
            review = review_for(document)
            review['rows']['0'].update({field: status, 'note': '合格条件を確認できないテストケース。'})
            self.assertEqual(apply_review(document, review)[1]['segments'], [])

    def test_invalid_status_and_missing_human_provenance_rejected(self):
        for key, value in [('reviewer', ''), ('reviewed_at', '2026-09-29'),
                           ('reviewed_at', 'bad'), ('listened_ranges', []), ('schema_version', 2)]:
            document = source()
            review = review_for(document)
            review[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                apply_review(document, review)
        document = source()
        review = review_for(document)
        review['rows']['0']['content'] = ['pass']
        with self.assertRaises(ValueError):
            apply_review(document, review)

    def test_unreviewed_rows_may_be_omitted_but_are_retained(self):
        document = source()
        review = review_for(document)
        del review['rows']['1']
        result, ready = apply_review(document, review)
        self.assertEqual(len(result['segments']), 2)
        self.assertFalse(result['segments'][1]['ready_for_anonymous_summary'])
        self.assertEqual(len(ready['segments']), 1)

    def test_multiple_speakers_cannot_become_single_person_attribution(self):
        document = source()
        review = review_for(document)
        review['rows']['0']['corrections'] = {'speakers': ['A', 'B']}
        result, ready = apply_review(document, review)
        self.assertTrue(ready['segments'][0]['ready_for_anonymous_summary'])
        self.assertFalse(ready['segments'][0]['ready_for_attributed_summary'])

    def test_cli_uses_new_directory_and_never_writes_source_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            document = source()
            original, review = root/'original.json', root/'review.json'
            original.write_text(json.dumps(document))
            review.write_text(json.dumps(review_for(document)))
            hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in (original, review)]
            command = [sys.executable, str(Path(__file__).with_name('apply_scene_review.py')), str(original), str(review), str(root/'output')]
            subprocess.run(command, check=True, capture_output=True)
            self.assertEqual(hashes, [hashlib.sha256(path.read_bytes()).hexdigest() for path in (original, review)])
            ready = json.loads((root/'output/ready-segments.json').read_text())
            self.assertEqual(len(ready['segments']), 1)
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)


if __name__ == '__main__':
    unittest.main()
