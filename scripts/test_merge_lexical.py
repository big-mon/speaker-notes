import copy
import pathlib
import tempfile
import unittest

from merge import export, merge


def inputs(texts, times, pieces, spans, extent=None):
    asr = [{'start': extent[0] if extent else times[0][0],
            'end': extent[1] if extent else times[-1][1], 'text': ''.join(texts),
            'units': [dict(text=t, start=a, end=b) for t, (a, b) in zip(texts, times)]}]
    lexical = [[{'text': t} for t in pieces]]
    diar = {'segments': [dict(start=a, end=b, speaker=s) for a, b, s in spans]}
    return asr, lexical, diar


class LexicalMergeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.input = pathlib.Path(self.directory.name) / 'audio.fixture'
        self.input.write_bytes(b'no audio inference in structural test')

    def run_merge(self, asr, lexical, diar):
        return merge(asr, diar, self.input, {}, lexical=lexical)

    def test_word_assignment_changes_without_rewriting_native_audit(self):
        asr, lexical, diar = inputs(['プッシ', 'ュ'], [(0, .4), (.4, .5)], ['プッシュ'],
                                    [(0, .42, 'S1'), (.42, .5, 'S2')])
        before = copy.deepcopy((asr, lexical, diar))
        native = merge(asr, diar, self.input, {})
        result = self.run_merge(asr, lexical, diar)
        self.assertEqual(result['units'], native['units'])
        self.assertEqual(result['diarization'], native['diarization'])
        self.assertEqual(result['speaker_mapping'], native['speaker_mapping'])
        self.assertEqual((asr, lexical, diar), before)
        self.assertEqual([r['text'] for r in result['segments']], ['プッシュ'])
        self.assertEqual(result['segments'][0]['speakers'], ['A'])
        self.assertEqual(result['lexical_buckets']['buckets'][0]['speakers'], ['S1'])
        self.assertEqual(result['lexical_tokens'], lexical)
        self.assertEqual(result['segments'][0]['bucket_ids'], [0])
        self.assertEqual((result['segments'][0]['unit_start'], result['segments'][0]['unit_end']), (0, 2))
        self.assertFalse(result['merge']['collar_enabled'])
        self.assertFalse(result['merge']['short_response_absorption'])

    def test_reading_groups_keep_short_other_speaker_response_separate(self):
        asr, lexical, diar = inputs(['説明', 'です', 'はい', '続き', 'です'],
                                    [(0, .9), (.9, 1), (1, 1.08), (1.08, 1.9), (1.9, 2)],
                                    ['説明', 'です', 'はい', '続き', 'です'],
                                    [(0, 1, 'S1'), (1, 1.08, 'S2'), (1.08, 2, 'S1')])
        rows = self.run_merge(asr, lexical, diar)['segments']
        self.assertEqual([r['text'] for r in rows], ['説明です', 'はい', '続きです'])
        self.assertEqual([r['speakers'] for r in rows], [['A'], ['B'], ['A']])
        self.assertEqual([r['bucket_ids'] for r in rows], [[0, 1], [2], [3, 4]])

    def test_brief_overlap_does_not_disappear_into_single_speaker_row(self):
        asr, lexical, diar = inputs(['前', '語', '後'], [(0, 1), (1, 2), (2, 3)],
                                    ['前', '語', '後'], [(0, 3, 'S1'), (1.45, 1.46, 'S2')])
        rows = self.run_merge(asr, lexical, diar)['segments']
        self.assertEqual([r['state'] for r in rows], ['single', 'overlap', 'single'])
        self.assertEqual(rows[1]['speakers'], ['A', 'B'])

    def test_unknown_timing_uses_parent_playback_only_and_exports(self):
        asr, lexical, diar = inputs(['前', '不明', '後'], [(0, 1), (None, None), (2, 3)],
                                    ['前', '不明', '後'], [(0, 3, 'S1')], extent=(0, 3))
        result = self.run_merge(asr, lexical, diar)
        rows = result['segments']
        self.assertEqual([r['text'] for r in rows], ['前', '不明', '後'])
        self.assertEqual(rows[1]['speakers'], [])
        self.assertEqual(rows[1]['state'], 'unknown')
        self.assertEqual((rows[1]['start'], rows[1]['end']), (0, 3))
        self.assertEqual(rows[1]['time_kind'], 'parent_asr_playback_envelope')
        self.assertIsNone(result['lexical_buckets']['buckets'][1]['start'])
        output = pathlib.Path(self.directory.name) / 'export'
        export(result, output)
        self.assertTrue((output / 'transcript.json').is_file())
        self.assertIn('不明', (output / 'transcript.md').read_text())

    def test_internal_time_gap_is_unknown_not_inherited(self):
        asr, lexical, diar = inputs(['プッシ', 'ュ'], [(0, .3), (.5, .6)], ['プッシュ'],
                                    [(0, 1, 'S1')], extent=(0, .6))
        row = self.run_merge(asr, lexical, diar)['segments'][0]
        self.assertEqual(row['state'], 'unknown')
        self.assertTrue(row['timing_fallback'])
        self.assertIn('native_timing_gap_exceeds_limit', row['reasons'])

    def test_native_multiword_run_keeps_original_time_and_whole_text(self):
        asr, lexical, diar = inputs(['こんにちは世界'], [(0, 1)], ['こんにちは', '世界'], [(0, 1, 'S1')])
        result = self.run_merge(asr, lexical, diar)
        bucket = result['lexical_buckets']['buckets'][0]
        self.assertEqual(bucket['text'], 'こんにちは世界')
        self.assertEqual(bucket['raw_timing'], [{'native_unit_index': 0, 'start': 0, 'end': 1}])
        self.assertEqual(len(result['lexical_buckets']['buckets']), 1)

    def test_invalid_parent_extent_does_not_create_a_fake_word_time(self):
        asr, lexical, diar = inputs(['不明'], [(None, None)], ['不明'], [], extent=(None, None))
        with self.assertRaisesRegex(ValueError, 'parent ASR playback extent'):
            self.run_merge(asr, lexical, diar)

    def test_lexical_text_mismatch_is_rejected(self):
        asr, lexical, diar = inputs(['本文'], [(0, 1)], ['別文'], [(0, 1, 'S1')])
        with self.assertRaises(ValueError):
            self.run_merge(asr, lexical, diar)


if __name__ == '__main__':
    unittest.main()
