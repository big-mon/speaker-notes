import copy
import math
import unittest

from lexical_speaker_buckets import build_buckets


def fixture(texts, times, pieces):
    return ([{'text': ''.join(texts), 'units': [dict(text=t, start=a, end=b)
                                             for t, (a, b) in zip(texts, times)]}],
            [[{'text': text} for text in pieces]])


def diar(*spans):
    return {'segments': [dict(start=a, end=b, speaker=s) for a, b, s in spans]}


class LexicalBucketTests(unittest.TestCase):
    def test_word_is_united_before_assignment_and_inputs_unchanged(self):
        a, l = fixture(['プッシ', 'ュ'], [(0, .4), (.4, .5)], ['プッシュ'])
        d = diar((0, .42, 'A'), (.42, .5, 'B'))
        before = copy.deepcopy((a, l, d))
        result = build_buckets(a, l, d)
        self.assertEqual(len(result['buckets']), 1)
        b = result['buckets'][0]
        self.assertEqual(b['text'], 'プッシュ')
        self.assertEqual(b['native_unit_indices'], [0, 1])
        self.assertEqual(b['character_span'], [0, 4])
        self.assertEqual(b['speakers'], ['A'])
        self.assertEqual((a, l, d), before)

    def test_native_unit_spanning_multiple_lexical_words_is_never_split(self):
        a, l = fixture(['こんにちは世界', '。'], [(0, 1), (1, 1.1)], ['こんにちは', '世界', '。'])
        result = build_buckets(a, l, diar((0, 2, 'A')))
        self.assertEqual([b['text'] for b in result['buckets']], ['こんにちは世界', '。'])
        self.assertEqual(result['buckets'][0]['raw_timing'], [{'native_unit_index': 0, 'start': 0, 'end': 1}])

    def test_unicode_offsets_derive_from_text_not_swift_grapheme_offsets(self):
        a, l = fixture(['か\u3099', '👩\u200d💻', '。'], [(0, .2), (.2, .4), (.4, .5)], ['か\u3099', '👩\u200d💻', '。'])
        for p in l[0]:
            p.update(start=0, end=1)  # Deliberately unusable Swift offsets.
        result = build_buckets(a, l, diar((0, 1, 'A')))
        self.assertEqual([b['character_span'] for b in result['buckets']], [[0, 2], [2, 5], [5, 6]])
        self.assertEqual(result['original_text'], 'か\u3099👩\u200d💻。')

    def test_short_other_speaker_acknowledgment_is_not_absorbed(self):
        a, l = fixture(['説明', 'はい', '続き'], [(0, 1), (1, 1.08), (1.08, 2)], ['説明', 'はい', '続き'])
        result = build_buckets(a, l, diar((0, 1, 'A'), (1, 1.08, 'B'), (1.08, 2, 'A')))
        self.assertEqual([b['speakers'] for b in result['buckets']], [['A'], ['B'], ['A']])
        self.assertEqual([b['text'] for b in result['buckets']], ['説明', 'はい', '続き'])

    def test_brief_overlap_is_explicit_even_below_coverage_cutoff(self):
        a, l = fixture(['発言'], [(0, 1)], ['発言'])
        b = build_buckets(a, l, diar((0, 1, 'A'), (.45, .46, 'B')))['buckets'][0]
        self.assertEqual(b['state'], 'overlap')
        self.assertEqual(b['speakers'], ['A', 'B'])
        self.assertAlmostEqual(b['overlap_seconds'], .01)
        self.assertAlmostEqual(b['speaker_scores']['B']['fraction'], .01)

    def test_duplicate_native_and_same_speaker_intervals_do_not_double_count(self):
        a, l = fixture(['同', '期'], [(0, 1), (0, 1)], ['同期'])
        b = build_buckets(a, l, diar((0, 1, 'A'), (0, .8, 'A')))['buckets'][0]
        self.assertEqual(b['native_time_seconds'], 1)
        self.assertEqual(b['speaker_scores']['A'], {'seconds': 1, 'fraction': 1})
        self.assertEqual(b['state'], 'single')
        self.assertEqual(b['overlap_seconds'], 0)

    def test_gap_unknown_keeps_all_text_and_overlap_evidence(self):
        a, l = fixture(['プッシ', 'ュ'], [(0, .3), (.5, .6)], ['プッシュ'])
        b = build_buckets(a, l, diar((0, 1, 'A'), (.51, .52, 'B')))['buckets'][0]
        self.assertEqual(b['state'], 'unknown')
        self.assertEqual(b['speakers'], [])
        self.assertEqual(b['overlap_speakers'], ['A', 'B'])
        self.assertEqual(b['text'], 'プッシュ')
        self.assertIn('native_timing_gap_exceeds_limit', b['reasons'])

    def test_invalid_missing_point_and_nonmonotonic_times_stay_unknown(self):
        for times in [[(None, 1)], [(0, math.nan)], [(0, 0)], [(1, 0)], [(True, 1)], [(-1, 1)]]:
            with self.subTest(times=times):
                a, l = fixture(['語'], times, ['語'])
                self.assertEqual(build_buckets(a, l, diar((0, 2, 'A')))['buckets'][0]['state'], 'unknown')
        a, l = fixture(['発', '言'], [(1, 2), (0, 1)], ['発言'])
        self.assertIn('nonmonotonic_native_timing', build_buckets(a, l, diar((0, 2, 'A')))['buckets'][0]['reasons'])

    def test_insufficient_coverage_does_not_inherit_neighbor(self):
        a, l = fixture(['はい'], [(0, 1)], ['はい'])
        b = build_buckets(a, l, diar((0, .5, 'A'), (.5, 1, 'B')))['buckets'][0]
        self.assertEqual(b['state'], 'unknown')
        self.assertEqual(b['overlap_seconds'], 0)

    def test_collar_defers_ambiguous_overlap_without_discarding_it(self):
        a, l = fixture(['語'], [(0, 1)], ['語'])
        d = diar((0, .7, 'A'), (.1, .7, 'B'))
        before = build_buckets(a, l, d)['buckets'][0]
        after = build_buckets(a, l, d, collar=True)['buckets'][0]
        self.assertEqual(before['state'], 'overlap')
        self.assertEqual(after['state'], 'unknown')
        self.assertEqual(after['overlap_speakers'], ['A', 'B'])
        self.assertIn('ambiguous_coverage_near_raw_speaker_change', after['reasons'])

    def test_collar_requires_both_ambiguity_and_nearby_change(self):
        a, l = fixture(['語'], [(5, 6)], ['語'])
        b = build_buckets(a, l, diar((0, 10, 'A'), (0, 10, 'B')), collar=True)['buckets'][0]
        self.assertEqual(b['state'], 'overlap')
        self.assertEqual(b['nearby_raw_speaker_change_times'], [])
        a, l = fixture(['語'], [(0, 1)], ['語'])
        b = build_buckets(a, l, diar((0, 1, 'A'), (.45, .46, 'B')), collar=True)['buckets'][0]
        self.assertEqual(b['state'], 'overlap')

    def test_global_offsets_and_unit_references_across_results(self):
        a, l = fixture(['最初'], [(0, 1)], ['最初'])
        a2, l2 = fixture(['次'], [(1, 2)], ['次'])
        result = build_buckets(a + a2, l + l2, diar((0, 2, 'A')))
        self.assertEqual(result['original_text'], '最初次')
        self.assertEqual(result['buckets'][1]['character_span'], [2, 3])
        self.assertEqual(result['buckets'][1]['native_unit_indices'], [1])
        self.assertEqual(result['buckets'][1]['source_unit_references'], [[1, 0]])

    def test_text_mismatches_and_bad_diarization_rejected(self):
        a, l = fixture(['本文'], [(0, 1)], ['本文'])
        for bad in [[], [[{'text': '本'}]], [[{'text': '本文余分'}]]]:
            with self.assertRaises(ValueError):
                build_buckets(a, bad, diar((0, 1, 'A')))
        a[0]['text'] = '違う本文'
        with self.assertRaises(ValueError):
            build_buckets(a, l, diar((0, 1, 'A')))
        a, l = fixture(['語'], [(0, 1)], ['語'])
        with self.assertRaises(ValueError):
            build_buckets(a, l, diar((1, 0, 'A')))


if __name__ == '__main__':
    unittest.main()
