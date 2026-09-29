"""Structural checks only; these tests do not measure acoustic accuracy."""
import copy
import unittest
from turn_candidates import build_candidates


def event(start, end, speaker):
    return {'start': start, 'end': end, 'speaker': speaker}


class CandidateTests(unittest.TestCase):
    def test_short_overlapped_speech_is_not_mislabeled_or_discarded(self):
        source = [event(0, 12, 'speaker-17'), event(4, 4.8, 'speaker-9')]
        original = copy.deepcopy(source)
        result = build_candidates(source)
        self.assertEqual(source, original)
        self.assertEqual(len(result['candidates']), 1)
        self.assertEqual(result['candidates'][0]['overlaps'][0]['speaker'], 'speaker-9')
        self.assertEqual(result['unresolved_fully_overlapped_event_ids'], [1])
        self.assertEqual([e['source_event'] for e in result['raw_events']], source)
        self.assertFalse(result['candidates'][0]['text_attribution_verified'])

    def test_short_exposed_answer_prevents_merging_surrounding_speaker(self):
        result = build_candidates([event(0, 4, 'x'), event(4, 4.1, 'y'),
                                   event(4.1, 8, 'x')])
        self.assertEqual([c['speaker'] for c in result['candidates']], ['x', 'y', 'x'])
        self.assertEqual(result['candidates'][1]['kind'], 'brief_exposed_activity')

    def test_transition_keeps_both_extents_and_boundary_uncertainty(self):
        result = build_candidates([event(0, 10, 'x'), event(9, 20, 'y')])
        first, second = result['candidates']
        self.assertEqual((first['end'], second['start']), (10, 9))
        for candidate in (first, second):
            boundary = candidate['uncertain_boundaries'][0]
            self.assertEqual((boundary['start'], boundary['end']), (9, 10))

    def test_same_speaker_silence_is_joined_but_long_gap_is_not(self):
        result = build_candidates([event(0, 3, 'x'), event(3.2, 6, 'x'),
                                   event(7, 10, 'x')])
        self.assertEqual(len(result['candidates']), 2)
        self.assertEqual(result['candidates'][0]['source_event_ids'], [0, 1])
        self.assertAlmostEqual(result['candidates'][0]['single_speaker_support_seconds'], 5.8)

    def test_sustained_overlap_stays_as_two_unresolved_candidates(self):
        result = build_candidates([event(0, 10, 'x'), event(2, 7, 'y')])
        self.assertEqual(len(result['candidates']), 2)
        self.assertTrue(all(c['overlaps'] for c in result['candidates']))

    def test_window_keeps_original_clock_ids_and_source_duration(self):
        result = build_candidates([event(50, 110, 'global-B'),
                                   event(99.5, 100.5, 'global-A')], start=100, end=102)
        self.assertEqual(next(e for e in result['raw_events'] if e['id'] == 0)['source_start'], 50)
        self.assertEqual(result['candidates'][0]['start'], 100)
        self.assertEqual(result['candidates'][0]['end'], 102)
        self.assertEqual(result['candidates'][0]['speaker'], 'global-B')
        self.assertEqual(result['candidates'][0]['kind'], 'sustained_activity')
        self.assertTrue(result['candidates'][0]['clipped_by_window'])
        self.assertEqual(len(result['raw_events']), 2)

    def test_union_not_double_counted_for_three_speakers(self):
        result = build_candidates([event(0, 10, 'x'), event(2, 7, 'y'), event(4, 8, 'z')])
        first = result['candidates'][0]
        self.assertEqual(first['single_speaker_support_seconds'], 4)
        self.assertEqual(len(first['overlaps']), 2)

    def test_invalid_times_rejected_and_empty_window_supported(self):
        with self.assertRaises(ValueError):
            build_candidates([event(float('nan'), 2, 'x')])
        with self.assertRaises(ValueError):
            build_candidates([event(3, 2, 'x')])
        self.assertEqual(build_candidates([event(0, 2, 'x')], start=5, end=6)['candidates'], [])


if __name__ == '__main__':
    unittest.main()
