"""Structural partition tests; no model, transcript or acoustic ground truth."""
import json
import struct
import unittest
from asr_chunking import partition_for_asr

def pcm(values):
    return struct.pack('<' + 'f' * len(values), *values)

class AsrChunkingTests(unittest.TestCase):

    def assert_coverage(self, data, result):
        parts = result['parts']
        self.assertEqual(parts[0]['start_sample'], 0)
        self.assertEqual(parts[-1]['end_sample'], len(data) // 4)
        self.assertEqual(b''.join((data[p['start_sample'] * 4:p['end_sample'] * 4] for p in parts)), data)
        self.assertEqual(sum((p['samples'] for p in parts)), len(data) // 4)
        for index, part in enumerate(parts):
            self.assertEqual(part['index'], index)
            self.assertGreater(part['samples'], 0)
            self.assertLessEqual(part['samples'], result['policy']['maximum_samples'])
            if index:
                self.assertEqual(parts[index - 1]['end_sample'], part['start_sample'])

    def test_defaults_and_metadata_distinguish_sources_and_adaptations(self):
        result = partition_for_asr(pcm([0.25] * 2000), rate=10)
        self.assert_coverage(pcm([0.25] * 2000), result)
        self.assertEqual(result['policy']['maximum_seconds'], 180)
        self.assertEqual(result['policy']['target_seconds'], 175)
        self.assertIn('github.com', result['policy']['source']['url'])
        self.assertTrue(result['policy']['local_adaptations'])
        self.assertFalse(result['policy']['verified_speech_end'])
        self.assertFalse(result['policy']['confidence_provided'])
        json.dumps(result, allow_nan=False)

    def test_qwen_uses_quiet_window_then_quiet_sample_not_window_start(self):
        values = [1.0] * 400
        values[240:250] = [0.1] * 10
        values[245] = -0.001
        result = partition_for_asr(pcm(values), rate=100, max_seconds=12)
        self.assertEqual(len(result['parts']), 1)
        values = [1.0] * 1400
        values[840:850] = [0.1] * 10
        values[845] = -0.001
        result = partition_for_asr(pcm(values), rate=100, max_seconds=12)
        first = result['parts'][0]
        self.assertEqual(first['quiet_window_start_sample'], 840)
        self.assertEqual(first['end_sample'], 845)
        self.assertEqual((first['search_start_sample'], first['search_end_sample']), (200, 1200))
        self.assert_coverage(pcm(values), result)

    def test_maximum_override_keeps_all_samples_and_is_recorded(self):
        data = pcm([0.125, -0.25, 0.0, 0.5] * 1001)
        result = partition_for_asr(data, rate=100, max_seconds=30)
        self.assertEqual(result['policy']['maximum_override_seconds'], 30)
        self.assertEqual(result['policy']['maximum_samples'], 3000)
        self.assert_coverage(data, result)

    def test_fractional_limit_rounds_down_and_flat_search_is_deterministic(self):
        data = pcm([0.25] * 701)
        result = partition_for_asr(data, rate=10, max_seconds=30.09)
        self.assertEqual(result['policy']['maximum_samples'], 300)
        self.assertEqual(result['policy']['maximum_seconds'], 30.0)
        self.assert_coverage(data, result)
        self.assertEqual(result['parts'][0]['end_sample'], 200)

    def test_exact_limit_and_subsecond_input_are_not_padded_or_cut(self):
        for count in (1, 2, 180 * 10):
            with self.subTest(count=count):
                data = pcm([0.125] * count)
                result = partition_for_asr(data, rate=10)
                self.assertEqual(len(result['parts']), 1)
                self.assertEqual(result['parts'][0]['end_sample'], count)
                self.assert_coverage(data, result)

    def test_qwen_short_tail_stays_short_and_unpadded(self):
        values = [1.0] * 18001
        values[17990:18000] = [0.1] * 10
        values[17999] = 0.0
        data = pcm(values)
        result = partition_for_asr(data, rate=100)
        self.assertEqual(result['parts'][-1]['samples'], 2)
        self.assert_coverage(data, result)

    def test_repeated_audio_is_retained_and_no_text_is_processed(self):
        data = pcm([0.0, 0.25, -0.5, 0.125] * 500)
        result = partition_for_asr(data, rate=10, max_seconds=30)
        self.assert_coverage(data, result)
        self.assertEqual(result['policy']['text_processing'], 'none')
        self.assertTrue(all(('text' not in p for p in result['parts'])))
        self.assertEqual(result, partition_for_asr(data, rate=10, max_seconds=30))

    def test_nonfinite_anywhere_is_rejected_even_without_a_search(self):
        for bad in (float('nan'), float('inf'), -float('inf')):
            for index in (0, 1, 49):
                values = [0.2] * 50
                values[index] = bad
                with self.subTest(bad=bad, index=index), self.assertRaisesRegex(ValueError, 'nonfinite'):
                    partition_for_asr(pcm(values), rate=10)

    def test_invalid_inputs(self):
        for invalid in (b'', b'abc', [0.0], 'audio'):
            with self.subTest(pcm=invalid), self.assertRaises(ValueError):
                partition_for_asr(invalid)
        for rate in (0, -1, True, 16000.0, float('nan')):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                partition_for_asr(pcm([0.0]), rate=rate)
        for maximum in (0, -1, True, '30', float('nan'), float('inf'), 5, 1e+308):
            with self.subTest(maximum=maximum), self.assertRaises(ValueError):
                partition_for_asr(pcm([0.0]), max_seconds=maximum)
if __name__ == '__main__':
    unittest.main()
