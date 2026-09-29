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
        self.assertEqual(b''.join(data[p['start_sample'] * 4:p['end_sample'] * 4] for p in parts), data)
        self.assertEqual(sum(p['samples'] for p in parts), len(data) // 4)
        for index, part in enumerate(parts):
            self.assertEqual(part['index'], index)
            self.assertGreater(part['samples'], 0)
            self.assertLessEqual(part['samples'], result['policy']['maximum_samples'])
            if index:
                self.assertEqual(parts[index - 1]['end_sample'], part['start_sample'])

    def test_defaults_and_metadata_distinguish_sources_and_adaptations(self):
        for engine, maximum, target in [('qwen', 180, 175), ('cohere', 35, 35)]:
            with self.subTest(engine=engine):
                result = partition_for_asr(pcm([.25] * 2000), engine, rate=10)
                self.assert_coverage(pcm([.25] * 2000), result)
                self.assertEqual(result['policy']['maximum_seconds'], maximum)
                self.assertEqual(result['policy']['target_seconds'], target)
                self.assertIn('github.com', result['policy']['source']['url'])
                self.assertTrue(result['policy']['local_adaptations'])
                self.assertFalse(result['policy']['verified_speech_end'])
                self.assertFalse(result['policy']['confidence_provided'])
                json.dumps(result, allow_nan=False)

    def test_qwen_uses_quiet_window_then_quiet_sample_not_window_start(self):
        values = [1.] * 400
        values[240:250] = [.1] * 10
        values[245] = -.001
        result = partition_for_asr(pcm(values), 'qwen', rate=100, max_seconds=12)
        # Input <= max must not be split even when it contains a quiet point.
        self.assertEqual(len(result['parts']), 1)
        values = [1.] * 1400
        values[840:850] = [.1] * 10
        values[845] = -.001
        result = partition_for_asr(pcm(values), 'qwen', rate=100, max_seconds=12)
        first = result['parts'][0]
        self.assertEqual(first['quiet_window_start_sample'], 840)
        self.assertEqual(first['end_sample'], 845)
        self.assertEqual((first['search_start_sample'], first['search_end_sample']), (200, 1200))
        self.assert_coverage(pcm(values), result)

    def test_cohere_uses_rms_grid_and_cuts_at_window_start(self):
        values = [1.] * 3700
        values[3210:3220] = [.01] * 10
        values[3214] = 0.
        result = partition_for_asr(pcm(values), 'cohere', rate=100)
        self.assertEqual(result['parts'][0]['end_sample'], 3210)
        self.assertEqual(result['parts'][0]['quiet_window_start_sample'], 3210)
        self.assert_coverage(pcm(values), result)

    def test_cohere_keeps_upstream_exclusive_grid_endpoint(self):
        values = [1.] * 3700
        values[3490:3500] = [0.] * 10
        result = partition_for_asr(pcm(values), 'cohere', rate=100)
        self.assertEqual(result['parts'][0]['end_sample'], 3000)

    def test_maximum_override_keeps_all_samples_and_is_recorded(self):
        data = pcm([.125, -.25, 0., .5] * 1001)
        for engine in ('qwen', 'cohere'):
            with self.subTest(engine=engine):
                result = partition_for_asr(data, engine, rate=100, max_seconds=30)
                self.assertEqual(result['policy']['maximum_override_seconds'], 30)
                self.assertEqual(result['policy']['maximum_samples'], 3000)
                self.assert_coverage(data, result)

    def test_fractional_limit_rounds_down_and_flat_search_is_deterministic(self):
        data = pcm([.25] * 701)
        for engine in ('qwen', 'cohere'):
            result = partition_for_asr(data, engine, rate=10, max_seconds=30.09)
            self.assertEqual(result['policy']['maximum_samples'], 300)
            self.assertEqual(result['policy']['maximum_seconds'], 30.)
            self.assert_coverage(data, result)
            self.assertEqual(result['parts'][0]['end_sample'], 200 if engine == 'qwen' else 250)

    def test_exact_limit_and_subsecond_input_are_not_padded_or_cut(self):
        for engine, maximum in [('qwen', 180), ('cohere', 35)]:
            for count in (1, 2, maximum * 10):
                with self.subTest(engine=engine, count=count):
                    data = pcm([.125] * count)
                    result = partition_for_asr(data, engine, rate=10)
                    self.assertEqual(len(result['parts']), 1)
                    self.assertEqual(result['parts'][0]['end_sample'], count)
                    self.assert_coverage(data, result)

    def test_qwen_short_tail_stays_short_and_unpadded(self):
        # A unique final-sample minimum selects 179.99 s, leaving 0.02 s.
        values = [1.] * 18001
        values[17990:18000] = [.1] * 10
        values[17999] = 0.
        data = pcm(values)
        result = partition_for_asr(data, 'qwen', rate=100)
        self.assertEqual(result['parts'][-1]['samples'], 2)
        self.assert_coverage(data, result)

    def test_repeated_audio_is_retained_and_no_text_is_processed(self):
        data = pcm([0., .25, -.5, .125] * 500)
        for engine in ('qwen', 'cohere'):
            result = partition_for_asr(data, engine, rate=10, max_seconds=30)
            self.assert_coverage(data, result)
            self.assertEqual(result['policy']['text_processing'], 'none')
            self.assertTrue(all('text' not in p for p in result['parts']))
            self.assertEqual(result, partition_for_asr(data, engine, rate=10, max_seconds=30))

    def test_nonfinite_anywhere_is_rejected_even_without_a_search(self):
        for engine in ('qwen', 'cohere'):
            for bad in (float('nan'), float('inf'), -float('inf')):
                for index in (0, 1, 49):
                    values = [.2] * 50
                    values[index] = bad
                    with self.subTest(engine=engine, bad=bad, index=index), self.assertRaisesRegex(ValueError, 'nonfinite'):
                        partition_for_asr(pcm(values), engine, rate=10)

    def test_invalid_inputs(self):
        for invalid in (b'', b'abc', [0.], 'audio'):
            with self.subTest(pcm=invalid), self.assertRaises(ValueError):
                partition_for_asr(invalid, 'qwen')
        for engine in ('apple', '', None):
            with self.subTest(engine=engine), self.assertRaises(ValueError):
                partition_for_asr(pcm([0.]), engine)
        for rate in (0, -1, True, 16000., float('nan')):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                partition_for_asr(pcm([0.]), 'qwen', rate=rate)
        for engine in ('qwen', 'cohere'):
            for maximum in (0, -1, True, '30', float('nan'), float('inf'), 5, 1e308):
                with self.subTest(engine=engine, maximum=maximum), self.assertRaises(ValueError):
                    partition_for_asr(pcm([0.]), engine, max_seconds=maximum)


if __name__ == '__main__':
    unittest.main()
