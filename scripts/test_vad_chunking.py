import copy
import contextlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest

from asr_chunking import partition_for_asr
from vad_chunking import main, partition_at_vad_pauses


def pcm(count, value=.25):
    return struct.pack('<f', value) * count


def vad(count, rate=100, width=10, lows=()):
    frames = []
    for a in range(0, count, width):
        b = min(a + width, count)
        frames.append(dict(start_sample=a, end_sample=b,
                           probability=.1 if any(x <= a and b <= y for x, y in lows) else .9))
    return dict(sample_count=count, sample_rate=rate, frames=frames)


class VadChunkingTests(unittest.TestCase):
    def assert_coverage(self, data, result):
        parts = result['parts']
        self.assertEqual(parts[0]['start_sample'], 0)
        self.assertEqual(parts[-1]['end_sample'], len(data) // 4)
        self.assertEqual(b''.join(data[p['start_sample'] * 4:p['end_sample'] * 4] for p in parts), data)
        for i, p in enumerate(parts):
            self.assertEqual(p['index'], i)
            self.assertGreater(p['samples'], 0)
            self.assertLessEqual(p['samples'], result['policy']['maximum_samples'])
            if i:
                self.assertEqual(parts[i - 1]['end_sample'], p['start_sample'])
        json.dumps(result, allow_nan=False)

    def test_pause_23_seconds_selected_outside_old_25_to_30_search(self):
        data = pcm(4000)
        record = vad(4000, lows=[(2290, 2310)])
        original = copy.deepcopy(record)
        result = partition_at_vad_pauses(data, record, 'cohere', rate=100)
        self.assertEqual(result['parts'][0]['end_sample'], 2300)
        self.assertEqual(result['parts'][0]['boundary_kind'], 'vad_pause_preferred')
        self.assertEqual(result['policy']['maximum_seconds'], 30)
        self.assertEqual(result['policy']['pause_candidates'][0]['frame_indices'], [229, 230])
        self.assertEqual(result['policy']['pause_candidates'][0]['frame_probabilities'], [.1, .1])
        self.assertEqual(record, original)
        self.assert_coverage(data, result)

    def test_latest_pause_is_chosen(self):
        result = partition_at_vad_pauses(pcm(4000), vad(4000, lows=[(1990, 2010), (2390, 2410)]), 'cohere', rate=100)
        self.assertEqual(result['parts'][0]['end_sample'], 2400)

    def test_earlier_pause_used_only_when_preferred_span_empty(self):
        data = pcm(4000)
        result = partition_at_vad_pauses(data, vad(4000, lows=[(590, 610)]), 'cohere', rate=100)
        self.assertEqual(result['parts'][0]['end_sample'], 600)
        self.assertEqual(result['parts'][0]['boundary_kind'], 'vad_pause_early')
        self.assertEqual(result['policy']['early_pause_count'], 1)
        self.assert_coverage(data, result)
        result = partition_at_vad_pauses(data, vad(4000, lows=[(590, 610), (2390, 2410)]), 'cohere', rate=100)
        self.assertEqual(result['parts'][0]['end_sample'], 2400)
        self.assertEqual(result['policy']['early_pause_count'], 0)

    def test_continuous_voice_fallback_visible_and_matches_existing_algorithm(self):
        data = pcm(8000)
        for engine in ('qwen', 'cohere'):
            result = partition_at_vad_pauses(data, vad(8000), engine, rate=100, max_seconds=30)
            old = partition_for_asr(data, engine, rate=100, max_seconds=30)
            self.assertEqual([p['end_sample'] for p in result['parts']], [p['end_sample'] for p in old['parts']])
            self.assertEqual(result['policy']['fallback_count'], len(result['parts']) - 1)
            for p in result['parts'][:-1]:
                self.assertEqual(p['boundary_kind'], 'forced_low_energy_no_vad_pause')
                self.assertTrue(p['unresolved'])
                self.assertFalse(p['verified_speech_end'])
            self.assert_coverage(data, result)

    def test_fallback_sample_metadata_is_offset_after_prior_pause(self):
        data = pcm(6000)
        result = partition_at_vad_pauses(data, vad(6000, lows=[(1990, 2010)]), 'cohere', rate=100)
        second = result['parts'][1]
        self.assertEqual(second['start_sample'], 2000)
        self.assertEqual(second['end_sample'], 4500)
        self.assertEqual(second['quiet_window_start_sample'], 4500)
        self.assertEqual(second['search_start_sample'], 4500)
        self.assertEqual(second['search_end_sample'], 5000)
        self.assertEqual(second['target_sample'], 5000)
        self.assert_coverage(data, result)

    def test_edge_low_runs_short_pauses_and_too_early_pauses_not_selected(self):
        record = vad(4000, lows=[(0, 300), (300, 310), (390, 410), (2290, 2300), (3800, 4000)])
        result = partition_at_vad_pauses(pcm(4000), record, 'cohere', rate=100)
        self.assertEqual(result['parts'][0]['boundary_kind'], 'forced_low_energy_no_vad_pause')
        self.assertEqual(len(result['policy']['excluded_edge_low_runs']), 2)
        self.assertEqual(len(result['policy']['pause_candidates']), 1)

    def test_equal_threshold_is_not_a_low_frame(self):
        record = vad(4000, lows=[(2290, 2310)])
        for frame in record['frames'][229:231]:
            frame['probability'] = .35
        result = partition_at_vad_pauses(pcm(4000), record, 'cohere', rate=100)
        self.assertEqual(result['policy']['pause_candidates'], [])

    def test_default_qwen_fitting_remainder_and_partial_frame_kept_whole(self):
        data = pcm(1003)
        for engine, maximum in [('qwen', 180), ('cohere', 30)]:
            result = partition_at_vad_pauses(data, vad(1003, lows=[(590, 610)]), engine, rate=100)
            self.assertEqual(result['policy']['default_maximum_seconds'], maximum)
            self.assertEqual(len(result['parts']), 1)
            self.assertEqual(result['parts'][0]['boundary_kind'], 'input_end')
            self.assert_coverage(data, result)

    def test_short_tail_and_real_silero_frame_sizes(self):
        data = pcm(3021)
        result = partition_at_vad_pauses(data, vad(3021, lows=[(2990, 3010)]), 'cohere', rate=100)
        self.assertEqual(result['parts'][0]['end_sample'], 3000)
        self.assertEqual(result['parts'][-1]['samples'], 21)
        self.assert_coverage(data, result)
        for width in (512, 4096):
            count = width * 3 + 13
            data = pcm(count)
            result = partition_at_vad_pauses(data, vad(count, rate=16000, width=width), 'cohere')
            self.assert_coverage(data, result)

    def test_incomplete_overlapping_gapped_or_misaligned_frames_rejected(self):
        good = vad(1003)
        bads = []
        for key, value in [('sample_count', 1000), ('sample_rate', 16000), ('sample_count', True)]:
            record = copy.deepcopy(good);record[key] = value;bads.append(record)
        record = copy.deepcopy(good);record['frames'].pop();bads.append(record)
        record = copy.deepcopy(good);record['frames'][1]['start_sample'] += 1;bads.append(record)
        record = copy.deepcopy(good);record['frames'][1]['start_sample'] -= 1;bads.append(record)
        record = copy.deepcopy(good);record['frames'][1]['end_sample'] += 1;bads.append(record)
        record = copy.deepcopy(good);record['frames'][0]['start_sample'] = False;bads.append(record)
        record = copy.deepcopy(good);record['frames'] = [];bads.append(record)
        for record in bads:
            with self.subTest(record=record), self.assertRaises(ValueError):
                partition_at_vad_pauses(pcm(1003), record, 'cohere', rate=100)

    def test_nonfinite_pcm_and_malformed_probability_rejected(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            data = pcm(1002) + pcm(1, value)
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                partition_at_vad_pauses(data, vad(1003), 'cohere', rate=100)
        for score in (float('nan'), float('inf'), -.1, 1.1, True, '0.1', None):
            record = vad(1003);record['frames'][-1]['probability'] = score
            with self.subTest(score=score), self.assertRaises(ValueError):
                partition_at_vad_pauses(pcm(1003), record, 'cohere', rate=100)

    def test_invalid_pcm_rate_engine_and_maximum_rejected(self):
        for data in (b'', b'123', [0.]):
            with self.assertRaises(ValueError):
                partition_at_vad_pauses(data, vad(1), 'cohere', rate=100)
        for rate in (0, True, 100., float('nan')):
            with self.assertRaises(ValueError):
                partition_at_vad_pauses(pcm(1), vad(1), 'cohere', rate=rate)
        for engine in ('apple', None):
            with self.assertRaises(ValueError):
                partition_at_vad_pauses(pcm(1), vad(1), engine, rate=100)
        for maximum in (True, 5, float('nan'), float('inf'), 1e308):
            with self.assertRaises(ValueError):
                partition_at_vad_pauses(pcm(1), vad(1), 'cohere', rate=100, max_seconds=maximum)

    def test_cli_verifies_identity_and_keeps_all_pcm_samples(self):
        from scene_pipeline import fingerprint, read_float_wav, write_float_wav
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio, record_path, output = root / 'source.wav', root / 'vad.json', root / 'new'
            data = pcm(640123)
            write_float_wav(audio, data)
            record = vad(len(data) // 4, rate=16000, width=512, lows=[(366080, 369152)])
            record.update(file=str(audio.resolve()), sha256=fingerprint(audio)['sha256'])
            record_path.write_text(json.dumps(record))
            args = [str(audio), str(record_path), str(output), '--engine', 'cohere']
            with contextlib.redirect_stdout(io.StringIO()):
                plan = main(args)
            self.assertTrue(plan['validation']['source_audio_identity_verified'])
            files = (output / 'inputs.txt').read_text().splitlines()
            self.assertEqual(b''.join(read_float_wav(Path(p))[1] for p in files), data)
            self.assertEqual(len(files), len(plan['parts']))
            self.assertEqual(plan, json.loads((output / 'plan.json').read_text()))
            with self.assertRaises(FileExistsError):
                main(args)
            for change in [{'sha256': '0' * 64}, {'file': str(root / 'different.wav')}]:
                bad = dict(record, **change)
                record_path.write_text(json.dumps(bad))
                bad_output = root / 'must-not-create'
                with self.assertRaisesRegex(ValueError, 'SHA-256'):
                    main([str(audio), str(record_path), str(bad_output), '--engine', 'cohere'])
                self.assertFalse(bad_output.exists())


if __name__ == '__main__':
    unittest.main()
