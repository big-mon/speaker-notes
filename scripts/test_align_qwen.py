import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from align_qwen import annotate_timestamps, capture_parser, json_safe, map_tokens, run


class MappingTests(unittest.TestCase):
    def test_exact_repeated_words_preserve_punctuation_whitespace_and_numbers(self):
        text = ' はい、はい。\r\n比率は-3.5%です。'
        words = ['はい', 'はい', '比率', 'は', '35', 'です']
        mapped = map_tokens(text, words)
        self.assertTrue(mapped['coverage']['all_tokens_mapped'])
        self.assertTrue(mapped['coverage']['original_text_preserved'])
        self.assertEqual(''.join(p['text'] for p in mapped['original_text_partition']), text)
        number = mapped['tokens'][4]
        self.assertEqual(number['original_raw_text'], '3.5')
        self.assertEqual(''.join(text[a:b] for a, b in number['original_character_spans']), '35')
        untimed = ''.join(p['text'] for p in mapped['original_text_partition'] if p['token_index'] is None)
        self.assertIn('-', untimed)
        self.assertIn('.', untimed)
        self.assertIn('%', untimed)
        self.assertIn('\r\n', untimed)

    def test_tokenizer_transformation_not_silently_normalized(self):
        mapped = map_tokens('ＡＩが未来を変える。', ['AI', 'が', '未来', 'を', '変える'])
        self.assertFalse(mapped['tokenizer_matches_filtered_original'])
        self.assertFalse(mapped['tokens'][0]['mapped'])
        self.assertTrue(mapped['tokens'][-1]['mapped'])
        self.assertEqual(mapped['coverage']['unmapped_lexical_characters'], 2)
        self.assertTrue(mapped['tokenizer_differences'])

    def test_missing_repeated_phrase_is_ambiguous(self):
        mapped = map_tokens('はいはい', ['はい'])
        self.assertFalse(mapped['tokens'][0]['mapped'])
        self.assertEqual(mapped['tokens'][0]['mapping_status'], 'repeated_partial_match_ambiguous')
        self.assertEqual(mapped['coverage']['unmapped_lexical_characters'], 4)

    def test_unicode_offsets_and_unmapped_insertions(self):
        mapped = map_tokens('🎧深津さん。', ['余分', '深津', 'さん'])
        self.assertFalse(mapped['tokens'][0]['mapped'])
        self.assertEqual(mapped['tokens'][1]['original_raw_span'], [1, 3])
        self.assertEqual(mapped['coverage']['original_characters'], 6)


class CaptureTests(unittest.TestCase):
    def test_capture_keeps_raw_and_restores_parser(self):
        class Processor:
            def parse_timestamp(self, words, times):
                return [{'text': word, 'start_time': 80, 'end_time': 160} for word in words]
        processor, captures = Processor(), []
        original = processor.parse_timestamp
        with capture_parser(processor, captures):
            result = processor.parse_timestamp(['言葉'], [160, 80])
        self.assertEqual(processor.parse_timestamp, original)
        self.assertEqual(captures[0]['raw_timestamp_ms'], [160., 80.])
        self.assertEqual(captures[0]['repaired_items_ms'], result)
        rows = annotate_timestamps(captures[0], 60, 480)
        self.assertEqual(rows[0]['raw_source_seconds'], [480.16, 480.08])
        self.assertEqual(rows[0]['repaired_source_seconds'], [480.08, 480.16])
        self.assertIn('timestamp_repaired_by_library', rows[0]['flags'])
        self.assertIn('raw_reversed_interval', rows[0]['flags'])
        self.assertFalse(rows[0]['unrepaired_timing_checks_pass'])

    def test_capture_survives_parser_failure(self):
        class Processor:
            def parse_timestamp(self, words, times):
                raise RuntimeError('parser failed')
        processor, captures = Processor(), []
        original = processor.parse_timestamp
        with self.assertRaisesRegex(RuntimeError, 'parser failed'):
            with capture_parser(processor, captures):
                processor.parse_timestamp(['原文'], [0, 80])
        self.assertEqual(processor.parse_timestamp, original)
        self.assertEqual(captures[0]['raw_timestamp_ms'], [0., 80.])
        self.assertIsNone(captures[0]['repaired_items_ms'])

    def test_zero_outside_nonmonotonic_and_nonfinite_flagged_without_clamp(self):
        capture = {'words': ['一', '二', '三', '四'],
                   'raw_timestamp_ms': [0, 0, 800, 1200, 400, 600, float('nan'), 800],
                   'repaired_items_ms': [
                       {'text': '一', 'start_time': 0, 'end_time': 0},
                       {'text': '二', 'start_time': 800, 'end_time': 1200},
                       {'text': '三', 'start_time': 400, 'end_time': 600},
                       {'text': '四', 'start_time': float('nan'), 'end_time': 800}]}
        rows = annotate_timestamps(capture, 1, 10)
        self.assertIn('raw_zero_length', rows[0]['flags'])
        self.assertIn('raw_outside_audio_extent', rows[1]['flags'])
        self.assertEqual(rows[1]['raw_source_seconds'], [10.8, 11.2])
        self.assertIn('raw_nonmonotonic_sequence', rows[2]['flags'])
        self.assertIn('raw_nonfinite_timestamp', rows[3]['flags'])
        self.assertIsNone(rows[3]['raw_source_seconds'])
        json.dumps(json_safe(rows), allow_nan=False)

    def test_timestamp_count_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, 'two raw timestamps'):
            annotate_timestamps({'words': ['一'], 'raw_timestamp_ms': [0],
                                 'repaired_items_ms': []}, 1)


class CLITests(unittest.TestCase):
    def test_import_and_help_without_site_packages(self):
        script = str(Path(__file__).with_name('align_qwen.py'))
        process = subprocess.run([sys.executable, '-S', script, '--help'], capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn('--source-offset', process.stdout)

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / 'keep.txt'
            marker.write_text('keep')
            with self.assertRaises(FileExistsError):
                run(argparse.Namespace(output=directory))
            self.assertEqual(marker.read_text(), 'keep')
            self.assertEqual(len(list(Path(directory).iterdir())), 1)

    def test_preparation_failure_is_recorded_without_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'out'
            args = argparse.Namespace(output=output, model=Path(directory)/'missing',
                                      audio='missing.wav', text='missing.txt', source_offset=0.)
            with self.assertRaisesRegex(ValueError, 'prepared local model'):
                run(args)
            manifest = json.loads((output/'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'failed')
            self.assertEqual(manifest['failed_stage'], 'preparing')
            self.assertTrue((output/'error.log').is_file())


if __name__ == '__main__':
    unittest.main()
