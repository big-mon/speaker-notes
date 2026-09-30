import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from align_qwen import map_tokens
from aligned_speaker_turns import build_aligned_turns, sentence_spans
from compose_alignment import compose_alignments


def alignment(text, words, times, duration=30., offset=0.):
    mapping = map_tokens(text, words)
    items = []
    for token, pair in zip(mapping['tokens'], times):
        items.append({'index': token['index'], 'text': token['text'],
                      'original_character_spans': token['original_character_spans'],
                      'raw_clip_seconds': list(pair), 'repaired_clip_seconds': list(pair),
                      'raw_source_seconds': [offset + v for v in pair],
                      'repaired_source_seconds': [offset + v for v in pair],
                      'library_repaired': False, 'flags': [],
                      'eligible_for_boundary_candidate': True})
    return {'schema_version': 1, 'mapping': mapping, 'items': items,
            'duration_seconds': duration, 'source_offset_seconds': offset}


def diar(*rows):
    return {'engine': 'synthetic', 'configuration': {'overlap': True},
            'segments': [{'start': a, 'end': b, 'speaker': s} for a, b, s in rows]}


class AlignedSpeakerTurnTests(unittest.TestCase):
    def test_primary_and_secondary_empty_results_are_independent(self):
        from review_aligned_scene import make_document
        for primary in ('', '本文です。'):
            for secondary in ([], [{'text': '比較文です。', 'units': [{'text': '比較文です。', 'start': 0., 'end': 5.}]}]):
                with self.subTest(primary=primary, secondary=bool(secondary)):
                    aligned = alignment(primary, [primary[:-1]] if primary else [], [(0, 5)] if primary else [], duration=10.)
                    document = make_document(aligned, diar(), secondary, {}, {}, {})
                    self.assertEqual(''.join(row['text'] for row in document['segments']), primary)
                    self.assertEqual(document['comparison']['apple_text'], '比較文です。' if secondary else '')
                    with tempfile.TemporaryDirectory() as folder:
                        from scene_transcript import save
                        save(document, Path(folder)/'result')
                        self.assertEqual(document['comparison']['qwen_text'], primary)

    def test_internal_time_regression_falls_back_and_keeps_raw_evidence(self):
        a = alignment('一二三。', ['一', '二', '三'], [(0, 5), (20, 25), (10, 15)])
        result = build_aligned_turns(a, diar((0, 30, 'speaker')))
        row = result['segments'][0]
        self.assertEqual((row['start'], row['end']), (0, 30))
        self.assertEqual(row['time_kind'], 'source_clip_envelope_fallback')
        self.assertEqual(row['speakers'], [])
        self.assertEqual(result['raw_alignment'], a)

    def test_asr_difference_links_use_half_open_spans_and_terminal_deletions(self):
        from review_aligned_scene import difference_overlaps
        rows = [(0, 4), (4, 8)]
        for span, expected in [([0, 4], [True, False]), ([4, 8], [False, True]),
                               ([4, 4], [False, True]), ([8, 8], [False, True]),
                               ([0, 0], [True, False]), ([3, 5], [True, True])]:
            self.assertEqual([difference_overlaps(span, a, b, 8) for a, b in rows], expected)

    def test_question_suffix_stays_before_new_speaker_start_snap(self):
        a = alignment('質問を任せみたいな。回答します。',
                      ['質問を', '任せ', 'みたいな', '回答します'],
                      [(0, 6), (6, 8), (8, 10.2), (10.2, 19)])
        d = diar((0, 9.8, 'host'), (9.2, 20, 'guest'))
        result = build_aligned_turns(a, d)
        transition = result['transitions'][0]
        self.assertEqual(transition['original_event_transition']['clip_seconds'], 9.2)
        self.assertEqual(transition['snapped_clip_seconds'], 10.2)
        self.assertEqual(result['segments'][0]['text'], '質問を任せみたいな。')
        self.assertEqual([r['speakers'] for r in result['segments']], [['host'], ['guest']])

    def test_short_response_is_separate_and_overlap_is_retained(self):
        a = alignment('質問です。そうですね。回答を続けます。',
                      ['質問です', 'そうですね', '回答を続けます'],
                      [(0, 10), (10, 10.8), (11, 20)])
        d = diar((0, 11.5, 'alpha'), (10.1, 21, 'beta'))
        result = build_aligned_turns(a, d)
        row = result['segments'][1]
        self.assertEqual(row['text'], 'そうですね。')
        self.assertEqual(row['candidate_main_speaker'], 'beta')
        self.assertEqual(row['speakers'], [])
        self.assertEqual(row['raw_overlap_intervals'][0]['start'], 10.1)
        self.assertFalse(row['human_verified'])
        self.assertFalse(row['ready_for_summary'])

    def test_no_punctuation_keeps_whole_text_unknown(self):
        a = alignment('質問と回答が連続します', ['質問と', '回答が', '連続します'],
                      [(0, 5), (5, 10), (10, 20)])
        result = build_aligned_turns(a, diar((0, 9, 'x'), (8, 21, 'y')))
        self.assertEqual(len(result['segments']), 1)
        self.assertEqual(result['segments'][0]['text'], a['mapping']['original_text'])
        self.assertEqual(result['segments'][0]['speakers'], [])
        self.assertIn('unresolved_major_transition_inside_sentence', result['segments'][0]['flags'])

    def test_decimal_whitespace_and_closing_punctuation_preserved(self):
        text = '「費用は3.5億円です。」\n次の話です？！\n'
        spans = sentence_spans(text)
        self.assertEqual([text[a:b] for a, b in spans], ['「費用は3.5億円です。」\n', '次の話です？！\n'])
        a = alignment(text, ['費用は', '35', '億円です', '次の話です'],
                      [(0, 2), (2, 4), (4, 7), (7, 12)])
        self.assertEqual(len(a['items'][1]['original_character_spans']), 2)
        result = build_aligned_turns(a, diar((0, 7, 'one'), (7, 15, 'two')))
        self.assertEqual(''.join(r['text'] for r in result['segments']), text)
        self.assertEqual(len(result['sentence_boundaries']), 1)

    def test_bad_last_adjacent_token_never_replaced_by_earlier_good_token(self):
        original = alignment('大きいんですね。次です。', ['大きい', 'んです', 'ね', '次です'],
                             [(0, 4), (4, 8), (8, 9), (9, 18)])
        for change in ('repaired', 'zero_length', 'flagged'):
            with self.subTest(change=change):
                a = copy.deepcopy(original)
                token = a['items'][2]
                if change == 'repaired':
                    token['library_repaired'] = True
                elif change == 'zero_length':
                    token['raw_clip_seconds'] = [9, 9]
                else:
                    token['flags'] = ['unreliable']
                token['eligible_for_boundary_candidate'] = False
                result = build_aligned_turns(a, diar((0, 8, 'a'), (7.5, 20, 'b')))
                boundary = result['sentence_boundaries'][0]
                self.assertFalse(boundary['eligible'])
                self.assertEqual(boundary['left_token_index'], 2)
                self.assertIsNone(boundary['clip_seconds'])

    def test_repair_gate_is_strictly_greater_than_ten_percent(self):
        original = alignment(''.join('語。' for _ in range(100)), ['語'] * 100,
                             [(i * 3, i * 3 + 3) for i in range(100)], duration=300)
        for count, blocked in ((10, False), (11, True)):
            a = copy.deepcopy(original)
            for token in a['items'][:count]:
                token.update(library_repaired=True, flags=['timestamp_repaired_by_library'],
                             eligible_for_boundary_candidate=False)
            result = build_aligned_turns(a, diar((0, 300, 'single')))
            self.assertEqual(result['diagnostics']['automatic_assignment_blocked'], blocked)
            self.assertEqual(result['diagnostics']['repaired_tokens'], count)
            if blocked:
                self.assertTrue(all(not row['speakers'] for row in result['segments']))
            else:
                self.assertTrue(any(row['speakers'] for row in result['segments']))
            self.assertFalse(result['ready_for_summary'])

    def test_source_offset_applied_once_and_no_speaker_count_assumption(self):
        a = alignment('質問。回答。訂正。', ['質問', '回答', '訂正'],
                      [(0, 10), (10, 20), (20, 30)], offset=1290.)
        result = build_aligned_turns(a, diar((0, 10.5, 'red'), (9, 20.5, 'green'), (19, 30, 'blue')))
        self.assertEqual([t['snapped_source_seconds'] for t in result['transitions']], [1300., 1310.])
        self.assertEqual([r['speakers'] for r in result['segments']], [['red'], ['green'], ['blue']])
        self.assertEqual(result['segments'][0]['playback'], {'start': 1290., 'end': 1300.})
        self.assertEqual(result['raw_diarization']['segments'][1]['start'], 9)

    def test_unreliable_child_repair_rate_not_diluted_by_healthy_child(self):
        a = alignment(''.join('語。' for _ in range(200)), ['語'] * 200,
                      [(i * 3, i * 3 + 3) for i in range(200)], duration=600)
        for index, token in enumerate(a['items']):
            token['alignment_part_index'] = index // 100
            if index < 100:
                token['flags'] = ['alignment_chunk_repair_gate']
                token['eligible_for_boundary_candidate'] = False
            if index < 11:
                token['library_repaired'] = True
                token['flags'].append('timestamp_repaired_by_library')
        result = build_aligned_turns(a, diar((0, 600, 'speaker')))
        self.assertFalse(result['diagnostics']['automatic_assignment_blocked'])
        self.assertAlmostEqual(result['diagnostics']['repaired_token_fraction'], .055)
        self.assertTrue(all(not r['speakers'] for r in result['segments'][:100]))
        self.assertTrue(all('alignment_chunk_repair_gate' in r['flags'] for r in result['segments'][:100]))
        self.assertTrue(all(r['speakers'] == ['speaker'] for r in result['segments'][100:]))
        self.assertEqual(result['segments'][100]['alignment_part_indices'], [1])

    def test_sentence_crossing_asr_chunk_seam_stays_unknown_even_with_same_diar_speaker(self):
        a = alignment('前半の説明後半の説明。', ['前半の説明', '後半の説明'], [(0, 5), (5, 12)])
        for index, token in enumerate(a['items']):
            token.update(alignment_part_index=index, original_token_index=0,
                         flags=['alignment_chunk_edge'], eligible_for_boundary_candidate=False)
        result = build_aligned_turns(a, diar((0, 12, 'same-global-speaker')))
        row = result['segments'][0]
        self.assertEqual(row['text'], a['mapping']['original_text'])
        self.assertEqual(row['candidate_main_speaker'], 'same-global-speaker')
        self.assertEqual(row['speakers'], [])
        self.assertEqual(row['alignment_part_indices'], [0, 1])
        self.assertIn('alignment_chunk_seam_inside_sentence', row['flags'])
        self.assertEqual(result['raw_alignment'], a)

    def test_validated_chunks_bound_unpunctuated_text_without_losing_separators(self):
        parts = []
        for index, text in enumerate(('前半の説明', '中盤の理由', '最後の具体例。')):
            start = index * 10.
            child = alignment(text, [text.rstrip('。')], [(0, 9)], duration=10., offset=100. + start)
            parts.append({'start': start, 'end': start + 10., 'alignment': child})
        composite = compose_alignments(parts, 100., 30.)
        before = copy.deepcopy(composite)
        result = build_aligned_turns(composite, diar((0, 30, 'same-speaker')))
        rows = result['segments']
        self.assertEqual([row['text'] for row in rows], ['前半の説明\n', '中盤の理由\n', '最後の具体例。'])
        self.assertEqual(''.join(row['text'] for row in rows), composite['mapping']['original_text'])
        self.assertEqual([row['alignment_part_indices'] for row in rows], [[0], [1], [2]])
        self.assertTrue(all(not row['speakers'] for row in rows))
        self.assertTrue(all('sentence_split_at_alignment_chunk_boundary' in row['flags'] for row in rows))
        self.assertTrue(all(not boundary['eligible'] for boundary in result['sentence_boundaries']))
        self.assertTrue(all('source_chunk_boundary_not_sentence_boundary' in boundary['flags']
                            for boundary in result['sentence_boundaries']))
        self.assertEqual(composite, before)

    def test_invalid_middle_chunk_times_fall_back_to_exact_source_chunk_only(self):
        parts = []
        for index, text in enumerate(('前。', '重要な理由。', '後。')):
            start = index * 10.
            child = alignment(text, [text.rstrip('。')], [(0, 9)], duration=10., offset=100. + start)
            parts.append({'start': start, 'end': start + 10., 'alignment': child})
        token = parts[1]['alignment']['items'][0]
        token.update(raw_clip_seconds=['nan', 9.], raw_source_seconds=None,
                     flags=['raw_nonfinite_timestamp'], eligible_for_boundary_candidate=False)
        composite = compose_alignments(parts, 100., 30.)
        result = build_aligned_turns(composite, diar((0, 30, 'speaker')))
        row = result['segments'][1]
        self.assertEqual(row['text'], '重要な理由。\n')
        self.assertEqual((row['start'], row['end']), (110., 120.))
        self.assertEqual(row['time_kind'], 'source_chunk_envelope_fallback')
        self.assertEqual(row['source_chunk_extent'], {'start': 110., 'end': 120., 'alignment_part_index': 1})
        self.assertEqual(row['speakers'], [])
        self.assertIn('broad_playback_timing_fallback', row['flags'])
        self.assertEqual(result['raw_alignment']['items'][1]['raw_clip_seconds'][0], 'nan')

    def test_child_time_outside_its_chunk_is_not_validated_against_full_recording(self):
        parts = []
        for index, text in enumerate(('前。', '中。', '後。')):
            start = index * 10.
            child = alignment(text, [text.rstrip('。')], [(0, 9)], duration=10., offset=100. + start)
            parts.append({'start': start, 'end': start + 10., 'alignment': child})
        token = parts[1]['alignment']['items'][0]
        token.update(raw_clip_seconds=[0., 15.], raw_source_seconds=[110., 125.],
                     flags=['raw_outside_audio'], eligible_for_boundary_candidate=False)
        composite = compose_alignments(parts, 100., 30.)
        row = build_aligned_turns(composite, diar((0, 30, 'speaker')))['segments'][1]
        self.assertEqual((row['start'], row['end']), (110., 120.))
        self.assertEqual(row['time_kind'], 'source_chunk_envelope_fallback')
        self.assertFalse(row['speakers'])

    def test_unmapped_child_text_uses_character_provenance_for_fallback(self):
        first = alignment('前。', ['前'], [(0, 9)], duration=10.)
        second = alignment('全文未対応。', ['全文未対応'], [(0, 9)], duration=10., offset=10.)
        second['items'][0]['original_character_spans'] = []
        composite = compose_alignments([
            {'start': 0., 'end': 10., 'alignment': first},
            {'start': 10., 'end': 20., 'alignment': second}], 0., 20.)
        row = build_aligned_turns(composite, diar((0, 20, 'speaker')))['segments'][1]
        self.assertEqual(row['text'], '全文未対応。')
        self.assertEqual((row['start'], row['end']), (10., 20.))
        self.assertEqual(row['source_chunk_extent']['alignment_part_index'], 1)
        self.assertEqual(row['time_kind'], 'source_chunk_envelope_fallback')
        self.assertFalse(row['speakers'])

    def test_valid_composite_uses_child_gate_when_overall_repair_rate_is_high(self):
        text, words = '語。' * 100, ['語'] * 100
        times = [(i * 3, i * 3 + 3) for i in range(100)]
        bad = alignment(text, words, times, duration=300.)
        healthy = alignment(text, words, times, duration=300., offset=300.)
        for token in bad['items'][:50]:
            token.update(library_repaired=True, flags=['timestamp_repaired_by_library'],
                         eligible_for_boundary_candidate=False)
        composite = compose_alignments([
            {'start': 0., 'end': 300., 'alignment': bad},
            {'start': 300., 'end': 600., 'alignment': healthy},
        ], 0., 600.)
        result = build_aligned_turns(composite, diar((0, 600, 'speaker')))
        self.assertEqual(result['diagnostics']['repaired_token_fraction'], .25)
        self.assertFalse(result['diagnostics']['automatic_assignment_blocked'])
        self.assertEqual(result['diagnostics']['repair_gate_scope'], 'validated_child_alignments')
        self.assertTrue(all(not r['speakers'] for r in result['segments'][:100]))
        self.assertTrue(all(r['speakers'] == ['speaker'] for r in result['segments'][100:]))
        for tamper in ('missing_children', 'removed_gate', 'part_count', 'text'):
            with self.subTest(tamper=tamper):
                invalid = copy.deepcopy(composite)
                if tamper == 'missing_children':
                    invalid.pop('raw_children')
                elif tamper == 'removed_gate':
                    invalid['items'][70]['flags'].remove('alignment_chunk_repair_gate')
                elif tamper == 'part_count':
                    invalid['composition_parts'][0]['repaired_tokens'] = 0
                else:
                    invalid['mapping']['original_text'] += '創作'
                with self.assertRaisesRegex(ValueError, 'Invalid composite alignment contract'):
                    build_aligned_turns(invalid, diar((0, 600, 'speaker')))
        # A kind-less aggregate is not trusted to disable the original gate.
        single = copy.deepcopy(composite)
        single.pop('kind')
        guarded = build_aligned_turns(single, diar((0, 600, 'speaker')))
        self.assertTrue(guarded['diagnostics']['automatic_assignment_blocked'])
        self.assertTrue(all(not r['speakers'] for r in guarded['segments']))

    def test_chunk_edge_does_not_support_transition_snap(self):
        a = alignment('質問の文章。回答の文章。', ['質問の文章', '回答の文章'], [(0, 10), (10, 20)])
        for index, token in enumerate(a['items']):
            token.update(alignment_part_index=index, flags=['alignment_chunk_edge'],
                         eligible_for_boundary_candidate=False)
        result = build_aligned_turns(a, diar((0, 10.5, 'a'), (9.5, 21, 'b')))
        self.assertFalse(result['sentence_boundaries'][0]['eligible'])
        self.assertEqual(result['transitions'][0]['state'], 'unresolved')
        self.assertEqual(result['segments'][0]['speakers'], [])
        self.assertEqual(result['segments'][1]['alignment_part_indices'], [1])

    def test_neighboring_same_speaker_major_events_join_but_brief_event_survives(self):
        a = alignment('一つ目。二つ目。', ['一つ目', '二つ目'], [(0, 4), (5, 9)])
        d = diar((0, 4, 'a'), (3, 3.5, 'b'), (5, 9, 'a'))
        result = build_aligned_turns(a, d)
        self.assertEqual(len(result['major_turns']), 1)
        self.assertEqual(result['transitions'], [])
        self.assertEqual(len(result['activity_candidates']['raw_events']), 3)
        self.assertEqual(result['activity_candidates']['unresolved_fully_overlapped_event_ids'], [1])
        self.assertTrue(result['segments'][0]['raw_overlap_intervals'])

    def test_raw_documents_unchanged_and_output_independent(self):
        a = alignment('本文。', ['本文'], [(0, 10)])
        d = diar((0, 10, 'a'))
        original_a, original_d = copy.deepcopy(a), copy.deepcopy(d)
        result = build_aligned_turns(a, d)
        result['raw_alignment']['items'][0]['raw_clip_seconds'][0] = 123
        result['raw_diarization']['segments'][0]['speaker'] = 'changed'
        self.assertEqual(a, original_a)
        self.assertEqual(d, original_d)

    def test_brief_sequential_correction_is_not_attributed_to_surrounding_speaker(self):
        a = alignment('説明しますいいえ訂正して続けます。', ['説明します', 'いいえ', '訂正して続けます'],
                      [(0, 5), (5, 6), (6, 12)])
        result = build_aligned_turns(a, diar((0, 5, 'a'), (5, 6, 'b'), (6, 12, 'a')))
        row = result['segments'][0]
        self.assertEqual(row['text'], a['mapping']['original_text'])
        self.assertEqual(row['candidate_main_speaker'], 'a')
        self.assertEqual(row['speakers'], [])
        self.assertIn('unseparated_brief_other_speaker_activity', row['flags'])
        self.assertEqual(len(result['activity_candidates']['raw_events']), 3)

    def test_unmapped_lexical_text_not_silently_given_neighbor_timestamps_and_speaker(self):
        a = alignment('前文重要な訂正後文。', ['前文', '重要な訂正', '後文'], [(0, 4), (4, 8), (8, 12)])
        a['items'][1]['original_character_spans'] = []
        result = build_aligned_turns(a, diar((0, 12, 'a')))
        self.assertEqual(result['segments'][0]['speakers'], [])
        self.assertIn('unmapped_lexical_text_timing_unresolved', result['segments'][0]['flags'])

    def test_nonfinite_raw_time_gets_broad_fallback_not_fabricated_word_time(self):
        a = alignment('本文。', ['本文'], [(0, 10)])
        a['items'][0]['raw_clip_seconds'] = ['nan', 10]
        result = build_aligned_turns(a, diar((0, 10, 'a')))
        self.assertEqual(result['segments'][0]['playback'], {'start': 0., 'end': 30.})
        self.assertEqual(result['segments'][0]['speakers'], [])
        self.assertEqual(result['raw_alignment']['items'][0]['raw_clip_seconds'][0], 'nan')

    def test_cli_keeps_hash_provenance_and_refuses_existing_output(self):
        a = alignment('本文。', ['本文'], [(0, 10)])
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            source, raw, output = directory / 'alignment.json', directory / 'diar.json', directory / 'out'
            source.write_text(json.dumps(a))
            raw.write_text(json.dumps(diar((0, 10, 'a'))))
            (directory / 'manifest.json').write_text(json.dumps({'model': 'synthetic'}))
            command = [sys.executable, str(Path(__file__).with_name('aligned_speaker_turns.py')), str(source), str(raw), str(output)]
            subprocess.run(command, check=True, capture_output=True)
            document = json.loads((output / 'candidates.json').read_text())
            self.assertEqual(len(document['raw_sources']['alignment']['sha256']), 64)
            self.assertEqual(document['alignment_manifest'], {'model': 'synthetic'})
            self.assertEqual((output / 'text.txt').read_text(), '本文。')
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual(json.loads((output / 'candidates.json').read_text()), document)


if __name__ == '__main__':
    unittest.main()
