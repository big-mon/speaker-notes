"""Structural safety checks; these do not measure acoustic ASR/diarization accuracy."""
import copy
import math
import unittest

from scene_transcript import assemble, nearby_handover


def segment(*units):
    return {'text': ''.join(u['text'] for u in units), 'units': list(units)}


def unit(text, start, end):
    return {'text': text, 'start': start, 'end': end}


def events(*spans):
    return {'segments': [{'start': a, 'end': b, 'speaker': speaker}
                         for a, b, speaker in spans]}


class SceneTranscriptTests(unittest.TestCase):
    def test_apple_text_is_unchanged_and_no_review_is_auto_accepted(self):
        apple = [segment(unit('第1回は仕組み、', 0, 3),
                         unit('第2回はAIです。', 3, 6))]
        original = copy.deepcopy(apple)
        document = assemble(apple, '第1回はAI、第2回は仕組みです。',
                            events((0, 6, 'speaker-9')), 6)
        self.assertEqual(''.join(r['text'] for r in document['segments']), apple[0]['text'])
        self.assertEqual(apple, original)
        self.assertEqual(document['comparison']['qwen_text'], '第1回はAI、第2回は仕組みです。')
        self.assertFalse(document['ready_for_attributed_summary'])
        self.assertFalse(document['validation']['accuracy_verified'])
        for row in document['segments']:
            self.assertEqual(set(row['quality'].values()), {'review_required'})
            self.assertFalse(row['ready_for_attributed_summary'])

    def test_substantial_question_and_answer_are_not_forced_to_main_speaker(self):
        text = 'なぜ始めたのですか理由は必要だと考えたからです'
        document = assemble([segment(unit(text, 0, 10))], text,
                            events((0, 6, 'question'), (6, 10, 'answer')), 10)
        row = document['segments'][0]
        self.assertEqual(row['speakers'], [])
        self.assertEqual(row['state'], 'unknown')
        self.assertIn('major_speaker_unresolved', row['review_reasons'])
        self.assertEqual(row['text'], text)

    def test_short_overlap_is_retained_and_not_called_unimportant(self):
        diarization = events((0, 10, 'long'), (4, 4.5, 'brief'))
        document = assemble([segment(unit('それは違いますという説明', 0, 10))],
                            'それは違いますという説明', diarization, 10)
        row = document['segments'][0]
        self.assertEqual(row['state'], 'mixed')
        self.assertIn(1, row['overlap_event_ids'])
        self.assertIn(1, document['activity']['unresolved_fully_overlapped_event_ids'])
        self.assertEqual(document['diarization'], diarization['segments'])
        self.assertIn('other_speech_not_separated_into_text', row['review_reasons'])
        self.assertFalse(row['ready_for_attributed_summary'])

    def test_substantial_other_speaker_is_not_absorbed_by_a_long_block(self):
        for duration, short in [(40, 4), (6, 2.28)]:
            document = assemble([segment(unit('重要な主張と別人による補足の本文', 0, duration))],
                                '重要な主張と別人による補足の本文',
                                events((0, duration-short, 'a'), (duration-short, duration, 'b')),
                                duration)
            self.assertEqual(document['segments'][0]['speakers'], [])

    def test_handover_support_uses_sustained_changes_and_has_no_word_time_claim(self):
        from turn_candidates import build_candidates
        activity = build_candidates(events((0, 10, 'a'), (9, 20, 'b'), (15, 15.4, 'a'))['segments'])
        self.assertEqual(len(nearby_handover(activity, 10)), 1)
        self.assertEqual(nearby_handover(activity, 15), [])
        self.assertFalse(nearby_handover(activity, 10)[0]['acoustically_verified'])

    def test_source_offset_does_not_shift_raw_times(self):
        apple = [segment(unit('原音声の途中です', 2, 7))]
        document = assemble(apple, '原音声の途中です', events((2, 7, 'x')), 10,
                            source_offset=1290)
        row = document['segments'][0]
        self.assertEqual((row['start'], row['end']), (1292, 1297))
        self.assertEqual((row['playback_start'], row['playback_end']), (1290, 1299))
        self.assertEqual((document['units'][0]['start'], document['units'][0]['end']), (2, 7))
        self.assertEqual(document['source_window'], {'start': 1290, 'end': 1300})

    def test_missing_timing_preserves_text_and_stays_unknown(self):
        document = assemble([segment(unit('時刻が不明でも残す本文', None, None))],
                            '時刻が不明でも残す本文', events((0, 10, 'x')), 10)
        row = document['segments'][0]
        self.assertEqual(row['text'], '時刻が不明でも残す本文')
        self.assertEqual(row['speakers'], [])
        self.assertEqual(row['time_kind'], 'input_extent')
        self.assertEqual((row['start'], row['end']), (0, 10))

    def test_partial_edge_units_preserve_native_extent_and_whole_text(self):
        apple = [segment(unit('前から続く言葉', -.12, .06), unit('説明の本体', .06, 4))]
        document = assemble(apple, apple[0]['text'], events((0, 10, 'x')), 10,
                            source_offset=1290)
        row = document['segments'][0]
        self.assertEqual(row['text'], apple[0]['text'])
        self.assertEqual(row['native_unit_envelope_clip_local'], [-.12, 4])
        self.assertEqual((row['start'], row['end']), (1290, 1294))
        self.assertEqual(row['time_kind'], 'native_apple_unit_envelope_clipped_to_window')
        self.assertEqual(document['units'][0]['start'], -.12)
        self.assertEqual(row['quality']['boundary'], 'review_required')
        tail = assemble([segment(unit('末尾の語', 8, 10.06))], '末尾の語',
                        events((0, 10, 'x')), 10)['segments'][0]
        self.assertEqual((tail['start'], tail['end']), (8, 10))
        self.assertEqual(tail['native_unit_envelope_clip_local'], [8, 10.06])

    def test_zero_duration_tail_keeps_row_envelope_and_is_explicit(self):
        document = assemble([segment(unit('最後の説明', 7, 10), unit('ー', 10, 10))],
                            '最後の説明ー', events((0, 10, 'x')), 10)
        row = document['segments'][0]
        self.assertEqual((row['start'], row['end']), (7, 10))
        self.assertEqual(row['text'], '最後の説明ー')
        self.assertEqual(row['timing_review_signals']['point_unit_references'], [[0, 1]])
        self.assertIn('native_zero_duration_units; point_times_not_word_intervals', row['review_reasons'])
        only_point = assemble([segment(unit('語', 10, 10))], '語', events((0, 10, 'x')), 10)
        self.assertEqual(only_point['segments'][0]['time_kind'], 'input_extent')

    def test_long_single_character_unit_is_a_review_signal_not_omission_proof(self):
        row = assemble([segment(unit('い', .48, 4.2), unit('説明', 4.2, 5))],
                       'い説明', events((0, 5, 'x')), 5)['segments'][0]
        self.assertEqual(row['timing_review_signals']['sparse_unit_references'], [[0, 0]])
        self.assertFalse(row['ready_for_attributed_summary'])

    def test_no_secondary_punctuation_uses_native_result_groups_explicitly(self):
        apple = [segment(unit('最初の説明です', 0, 5)), segment(unit('次の説明です', 5, 10))]
        for i, r in enumerate(apple):
            r.update(start=i*5, end=(i+1)*5)
        document = assemble(apple, '最初の説明です次の説明です', events((0, 10, 'x')), 10)
        self.assertEqual([r['text'] for r in document['segments']], ['最初の説明です', '次の説明です'])
        for row in document['segments']:
            self.assertIn('native_result_group_boundary; not_verified_sentence_or_speaker_boundary', row['review_reasons'])
            self.assertFalse(row['ready_for_attributed_summary'])

    def test_diarization_change_cannot_resolve_unmatched_sentence_suffix(self):
        text = 'これが順番だと思うんですねもう一個の説明をします'
        units = [unit(c, i*.2, (i+1)*.2) for i, c in enumerate(text)]
        apple = [dict(segment(*units), start=0, end=len(text)*.2)]
        cut = text.index('んです')*.2
        doc = assemble(apple, 'これが順番だと思う。でねもう一個の説明をします。',
                       events((0, cut+.1, 'a'), (cut, len(text)*.2, 'b')), len(text)*.2)
        self.assertTrue(doc['deferred_boundary_proposals'])
        self.assertIn('思うんですね', ''.join(doc['segments'][0]['text']))
        self.assertFalse(any(r['text'].endswith('思う') for r in doc['segments']))

    def test_segment_without_native_units_is_preserved_without_claiming_fine_times(self):
        apple = [{'text': '古い段落形式も本文は残します', 'start': 1, 'end': 9}]
        document = assemble(apple, apple[0]['text'], events((0, 10, 'x')), 10)
        row = document['segments'][0]
        self.assertEqual(row['text'], apple[0]['text'])
        self.assertEqual(row['speakers'], [])
        self.assertEqual(row['time_kind'], 'input_extent')

    def test_brief_reply_does_not_inherit_the_surrounding_main_speaker(self):
        document = assemble([segment(unit('違います', 4, 5))], '違います',
                            events((0, 10, 'long')), 10)
        self.assertEqual(document['segments'][0]['speakers'], [])
        self.assertIn('brief_text_not_inherited_from_surrounding_main_speaker',
                      document['segments'][0]['review_reasons'])

    def test_qwen_punctuation_cannot_split_a_native_multi_character_unit(self):
        first = '北欧の暮らしを話します'
        apple = [segment(unit(first, 0, 5), unit('終了です', 5, 7))]
        document = assemble(apple, '北欧の暮らし。を話します。終了です。',
                            events((0, 7, 'x')), 7)
        rows = document['segments']
        self.assertEqual(rows[0]['text'], first)
        self.assertEqual((rows[0]['start'], rows[0]['end']), (0, 5))
        references = [reference for row in rows for reference in row['source_unit_references']]
        self.assertEqual(references, [[0, 0], [0, 1]])
        self.assertEqual(''.join(row['text'] for row in rows), first + '終了です')

    def test_standalone_native_punctuation_stays_with_the_preceding_text(self):
        apple = [segment(unit('最初の説明です', 0, 4), unit('。', 4, 4.1),
                         unit('次の説明です', 4.1, 8), unit('。', 8, 8.1))]
        document = assemble(apple, apple[0]['text'], events((0, 8.1, 'x')), 8.1)
        self.assertEqual([r['text'] for r in document['segments']],
                         ['最初の説明です。', '次の説明です。'])
        self.assertEqual(document['segments'][0]['source_unit_references'], [[0, 0], [0, 1]])

    def test_qwen_only_tail_is_visible_as_a_difference_for_review(self):
        document = assemble([segment(unit('これは説明です', 0, 4))],
                            'これは説明です。重要な理由は別にあります。', events((0, 4, 'x')), 4)
        tail_ids = {index for index, difference in enumerate(document['comparison']['differences'])
                    if difference['operation'] == 'insert' and
                    difference['apple_raw_span'] == [len('これは説明です')] * 2}
        self.assertTrue(tail_ids)
        shown_ids = {index for row in document['segments'] for index in row['asr_difference_ids']}
        self.assertTrue(tail_ids.issubset(shown_ids), 'Qwen-only tail must not disappear from review rows')

    def test_invalid_source_offsets_are_rejected(self):
        for offset in (math.nan, math.inf, -math.inf, -1):
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                assemble([segment(unit('本文です', 0, 1))], '本文です',
                         events((0, 1, 'x')), 1, source_offset=offset)

    def test_non_monotonic_native_times_are_not_promoted_to_speaker_timing(self):
        apple = [segment(unit('先の文章', 5, 6), unit('後の文章', 0, 1))]
        document = assemble(apple, '先の文章後の文章', events((0, 6, 'x')), 6)
        row = document['segments'][0]
        self.assertEqual(row['text'], '先の文章後の文章')
        self.assertEqual(row['speakers'], [])
        self.assertEqual(row['time_kind'], 'input_extent')

    def test_outside_timing_tolerance_never_creates_a_reversed_output_range(self):
        document = assemble([segment(unit('入力の外側の時刻です', 10.1, 10.2))],
                            '入力の外側の時刻です', events((0, 10, 'x')), 10)
        row = document['segments'][0]
        self.assertLess(row['start'], row['end'])
        self.assertGreaterEqual(row['start'], document['source_window']['start'])
        self.assertLessEqual(row['end'], document['source_window']['end'])
        self.assertEqual(row['speakers'], [])
        self.assertEqual(row['time_kind'], 'input_extent')

    def test_sentence_cuts_cannot_hide_non_monotonic_source_times(self):
        apple = [segment(unit('最初の説明です', 5, 6), unit('次の説明です', 0, 1))]
        document = assemble(apple, '最初の説明です。ここに異なる補足。次の説明です。',
                            events((0, 6, 'x')), 6)
        self.assertEqual(''.join(row['text'] for row in document['segments']), apple[0]['text'])
        for row in document['segments']:
            self.assertEqual(row['time_kind'], 'input_extent')
            self.assertEqual(row['speakers'], [])


if __name__ == '__main__':
    unittest.main()
