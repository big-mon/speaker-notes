import unittest

from text_anchor_audit import audit, native_result_boundaries, supported_sentence_boundaries


def timed(text, start=0):
    return [{"text": text, "units": [
        {"text": character, "start": start + index, "end": start + index + 1}
        for index, character in enumerate(text)
    ]}]


def compact_timed(text):
    segments = timed(text)
    for unit in segments[0]["units"]:
        unit["start"] /= 10
        unit["end"] /= 10
    return segments


class TextAnchorAuditTests(unittest.TestCase):
    def test_punctuation_offsets_and_native_envelopes(self):
        result = audit(timed("皆さん、こんにちは。"), "皆さん こんにちは！")
        self.assertEqual(result["differences"], [])
        anchor = result["anchors"][0]
        self.assertEqual(anchor["apple_text"], "皆さん、こんにちは")
        self.assertEqual(anchor["qwen_text"], "皆さん こんにちは")
        self.assertEqual(anchor["apple_time_range"], [0, 9])
        self.assertEqual(result["qwen_text"], "皆さん こんにちは！")

    def test_repeated_greeting_is_not_arbitrarily_timed(self):
        result = audit(timed("よろしくお願いします。よろしくお願いします。"), "よろしくお願いします。")
        self.assertEqual(result["anchors"], [])
        self.assertIn("repeated_exact_phrase_ambiguous", [r["reason"] for r in result["rejected_anchors"]])
        self.assertIsNone(result["unanchored"]["qwen"][0]["time_range"])

    def test_context_disambiguates_repeated_phrase(self):
        text = "最初によろしくお願いします。その次もよろしくお願いします。"
        result = audit(timed(text), text)
        self.assertEqual(len(result["anchors"]), 1)
        self.assertEqual(result["differences"], [])

    def test_insertions_deletions_do_not_receive_times(self):
        result = audit(timed("会社の目的は街を守ることです。"), "まず会社の目的は守ることです。以上。")
        self.assertEqual({d["operation"] for d in result["differences"]}, {"insert", "delete"})
        self.assertTrue(all(d["qwen_time_range"] is None for d in result["differences"]))
        self.assertTrue(all(g["time_range"] is None for g in result["unanchored"]["qwen"]))
        self.assertIsNone(result["differences"][0]["neighboring_anchor_ids"]["before"])
        self.assertIsNone(result["differences"][-1]["neighboring_anchor_ids"]["after"])

    def test_names_numbers_and_negation_are_review_candidates(self):
        result = audit(timed("深津さんは会社を3つ作らないと話した。"),
                       "復活さんは会社を5つ作ると話した。", critical_terms=["深津"])
        signals = {flag for difference in result["differences"] for flag in difference["review_signals"]}
        self.assertIn("critical_term_difference", signals)
        self.assertIn("number_or_numeric_character_difference", signals)
        self.assertIn("negation_difference_candidate", signals)
        self.assertEqual(result["apple_text"], "深津さんは会社を3つ作らないと話した。")

    def test_missing_times_and_long_unit_are_not_interpolated(self):
        missing = audit([{"text": "目的を説明する", "start": 0, "end": 12}], "目的を説明する")
        self.assertIsNone(missing["anchors"][0]["apple_time_range"])
        long_unit = [{"text": "目的を説明する", "units": [{"text": "目的を説明する", "start": 4, "end": 20}]}]
        result = audit(long_unit, "まず目的を説明する")
        self.assertEqual(result["anchors"][0]["apple_time_range"], [4, 20])
        self.assertIsNone(result["differences"][0]["qwen_time_range"])

    def test_clip_midpoint_selection_does_not_duplicate_units(self):
        apple = timed("会社の目的です")
        left = audit(apple, "会社の", clip_start=0, clip_end=2.5)
        right = audit(apple, "目的です", clip_start=2.5, clip_end=7)
        self.assertEqual(left["apple_text"] + right["apple_text"], "会社の目的です")

    def test_nfkc_preserves_source_offsets(self):
        result = audit(timed("ﾊﾟｰﾄはＡＩ３つです"), "パートはAI3つです")
        self.assertEqual(result["differences"], [])
        self.assertEqual(result["anchors"][0]["apple_text"], "ﾊﾟｰﾄはＡＩ３つです")

    def test_decimal_and_negative_sign_are_not_erased(self):
        for apple, qwen in [("予算は3.5億円です", "予算は35億円です"),
                            ("変化は-3です", "変化は3です"),
                            ("割合は30%です", "割合は30です")]:
            result = audit(timed(apple), qwen)
            self.assertTrue(result["differences"])
            self.assertIn("number_or_numeric_character_difference", {
                flag for difference in result["differences"] for flag in difference["review_signals"]
            })

    def test_invalid_and_nonmonotonic_times_stay_untimed(self):
        apple = timed("会社の目的です")
        apple[0]["units"][3]["start"] = -1
        result = audit(apple, "会社の目的です")
        self.assertIsNone(result["anchors"][0]["apple_time_range"])
        apple = timed("会社の目的です")
        apple[0]["units"][3].update(start=10, end=11)
        result = audit(apple, "会社の目的です")
        self.assertEqual(result["anchors"][0]["timing_status"], "non_monotonic_source_times")

    def test_malformed_apple_units_fail_instead_of_dropping_text(self):
        with self.assertRaises(ValueError):
            audit([{"text": "原文", "units": [{"text": "違う文", "start": 0, "end": 1}]}], "原文")

    def test_sentence_boundary_uses_native_unit_end_inside_long_anchor(self):
        text = "これは最初の話ですこれは次の話です"
        proposals = supported_sentence_boundaries(timed(text), "これは最初の話です。これは次の話です。")
        self.assertEqual([p["candidate_apple_unit_end"] for p in proposals], [9, 17])
        self.assertTrue(all(p["requires_human_review"] for p in proposals))
        self.assertEqual(proposals[0]["apple_source_unit_time_range"], [8, 9])

    def test_unmatched_sentence_end_has_no_proposal(self):
        proposals = supported_sentence_boundaries(timed("会社の目的を説明します"), "会社の目的を説明しません。")
        self.assertEqual(proposals, [])

    def test_decimal_period_is_not_a_sentence_boundary(self):
        proposals = supported_sentence_boundaries(timed("費用は3.5億円です"), "費用は3.5億円です。")
        self.assertEqual(len(proposals), 1)

    def test_repeated_sentence_without_context_has_no_boundary(self):
        proposals = supported_sentence_boundaries(timed("よろしくお願いしますよろしくお願いします"), "よろしくお願いします。")
        self.assertEqual(proposals, [])

    def test_short_backchannel_does_not_borrow_previous_sentence_context(self):
        proposals = supported_sentence_boundaries(timed("会社の目的ですはい"), "会社の目的です。はい。")
        self.assertEqual(proposals, [])

    def test_omitted_apple_suffix_prevents_cut(self):
        apple = "これは順番だと思うんですねもう一個実は重要な話があります"
        qwen = "これは順番だと思う。もう一個実は重要な話があります。"
        proposals = supported_sentence_boundaries(timed(apple), qwen)
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]["qwen_sentence_text"], "もう一個実は重要な話があります。")
        self.assertTrue(all(p["candidate_apple_unit_end"] != len("これは順番だと思う") for p in proposals))

    def test_mismatched_next_sentence_prefix_prevents_cut(self):
        apple = "ここまでが最初の話ですさて続きの説明をします"
        qwen = "ここまでが最初の話です。ところで続きの説明をします。"
        candidates = supported_sentence_boundaries(timed(apple), qwen)
        proposals = [p for p in candidates if not p["requires_diarization_handover"]]
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]["boundary_support"], "both_normalized_texts_end")

    def test_omitted_terminal_suffix_prevents_terminal_cut(self):
        proposals = supported_sentence_boundaries(timed("これは順番だと思うんですね"), "これは順番だと思う。")
        self.assertEqual(proposals, [])

    def test_apple_punctuation_gap_is_allowed(self):
        apple = "ここまでが最初の話です。　（続きの説明をします。）"
        qwen = "ここまでが最初の話です。続きの説明をします。"
        proposals = supported_sentence_boundaries(timed(apple), qwen)
        self.assertEqual(len(proposals), 2)
        self.assertEqual(proposals[0]["boundary_support"], "two_sided_adjacent_exact_text")
        self.assertEqual(proposals[0]["candidate_apple_unit_end"], len("ここまでが最初の話です"))

    def test_substantial_question_answer_handover_remains_supported(self):
        question = "それとも結構もう佐藤さんにお任せみたいな"
        answer = "なんかやっぱりフェースによって徐々に変化してきた感じです"
        proposals = supported_sentence_boundaries(timed(question + answer), question + "。" + answer + "。")
        self.assertEqual(len(proposals), 2)
        self.assertEqual(proposals[0]["candidate_apple_unit_end"], len(question))
        self.assertIsNotNone(proposals[0]["following_anchor_id"])

    def test_short_next_sentence_does_not_borrow_later_sentence_context(self):
        text = "ここまでが最初の話ですはい続いて別の話です"
        candidates = supported_sentence_boundaries(timed(text), "ここまでが最初の話です。はい。続いて別の話です。")
        proposals = [p for p in candidates if not p["requires_diarization_handover"]]
        self.assertEqual([p["qwen_sentence_text"] for p in proposals], ["続いて別の話です。"])

    def test_qwen_only_prefix_restores_adjacent_apple_boundary(self):
        apple = "過去からの自分が影響されているコンテクスト"
        qwen = "過去からの。で自分が影響されているコンテクスト。"
        proposals = supported_sentence_boundaries(compact_timed(apple), qwen)
        first = proposals[0]
        self.assertEqual(first["qwen_sentence_text"], "過去からの。")
        self.assertEqual(first["boundary_support"], "qwen_only_prefix_gap")
        self.assertFalse(first["requires_diarization_handover"])
        self.assertEqual(first["boundary_uncertainty"]["qwen_text"], "で")
        self.assertIsNone(first["boundary_uncertainty"]["qwen_time_range"])
        self.assertEqual(first["boundary_uncertainty"]["apple_normalized_character_count"], 0)

    def test_qwen_only_response_is_preserved_as_untimed_gap(self):
        apple = "事業そのものは佐藤さんのパートで多分僕が見ている部分は土台です"
        qwen = "事業そのものは佐藤さんのパート。そうですね。で多分僕が見ている部分は土台です。"
        first = supported_sentence_boundaries(compact_timed(apple), qwen)[0]
        self.assertEqual(first["boundary_support"], "qwen_only_prefix_gap")
        self.assertEqual(first["boundary_uncertainty"]["qwen_text"], "そうですね。")
        self.assertIsNone(first["boundary_uncertainty"]["qwen_time_range"])

    def test_bounded_name_substitution_requires_independent_handover(self):
        left = "彼女の担当している部分がすごく大きい"
        right = "さんの事業そのものは佐藤さんのパート"
        apple = left + "ん国王" + right
        qwen = left + "。だんだんそこ北尾" + right + "。"
        first = supported_sentence_boundaries(compact_timed(apple), qwen)[0]
        self.assertTrue(first["requires_diarization_handover"])
        self.assertFalse(first["standalone_automatic_boundary_selection_allowed"])
        self.assertEqual(first["candidate_apple_unit_end"], len(left) / 10)
        gap = first["boundary_uncertainty"]
        self.assertEqual(gap["apple_text"], "ん国王")
        self.assertEqual(gap["qwen_text"], "だんだんそこ北尾")
        self.assertEqual(gap["native_endpoint_envelope"], [len(left) / 10, (len(left) + 3) / 10])
        self.assertEqual(gap["right_reagreement_cut_candidate"]["candidate_apple_unit_end"], (len(left) + 3) / 10)
        self.assertFalse(gap["right_reagreement_cut_candidate"]["automatically_selected"])

    def test_pure_apple_suffix_is_not_unconditionally_extended(self):
        left = "これは順番だと思う"
        apple = left + "んですねもう一個実は重要な話があります"
        qwen = left + "。もう一個実は重要な話があります。"
        first = supported_sentence_boundaries(compact_timed(apple), qwen)[0]
        self.assertTrue(first["requires_diarization_handover"])
        self.assertEqual(first["candidate_apple_unit_end"], len(left) / 10)
        self.assertEqual(first["boundary_uncertainty"]["apple_text"], "んですね")
        self.assertEqual(first["boundary_uncertainty"]["qwen_normalized_character_count"], 0)
        self.assertEqual(first["boundary_uncertainty"]["right_reagreement_cut_candidate"]["candidate_apple_unit_end"], (len(left) + 4) / 10)

    def test_fallback_rejects_excessive_character_or_native_time_gap(self):
        left = "ここまでが最初の話です"
        right = "続いて別の説明をします"
        cases = [
            (compact_timed(left + right), left + "。" + "あ" * 11 + right + "。"),
            (compact_timed(left + "あ" * 11 + right), left + "。" + right + "。"),
            (timed(left + "あいう" + right), left + "。" + right + "。"),
        ]
        for apple, qwen in cases:
            proposals = supported_sentence_boundaries(apple, qwen)
            self.assertFalse(any(p["qwen_sentence_text"] == left + "。" for p in proposals))

    def test_fallback_requires_native_time_for_every_apple_gap_unit(self):
        left = "ここまでが最初の話です"
        apple = compact_timed(left + "あいう続いて別の説明をします")
        apple[0]["units"][len(left) + 1]["start"] = None
        proposals = supported_sentence_boundaries(apple, left + "。続いて別の説明をします。")
        self.assertFalse(any(p["qwen_sentence_text"] == left + "。" for p in proposals))

    def test_native_result_fallback_is_opt_in_and_preserves_result_boundaries(self):
        apple = timed("最初の結果", start=0) + timed("次の結果", start=10)
        qwen = "最初の結果から次の結果へ句読点なしで続く"
        self.assertEqual(supported_sentence_boundaries(apple, qwen), [])
        proposals = supported_sentence_boundaries(apple, qwen, include_native_fallback=True)
        self.assertEqual([p["candidate_apple_unit_end"] for p in proposals], [5, 14])
        self.assertEqual([p["apple_last_matched_raw_span"][1] for p in proposals], [5, 9])
        self.assertTrue(all(p["kind"] == "native_asr_result_boundary" for p in proposals))
        self.assertTrue(all(p["semantic_verified"] is False and p["anchor_id"] is None for p in proposals))
        self.assertTrue(all(p["requires_human_review"] for p in proposals))

    def test_native_fallback_does_not_replace_available_anchored_boundaries(self):
        apple = timed("ここまでが最初の話です続いて別の話です")
        proposals = supported_sentence_boundaries(apple, "ここまでが最初の話です。続いて別の話です。",
                                                  include_native_fallback=True)
        self.assertTrue(proposals)
        self.assertFalse(any(p["boundary_support"] == "native_asr_result_boundary" for p in proposals))

    def test_native_fallback_never_cuts_partial_result_at_arbitrary_clip_end(self):
        apple = timed("最初の結果", start=0) + timed("次の結果", start=10)
        proposals = native_result_boundaries(apple, clip_start=0, clip_end=12)
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]["candidate_apple_unit_end"], 5)

    def test_native_fallback_requires_a_real_final_unit_time(self):
        apple = timed("最初の結果", start=0)
        apple[0]["units"][-1]["end"] = None
        self.assertEqual(native_result_boundaries(apple), [])
        self.assertEqual(native_result_boundaries([{"text": "原文だけ", "start": 0, "end": 10}]), [])

    def test_large_qwen_only_span_is_review_signal_not_established_omission(self):
        apple = timed("はい。会社の説明をします。")
        qwen = "この会社は何を目指していてどんなメッセージを出しているかを教えてください。会社の説明をします。"
        result = audit(apple, qwen)
        flagged = [d for d in result["differences"] if "potential_substantive_omission" in d["review_signals"]]
        self.assertTrue(flagged)
        for difference in flagged:
            evidence = difference["review_signal_evidence"]["potential_substantive_omission"]
            self.assertFalse(evidence["error_confirmed"])
            self.assertFalse(evidence["qwen_correctness_verified"])
            self.assertTrue(evidence["requires_audio_review"])
            self.assertIsNone(difference["qwen_time_range"])
        self.assertEqual(result["apple_text"], "はい。会社の説明をします。")
        self.assertEqual(result["qwen_text"], qwen)

    def test_substantive_omission_threshold_does_not_flag_small_or_two_long_spans(self):
        for apple, qwen in [("甲", "乙" * 20), ("甲" * 6, "乙" * 30)]:
            result = audit(timed(apple), qwen)
            self.assertFalse(any("potential_substantive_omission" in d["review_signals"]
                                 for d in result["differences"]))


if __name__ == "__main__":
    unittest.main()
