import unittest

from text_anchor_audit import audit


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
