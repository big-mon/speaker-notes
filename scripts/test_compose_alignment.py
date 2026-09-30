import copy
import unittest

from align_qwen import map_tokens
from compose_alignment import compose_alignments


def part(start, end, text, words, offset=100.):
    mapping = map_tokens(text, words)
    step = (end - start) / len(words)
    items = []
    for index, token in enumerate(mapping['tokens']):
        pair = [index * step, (index + 1) * step]
        source = [offset + start + v for v in pair]
        items.append({'index': index, 'text': token['text'],
                      'original_character_spans': token['original_character_spans'],
                      'raw_clip_seconds': pair, 'repaired_clip_seconds': pair.copy(),
                      'raw_source_seconds': source, 'repaired_source_seconds': source.copy(),
                      'library_repaired': False, 'flags': [],
                      'eligible_for_boundary_candidate': True})
    return {'start': start, 'end': end, 'source': {'sha256': 'example'}, 'alignment': {
        'schema_version': 1, 'mapping': mapping, 'items': items,
        'source_offset_seconds': offset + start, 'duration_seconds': end - start}}


class CompositionTests(unittest.TestCase):
    def test_text_offsets_times_and_raw_children_preserved(self):
        parts = [part(0, 3, '費用3.5円。\r\n', ['費用', '35', '円']),
                 part(3, 6, ' 次です？！', ['次', 'です'])]
        before = copy.deepcopy(parts)
        result = compose_alignments(parts, 100, 6)
        original = parts[0]['alignment']['mapping']['original_text']
        self.assertEqual(result['mapping']['original_text'], original + '\n 次です？！')
        item = result['items'][3]
        self.assertEqual(item['index'], 3)
        self.assertEqual(item['original_token_index'], 0)
        self.assertEqual(item['alignment_part_index'], 1)
        self.assertEqual(item['original_part_character_spans'], [[1, 2]])
        self.assertEqual(item['original_character_spans'], [[len(original) + 2, len(original) + 3]])
        self.assertEqual(item['raw_clip_seconds'], [3., 4.5])
        self.assertEqual(item['raw_source_seconds'], [103., 104.5])
        self.assertEqual(item['repaired_source_seconds'], [103., 104.5])
        self.assertEqual(parts, before)
        self.assertEqual(result['raw_children'], before)
        result['raw_children'][0]['alignment']['items'][0]['text'] = 'changed'
        self.assertEqual(parts, before)

    def test_only_internal_seam_tokens_get_edge_marker(self):
        parts = [part(0, 3, '一二三。', list('一二三')),
                 part(3, 6, '四五六。', list('四五六'))]
        result = compose_alignments(parts, 100, 6)
        marked = [i['index'] for i in result['items'] if 'alignment_chunk_edge' in i['flags']]
        self.assertEqual(marked, [2, 3])
        self.assertEqual([i['eligible_for_boundary_candidate'] for i in result['items']],
                         [True, True, False, False, True, True])
        single = compose_alignments(parts[:1], 100, 3)
        self.assertTrue(all('alignment_chunk_edge' not in i['flags'] for i in single['items']))

    def test_contiguity_and_child_extents_are_required(self):
        original = [part(0, 2, '一。', ['一']), part(2, 4, '二。', ['二'])]
        for change in ('first', 'gap', 'overlap', 'reversed', 'duration', 'offset', 'end', 'nan'):
            with self.subTest(change=change):
                parts = copy.deepcopy(original)
                if change == 'first': parts[0]['start'] = .1
                elif change == 'gap': parts[1]['start'] = 2.1
                elif change == 'overlap': parts[1]['start'] = 1.9
                elif change == 'reversed': parts[1]['end'] = 1
                elif change == 'duration': parts[1]['alignment']['duration_seconds'] = 3
                elif change == 'offset': parts[1]['alignment']['source_offset_seconds'] = 100
                elif change == 'end': parts[1]['end'] = 5; parts[1]['alignment']['duration_seconds'] = 3
                else: parts[1]['start'] = float('nan')
                with self.assertRaises(ValueError): compose_alignments(parts, 100, 4)

    def test_nonfinite_evidence_is_preserved_without_fabricated_source_time(self):
        parts = [part(0, 2, '一。', ['一']), part(2, 4, '二。', ['二'])]
        token = parts[1]['alignment']['items'][0]
        token.update(raw_clip_seconds=['nan', 2.], raw_source_seconds=None,
                     flags=['raw_nonfinite_timestamp'], eligible_for_boundary_candidate=False)
        result = compose_alignments(parts, 100, 4)
        item = result['items'][1]
        self.assertEqual(item['raw_clip_seconds'], ['nan', 4.])
        self.assertEqual(item['original_part_raw_clip_seconds'], ['nan', 2.])
        self.assertIsNone(item['raw_source_seconds'])
        self.assertEqual(item['repaired_source_seconds'], [102., 104.])
        self.assertIn('raw_nonfinite_timestamp', item['flags'])

    def test_inconsistent_source_times_fail_instead_of_double_adding_offset(self):
        for kind in ('raw', 'repaired'):
            p = part(0, 2, '一。', ['一'])
            p['alignment']['items'][0][kind + '_source_seconds'] = [0, 2]
            with self.assertRaisesRegex(ValueError, 'Full-source time'):
                compose_alignments([p], 100, 2)

    def test_repair_gate_remains_local_even_when_global_fraction_is_small(self):
        parts = [part(0, 10, '語' * 10, ['語'] * 10), part(10, 100, '文' * 90, ['文'] * 90)]
        for token in parts[0]['alignment']['items'][:2]:
            token.update(library_repaired=True, flags=['timestamp_repaired_by_library'],
                         eligible_for_boundary_candidate=False)
        result = compose_alignments(parts, 100, 100)
        self.assertEqual(sum(i['library_repaired'] for i in result['items']), 2)
        self.assertTrue(result['composition_parts'][0]['repair_gate'])
        self.assertFalse(result['composition_parts'][1]['repair_gate'])
        self.assertTrue(all('alignment_chunk_repair_gate' in i['flags'] for i in result['items'][:10]))
        self.assertTrue(all(not i['eligible_for_boundary_candidate'] for i in result['items'][:10]))
        self.assertFalse(any('alignment_chunk_repair_gate' in i['flags'] for i in result['items'][10:]))
        parts[0]['alignment']['items'][1]['library_repaired'] = False
        boundary = compose_alignments(parts, 100, 100)
        self.assertFalse(boundary['composition_parts'][0]['repair_gate'])


if __name__ == '__main__':
    unittest.main()
