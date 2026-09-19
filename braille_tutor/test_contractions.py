"""Tests for contractions.py (Grade 2 English braille) and how the tutor uses it.

Most tests pass a tiny fixed dictionary so they don't depend on a spell-checker; a few use the real one when installed.
"""
import unittest

import contractions as ct
from contractions import decode_word, decode_words

WORDS = {"but", "can", "do", "the", "and", "you", "it", "as", "that", "thing", "children", "bread", "discover", "connect",
         "beside", "what", "braille", "this", "shall", "which", "sing", "letter", "was", "his", "to", "in", "enough", "be",
         "for", "of", "with", "child", "out", "still", "sand", "father", "eat", "bell", "dog", "cat", "a", "i", "hello", "don't", "rabbit"}
is_word = WORDS.__contains__


def d(text):
    return decode_word(text.split(), is_word)


class WordsignTests(unittest.TestCase):
    def test_alphabetic_wordsigns_when_alone(self):
        for cell, word in [("12", "but"), ("14", "can"), ("145", "do"), ("13456", "you"), ("1346", "it"), ("1356", "as"),
                           ("2345", "that")]:
            self.assertEqual(d(cell).text, word, cell)

    def test_strong_and_lower_wordsigns(self):
        for cell, word in [("12346", "and"), ("123456", "for"), ("12356", "of"), ("2346", "the"), ("23456", "with"),
                           ("16", "child"), ("146", "shall"), ("1456", "this"), ("156", "which"), ("1256", "out"),
                           ("34", "still"), ("35", "in"), ("26", "enough"), ("23", "be"), ("235", "to"), ("356", "was"),
                           ("236", "his")]:
            self.assertEqual(d(cell).text, word, cell)

    def test_a_and_i_are_words_on_their_own(self):
        self.assertEqual(d("1").text, "a")
        self.assertEqual(d("24").text, "i")

    def test_a_wordsign_needs_the_cell_to_be_alone(self):
        r = d("12 1235")  # b then r is not "but" + "rather": in a longer word the cells are letters
        self.assertEqual(r.text, "br")
        self.assertFalse(r.valid)


class GroupsignTests(unittest.TestCase):
    def test_strong_groupsigns_inside_words(self):
        self.assertEqual(d("1456 346").text, "thing")           # th + ing
        self.assertEqual(d("234 12346").text, "sand")            # s + and
        self.assertEqual(d("124 1 2346 1235").text, "father")    # f a the r
        self.assertEqual(d("16 24 123 145 1235 26").text, "children")  # ch i l d r en

    def test_ing_can_never_start_a_word(self):
        # 346 has no legal reading in first position, so it stays visible as "?" instead of becoming "ing"
        self.assertTrue(all(t.startswith("?") for t, _ in ct._readings(["346", "234"])))
        self.assertEqual(d("234 346").text, "sing")                     # s + ing is fine: ing is not first

    def test_lower_groupsigns_only_between_cells(self):
        mid = [t for t, _ in ct._readings(["12", "1235", "2", "145"])]
        self.assertIn("bread", mid)                              # ea in the middle
        first = [t for t, _ in ct._readings(["2", "1235"])]
        self.assertNotIn("earr", first)                          # never first
        last = [t for t, _ in ct._readings(["1235", "2"])]
        self.assertNotIn("rea", last)                            # never last
        self.assertEqual(d("1235 1 23 24 2345").text, "rabbit")   # bb between letters
        self.assertNotIn("bbb", [t for t, _ in ct._readings(["23", "12"])])  # bb never first

    def test_initial_groupsigns_only_at_the_start(self):
        self.assertEqual(d("256 14 135 1236 15 1235").text, "discover")   # dis
        self.assertEqual(d("25 1345 15 14 2345").text, "connect")          # con
        self.assertEqual(d("23 234 24 145 15").text, "beside")             # be
        self.assertNotIn("dis", [t for t, _ in ct._readings(["14", "256"])])  # 256 is not "dis" after the first cell

    def test_er_at_the_end(self):
        self.assertEqual(d("123 15 2345 2345 12456").text, "letter")


class SignTests(unittest.TestCase):
    def test_capitals(self):
        self.assertEqual(d("6 12 1235 1 24 123 123 15").text, "Braille")
        self.assertEqual(d("6 6 2346").text, "THE")

    def test_numbers(self):
        self.assertEqual(d("3456 1 12 14").text, "123")
        self.assertEqual(d("3456 245").text, "0")
        self.assertEqual(d("3456 1 245").text, "10")

    def test_trailing_punctuation(self):
        self.assertEqual(d("2346 2").text, "the,")
        self.assertEqual(d("2456 125 1 2345 236").text, "what?")
        self.assertEqual(d("125 15 123 123 135 235").text, "hello!")
        self.assertEqual(d("12 2").text, "but,")     # a wordsign with punctuation still counts as a wordsign

    def test_comma_is_not_read_as_ea_at_the_end(self):
        self.assertEqual(d("145 135 1245 2").text, "dog,")

    def test_apostrophe_inside_a_word(self):
        self.assertEqual(d("145 135 1345 3 2345").text, "don't")


class RobustnessTests(unittest.TestCase):
    def test_empty_and_unknown_cells(self):
        self.assertEqual(decode_word([], is_word).text, "")
        self.assertIn("?", decode_word(["123456789"], is_word).text)   # not a real cell: kept visible
        self.assertFalse(decode_word(["12", "1235", "12", "12"], is_word).valid)  # no dictionary word: says so

    def test_accepts_sets_of_dot_numbers(self):
        self.assertEqual(decode_word([{2, 3, 4, 6}], is_word).text, "the")
        self.assertEqual(decode_word([frozenset({1, 4, 5}), frozenset({1, 4, 5, 6})][:1], is_word).text, "do")

    def test_long_words_stay_fast(self):
        import time
        t0 = time.time()
        decode_word(["2", "23", "25", "235", "2356", "26", "35", "16"] * 3, is_word)   # every cell has several readings
        self.assertLess(time.time() - t0, 2.0)

    def test_valid_flag_and_alternatives(self):
        r = d("1456 346")
        self.assertTrue(r.valid)
        self.assertIn("thing", r.candidates)
        self.assertFalse(decode_word(["12", "12", "12", "12"], is_word).valid)

    def test_decode_words_helper(self):
        self.assertEqual(decode_words([["2346"], ["14", "1", "2345"]], is_word), ["the", "cat"])


@unittest.skipIf(ct.default_is_word() is None, "pyspellchecker not installed")
class WithTheRealDictionaryTests(unittest.TestCase):
    def test_common_words(self):
        for cells, word in [("12", "but"), ("1456 346", "thing"), ("16 24 123 145 1235 26", "children"),
                            ("12 1235 2 145", "bread"), ("256 14 135 1236 15 1235", "discover")]:
            r = decode_word(cells.split())
            self.assertEqual(r.text, word)
            self.assertTrue(r.valid)

    def test_plain_letters_still_win_when_no_contraction_makes_a_word(self):
        self.assertEqual(decode_word(["14", "1", "2345"]).text, "cat")


class ReaderIntegrationTests(unittest.TestCase):
    def test_decode_lines_reads_a_row_of_detected_cells(self):
        import reader
        from detect import _make_cell, _assign_grid
        # "the cat" laid out with a word gap: 2346 | 14 1 2345
        cells = [_make_cell(10.0 + i * 12.0 + (12.0 if i >= 1 else 0.0), 10.0, 6.0, 9.0, "000000", 1.0) for i in range(4)]
        labels = ["011101", "100100", "100000", "011110"]  # dots 2346, 14, 1, 2345 as 6-char labels
        cells = _assign_grid([{**c, "label": lab, "dots": frozenset(i + 1 for i, ch in enumerate(lab) if ch == "1")}
                              for c, lab in zip(cells, labels)])
        lines = reader.decode_lines(cells, is_word)
        self.assertEqual(lines, ["the cat"])


if __name__ == "__main__":
    unittest.main()
