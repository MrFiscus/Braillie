import braillie.word_correction as wc


def make_redetect(words):
    """Build a redetect callback from a list of words; counts calls."""
    calls = {"n": 0}
    it = iter(words)

    def redetect():
        calls["n"] += 1
        return next(it)

    return redetect, calls


class TestIsValidWord:
    def test_valid(self):
        assert wc.is_valid_word("cap") is True
        assert wc.is_valid_word(" CAP ") is True

    def test_invalid(self):
        assert wc.is_valid_word("cax") is False
        assert wc.is_valid_word("") is False
        assert wc.is_valid_word("   ") is False


class TestReadMode:
    def test_exact(self):
        redetect, calls = make_redetect(["cap"])
        r = wc.correct_word_read_mode("cap", redetect=redetect)
        assert r.word == "cap" and r.corrected is False
        assert r.source == "exact" and r.attempts == 0
        assert calls["n"] == 0

    def test_redetect_first_try(self):
        redetect, calls = make_redetect(["cap"])
        r = wc.correct_word_read_mode("cax", redetect=redetect)
        assert r.word == "cap" and r.corrected is True
        assert r.source == "redetect" and r.attempts == 1
        assert calls["n"] == 1

    def test_redetect_second_try(self):
        redetect, calls = make_redetect(["cxp", "cap"])
        r = wc.correct_word_read_mode("cax", redetect=redetect)
        assert r.word == "cap" and r.corrected is True
        assert r.source == "redetect" and r.attempts == 2
        assert calls["n"] == 2

    def test_redetect_exhausted(self):
        redetect, calls = make_redetect(["cxp", "caq"])
        r = wc.correct_word_read_mode("cax", redetect=redetect)
        assert r.word == "cax" and r.corrected is False
        assert r.source == "none" and r.attempts == 2
        assert calls["n"] == 2

    def test_no_redetect(self):
        r = wc.correct_word_read_mode("cax")
        assert r.word == "cax" and r.corrected is False
        assert r.source == "none" and r.attempts == 0

    def test_max_redetects_zero(self):
        redetect, calls = make_redetect(["cap"])
        r = wc.correct_word_read_mode("cax", redetect=redetect, max_redetects=0)
        assert r.word == "cax" and r.attempts == 0
        assert calls["n"] == 0

    def test_redetect_raises(self):
        def boom():
            raise RuntimeError("camera gone")

        r = wc.correct_word_read_mode("cax", redetect=boom)
        assert r.word == "cax" and r.corrected is False
        assert r.source == "none" and r.attempts == 1


class TestQuizMode:
    def test_exact(self):
        redetect, calls = make_redetect(["cap"])
        r = wc.check_word_quiz_mode("cap", "cap", redetect=redetect)
        assert r.correct is True and r.corrected is False and r.attempts == 0
        assert calls["n"] == 0

    def test_redetect_match(self):
        redetect, _ = make_redetect(["cap"])
        r = wc.check_word_quiz_mode("cax", "cap", redetect=redetect)
        assert r.correct is True and r.corrected is True and r.attempts == 1

    def test_redetect_exhausted(self):
        redetect, calls = make_redetect(["cax", "cax"])
        r = wc.check_word_quiz_mode("cax", "cap", redetect=redetect)
        assert r.correct is False and r.corrected is False and r.attempts == 2
        assert calls["n"] == 2

    def test_wrong_word(self):
        redetect, _ = make_redetect(["dog"])
        r = wc.check_word_quiz_mode("dog", "cap", redetect=redetect)
        assert r.correct is False

    def test_case_and_whitespace(self):
        r = wc.check_word_quiz_mode(" CAP ", "cap")
        assert r.correct is True and r.corrected is False and r.attempts == 0
