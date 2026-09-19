"""Build a simulated embossed page of real contracted English, for scoring the whole reading pipeline end to end."""
from __future__ import annotations

import sim_embossed as sim

# how each word is written in Grade 2 braille (dot-number strings), following the rules in contractions.py
SPELLING = {"the": ["2346"], "child": ["16"], "can": ["14"], "go": ["1245"], "with": ["23456"], "you": ["13456"],
            "and": ["12346"], "that": ["2345"], "but": ["12"], "do": ["145"], "not": ["1345"], "it": ["1346"], "so": ["234"],
            "will": ["2456"], "from": ["124"], "read": ["1235", "2", "145"], "book": ["12", "135", "135", "13"],
            "this": ["1456"], "thing": ["1456", "346"], "bread": ["12", "1235", "2", "145"],
            "children": ["16", "24", "123", "145", "1235", "26"], "sing": ["234", "346"],
            "friend": ["124", "1235", "24", "26", "145"], "letter": ["123", "15", "2345", "2345", "12456"],
            "shall": ["146"], "which": ["156"]}
SENTENCES = ["the child can go with you", "you can read that book", "but do not go from it", "this thing and that thing",
             "the children will sing", "you will read the letter", "shall the friend read bread", "which child can go so"]


def build(contrast: float = 28.0, sentences=SENTENCES, px_per_mm: float = 10.0):
    """(flat image, truth cells, truth words per line, page size in mm)."""
    rows, words = [], []
    for s in sentences:
        row = []
        for w in s.split():
            row += [frozenset(int(c) for c in k) for k in SPELLING[w]] + [None]
        rows.append(row[:-1])
        words.append(s.split())
    width = max(len(r) for r in rows)
    rows = [r + [None] * (width - len(r)) for r in rows]
    size = (10 + width * 6 + 14, 10 + len(rows) * 10 + 14)
    flat, truth = sim.render_relief(rows, 2.5, 6.0, 10.0, px_per_mm, size, contrast=contrast)
    return flat, truth, words, size


def words_right(lines: list, truth_words: list) -> tuple:
    """(right, total): words that match the truth at the same place."""
    right = total = 0
    for i, words in enumerate(truth_words):
        got = lines[i].split() if i < len(lines) else []
        for j, w in enumerate(words):
            total += 1
            right += j < len(got) and got[j] == w
    return right, total
