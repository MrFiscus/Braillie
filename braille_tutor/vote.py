"""Temporal voting: pool detections from the last few scans so labels stop flickering from frame to frame."""
from __future__ import annotations

import collections
from collections import deque

import numpy as np

from detect import Cell, _assign_grid, _make_cell, dots_to_char


def keep_inside(cells: list, w_mm: float, h_mm: float) -> list:
    """Drop cells whose centre is off the page (the receipt, the binder, a neighbouring page)."""
    return [c for c in cells if 0 <= c["x"] <= w_mm and 0 <= c["y"] <= h_mm]


def first_rows(cells: list, n: int) -> list:
    """Keep only the first n rows (lines of braille) of a row-numbered cell list."""
    return [c for c in cells if c["row"] < n]


class CellVoter:
    """Collects cells (in page mm) from the last `frames` scans; result() gives one stable cell per position.

    A cell must show up in at least `min_presence` of the scans to count, which removes one-frame ghosts. Its label
    is the one with the most total confidence; its confidence is (share of votes for that label) x (share of scans
    it appeared in).
    """

    def __init__(self, frames: int = 8, min_presence: float = 0.4):
        self.history: deque = deque(maxlen=frames)
        self.min_presence = min_presence

    def add(self, cells: list) -> None:
        """Record one scan's cells (page mm)."""
        self.history.append(cells)

    def reset(self) -> None:
        """Forget everything (the page moved or was lost)."""
        self.history.clear()

    def result(self) -> list:
        """One voted Cell per position, row/col numbered."""
        pool = [(i, c) for i, scan in enumerate(self.history) for c in scan]
        if not pool:
            return []
        radius = 0.4 * float(np.median([c["w"] for _, c in pool]))
        clusters: list = []
        for i, c in sorted(pool, key=lambda p: -p[1]["confidence"]):
            for cl in clusters:
                if np.hypot(c["x"] - cl[0][1]["x"], c["y"] - cl[0][1]["y"]) < radius:
                    cl.append((i, c))
                    break
            else:
                clusters.append([(i, c)])
        need = self.min_presence * len(self.history) if len(self.history) >= 3 else 0
        out = []
        for cl in clusters:
            scans = {i for i, _ in cl}
            if len(scans) < need:
                continue
            votes: collections.Counter = collections.Counter()
            for _, c in cl:
                votes[c["label"]] += c["confidence"]
            label, best = votes.most_common(1)[0]
            conf = (best / sum(votes.values())) * (len(scans) / len(self.history))
            out.append(_make_cell(float(np.mean([c["x"] for _, c in cl])), float(np.mean([c["y"] for _, c in cl])),
                                  float(np.mean([c["w"] for _, c in cl])), float(np.mean([c["h"] for _, c in cl])),
                                  label, conf))
        return _assign_grid(out)


class CellLocker:
    """Latches each cell's reading once it can be trusted, so camera shake and blurry frames cannot flicker it.

    Voting smooths a reading over a sliding window, so a run of bad frames still wins eventually. Locking is different: a cell
    that has been read right is HELD, and one wrong or missing reading does nothing. It only lets go if a different reading
    persists for `unlock_after` scans in a row (the sheet really changed: a dot was added, another sheet was put down).

    Two ways to decide a reading is trustworthy:
      known sheet    the expected pattern is given (update_known): a cell locks as soon as it reads as expected `lock_after` scans
                     running. This is "we know it is correct", the way the four markers used to hold the page.
      unknown page   (update): a cell locks once the same reading has made up `agree` of its recent scans, at least `lock_after` times.
    Cells are matched by position in page millimetres, so they stay put when the camera moves.
    """

    def __init__(self, lock_after: int = 2, unlock_after: int = 6, window: int = 10, agree: float = 0.7):
        self.lock_after, self.unlock_after, self.window, self.agree = lock_after, unlock_after, window, agree
        self.slots: list = []

    def reset(self) -> None:
        """Forget everything and read from scratch (the u key)."""
        self.slots = []

    def prelock(self, cells: list) -> None:
        """Start with these cells already locked, for example a first line someone has checked by eye against the page."""
        self.reset()
        for c in cells:
            self.slots.append(self._new(c, c["label"], locked=True))

    def _new(self, cell: dict, expected: Optional[str] = None, locked: bool = False) -> dict:
        return {"cell": cell, "hist": collections.deque(maxlen=self.window), "locked": cell["label"] if locked else None,
                "expected": expected, "run": 0, "other": None, "other_run": 0, "unseen": 0}

    def _follow(self, slot: dict, label: Optional[str]) -> None:
        """Move a slot's lock state on by one scan, given what was read (None = nothing seen)."""
        slot["unseen"] = 0 if label is not None else slot["unseen"] + 1
        if label is None:
            return  # a dropout says nothing: a locked cell stays locked
        slot["hist"].append(label)
        if slot["expected"] is not None:
            # Known printed sheet (and the photos that layout came from): the name is the sheet.
            # A finger covering a dot is a worse live reading, not a new page — never rename or unlock.
            if slot["locked"] is not None:
                return
            slot["run"] = slot["run"] + 1 if label == slot["expected"] else 0
            if slot["run"] >= self.lock_after:
                slot["locked"] = slot["expected"]
            return
        if slot["locked"] is not None:
            if label == slot["locked"]:
                slot["other"], slot["other_run"] = None, 0
                return
            slot["other_run"] = slot["other_run"] + 1 if slot["other"] == label else 1
            slot["other"] = label
            if slot["other_run"] >= self.unlock_after:  # a different reading, again and again: the page really changed
                slot["locked"], slot["other"], slot["other_run"], slot["run"] = None, None, 0, 0
            return
        top, count = collections.Counter(slot["hist"]).most_common(1)[0]  # unknown page: lock on a steady majority
        if count >= self.lock_after and count >= self.agree * len(slot["hist"]):
            slot["locked"] = top

    def update_known(self, observed: list, expected: list) -> list:
        """Add one scan of a known sheet: `observed` and `expected` are lists of Cells in the same order.

        Centres stay on the printed layout (the demo-sheet photos). Live observation under a covering finger
        used to drag boxes onto neighbours — especially on the denser words and lookalikes sheets — so the
        green ring looked right while grading looked at a shifted square. Names stay pinned either way."""
        if not self.slots or len(self.slots) != len(expected):
            self.slots = [self._new(e, e["label"]) for e in expected]
        for slot, o, e in zip(self.slots, observed, expected):
            # Keep the sheet's own centre; only refresh size if the layout carried one.
            slot["cell"] = {**slot["cell"], "x": e["x"], "y": e["y"],
                            **{k: e[k] for k in ("w", "h") if k in e}}
            self._follow(slot, o["label"])
        return self.result()

    def update(self, cells: list) -> list:
        """Add one scan of an unknown page (Cells in page mm)."""
        if not cells:
            return self.result()
        radius = 0.4 * float(np.median([c["w"] for c in cells]))
        seen = set()
        for c in cells:
            free = [(np.hypot(c["x"] - s["cell"]["x"], c["y"] - s["cell"]["y"]), i) for i, s in enumerate(self.slots) if i not in seen]
            dist, i = min(free, default=(np.inf, -1))
            if dist > radius:
                self.slots.append(self._new(c))
                i = len(self.slots) - 1
            seen.add(i)
            slot = self.slots[i]
            slot["cell"] = {**c, "x": 0.5 * (slot["cell"]["x"] + c["x"]), "y": 0.5 * (slot["cell"]["y"] + c["y"])}
            self._follow(slot, c["label"])
        for i, slot in enumerate(self.slots):
            if i not in seen:
                self._follow(slot, None)
        self.slots = [s for s in self.slots if s["locked"] is not None or s["unseen"] <= self.window]  # unlocked ghosts age out
        return self.result()

    def result(self) -> list:
        """Every cell as it should be shown now: locked cells with their held reading (flag `locked`), others as last read."""
        out = []
        for s in self.slots:
            cell = {**s["cell"], "locked": s["locked"] is not None}
            if s["locked"] is not None:
                dots = frozenset(i + 1 for i, ch in enumerate(s["locked"]) if ch == "1")
                cell = {**cell, "label": s["locked"], "dots": dots, "char": dots_to_char(dots), "confidence": 1.0}
            out.append(cell)
        return out

    @property
    def locked_count(self) -> int:
        return sum(s["locked"] is not None for s in self.slots)
