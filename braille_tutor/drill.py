"""Per-symbol weighted target selection for the letters quiz mode.

Weight formula:   w(symbol) = (1 + 2 * misses) / (1 + correct)

Symbols with more misses get higher weight; symbols the learner has
answered correctly many times get lower weight.  The draw is weighted
random without replacement, with optional exclusion of the most-recently-
asked symbol from the first position so the same letter never leads two
consecutive quizzes.

This module is the fallback path inside TutorSession._fallback_items()
when AdaptiveTargetPlanner has no ready OpenAI plan.  It has no imports
from tutor.py and no side effects, so it can be unit-tested independently.
"""
from __future__ import annotations

import random
from typing import Optional


class LetterDrill:
    """Stateless weighted random target selector.

    Instantiate once (or use the module-level singleton ``drill``) and call
    ``pick()`` each time a new quiz is started.
    """

    @staticmethod
    def weight(misses: int, correct: int) -> float:
        """(1 + 2*misses) / (1 + correct) — miss-weighted, correct-discounted."""
        return (1 + 2 * misses) / (1 + correct)

    def pick(
        self,
        candidates: list[str],
        count: int,
        stats: dict[str, dict],
        rng: random.Random,
        exclude_first: Optional[str] = None,
    ) -> list[str]:
        """Weighted random draw of ``count`` symbols from ``candidates`` without replacement.

        Args:
            candidates:    All symbols eligible to be asked this quiz.
            count:         How many to return; capped silently at len(candidates).
            stats:         Per-symbol history as ``{symbol: {"misses": int, "correct": int}}``.
                           Symbols absent from the dict are treated as 0/0 (weight = 1.0).
            rng:           The session's own ``random.Random`` instance (keeps the session
                           seed deterministic; do not pass ``random`` directly).
            exclude_first: If set and the pool has more than one symbol, this symbol is
                           skipped for the first draw only, so the same letter never opens
                           two consecutive quizzes back-to-back.

        Returns:
            A ``list[str]`` of exactly ``min(count, len(candidates))`` symbols in the
            order they should be asked.  Same type and shape as the old ``rng.sample()``
            return value.
        """
        pool = list(candidates)
        n = min(count, len(pool))
        if not pool or n <= 0:
            return []

        out: list[str] = []
        for i in range(n):
            # First pick: skip the immediately-prior quiz's leading symbol so the
            # learner never faces the same opening letter twice in a row.
            excluded = (
                {exclude_first}
                if i == 0 and exclude_first is not None and len(pool) - len(out) > 1
                else set()
            )
            eligible = [s for s in pool if s not in out and s not in excluded]
            if not eligible:            # exclusion would leave an empty set — drop it
                eligible = [s for s in pool if s not in out]
            if not eligible:
                break

            weights = [
                self.weight(
                    stats.get(s, {}).get("misses", 0),
                    stats.get(s, {}).get("correct", 0),
                )
                for s in eligible
            ]
            total = sum(weights)
            r = rng.random() * total
            acc = 0.0
            chosen = eligible[-1]       # guard against floating-point rounding
            for sym, w in zip(eligible, weights):
                acc += w
                if r <= acc:
                    chosen = sym
                    break
            out.append(chosen)

        return out


# Module-level singleton — import and use directly:
#   from drill import drill
#   items = drill.pick(candidates, count, stats, rng)
drill = LetterDrill()
