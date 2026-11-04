"""The fast locator DP (prefix minimum over earlier paths, 2026-11-04) must choose exactly what the original
all-pairs DP chose. The original is kept here as the reference."""

from __future__ import annotations

import contextlib
import io
import random
import unittest

from paper_structure.locate import (MAX_CANDIDATES_PER_HEADING, _ordered_candidates, find_all_positions,
                                    locate_headings)


def reference(full_text, headings, toc_skip_frac=0.05):
    """The all-pairs DP as it was before 11-04 (every candidate against every earlier path, full keys), returning
    what locate_headings returns: located (predicted_index, char_start, matched form) in order, and every unlocated
    (predicted_index, reason) sorted by index."""
    toc_skip_chars = int(len(full_text) * toc_skip_frac)

    def key(path):
        return (-len(path), sum(c.preference_rank for c in path), sum(c.position[0] for c in path),
                tuple((c.predicted_index, c.position[0], c.candidate_order) for c in path))

    states, best, missing, found = [], (), [], []
    for i, h in enumerate(headings):
        positions = find_all_positions(full_text, h)
        if not positions:
            missing.append(i)
            continue
        found.append(i)
        current = []
        for cand in _ordered_candidates(positions, i, toc_skip_chars):
            cand_best = (cand,)
            for prev in states:
                if prev[-1].position[0] >= cand.position[0]:
                    continue
                contender = prev + (cand,)
                if key(contender) < key(cand_best):
                    cand_best = contender
            current.append(cand_best)
            if key(cand_best) < key(best):
                best = cand_best
        states.extend(current)
    chosen = {c.predicted_index for c in best}
    unlocated = sorted([(i, "no_text_match") for i in missing]
                       + [(i, "dropped_for_monotonic_consistency") for i in found if i not in chosen])
    return [(c.predicted_index, c.position[0], c.position[1]) for c in best], unlocated


class TestFastDP(unittest.TestCase):
    def check(self, text, heads, frac=0.05, msg=""):
        with contextlib.redirect_stderr(io.StringIO()):
            located, unlocated = locate_headings(text, heads, toc_skip_frac=frac)
        want_located, want_unlocated = reference(text, heads, frac)
        # located carries no index; tie it to the reference's choice through the unlocated indices and the titles
        self.assertEqual([(u.predicted_index, u.reason) for u in unlocated], want_unlocated, msg)
        kept = [i for i in range(len(heads)) if i not in {u.predicted_index for u in unlocated}]
        self.assertEqual([(i, h.char_start, h.matched_title) for i, h in zip(kept, located)], want_located, msg)
        self.assertEqual([h.predicted_title for h in located], [heads[i] for i in kept], msg)

    def test_matches_the_all_pairs_dp_on_random_documents(self):
        rng = random.Random(20261104)
        words = ["Intro", "Method", "Results", "Data", "Notes", "Summary", "Usage", "Options"]
        for trial in range(400):
            lines = []
            for _ in range(rng.randint(3, 40)):
                w = rng.choice(words)
                num = f"{rng.randint(1, 4)}.{rng.randint(1, 3)} " if rng.random() < 0.4 else ""
                lines.append(num + w)   # headings repeat, as running headers and contents lines do
                lines.append(" ".join(rng.choice(words).lower() for _ in range(rng.randint(0, 8))))
            text = "\n".join(lines) + "\n"
            heads = [rng.choice(lines[::2]) if rng.random() < 0.9 else "Missing heading"
                     for _ in range(rng.randint(1, 25))]
            self.check(text, heads, rng.choice([0, 0.05, 0.3, 0.7, 0.99]), f"trial {trial}")

    def test_edge_cases(self):
        self.check("", [])
        self.check("", ["A"])
        self.check("A\n", [])
        self.check("A\nA\nA\n", ["A"] * 4, frac=0)   # one more heading than places: index 3 is dropped
        self.check("AB\nA\nx\n", ["AB", "A"])        # headings that start at the same character
        self.check("A\nx\nA\n", ["A", "A"], frac=0.99)

    def test_more_candidates_than_the_cap(self):
        n = MAX_CANDIDATES_PER_HEADING + 10
        text = "\n".join(["Notes", "body"] * n) + "\n"
        self.check(text, ["Notes"] * 5)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            locate_headings(text, ["Notes"])
        self.assertIn(f"0:{n}->{MAX_CANDIDATES_PER_HEADING}", err.getvalue())


if __name__ == "__main__":
    unittest.main()
