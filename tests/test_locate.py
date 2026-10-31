from __future__ import annotations

import unittest

from paper_structure.locate import locate_headings


class LocateHeadingTests(unittest.TestCase):
    def test_max_retention_prevents_cascade_from_late_false_match(self):
        # Mirrors A3: the old greedy rule chose the preferred standalone copy
        # of "Trap" near the end, then discarded every real heading before it.
        text = (
            "front matter mentions Trap inline\n"
            "1 Introduction\n"
            "body\n"
            "2 Method\n"
            "body\n"
            "3 Results\n"
            "body\n"
            "Trap\n"
        )
        headings = ["Trap", "1 Introduction", "2 Method", "3 Results"]
        located, unlocated = locate_headings(text, headings, toc_skip_frac=0)

        self.assertEqual(unlocated, [])
        self.assertEqual(
            [item.predicted_title for item in located],
            headings,
        )
        self.assertEqual(located[0].char_start, text.find("Trap"))

    def test_failed_location_is_explicit(self):
        located, unlocated = locate_headings(
            "1 Introduction\nBody text.\n",
            ["1 Introduction", "2 Missing"],
            toc_skip_frac=0,
        )
        self.assertEqual([item.matched_title for item in located], ["1 Introduction"])
        self.assertEqual(len(unlocated), 1)
        self.assertEqual(unlocated[0].predicted_title, "2 Missing")
        self.assertEqual(unlocated[0].reason, "no_text_match")

    def test_unlocated_reasons_distinguish_no_match_from_monotonic_drop(self):
        text = "2 Later\nbody\n1 Earlier\n"
        located, unlocated = locate_headings(
            text,
            ["1 Earlier", "2 Later", "3 Missing"],
            toc_skip_frac=0,
        )

        self.assertEqual(
            [item.predicted_title for item in located],
            ["2 Later"],
        )
        self.assertEqual(
            [(item.predicted_title, item.reason) for item in unlocated],
            [
                ("1 Earlier", "dropped_for_monotonic_consistency"),
                ("3 Missing", "no_text_match"),
            ],
        )
        self.assertEqual(
            unlocated[0].candidate_positions,
            (text.find("1 Earlier"),),
        )

    def test_non_monotonic_match_retries_after_previous_heading(self):
        text = "Repeat\nfront matter\n1 First\nbody\nRepeat\nlater body\n"
        located, unlocated = locate_headings(
            text,
            ["1 First", "Repeat"],
            toc_skip_frac=0,
        )
        self.assertEqual(unlocated, [])
        self.assertEqual([item.matched_title for item in located], ["1 First", "Repeat"])
        self.assertGreater(located[1].char_start, located[0].char_start)
        self.assertEqual(located[1].char_start, text.rfind("Repeat"))

    def test_toc_skip_zero_selects_front_match(self):
        heading = "2 Method"
        filler = "x" * 200
        text = f"{heading}\n{filler}\n{heading}\nbody\n"
        with_skip, _ = locate_headings(text, [heading], toc_skip_frac=0.05)
        without_skip, _ = locate_headings(text, [heading], toc_skip_frac=0)
        self.assertEqual(with_skip[0].char_start, text.rfind(heading))
        self.assertEqual(without_skip[0].char_start, 0)

    def test_assignment_is_deterministic_across_repeated_runs(self):
        text = "Shared\n1 First\nShared\n2 Second\nShared\n"
        headings = ["Shared", "1 First", "Shared", "2 Second", "Shared"]

        first = locate_headings(text, headings, toc_skip_frac=0)
        second = locate_headings(text, headings, toc_skip_frac=0)

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
