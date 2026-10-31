from __future__ import annotations

import os
import unittest
from pathlib import Path

from paper_structure import build_document
from paper_structure.predictions import load_predicted_headings
from paper_structure.tree import infer_depth


class ApiTests(unittest.TestCase):
    def test_depth_rules(self):
        self.assertEqual(infer_depth("3 Method"), 1)
        self.assertEqual(infer_depth("3.1 Method"), 2)
        self.assertEqual(infer_depth("A.2 Details"), 2)
        self.assertEqual(infer_depth("3.1.2 Encoder"), 3)
        self.assertEqual(infer_depth("Abstract"), 1)
        self.assertEqual(infer_depth("Introduction"), 1)
        self.assertEqual(infer_depth("References"), 1)

    def test_ieee_numbering(self):
        # IEEE surveys (RAG for LLMs, 10-20): letters restart in every Roman section, so they were all depth 1 with
        # ids "A", "A#2", "A#3", ... and "II. OVERVIEW OF RAG" got a slug while "I. INTRODUCTION" got "I"
        from paper_structure.tree import outline_numbering
        titles = ["Abstract", "I. INTRODUCTION", "II. OVERVIEW", "A. Naive RAG", "B. Advanced RAG", "III. RETRIEVAL",
                  "A. Retrieval Source", "Sparse and dense embedding", "1) Fine-tuning Embedding Model: In",
                  "B. Indexing", "V. AUGMENTATION", "A. Iterative", "REFERENCES", "A. Proofs"]
        self.assertEqual(outline_numbering(titles), [
            (None, 1), ("I", 1), ("II", 1), ("II.A", 2), ("II.B", 2), ("III", 1), ("III.A", 2), (None, 3),
            ("III.A.1", 3), ("III.B", 2), ("V", 1), ("V.A", 2), (None, 1), ("A", 1)])   # IV missing: still Roman

    def test_ieee_back_matter_is_the_whole_title(self):
        # Grok review 10-20 (F5): "Supplemental Results" between two lettered subsections ended the Roman section
        from paper_structure.tree import outline_numbering
        out = outline_numbering(["I. INTRO", "II. METHOD", "A. Setup", "Supplemental Results", "B. Analysis",
                                 "Appendix A", "A. Proofs"])
        self.assertEqual(out[4], ("II.B", 2))
        self.assertEqual(out[6], ("A", 1))   # after "Appendix A" the letters are the appendix's own
        out = outline_numbering(["I. INTRO", "II. METHOD", "A. Setup", "Abstract Interpretation", "B. Analysis"])
        self.assertEqual(out[4], ("II.B", 2))   # a title that only starts with a back-matter word

    def test_ieee_heading_cut_after_its_letter(self):
        # RAG survey: the model's heading was just "B." (the title went on on the next line); it became "B", depth 1
        from paper_structure.tree import outline_numbering
        self.assertEqual(outline_numbering(["I. INTRO", "II. GENERATION", "A. Context Curation", "B."])[3], ("II.B", 2))

    def test_ieee_letter_i_after_h_is_a_letter(self):
        from paper_structure.tree import outline_numbering
        titles = ["I. INTRODUCTION", "II. METHODS"] + [f"{c}. Part {c}" for c in "ABCDEFGHI"] + ["III. RESULTS"]
        out = outline_numbering(titles)
        self.assertEqual(out[10], ("II.I", 2))
        self.assertEqual(out[11], ("III", 1))

    def test_other_layouts_number_as_before(self):
        from paper_structure.tree import numbering_token, outline_numbering
        for titles in (["1 Intro", "2 Method", "2.1 Data", "A Appendix", "A.1 More", "Results"],
                       ["I. Only one Roman heading", "A. Details", "B. More"],      # one Roman: not IEEE
                       ["II. Starts at two", "IV. Skips one too many", "A. x"]):   # II then IV is in order (one gap)
            want = [(numbering_token(t), infer_depth(t)) for t in titles]
            if titles[0].startswith("II."):
                want = [("II", 1), ("IV", 1), ("IV.A", 2)]
            self.assertEqual(outline_numbering(titles), want, titles)

    def test_ieee_sections_get_parents_and_subsections(self):
        text = ("I. INTRODUCTION\nintro\nII. METHOD\nmethod\nA. Data\ndata\nB. Model\nmodel\n"
                "III. RESULTS\nresults\nA. Data\nmore data\n")
        document = build_document(text, ["I. INTRODUCTION", "II. METHOD", "A. Data", "B. Model", "III. RESULTS",
                                         "A. Data"], toc_skip_frac=0)
        self.assertEqual([s.id for s in document.sections], ["I", "II", "II.A", "II.B", "III", "III.A"])
        self.assertEqual(document.get_section("III.A").parent_id, "III")
        self.assertIn("B. Model", document.read_section("II", include_descendants=True))
        self.assertNotIn("III. RESULTS", document.read_section("II", include_descendants=True))

    def test_exact_slices_reconstruct_structured_interval(self):
        text = "preamble\n1 Intro\nalpha\n1.1 Detail\nbeta\n2 End\ngamma\n"
        document = build_document(
            text,
            ["1 Intro", "1.1 Detail", "2 End"],
            toc_skip_frac=0,
        )
        reconstructed = "".join(
            document.read_section(section.id) for section in document.sections
        )
        self.assertEqual(
            reconstructed,
            text[document.sections[0].char_start:document.sections[-1].char_end],
        )

    def test_nested_read_parent_children_and_search(self):
        text = (
            "1 Intro\nchapter intro\n"
            "1.1 Setup\nchild needle\n"
            "1.1.1 Detail\ndeep body\n"
            "1.2 Baselines\nsecond child\n"
            "2 Results\nresult body\n"
        )
        document = build_document(
            text,
            ["1 Intro", "1.1 Setup", "1.1.1 Detail", "1.2 Baselines", "2 Results"],
            toc_skip_frac=0,
        )

        direct = document.read_section("1")
        nested = document.read_section("1", include_descendants=True)
        self.assertNotIn("1.1 Setup", direct)
        self.assertIn("1.1 Setup", nested)
        self.assertIn("1.2 Baselines", nested)
        self.assertNotIn("2 Results", nested)
        self.assertEqual(document.children("1"), ["1.1", "1.2"])
        self.assertEqual(document.children("1.1"), ["1.1.1"])
        self.assertEqual(document.parent("1.1"), "1")
        self.assertEqual(document.parent("1.1.1"), "1.1")
        self.assertEqual(document.parent("1"), None)
        self.assertEqual(document.search_in_section("1", "needle"), [])
        self.assertEqual(
            len(document.search_in_section(
                "1", "needle", include_descendants=True
            )),
            1,
        )

    def test_unnumbered_headings_do_not_nest(self):
        text = "Abstract\nsummary\nIntroduction\nopening\n"
        document = build_document(
            text,
            ["Abstract", "Introduction"],
            toc_skip_frac=0,
        )
        self.assertIsNone(document.parent("abstract"))
        self.assertIsNone(document.parent("introduction"))
        self.assertEqual(document.children("abstract"), [])
        self.assertEqual(document.children("introduction"), [])

    def test_to_dict_is_deterministic_and_includes_parent_id(self):
        text = "1 Intro\nbody\n1.1 Detail\nchild\n2 End\ndone\n"
        headings = ["1 Intro", "1.1 Detail", "2 End"]
        first = build_document(text, headings, toc_skip_frac=0).to_dict()
        second = build_document(text, headings, toc_skip_frac=0).to_dict()
        self.assertEqual(first, second)
        self.assertEqual(
            [section["parent_id"] for section in first["sections"]],
            [None, "1", None],
        )
        self.assertEqual(
            [section["has_empty_body"] for section in first["sections"]],
            [False, False, False],
        )

    def test_empty_body_flag_and_ordered_empty_sections(self):
        text = "1 Intro\n1.1 Detail\nbody\n2 Empty\n"
        document = build_document(
            text,
            ["1 Intro", "1.1 Detail", "2 Empty"],
            toc_skip_frac=0,
        )
        self.assertEqual(
            [section.has_empty_body for section in document.sections],
            [True, False, True],
        )
        self.assertEqual(document.empty_sections(), ["1", "2"])
        self.assertEqual(
            document.empty_sections(),
            [
                section.id
                for section in document.sections
                if section.has_empty_body
            ],
        )

    def test_a1_saved_predictions_preserve_section_slices(self):
        tool_root = Path(__file__).resolve().parents[1]
        text_relative = (
            Path("data") / "full_text" / "A1_Transformer_Attention.txt"
        )
        predictions_relative = (
            Path("data")
            / "predictions"
            / "v13_2"
            / "A1_Transformer_Attention.jsonl"
        )
        configured_root = os.environ.get("PAPER_STRUCTURE_BENCHMARK_DIR")
        if configured_root:
            benchmark = Path(configured_root).expanduser().resolve()
        else:
            candidates = [
                candidate
                for candidate in tool_root.parent.iterdir()
                if (candidate / text_relative).is_file()
                and (candidate / predictions_relative).is_file()
            ]
            if not candidates:
                self.skipTest(
                    "set PAPER_STRUCTURE_BENCHMARK_DIR to run the external "
                    "A1 saved-prediction regression"
                )
            self.assertEqual(
                len(candidates),
                1,
                "external A1 fixtures are ambiguous; set "
                "PAPER_STRUCTURE_BENCHMARK_DIR explicitly",
            )
            benchmark = candidates[0]

        text_path = benchmark / text_relative
        predictions_path = benchmark / predictions_relative
        text = text_path.read_text(encoding="utf-8")
        document = build_document(
            text,
            load_predicted_headings(predictions_path),
        )
        self.assertEqual(len(document.sections), 24)
        section = document.get_section("3")
        self.assertEqual(
            document.read_section("3").encode("utf-8"),
            text[section.char_start:section.char_end].encode("utf-8"),
        )

    def test_keyword_search_is_scoped(self):
        text = "1 Intro\nneedle one\n2 Method\nneedle two\n"
        document = build_document(text, ["1 Intro", "2 Method"], toc_skip_frac=0)
        hits = document.search_in_section("2", "needle")
        self.assertEqual(len(hits), 1)
        self.assertIn("needle two", hits[0].snippet)
        self.assertNotIn("needle one", hits[0].snippet)
        self.assertEqual(document.search_in_section("2", "needle", max_results=0), [])


if __name__ == "__main__":
    unittest.main()
