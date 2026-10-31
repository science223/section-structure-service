"""Jev tagger engine (paper_structure/jev.py): decoding helpers without a model, dispatch by config.json, and an optional
end-to-end check against a real checkpoint (set PAPER_STRUCTURE_JEV_MODEL to its folder)."""
import json
import os
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

from paper_structure import jev


class DecodeTests(unittest.TestCase):
    def test_pieces_split_digits_and_punctuation(self):
        text = "opics.A.3 Argument\n22.3 Opt"
        pieces = [m.group() for m in jev.PIECE.finditer(text)]
        self.assertEqual("".join(pieces), text)
        self.assertEqual(pieces[:6], ["opics", ".", "A", ".", "3", " Argument"])
        self.assertIn("2", pieces)

    @unittest.skipUnless(find_spec("torch"), "torch not installed")
    def test_viterbi_drops_a_stray_inside_tag(self):
        import torch
        lp = torch.log(torch.tensor([[0.6, 0.1, 0.3], [0.2, 0.1, 0.7], [0.9, 0.05, 0.05]]))
        self.assertEqual(lp.argmax(-1).tolist(), [jev.O, jev.I, jev.O])   # argmax would make a fragment
        self.assertEqual(jev.viterbi(lp), [jev.O, jev.O, jev.O])

    @unittest.skipUnless(find_spec("torch"), "torch not installed")
    def test_viterbi_keeps_a_begin_inside_run(self):
        import torch
        lp = torch.log(torch.tensor([[0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.8, 0.1, 0.1]]))
        self.assertEqual(jev.viterbi(lp), [jev.B, jev.I, jev.O])

    def test_spans_to_verbatim_headings(self):
        text = "intro.\n2.1. Compatibility conditions. In the\nRelated\nWork\nbody"
        offs = [(0, 5), (5, 6), (7, 8), (8, 9), (9, 10), (10, 11), (11, 25), (25, 36), (36, 37), (37, 40), (40, 44),
                (45, 52), (53, 57), (58, 62)]
        tags = [0, 0, 1, 2, 2, 2, 2, 2, 0, 0, 0, 1, 2, 0]
        hs = jev.headings_from_spans(text, jev.spans(offs, tags))
        self.assertEqual(hs, ["2.1. Compatibility conditions", "Related Work"])

    @unittest.skipUnless(find_spec("torch"), "torch not installed")
    def test_numpy_viterbi_matches_torch(self):
        import numpy as np
        import torch
        rng = np.random.default_rng(7)
        for n in (1, 2, 5, 40):
            lp = np.log(rng.dirichlet([1, 1, 1], size=n))
            self.assertEqual(jev.viterbi_np(lp), jev.viterbi(torch.tensor(lp)))

    def test_onnx_export_detected(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "config.json").write_text(json.dumps({"architectures": ["ModernBertForTokenClassification"]}))
            self.assertFalse(jev.is_jev_onnx(d))
            (Path(d) / "model.onnx").write_bytes(b"x")
            (Path(d) / "tokenizer.json").write_text("{}")
            self.assertTrue(jev.is_jev_onnx(d))

    def test_dispatch_by_config(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "config.json").write_text(json.dumps({"architectures": ["ModernBertForTokenClassification"]}))
            self.assertTrue(jev.is_jev_checkpoint(d))
            (Path(d) / "config.json").write_text(json.dumps({"architectures": ["Qwen2ForCausalLM"]}))
            self.assertFalse(jev.is_jev_checkpoint(d))
        self.assertFalse(jev.is_jev_checkpoint(Path(tempfile.gettempdir()) / "no_such_model_dir"))


@unittest.skipUnless(os.environ.get("PAPER_STRUCTURE_JEV_MODEL"), "set PAPER_STRUCTURE_JEV_MODEL to a Jev checkpoint")
class RealCheckpointTests(unittest.TestCase):
    def test_tags_a_small_document(self):
        from paper_structure.infer import get_engine
        eng = get_engine(os.environ["PAPER_STRUCTURE_JEV_MODEL"], "cpu")
        self.assertIsInstance(eng, jev.JevEngine)
        # one occurrence each: a heading repeated several times reads as a running header or contents line
        text = ("Abstract\nWe study how to find section headings in the plain text of scientific papers.\n"
                "1 Introduction\nReaders and agents navigate papers by their sections, so the headings matter. "
                "We tag every token of a chunk instead of generating the heading strings.\n"
                "2 Related Work\nPrior work extracts headings with rules or with generative models.\n"
                "2.1 Generative models\nThey write the headings out token by token and may change a character.\n"
                "References\n[1] A. Author. A paper about headings. 2024.")
        heads = [h for p in eng.iter_predictions(text) for h in p.predicted_headings]
        for h in ("1 Introduction", "2 Related Work", "2.1 Generative models", "References"):
            self.assertIn(h, heads)


@unittest.skipUnless(os.environ.get("PAPER_STRUCTURE_JEV_ONNX"), "set PAPER_STRUCTURE_JEV_ONNX to an ONNX Jev export")
class OnnxCheckpointTests(unittest.TestCase):
    def test_tags_a_small_document(self):
        from paper_structure.infer import get_engine
        eng = get_engine(os.environ["PAPER_STRUCTURE_JEV_ONNX"], "cpu")
        self.assertIsInstance(eng, jev.JevOnnxEngine)
        text = ("Abstract\nWe study how to find section headings in the plain text of scientific papers.\n"
                "1 Introduction\nReaders and agents navigate papers by their sections, so the headings matter. "
                "We tag every token of a chunk instead of generating the heading strings.\n"
                "2 Related Work\nPrior work extracts headings with rules or with generative models.\n"
                "2.1 Generative models\nThey write the headings out token by token and may change a character.\n"
                "References\n[1] A. Author. A paper about headings. 2024.")
        heads = [h for p in eng.iter_predictions(text) for h in p.predicted_headings]
        for h in ("1 Introduction", "2 Related Work", "2.1 Generative models", "References"):
            self.assertIn(h, heads)


if __name__ == "__main__":
    unittest.main()
