"""Tests for the 2026-10-19 polish: friendly CLI errors, single-line outline titles, short slugs, device option."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from paper_structure import UnknownSectionError, build_document, cli
from paper_structure._install import SOURCE
from paper_structure.api import Document, Section
from paper_structure.infer import resolve_device
from paper_structure.tree import section_id_base

TEXT = "Paper title\n\n1 Introduction\nWe study things.\n\n2 Method\nWe do things.\n"


class TempFiles(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.dir = Path(self._t.name)
        self.paper = self.dir / "paper.txt"
        self.paper.write_text(TEXT, encoding="utf-8")
        self.preds = self.dir / "preds.jsonl"
        self.preds.write_text(json.dumps({"chunk_id": 0, "predicted_headings": ["1 Introduction", "2 Method"]}) + "\n",
                              encoding="utf-8")

    def tearDown(self):
        self._t.cleanup()

    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cli.main(list(argv))
        return code, out.getvalue(), err.getvalue()


class TestCliErrors(TempFiles):
    def test_unknown_section_is_one_line(self):
        code, out, err = self.run_cli("read", str(self.paper), "--section", "3", "--headings-jsonl", str(self.preds))
        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertIn("error: Unknown section id '3'. Known ids: 1, 2", err)
        self.assertNotIn("Traceback", err)

    def test_unknown_section_in_search(self):
        code, _, err = self.run_cli("search", str(self.paper), "--section", "9", "-q", "x",
                                    "--headings-jsonl", str(self.preds))
        self.assertEqual(code, 2)
        self.assertIn("Unknown section id '9'", err)

    def test_missing_file(self):
        code, _, err = self.run_cli("outline", str(self.dir / "nope.txt"), "--headings-jsonl", str(self.preds))
        self.assertEqual(code, 2)
        self.assertIn("error:", err)

    def test_no_model_configured(self):
        with mock.patch.dict(os.environ, {"PAPER_STRUCTURE_MODEL": ""}):
            code, _, err = self.run_cli("outline", str(self.paper))
        self.assertEqual(code, 2)
        self.assertIn("model path is required", err)

    def test_valid_read_unchanged(self):
        code, out, err = self.run_cli("read", str(self.paper), "--section", "1", "--headings-jsonl", str(self.preds))
        self.assertEqual(code, 0)
        self.assertTrue(out.startswith("1 Introduction"))

    def test_device_choices(self):
        with self.assertRaises(SystemExit):
            with redirect_stderr(io.StringIO()):
                cli.build_parser().parse_args(["outline", "x.txt", "--device", "tpu"])
        a = cli.build_parser().parse_args(["outline", "x.txt", "--device", "cpu"])
        self.assertEqual(a.device, "cpu")


class TestDisplay(unittest.TestCase):
    def test_outline_title_is_single_line(self):
        doc = Document(sections=[Section("x", "Inference When generating\nfinetuning  done", 1, 0, 10)],
                       unlocated=[], _text="a" * 10)
        self.assertEqual(doc.outline(), ["x\tInference When generating finetuning done"])
        self.assertEqual(doc.sections[0].title, "Inference When generating\nfinetuning  done")   # exact text kept

    def test_slug_is_short(self):
        self.assertEqual(section_id_base("Model Finetuning After sampling and filtering, then more words", 1),
                         "model-finetuning-after-sampling-and-filtering")
        self.assertEqual(section_id_base("Abstract", 1), "abstract")
        self.assertEqual(section_id_base("4.3 Results", 1), "4.3")
        self.assertEqual(section_id_base("!!!", 7), "section-7")

    def test_short_slugs_stay_unique(self):
        text = "x\nAlpha beta gamma delta epsilon zeta one\nbody\nAlpha beta gamma delta epsilon zeta two\nbody\n"
        doc = build_document(text, ["Alpha beta gamma delta epsilon zeta one", "Alpha beta gamma delta epsilon zeta two"],
                             toc_skip_frac=0)
        ids = [s.id for s in doc.sections]
        self.assertEqual(len(ids), len(set(ids)), ids)

    def test_unknown_section_error_is_keyerror(self):
        doc = build_document(TEXT, ["1 Introduction"], toc_skip_frac=0)
        with self.assertRaises(KeyError):
            doc.read_section("zzz")
        try:
            doc.read_section("zzz")
        except UnknownSectionError as exc:
            self.assertEqual(str(exc), "Unknown section id 'zzz'. Known ids: 1")


class TestReviewFollowUps(TempFiles):
    """Gaps found by the read-only review of the polish (2026-10-19)."""

    def test_device_reaches_recover_structure(self):
        seen = {}

        def fake(text, **kw):
            seen.update(kw)
            return build_document(text, ["1 Introduction"], toc_skip_frac=0)

        with mock.patch.object(cli, "recover_structure", fake):
            code, _, _ = self.run_cli("outline", str(self.paper), "--model", "m", "--device", "cpu")
        self.assertEqual(code, 0)
        self.assertEqual(seen["device"], "cpu")

    def test_device_with_headings_file_is_noted(self):
        code, _, err = self.run_cli("outline", str(self.paper), "--headings-jsonl", str(self.preds), "--device", "cpu")
        self.assertEqual(code, 0)
        self.assertIn("--device is not used", err)

    def test_non_utf8_paper(self):
        bad = self.dir / "latin1.txt"
        bad.write_bytes("caf\xe9\n".encode("latin-1"))
        code, out, err = self.run_cli("outline", str(bad), "--headings-jsonl", str(self.preds))
        self.assertEqual((code, out), (2, ""))
        self.assertIn("is not UTF-8 plain text", err)
        self.assertNotIn("Traceback", err)

    def test_model_load_error_is_one_line(self):
        with mock.patch.object(cli, "recover_structure", side_effect=OSError("Incorrect path_or_model_id: 'x'")):
            code, _, err = self.run_cli("outline", str(self.paper), "--model", "x")
        self.assertEqual(code, 2)
        self.assertIn("error: Incorrect path_or_model_id", err)

    def test_unknown_section_error_copies_and_pickles(self):
        import copy
        import pickle

        exc = UnknownSectionError("9", ["1", "2"])
        self.assertEqual(exc.args[0], "Unknown section id '9'. Known ids: 1, 2")   # as the old plain KeyError
        for clone in (copy.deepcopy(exc), pickle.loads(pickle.dumps(exc))):
            self.assertEqual((str(clone), clone.section_id, clone.known), (str(exc), "9", ["1", "2"]))


class TestEngineCache(unittest.TestCase):
    def setUp(self):
        from paper_structure import infer
        self.infer = infer
        infer._cached_engine.cache_clear()
        self.addCleanup(infer._cached_engine.cache_clear)

    def fake_init(self, engine, path, device=None):
        engine.model_path, engine.device = str(path), self.infer.resolve_device(device)

    def test_one_engine_per_path_and_device(self):
        with tempfile.TemporaryDirectory() as m, mock.patch.object(self.infer.InferenceEngine, "__init__",
                                                                   lambda e, p, d=None: self.fake_init(e, p, d)):
            cpu = self.infer.get_engine(m, "cpu")
            self.assertIs(self.infer.get_engine(m, "cpu"), cpu)
            cuda = self.infer.get_engine(m, "cuda")
        self.assertIsNot(cuda, cpu)
        self.assertEqual((cpu.device, cuda.device), ("cpu", "cuda"))

    def test_without_torch_says_what_to_install(self):
        import sys
        with tempfile.TemporaryDirectory() as m, mock.patch.dict(sys.modules, {"torch": None}):
            with self.assertRaises(ImportError) as cm:
                self.infer.get_engine(m, "cpu")
            self.assertIn(f'pip install -e "{SOURCE}[infer]"', str(cm.exception))   # run from the checkout
            paper = Path(m) / "p.txt"
            paper.write_text(TEXT, encoding="utf-8")
            err = io.StringIO()
            with redirect_stdout(io.StringIO()), redirect_stderr(err):
                code = cli.main(["outline", str(paper), "--model", m])
        self.assertEqual(code, 2)
        self.assertIn(f'pip install -e "{SOURCE}[infer]"', err.getvalue())
        self.assertIn("unrelated package", err.getvalue())
        self.assertNotIn("pip install 'paper-structure", err.getvalue())   # PyPI's paper-structure is someone else's
        self.assertNotIn("Traceback", err.getvalue())

    def test_cuda_asked_but_torch_has_no_gpu(self):
        import sys
        from types import SimpleNamespace
        tf = mock.MagicMock()
        for build, says in ((None, "CPU-only build"), ("12.8", "no usable NVIDIA GPU")):
            torch = SimpleNamespace(__version__="2.14.1", cuda=SimpleNamespace(is_available=lambda: False),
                                    version=SimpleNamespace(cuda=build))
            with tempfile.TemporaryDirectory() as m, mock.patch.dict(sys.modules, {"torch": torch, "transformers": tf}):
                with self.assertRaises(ValueError) as cm:   # ValueError: the CLI and the agent tools report it
                    self.infer.get_engine(m, "cuda")
                self.assertIn(says, str(cm.exception))
                self.assertIn("--device cpu", str(cm.exception))
                paper = Path(m) / "p.txt"
                paper.write_text(TEXT, encoding="utf-8")
                err = io.StringIO()
                with redirect_stdout(io.StringIO()), redirect_stderr(err):
                    code = cli.main(["outline", str(paper), "--model", m, "--device", "cuda"])
            self.assertEqual(code, 2)
            self.assertIn(says, err.getvalue())
            self.assertNotIn("Traceback", err.getvalue())
        tf.AutoTokenizer.from_pretrained.assert_not_called()   # refused before any loading
        tf.AutoModelForCausalLM.from_pretrained.assert_not_called()

    def test_missing_model_folder(self):
        with self.assertRaises(FileNotFoundError) as cm:
            self.infer.get_engine(str(Path(tempfile.gettempdir()) / "no_such_model_dir_20261019"), "cpu")
        self.assertIn("Model folder not found", str(cm.exception))

    def test_engine_reads_device_env_before_loading(self):
        with mock.patch.dict(os.environ, {"PAPER_STRUCTURE_DEVICE": "tpu"}):
            with self.assertRaises(ValueError):
                self.infer.InferenceEngine("no/such/model")

    def test_mcp_cache_follows_device_env(self):
        from paper_structure import mcp_server
        mcp_server._cached_document.cache_clear()
        self.addCleanup(mcp_server._cached_document.cache_clear)
        calls = []

        def fake(text, **kw):
            calls.append(kw["device"])
            return build_document(text, ["1 Introduction"], toc_skip_frac=0)

        with tempfile.TemporaryDirectory() as d:
            paper = Path(d) / "p.txt"
            paper.write_text(TEXT, encoding="utf-8")
            with mock.patch.object(mcp_server, "recover_structure", fake):
                for dev in ("cuda", "cuda", "cpu"):
                    with mock.patch.dict(os.environ, {"PAPER_STRUCTURE_DEVICE": dev}):
                        mcp_server._document(str(paper), "m", 0.05)
        self.assertEqual(calls, ["cuda", "cpu"])   # cached once per device


class TestDtypeKeyword(unittest.TestCase):
    def test_boundaries(self):
        try:
            import transformers  # noqa: F401
        except ImportError:
            self.skipTest("transformers not installed")
        from paper_structure.infer import _dtype_keyword
        for version, want in (("4.55.2", "torch_dtype"), ("4.56.0", "dtype"), ("4.6.1", "torch_dtype"),
                              ("5.0.0", "dtype")):
            with mock.patch("transformers.__version__", version):
                self.assertEqual(_dtype_keyword(), want, version)

    def test_generation_config_cleared_and_cpu_dtype(self):
        try:
            import torch
            import transformers
        except ImportError:
            self.skipTest("torch/transformers not installed")
        from types import SimpleNamespace

        from paper_structure.infer import InferenceEngine, _dtype_keyword
        model = mock.MagicMock()
        model.generation_config = SimpleNamespace(max_length=20, temperature=0.7, top_p=0.8, top_k=20)
        with mock.patch.object(transformers.AutoTokenizer, "from_pretrained", return_value=mock.MagicMock()), \
                mock.patch.object(transformers.AutoModelForCausalLM, "from_pretrained", return_value=model) as load:
            engine = InferenceEngine("m", "cpu")
        self.assertEqual(vars(model.generation_config), dict(max_length=None, temperature=None, top_p=None, top_k=None))
        self.assertEqual(load.call_args.kwargs, {_dtype_keyword(): torch.float32})   # no device_map on cpu
        self.assertEqual(engine.device, "cpu")


class TestInstallHints(unittest.TestCase):
    def test_pdf_hint_points_at_the_source(self):
        import sys
        from paper_structure.pdf import extract_pdf_text
        with mock.patch.dict(sys.modules, {"pypdfium2": None}):
            with self.assertRaises(ImportError) as cm:
                extract_pdf_text(b"%PDF-1.4")
        self.assertIn(f'pip install -e "{SOURCE}[pdf]"', str(cm.exception))
        self.assertIn("unrelated package", str(cm.exception))

    def test_installed_without_the_source_names_the_packages(self):
        from paper_structure import _install
        with tempfile.TemporaryDirectory() as d, mock.patch.object(_install, "SOURCE", Path(d)):
            self.assertEqual(_install.install_hint("pdf", "pypdfium2"), "pip install pypdfium2")

    def test_extras_that_load_on_cuda_bring_accelerate(self):
        try:
            import tomllib
        except ImportError:
            self.skipTest("tomllib needs Python 3.11")
        extras = tomllib.loads((SOURCE / "pyproject.toml").read_text(encoding="utf-8"))["project"][
            "optional-dependencies"]
        for name in ("infer", "app"):   # device_map="auto" needs accelerate
            self.assertTrue(any(d.startswith("accelerate") for d in extras[name]), name)


class TestDevice(unittest.TestCase):
    def test_resolve(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PAPER_STRUCTURE_DEVICE", None)
            self.assertEqual(resolve_device(None), "cuda")
            self.assertEqual(resolve_device("cpu"), "cpu")
        with mock.patch.dict(os.environ, {"PAPER_STRUCTURE_DEVICE": "cpu"}):
            self.assertEqual(resolve_device(None), "cpu")
        with self.assertRaises(ValueError):
            resolve_device("tpu")


if __name__ == "__main__":
    unittest.main()
