from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paper_structure.cli import build_parser
from paper_structure.mcp_server import _cached_document, _document, create_server


class InterfaceTests(unittest.TestCase):
    def test_cli_contract_parses(self):
        parser = build_parser()
        outline_args = parser.parse_args(
            ["outline", "paper.txt", "--headings-jsonl", "predictions.jsonl"]
        )
        read_args = parser.parse_args([
            "read",
            "paper.txt",
            "--section",
            "4.3",
            "--headings-jsonl",
            "predictions.jsonl",
            "--include-descendants",
        ])
        search_args = parser.parse_args(
            [
                "search",
                "paper.txt",
                "--section",
                "4.3",
                "-q",
                "baseline",
                "--headings-jsonl",
                "predictions.jsonl",
                "--include-descendants",
            ]
        )
        self.assertEqual(outline_args.headings_jsonl.name, "predictions.jsonl")
        self.assertEqual(read_args.section, "4.3")
        self.assertTrue(read_args.include_descendants)
        self.assertEqual(search_args.query, "baseline")
        self.assertTrue(search_args.include_descendants)

    def test_mcp_saved_headings_path_bypasses_inference(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            text_path = Path(temp_dir) / "paper.txt"
            predictions_path = Path(temp_dir) / "predictions.jsonl"
            text_path.write_text("1 Intro\nbody\n", encoding="utf-8")
            predictions_path.write_text(
                json.dumps({"predicted_headings": ["1 Intro"]}) + "\n",
                encoding="utf-8",
            )
            _cached_document.cache_clear()
            with patch(
                "paper_structure.mcp_server.recover_structure",
                side_effect=AssertionError("inference path must not be called"),
            ):
                document = _document(
                    str(text_path),
                    "unused-model",
                    0,
                    str(predictions_path),
                )
            self.assertEqual(document.outline(), ["1\t1 Intro"])
            _cached_document.cache_clear()

    def test_missing_mcp_dependency_is_explicit(self):
        if importlib.util.find_spec("mcp") is not None:
            self.skipTest("MCP SDK is installed in this environment")
        with self.assertRaisesRegex(RuntimeError, "optional MCP SDK is not installed"):
            create_server()


if __name__ == "__main__":
    unittest.main()
