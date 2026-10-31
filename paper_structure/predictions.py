"""Load saved per-chunk heading predictions without running inference."""

from __future__ import annotations

import json
from pathlib import Path


def load_predicted_headings(path: str | Path) -> list[str]:
    """Flatten ``predicted_headings`` lists in JSONL row order."""
    with Path(path).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    return [
        heading
        for row in rows
        for heading in row["predicted_headings"]
    ]
