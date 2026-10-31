"""Pack zotero-plugin/ into dist/paper-structure-<version>.xpi (a zip with forward-slash paths, manifest at the root).
  python scripts/build_zotero_xpi.py"""

import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "zotero-plugin"


def main() -> int:
    version = json.loads((SRC / "manifest.json").read_text(encoding="utf-8"))["version"]
    out = ROOT / "dist" / f"paper-structure-{version}.xpi"
    out.parent.mkdir(exist_ok=True)
    files = sorted(p for p in SRC.rglob("*") if p.is_file() and p.name != "updates.json")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(SRC).as_posix())
    print(out, [p.relative_to(SRC).as_posix() for p in files])
    return 0


if __name__ == "__main__":
    sys.exit(main())
