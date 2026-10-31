"""What to tell the user (or an agent) when an optional dependency is missing.

This package is installed from its source folder. PyPI has an unrelated project with the same name (paper-structure
0.1.0 / 0.1.1, a layout-detection and OCR tool, uploaded 2026-02-11; checked 2026-10-20), so a hint must never be
``pip install paper-structure[...]``: run as written, or with ``-U``, that installs someone else's code.
"""

from __future__ import annotations

from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent
NOT_ON_PYPI = "the PyPI project named paper-structure is an unrelated package, do not install it"


def install_hint(extra: str, packages: str) -> str:
    """The pip command that adds ``extra``: from the source folder when this package was installed from one."""
    if (SOURCE / "pyproject.toml").is_file():   # an editable install (pip install -e .) or a checkout on sys.path
        return f'pip install -e "{SOURCE}[{extra}]"'
    return f"pip install {packages}"
