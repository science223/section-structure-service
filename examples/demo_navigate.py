"""Offline deterministic agent-loop demo; this is not an experiment."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Protocol

# Keep ``python examples/demo_navigate.py`` usable without installing the package.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_structure import Document, build_document


DEMO_TEXT = """4.2 Baselines
Baseline definition: Regex marks standalone numbered lines as headings.
The definition is intentionally stored in this preceding section.

4.3 Results
The recovered structure is evaluated here. The baseline definition is not
repeated in this section, so a reader must navigate backward for it.

4.4 Discussion
The result is descriptive. This demo makes no downstream RAG or QA claim.
"""

DEMO_HEADINGS = ["4.2 Baselines", "4.3 Results", "4.4 Discussion"]

NESTED_DEMO_TEXT = """4 Method
Chapter introduction only.
4.1 Setup
Deterministic setup details.
4.2 Baselines
Deterministic baseline details.
5 Results
Results are outside the Method chapter.
"""

NESTED_DEMO_HEADINGS = ["4 Method", "4.1 Setup", "4.2 Baselines", "5 Results"]


class NavigationStrategy(Protocol):
    """Pluggable policy surface; no network model is required."""

    def next_section(self, document: Document, visited: list[str]) -> str | None:
        ...


class DeterministicBaselineStrategy:
    """Hard-coded route: 4.3 -> missing definition -> 4.2 -> 4.4 -> stop."""

    def next_section(self, document: Document, visited: list[str]) -> str | None:
        if not visited:
            return "4.3"
        if visited == ["4.3"]:
            exact_definition = document.search_in_section(
                "4.3", "Baseline definition:"
            )
            return "4.4" if exact_definition else "4.2"
        if visited == ["4.3", "4.2"]:
            return "4.4"
        return None


def run_demo(strategy: NavigationStrategy) -> None:
    document = build_document(DEMO_TEXT, DEMO_HEADINGS, toc_skip_frac=0)
    print("1. inspect outline")
    print("\n".join(document.outline(indented=True)))

    visited: list[str] = []
    while True:
        section_id = strategy.next_section(document, visited)
        if section_id is None:
            print("5. stop")
            break
        visited.append(section_id)
        body = document.read_section(section_id)
        if section_id == "4.3":
            print("2. read 4.3; baseline definition missing here")
        elif section_id == "4.2":
            hits = document.search_in_section(section_id, "Baseline definition:")
            print(f"3. read 4.2; baseline definition hits={len(hits)}")
        else:
            print("4. read 4.4")
        print(body.splitlines()[0])


def run_nested_demo() -> None:
    """Contrast the default direct slice with opt-in whole-chapter reading."""
    document = build_document(
        NESTED_DEMO_TEXT,
        NESTED_DEMO_HEADINGS,
        toc_skip_frac=0,
    )
    direct = document.read_section("4")
    nested = document.read_section("4", include_descendants=True)

    print("\n6. nested navigation demo")
    print(f'children("4") == {document.children("4")}')
    print("read 4 (direct; excludes 4.1/4.2):")
    print(direct.rstrip())
    print("read 4 include_descendants (includes 4.1/4.2; stops before 5):")
    print(nested.rstrip())


if __name__ == "__main__":
    run_demo(DeterministicBaselineStrategy())
    run_nested_demo()
