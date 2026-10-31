"""Public document-structure API assembled from inference and location stages.

New TASK X adapter. It composes the one-way-copied chunk/inference/location
logic and slices original text between adjacent located heading offsets; it does
not invent or regenerate section body text.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .infer import ModelArgument, detect_headings
from .locate import LocatedHeading, UnlocatedHeading, locate_headings
from .tree import infer_depth, outline_numbering, section_id_base  # noqa: F401 (infer_depth: API)


class UnknownSectionError(KeyError):
    """A section id that is not in the document. Still a KeyError (earlier callers keep working); its message is
    plain text instead of KeyError's quoted repr."""

    def __init__(self, section_id: str, known: list[str]):
        self.section_id, self.known = section_id, list(known)
        # args[0] is the full message, as with the plain KeyError raised before
        super().__init__(f"Unknown section id {section_id!r}. Known ids: {', '.join(self.known)}")

    def __str__(self) -> str:
        return self.args[0]

    def __reduce__(self):   # copy/pickle rebuild from the constructor's arguments, not from args
        return type(self), (self.section_id, self.known)


@dataclass(frozen=True)
class Section:
    id: str
    title: str
    depth: int
    char_start: int
    char_end: int
    parent_id: str | None = None
    has_empty_body: bool = False


@dataclass(frozen=True)
class SearchHit:
    section_id: str
    char_start: int
    char_end: int
    snippet: str


@dataclass
class Document:
    sections: list[Section]
    unlocated: list[UnlocatedHeading]
    _text: str = field(repr=False)

    def outline(self, *, indented: bool = False) -> list[str]:
        """Return a flat list of display lines, optionally indented by depth."""
        lines = []
        for section in self.sections:
            prefix = "  " * (section.depth - 1) if indented else ""
            # display only: a heading matched across a line break keeps its exact text in section.title
            lines.append(f"{prefix}{section.id}\t{' '.join(section.title.split())}")
        return lines

    def get_section(self, section_id: str) -> Section:
        matches = [section for section in self.sections if section.id == section_id]
        if not matches:
            raise UnknownSectionError(section_id, [section.id for section in self.sections])
        return matches[0]

    def parent(self, section_id: str) -> str | None:
        """Return the nearest preceding shallower section id, if any."""
        return self.get_section(section_id).parent_id

    def children(self, section_id: str) -> list[str]:
        """Return direct child section ids in source order."""
        section = self.get_section(section_id)
        section_index = self.sections.index(section)
        child_ids: list[str] = []
        for candidate in self.sections[section_index + 1:]:
            if candidate.depth <= section.depth:
                break
            if candidate.depth == section.depth + 1:
                child_ids.append(candidate.id)
        return child_ids

    def empty_sections(self) -> list[str]:
        """Return ids whose direct body is blank, in source order."""
        return [
            section.id
            for section in self.sections
            if section.has_empty_body
        ]

    def _descendant_end(self, section_id: str) -> int:
        section = self.get_section(section_id)
        section_index = self.sections.index(section)
        for candidate in self.sections[section_index + 1:]:
            if candidate.depth <= section.depth:
                return candidate.char_start
        return len(self._text)

    def read_section(
        self,
        section_id: str,
        *,
        include_descendants: bool = False,
    ) -> str:
        """Return an exact direct slice, optionally extended through descendants."""
        section = self.get_section(section_id)
        char_end = (
            self._descendant_end(section_id)
            if include_descendants
            else section.char_end
        )
        return self._text[section.char_start:char_end]

    def search_in_section(
        self,
        section_id: str,
        query: str,
        *,
        include_descendants: bool = False,
        context_chars: int = 80,
        max_results: int | None = None,
    ) -> list[SearchHit]:
        """Run case-insensitive literal keyword search inside one section."""
        if not query:
            raise ValueError("query must not be empty")
        if context_chars < 0:
            raise ValueError("context_chars must be non-negative")
        if max_results is not None and max_results < 0:
            raise ValueError("max_results must be non-negative or None")
        if max_results == 0:
            return []

        section = self.get_section(section_id)
        section_text = self.read_section(
            section_id,
            include_descendants=include_descendants,
        )
        hits: list[SearchHit] = []
        for match in re.finditer(re.escape(query), section_text, flags=re.IGNORECASE):
            local_start = match.start()
            local_end = match.end()
            snippet_start = max(0, local_start - context_chars)
            snippet_end = min(len(section_text), local_end + context_chars)
            hits.append(SearchHit(
                section_id=section_id,
                char_start=section.char_start + local_start,
                char_end=section.char_start + local_end,
                snippet=section_text[snippet_start:snippet_end],
            ))
            if max_results is not None and len(hits) >= max_results:
                break
        return hits

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable structure metadata without body text."""
        return {
            "sections": [asdict(section) for section in self.sections],
            "unlocated": [asdict(item) for item in self.unlocated],
        }


def _unique_section_id(base: str, seen: dict[str, int]) -> str:
    count = seen.get(base, 0) + 1
    seen[base] = count
    return base if count == 1 else f"{base}#{count}"


def _direct_body_is_empty(section_text: str) -> bool:
    """Test the exact direct slice after removing its first (title) line."""
    line_break = section_text.find("\n")
    body = "" if line_break < 0 else section_text[line_break + 1:]
    return not body.strip()


def _document_from_locations(
    text: str,
    locations: list[LocatedHeading],
    unlocated: list[UnlocatedHeading],
) -> Document:
    sections: list[Section] = []
    seen_ids: dict[str, int] = {}
    numbering = outline_numbering([location.matched_title for location in locations])
    for index, location in enumerate(locations):
        next_start = locations[index + 1].char_start if index + 1 < len(locations) else len(text)
        token, depth = numbering[index]
        base_id = token or section_id_base(location.matched_title, index + 1)
        parent_id = next(
            (
                preceding.id
                for preceding in reversed(sections)
                if preceding.depth < depth
            ),
            None,
        )
        sections.append(Section(
            id=_unique_section_id(base_id, seen_ids),
            title=location.matched_title,
            depth=depth,
            char_start=location.char_start,
            char_end=next_start,
            parent_id=parent_id,
            has_empty_body=_direct_body_is_empty(
                text[location.char_start:next_start]
            ),
        ))
    return Document(sections=sections, unlocated=unlocated, _text=text)


def build_document(
    text: str,
    predicted_headings: list[str],
    *,
    toc_skip_frac: float = 0.05,
) -> Document:
    """Build a Document from already-available predictions (no inference)."""
    locations, unlocated = locate_headings(
        text,
        predicted_headings,
        toc_skip_frac=toc_skip_frac,
    )
    return _document_from_locations(text, locations, unlocated)


def recover_structure(
    text: str,
    *,
    model: ModelArgument = None,
    toc_skip_frac: float = 0.05,
    device: str | None = None,
) -> Document:
    """Detect, locate, and expose navigable sections for plain paper text.

    device: None (PAPER_STRUCTURE_DEVICE, else "cuda": the benchmark's bf16 GPU path) or "cpu" (float32; slower,
    and the output can differ slightly from the benchmark path)."""
    predicted_headings = detect_headings(text, model=model, device=device)
    return build_document(text, predicted_headings, toc_skip_frac=toc_skip_frac)
