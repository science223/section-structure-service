"""Locate predicted headings in original plain text.

Source: paper_benchmark/src/chunk_and_align.py, read 2026-08-23.
Copied content: strip_for_loose(), _is_standalone(), find_all_positions(), and
select_position() (L1 exact, L2 whitespace/case, L3 punctuation-stripped window;
standalone lines preferred and front-of-document matches deprioritized).
New tool-only behavior is isolated in locate_headings(): explicit failures,
maximum-retention monotonic position assignment, and configurable ToC skip.
This is a one-way copy; the two files are not kept in bidirectional sync.
"""

from __future__ import annotations

import bisect
import re
import string
import sys
from dataclasses import dataclass


PUNCT_TBL = str.maketrans("", "", string.punctuation)


def strip_for_loose(s: str) -> str:
    return s.translate(PUNCT_TBL).replace(" ", "").replace("\n", "").replace("\t", "").lower()


def _is_standalone(full_text: str, pos: int, end: int) -> bool:
    """
    Return True if the match at [pos, end) sits on its own line:
    - the character immediately before pos is '\n' or pos == 0
    - after skipping optional trailing spaces/tabs, the character at `end` is '\n' or end == len(full_text)
    """
    if pos > 0 and full_text[pos - 1] != "\n":
        return False
    after = end
    while after < len(full_text) and full_text[after] in (" ", "\t"):
        after += 1
    return after == len(full_text) or full_text[after] == "\n"


def find_all_positions(full_text: str, heading: str):
    """
    返回 [(pos, matched_form, level, is_standalone), ...] 所有匹配位置(按 pos 升序)。
    is_standalone=True 表示该位置是独立行标题(前面是换行/行首,后面是换行/行尾)。
    """
    results = []

    # L1: 严格 substring,找全部
    start = 0
    while True:
        pos = full_text.find(heading, start)
        if pos == -1:
            break
        standalone = _is_standalone(full_text, pos, pos + len(heading))
        results.append((pos, heading, "L1", standalone))
        start = pos + 1

    # 如果 L1 命中了,直接返回(不混用层级,避免重复)
    if results:
        return results

    # L2: 大小写不敏感 + 空白容错
    h_collapsed = re.sub(r"\s+", " ", heading)
    pattern = re.compile(r"(?i)" + re.escape(h_collapsed).replace(r"\ ", r"\s+"))
    for m in pattern.finditer(full_text):
        standalone = _is_standalone(full_text, m.start(), m.end())
        results.append((m.start(), m.group(0), "L2", standalone))
    if results:
        return results

    # L3: 滑窗,做 stripped 比较
    h_loose = strip_for_loose(heading)
    if not h_loose:
        return []
    word_matches = list(re.finditer(r"\S+", full_text))
    n = len(word_matches)
    h_len = len(h_loose)
    for start_i in range(n):
        for end_i in range(start_i + 1, min(start_i + 16, n + 1)):
            window_text = full_text[word_matches[start_i].start():word_matches[end_i - 1].end()]
            window_loose = strip_for_loose(window_text)
            if len(window_loose) > h_len + 5:
                break
            if window_loose == h_loose:
                standalone = _is_standalone(
                    full_text,
                    word_matches[start_i].start(),
                    word_matches[end_i - 1].end(),
                )
                results.append((word_matches[start_i].start(), window_text, "L3", standalone))
                break  # 同一起点不重复匹配
    return results


def select_position(positions, full_text_len, toc_skip_chars):
    """
    从所有命中位置中选一个作为 expected_heading 位置。

    优先级(高到低):
    1. 独立行 + 在 TOC skip 之后 → 最可靠的正文章节标题
    2. 独立行 + 在 TOC skip 之前 → 真正的章节标题恰好在目录区字符数之前
       (例如 'Abstract' 在页 1, toc_skip 因为有目录被设为 5%,但章节本身比目录更早)
    3. 非独立行 + 在 TOC skip 之后 → 正文里的子串匹配(表格单元格等)
    4. 非独立行 + 在 TOC skip 之前 → 目录条目里的子串,最后 fallback

    每个优先级内取 pos 最小的(最早出现)。
    """
    standalone_after  = [p for p in positions if p[3] and p[0] >= toc_skip_chars]
    standalone_before = [p for p in positions if p[3] and p[0] <  toc_skip_chars]
    inline_after      = [p for p in positions if not p[3] and p[0] >= toc_skip_chars]
    inline_before     = [p for p in positions if not p[3] and p[0] <  toc_skip_chars]

    for bucket in (standalone_after, standalone_before, inline_after, inline_before):
        if bucket:
            return bucket[0]
    return None


@dataclass(frozen=True)
class LocatedHeading:
    predicted_title: str
    matched_title: str
    char_start: int
    char_end: int
    match_level: str
    is_standalone: bool


@dataclass(frozen=True)
class UnlocatedHeading:
    predicted_title: str
    predicted_index: int
    reason: str
    candidate_positions: tuple[int, ...] = ()


MAX_CANDIDATES_PER_HEADING = 64


@dataclass(frozen=True)
class _Candidate:
    predicted_index: int
    position: tuple[int, str, str, bool]
    preference_rank: int
    candidate_order: int


class _Path:
    """A DP path stored as its last choice plus the path it extends, so extending costs O(1) memory.
    Paths are ordered by _path_key: more choices first, then lower preference cost, then lower position sum, then the
    (predicted_index, char position, candidate_order) sequence of the choices. No two paths share that sequence."""

    __slots__ = ("candidate", "parent", "length", "preference_cost", "position_sum")

    def __init__(self, candidate: _Candidate, parent: "_Path | None"):
        self.candidate, self.parent = candidate, parent
        self.length = 1 + (parent.length if parent else 0)
        self.preference_cost = candidate.preference_rank + (parent.preference_cost if parent else 0)
        self.position_sum = candidate.position[0] + (parent.position_sum if parent else 0)

    def choices(self) -> tuple[_Candidate, ...]:
        out, p = [], self
        while p is not None:
            out.append(p.candidate)
            p = p.parent
        return tuple(reversed(out))

    def __lt__(self, other: "_Path") -> bool:
        a = (-self.length, self.preference_cost, self.position_sum)
        b = (-other.length, other.preference_cost, other.position_sum)
        if a != b:
            return a < b
        return _path_key(self) < _path_key(other)   # rare: walk both paths only on a tie


def _path_key(path: _Path):
    """Minimization key implementing the three specified DP tie-breaks."""
    return (-path.length, path.preference_cost, path.position_sum,
            tuple((c.predicted_index, c.position[0], c.candidate_order) for c in path.choices()))


class _PrefixMin:
    """Fenwick tree over sorted positions: the best path ending before a position, in O(log n).
    11-04: the earlier DP compared every candidate with every earlier path, rebuilding both keys each time; the pytest
    manual's 430 headings took 295 s and PostgreSQL's 1,899 did not finish in over an hour. The best extension of a
    candidate is the best earlier path that ends before it, because adding the same candidate to every earlier path
    keeps their order (each key part grows by the same amount, and paths of equal length get the same last step)."""

    def __init__(self, positions: list[int]):
        self.positions = positions
        self.tree: list[_Path | None] = [None] * (len(positions) + 1)

    def add(self, pos: int, path: _Path) -> None:
        i = self._index(pos) + 1
        while i < len(self.tree):
            if self.tree[i] is None or path < self.tree[i]:
                self.tree[i] = path
            i += i & -i

    def best_before(self, pos: int) -> "_Path | None":
        i, best = self._index(pos), None   # tree prefix [1, i] covers positions strictly below pos
        while i > 0:
            if self.tree[i] is not None and (best is None or self.tree[i] < best):
                best = self.tree[i]
            i -= i & -i
        return best

    def _index(self, pos: int) -> int:
        return bisect.bisect_left(self.positions, pos)


def _position_preference_rank(position, toc_skip_chars: int) -> int:
    """Return the bucket order documented by select_position()."""
    pos, _, _, is_standalone = position
    if is_standalone and pos >= toc_skip_chars:
        return 0
    if is_standalone and pos < toc_skip_chars:
        return 1
    if not is_standalone and pos >= toc_skip_chars:
        return 2
    return 3


def _ordered_candidates(
    positions,
    predicted_index: int,
    toc_skip_chars: int,
) -> list[_Candidate]:
    """Sort like select_position(), then cap the DP input deterministically."""
    ranked = sorted(
        enumerate(positions),
        key=lambda item: (
            _position_preference_rank(item[1], toc_skip_chars),
            item[1][0],
            item[0],
        ),
    )[:MAX_CANDIDATES_PER_HEADING]
    return [
        _Candidate(
            predicted_index=predicted_index,
            position=position,
            preference_rank=_position_preference_rank(position, toc_skip_chars),
            candidate_order=candidate_order,
        )
        for candidate_order, (_, position) in enumerate(ranked)
    ]


def locate_headings(
    full_text: str,
    predicted_headings: list[str],
    *,
    toc_skip_frac: float = 0.05,
) -> tuple[list[LocatedHeading], list[UnlocatedHeading]]:
    """Choose a maximum-cardinality, strictly increasing heading assignment."""
    if not 0 <= toc_skip_frac < 1:
        raise ValueError("toc_skip_frac must be in the half-open interval [0, 1).")

    toc_skip_chars = int(len(full_text) * toc_skip_frac)
    unlocated: list[UnlocatedHeading] = []
    raw_positions: dict[int, list[tuple[int, str, str, bool]]] = {}
    per_heading: list[list[_Candidate]] = []
    truncations: list[tuple[int, int]] = []

    for predicted_index, heading in enumerate(predicted_headings):
        positions = find_all_positions(full_text, heading)
        if not positions:
            unlocated.append(UnlocatedHeading(
                predicted_title=heading,
                predicted_index=predicted_index,
                reason="no_text_match",
            ))
            continue

        raw_positions[predicted_index] = positions
        if len(positions) > MAX_CANDIDATES_PER_HEADING:
            truncations.append((predicted_index, len(positions)))

        per_heading.append(_ordered_candidates(
            positions,
            predicted_index,
            toc_skip_chars,
        ))

    # For each candidate, the best path ending at it extends the best path of the earlier headings that ends before
    # it (or is the candidate alone); paths of one heading do not see each other (strictly increasing headings).
    ends = _PrefixMin(sorted({c.position[0] for cands in per_heading for c in cands}))
    best: _Path | None = None
    for candidates in per_heading:
        current: list[_Path] = []
        for candidate in candidates:
            path = _Path(candidate, ends.best_before(candidate.position[0]))
            current.append(path)
            if best is None or path < best:
                best = path
        for path in current:
            ends.add(path.candidate.position[0], path)
    best_choices = best.choices() if best is not None else ()

    if truncations:
        details = ",".join(
            f"{index}:{count}->{MAX_CANDIDATES_PER_HEADING}"
            for index, count in truncations
        )
        print(   # a diagnostic: stderr, so it can never mix into a program's stdout (the MCP stdio stream)
            "LOCATE CANDIDATE TRUNCATION "
            f"max={MAX_CANDIDATES_PER_HEADING} headings={details}",
            file=sys.stderr,
        )

    selected = {choice.predicted_index: choice for choice in best_choices}
    located: list[LocatedHeading] = []
    for choice in best_choices:
        pos, matched_form, level, is_standalone = choice.position
        located.append(LocatedHeading(
            predicted_title=predicted_headings[choice.predicted_index],
            matched_title=matched_form,
            char_start=pos,
            char_end=pos + len(matched_form),
            match_level=level,
            is_standalone=is_standalone,
        ))

    for predicted_index, positions in raw_positions.items():
        if predicted_index not in selected:
            unlocated.append(UnlocatedHeading(
                predicted_title=predicted_headings[predicted_index],
                predicted_index=predicted_index,
                reason="dropped_for_monotonic_consistency",
                candidate_positions=tuple(position[0] for position in positions),
            ))

    unlocated.sort(key=lambda item: item.predicted_index)

    return located, unlocated
