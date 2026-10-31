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


@dataclass(frozen=True)
class _PathState:
    choices: tuple[_Candidate, ...]
    preference_cost: int
    position_sum: int


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


def _path_key(state: _PathState):
    """Minimization key implementing the three specified DP tie-breaks."""
    return (
        -len(state.choices),
        state.preference_cost,
        state.position_sum,
        tuple(
            (
                choice.predicted_index,
                choice.position[0],
                choice.candidate_order,
            )
            for choice in state.choices
        ),
    )


def _better(left: _PathState, right: _PathState) -> _PathState:
    return left if _path_key(left) < _path_key(right) else right


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
    states: list[_PathState] = []
    best = _PathState(choices=(), preference_cost=0, position_sum=0)
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

        candidates = _ordered_candidates(
            positions,
            predicted_index,
            toc_skip_chars,
        )
        current_states: list[_PathState] = []
        for candidate in candidates:
            candidate_pos = candidate.position[0]
            candidate_best = _PathState(
                choices=(candidate,),
                preference_cost=candidate.preference_rank,
                position_sum=candidate_pos,
            )
            for previous in states:
                if previous.choices[-1].position[0] >= candidate_pos:
                    continue
                contender = _PathState(
                    choices=previous.choices + (candidate,),
                    preference_cost=(
                        previous.preference_cost + candidate.preference_rank
                    ),
                    position_sum=previous.position_sum + candidate_pos,
                )
                candidate_best = _better(contender, candidate_best)
            current_states.append(candidate_best)
            best = _better(candidate_best, best)
        states.extend(current_states)

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

    selected = {choice.predicted_index: choice for choice in best.choices}
    located: list[LocatedHeading] = []
    for choice in best.choices:
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
