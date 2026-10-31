"""Read-only TASK X regression gates against the paper_benchmark artifacts."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path

from paper_structure import build_document, chunk_text, recover_structure
from paper_structure.infer import get_engine, predict_chunks
from paper_structure.locate import locate_headings


CHUNK_FIELDS = ("chunk_id", "char_start", "char_end", "text")
TASK_Y_BASELINE = Path(__file__).with_name("task_y_locator_baseline.json")
GATEG_OUTPUT = Path(__file__).with_name("gateG_empty_sections.json")
EXPECTED_PAPERS = 26
EXPECTED_LOCATED = 1082
EXPECTED_EMPTY_DIRECT = 79
DEFAULT_GATEJ_PAPERS = (
    "A1_Transformer_Attention",
    "B1_Survey_TransferLearning",
)
EXPECTED_UNLOCATED = {
    "no_text_match": 24,
    "dropped_for_monotonic_consistency": 9,
}


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _task_y_baseline(path: Path) -> dict[str, dict[str, int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["papers"]


def _saved_document(text_path: Path, predictions_dir: Path):
    predictions = _jsonl(predictions_dir / f"{text_path.stem}.jsonl")
    headings = [
        heading
        for row in predictions
        for heading in row["predicted_headings"]
    ]
    text = text_path.read_text(encoding="utf-8")
    return build_document(text, headings), len(headings)


def gate_a(
    full_text_dir: Path,
    predictions_dir: Path,
    baseline_path: Path,
) -> int:
    """Audit locator retention on all saved v13_2 predictions without inference."""
    baseline = _task_y_baseline(baseline_path)
    text_paths = sorted(full_text_dir.glob("*.txt"))
    rows: list[dict] = []
    unexpected_reasons: list[tuple[str, str]] = []

    print(
        "paper_id\tpredicted\tlocated\tunlocated(no_text_match)\t"
        "unlocated(dropped)\tlocated_rate"
    )
    for text_path in text_paths:
        paper_id = text_path.stem
        predictions = _jsonl(predictions_dir / f"{paper_id}.jsonl")
        headings = [
            heading
            for row in predictions
            for heading in row["predicted_headings"]
        ]
        text = text_path.read_text(encoding="utf-8")
        document = build_document(text, headings)
        reasons = Counter(item.reason for item in document.unlocated)
        for reason in reasons:
            if reason not in {
                "no_text_match",
                "dropped_for_monotonic_consistency",
            }:
                unexpected_reasons.append((paper_id, reason))

        located = len(document.sections)
        predicted = len(headings)
        rate = located / predicted if predicted else 0.0
        row = {
            "paper_id": paper_id,
            "predicted": predicted,
            "located": located,
            "no_text_match": reasons["no_text_match"],
            "dropped": reasons["dropped_for_monotonic_consistency"],
            "located_rate": rate,
        }
        rows.append(row)
        print(
            f"{paper_id}\t{predicted}\t{located}\t{row['no_text_match']}\t"
            f"{row['dropped']}\t{rate:.6f}"
        )

    total_predicted = sum(row["predicted"] for row in rows)
    total_located = sum(row["located"] for row in rows)
    total_no_text = sum(row["no_text_match"] for row in rows)
    total_dropped = sum(row["dropped"] for row in rows)
    total_rate = total_located / total_predicted if total_predicted else 0.0
    print(
        f"TOTAL\t{total_predicted}\t{total_located}\t{total_no_text}\t"
        f"{total_dropped}\t{total_rate:.6f}"
    )

    rows_by_id = {row["paper_id"]: row for row in rows}
    paper_sets_match = set(rows_by_id) == set(baseline)
    regressions = [
        (
            paper_id,
            rows_by_id[paper_id]["located"],
            baseline[paper_id]["located"],
        )
        for paper_id in sorted(set(rows_by_id) & set(baseline))
        if rows_by_id[paper_id]["located"] < baseline[paper_id]["located"]
    ]
    a3_located = rows_by_id.get("A3_GPT4_TechReport", {}).get("located", 0)
    low_rate_papers = [
        row["paper_id"] for row in rows if row["located_rate"] < 0.85
    ]

    checks = {
        "papers_26": len(rows) == 26 and paper_sets_match,
        "a3_at_least_50": a3_located >= 50,
        "total_rate_at_least_0_90": total_rate >= 0.90,
        "no_paper_regressed": not regressions,
        "known_reasons_only": not unexpected_reasons,
        "gate_c_at_most_2_below_0_85": len(low_rate_papers) <= 2,
    }
    print(
        f"GATEA CHECK papers={len(rows)}/26 baseline_ids_match={paper_sets_match} "
        f"status={'PASS' if checks['papers_26'] else 'FAIL'}"
    )
    print(
        f"GATEA CHECK A3 located={a3_located} threshold=50 "
        f"status={'PASS' if checks['a3_at_least_50'] else 'FAIL'}"
    )
    print(
        f"GATEA CHECK total_rate={total_rate:.6f} threshold=0.900000 "
        f"status={'PASS' if checks['total_rate_at_least_0_90'] else 'FAIL'}"
    )
    print(
        f"GATEA CHECK regressions={regressions} "
        f"status={'PASS' if checks['no_paper_regressed'] else 'FAIL'}"
    )
    print(
        f"GATEA CHECK unexpected_reasons={unexpected_reasons} "
        f"status={'PASS' if checks['known_reasons_only'] else 'FAIL'}"
    )
    print(
        f"GATEC CHECK below_0.85={len(low_rate_papers)} papers={low_rate_papers} "
        f"max_allowed=2 "
        f"status={'PASS' if checks['gate_c_at_most_2_below_0_85'] else 'FAIL'}"
    )
    passed = all(checks.values())
    print(f"GATEA SUMMARY status={'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


def gate1a(full_text_dir: Path, chunks_dir: Path) -> int:
    text_paths = sorted(full_text_dir.glob("*.txt"))
    passed = 0
    total_chunks = 0
    for text_path in text_paths:
        reference_path = chunks_dir / f"{text_path.stem}.jsonl"
        generated = chunk_text(text_path.read_text(encoding="utf-8"))
        reference = _jsonl(reference_path)
        total_chunks += len(generated)
        equal = len(generated) == len(reference) and all(
            all(new[field] == old[field] for field in CHUNK_FIELDS)
            for new, old in zip(generated, reference)
        )
        if equal:
            passed += 1
            print(f"GATE1A {text_path.stem}: EQUAL chunks={len(generated)}")
        else:
            print(
                f"GATE1A {text_path.stem}: MISMATCH "
                f"generated={len(generated)} reference={len(reference)}"
            )
    print(f"GATE1A SUMMARY papers={passed}/{len(text_paths)} chunks={total_chunks}")
    return 0 if passed == len(text_paths) == 26 else 1


def _first_difference(left: str, right: str) -> int | None:
    for index, (left_char, right_char) in enumerate(zip(left, right)):
        if left_char != right_char:
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


def gate2(full_text_dir: Path, predictions_dir: Path, limit: int) -> int:
    text_paths = sorted(full_text_dir.glob("*.txt"))[:limit]
    passed = 0
    for text_path in text_paths:
        predictions = _jsonl(predictions_dir / f"{text_path.stem}.jsonl")
        headings = [
            heading
            for row in predictions
            for heading in row["predicted_headings"]
        ]
        text = text_path.read_text(encoding="utf-8")
        document = build_document(text, headings)
        if document.sections:
            reconstructed = "".join(
                document.read_section(section.id) for section in document.sections
            )
            expected = text[
                document.sections[0].char_start:document.sections[-1].char_end
            ]
            equal = reconstructed == expected
            difference = _first_difference(reconstructed, expected)
        else:
            equal = False
            difference = 0
        passed += int(equal)
        print(
            f"GATE2 {text_path.stem}: {'EQUAL' if equal else 'MISMATCH'} "
            f"sections={len(document.sections)} unlocated={len(document.unlocated)}"
        )
        if not equal:
            first_section_start = (
                document.sections[0].char_start if document.sections else 0
            )
            print(
                f"GATE2 DIFF {text_path.stem}: relative_char={difference} "
                f"absolute_char={first_section_start + difference}"
            )
    print(f"GATE2 SUMMARY papers={passed}/{len(text_paths)}")
    return 0 if passed == len(text_paths) == limit else 1


def _direct_body_is_empty(document, section_id: str) -> bool:
    direct = document.read_section(section_id)
    line_break = direct.find("\n")
    body = "" if line_break < 0 else direct[line_break + 1:]
    return not body.strip()


def gate_e(full_text_dir: Path, predictions_dir: Path) -> int:
    """Inventory nested navigation over saved predictions without inference."""
    text_paths = sorted(full_text_dir.glob("*.txt"))
    rows: list[dict] = []
    documents = {}

    print(
        "paper_id\tn_sections\tmax_depth\tn_with_children\t"
        "n_empty_direct\tn_unlocated"
    )
    for text_path in text_paths:
        document, predicted = _saved_document(text_path, predictions_dir)
        documents[text_path.stem] = document
        row = {
            "paper_id": text_path.stem,
            "n_sections": len(document.sections),
            "max_depth": max(
                (section.depth for section in document.sections),
                default=0,
            ),
            "n_with_children": sum(
                bool(document.children(section.id))
                for section in document.sections
            ),
            "n_empty_direct": sum(
                _direct_body_is_empty(document, section.id)
                for section in document.sections
            ),
            "n_unlocated": len(document.unlocated),
            "predicted": predicted,
        }
        rows.append(row)
        print(
            f"{row['paper_id']}\t{row['n_sections']}\t{row['max_depth']}\t"
            f"{row['n_with_children']}\t{row['n_empty_direct']}\t"
            f"{row['n_unlocated']}"
        )

    total_sections = sum(row["n_sections"] for row in rows)
    total_unlocated = sum(row["n_unlocated"] for row in rows)
    total_predicted = sum(row["predicted"] for row in rows)
    print(
        f"TOTAL\t{total_sections}\t"
        f"{max((row['max_depth'] for row in rows), default=0)}\t"
        f"{sum(row['n_with_children'] for row in rows)}\t"
        f"{sum(row['n_empty_direct'] for row in rows)}\t{total_unlocated}"
    )

    print("A1 TREE (first 15 located sections)")
    print("id\tdepth\tparent_id\tchildren")
    a1_document = documents.get("A1_Transformer_Attention")
    if a1_document is not None:
        for section in a1_document.sections[:15]:
            print(
                f"{section.id}\t{section.depth}\t{section.parent_id}\t"
                f"{a1_document.children(section.id)}"
            )

    checks = {
        "papers_26": len(rows) == EXPECTED_PAPERS,
        "located_1082": total_sections == EXPECTED_LOCATED,
        "located_plus_unlocated_equals_predicted": (
            total_sections + total_unlocated == total_predicted
        ),
        "a1_present": a1_document is not None,
    }
    for name, passed in checks.items():
        print(f"GATEE CHECK {name} status={'PASS' if passed else 'FAIL'}")
    passed = all(checks.values())
    print(f"GATEE SUMMARY status={'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


def gate_f(full_text_dir: Path, predictions_dir: Path) -> int:
    """Print every unlocated saved prediction and verify known reason counts."""
    text_paths = sorted(full_text_dir.glob("*.txt"))
    reason_counts: Counter[str] = Counter()
    examples: dict[str, list[tuple[str, int, str]]] = {
        reason: [] for reason in EXPECTED_UNLOCATED
    }

    print("paper_id\tpredicted_index\tpredicted_title\treason")
    for text_path in text_paths:
        document, _ = _saved_document(text_path, predictions_dir)
        for item in document.unlocated:
            reason_counts[item.reason] += 1
            print(
                f"{text_path.stem}\t{item.predicted_index}\t"
                f"{json.dumps(item.predicted_title, ensure_ascii=False)}\t"
                f"{item.reason}"
            )
            if item.reason in examples and len(examples[item.reason]) < 3:
                examples[item.reason].append(
                    (text_path.stem, item.predicted_index, item.predicted_title)
                )

    print("GATEF COUNTS")
    for reason, expected in EXPECTED_UNLOCATED.items():
        actual = reason_counts[reason]
        print(f"{reason}\tactual={actual}\texpected={expected}")
        for paper_id, predicted_index, predicted_title in examples[reason]:
            print(
                f"GATEF EXAMPLE reason={reason} paper={paper_id} "
                f"predicted_index={predicted_index} "
                f"predicted_title={predicted_title!r}"
            )

    unexpected = {
        reason: count
        for reason, count in reason_counts.items()
        if reason not in EXPECTED_UNLOCATED
    }
    checks = {
        "papers_26": len(text_paths) == EXPECTED_PAPERS,
        "reason_counts": all(
            reason_counts[reason] == expected
            for reason, expected in EXPECTED_UNLOCATED.items()
        ),
        "known_reasons_only": not unexpected,
    }
    for name, passed in checks.items():
        print(f"GATEF CHECK {name} status={'PASS' if passed else 'FAIL'}")
    passed = all(checks.values())
    print(f"GATEF SUMMARY status={'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


def _body_after_title(section_text: str) -> str:
    line_break = section_text.find("\n")
    return "" if line_break < 0 else section_text[line_break + 1:]


def _looks_like_toc_residue(
    *,
    title: str,
    pos_frac: float,
    context_before: str,
    context_after: str,
) -> bool:
    """Return a conservative ToC signal based on visible source evidence."""
    if pos_frac >= 0.15:
        return False
    context = f"{context_before}{context_after}"
    return title.casefold() == "contents" or ". . ." in context


def _model_furniture_signals(title: str) -> list[str]:
    """Return the reproducible text-shape signals used by TASK AB."""
    stripped = title.strip()
    words = stripped.split()
    signals = []
    if stripped and re.fullmatch(r"[\d.,%+\-/\s]+", stripped):
        signals.append("title is numeric/punctuation-only")
    if re.fullmatch(r"\d+(?:\.\d+)?\s*[BbMmKk]", stripped):
        signals.append("title is a model-size token")
    if len(words) <= 2 and stripped and stripped[0].isupper():
        signals.append("title is an uppercase-initial label of at most two words")
    if stripped.endswith(".") and len(words) > 4:
        signals.append("title is a period-terminated sentence of more than four words")
    if re.match(r"^[A-Z](?:\.\d+)?\s+\S", stripped):
        signals.append("title starts with an uppercase appendix/designator token")
    if re.fullmatch(r"\d+\.\s+.+:\s*", stripped):
        signals.append("title is a numbered colon-terminated field label")
    return signals


def _classify_empty_section(
    *,
    title: str,
    pos_frac: float,
    match_level: str,
    is_standalone: bool,
    next_section_id: str | None,
    child_ids: list[str],
    context_before: str,
    context_after: str,
) -> tuple[str, str, list[str]]:
    """Classify only when one of TASK AA's evidence rules is observable."""
    evidence = ["direct body after the matched title is whitespace-only"]
    if _looks_like_toc_residue(
        title=title,
        pos_frac=pos_frac,
        context_before=context_before,
        context_after=context_after,
    ):
        evidence.extend([
            f"pos_frac={pos_frac:.6f} is below 0.15",
            "context contains a Contents label or dot-leader page-list text",
        ])
        return "b_toc_residue", "directly_proven", evidence

    locator_signals = []
    if match_level == "L3":
        locator_signals.append("match_level=L3")
    if not is_standalone:
        locator_signals.append("is_standalone=false")
    if locator_signals:
        evidence.extend(locator_signals)
        evidence.append("no conservative table-of-contents marker was observed")
        return "c_locator_misplacement", "strongly_correlated_not_proven", evidence

    furniture_signals = _model_furniture_signals(title)
    if (
        match_level == "L1"
        and is_standalone
        and not child_ids
        and furniture_signals
    ):
        evidence.extend([
            "match_level=L1 and is_standalone=true",
            "the section has no children",
            *furniture_signals,
        ])
        return "d_model_furniture", "mechanical_shape_match", evidence

    if next_section_id is not None and next_section_id in child_ids:
        evidence.extend([
            f"the immediately following section {next_section_id!r} is a direct child",
            f"match_level={match_level} and is_standalone=true",
        ])
        return "a_genuine_adjacent_headings", "strongly_supported", evidence

    evidence.extend([
        f"match_level={match_level} and is_standalone={str(is_standalone).lower()}",
        "no conservative ToC marker, weak-match signal, or immediate-child relation distinguishes the cause",
    ])
    return "unknown", "evidence_insufficient", evidence


def gate_g(
    full_text_dir: Path,
    predictions_dir: Path,
    output_path: Path,
) -> int:
    """Inventory and conservatively classify every empty direct section body."""
    text_paths = sorted(full_text_dir.glob("*.txt"))
    records: list[dict] = []
    total_sections = 0
    metadata_mismatches: list[str] = []

    for text_path in text_paths:
        paper_id = text_path.stem
        predictions = _jsonl(predictions_dir / f"{paper_id}.jsonl")
        headings = [
            heading
            for row in predictions
            for heading in row["predicted_headings"]
        ]
        text = text_path.read_text(encoding="utf-8")
        document = build_document(text, headings)
        total_sections += len(document.sections)

        empty_indexes = [
            index
            for index, section in enumerate(document.sections)
            if not _body_after_title(document.read_section(section.id)).strip()
        ]
        if not empty_indexes:
            continue

        locations, _ = locate_headings(text, headings)
        if len(locations) != len(document.sections):
            metadata_mismatches.append(
                f"{paper_id}: locations={len(locations)} "
                f"sections={len(document.sections)}"
            )
            continue
        if any(
            location.char_start != section.char_start
            or location.matched_title != section.title
            for location, section in zip(locations, document.sections)
        ):
            metadata_mismatches.append(f"{paper_id}: location metadata differs")
            continue

        previous_gap = None
        empty_index_set = set(empty_indexes)
        for index, (section, location) in enumerate(
            zip(document.sections, locations)
        ):
            gap_chars = (
                section.char_end
                - section.char_start
                - len(location.matched_title)
            )
            if index not in empty_index_set:
                previous_gap = gap_chars
                continue

            next_section = (
                document.sections[index + 1]
                if index + 1 < len(document.sections)
                else None
            )
            child_ids = document.children(section.id)
            pos_frac = section.char_start / len(text) if text else 0.0
            context_before = text[max(0, section.char_start - 120):section.char_start]
            context_after = text[
                section.char_end:min(len(text), section.char_end + 120)
            ]
            classification, confidence, evidence = _classify_empty_section(
                title=section.title,
                pos_frac=pos_frac,
                match_level=location.match_level,
                is_standalone=location.is_standalone,
                next_section_id=next_section.id if next_section else None,
                child_ids=child_ids,
                context_before=context_before,
                context_after=context_after,
            )
            descendant_text = document.read_section(
                section.id,
                include_descendants=True,
            )
            records.append({
                "paper_id": paper_id,
                "section_id": section.id,
                "title": section.title,
                "depth": section.depth,
                "char_start": section.char_start,
                "pos_frac": pos_frac,
                "next_title": next_section.title if next_section else None,
                "gap_chars": gap_chars,
                "prev_gap": previous_gap,
                "match_level": location.match_level,
                "is_standalone": location.is_standalone,
                "context_before": context_before,
                "context_after": context_after,
                "children": child_ids,
                "include_descendants_chars": len(descendant_text),
                "include_descendants_has_content": bool(
                    _body_after_title(descendant_text).strip()
                ),
                "classification": classification,
                "confidence_tier": confidence,
                "classification_evidence": evidence,
            })
            previous_gap = gap_chars

    output_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    paper_counts = Counter(record["paper_id"] for record in records)
    class_counts = Counter(record["classification"] for record in records)
    front_records = [record for record in records if record["pos_frac"] < 0.15]
    front_papers = Counter(record["paper_id"] for record in front_records)
    with_children = [record for record in records if record["children"]]
    position_counts = {}
    for classification in sorted(class_counts):
        grouped = [
            record
            for record in records
            if record["classification"] == classification
        ]
        position_counts[classification] = {
            "lt_0.15": sum(record["pos_frac"] < 0.15 for record in grouped),
            "0.15_to_lt_0.75": sum(
                0.15 <= record["pos_frac"] < 0.75 for record in grouped
            ),
            "gte_0.75": sum(record["pos_frac"] >= 0.75 for record in grouped),
        }

    print("GATEG EMPTY SECTIONS")
    for record in records:
        print(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
    print(
        "GATEG PAPER COUNTS "
        + json.dumps(dict(sorted(paper_counts.items())), ensure_ascii=False)
    )
    print(
        "GATEG CLASS COUNTS "
        + json.dumps(dict(sorted(class_counts.items())), ensure_ascii=False)
    )
    print(
        "GATEG POS_FRAC BY CLASS "
        + json.dumps(position_counts, ensure_ascii=False)
    )
    print(
        f"GATEG FRONT15 count={len(front_records)} papers="
        + json.dumps(dict(sorted(front_papers.items())), ensure_ascii=False)
    )
    print(f"GATEG WITH_CHILDREN count={len(with_children)}")
    print(f"GATEG OUTPUT path={output_path} records={len(records)}")

    checks = {
        "papers_26": len(text_paths) == EXPECTED_PAPERS,
        "located_1082": total_sections == EXPECTED_LOCATED,
        "empty_79": len(records) == EXPECTED_EMPTY_DIRECT,
        "metadata_agrees": not metadata_mismatches,
        "descendant_reads_nonempty": all(
            record["include_descendants_has_content"]
            for record in with_children
        ),
        "model_furniture_37": class_counts["d_model_furniture"] == 37,
        "unknown_0": class_counts["unknown"] == 0,
    }
    if metadata_mismatches:
        print(f"GATEG METADATA MISMATCHES {metadata_mismatches}")
    for name, passed in checks.items():
        print(f"GATEG CHECK {name} status={'PASS' if passed else 'FAIL'}")
    passed = all(checks.values())
    print(f"GATEG SUMMARY status={'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


def gate_i(
    full_text_dir: Path,
    predictions_dir: Path,
    gate_g_path: Path,
) -> int:
    """Match the API's empty-section ids to TASK AA/AB's saved inventory."""
    expected_records = json.loads(gate_g_path.read_text(encoding="utf-8"))
    expected_pairs = [
        (record["paper_id"], record["section_id"])
        for record in expected_records
    ]

    text_paths = sorted(full_text_dir.glob("*.txt"))
    actual_pairs = []
    total_sections = 0
    for text_path in text_paths:
        document, _ = _saved_document(text_path, predictions_dir)
        total_sections += len(document.sections)
        actual_pairs.extend(
            (text_path.stem, section_id)
            for section_id in document.empty_sections()
        )

    expected_set = set(expected_pairs)
    actual_set = set(actual_pairs)
    missing = sorted(expected_set - actual_set)
    unexpected = sorted(actual_set - expected_set)
    checks = {
        "papers_26": len(text_paths) == EXPECTED_PAPERS,
        "sections_1082": total_sections == EXPECTED_LOCATED,
        "api_empty_79": len(actual_pairs) == EXPECTED_EMPTY_DIRECT,
        "inventory_79": len(expected_pairs) == EXPECTED_EMPTY_DIRECT,
        "api_ids_unique": len(actual_pairs) == len(actual_set),
        "inventory_ids_unique": len(expected_pairs) == len(expected_set),
        "id_sets_equal": actual_set == expected_set,
    }
    print(
        f"GATEI COUNTS api={len(actual_pairs)} inventory={len(expected_pairs)} "
        f"sections={total_sections}"
    )
    print(f"GATEI SETS missing={missing} unexpected={unexpected}")
    for name, passed in checks.items():
        print(f"GATEI CHECK {name} status={'PASS' if passed else 'FAIL'}")
    passed = all(checks.values())
    print(f"GATEI SUMMARY status={'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


def gate1b(
    model_path: Path,
    full_text_dir: Path,
    predictions_dir: Path,
) -> int:
    engine = get_engine(model_path)
    text_paths = sorted(full_text_dir.glob("*.txt"))
    identical_chunks = 0
    total_chunks = 0
    mismatches: list[str] = []
    unlocated_by_paper: Counter[str] = Counter()
    unlocated_examples: list[tuple[str, str, str]] = []

    for text_path in text_paths:
        paper_id = text_path.stem
        text = text_path.read_text(encoding="utf-8")
        generated = predict_chunks(text, engine)
        reference = _jsonl(predictions_dir / f"{paper_id}.jsonl")
        paper_identical = 0
        for new, old in zip(generated, reference):
            total_chunks += 1
            if (
                new.chunk_id == old["chunk_id"]
                and new.predicted_headings == old["predicted_headings"]
            ):
                identical_chunks += 1
                paper_identical += 1
            elif len(mismatches) < 10:
                mismatches.append(
                    f"{paper_id} chunk={old['chunk_id']} "
                    f"new={new.predicted_headings!r} old={old['predicted_headings']!r}"
                )
        if len(generated) != len(reference):
            total_chunks += abs(len(generated) - len(reference))
            mismatches.append(
                f"{paper_id} length new={len(generated)} old={len(reference)}"
            )

        headings = [
            heading
            for prediction in generated
            for heading in prediction.predicted_headings
        ]
        document = build_document(text, headings)
        unlocated_by_paper[paper_id] = len(document.unlocated)
        for item in document.unlocated:
            if len(unlocated_examples) < 3:
                unlocated_examples.append(
                    (paper_id, item.predicted_title, item.reason)
                )
        print(
            f"GATE1B {paper_id}: identical={paper_identical}/{len(reference)} "
            f"unlocated={len(document.unlocated)}",
            flush=True,
        )

    print(f"GATE1B SUMMARY chunks={identical_chunks}/{total_chunks}")
    print(f"UNLOCATED TOTAL={sum(unlocated_by_paper.values())}")
    for paper_id in sorted(unlocated_by_paper):
        print(f"UNLOCATED {paper_id}={unlocated_by_paper[paper_id]}")
    for paper_id, title, reason in unlocated_examples:
        print(f"UNLOCATED EXAMPLE paper={paper_id} title={title!r} reason={reason}")
    for mismatch in mismatches:
        print(f"MISMATCH {mismatch}")

    return 0 if identical_chunks == total_chunks == 1274 else 1


def _comma_separated_papers(value: str) -> list[str]:
    paper_ids = [paper_id.strip() for paper_id in value.split(",")]
    if not paper_ids or any(not paper_id for paper_id in paper_ids):
        raise argparse.ArgumentTypeError(
            "--papers must be a comma-separated list of non-empty paper ids"
        )
    return paper_ids


def _heading_sequence_bytes(headings: list[str]) -> bytes:
    """Encode a heading sequence without losing item boundaries or Unicode."""
    return json.dumps(
        headings,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def gate_j(
    model_path: Path,
    full_text_dir: Path,
    predictions_dir: Path,
    paper_ids: list[str],
) -> int:
    """Smoke-test the public model path against saved offline predictions."""
    engine = None
    passed_papers = 0

    for paper_id in paper_ids:
        text_path = full_text_dir / f"{paper_id}.txt"
        predictions_path = predictions_dir / f"{paper_id}.jsonl"
        text = text_path.read_text(encoding="utf-8")
        reference_rows = _jsonl(predictions_path)
        reference_headings = [
            heading
            for row in reference_rows
            for heading in row["predicted_headings"]
        ]

        started = time.perf_counter()
        if engine is None:
            engine = get_engine(model_path)

        captured_predictions = []
        original_iter_predictions = engine.iter_predictions

        def recording_iter_predictions(source_text: str):
            for prediction in original_iter_predictions(source_text):
                captured_predictions.append(prediction)
                yield prediction

        engine.iter_predictions = recording_iter_predictions
        try:
            model_document = recover_structure(text, model=model_path)
        finally:
            engine.iter_predictions = original_iter_predictions
        elapsed = time.perf_counter() - started

        model_headings = [
            heading
            for prediction in captured_predictions
            for heading in prediction.predicted_headings
        ]
        chunks_equal = (
            len(captured_predictions) == len(reference_rows)
            and all(
                prediction.chunk_id == row["chunk_id"]
                for prediction, row in zip(captured_predictions, reference_rows)
            )
        )
        headings_byte_equal = (
            _heading_sequence_bytes(model_headings)
            == _heading_sequence_bytes(reference_headings)
        )
        offline_document = build_document(text, reference_headings)
        document_equal = model_document.to_dict() == offline_document.to_dict()
        paper_passed = chunks_equal and headings_byte_equal and document_equal
        passed_papers += int(paper_passed)

        print(
            f"GATEJ {paper_id}: chunks={len(captured_predictions)}/"
            f"{len(reference_rows)} predicted_headings={len(model_headings)}/"
            f"{len(reference_headings)} "
            f"chunk_sequence={'EQUAL' if chunks_equal else 'MISMATCH'} "
            f"heading_bytes={'EQUAL' if headings_byte_equal else 'MISMATCH'} "
            f"to_dict={'EQUAL' if document_equal else 'MISMATCH'} "
            f"elapsed_seconds={elapsed:.3f}",
            flush=True,
        )

        if not headings_byte_equal:
            first_difference = next(
                (
                    index
                    for index, (actual, expected) in enumerate(
                        zip(model_headings, reference_headings)
                    )
                    if actual != expected
                ),
                min(len(model_headings), len(reference_headings)),
            )
            print(
                f"GATEJ DIFF {paper_id}: first_heading_index={first_difference}"
            )

    passed = passed_papers == len(paper_ids)
    print(
        f"GATEJ SUMMARY papers={passed_papers}/{len(paper_ids)} "
        f"status={'PASS' if passed else 'FAIL'}"
    )
    return 0 if passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="gate", required=True)

    one_a = subparsers.add_parser("gate1a")
    one_a.add_argument("--full-text-dir", type=Path, required=True)
    one_a.add_argument("--chunks-dir", type=Path, required=True)

    gate_a_parser = subparsers.add_parser("gateA")
    gate_a_parser.add_argument("--full-text-dir", type=Path, required=True)
    gate_a_parser.add_argument("--predictions-dir", type=Path, required=True)
    gate_a_parser.add_argument(
        "--baseline",
        type=Path,
        default=TASK_Y_BASELINE,
    )

    two = subparsers.add_parser("gate2")
    two.add_argument("--full-text-dir", type=Path, required=True)
    two.add_argument("--predictions-dir", type=Path, required=True)
    two.add_argument("--limit", type=int, default=26)

    gate_e_parser = subparsers.add_parser("gateE")
    gate_e_parser.add_argument("--full-text-dir", type=Path, required=True)
    gate_e_parser.add_argument("--predictions-dir", type=Path, required=True)

    gate_f_parser = subparsers.add_parser("gateF")
    gate_f_parser.add_argument("--full-text-dir", type=Path, required=True)
    gate_f_parser.add_argument("--predictions-dir", type=Path, required=True)

    gate_g_parser = subparsers.add_parser("gateG")
    gate_g_parser.add_argument("--full-text-dir", type=Path, required=True)
    gate_g_parser.add_argument("--predictions-dir", type=Path, required=True)
    gate_g_parser.add_argument("--output", type=Path, default=GATEG_OUTPUT)

    gate_i_parser = subparsers.add_parser("gateI")
    gate_i_parser.add_argument("--full-text-dir", type=Path, required=True)
    gate_i_parser.add_argument("--predictions-dir", type=Path, required=True)
    gate_i_parser.add_argument("--gate-g", type=Path, default=GATEG_OUTPUT)

    one_b = subparsers.add_parser("gate1b")
    one_b.add_argument("--model", type=Path, required=True)
    one_b.add_argument("--full-text-dir", type=Path, required=True)
    one_b.add_argument("--predictions-dir", type=Path, required=True)

    gate_j_parser = subparsers.add_parser("gateJ")
    gate_j_parser.add_argument("--model", type=Path, required=True)
    gate_j_parser.add_argument("--full-text-dir", type=Path, required=True)
    gate_j_parser.add_argument("--predictions-dir", type=Path, required=True)
    gate_j_parser.add_argument(
        "--papers",
        type=_comma_separated_papers,
        default=list(DEFAULT_GATEJ_PAPERS),
        help="comma-separated paper ids",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.gate == "gate1a":
        return gate1a(args.full_text_dir, args.chunks_dir)
    if args.gate == "gateA":
        return gate_a(args.full_text_dir, args.predictions_dir, args.baseline)
    if args.gate == "gate2":
        return gate2(args.full_text_dir, args.predictions_dir, args.limit)
    if args.gate == "gateE":
        return gate_e(args.full_text_dir, args.predictions_dir)
    if args.gate == "gateF":
        return gate_f(args.full_text_dir, args.predictions_dir)
    if args.gate == "gateG":
        return gate_g(args.full_text_dir, args.predictions_dir, args.output)
    if args.gate == "gateI":
        return gate_i(args.full_text_dir, args.predictions_dir, args.gate_g)
    if args.gate == "gateJ":
        return gate_j(
            args.model,
            args.full_text_dir,
            args.predictions_dir,
            args.papers,
        )
    return gate1b(args.model, args.full_text_dir, args.predictions_dir)


if __name__ == "__main__":
    raise SystemExit(main())
