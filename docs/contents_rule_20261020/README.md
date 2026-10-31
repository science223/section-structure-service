# Contents pages, page headers and IEEE numbering (2026-10-20)

All of this was found by checking real outlines after the agent acceptance run, not by a user report. The model and
its predicted headings are unchanged; what changed is how predicted headings become sections
(`tidy_headings` in `paper_structure/service.py`, `outline_numbering` in `paper_structure/tree.py`).

## 1. Contents pages (commit 4ffd531)

RFC 9110: six contents-page entries (`13.2`, `13.2.1`, `14.6`, `15`, `Acknowledgements`, `Index`) still held the
section ids, so `read_section("13.2")` returned one contents line. The model had marked those contents lines but not
the headings in the text. A seventh entry, `15.2.2.` (the page was cut there at a chunk boundary), took the id of the
real `15.2.2. 101 Switching Protocols`.

Beyond dropping entries whose title is a later section:

- an empty entry whose title comes later only as an unmarked line is **moved** to that line;
- an entry whose body is mostly further contents lines (the rest of the page) is part of the run, and is dropped
  when its id belongs to a later section;
- once a run is found, headings are located with the page blanked out. Without this the locator, which prefers the
  earliest of equally good matches, puts the remaining real headings back on the contents lines whenever the page
  lies past the first 5% of the text (an appendix contents page). The previous rule had this problem too.

## 2. Page headers and IEEE numbering (this commit)

- **Running page headers** the model marked: 32 `REFERENCES` headers and numbered headers such as `3.2 Stage 2 ...`
  (four of them; `3.2` itself was 66 characters, its text in `3.2#2`). Sections with the same title and the same
  printed number keep one, the first that is not a contents line or a header right above its heading; of three or
  more same-titled sections in a row one stays. A first version kept the first occurrence and so, in DeepSeek-V3 and
  The Prompt Report, kept a single dotted contents line and dropped the real `2.1.1` / `2.3.1`; caught by the
  comparison below, fixed, and covered by tests.
- **IEEE layout** (Roman sections, letters restarting in each): letters belong to their Roman section (`II.A`,
  depth 2) and `1)` to the letter (`II.A.1`). Only when at least two Roman headings run in order; other layouts are
  numbered exactly as before (tested).
- **Out-of-order contents lines**: `Authors' Addresses` was predicted ahead of the other contents lines of its chunk,
  so it could not be located; it now goes to its line after the page. A page header predicted on the contents page
  (`RFC 9110 HTTP Semantics June 2022`, on every page) is dropped instead of moved.

## Correction

The first version of this note said "RFC 9110 after the change: 249 sections, 13 entries dropped, 6 moved". That
was measured before the blanking step was added, and I did not measure again before committing 4ffd531: with
blanking, `Authors' Addresses` lost its contents line and was not located (248 sections). It had not been located
before today either (044b940), so nothing was lost against the previous release, but the number was wrong.
Now: **249 sections, 14 entries dropped (one is the page header), 7 moved, no repeated ids, 3 headings not
located** (`Appendices`, which is not in the text, and two predictions that joined several headings into one).

## The same model output on 26 long documents

The texts are 26 surveys and technical reports (65,000-380,000 characters) from the benchmark's 2026-05 archive.
None of them is a benchmark test paper: their arXiv ids were checked against all 126 AKC ids in the benchmark
repository. The model ran once per text; every version below works on the same cached predictions.

Contents rule, 044b940 against 4ffd531 (`surveys26_old_vs_new.json`, `scripts/compare_contents_rule.py`): 23 of 26
identical; 3 changed, all for the better (a contents entry with page numbers held the id and the real section was
`#2`: DeepSeek-V3 `3.2.3`, Large Multimodal Reasoning Models `3.2.1`, The Prompt Report `1` and `1.1`).

This commit against 4ffd531 (`surveys26_4ffd531_vs_this.json`, `scripts/dump_outlines.py`): 19 of 26 identical.

| document | sections | repeated ids | deepest level |
|---|---|---|---|
| RAG for LLMs (IEEE) | 36 → 36 | 18 → 0 | 1 → 3 |
| Large Language Models: A Survey (IEEE) | 49 → 49 | 24 → 1 | 1 → 3 |
| Internal Consistency (IEEE) | 51 → 51 | 32 → 5 | 1 → 2 |
| Foundation Models in Medicine (IEEE) | 60 → 60 | 27 → 8 | 1 → 3 |
| Large Multimodal Reasoning Models | 96 → 43 (53 page headers) | 54 → 1 | 3 → 3 |
| DeepSeek-V3 | 54 → 53 (one contents line) | 1 → 0 | 3 → 3 |
| The Prompt Report | 140 → 139 (one contents line) | 2 → 1 | 4 → 4 |

The repeated ids left in the IEEE surveys come from Roman headings the model did not mark (the letters of a missing
section `VI` join section `V`); the tool cannot fix those.

## 3. A scan of the 26 benchmark v1 PDFs, and two corrections it led to (about 05:00)

`scripts/scan_outlines.py` over the 26 v1 papers (`scan_v1pdfs26.json`; these were used for diagnosis before, not
test papers) lists repeated ids, empty or very short leaf sections and unlocated headings. 19 are clean or have a
few unlocated headings. The rest:

- **D2 (LLM research ideas)**: every example idea in the appendix repeats the same numbered headings ("4.
  Step-by-Step Experiment Plan", "5. Test Case Examples"). The repeat rule above took the later ones for page headers
  and merged them away (3 of them). **Corrected**: a repeat is a page header only if no *numbered* heading of the same
  or a higher level comes between it and the last one kept (page headers repeat inside their own section, with only
  deeper headings in between); a repeat after "Q Example Idea ..." is a new instance and stays (`4#2`). Unnumbered
  headings do not count as separators: in the Large Multimodal Reasoning survey a figure caption the model marked
  sat between `3.1` and its page header.
- **A3 (GPT-4 report)**: the PDF holds the report and its system card, each with its own `1 Introduction`. The report's
  is at 0.3% of the text (short abstract, author list at the end), so "in the first 5% while a namesake is past it"
  made it a contents line and it was dropped. **Corrected**: in the first 5% a line is a contents line only if it also
  has no body or its next line is the title of a later section (a contents line is followed by the next contents
  line; a heading is followed by text).
- **C6 (PaLM)**: table column headers ("PaLM 540B", "T5 XXL", "8B") marked as headings give a row of empty sections.
  A model error; no rule for it. Appendix headings `H.2` and `H.3` that were predicted on the contents page now
  reach their lines (the out-of-order rule of part 2).
- **C7 (Sparks of AGI)**: the dotted contents line "2.2.1 Image generation beyond memorization . . . 16" goes, the real
  2.2.1 stays. Its repeated "GPT-4 vs ChatGPT" figure labels stay (unnumbered, with text under them).

The same model output, 4ffd531 against the final code: 23 of the 26 v1 PDFs identical (C6 and C7 as above; D7, an
IEEE survey: repeated ids 18 -> 0, depth 1 -> 3), `v1pdfs26_4ffd531_vs_this.json`; the 26 surveys as in part 2
(`surveys26_4ffd531_vs_this.json`, regenerated with the final code); RFC 9110: 249 sections, no repeated ids.
