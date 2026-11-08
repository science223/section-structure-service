# 62-paper section-heading benchmark: gold labels

These are the human gold labels behind the accuracy numbers in the main README: 62 papers, 2,072 text chunks and 1,974
gold headings. They let anyone check the reported scores and evaluate another system on the same papers. The papers
themselves are not redistributed. The file holds arXiv ids, PDF hashes, chunk offsets, text hashes and the heading
strings, and no paper text.

## What is in `gold_labels_62.jsonl`

One line per paper:

| field | meaning |
|---|---|
| `paper_id` | benchmark id (A–D: the first 26 papers; `AKC_…`: the 36 added later, all arXiv 2601/2602) |
| `arxiv_id`, `arxiv_version` | the arXiv paper and the exact version whose PDF was used |
| `doi` | set only for `B7_GNN_Review`: the PDF is the journal version (AI Open), not an arXiv version |
| `version_source` | how the version is known: pinned at download, or the arXiv stamp printed in the PDF |
| `pdf_sha256` | SHA-256 of the PDF the text was extracted from |
| `n_chars` | length of the extracted text |
| `chunks` | the 500-word chunks, in order (below) |

Each chunk:

| field | meaning |
|---|---|
| `chunk_id`, `char_start`, `char_end` | the chunk's slice of the extracted text |
| `is_toc` | a contents-page chunk; not scored |
| `in_main_table` | false for the 11 chunks that at least one LLM API refused; the main table leaves them out for every system |
| `text_sha256` | SHA-256 of the chunk's text (UTF-8), to check that you extracted the same text |
| `gold` | the gold headings in this chunk: `heading` (the string scored), `char_pos` (its offset in the extracted text) and `level` |

Six headings straddle a chunk boundary. They start in their chunk and run into the next one, and are scored in the
chunk where they start.

## How the labels were made

Human annotators adjudicated the headings of every paper. Agreement with an independent second annotator (heading-set
F1) is 0.930 on 10 of the first 26 papers and 0.860 on 6 of the 36 added ones. The released tagger's weights and
decoder were fixed on separate silver-labelled data before it was scored on these labels.

## Reproducing the text and the scores

```bash
pip install -e ".[pdf,lite]"
python benchmark/fetch_pdfs.py pdfs/            # 61 arXiv PDFs; B7 from the publisher (DOI in the file)
python benchmark/make_chunks.py pdfs/ chunks/   # extract, cut and check every chunk's hash
python benchmark/predict_tagger.py jev_j01b_a_q5b_onnx/ chunks/ preds/ --threads 4
python benchmark/score.py preds/
```

`jev_j01b_a_q5b_onnx/` is the unzipped model asset from the Releases page. `chunks/` holds the papers' text, for your
own use.

**Use Linux (or WSL or Docker) for the extraction.** With pypdfium2 5.8.0, 5.13.0 or 5.15.1 on Linux, all 62 papers reproduce
every chunk byte for byte. On Windows (5.8.0 and 5.13.0 tested), 59 do. AKC_2601_01473, AKC_2601_21611 and AKC_2602_03887
then differ in spacing and repeated glyphs inside figure and math text, which shifts the character positions in two of
them.

`score.py` reads one `<paper_id>.jsonl` per paper, one line per chunk: `{"chunk_id": 3, "predicted_headings": [...]}`.
The metric is strict heading micro-F1:
- headings are compared after lower-casing and collapsing whitespace, with punctuation and numbering kept and no
  fuzzy matching;
- a heading predicted in the chunk next to its gold chunk still counts (±1-chunk tolerance);
- contents-page chunks are skipped.

By default `score.py` scores the main-table chunks; `--all-chunks` scores all 2,072. On the saved predictions it
reproduces the reported table exactly for all 24 systems scored on the 62 papers, for example 0.9332 for the released
tagger, 0.9116 for Claude Opus 5.5 and 0.6075 for the regex baseline.

To evaluate another system, give it each chunk's `text` from `chunks/`, ask for the headings that start in the chunk,
and write them in that format.

## Licence

The labels in this directory are released under CC BY 4.0 (see `LICENSE` here). The scripts are under the repository's
MIT licence. The papers belong to their authors and are not included.
