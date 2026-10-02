# Scope

## Documents Processed

The corpus contains 11 files across three formats:

- **9 PDFs**: Odisha gender budget 2024-25 (`14-Gender_Budget.pdf`), Bihar gender budget 2024-25 (`17107782611749468962.pdf`), and seven Delhi gender budgets spanning 2011-12 through 2026-27.
- **1 CSV**: `MRF_13_Union_Budget.csv` — the Union Gender Budget Statement 2023-24 with Part A and Part B scheme-level rows.
- **1 XLS**: `stat20.xls` — Statement 20, Union Budget 2013-14, with messy structure and a narrative note section.

## Formats Supported

PDF extraction uses PyMuPDF (text-based extraction). CSV parsing uses pandas with cp1252/latin-1 fallback encoding. XLS parsing uses xlrd 2.0.1 (required for legacy `.xls`; openpyxl handles `.xlsx`).

## Multilingual Handling

Delhi PDFs are bilingual: each page interleaves legacy-font ASCII-encoded Hindi with English scheme names and numbers. The Hindi text is not proper Unicode Devanagari — it is legacy-font ASCII that cannot be semantically matched. The system handles this by: (1) using the `paraphrase-multilingual-mpnet-base-v2` model which provides cross-lingual embeddings for Hindi–English queries, and (2) extracting scheme-level sub-chunks from Delhi PDFs that isolate English scheme name + numbers, making them retrievable via both English and Hindi queries.

Bihar PDF is bilingual but the English section is clean and parseable. The bilingual Hindi sections are not reliably extractable.

## What Was Skipped

No OCR was applied (PyMuPDF text extraction only). Scanned-image pages would yield empty text. The corpus appears to be text-based throughout. Merged-cell information in XLS is approximated by including all non-empty cells per row.

## Extraction Limitations

- Delhi PDFs use a legacy Hindi font that renders as ASCII garbled text. Scheme names and numbers are present on the English side and are reliably extracted.
- The 2026-27 Delhi PDF shows `...` for some Ladli Yojna budget columns, indicating the figure was not disclosed in the document.
- `stat20.xls` has blank rows and merged cells. The header row detection heuristic (first row with ≥3 non-empty strings) correctly identifies the real header at row 10.
- The Bihar PDF has duplicate text blocks on some pages (a PDF artefact).
- No Kerala data exists in the corpus; questions about Kerala will correctly return `answered: false`.

## Retrieval Approach

Semantic retrieval with `paraphrase-multilingual-mpnet-base-v2` (768-dim) plus soft metadata boosting: +0.10 for matching financial year, +0.05 for matching jurisdiction, +0.03 for Part A/B match. Boosts are additive and small enough that a correct chunk is never eliminated — only re-ranked upward when metadata aligns with the query.

## Shortcuts Taken

Scheme-level sub-chunking is applied to Delhi PDFs using a heuristic (lines with >85% ASCII and not pure numbers). This handles the key retrieval problem for Delhi bilingual tables but may occasionally create noisy sub-chunks from header text.
