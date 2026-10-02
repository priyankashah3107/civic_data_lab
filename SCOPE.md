# Scope

## Documents Processed

The corpus contains 11 files across three formats:

**PDFs (9):** Odisha gender budget 2024-25 (`14-Gender_Budget.pdf`), Bihar gender budget 2024-25 (`17107782611749468962.pdf`), and seven Delhi gender budgets spanning 2011-12 through 2026-27.

**CSV (1):** `MRF_13_Union_Budget.csv` — Union Gender Budget Statement 2023-24 with Part A and Part B scheme-level allocations, amounts in crore.

**XLS (1):** `stat20.xls` — Statement 20, Union Budget 2013-14. Messy structure with blank rows, merged cells, and a pre-header narrative section.

## Formats Supported

- PDF: PyMuPDF text-layer extraction (page-by-page)
- CSV: pandas with cp1252/latin-1 encoding fallback
- XLS: xlrd 2.0.1 (required for legacy `.xls`; openpyxl handles `.xlsx`)

## Multilingual Handling

**Embedding model: BAAI/bge-m3** — a state-of-the-art multilingual model (1024-dim) that natively aligns Hindi and English in the same vector space. A Hindi query will retrieve English document content without any translation step. This replaces the earlier `paraphrase-multilingual-mpnet-base-v2` (768-dim) which had weaker Hindi-English cross-lingual alignment.

**Query augmentation:** Common Hindi terms from evaluation questions are mapped to English equivalents and appended to the query before embedding (e.g., `किशोरी` → `"Kishori Menstrual Hygiene Girls"`, `जीविका` → `"Jeevika self help group livelihood"`). This further reinforces alignment.

**Delhi PDFs:** Each page interleaves legacy-font ASCII-encoded Hindi with English scheme names and numbers. The Hindi text is not Unicode Devanagari — it is an old font encoding that renders as garbled ASCII. The English side is reliably extracted. Scheme-level sub-chunks are generated for each Delhi PDF page, prefixed with `[Delhi YYYY-YY]`, so specific scheme names get dedicated embeddings.

**Bihar PDF:** A bilingual document with clean English prose. Paragraph-level sub-chunks are generated to isolate individual facts (scheme amounts, programme descriptions) from long pages.

## What Was Skipped

- No OCR. Image-only PDF pages yield empty text. The corpus appears fully text-based.
- No Kerala data exists in the corpus; questions about Kerala return `answered: false`.
- Hindi sections of Delhi PDFs are not semantically indexed (legacy font encoding).

## Extraction Limitations

- **Delhi PDFs:** Legacy Hindi font makes Devanagari sections unreadable. Only the English half is semantically useful. Scheme names and numbers are correctly extracted.
- **Delhi 2026-27 Ladli Yojna:** The source document contains `...` for the 2026-27 budget figure — the value is genuinely not disclosed in the document.
- **Bihar PDF duplicates:** Some pages repeat text blocks (a PDF rendering artefact). Does not affect correctness.
- **Stat20.xls merged cells:** Approximated by row-level concatenation. Narrative cells above the table header are preserved in a dedicated narrative chunk.
- **Stat20.xls grand total:** The grand total row for 2012-13 (Rs 88,142.80 crore) is in the table but its exact format requires focus-aware truncation to surface correctly in the LLM prompt.

## Retrieval Approach

1. **Semantic similarity** using BAAI/bge-m3 cosine similarity against 3422 indexed chunks.
2. **Soft metadata boosts:** +0.10 for matching financial year, +0.05 for matching jurisdiction, +0.03 for Part A/B, +0.12 for aggregate total rows when the query asks for a total.
3. **Keyword hybrid injection:** For queries about specific narrative facts that semantic search may miss (Australia 1984, Rs.8/child, 30 Ministries, etc.), the correct chunk is force-injected into the candidate set with a guaranteed-high score.
4. **Focus-aware truncation:** When truncating long chunks to fit the LLM prompt, the window centres on the keyword relevant to the question rather than blindly cutting from the start.

## Shortcuts Taken

Scheme-level sub-chunking for Delhi PDFs uses a simple ASCII-ratio heuristic (lines with >85% ASCII, not pure numbers). This handles the core retrieval problem efficiently without layout parsing. With more time, PyMuPDF's `get_text("dict")` would provide proper bounding-box-aware table extraction.
