# Notes

## Architecture

The system is a pure Python RAG pipeline: ingestion → chunking → embedding → retrieval → Groq generation. No LangChain or LlamaIndex is used. All components are transparent and auditable.

## Ingestion

Three separate parsers handle PDF, CSV, and XLS/XLSX. Each emits chunks with a standard schema: `{text, source_file, page, financial_year, jurisdiction, language, part}`. Jurisdiction and financial year are inferred from filenames first, then from page content as a fallback (required for Odisha PDF which has year only on page 2).

## Chunking

PDFs produce one chunk per page. Delhi PDFs additionally produce scheme-level sub-chunks by extracting lines that are predominantly ASCII (English scheme names + numbers), prefixed with `[Delhi YYYY-YY]`. This dramatically improves retrieval for specific scheme queries since the page-level chunks are diluted by Hindi garbled text. CSV rows each become a natural-language string preserving Part A/B, ministry, scheme name, and all year columns. XLS emits a narrative chunk (pre-header rows) and row-level chunks (post-header).

## Embeddings

Model: `paraphrase-multilingual-mpnet-base-v2` (768-dim). This model provides cross-lingual alignment between Hindi and English without translation, enabling Hindi queries to retrieve English content. Embeddings are stored in `index.npz` (float32) and metadata in `chunks.json`.

## Retrieval

Cosine similarity on L2-normalised vectors (numpy dot product). Soft boosts: +0.10 for matching financial year, +0.05 for matching jurisdiction, +0.03 for Part A/B. Boosts are intentionally small to avoid eliminating correct answers from adjacent years or jurisdictions.

## Groq Generation

Model: `llama-3.3-70b-versatile` (configurable via `GROQ_MODEL`). Temperature 0.0. The system prompt strictly instructs the model to answer only from context, distinguish years and jurisdictions exactly, show unit conversion working, and return `answered: false` when evidence is insufficient. The response is parsed as JSON with fallback extraction.

## Source Attribution

Every chunk carries source metadata. The LLM is instructed to preserve file and page references in the JSON response. Page numbers come from the actual PDF page (1-based) and are never invented for CSV/XLS.

## Evaluation

Running `uv run python -m rag_assignment_civicdatalab.evaluate --input questions_dev.jsonl` prints each question's retrieved sources, generated answer, and gold answer for comparison. The `run_eval.py` script generates `answers.jsonl` with exactly 18 lines.

## AI Mistake Observed During Development

During early testing, the system confused Delhi 2024-25 and 2026-27 allocations for "Self Defence for Girls in Schools" — it returned Rs 2 crore (2024-25 value) for a 2026-27 question. The cause was that page-level Delhi PDF chunks have nearly identical embeddings across years (all contain the same Hindi headers and English column labels), so year-based re-ranking was insufficient with small boost weights. The fix was (1) introducing scheme-level sub-chunks prefixed with `[Delhi YYYY-YY]` which encode the year directly into the text, and (2) increasing the year boost from 0.05 to 0.10. After this change, the correct year chunk appears at rank 1 for both D03 and D04.

## Limitations

- No OCR; scanned pages yield empty text.
- Hindi legacy font text is unreadable semantically.
- Kerala data is absent from the corpus; those questions are correctly marked unanswered.
- The 2026-27 Ladli Yojna figure is `...` in the source document.
- Stat20.xls merged-cell context is approximated.

## What Would Be Improved With More Time

- Pairwise reranker (cross-encoder) for top-k re-scoring
- OCR fallback for image-based pages
- Better table extraction using PyMuPDF's `get_text("dict")` for layout-aware parsing
- Query expansion/translation for Hindi queries to improve recall
