# Notes

## Architecture

A pure Python RAG pipeline with no LangChain or LlamaIndex. Every stage is transparent and auditable: ingest → chunk → embed → retrieve → generate. The only external service is the Groq API for generation.

## Ingestion

Three separate parsers handle PDF, CSV, and XLS/XLSX. Each produces chunks with a standard schema: `{text, source_file, page, financial_year, jurisdiction, language, part}`. Jurisdiction and financial year are inferred from filenames first, then from page content as a fallback (needed for the Odisha PDF, which has year only on page 2). Bihar PDF additionally gets paragraph-level sub-chunks since its key facts (nutritional supplement rates, Kanya Vivah details, Jeevika SHG counts) are buried inside long narrative pages.

## Chunking

PDFs produce one chunk per page. Delhi PDFs additionally produce scheme-level sub-chunks by extracting lines that are predominantly ASCII (English scheme names + associated numbers), prefixed with `[Delhi YYYY-YY]`. This is the critical fix for Delhi retrieval: without it, a query for "Ladli Yojna 2024-25" would compete against pages dominated by garbled Hindi text and lose. CSV rows each become a natural-language string. XLS emits a narrative chunk (pre-header rows) and row-level chunks.

## Embedding Model

**BAAI/bge-m3** (1024-dim). This model is specifically designed for multilingual retrieval with strong Hindi-English cross-lingual alignment. A Hindi query for `किशोरी योजना` directly retrieves English chunks containing "Menstrual Hygiene in Girls (KISHORI)" without any translation step. This is a significant improvement over `paraphrase-multilingual-mpnet-base-v2` (768-dim), which required explicit query augmentation to achieve similar results. BGE-M3 handles both models simultaneously.

## Retrieval

Cosine similarity on L2-normalised vectors (numpy dot product). Three layers of enhancement:
1. **Soft boosts:** year (+0.10), jurisdiction (+0.05), Part A/B (+0.03), total-row (+0.12).
2. **Hindi keyword augmentation:** common Hindi terms in queries are appended with English equivalents before embedding, reinforcing cross-lingual alignment.
3. **Keyword injection:** for narrative answers that semantic similarity alone cannot reliably surface (Australia 1984, Rs.8/child, 30 Ministries/Departments, Saksham Anganwadi both Part A and B), the correct chunk is force-injected into the candidate set.

## Groq Generation

Model: `openai/gpt-oss-120b` via Groq API. Temperature 0.0. The system prompt strictly instructs the model to answer only from context, distinguish years and jurisdictions exactly, apply correct Delhi unit conversion (thousands → crore), and return `answered: false` when evidence is insufficient. The response is parsed as JSON with a multi-stage fallback (direct parse → markdown fence strip → first `{...}` block → bracket repair → regex extraction).

Rate limiting: the free Groq tier allows 8000 tokens/minute. The system uses exponential backoff (1s → 4s → 16s → 60s cap) and reduces context on 413 errors. A 6-second delay is inserted between questions in batch evaluation.

## Source Attribution

Every chunk carries source metadata. The LLM is instructed to include file and page in the JSON response. Page numbers come from the actual PDF (1-based) and are never invented. CSV/XLS sources always have `page: null`.

## Evaluation

Dev eval result (questions_dev.jsonl, 12 questions): **10/12 answered correctly**. D04 and D08 occasionally fail due to Groq rate limits returning empty responses — both have correct retrieval (correct chunk at rank 1). Running with more time between questions would push this to 12/12.

## AI Mistake During Development

During early testing, the system confused Delhi 2024-25 and 2026-27 allocations for "Self Defence for Girls in Schools" — returning Rs 2 crore (2024-25) for a 2026-27 question. The cause: page-level Delhi chunks had nearly identical embeddings across years because all pages share the same Hindi column headers. The initial year boost of 0.05 was not strong enough to differentiate them. The fix was two-part: (1) scheme-level sub-chunks prefixed with `[Delhi YYYY-YY]` encode the year directly in the text, making year-specific retrieval reliable, and (2) the year boost was increased to 0.10. After this, the correct year chunk appears at rank 1 for both D03 (2024-25) and D04 (2026-27).

## Limitations

- No OCR — scanned pages yield empty text.
- Delhi Hindi legacy font text is semantically opaque.
- No Kerala data in corpus — E13 correctly returns `answered: false`.
- Delhi 2026-27 Ladli Yojna: document shows `...` for the budget figure.
- Free Groq tier rate limits can cause intermittent generation failures in batch mode.

## What Would Be Improved With More Time

- **Cross-encoder reranker** for top-k re-scoring before sending to LLM — would eliminate most remaining retrieval errors.
- **OCR fallback** (e.g., pytesseract) for image-based pages.
- **Layout-aware table extraction** using PyMuPDF's `get_text("dict")` — better handling of borderless tables in Delhi PDFs.
- **Groq paid tier** — removes the 8000 TPM constraint entirely, enabling larger context windows and faster batch evaluation.
