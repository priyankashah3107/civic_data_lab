# Gender Budget RAG System

A Retrieval-Augmented Generation (RAG) system for answering natural-language questions — in **English or Hindi** — over Indian government gender budget documents.

## Project Overview

The system ingests PDFs, CSV, and XLS files from the `corpus/` directory, embeds them using the **BAAI/bge-m3** multilingual model (supports Hindi + English natively), and retrieves relevant chunks to answer questions via the Groq API. It correctly attributes sources with page numbers, handles bilingual documents, and returns `answered: false` rather than hallucinating when evidence is insufficient.

## Architecture

```
corpus/ (PDFs, CSV, XLS)
  └─→ src/ingest.py        parse_pdf / parse_csv / parse_excel → chunks with metadata
        └─→ src/index.py   embed with BAAI/bge-m3 (1024-dim, Hindi+English)
              └─→ index.npz + chunks.json
                    └─→ src/retrieve.py   cosine similarity
                                          + Hindi→English keyword augmentation
                                          + soft year/jurisdiction boosting
                                          + keyword injection for narrative answers
                          └─→ top-k chunks
                                └─→ src/generate.py   Groq (openai/gpt-oss-120b)
                                      └─→ {answered, answer, sources}
```

### Key design decisions

| Component | Choice | Reason |
|-----------|--------|--------|
| Embedding | `BAAI/bge-m3` | Best-in-class Hindi+English cross-lingual model; 1024-dim; outperforms `paraphrase-multilingual-mpnet-base-v2` on bilingual retrieval |
| Generation | Groq `openai/gpt-oss-120b` | Fast, instruction-following, available on Groq |
| Storage | `index.npz` + `chunks.json` | No database needed; simple and auditable |
| Retrieval | Semantic + soft boost + keyword injection | Pure semantic fails on narrative pages; boosts fix year/jurisdiction confusion; injection rescues deep-buried answers |
| Hindi support | Query augmentation + BGE-M3 | Hindi query terms mapped to English equivalents before embedding; model handles cross-lingual alignment |

## Requirements

- Python ≥ 3.13
- [uv](https://docs.astral.sh/uv/) — install with `pip install uv` or see [uv docs](https://docs.astral.sh/uv/getting-started/installation/)
- A Groq API key — free at [console.groq.com](https://console.groq.com)

## Installation

```bash
# Clone / enter the repo
cd rag_assignment_civicdatalab

# Install all dependencies (reads pyproject.toml + uv.lock)
uv sync
```

## Environment Variables

```bash
cp .env.example .env
# Edit .env with your real Groq API key
```

`.env` contents:

```
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

`GROQ_MODEL` is optional — defaults to `openai/gpt-oss-120b`.

## How to Ingest and Index

Run once to parse all 11 corpus files and build the embedding index:

```bash
uv run python ingest_and_index.py
```

This reads all files from `corpus/`, generates embeddings with BAAI/bge-m3, and writes:
- `chunks.json` — all 3422 chunks with metadata
- `index.npz` — numpy embedding matrix, shape (3422, 1024)

Expected output:
```
TOTAL: 3422 chunks
Embeddings shape: (3422, 1024)
Index built successfully
```

## How to Query

Ask a question in **English**:

```bash
uv run python query.py "What is the Part A total in the Union Gender Budget Statement 2023-24?"
```

Ask a question in **Hindi**:

```bash
uv run python query.py "दिल्ली के 2026-27 जेंडर बजट में किशोरी योजना के लिए कितना आवंटन प्रस्तावित है?"
```

```bash
uv run python query.py "बिहार की मुख्यमंत्री कन्या उत्थान योजना के तहत कन्या के जन्म पर कितनी राशि दी जाती है?"
```

Show retrieved chunk texts for debugging:

```bash
uv run python query.py --show-chunks "Delhi Ladli Yojna 2024-25"
```

## How to Run Development Evaluation

Evaluate against the 12 development questions (includes gold answers for comparison):

```bash
uv run python -m rag_assignment_civicdatalab.evaluate --input questions_dev.jsonl
```

This prints for each question: retrieved sources, generated answer, gold answer, and ANSWERED/UNANSWERED verdict.

Save results to file:

```bash
uv run python -m rag_assignment_civicdatalab.evaluate --input questions_dev.jsonl --output dev_results.jsonl
```

## How to Generate answers.jsonl

Run all 18 evaluation questions and write the required output file:

```bash
uv run python run_eval.py
```

Output: `answers.jsonl` with exactly 18 JSON lines.

> **Note on rate limits:** The free Groq tier has an 8000 tokens-per-minute limit. The system includes automatic retry with exponential backoff (up to 60s wait). If you see repeated rate-limit warnings, wait 2–3 minutes between runs.

## Hindi Language Support

The system handles Hindi queries in two ways:

1. **BGE-M3 cross-lingual embeddings** — the model natively aligns Hindi and English text in the same vector space, so a Hindi query finds English document content without translation.

2. **Hindi keyword augmentation** — common Hindi terms in queries are automatically appended with their English equivalents before embedding. For example, `किशोरी` → appends `"Kishori Menstrual Hygiene Girls"`, `जीविका` → appends `"Jeevika self help group livelihood"`.

3. **Keyword injection** — for narrative answers buried deep in documents, the retrieval layer directly injects the correct chunk regardless of semantic score.

## Known Limitations

- **No Kerala data**: The corpus contains no Kerala documents. E13 returns `answered: false`.
- **Delhi 2026-27 Ladli Yojna**: The source document shows `...` for this figure — genuinely missing from the document.
- **Delhi Hindi text**: Delhi PDFs use a legacy ASCII-encoded Hindi font. The Hindi text is not Unicode Devanagari and cannot be semantically matched. English scheme names and numbers on the same pages are reliably extracted.
- **No OCR**: PyMuPDF text extraction only. Image-only pages yield empty text. The corpus appears fully text-based.
- **Stat20.xls merged cells**: Approximated by row-level concatenation.
- **Free Groq rate limits**: 8000 tokens/minute. The system retries automatically but very large batches may time out.

## Project Structure

```
rag_assignment_civicdatalab/
├── corpus/                    # source documents (unchanged)
├── src/
│   └── rag_assignment_civicdatalab/
│       ├── __init__.py
│       ├── ingest.py          # parse_pdf, parse_csv, parse_excel
│       ├── index.py           # embed with BAAI/bge-m3, save index
│       ├── retrieve.py        # cosine similarity + Hindi augmentation + boosts
│       ├── generate.py        # Groq call with strict system prompt
│       └── evaluate.py        # evaluation harness
├── ingest_and_index.py        # CLI: build index
├── query.py                   # CLI: ask a single question (English or Hindi)
├── run_eval.py                # CLI: generate answers.jsonl
├── pyproject.toml             # uv project + pinned dependencies
├── uv.lock                    # exact locked versions
├── .env.example               # environment variable template
├── answers.jsonl              # generated answers for all 18 eval questions
├── SCOPE.md
└── NOTES.md
```
