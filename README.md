# Gender Budget RAG System

A Retrieval-Augmented Generation (RAG) system for answering natural-language questions over Indian government gender budget documents.

## Project Overview

The system ingests PDFs, CSV, and XLS files from the `corpus/` directory, embeds them using a multilingual sentence-transformer model, and retrieves relevant chunks to answer questions via the Groq API. It supports English and Hindi questions, handles bilingual documents, and correctly attributes sources with page numbers.

## Architecture

```
corpus/ (PDFs, CSV, XLS)
  └─→ src/ingest.py       parse_pdf / parse_csv / parse_excel
        └─→ chunks (list[dict] with metadata)
              └─→ src/index.py    embed with paraphrase-multilingual-mpnet-base-v2
                    └─→ index.npz + chunks.json
                          └─→ src/retrieve.py   cosine similarity + soft year/jurisdiction boosting
                                └─→ top-k chunks
                                      └─→ src/generate.py   Groq (llama-3.3-70b-versatile)
                                            └─→ {answered, answer, sources}
```

## Requirements

- Python ≥ 3.13
- [uv](https://docs.astral.sh/uv/) package manager
- A Groq API key (free tier available at [console.groq.com](https://console.groq.com))

## Installation

```bash
# Clone and enter the repo
cd rag_assignment_civicdatalab

# Install all dependencies (reads pyproject.toml + uv.lock)
uv sync
```

## Environment Variables

Copy `.env.example` to `.env` and fill in your Groq API key:

```bash
cp .env.example .env
```

Edit `.env`:

```
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

`GROQ_MODEL` is optional — defaults to `llama-3.3-70b-versatile`.

## How to Ingest and Index

Run once to parse all corpus files and build the embedding index:

```bash
uv run python ingest_and_index.py
```

This reads all files from `corpus/`, generates embeddings, and writes:
- `chunks.json` — all chunks with metadata
- `index.npz` — numpy embedding matrix

Expected output: ~3000 chunks, embeddings shape `(3008, 768)`.

## How to Query

Ask a single question:

```bash
uv run python query.py "What is the Part A total in the Union Gender Budget Statement 2023-24?"
```

Hindi queries work too:

```bash
uv run python query.py "दिल्ली के 2026-27 जेंडर बजट में किशोरी योजना के लिए कितना आवंटन प्रस्तावित है?"
```

## How to Run Development Evaluation

Evaluate against the 12 development questions (with gold answers for comparison):

```bash
uv run python -m rag_assignment_civicdatalab.evaluate --input questions_dev.jsonl
```

This prints for each question:
- Retrieved source files and pages
- Generated answer
- Gold answer
- Whether the question was answered

To save dev results to a file:

```bash
uv run python -m rag_assignment_civicdatalab.evaluate --input questions_dev.jsonl --output dev_results.jsonl
```

## How to Generate answers.jsonl

Run all 18 evaluation questions and write the required output file:

```bash
uv run python run_eval.py
```

Output: `answers.jsonl` with exactly 18 JSON lines, one per evaluation question.

## Known Limitations

- **No Kerala data**: The corpus does not contain Kerala documents. E13 (Kerala 2024-25 total) returns `answered: false`.
- **Delhi 2026-27 Ladli Yojna**: The source document shows `...` for this budget figure; it cannot be retrieved.
- **Hindi text in Delhi PDFs**: Legacy-font ASCII encoding makes Hindi sections unreadable semantically. English scheme names and numbers are reliably extracted.
- **No OCR**: Image-only PDF pages would yield empty text. The corpus appears fully text-based.
- **Stat20.xls merged cells**: Approximated by row-level concatenation.
- **Bihar PDF duplicates**: Some pages have duplicate text blocks (PDF artefact); does not affect correctness.

## Project Structure

```
rag_assignment_civicdatalab/
├── corpus/                    # source documents (unchanged)
├── src/
│   └── rag_assignment_civicdatalab/
│       ├── __init__.py
│       ├── ingest.py          # parse_pdf, parse_csv, parse_excel
│       ├── index.py           # embed chunks, save index.npz + chunks.json
│       ├── retrieve.py        # cosine similarity + soft metadata reranking
│       ├── generate.py        # Groq API call with strict system prompt
│       └── evaluate.py        # evaluation harness for dev/eval questions
├── ingest_and_index.py        # CLI: build the index
├── query.py                   # CLI: ask a single question
├── run_eval.py                # CLI: generate answers.jsonl
├── pyproject.toml             # uv project + pinned dependencies
├── uv.lock                    # exact locked versions
├── .env.example               # environment variable template
├── answers.jsonl              # generated answers for all 18 eval questions
├── SCOPE.md                   # scope and limitations
└── NOTES.md                   # architecture and development notes
```
