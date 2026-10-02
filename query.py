#!/usr/bin/env python3
"""
query.py — Interactive query interface for the Gender Budget RAG system.

Usage:
    uv run python query.py "What is the Part A total in Union Budget 2023-24?"
    uv run python query.py --top-k 15 "Delhi Ladli Yojna 2024-25 allocation?"
"""

import argparse
import json
import logging
import sys

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Query the Gender Budget RAG system.")
    parser.add_argument("question", nargs="?", help="Question to answer (or use --question)")
    parser.add_argument("-q", "--question", dest="question_flag", help="Question to answer")
    parser.add_argument("--top-k", type=int, default=20, help="Number of chunks to retrieve")
    parser.add_argument("--index", default="index.npz", help="Path to index.npz")
    parser.add_argument("--chunks", default="chunks.json", help="Path to chunks.json")
    parser.add_argument("--show-chunks", action="store_true", help="Print retrieved chunk texts")
    args = parser.parse_args()

    question = args.question or args.question_flag
    if not question:
        print("Usage: uv run python query.py \"Your question here\"")
        sys.exit(1)

    try:
        from rag_assignment_civicdatalab.retrieve import retrieve
        from rag_assignment_civicdatalab.generate import generate_answer
    except ImportError as e:
        print(f"ERROR: {e}. Run 'uv sync' first.", file=sys.stderr)
        sys.exit(1)

    print(f"\nQuestion: {question}\n")

    # Retrieve
    try:
        chunks = retrieve(question, top_k=args.top_k, index_path=args.index, chunks_path=args.chunks)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Retrieval error: {e}", file=sys.stderr)
        sys.exit(1)

    if not chunks:
        print("No relevant chunks found.")
        sys.exit(0)

    print(f"Retrieved {len(chunks)} chunks:")
    for i, c in enumerate(chunks[:5], 1):
        src = c.get("source_file", "?")
        page = c.get("page")
        fy = c.get("financial_year", "")
        jur = c.get("jurisdiction", "")
        score = c.get("score", 0.0)
        pg_str = f" p.{page}" if page else ""
        print(f"  {i}. {src}{pg_str} | {jur} {fy} | score={score:.3f}")

    if args.show_chunks:
        print("\n--- Retrieved chunk texts ---")
        for i, c in enumerate(chunks, 1):
            print(f"\n[{i}] {c.get('source_file')} p.{c.get('page')} ({c.get('jurisdiction')} {c.get('financial_year')})")
            print(c.get("text", "")[:500])

    # Generate
    try:
        result = generate_answer(question, chunks)
    except RuntimeError as e:
        err_msg = str(e)
        if "rate" in err_msg.lower() or "429" in err_msg:
            print(f"\n{'='*60}")
            print("ERROR: Groq API rate limit hit after all retries.")
            print("Wait 1-2 minutes and try again, or switch GROQ_MODEL in .env")
            print(f"Details: {err_msg[:120]}")
        else:
            print(f"\nGeneration error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nGeneration error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'='*60}")
    if result.get("answered"):
        print(f"Answer: {result['answer']}")
    else:
        print("Answer: [Cannot answer — insufficient evidence in corpus]")

    sources = result.get("sources", [])
    if sources:
        print("\nSources:")
        for s in sources:
            f = s.get("file", "?")
            p = s.get("page")
            print(f"  - {f}" + (f" (page {p})" if p else ""))
    print()


if __name__ == "__main__":
    main()
