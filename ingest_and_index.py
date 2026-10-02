#!/usr/bin/env python3
"""
ingest_and_index.py — Build the RAG index from the corpus.

Usage:
    uv run python ingest_and_index.py
    uv run python ingest_and_index.py --data-dir corpus --index index.npz --chunks chunks.json
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="Ingest corpus and build the vector index.")
    parser.add_argument("--data-dir", default="corpus", help="Directory containing corpus files")
    parser.add_argument("--index", default="index.npz", help="Output path for index.npz")
    parser.add_argument("--chunks", default="chunks.json", help="Output path for chunks.json")
    args = parser.parse_args()

    try:
        from rag_assignment_civicdatalab.index import build_index
    except ImportError as e:
        print(f"ERROR: Could not import modules: {e}", file=sys.stderr)
        print("Make sure you have run: uv sync", file=sys.stderr)
        sys.exit(1)

    try:
        embeddings, chunks = build_index(
            data_dir=args.data_dir,
            index_path=args.index,
            chunks_path=args.chunks,
        )
        print(f"\nIndex built successfully: {len(chunks)} chunks, embeddings shape {embeddings.shape}")
    except Exception as e:
        print(f"ERROR: Indexing failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
