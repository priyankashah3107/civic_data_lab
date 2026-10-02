#!/usr/bin/env python3
"""
run_eval.py — Generate answers.jsonl for all 18 evaluation questions.

Usage:
    uv run python run_eval.py
    uv run python run_eval.py --input questions_eval.jsonl --output answers.jsonl

Output: answers.jsonl with exactly 18 lines, one JSON object per question.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


def main():
    parser = argparse.ArgumentParser(description="Run all evaluation questions and write answers.jsonl")
    parser.add_argument("--input", default="questions_eval.jsonl")
    parser.add_argument("--output", default="answers.jsonl")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--index", default="index.npz")
    parser.add_argument("--chunks", default="chunks.json")
    args = parser.parse_args()

    try:
        from rag_assignment_civicdatalab.evaluate import run_evaluation
    except ImportError as e:
        print(f"ERROR: {e}. Run 'uv sync' first.", file=sys.stderr)
        sys.exit(1)

    results = run_evaluation(
        input_path=args.input,
        output_path=args.output,
        top_k=args.top_k,
        index_path=args.index,
        chunks_path=args.chunks,
    )

    # Final validation
    print(f"\n=== Validation ===")
    print(f"Total results: {len(results)}")

    ids_seen = set()
    errors = []
    for r in results:
        qid = r.get("question_id")
        if qid in ids_seen:
            errors.append(f"Duplicate question_id: {qid}")
        ids_seen.add(qid)

        if not r.get("answered") and r.get("answer"):
            errors.append(f"{qid}: answered=false but answer is non-empty")

    if errors:
        print("VALIDATION ERRORS:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("All validation checks passed.")

    print(f"\nDone. answers.jsonl written to {args.output}")


if __name__ == "__main__":
    main()
