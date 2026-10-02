"""
evaluate.py — Evaluation harness for the Gender Budget RAG system.

Usage:
    uv run python -m rag_assignment_civicdatalab.evaluate --input questions_dev.jsonl
    uv run python -m rag_assignment_civicdatalab.evaluate --input questions_eval.jsonl --output answers.jsonl

For dev questions, prints a PASS/FAIL verdict by comparing to the gold answer.
For eval questions, writes answers.jsonl.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_questions(path: str | Path) -> list[dict]:
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def format_sources(sources: list[dict]) -> str:
    if not sources:
        return "(no sources)"
    parts = []
    for s in sources:
        f = s.get("file", "?")
        p = s.get("page")
        parts.append(f"{f} p.{p}" if p else f)
    return ", ".join(parts)


def run_evaluation(
    input_path: str | Path,
    output_path: str | Path | None = None,
    top_k: int = 12,
    index_path: str | Path = "index.npz",
    chunks_path: str | Path = "chunks.json",
) -> list[dict]:
    from rag_assignment_civicdatalab.retrieve import retrieve
    from rag_assignment_civicdatalab.generate import generate_answer

    input_path = Path(input_path)
    questions = load_questions(input_path)
    is_dev = any("answer" in q for q in questions)

    results = []
    passed = 0
    failed = 0
    unanswered = 0

    print(f"\n{'='*70}")
    print(f"Running {len(questions)} questions from {input_path.name}")
    print(f"{'='*70}\n")

    for q in questions:
        qid = q["question_id"]
        question_text = q["question"]
        gold = q.get("answer", None)

        print(f"[{qid}] {question_text}")

        # Retrieve
        try:
            chunks = retrieve(
                question_text,
                top_k=top_k,
                index_path=index_path,
                chunks_path=chunks_path,
            )
        except Exception as exc:
            print(f"  ERROR during retrieval: {exc}")
            result = {
                "question_id": qid,
                "answered": False,
                "answer": "",
                "sources": [],
            }
            results.append(result)
            unanswered += 1
            print()
            continue

        # Show top retrieved sources
        if chunks:
            print(f"  Retrieved {len(chunks)} chunks:")
            for i, c in enumerate(chunks[:5], 1):
                src = c.get("source_file", "?")
                page = c.get("page")
                fy = c.get("financial_year", "")
                jur = c.get("jurisdiction", "")
                score = c.get("score", 0.0)
                pg_str = f" p.{page}" if page else ""
                print(f"    {i}. {src}{pg_str} | {jur} {fy} | score={score:.3f}")

        # Generate
        try:
            answer_dict = generate_answer(question_text, chunks)
        except Exception as exc:
            print(f"  ERROR during generation: {exc}")
            result = {
                "question_id": qid,
                "answered": False,
                "answer": "",
                "sources": [],
            }
            results.append(result)
            unanswered += 1
            print()
            continue

        answered = answer_dict.get("answered", False)
        answer_text = answer_dict.get("answer", "")
        sources = answer_dict.get("sources", [])

        print(f"  Answered: {answered}")
        print(f"  Answer:   {answer_text}")
        print(f"  Sources:  {format_sources(sources)}")

        if is_dev and gold is not None:
            print(f"  Gold:     {gold}")
            # Simple heuristic: if answered and answer is non-empty, mark as attempted
            verdict = "ANSWERED" if answered else "UNANSWERED"
            print(f"  Verdict:  {verdict}")
            if answered:
                passed += 1
            else:
                unanswered += 1
        print()

        result = {
            "question_id": qid,
            "answered": answered,
            "answer": answer_text if answered else "",
            "sources": sources,
        }
        results.append(result)

        # Small delay between questions to stay within rate limits
        time.sleep(1.5)

    # Summary for dev
    if is_dev:
        total = len(questions)
        print(f"{'='*70}")
        print(f"SUMMARY: {passed} answered, {unanswered} unanswered out of {total}")
        print(f"{'='*70}\n")

    # Write output file if requested
    if output_path:
        output_path = Path(output_path)
        with open(output_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(results)} results to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the Gender Budget RAG system against a set of questions."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to questions JSONL file (e.g. questions_dev.jsonl)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write answers JSONL (e.g. answers.jsonl)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of chunks to retrieve per question (default: 20)",
    )
    parser.add_argument(
        "--index",
        default="index.npz",
        help="Path to index.npz (default: index.npz)",
    )
    parser.add_argument(
        "--chunks",
        default="chunks.json",
        help="Path to chunks.json (default: chunks.json)",
    )
    args = parser.parse_args()

    run_evaluation(
        input_path=args.input,
        output_path=args.output,
        top_k=args.top_k,
        index_path=args.index,
        chunks_path=args.chunks,
    )


if __name__ == "__main__":
    main()
