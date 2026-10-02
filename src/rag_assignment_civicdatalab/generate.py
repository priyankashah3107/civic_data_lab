"""
generate.py — Groq-based answer generation for the RAG system.

The LLM receives only the retrieved context chunks.
It is instructed to:
  - answer only from context
  - distinguish years and jurisdictions exactly
  - distinguish Part A and Part B
  - preserve and convert units explicitly
  - return answered=false if evidence is insufficient
  - respond in JSON

Usage:
    from rag_assignment_civicdatalab.generate import generate_answer
    result = generate_answer(question, chunks)
    # result = {"answered": True/False, "answer": "...", "sources": [...]}
"""

import json
import logging
import os
import re
import time

logger = logging.getLogger(__name__)

# Max chars per chunk sent to LLM — keeps total prompt within token limits
_MAX_CHUNK_CHARS = 1500
# Max number of chunks to include in prompt
_MAX_CHUNKS_IN_PROMPT = 4

SYSTEM_PROMPT = """You are a precise research assistant answering questions about Indian government gender budget documents.

STRICT RULES — follow every one:
1. Answer ONLY using the provided context. Do NOT use any outside knowledge.
2. Distinguish financial years EXACTLY. "2024-25" and "2026-27" are different years. Never confuse them.
3. Distinguish jurisdictions EXACTLY. Delhi, Bihar, Odisha, Kerala, and Union Budget are all different. Do not mix them up.
4. Distinguish Part A (100% women-specific) and Part B (at least 30% women-specific). If both are relevant, report both.
5. UNITS AND COLUMN ORDER FOR DELHI DOCUMENTS:
   - Delhi gender budget documents report all amounts IN THOUSANDS of rupees (Rs in thousands).
   - The column order for each scheme row is: [Scheme name] | [Revised Estimates prev year] | [Annual Budget outlay current year] | [Women Component flow]
   - The OUTLAY / ALLOCATION for the current year is the SECOND number (Annual Budget), NOT the first (Revised Estimates).
   - Example: "Self Defence | 18040 | 20000 | 20000" → the 2024-25 outlay is 20,000 thousands = Rs 2 crore.
   - Example: "Ladli Yojna | 1000000 | 1000000 | 1000000" → the outlay is 10,00,000 thousands = Rs 100 crore.
   - To convert thousands to crore: divide by 10,000.
   - To convert thousands to lakh: divide by 100.
6. For Union Budget (MRF_13_Union_Budget.csv), amounts are already in crore.
7. For Odisha gender budget (14-Gender_Budget.pdf), amounts are in lakh.
8. If the context does not contain sufficient evidence, set answered to false and leave answer empty.
9. Do NOT hallucinate figures, scheme names, or dates not in the context.
10. Include sources for every claim. For CSV/XLS files, page must be null.
11. For Hindi questions, find the answer in the English portion of bilingual documents.

Respond ONLY with valid JSON (no text before or after):
{
  "answered": true,
  "answer": "concise answer here",
  "sources": [{"file": "filename.pdf", "page": 1}]
}

When you cannot answer:
{
  "answered": false,
  "answer": "",
  "sources": []
}"""


def _load_groq_client():
    """Load Groq client, reading API key from .env."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not found. Create a .env file with GROQ_API_KEY=your_key_here"
        )

    try:
        from groq import Groq
        return Groq(api_key=api_key)
    except Exception as exc:
        raise RuntimeError(f"Failed to initialise Groq client: {exc}") from exc


def _truncate_text(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, preserving whole lines where possible."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    # Try to cut at a newline boundary
    last_nl = truncated.rfind("\n")
    if last_nl > max_chars // 2:
        return truncated[:last_nl] + "\n[...truncated]"
    return truncated + "[...truncated]"


def _format_chunks_for_prompt(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a labelled context block for the prompt.
    Deduplicates by source+page, limits each chunk to _MAX_CHUNK_CHARS,
    and caps total chunks at _MAX_CHUNKS_IN_PROMPT.
    """
    # Deduplicate by (source_file, page) — keep highest-scored version
    seen = {}
    for chunk in chunks:
        key = (chunk.get("source_file", ""), chunk.get("page"))
        if key not in seen or chunk.get("score", 0) > seen[key].get("score", 0):
            seen[key] = chunk
    deduped = sorted(seen.values(), key=lambda c: c.get("score", 0), reverse=True)

    lines = []
    for i, chunk in enumerate(deduped[:_MAX_CHUNKS_IN_PROMPT], 1):
        src = chunk.get("source_file", "unknown")
        page = chunk.get("page")
        fy = chunk.get("financial_year", "")
        jur = chunk.get("jurisdiction", "")
        part = chunk.get("part", "")

        meta_parts = []
        if jur:
            meta_parts.append(f"jurisdiction={jur}")
        if fy:
            meta_parts.append(f"year={fy}")
        if part:
            meta_parts.append(f"part={part}")
        if page:
            meta_parts.append(f"page={page}")
        meta = ", ".join(meta_parts)

        chunk_text = _truncate_text(chunk.get("text", "").strip(), _MAX_CHUNK_CHARS)

        lines.append(f"--- Context chunk {i} [{src}] ({meta}) ---")
        lines.append(chunk_text)
        lines.append("")

    return "\n".join(lines)


def _parse_json_response(raw: str) -> dict:
    """
    Parse the LLM's JSON response. Returns a valid result dict or a fallback.
    Handles truncated JSON by attempting repair.
    """
    if not raw or not raw.strip():
        return {"answered": False, "answer": "", "sources": []}

    text = raw.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code blocks
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding the first complete {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        # Try to repair truncated JSON by closing open brackets
        repaired = candidate
        # Count open/close braces and brackets
        opens = repaired.count("{") - repaired.count("}")
        arr_opens = repaired.count("[") - repaired.count("]")
        if arr_opens > 0:
            repaired += "]" * arr_opens
        if opens > 0:
            repaired += "}" * opens
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

    # Last resort: try to extract answered + answer from text directly
    answered_match = re.search(r'"answered"\s*:\s*(true|false)', text, re.IGNORECASE)
    answer_match = re.search(r'"answer"\s*:\s*"([^"]*)"', text)
    if answered_match:
        answered = answered_match.group(1).lower() == "true"
        answer = answer_match.group(1) if answer_match else ""
        return {"answered": answered, "answer": answer, "sources": []}

    logger.warning(f"Could not parse JSON from LLM response: {text[:300]}")
    return {"answered": False, "answer": "", "sources": []}


def _call_groq_with_retry(client, model: str, messages: list, max_retries: int = 3) -> str:
    """Call Groq API with exponential backoff on rate limit errors."""
    last_exc = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=512,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            err_str = str(exc)
            last_exc = exc
            # Rate limit (429) or token overflow (413) — wait and retry
            if "429" in err_str or "rate_limit" in err_str:
                wait = 2 ** attempt  # 1s, 2s, 4s
                logger.warning(f"Rate limit hit (attempt {attempt+1}/{max_retries}), waiting {wait}s...")
                time.sleep(wait)
                continue
            elif "413" in err_str:
                # Request too large — this shouldn't happen with our truncation,
                # but if it does, raise immediately so caller knows
                raise RuntimeError(f"Groq API call failed: {exc}") from exc
            else:
                raise RuntimeError(f"Groq API call failed: {exc}") from exc
    raise RuntimeError(f"Groq API call failed after {max_retries} retries: {last_exc}") from last_exc


def generate_answer(
    question: str,
    chunks: list[dict],
    model: str | None = None,
) -> dict:
    """
    Generate a structured answer using Groq.

    Args:
        question: The user's question.
        chunks: Retrieved context chunks (with metadata).
        model: Optional model override. Falls back to GROQ_MODEL env var or default.

    Returns:
        dict with keys: answered (bool), answer (str), sources (list of dicts)
    """
    if not chunks:
        logger.warning("No chunks provided to generate_answer; returning unanswered.")
        return {"answered": False, "answer": "", "sources": []}

    try:
        client = _load_groq_client()
    except (EnvironmentError, RuntimeError) as exc:
        logger.error(str(exc))
        raise

    # Determine model
    if model is None:
        from dotenv import load_dotenv
        load_dotenv()
        model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

    context_text = _format_chunks_for_prompt(chunks)
    user_prompt = f"""Question: {question}

Retrieved context:
{context_text}

Answer the question using ONLY the context above. Respond with valid JSON only."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    try:
        raw = _call_groq_with_retry(client, model, messages)
    except RuntimeError as exc:
        logger.error(str(exc))
        raise

    result = _parse_json_response(raw)

    # Ensure required keys exist
    if "answered" not in result:
        result["answered"] = False
    if "answer" not in result:
        result["answer"] = ""
    if "sources" not in result:
        result["sources"] = []

    # Enforce: answered=false must have empty answer
    if not result["answered"]:
        result["answer"] = ""

    # Fix: CSV/XLS sources should not have page numbers
    fixed_sources = []
    for src in result.get("sources", []):
        fname = src.get("file", "")
        page = src.get("page")
        if fname.endswith(".csv") or fname.endswith(".xls") or fname.endswith(".xlsx"):
            page = None
        fixed_sources.append({"file": fname, "page": page})
    result["sources"] = fixed_sources

    return result
