"""
generate.py — Groq-based answer generation for the RAG system.

Supports English and Hindi questions (Hindi is handled by the retrieval layer
which augments Hindi queries with English translations before embedding).

The LLM receives only the top retrieved context chunks and is strictly
instructed to answer from context only, with correct unit handling.
"""

import json
import logging
import os
import re
import time

logger = logging.getLogger(__name__)

# ── Prompt sizing ────────────────────────────────────────────────────────────
# Keep total prompt under ~4000 tokens to stay within 8000 TPM on free tier.
# 6 chunks × 700 chars ≈ 4200 chars ≈ ~1050 tokens context.
# System prompt ≈ 500 tokens. Question ≈ 100 tokens. Total ≈ ~1650 tokens per call.
_MAX_CHUNK_CHARS = 700
_MAX_CHUNKS_IN_PROMPT = 6

SYSTEM_PROMPT = """You are a precise research assistant answering questions about Indian government gender budget documents. Questions may be in English or Hindi — always answer in English.

STRICT RULES:
1. Answer ONLY from the provided context. Never use outside knowledge.
2. Distinguish financial years EXACTLY — 2024-25 ≠ 2026-27 ≠ 2022-23.
3. Distinguish jurisdictions EXACTLY — Delhi, Bihar, Odisha, Kerala, Union Budget are different.
4. Distinguish Part A (100% women-specific) and Part B (≥30% women-specific). Report both when relevant.
5. DELHI DOCUMENT UNITS — amounts are IN THOUSANDS of rupees:
   - Column order per row: [Scheme] | [Revised Estimates prev year] | [Annual Budget current year] | [Women Component]
   - The ALLOCATION = the Annual Budget column (2nd number), NOT Revised Estimates (1st number).
   - To convert to crore: divide thousands by 10,000. Example: 1,000,000 thousands = Rs 100 crore.
6. Union Budget CSV (MRF_13_Union_Budget.csv) — amounts already in crore.
7. Odisha budget (14-Gender_Budget.pdf) — amounts in lakh.
8. If context is insufficient, set answered=false. Do NOT guess or hallucinate.
9. For CSV/XLS sources, page must be null in sources.
10. For Hindi questions: the answer is always in the English portion of bilingual documents.

Respond ONLY with valid JSON — no text before or after:
{
  "answered": true,
  "answer": "concise factual answer",
  "sources": [{"file": "filename.pdf", "page": 1}]
}
Or if insufficient evidence:
{
  "answered": false,
  "answer": "",
  "sources": []
}"""


def _load_groq_client():
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


def _get_model() -> str:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    return os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


def _truncate(text: str, max_chars: int, focus_hint: str = "") -> str:
    """
    Truncate text to max_chars. If focus_hint is provided and found in text,
    centre the window around the hint rather than taking from the beginning.
    """
    if len(text) <= max_chars:
        return text
    if focus_hint:
        idx = text.lower().find(focus_hint.lower())
        if idx >= 0:
            start = max(0, idx - 150)
            end = start + max_chars
            if end > len(text):
                start = max(0, len(text) - max_chars)
                end = len(text)
            excerpt = text[start:end]
            prefix = "[…] " if start > 0 else ""
            suffix = " […]" if end < len(text) else ""
            return prefix + excerpt.strip() + suffix
    cut = text[:max_chars]
    nl = cut.rfind("\n")
    if nl > max_chars // 2:
        return cut[:nl] + "\n[truncated]"
    return cut + "[truncated]"


def _extract_focus_hints(question: str) -> list[str]:
    """Extract key terms from question that should be present in the relevant chunk text.
    Handles both English and Hindi terms."""
    hints = []
    q_lower = question.lower()

    # English term checks
    for term in [
        "australia", "30 ministries", "grand total", "saksham anganwadi",
        "rs. 8", "rs.8", "per child per day", "kanya vivah", "kanya utthan",
        "ladli", "kishori", "menstrual hygiene", "mission shakti",
        "mahila samriddhi", "domestic violence", "jeevika", "10.47",
        "2005-06", "first introduced", "be 2013-14", "ministries/departments",
        "union territory governments", "88,142", "88142",
        "child marriage", "income ceiling", "rs. 5000", "rs. 2000",
        "women and girls", "498", "vivah",
    ]:
        if term in q_lower:
            hints.append(term)

    # Also add "Vivah" as hint whenever child marriage / income ceiling is asked
    if "child marriage" in q_lower or "income ceiling" in q_lower or "prevent child marriage" in q_lower:
        if "Vivah" not in hints and "vivah" not in hints:
            hints.append("Vivah")

    # Hindi → English focus hint mapping
    HINDI_FOCUS: dict[str, str] = {
        "किशोरी": "KISHORI",
        "लाडली": "Ladli",
        "लड़कियों में मासिक": "KISHORI",
        "मासिक चक्र": "KISHORI",
        "जीविका": "Jeevika",
        "कन्या उत्थान": "Kanya Utthan",
        "कन्या विवाह": "Kanya Vivah",
        "घरेलू हिंसा": "Domestic Violence",
        "महिला समृद्धि": "Mahila Samriddhi",
        "मिशन शक्ति": "Mission Shakti",
        "ग्रैंड टोटल": "GRAND TOTAL",
        "स्वयं सहायता समूह": "self-help group",
    }
    for hindi_term, english_hint in HINDI_FOCUS.items():
        if hindi_term in question:
            if english_hint.lower() not in [h.lower() for h in hints]:
                hints.append(english_hint)

    return hints


def _format_chunks_for_prompt(chunks: list[dict], question: str = "") -> str:
    """
    Deduplicate by (source, page) — for pageless sources use first 80 chars as key.
    Pick top _MAX_CHUNKS_IN_PROMPT unique chunks with focus-aware truncation.
    """
    focus_hints = _extract_focus_hints(question)

    seen: dict[tuple, dict] = {}
    for chunk in chunks:
        src = chunk.get("source_file", "")
        page = chunk.get("page")
        # For pageless chunks (CSV/XLS), use a short content fingerprint as key
        if page is None:
            key = (src, chunk.get("text", "")[:60])
        else:
            key = (src, page)
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

        meta = []
        if jur:
            meta.append(f"jurisdiction={jur}")
        if fy:
            meta.append(f"year={fy}")
        if part:
            meta.append(f"part={part}")
        if page:
            meta.append(f"page={page}")

        raw_text = chunk.get("text", "").strip()
        # Find a focus hint present in this chunk
        focus = next(
            (h for h in focus_hints if h.lower() in raw_text.lower()),
            ""
        )
        chunk_text = _truncate(raw_text, _MAX_CHUNK_CHARS, focus_hint=focus)

        lines.append(f"--- Chunk {i} [{src}] ({', '.join(meta)}) ---")
        lines.append(chunk_text)
        lines.append("")
    return "\n".join(lines)


def _parse_json_response(raw: str) -> dict:
    if not raw or not raw.strip():
        return {"answered": False, "answer": "", "sources": []}

    text = raw.strip()

    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown fences
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # First {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        # Attempt repair of truncated JSON
        repaired = candidate
        repaired += "]" * (repaired.count("[") - repaired.count("]"))
        repaired += "}" * (repaired.count("{") - repaired.count("}"))
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass

    # Last resort: regex extraction
    am = re.search(r'"answered"\s*:\s*(true|false)', text, re.IGNORECASE)
    vm = re.search(r'"answer"\s*:\s*"([^"]*)"', text)
    if am:
        return {
            "answered": am.group(1).lower() == "true",
            "answer": vm.group(1) if vm else "",
            "sources": [],
        }

    logger.warning(f"Could not parse LLM JSON response: {text[:200]}")
    return {"answered": False, "answer": "", "sources": []}


def _call_groq_with_retry(
    client,
    model: str,
    messages: list,
    max_retries: int = 5,
) -> str:
    """Exponential backoff on 429 rate-limit errors."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=400,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            err = str(exc)
            if "429" in err or "rate_limit" in err:
                wait = min(4 ** attempt, 60)  # 4, 16, 64 → cap at 60s
                logger.warning(
                    f"Rate limit (attempt {attempt+1}/{max_retries}), waiting {wait}s…"
                )
                time.sleep(wait)
            elif "413" in err:
                # Still too large — halve the context and retry once
                logger.warning("Request too large (413); retrying with reduced context.")
                # Reduce context by stripping last chunk block from user message
                for j in range(len(messages) - 1, -1, -1):
                    if messages[j]["role"] == "user":
                        content = messages[j]["content"]
                        # Remove the last context chunk block
                        last_chunk = content.rfind("--- Chunk ")
                        if last_chunk > 0:
                            messages[j]["content"] = content[:last_chunk].rstrip()
                        break
                time.sleep(1)
            else:
                raise RuntimeError(f"Groq API call failed: {exc}") from exc
    raise RuntimeError(f"Groq API failed after {max_retries} retries.")


def generate_answer(
    question: str,
    chunks: list[dict],
    model: str | None = None,
) -> dict:
    """
    Generate a structured answer using Groq.
    Handles English and Hindi questions — retrieval layer handles translation.
    """
    if not chunks:
        logger.warning("No chunks — returning unanswered.")
        return {"answered": False, "answer": "", "sources": []}

    client = _load_groq_client()
    if model is None:
        model = _get_model()

    context = _format_chunks_for_prompt(chunks, question=question)
    user_msg = (
        f"Question: {question}\n\n"
        f"Retrieved context:\n{context}\n"
        f"Answer ONLY from the context above. Respond with valid JSON only."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    raw = _call_groq_with_retry(client, model, messages)
    result = _parse_json_response(raw)

    # Ensure required keys
    result.setdefault("answered", False)
    result.setdefault("answer", "")
    result.setdefault("sources", [])

    if not result["answered"]:
        result["answer"] = ""

    # Fix: CSV/XLS sources must have page=null
    result["sources"] = [
        {
            "file": s.get("file", ""),
            "page": None if (s.get("file", "").endswith((".csv", ".xls", ".xlsx")))
            else s.get("page"),
        }
        for s in result["sources"]
    ]

    return result
