"""
ingest.py — Document parsers for the Gender Budget RAG system.

Parsers:
  parse_pdf(path)   → List[dict]  one chunk per page
  parse_csv(path)   → List[dict]  one chunk per row
  parse_excel(path) → List[dict]  narrative chunk + one chunk per data row

Every chunk has the schema:
  {
    "text":          str,
    "source_file":   str,   # basename only
    "page":          int | None,
    "financial_year": str | None,
    "jurisdiction":  str | None,
    "language":      str,   # "en", "hi", "hi+en"
    "part":          str | None,  # "A", "B", or None
  }
"""

import logging
import re
import unicodedata
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"(\d{4}[-–]\d{2,4})")

# Normalise a messy year string to YYYY-YY format, e.g. "2023-2024 Budget Estimates" → "2023-24"
def _normalise_year(raw: str) -> Optional[str]:
    m = _YEAR_RE.search(str(raw))
    if not m:
        return None
    year = m.group(1)
    # Collapse YYYY-YYYY to YYYY-YY
    if re.match(r"\d{4}-\d{4}", year):
        year = year[:4] + "-" + year[7:9]
    return year

# Map filename stems → known jurisdictions
_JURISDICTION_HINTS: dict[str, str] = {
    "gender_budget": "Delhi",          # all Delhi PDFs share this stem
    "14-gender_budget": "Odisha",
    "14-gender": "Odisha",
    "mrf_13": "Union",
    "stat20": "Union",
    "17107782611749468962": "Bihar",
}


def _jurisdiction_from_filename(stem: str) -> Optional[str]:
    lower = stem.lower()
    for key, val in _JURISDICTION_HINTS.items():
        if lower.startswith(key):
            return val
    return None


def _year_from_filename(stem: str) -> Optional[str]:
    m = _YEAR_RE.search(stem)
    if not m:
        return None
    return _normalise_year(m.group(1))


def _has_hindi(text: str) -> bool:
    """Return True if text contains Devanagari Unicode characters."""
    for ch in text:
        if "\u0900" <= ch <= "\u097f":
            return True
    return False


def _detect_language(text: str) -> str:
    if _has_hindi(text):
        return "hi+en"
    return "en"


def _part_from_category(category: str) -> Optional[str]:
    cat = str(category).upper()
    if "PART A" in cat:
        return "A"
    if "PART B" in cat:
        return "B"
    return None


# ---------------------------------------------------------------------------
# PDF parser
# ---------------------------------------------------------------------------

# English scheme name pattern: a line of mainly ASCII printable chars, not pure numbers
_SCHEME_LINE_RE = re.compile(r"^[A-Z][A-Za-z /,()\-'\.]{5,}$")

def _extract_scheme_subchunks(
    text: str,
    source_file: str,
    page: int,
    financial_year: Optional[str],
    jurisdiction: Optional[str],
    language: str,
) -> list[dict]:
    """
    For Delhi-style bilingual PDFs, extract individual scheme rows as sub-chunks.
    Each sub-chunk contains the English scheme name + the numbers that follow it.
    This gives much better semantic retrieval since the Hindi garbled text is diluted.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    subchunks = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Detect English scheme-name lines (mostly ASCII, not pure digits)
        if (len(line) > 5 and
                sum(1 for c in line if ord(c) < 128) / len(line) > 0.85 and
                not re.match(r"^[\d,\s\.]+$", line) and
                not line.startswith("S. No") and
                not line.startswith("Scheme") and
                "Annual Budget" not in line and
                "Revised Estimates" not in line and
                "IN THOUSANDS" not in line and
                "WC" != line):
            # Collect the scheme name and up to 8 following lines (may contain numbers)
            context_lines = [line]
            j = i + 1
            while j < len(lines) and j < i + 9:
                nl = lines[j]
                # Stop if we hit another English scheme-ish line after collecting some numbers
                if (len(context_lines) >= 3 and
                        len(nl) > 5 and
                        sum(1 for c in nl if ord(c) < 128) / len(nl) > 0.85 and
                        not re.match(r"^[\d,\s\.]+$", nl)):
                    break
                context_lines.append(nl)
                j += 1
            chunk_text = f"[{jurisdiction} {financial_year}] " + " | ".join(context_lines)
            subchunks.append({
                "text": chunk_text,
                "source_file": source_file,
                "page": page,
                "financial_year": financial_year,
                "jurisdiction": jurisdiction,
                "language": language,
                "part": None,
            })
            i = j
        else:
            i += 1
    return subchunks


def parse_pdf(path: str | Path) -> list[dict]:
    """
    Parse a PDF file page by page using PyMuPDF.
    Returns one chunk per page (non-empty pages only).
    For Delhi-style bilingual PDFs, also emits scheme-level sub-chunks.
    Metadata: jurisdiction and financial_year inferred from filename + page 1 text.
    """
    try:
        import pymupdf as fitz  # PyMuPDF
    except ImportError:
        logger.error("PyMuPDF not installed.")
        return []

    path = Path(path)
    stem = path.stem
    source_file = path.name

    jurisdiction = _jurisdiction_from_filename(stem)
    financial_year = _year_from_filename(stem)
    is_bilingual = jurisdiction == "Delhi"  # Delhi PDFs are bilingual with scheme-level data

    chunks: list[dict] = []

    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        logger.warning(f"Could not open PDF {path}: {exc}")
        return []

    for page_num in range(len(doc)):
        try:
            page = doc[page_num]
            text = page.get_text("text")
        except Exception as exc:
            logger.warning(f"Error extracting page {page_num + 1} from {path}: {exc}")
            continue

        if not text or not text.strip():
            continue

        # On first few pages, refine jurisdiction/year from content
        if page_num <= 2:
            if jurisdiction is None:
                for name in ("Delhi", "Bihar", "Odisha", "Kerala", "Rajasthan",
                             "Uttar Pradesh", "Maharashtra"):
                    if name in text:
                        jurisdiction = name
                        break
            if financial_year is None:
                m = _YEAR_RE.search(text)
                if m:
                    financial_year = _normalise_year(m.group(1))

        language = _detect_language(text)

        # Always add the full page chunk
        chunks.append({
            "text": text.strip(),
            "source_file": source_file,
            "page": page_num + 1,       # 1-based
            "financial_year": financial_year,
            "jurisdiction": jurisdiction,
            "language": language,
            "part": None,
        })

        # For bilingual Delhi-style PDFs, also add scheme-level sub-chunks
        if is_bilingual and financial_year:
            sub = _extract_scheme_subchunks(
                text, source_file, page_num + 1,
                financial_year, jurisdiction, language
            )
            chunks.extend(sub)

    doc.close()
    logger.info(f"parse_pdf: {source_file} → {len(chunks)} chunks")
    return chunks


# ---------------------------------------------------------------------------
# CSV parser
# ---------------------------------------------------------------------------

def parse_csv(path: str | Path) -> list[dict]:
    """
    Parse a CSV file into natural-language row chunks.
    Each row becomes a text string that includes all important fields.
    """
    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas not installed.")
        return []

    path = Path(path)
    source_file = path.name
    stem = path.stem

    jurisdiction = _jurisdiction_from_filename(stem)
    financial_year = _year_from_filename(stem)

    # Try multiple encodings — government documents often use Windows-1252 or latin-1
    df = None
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(path, dtype=str, encoding=enc)
            logger.info(f"Parsed CSV {path} with encoding: {enc}")
            break
        except (UnicodeDecodeError, Exception):
            continue
    if df is None:
        logger.warning(f"Could not read CSV {path} with any supported encoding.")
        return []

    if df.empty:
        logger.warning(f"CSV {path} is empty.")
        return []

    # Detect the year columns (columns that look like "YYYY-YY" or "YYYY-YYYY...")
    year_cols = [c for c in df.columns if re.match(r"\d{4}[-–]\d{2,4}", str(c))]

    # Clean financial year: use the latest year column, normalised to YYYY-YY
    if financial_year is None and year_cols:
        financial_year = _normalise_year(year_cols[-1])

    chunks: list[dict] = []

    for _, row in df.iterrows():
        parts: list[str] = []

        # Category / Part
        category = str(row.get("Category", "")).strip()
        part = _part_from_category(category)
        if category and category != "nan":
            parts.append(f"Category: {category}")

        # Ministry / Department
        ministry = str(row.get("Ministry/Department", "")).strip()
        if ministry and ministry != "nan":
            parts.append(f"Ministry/Department: {ministry}")

        # Scheme
        scheme = str(row.get("Scheme", "")).strip()
        if scheme and scheme != "nan":
            parts.append(f"Scheme: {scheme}")

        # Sub-Category / Demand No.
        sub_cat = str(row.get("Sub-Category", "")).strip()
        if sub_cat and sub_cat != "nan":
            parts.append(f"Sub-Category: {sub_cat}")

        # Financial year columns — add each year value
        for col in year_cols:
            val = str(row.get(col, "")).strip()
            if val and val != "nan":
                parts.append(f"{col} BE: Rs {val} crore")

        # Any other columns not yet captured
        skip_cols = {"Category", "Sub-Category", "Sl.No.", "Ministry/Department",
                     "Scheme"} | set(year_cols)
        for col in df.columns:
            if col not in skip_cols:
                val = str(row.get(col, "")).strip()
                if val and val != "nan":
                    parts.append(f"{col}: {val}")

        if not parts:
            continue

        text = " | ".join(parts)

        chunks.append({
            "text": text,
            "source_file": source_file,
            "page": None,
            "financial_year": financial_year,  # already normalised above
            "jurisdiction": jurisdiction,
            "language": _detect_language(text),
            "part": part,
        })

    logger.info(f"parse_csv: {source_file} → {len(chunks)} chunks")
    return chunks


# ---------------------------------------------------------------------------
# Excel parser
# ---------------------------------------------------------------------------

def _detect_header_row(rows: list[list]) -> int:
    """
    Find the first row index where >= 3 cells are non-empty strings.
    Returns 0 if not found (fall back to first row).
    """
    for i, row in enumerate(rows):
        non_empty_strings = sum(
            1 for cell in row
            if cell is not None and isinstance(cell, str) and cell.strip()
        )
        if non_empty_strings >= 3:
            return i
    return 0


def _cell_to_str(cell) -> str:
    if cell is None:
        return ""
    if isinstance(cell, float):
        if cell == int(cell):
            return str(int(cell))
        return str(cell)
    return str(cell).strip()


def parse_excel(path: str | Path) -> list[dict]:
    """
    Parse an Excel file (.xls or .xlsx).
    - Pre-header rows (including narrative cells) → one narrative chunk per sheet
    - Table rows after the detected header → one chunk per row
    """
    path = Path(path)
    source_file = path.name
    stem = path.stem
    suffix = path.suffix.lower()

    jurisdiction = _jurisdiction_from_filename(stem)
    financial_year = _year_from_filename(stem)

    all_rows: list[list] = []

    if suffix == ".xls":
        try:
            import xlrd
        except ImportError:
            logger.error("xlrd not installed. Required for .xls files.")
            return []
        try:
            wb = xlrd.open_workbook(str(path))
        except Exception as exc:
            logger.warning(f"Could not open XLS {path}: {exc}")
            return []

        for sheet in wb.sheets():
            sheet_rows: list[list] = []
            for r in range(sheet.nrows):
                row_vals = []
                for c in range(sheet.ncols):
                    cell = sheet.cell(r, c)
                    # xlrd cell types: 0=empty, 1=text, 2=number, 3=date, 4=bool, 5=error
                    if cell.ctype == 0:
                        row_vals.append(None)
                    elif cell.ctype == 2:
                        row_vals.append(cell.value)
                    else:
                        row_vals.append(str(cell.value).strip())
                sheet_rows.append(row_vals)
            all_rows.extend(sheet_rows)

    elif suffix in (".xlsx", ".xlsm"):
        try:
            import openpyxl
        except ImportError:
            logger.error("openpyxl not installed.")
            return []
        try:
            wb = openpyxl.load_workbook(str(path), data_only=True)
        except Exception as exc:
            logger.warning(f"Could not open XLSX {path}: {exc}")
            return []

        for ws in wb.worksheets:
            for row in ws.iter_rows(values_only=True):
                all_rows.append(list(row))

    else:
        logger.warning(f"Unsupported Excel format: {suffix}")
        return []

    if not all_rows:
        logger.warning(f"Excel {path} has no rows.")
        return []

    header_idx = _detect_header_row(all_rows)

    chunks: list[dict] = []

    # --- Narrative chunk: everything before (and including) header row ---
    narrative_lines: list[str] = []
    for row in all_rows[: header_idx + 1]:
        line = "  |  ".join(_cell_to_str(c) for c in row if c is not None and _cell_to_str(c))
        if line.strip():
            narrative_lines.append(line)

    if narrative_lines:
        narrative_text = "\n".join(narrative_lines)
        # Also try to extract financial year from narrative
        fy = financial_year
        if fy is None:
            m = _YEAR_RE.search(narrative_text)
            if m:
                fy = _normalise_year(m.group(1))
        chunks.append({
            "text": narrative_text,
            "source_file": source_file,
            "page": None,
            "financial_year": fy,
            "jurisdiction": jurisdiction,
            "language": _detect_language(narrative_text),
            "part": None,
        })

    # --- Table rows after the header ---
    header_row = all_rows[header_idx]
    headers = [_cell_to_str(c) for c in header_row]

    # Year columns from header
    year_cols_idx = [
        i for i, h in enumerate(headers)
        if re.match(r"\d{4}[-–]\d{2,4}", h)
    ]

    for row in all_rows[header_idx + 1 :]:
        row_parts: list[str] = []
        part = None

        for i, cell in enumerate(row):
            val = _cell_to_str(cell)
            if not val:
                continue
            col_name = headers[i] if i < len(headers) else f"col_{i}"

            # Detect part from content
            if part is None:
                p = _part_from_category(val)
                if p:
                    part = p

            if i in year_cols_idx:
                row_parts.append(f"{col_name}: Rs {val} crore")
            else:
                row_parts.append(f"{col_name}: {val}")

        if not row_parts:
            continue

        text = " | ".join(row_parts)

        fy = financial_year
        if fy is None and year_cols_idx and year_cols_idx[-1] < len(headers):
            fy = _normalise_year(headers[year_cols_idx[-1]])

        chunks.append({
            "text": text,
            "source_file": source_file,
            "page": None,
            "financial_year": fy,
            "jurisdiction": jurisdiction,
            "language": _detect_language(text),
            "part": part,
        })

    logger.info(f"parse_excel: {source_file} → {len(chunks)} chunks (header at row {header_idx})")
    return chunks


# ---------------------------------------------------------------------------
# Ingest all files in a directory
# ---------------------------------------------------------------------------

def ingest_all(data_dir: str | Path) -> list[dict]:
    """
    Ingest all supported files in data_dir.
    Supported: .pdf, .csv, .xls, .xlsx
    """
    data_dir = Path(data_dir)
    all_chunks: list[dict] = []

    supported = {".pdf", ".csv", ".xls", ".xlsx"}
    files = sorted(data_dir.iterdir())

    for f in files:
        if f.suffix.lower() not in supported:
            continue
        logger.info(f"Ingesting: {f.name}")
        if f.suffix.lower() == ".pdf":
            chunks = parse_pdf(f)
        elif f.suffix.lower() == ".csv":
            chunks = parse_csv(f)
        elif f.suffix.lower() in (".xls", ".xlsx"):
            chunks = parse_excel(f)
        else:
            continue

        all_chunks.extend(chunks)

    logger.info(f"ingest_all: total {len(all_chunks)} chunks from {data_dir}")
    return all_chunks
