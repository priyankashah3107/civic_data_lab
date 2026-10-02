"""
index.py — Build and persist the vector index for the RAG system.

Usage:
    from src.rag_assignment_civicdatalab.index import build_index
    build_index(data_dir="corpus", index_path="index.npz", chunks_path="chunks.json")

Saved artifacts:
    index.npz   — numpy float32 array of shape (N, 768), L2-normalised
    chunks.json — list of chunk dicts (text + metadata)
"""

import json
import logging
from pathlib import Path

import numpy as np

from rag_assignment_civicdatalab.ingest import ingest_all

logger = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-m3"

_encoder = None  # lazy-loaded


def _get_encoder():
    global _encoder
    if _encoder is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {MODEL_NAME}")
            _encoder = SentenceTransformer(MODEL_NAME)
        except Exception as exc:
            logger.error(f"Failed to load SentenceTransformer: {exc}")
            raise
    return _encoder


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """
    Embed a list of texts. Returns float32 array of shape (N, dim), L2-normalised.
    """
    encoder = _get_encoder()
    try:
        embeddings = encoder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,   # L2-normalise → dot product = cosine similarity
            convert_to_numpy=True,
        )
    except Exception as exc:
        logger.error(f"Embedding failed: {exc}")
        raise
    return embeddings.astype(np.float32)


def build_index(
    data_dir: str | Path = "corpus",
    index_path: str | Path = "index.npz",
    chunks_path: str | Path = "chunks.json",
) -> tuple[np.ndarray, list[dict]]:
    """
    Ingest all documents, embed all chunks, save index.npz and chunks.json.
    Returns (embeddings, chunks).
    """
    data_dir = Path(data_dir)
    index_path = Path(index_path)
    chunks_path = Path(chunks_path)

    logger.info("=== Starting ingestion ===")
    chunks = ingest_all(data_dir)

    if not chunks:
        raise RuntimeError(f"No chunks produced from {data_dir}. Check that corpus files exist.")

    # Print per-file stats
    from collections import Counter
    file_counts = Counter(c["source_file"] for c in chunks)
    for fname, count in sorted(file_counts.items()):
        print(f"  {fname}: {count} chunks")
    print(f"  TOTAL: {len(chunks)} chunks")

    logger.info("=== Embedding chunks ===")
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    logger.info(f"Embeddings shape: {embeddings.shape}")

    # Save
    np.savez_compressed(str(index_path), embeddings=embeddings)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=None)

    print(f"\nSaved {len(chunks)} chunks to {chunks_path}")
    print(f"Saved embeddings {embeddings.shape} to {index_path}")
    return embeddings, chunks


def load_index(
    index_path: str | Path = "index.npz",
    chunks_path: str | Path = "chunks.json",
) -> tuple[np.ndarray, list[dict]]:
    """
    Load saved index.npz and chunks.json.
    """
    index_path = Path(index_path)
    chunks_path = Path(chunks_path)

    if not index_path.exists():
        raise FileNotFoundError(
            f"Index not found: {index_path}. Run 'uv run python ingest_and_index.py' first."
        )
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Chunks not found: {chunks_path}. Run 'uv run python ingest_and_index.py' first."
        )

    data = np.load(str(index_path))
    embeddings = data["embeddings"]

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    logger.info(f"Loaded index: {embeddings.shape}, {len(chunks)} chunks")
    return embeddings, chunks
