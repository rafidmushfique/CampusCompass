from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ollama_client import embed

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


@dataclass
class Chunk:
    text: str
    url: str
    title: str


def chunk_pages(pages, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    # slice each page up so we don't blow past the model's context window,
    # and so search can zero in on the right bit instead of the whole page
    chunks: list[Chunk] = []
    for page in pages:
        text = page.text
        start = 0
        while start < len(text):
            end = start + chunk_size
            piece = text[start:end].strip()
            if piece:
                chunks.append(Chunk(text=piece, url=page.url, title=page.title))
            if end >= len(text):
                break
            start = end - overlap
    return chunks


def embed_chunks(chunks: list[Chunk], model: str, host: str, progress_callback=None) -> np.ndarray:
    vectors = []
    for i, chunk in enumerate(chunks):
        vectors.append(embed(chunk.text, model=model, host=host))
        if progress_callback:
            progress_callback(i + 1, len(chunks))
    return np.array(vectors, dtype=np.float32)


def top_k_chunks(query_vec: np.ndarray, chunk_vectors: np.ndarray, chunks: list[Chunk], k: int = 5) -> list[Chunk]:
    if len(chunks) == 0:
        return []
    # cosine similarity - normalize both sides then dot product
    a_norm = chunk_vectors / (np.linalg.norm(chunk_vectors, axis=1, keepdims=True) + 1e-8)
    q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    scores = a_norm @ q_norm
    top_idx = np.argsort(-scores)[:k]
    return [chunks[i] for i in top_idx]


def build_prompt(question: str, context_chunks: list[Chunk]) -> list[dict]:
    # stuff the best-matching chunks into the system prompt so the model
    # answers from the actual site instead of making things up - only the
    # page title goes in, not the url, so there's nothing for it to turn
    # into a link
    blocks = [f"[{c.title}]\n{c.text}" for c in context_chunks]
    context_text = "\n\n---\n\n".join(blocks) if blocks else "(nothing relevant found)"

    system_prompt = (
        "Answer using only the context below, which was scraped from a "
        "website. If it's not in there, say you don't know instead of "
        "guessing. Once you've answered, add a short line explaining, in "
        "plain words, which page(s) that came from and why - just name "
        "them, don't include links or URLs.\n\n"
        f"CONTEXT:\n{context_text}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
