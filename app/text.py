"""Text normalisation shared by the sparse index, the embedder and the chunker."""

from __future__ import annotations

import re

# Letters, digits and the joiners that appear inside identifiers such as
# "CVE-2024-3094", "pg_vector" or "v1.2". Splitting those apart would make
# exact-identifier search impossible, which is the main job of BM25 in RAG.
_TOKEN = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")

STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
        "from", "has", "have", "how", "i", "in", "is", "it", "its", "of", "on", "or",
        "that", "the", "this", "to", "was", "what", "when", "where", "which", "who",
        "why", "will", "with", "you", "your",
    }
)


def tokenize(text: str, *, drop_stopwords: bool = True) -> list[str]:
    """Lower-case and split text into search tokens."""
    tokens = _TOKEN.findall(text.lower())
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences on ., ? or ! followed by whitespace."""
    parts = re.split(r"(?<=[.?!])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def approx_tokens(text: str) -> int:
    """Rough model-token estimate: about 0.75 words per token for English prose."""
    words = len(text.split())
    return max(1, round(words / 0.75)) if words else 0
