"""Chunking strategies: sliding window, recursive, Markdown, semantic, parent-child, code."""

from __future__ import annotations

import ast
import hashlib
import re
import statistics
from dataclasses import dataclass, field

from app.embeddings import Embedder, cosine


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit of text plus the metadata needed to cite it."""

    chunk_id: str
    doc_id: str
    text: str
    position: int
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]


def sliding_window(words: list[str], size: int, overlap: int) -> list[list[str]]:
    """Fixed-size windows of `size` words, each sharing `overlap` words with the previous one."""
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")
    step = size - overlap
    windows: list[list[str]] = []
    for start in range(0, len(words), step):
        windows.append(words[start : start + size])
        if start + size >= len(words):
            break
    return windows


DEFAULT_SEPARATORS = ("\n\n", "\n", ". ", " ")


def recursive_split(
    text: str,
    max_chars: int = 500,
    separators: tuple[str, ...] = DEFAULT_SEPARATORS,
) -> list[str]:
    """Split on the coarsest separator that works, recursing into oversized pieces.

    Paragraphs are tried first, then lines, then sentences, then words. A piece
    that is still too long after the last separator is cut every `max_chars`.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    if not separators:
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]

    sep, rest = separators[0], separators[1:]
    if sep not in text:
        return recursive_split(text, max_chars, rest)

    # A separator such as ". " carries punctuation that belongs to the text:
    # keep the "." on the left part and join pieces back with the whitespace.
    keep = sep.strip()
    joiner = sep[len(keep) :] if keep else sep
    parts = text.split(sep)
    parts = [p + keep for p in parts[:-1]] + [parts[-1]]

    pieces: list[str] = []
    current = ""
    for part in parts:
        candidate = f"{current}{joiner}{part}" if current else part
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            pieces.append(current.strip())
        if len(part) > max_chars:
            pieces.extend(recursive_split(part, max_chars, rest))
            current = ""
        else:
            current = part
    if current.strip():
        pieces.append(current.strip())
    return pieces


_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def markdown_sections(markdown: str) -> list[tuple[list[str], str]]:
    """Split Markdown at headings and return (breadcrumb, body) pairs.

    Fenced code blocks are kept whole: a "# comment" inside ``` is not a heading.
    """
    sections: list[tuple[list[str], str]] = []
    trail: list[str] = []
    body: list[str] = []
    in_fence = False

    def flush() -> None:
        text = "\n".join(body).strip()
        if text:
            sections.append((list(trail), text))
        body.clear()

    for line in markdown.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        match = None if in_fence else _HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            del trail[level - 1 :]
            trail.append(match.group(2))
        else:
            body.append(line)
    flush()
    return sections


def chunk_markdown(doc_id: str, markdown: str, max_chars: int = 500) -> list[Chunk]:
    """Structure-aware chunking: split by heading, then recursively by size.

    Each chunk's text starts with its heading breadcrumb so that the embedding
    and the BM25 index both see the section topic.
    """
    chunks: list[Chunk] = []
    for trail, body in markdown_sections(markdown):
        crumb = " > ".join(trail)
        for piece in recursive_split(body, max_chars):
            text = f"[{crumb}] {piece}" if crumb else piece
            position = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}#{position}",
                    doc_id=doc_id,
                    text=text,
                    position=position,
                    metadata={"section": crumb},
                )
            )
    return chunks


def semantic_chunks(sentences: list[str], embedder: Embedder, sensitivity: float = 1.0) -> list[str]:
    """Group consecutive sentences, starting a new chunk where the topic shifts.

    The distance between neighbours is 1 - cosine. A boundary goes wherever a
    distance exceeds mean + sensitivity * stdev of all distances in the text.
    """
    if len(sentences) < 3:
        return [" ".join(sentences)] if sentences else []
    vectors = embedder.embed(sentences)
    distances = [1 - cosine(a, b) for a, b in zip(vectors, vectors[1:], strict=False)]
    threshold = statistics.fmean(distances) + sensitivity * statistics.pstdev(distances)
    groups: list[list[str]] = [[sentences[0]]]
    for sentence, distance in zip(sentences[1:], distances, strict=True):
        if distance > threshold:
            groups.append([])
        groups[-1].append(sentence)
    return [" ".join(g) for g in groups]


@dataclass(frozen=True)
class ParentChild:
    parents: dict[str, str]  # parent_id -> large passage handed to the LLM
    children: list[tuple[str, str]]  # (parent_id, small passage that gets embedded)


def parent_child_chunks(
    doc_id: str, text: str, parent_chars: int = 2000, child_chars: int = 400
) -> ParentChild:
    """Small-to-big indexing: embed small children, return their large parent at query time."""
    parents: dict[str, str] = {}
    children: list[tuple[str, str]] = []
    for i, parent in enumerate(recursive_split(text, parent_chars)):
        parent_id = f"{doc_id}/p{i}"
        parents[parent_id] = parent
        children.extend((parent_id, child) for child in recursive_split(parent, child_chars))
    return ParentChild(parents, children)


def python_code_chunks(source: str) -> list[dict[str, object]]:
    """One chunk per top-level function or class, decorators and docstring included.

    Uses the standard-library `ast` module; for other languages tree-sitter
    plays the same role.
    """
    tree = ast.parse(source)
    lines = source.splitlines()
    chunks: list[dict[str, object]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = min([node.lineno] + [d.lineno for d in node.decorator_list])
            end = node.end_lineno or node.lineno
            chunks.append(
                {
                    "name": node.name,
                    "kind": type(node).__name__,
                    "start_line": start,
                    "end_line": end,
                    "text": "\n".join(lines[start - 1 : end]),
                }
            )
    return chunks
