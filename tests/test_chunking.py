import pytest

from app.chunking import (
    chunk_markdown,
    markdown_sections,
    parent_child_chunks,
    python_code_chunks,
    recursive_split,
    semantic_chunks,
    sliding_window,
)
from app.embeddings import HashingEmbedder


def test_sliding_window_overlap():
    words = list("ABCDEFGHIJ")
    windows = sliding_window(words, size=4, overlap=2)
    assert windows[0] == list("ABCD")
    assert windows[1] == list("CDEF")
    assert windows[-1][-1] == "J"
    # every adjacent pair shares exactly `overlap` words
    for left, right in zip(windows, windows[1:], strict=False):
        assert left[-2:] == right[:2]


def test_sliding_window_rejects_bad_overlap():
    with pytest.raises(ValueError):
        sliding_window(["a"], size=3, overlap=3)


def test_recursive_split_respects_limit_and_keeps_text():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 30 for i in range(5))
    pieces = recursive_split(text, max_chars=120)
    assert all(len(p) <= 120 for p in pieces)
    assert " ".join(pieces).split() == text.split()


def test_recursive_split_prefers_paragraph_boundaries():
    text = "First paragraph is short.\n\nSecond paragraph is short too."
    assert recursive_split(text, max_chars=40) == [
        "First paragraph is short.",
        "Second paragraph is short too.",
    ]


def test_markdown_sections_build_breadcrumbs_and_ignore_code_fences():
    md = "# Guide\nintro\n## Install\n```bash\n# not a heading\npip install x\n```\n## Use\nrun it\n"
    sections = markdown_sections(md)
    assert [crumb for crumb, _ in sections] == [["Guide"], ["Guide", "Install"], ["Guide", "Use"]]
    assert "# not a heading" in sections[1][1]


def test_chunk_markdown_prefixes_breadcrumb_and_ids():
    chunks = chunk_markdown("doc", "# A\n## B\nBody text here.")
    assert chunks[0].chunk_id == "doc#0"
    assert chunks[0].text == "[A > B] Body text here."
    assert chunks[0].metadata["section"] == "A > B"


def test_semantic_chunks_split_where_topic_changes():
    sentences = [
        "Kafka partitions spread a topic across brokers.",
        "Kafka consumer groups share the partitions of a topic.",
        "Kafka brokers replicate partitions for durability.",
        "Postgres vacuum reclaims dead tuples.",
        "Postgres autovacuum runs vacuum on busy tables.",
    ]
    groups = semantic_chunks(sentences, HashingEmbedder(), sensitivity=0.5)
    assert len(groups) == 2
    assert groups[1].startswith("Postgres vacuum")


def test_parent_child_children_point_at_their_parent():
    text = "\n\n".join(f"Section {i}. " + "detail " * 60 for i in range(4))
    pc = parent_child_chunks("manual", text, parent_chars=900, child_chars=200)
    assert len(pc.children) > len(pc.parents)
    for parent_id, child in pc.children:
        assert child in pc.parents[parent_id]


def test_python_code_chunks_keep_functions_whole():
    source = (
        "import os\n\n"
        "@cache\n"
        "def load(path):\n"
        '    """Read a file."""\n'
        "    return open(path).read()\n\n"
        "class Store:\n"
        "    def get(self):\n"
        "        return 1\n"
    )
    chunks = python_code_chunks(source)
    assert [c["name"] for c in chunks] == ["load", "Store"]
    assert chunks[0]["text"].startswith("@cache")
    assert "Read a file." in chunks[0]["text"]
    assert chunks[1]["end_line"] == 10
