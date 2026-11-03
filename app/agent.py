"""A bounded multi-hop retrieval loop with a fact scratchpad."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from app.pipeline import Hit, RagPipeline
from app.text import split_sentences, tokenize

StopReason = Literal["done", "budget", "missing"]


@dataclass(frozen=True)
class Fact:
    sub_question: str
    chunk_id: str
    sentence: str


@dataclass
class Trace:
    facts: list[Fact] = field(default_factory=list)
    searches: int = 0
    missing: list[str] = field(default_factory=list)
    stopped: StopReason = "done"


def best_sentence(question: str, hit: Hit) -> str | None:
    """The sentence in `hit` that shares the most terms with `question`."""
    terms = set(tokenize(question))
    body = hit.chunk.text.split("] ", 1)[-1]
    scored = [(len(terms & set(tokenize(s))), s) for s in split_sentences(body)]
    scored = [pair for pair in scored if pair[0] > 0]
    return max(scored, key=lambda pair: pair[0])[1] if scored else None


def multi_hop(pipe: RagPipeline, sub_questions: list[str], max_searches: int = 4) -> Trace:
    """Answer each sub-question with its own retrieval, keeping provenance.

    Stops early when the search budget is spent ("budget") or when a
    sub-question finds no relevant evidence ("missing"), because later hops
    usually depend on earlier ones. In an LLM agent the sub-questions come
    from a decomposition call, and later ones may be rewritten using the
    facts found so far.
    """
    trace = Trace()
    for question in sub_questions:
        if trace.searches >= max_searches:
            trace.stopped = "budget"
            return trace
        hits = pipe.search(question, top_k=3)
        trace.searches += 1
        if pipe.grade(question, hits) == "incorrect":
            trace.missing.append(question)
            trace.stopped = "missing"
            return trace
        for hit in hits:
            sentence = best_sentence(question, hit)
            if sentence:
                trace.facts.append(Fact(question, hit.chunk.chunk_id, sentence))
                break
    return trace


def render_facts(trace: Trace) -> str:
    """Scratchpad text for the final generation prompt, one cited fact per line."""
    return "\n".join(f"- {f.sentence} [{f.chunk_id}]" for f in trace.facts)
