"""Small lexical sanity baseline, separate from neural vector retrieval."""

import math
import re
from collections import Counter
from collections.abc import Iterable
from time import perf_counter

from pydantic import Field

from graphrag_bench.corpus import CorpusIndex
from graphrag_bench.models import Chunk, Document, Record, RetrievalHit, RetrievalResult
from graphrag_bench.retrieval.vector import validate_request


class BM25Config(Record):
    k1: float = Field(default=1.5, gt=0, strict=True)
    b: float = Field(default=0.75, ge=0, le=1, strict=True)


def tokenize(text: str) -> tuple[str, ...]:
    """Unicode word tokens, case-folded; no stemming, stoplist, or synonym expansion."""
    return tuple(re.findall(r"\w+", text.casefold()))


class BM25Retriever:
    def __init__(
        self,
        documents: Iterable[Document],
        chunks: Iterable[Chunk],
        config: BM25Config | None = None,
    ) -> None:
        self.config = config if config is not None else BM25Config()
        corpus = CorpusIndex(documents, chunks)
        self.chunks = tuple(corpus.chunks.values())
        self._terms = tuple(Counter(tokenize(chunk.text)) for chunk in self.chunks)
        self._lengths = tuple(sum(terms.values()) for terms in self._terms)
        self._average = sum(self._lengths) / len(self.chunks) if self.chunks else 0
        self._frequency = Counter(term for terms in self._terms for term in terms)

    def retrieve(self, query: str, *, top_k: int = 5) -> RetrievalResult:
        validate_request(query, top_k)
        started = perf_counter()
        scores = [0.0] * len(self.chunks)
        # Repeating a query word does not increase its weight in this declared variant.
        for term in sorted(set(tokenize(query))):
            frequency = self._frequency[term]
            if not frequency:
                continue
            idf = math.log1p((len(self.chunks) - frequency + 0.5) / (frequency + 0.5))
            for i, terms in enumerate(self._terms):
                tf = terms[term]
                if tf:
                    length_penalty = (
                        1 - self.config.b + self.config.b * (self._lengths[i] / self._average)
                    )
                    scores[i] += (
                        idf * tf * (self.config.k1 + 1) / (tf + self.config.k1 * length_penalty)
                    )
        ranked = sorted(
            (i for i, score in enumerate(scores) if score > 0),
            key=lambda i: (-scores[i], self.chunks[i].chunk_id),
        )
        return RetrievalResult(
            query=query,
            strategy="bm25",
            hits=tuple(
                RetrievalHit(chunk_id=self.chunks[i].chunk_id, score=scores[i])
                for i in ranked[:top_k]
            ),
            elapsed_ms=(perf_counter() - started) * 1000,
        )
