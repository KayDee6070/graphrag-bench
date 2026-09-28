"""Route all existing retrievers into the same answer-generation interface."""

from hashlib import sha256
from pathlib import Path

from graphrag_bench.embeddings.config import EmbeddingConfig
from graphrag_bench.embeddings.sentence_transformers import SentenceTransformerProvider
from graphrag_bench.generation.config import AnswerRetrievalConfig, GenerationError
from graphrag_bench.generation.contracts import RetrievalReceipt
from graphrag_bench.graph.pipeline import load_graph
from graphrag_bench.ingestion.reader import load_ingestion
from graphrag_bench.retrieval.artifacts import load_vector_index
from graphrag_bench.retrieval.bm25 import BM25Retriever
from graphrag_bench.retrieval.graph import GraphRetriever
from graphrag_bench.retrieval.hybrid import HybridRetriever
from graphrag_bench.retrieval.vector import validate_request


def retrieve_for_answer(
    source: Path,
    query: str,
    config: AnswerRetrievalConfig,
    *,
    graph_directory: Path | None = None,
    index_directory: Path | None = None,
    allow_download: bool = False,
    cache_folder: Path | None = None,
) -> RetrievalReceipt:
    validate_request(query, config.top_k)
    needs_graph = config.strategy in {"graph", "hybrid"}
    needs_index = config.strategy in {"vector", "hybrid"}
    if needs_graph != (graph_directory is not None):
        raise GenerationError("--graph is required exactly for graph and hybrid answers")
    if needs_index != (index_directory is not None):
        raise GenerationError("--index is required exactly for vector and hybrid answers")
    batch, _ = load_ingestion(source)
    # Verify both artifacts before loading an embedding model.
    graph = load_graph(graph_directory, source) if graph_directory is not None else None
    index = load_vector_index(index_directory, source) if index_directory is not None else None
    directories = {"graph": graph_directory, "vector": index_directory}
    try:
        index_hashes = {
            name: sha256((directory / "manifest.json").read_bytes()).hexdigest()
            for name, directory in directories.items()
            if directory is not None
        }
    except OSError as error:
        raise GenerationError(f"cannot record retrieval provenance: {error}") from error
    audit = {}
    graph_retriever = (
        GraphRetriever(graph, batch.documents, batch.chunks, config.graph)
        if graph is not None
        else None
    )
    provider = (
        SentenceTransformerProvider(
            EmbeddingConfig.model_validate(index.spec.settings),
            allow_download=allow_download,
            cache_folder=cache_folder,
        )
        if index is not None
        else None
    )
    if provider is not None:
        audit["embedding"] = provider.spec.model_dump(mode="json")
    if config.strategy == "bm25":
        result = BM25Retriever(batch.documents, batch.chunks, config.bm25).retrieve(
            query, top_k=config.top_k
        )
    elif config.strategy == "vector":
        result = index.retrieve(query, provider, top_k=config.top_k)
    else:
        retriever = (
            graph_retriever
            if config.strategy == "graph"
            else HybridRetriever(index, graph_retriever, provider, config.hybrid)
        )
        trace = retriever.retrieve_with_trace(query, top_k=config.top_k)
        result = trace.result
        audit["trace"] = trace.model_dump(mode="json")
        assertion_ids = {
            identifier
            for hit in result.hits
            for path in hit.paths
            for identifier in path.assertion_ids
        }
        audit["assertions"] = [
            graph.assertion(identifier).model_dump(mode="json")
            for identifier in sorted(assertion_ids)
        ]
    return RetrievalReceipt(result=result, config=config, index_hashes=index_hashes, audit=audit)
