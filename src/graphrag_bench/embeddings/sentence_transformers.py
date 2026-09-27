"""CPU Sentence Transformers adapter with explicit downloads and no silent truncation."""

from importlib.metadata import version
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from graphrag_bench.embeddings.base import EmbeddingError, EmbeddingSpec
from graphrag_bench.embeddings.config import EmbeddingConfig

PROVIDER_VERSION = "sentence-transformers-cpu-v1"


class SentenceTransformerProvider:
    def __init__(
        self,
        config: EmbeddingConfig,
        *,
        allow_download: bool = False,
        cache_folder: Path | None = None,
    ) -> None:
        if Path(config.model_id).exists():
            raise EmbeddingError("model_id must identify a Hub repository, not a local directory")
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise EmbeddingError(
                "Neural embeddings require the optional embeddings dependencies. "
                "See requirements-embeddings-cpu.txt and docs/vector-retrieval.md."
            ) from error
        self.config = config
        try:
            # A small explicit thread count avoids severe oversubscription on laptop CPUs.
            torch.set_num_threads(config.cpu_threads)
            self._model = SentenceTransformer(
                config.model_id,
                revision=config.revision,
                device=config.device,
                cache_folder=str(cache_folder) if cache_folder is not None else None,
                local_files_only=not allow_download,
                trust_remote_code=False,
                model_kwargs={"use_safetensors": True},
            )
            if config.max_seq_length > self._model.max_seq_length:
                raise EmbeddingError(
                    f"max_seq_length {config.max_seq_length} exceeds this model's limit "
                    f"{self._model.max_seq_length}"
                )
            self._model.max_seq_length = config.max_seq_length
            self._model.eval()
            self._spec = EmbeddingSpec(
                provider=PROVIDER_VERSION,
                model_id=config.model_id,
                revision=config.revision,
                dimensions=self._model.get_embedding_dimension(),
                settings=config.model_dump(mode="json"),
                versions={
                    name: version(name)
                    for name in (
                        "sentence-transformers",
                        "transformers",
                        "torch",
                        "tokenizers",
                        "huggingface-hub",
                        "numpy",
                    )
                },
            )
        except EmbeddingError:
            raise
        except Exception as error:
            # Provider libraries expose several unrelated load/network/tokenizer exceptions.
            raise EmbeddingError(
                f"Cannot load {config.model_id}@{config.revision}. "
                "Use --allow-download once if the pinned model is not cached. "
                f"Provider error: {error}"
            ) from error

    @property
    def spec(self) -> EmbeddingSpec:
        return self._spec.model_copy(deep=True)

    def _encode(self, texts: tuple[str, ...], *, query: bool) -> NDArray[np.float32]:
        if not texts:
            return np.empty((0, self._spec.dimensions), dtype=np.float32)
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise EmbeddingError("embedding inputs must be nonblank strings")
        prefix = self.config.query_prefix if query else self.config.document_prefix
        try:
            # The supported adapter is for standard text tokenizers, not arbitrary multimodal
            # or multi-tokenizer Router models. Prefixes and special tokens consume capacity.
            tokenized = self._model.tokenizer(
                [prefix + text for text in texts],
                truncation=False,
                padding=False,
                add_special_tokens=True,
            )["input_ids"]
            for position, tokens in enumerate(tokenized):
                if len(tokens) > self.config.max_seq_length:
                    raise EmbeddingError(
                        f"input {position} has {len(tokens)} tokens; limit is "
                        f"{self.config.max_seq_length}. Shorten the query or re-ingest "
                        "with smaller chunks; silent truncation is disabled."
                    )
            encode = self._model.encode_query if query else self._model.encode_document
            return np.asarray(
                encode(
                    list(texts),
                    prompt=prefix,
                    batch_size=self.config.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                    precision="float32",
                    processing_kwargs={"text": {"truncation": False}},
                ),
                dtype=np.float32,
            )
        except EmbeddingError:
            raise
        except Exception as error:
            raise EmbeddingError(f"embedding failed: {error}") from error

    def embed_documents(self, texts: tuple[str, ...]) -> NDArray[np.float32]:
        return self._encode(texts, query=False)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        return self._encode((text,), query=True)[0]
