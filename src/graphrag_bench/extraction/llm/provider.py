"""Optional, pinned CPU Transformers inference; no hosted service or implicit downloads."""

from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from graphrag_bench.extraction.llm.config import LLMError, LocalModelConfig
from graphrag_bench.extraction.llm.contracts import Completion, ExtractionRequest, ModelSpec
from graphrag_bench.models import text_sha256

PROVIDER_VERSION = "transformers-causal-cpu-v1"


class LocalTransformersProvider:
    def __init__(
        self,
        config: LocalModelConfig,
        *,
        allow_download: bool = False,
        cache_folder: Path | None = None,
    ) -> None:
        if Path(config.model_id).exists():
            raise LLMError("model_id must be a Hub repository, not a local directory")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        except ImportError as error:
            raise LLMError(
                "Local extraction requires optional torch/transformers dependencies; "
                "see requirements-embeddings-cpu.txt and docs/llm-extraction.md."
            ) from error
        self._config = config
        self._torch = torch
        try:
            torch.set_num_threads(config.cpu_threads)
            loading = {
                "revision": config.revision,
                "local_files_only": not allow_download,
                "trust_remote_code": False,
                "cache_dir": str(cache_folder) if cache_folder is not None else None,
            }
            self._tokenizer = AutoTokenizer.from_pretrained(config.model_id, **loading)
            template = self._tokenizer.get_chat_template()
            self._model = AutoModelForCausalLM.from_pretrained(
                config.model_id,
                **loading,
                use_safetensors=True,
                dtype=torch.float32,
                attn_implementation="eager",
            ).to("cpu")
            self._model.eval()
            capacity = self._model.config.max_position_embeddings
            if config.max_input_tokens + config.max_new_tokens > capacity:
                raise LLMError("input plus output token limits exceed model capacity")
            eos = self._model.generation_config.eos_token_id
            if eos is None:
                raise LLMError("model must declare an end-of-sequence token")
            self._eos = {eos} if isinstance(eos, int) else set(eos)
            pad = self._tokenizer.pad_token_id
            self._generation = GenerationConfig(
                do_sample=False,
                num_beams=1,
                temperature=1.0,
                top_p=1.0,
                top_k=50,
                repetition_penalty=1.0,
                max_new_tokens=config.max_new_tokens,
                use_cache=True,
                eos_token_id=eos,
                pad_token_id=pad if pad is not None else min(self._eos),
            )
            # Transformers fills unspecified fields from model defaults. Replace those
            # defaults too so a publisher's sampling/repetition settings cannot leak in.
            self._model.generation_config = self._generation
            self._spec = ModelSpec(
                provider=PROVIDER_VERSION,
                model_id=config.model_id,
                revision=config.revision,
                settings=config.model_dump(mode="json")
                | {
                    "do_sample": False,
                    "num_beams": 1,
                    "generation_config": self._generation.to_dict(),
                    "attention": "eager",
                    "chat_template_sha256": text_sha256(template),
                    "eos_token_ids": sorted(self._eos),
                    "pad_token_id": self._generation.pad_token_id,
                },
                versions={
                    name: version(name)
                    for name in ("torch", "transformers", "tokenizers", "huggingface-hub")
                },
            )
        except LLMError:
            raise
        except Exception as error:
            raise LLMError(
                f"Cannot load {config.model_id}@{config.revision}. "
                "Cached-only by default; --allow-download explicitly permits a download. "
                f"Provider error: {error}"
            ) from error

    @property
    def spec(self) -> ModelSpec:
        return self._spec.model_copy(deep=True)

    def complete(self, request: ExtractionRequest) -> Completion:
        started = perf_counter()
        try:
            messages = [
                {"role": message.role, "content": message.content} for message in request.messages
            ]
            rendered = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._tokenizer(
                rendered, return_tensors="pt", add_special_tokens=False, truncation=False
            )
            input_count = inputs["input_ids"].shape[-1]
            if input_count > self._config.max_input_tokens:
                raise LLMError(
                    f"prompt has {input_count} tokens; limit is {self._config.max_input_tokens}. "
                    "Use smaller chunks or an explicitly larger input limit; no truncation."
                )
            with self._torch.inference_mode():
                output = self._model.generate(**inputs, generation_config=self._generation)
            generated = output[0][input_count:]
            return Completion(
                text=self._tokenizer.decode(generated, skip_special_tokens=True),
                input_tokens=input_count,
                output_tokens=len(generated),
                finish_reason="eos"
                if len(generated) and int(generated[-1]) in self._eos
                else "length",
                elapsed_ms=(perf_counter() - started) * 1000,
            )
        except LLMError:
            raise
        except Exception as error:
            raise LLMError(f"local extraction inference failed: {error}") from error
