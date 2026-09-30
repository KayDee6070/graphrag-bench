import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from graphrag_bench.extraction.llm.config import LLMError, LocalModelConfig
from graphrag_bench.extraction.llm.prompt import make_request
from graphrag_bench.extraction.llm.provider import LocalTransformersProvider


@pytest.fixture
def inference_stub(monkeypatch):
    state = {
        "input_tokens": 4,
        "generated": [20, 7],
        "capacity": 16,
        "eos": [7, 8],
        "cuda": False,
        "calls": [],
    }

    class Tokenizer:
        pad_token_id = None

        def get_chat_template(self):
            return "pinned template"

        def apply_chat_template(self, messages, **kwargs):
            state["calls"].append(("template", messages, kwargs))
            return "rendered chat"

        def __call__(self, text, **kwargs):
            state["calls"].append(("tokenize", text, kwargs))
            return {"input_ids": Tensor(state["input_tokens"])}

        def decode(self, tokens, **kwargs):
            state["calls"].append(("decode", tokens, kwargs))
            return '{"entities": [], "relations": []}'

        def encode(self, text, **kwargs):
            state["calls"].append(("encode", text, kwargs))
            return list(range(len(text)))

    class Tensor:
        def __init__(self, count):
            self.shape = (1, count)

        def to(self, device):
            state["calls"].append(("place_input", device))
            return self

    class Model:
        def __init__(self):
            self.config = SimpleNamespace(max_position_embeddings=state["capacity"])
            self.generation_config = SimpleNamespace(eos_token_id=state["eos"])

        def to(self, device):
            state["calls"].append(("device", device))
            return self

        def eval(self):
            state["calls"].append(("eval",))

        def generate(self, **kwargs):
            state["calls"].append(("generate", kwargs))
            if state.get("fail"):
                raise RuntimeError("inference failure")
            return [[0] * state["input_tokens"] + state["generated"]]

    def load_tokenizer(model_id, **kwargs):
        state["calls"].append(("load_tokenizer", model_id, kwargs))
        return Tokenizer()

    def load_model(model_id, **kwargs):
        state["calls"].append(("load_model", model_id, kwargs))
        return Model()

    class GenerationConfig(SimpleNamespace):
        def to_dict(self):
            return vars(self).copy()

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            set_num_threads=lambda n: state["calls"].append(("threads", n)),
            float32="float32",
            float16="float16",
            bfloat16="bfloat16",
            cuda=SimpleNamespace(is_available=lambda: state["cuda"]),
            inference_mode=nullcontext,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=load_tokenizer),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=load_model),
            GenerationConfig=GenerationConfig,
        ),
    )
    monkeypatch.setattr("graphrag_bench.extraction.llm.provider.version", lambda _: "test")
    return state


def config():
    return LocalModelConfig(
        model_id="test/model", revision="a" * 40, max_input_tokens=8, max_new_tokens=4
    )


def test_pinned_cached_cpu_loading(inference_stub, tmp_path):
    provider = LocalTransformersProvider(config(), cache_folder=tmp_path)
    model = next(c for c in inference_stub["calls"] if c[0] == "load_model")
    assert model[2]["revision"] == "a" * 40
    assert model[2]["local_files_only"] is True
    assert model[2]["trust_remote_code"] is False
    assert model[2]["use_safetensors"] is True
    assert model[2]["dtype"] == "float32"
    assert model[2]["cache_dir"] == str(tmp_path)
    assert ("device", "cpu") in inference_stub["calls"]
    assert provider.spec.settings["do_sample"] is False
    spec = provider.spec
    spec.settings["do_sample"] = True
    assert provider.spec.settings["do_sample"] is False


def test_explicit_download_flag(inference_stub):
    LocalTransformersProvider(config(), allow_download=True)
    for call in inference_stub["calls"]:
        if call[0].startswith("load_"):
            assert call[2]["local_files_only"] is False


def test_complete_counts_chat_tokens_and_decodes_only_new_tokens(
    inference_stub, source_pair, llm_config
):
    provider = LocalTransformersProvider(config())
    response = provider.complete(make_request(source_pair[1][0], llm_config))
    assert (response.input_tokens, response.output_tokens, response.finish_reason) == (4, 2, "eos")
    tokenization = next(c for c in inference_stub["calls"] if c[0] == "tokenize")
    assert tokenization[2] == {
        "return_tensors": "pt",
        "add_special_tokens": False,
        "truncation": False,
    }
    decoding = next(c for c in inference_stub["calls"] if c[0] == "decode")
    assert decoding[1] == [20, 7]
    generation = next(c for c in inference_stub["calls"] if c[0] == "generate")[1][
        "generation_config"
    ]
    assert generation.do_sample is False and generation.num_beams == 1
    assert generation.max_new_tokens == 4


def test_oversized_prompt_fails_before_generation(inference_stub, source_pair, llm_config):
    inference_stub["input_tokens"] = 9
    provider = LocalTransformersProvider(config())
    with pytest.raises(LLMError, match="9 tokens"):
        provider.complete(make_request(source_pair[1][0], llm_config))
    assert not any(c[0] == "generate" for c in inference_stub["calls"])


def test_unfinished_response_marked_length(inference_stub, source_pair, llm_config):
    inference_stub["generated"] = [1, 2, 3, 4]
    response = LocalTransformersProvider(config()).complete(
        make_request(source_pair[1][0], llm_config)
    )
    assert response.finish_reason == "length"


def test_capacity_and_eos_checks(inference_stub):
    inference_stub["capacity"] = 10
    with pytest.raises(LLMError, match="exceed model capacity"):
        LocalTransformersProvider(config())
    inference_stub["capacity"] = 16
    inference_stub["eos"] = None
    with pytest.raises(LLMError, match="end-of-sequence"):
        LocalTransformersProvider(config())


def test_provider_error_is_actionable(inference_stub, source_pair, llm_config):
    inference_stub["fail"] = True
    provider = LocalTransformersProvider(config())
    with pytest.raises(LLMError, match="inference failure"):
        provider.complete(make_request(source_pair[1][0], llm_config))


def test_dependencies_are_optional(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(LLMError, match="optional torch/transformers"):
        LocalTransformersProvider(config())


def test_evidence_count_uses_same_tokenizer_without_special_tokens(inference_stub):
    provider = LocalTransformersProvider(config())
    assert provider.count_tokens("source") == 6
    assert (
        "encode",
        "source",
        {"add_special_tokens": False, "truncation": False},
    ) in inference_stub["calls"]


def test_shared_provider_accepts_answer_messages_without_an_extraction_chunk(inference_stub):
    from graphrag_bench.extraction.llm.contracts import Message

    request = SimpleNamespace(messages=(Message(role="user", content="Answer this question."),))
    assert LocalTransformersProvider(config()).complete(request).finish_reason == "eos"


def gpu_config(dtype="float16"):
    return LocalModelConfig(
        model_id="test/model",
        revision="a" * 40,
        device="cuda",
        dtype=dtype,
        max_input_tokens=8,
        max_new_tokens=4,
    )


def test_cpu_recipe_is_pinned_to_full_precision():
    with pytest.raises(ValueError, match="cpu inference in this project is pinned to float32"):
        LocalModelConfig(model_id="test/model", revision="a" * 40, dtype="float16")


@pytest.mark.parametrize("dtype", ["float64", "int8", ""])
def test_unsupported_dtypes_are_rejected(dtype):
    with pytest.raises(ValueError):
        LocalModelConfig(model_id="test/model", revision="a" * 40, device="cuda", dtype=dtype)


def test_unsupported_devices_are_rejected():
    with pytest.raises(ValueError):
        LocalModelConfig(model_id="test/model", revision="a" * 40, device="mps")


def test_cuda_request_without_a_cuda_build_fails_before_loading(inference_stub):
    inference_stub["cuda"] = False

    with pytest.raises(LLMError, match="reports no available device"):
        LocalTransformersProvider(gpu_config())

    assert not [call for call in inference_stub["calls"] if call[0].startswith("load_")]


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_cuda_recipe_places_model_and_inputs_on_the_device(
    inference_stub, source_pair, llm_config, dtype
):
    inference_stub["cuda"] = True

    provider = LocalTransformersProvider(gpu_config(dtype))
    provider.complete(make_request(source_pair[1][0], llm_config))

    model = next(c for c in inference_stub["calls"] if c[0] == "load_model")
    assert model[2]["dtype"] == dtype
    assert ("device", "cuda") in inference_stub["calls"]
    assert ("place_input", "cuda") in inference_stub["calls"]


def test_cuda_recipe_is_a_different_provider_version(inference_stub):
    inference_stub["cuda"] = True

    cpu = LocalTransformersProvider(config()).spec
    gpu = LocalTransformersProvider(gpu_config()).spec

    assert cpu.provider == "transformers-causal-cpu-v1"
    assert gpu.provider == "transformers-causal-cuda-v1"
    assert gpu.settings["device"] == "cuda" and gpu.settings["dtype"] == "float16"


def test_cpu_path_never_moves_inputs(inference_stub, source_pair, llm_config):
    provider = LocalTransformersProvider(config())
    provider.complete(make_request(source_pair[1][0], llm_config))

    assert not [call for call in inference_stub["calls"] if call[0] == "place_input"]
