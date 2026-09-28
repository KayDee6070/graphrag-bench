import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from graphrag_bench.extraction.llm.config import LLMError, LocalModelConfig
from graphrag_bench.extraction.llm.prompt import make_request
from graphrag_bench.extraction.llm.provider import LocalTransformersProvider


@pytest.fixture
def inference_stub(monkeypatch):
    state = {"input_tokens": 4, "generated": [20, 7], "capacity": 16, "eos": [7, 8], "calls": []}

    class Tokenizer:
        pad_token_id = None

        def get_chat_template(self):
            return "pinned template"

        def apply_chat_template(self, messages, **kwargs):
            state["calls"].append(("template", messages, kwargs))
            return "rendered chat"

        def __call__(self, text, **kwargs):
            state["calls"].append(("tokenize", text, kwargs))
            return {"input_ids": SimpleNamespace(shape=(1, state["input_tokens"]))}

        def decode(self, tokens, **kwargs):
            state["calls"].append(("decode", tokens, kwargs))
            return '{"entities": [], "relations": []}'

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
